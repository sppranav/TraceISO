"""
Blank uncertainty extraction for TraceISO.

SDs and Pearson r are computed within one blank measurement. When
before/after averaging combines two blanks, each blank is evaluated
separately and its contribution is scaled by the weight the *producer*
actually applied to that blank for that isotope channel — one half for a
complete pair, one for a channel only one side supports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np

from domain.filters.outlier import (
    get_filtered_values,
    resolve_cycle_range,
    sample_cycle_key,
)
from domain.corrections.blank import BLANK_CHANNEL_WEIGHTS_KEY
from domain.models import Sample
from domain.uncertainty.propagation import (
    u_blank_correlated,
    u_blank_uncorrelated,
)
from domain.uncertainty.welch_satterthwaite import effective_dof


MIN_BLANK_CYCLES_FOR_CORRELATION = 10
BLANK_SPIKE_THRESHOLD = 5.0

#: Bracketing-blank roles in the order ``compute_blank_uncertainty`` maps them
#: onto the resolved blank list (first entry, last entry).
BLANK_ROLES: Tuple[str, str] = ("before", "after")


def resolve_blank_channel_weights(
    sample: Optional[Sample],
    isotopes: Tuple[str, ...],
    *,
    blank_mode: str,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Return ``{role: {isotope: weight}}`` as recorded by the blank producer.

    Returns ``None`` when the sample carries no producer record — an older
    session, or a sample whose correction never ran — so the caller keeps the
    historical equal-split assumption rather than silently zeroing a term.
    """
    if sample is None:
        return None
    recorded = (sample.metadata or {}).get(BLANK_CHANNEL_WEIGHTS_KEY)
    if not isinstance(recorded, dict):
        return None

    roles = BLANK_ROLES if blank_mode == "before_and_after" else ("before",)
    weights: Dict[str, Dict[str, float]] = {role: {} for role in roles}
    for isotope in isotopes:
        per_role = recorded.get(isotope)
        for role in roles:
            value = 0.0
            if isinstance(per_role, dict):
                try:
                    value = float(per_role.get(role, 0.0))
                except (TypeError, ValueError):
                    value = 0.0
            weights[role][isotope] = value if np.isfinite(value) else 0.0
    return weights


def _role_weight(
    channel_weights: Optional[Dict[str, Dict[str, float]]],
    role: str,
    isotope: str,
    default: float,
) -> float:
    """Weight applied to *isotope* from the blank in *role*, else *default*."""
    if not channel_weights:
        return default
    per_role = channel_weights.get(role)
    if not isinstance(per_role, dict) or isotope not in per_role:
        return default
    return float(per_role[isotope])


@dataclass
class BlankUncertaintyResult:
    """Result of blank uncertainty computation for a single ratio."""

    u_blank_abs: float = 0.0
    u_blank_uncorrelated_abs: float = 0.0
    u_blank_correlation_term_abs2: float = 0.0
    u_num_sd: float = 0.0
    u_den_sd: float = 0.0
    u_num_input: float = 0.0
    u_den_input: float = 0.0
    correlation: float = 0.0
    degrees_of_freedom: float = float("inf")
    num_corrected_mean: float = 0.0
    den_corrected_mean: float = 0.0
    blank_mode: str = "single"
    n_blanks_used: int = 1
    n_blank_cycles: int = 0
    blank_uncertainty_input: str = "sd"
    correlation_warning: str = ""
    model_dimension: int = 2
    aux_isotope: str = ""
    aux_corrected_mean: float = 0.0
    u_aux_sd: float = 0.0
    u_aux_input: float = 0.0
    correlation_labels: Tuple[str, ...] = field(default_factory=tuple)
    correlation_matrix: Optional[np.ndarray] = None
    per_blank_results: List["BlankUncertaintyResult"] = field(default_factory=list)
    num_weight: float = 1.0
    den_weight: float = 1.0


@dataclass
class BlankStats3Var:
    """Cycle-paired 3-variable blank statistics for Sr-style propagation."""

    isotopes: Tuple[str, str, str]
    means: Dict[str, float]
    sds: Dict[str, float]
    input_sds: Dict[str, float]
    empirical_correlation_matrix: np.ndarray
    applied_correlation_matrix: np.ndarray
    covariance_matrix: np.ndarray
    n_pairs: int
    degrees_of_freedom: float
    blank_uncertainty_input: str = "sd"
    warning: str = ""


def normalize_blank_uncertainty_input(value: str | None) -> str:
    """Normalize the blank uncertainty input mode."""
    normalized = str(value or "sd").strip().lower()
    return normalized if normalized in {"sd", "se"} else "sd"


def describe_blank_input_model(mode: str | None, n_cycles: int = 0) -> str:
    """Return one sentence naming the *configured* blank input model.

    The blank input is user-selectable between the cycle SD and the standard
    error of the blank mean, so a contributor description must not assert one
    of them unconditionally. This is the single source of that wording for
    every engine; see :func:`blank_input_sigma` for the matching arithmetic.
    """
    if normalize_blank_uncertainty_input(mode) == "se":
        if n_cycles > 0:
            return (
                "Blank input sigma is the standard error of the blank mean, "
                f"SD/sqrt(n) with n = {int(n_cycles)} blank cycles."
            )
        return (
            "Blank input sigma is the standard error of the blank mean, "
            "SD/sqrt(n)."
        )
    return (
        "Blank input sigma is the blank cycle SD, not the standard error of "
        "the mean."
    )


def blank_input_sigma(cycle_sd: float, n_cycles: int, mode: str | None) -> float:
    """Blank input sigma per *mode*: the cycle SD (``"sd"``) or the standard
    error of the mean ``SD / sqrt(n)`` (``"se"``)."""
    sigma = float(cycle_sd)
    if normalize_blank_uncertainty_input(mode) == "se" and n_cycles > 0:
        return sigma / float(np.sqrt(float(n_cycles)))
    return sigma


def blank_input_sigmas(
    cycle_sds: np.ndarray,
    n_cycles: int,
    mode: str | None,
) -> np.ndarray:
    """Vectorized blank input sigmas for covariance construction."""
    sigmas = np.asarray(cycle_sds, dtype=np.float64)
    if normalize_blank_uncertainty_input(mode) == "se" and n_cycles > 0:
        return sigmas / float(np.sqrt(float(n_cycles)))
    return sigmas


def resolve_blank_correction_mode(
    sample: Sample,
    configured_mode: str | None = None,
) -> str:
    """Resolve the blank correction mode for a sample."""
    if configured_mode in {"none", "before", "before_and_after"}:
        return configured_mode

    if "after" in sample.used_blanks:
        return "before_and_after"

    if sample.used_blanks:
        return "before"

    return "before"


def compute_blank_uncertainty(
    blank_samples: List[Sample],
    num_isotope: str,
    den_isotope: str,
    num_corrected_mean: float,
    den_corrected_mean: float,
    *,
    correlation_method: str = "pearson_from_data",
    fixed_r: float = 0.0,
    blank_correction_mode: str = "before",
    blank_uncertainty_input: str = "sd",
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    channel_weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> BlankUncertaintyResult:
    """Blank uncertainty for the ratio ``num_isotope / den_isotope``.

    Evaluates each blank measurement and propagates the numerator/denominator
    blank SDs through the ratio sensitivity coefficients — correlated or
    independent per *correlation_method*. Honours the correction mode
    (``"none"``, ``"before"``, ``"before_and_after"``). *channel_weights* is
    ``{role: {isotope: weight}}`` as recorded by the blank producer; when it is
    omitted a ``before_and_after`` pair falls back to the historical equal
    halves. Returns a :class:`BlankUncertaintyResult`."""
    blank_input_mode = normalize_blank_uncertainty_input(blank_uncertainty_input)
    mode = blank_correction_mode if blank_correction_mode in {
        "none", "before", "before_and_after",
    } else "before"

    if mode == "none":
        return BlankUncertaintyResult(
            blank_mode="none",
            n_blanks_used=0,
            num_corrected_mean=num_corrected_mean,
            den_corrected_mean=den_corrected_mean,
            blank_uncertainty_input=blank_input_mode,
        )

    if not blank_samples:
        return BlankUncertaintyResult(
            blank_mode="single",
            n_blanks_used=0,
            num_corrected_mean=num_corrected_mean,
            den_corrected_mean=den_corrected_mean,
            blank_uncertainty_input=blank_input_mode,
        )

    if mode == "before_and_after" and len(blank_samples) >= 2:
        # Each side carries the weight the producer actually applied to that
        # isotope channel: 1/2 for a complete pair, 1 where only this side
        # supplied a mean, 0 where it supplied none.
        weights_before = (
            _role_weight(channel_weights, "before", num_isotope, 0.5),
            _role_weight(channel_weights, "before", den_isotope, 0.5),
        )
        weights_after = (
            _role_weight(channel_weights, "after", num_isotope, 0.5),
            _role_weight(channel_weights, "after", den_isotope, 0.5),
        )
        result_before = _compute_single_blank_uncertainty(
            blank_samples[0],
            num_isotope,
            den_isotope,
            num_corrected_mean,
            den_corrected_mean,
            correlation_method,
            fixed_r,
            blank_input_mode,
            cycle_ranges=cycle_ranges,
            num_weight=weights_before[0],
            den_weight=weights_before[1],
        )
        result_after = _compute_single_blank_uncertainty(
            blank_samples[-1],
            num_isotope,
            den_isotope,
            num_corrected_mean,
            den_corrected_mean,
            correlation_method,
            fixed_r,
            blank_input_mode,
            cycle_ranges=cycle_ranges,
            num_weight=weights_after[0],
            den_weight=weights_after[1],
        )

        # The weights are already inside each side's propagation, so the two
        # independent blank measurements combine by plain quadrature here.
        scaled_before = result_before.u_blank_abs
        scaled_after = result_after.u_blank_abs
        u_combined = float(np.sqrt(scaled_before ** 2 + scaled_after ** 2))
        scaled_before_uncorr = result_before.u_blank_uncorrelated_abs
        scaled_after_uncorr = result_after.u_blank_uncorrelated_abs
        u_combined_uncorr = float(
            np.sqrt(scaled_before_uncorr ** 2 + scaled_after_uncorr ** 2)
        )
        correlation_term_abs2 = u_combined ** 2 - u_combined_uncorr ** 2
        dof = effective_dof([
            (scaled_before, result_before.degrees_of_freedom),
            (scaled_after, result_after.degrees_of_freedom),
        ])

        warning_parts = []
        if result_before.correlation_warning:
            warning_parts.append(
                f"{blank_samples[0].name}: {result_before.correlation_warning}"
            )
        if result_after.correlation_warning:
            warning_parts.append(
                f"{blank_samples[-1].name}: {result_after.correlation_warning}"
            )

        return BlankUncertaintyResult(
            u_blank_abs=u_combined,
            u_blank_uncorrelated_abs=u_combined_uncorr,
            u_blank_correlation_term_abs2=correlation_term_abs2,
            u_num_sd=_weighted_channel_mean(
                (result_before.u_num_sd, weights_before[0]),
                (result_after.u_num_sd, weights_after[0]),
            ),
            u_den_sd=_weighted_channel_mean(
                (result_before.u_den_sd, weights_before[1]),
                (result_after.u_den_sd, weights_after[1]),
            ),
            u_num_input=_weighted_channel_mean(
                (result_before.u_num_input, weights_before[0]),
                (result_after.u_num_input, weights_after[0]),
            ),
            u_den_input=_weighted_channel_mean(
                (result_before.u_den_input, weights_before[1]),
                (result_after.u_den_input, weights_after[1]),
            ),
            # item 53: use Fisher-z averaging for scalar correlation, consistent
            # with the matrix combination in engine_internal_sr.py.
            correlation=float(np.tanh(
                (np.arctanh(np.clip(result_before.correlation, -0.9999, 0.9999))
                 + np.arctanh(np.clip(result_after.correlation, -0.9999, 0.9999))) / 2.0
            )),
            degrees_of_freedom=dof,
            num_corrected_mean=num_corrected_mean,
            den_corrected_mean=den_corrected_mean,
            blank_mode="before_and_after",
            n_blanks_used=sum(
                1 for w in (weights_before, weights_after) if max(w) > 0.0
            ),
            n_blank_cycles=_contributing_blank_cycles(
                (result_before, weights_before),
                (result_after, weights_after),
            ),
            blank_uncertainty_input=blank_input_mode,
            correlation_warning=" | ".join(warning_parts),
            per_blank_results=[result_before, result_after],
            num_weight=weights_before[0] + weights_after[0],
            den_weight=weights_before[1] + weights_after[1],
        )

    result = _compute_single_blank_uncertainty(
        blank_samples[0],
        num_isotope,
        den_isotope,
        num_corrected_mean,
        den_corrected_mean,
        correlation_method,
        fixed_r,
        blank_input_mode,
        cycle_ranges=cycle_ranges,
        num_weight=_role_weight(channel_weights, "before", num_isotope, 1.0),
        den_weight=_role_weight(channel_weights, "before", den_isotope, 1.0),
    )
    result.blank_mode = "single"
    result.n_blanks_used = 1
    return result


@dataclass
class BlankSelection:
    """The blank observations behind one sample, plus what could not be resolved.

    Resolution is per *role*: a ``before_and_after`` record with one resolvable
    side is not a resolved pair, and reporting it as one would quietly halve
    the reference the correction actually used.
    """

    samples: List[Sample] = field(default_factory=list)
    ambiguous_names: List[str] = field(default_factory=list)
    unresolved_names: List[str] = field(default_factory=list)
    unresolved_ids: List[str] = field(default_factory=list)
    unresolved_roles: List[str] = field(default_factory=list)

    @property
    def is_fully_resolved(self) -> bool:
        """Whether every recorded role resolved to exactly one observation.

        A record with nothing at all in it is fully resolved *vacuously*: there
        is no selection to fail to resolve, which is a different state from a
        selection that was recorded and then could not be found.
        """
        return not (
            self.ambiguous_names
            or self.unresolved_names
            or self.unresolved_ids
            or self.unresolved_roles
        )

    def describe_unresolved(self) -> str:
        """One phrase naming what could not be resolved, for a refusal reason."""
        parts: List[str] = []
        if self.ambiguous_names:
            parts.append(
                "blank label(s) "
                + ", ".join(f"'{name}'" for name in self.ambiguous_names)
                + " match more than one observation"
            )
        if self.unresolved_names:
            parts.append(
                "blank label(s) "
                + ", ".join(f"'{name}'" for name in self.unresolved_names)
                + " are not in this session"
            )
        if self.unresolved_ids:
            parts.append(
                "blank observation(s) "
                + ", ".join(f"'{value}'" for value in self.unresolved_ids)
                + " are not in this session"
            )
        if not parts and self.unresolved_roles:
            parts.append("blank role(s) " + ", ".join(self.unresolved_roles))
        return "; ".join(parts)


def resolve_blank_selection(
    sample: Sample,
    all_samples: List[Sample],
) -> BlankSelection:
    """Resolve the blank observations the correction actually subtracted.

    The blank correction records both the label and the observation ID of each
    blank it used; the ID is what identifies the observation, because two
    blanks may share a label. A record that carries only a label — an older
    session, or a sample assembled by hand — is still honoured, but only when
    that label matches exactly one observation. An ambiguous label is reported
    rather than resolved to an arbitrary candidate: picking the first match is
    precisely how a consumer ends up propagating a different blank from the one
    that was subtracted.

    An explicit ID that names no observation in this session is reported too,
    and never falls back to the label recorded beside it. The label is the
    weaker record of the two; if the ID it accompanies is gone, agreeing to
    read the label instead would re-admit exactly the guess the ID exists to
    prevent.
    """
    references: List[Tuple[str, str, str]] = []
    for role, name in sample.used_blanks.items():
        references.append((
            str(role),
            str(sample.used_blank_ids.get(role, "") or ""),
            str(name or ""),
        ))
    for role, observation_id in sample.used_blank_ids.items():
        if role not in sample.used_blanks and observation_id:
            references.append((str(role), str(observation_id), ""))

    if not references:
        return BlankSelection()

    by_id = {
        candidate.observation_id: candidate
        for candidate in all_samples
        if candidate.observation_id
    }

    wanted_ids: set = set()
    ambiguous: List[str] = []
    unresolved: List[str] = []
    unresolved_ids: List[str] = []
    unresolved_roles: List[str] = []

    for role, observation_id, name in sorted(references):
        if observation_id:
            found = by_id.get(observation_id)
            if found is not None:
                wanted_ids.add(found.observation_id)
            else:
                if observation_id not in unresolved_ids:
                    unresolved_ids.append(observation_id)
                unresolved_roles.append(role)
            continue
        if not name:
            # A role recorded with neither an identity nor a label says
            # nothing; it never named a blank to resolve.
            continue
        matches = [c for c in all_samples if c.name == name]
        if len(matches) == 1:
            wanted_ids.add(matches[0].observation_id)
        elif len(matches) > 1:
            if name not in ambiguous:
                ambiguous.append(name)
            unresolved_roles.append(role)
        else:
            if name not in unresolved:
                unresolved.append(name)
            unresolved_roles.append(role)

    blanks: List[Sample] = []
    seen: set = set()
    for candidate in all_samples:
        if candidate.observation_id in wanted_ids and candidate.observation_id not in seen:
            blanks.append(candidate)
            seen.add(candidate.observation_id)

    return BlankSelection(
        samples=blanks,
        ambiguous_names=ambiguous,
        unresolved_names=unresolved,
        unresolved_ids=unresolved_ids,
        unresolved_roles=unresolved_roles,
    )


def resolve_blank_samples_for_uncertainty(
    sample: Sample,
    all_samples: List[Sample],
) -> Tuple[Optional[List[Sample]], BlankSelection]:
    """The blanks an uncertainty engine may propagate, and why, if it may not.

    Returns ``(None, selection)`` when a selection was recorded and could not
    be resolved. Every engine used to answer that case by falling back to all
    the blanks in the session, which reports the scatter of blanks that were
    never subtracted from this sample — the substitution the resolver exists to
    refuse, reintroduced one layer down.

    An empty list is the different, legitimate case: nothing was recorded at
    all, and the caller's own session-wide fallback still applies.
    """
    selection = resolve_blank_selection(sample, all_samples)
    if not selection.is_fully_resolved:
        return None, selection
    return list(selection.samples), selection


def resolve_blank_samples(
    sample: Sample,
    all_samples: List[Sample],
) -> List[Sample]:
    """Find the blank Sample(s) that were used to correct *sample*."""
    return resolve_blank_selection(sample, all_samples).samples


# Internal helpers

def _compute_single_blank_uncertainty(
    blank: Sample,
    num_isotope: str,
    den_isotope: str,
    num_corrected_mean: float,
    den_corrected_mean: float,
    correlation_method: str,
    fixed_r: float,
    blank_uncertainty_input: str,
    *,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    num_weight: float = 1.0,
    den_weight: float = 1.0,
) -> BlankUncertaintyResult:
    """Compute blank uncertainty from a single blank measurement.

    *num_weight*/*den_weight* are the coefficients this blank's mean carried in
    the subtraction for each channel. They scale the input sigmas, not the
    ratio sensitivities, so a complete pair at 1/2 each reproduces exactly the
    historical "divide the combined result by two" behaviour.
    """
    blank_input_mode = normalize_blank_uncertainty_input(blank_uncertainty_input)
    num_weight = float(num_weight) if np.isfinite(num_weight) else 0.0
    den_weight = float(den_weight) if np.isfinite(den_weight) else 0.0
    empty = BlankUncertaintyResult(
        blank_mode="single",
        n_blanks_used=1,
        num_corrected_mean=num_corrected_mean,
        den_corrected_mean=den_corrected_mean,
        blank_uncertainty_input=blank_input_mode,
        num_weight=num_weight,
        den_weight=den_weight,
    )

    if num_weight <= 0.0 and den_weight <= 0.0:
        # This blank supplied no channel of this ratio, so it subtracted
        # nothing and contributes no uncertainty.
        return empty

    if num_weight > 0.0 and den_weight > 0.0:
        v_num, v_den = get_paired_blank_voltages(
            blank,
            num_isotope,
            den_isotope,
            cycle_ranges=cycle_ranges,
        )
    else:
        # Only one channel of this ratio was corrected from this blank, so
        # there is no cycle pairing to establish and no correlation to apply.
        v_num = (
            _get_blank_voltages(blank, num_isotope, cycle_ranges=cycle_ranges)
            if num_weight > 0.0
            else np.array([], dtype=np.float64)
        )
        v_den = (
            _get_blank_voltages(blank, den_isotope, cycle_ranges=cycle_ranges)
            if den_weight > 0.0
            else np.array([], dtype=np.float64)
        )

    usable_num = len(v_num) >= 2
    usable_den = len(v_den) >= 2
    if not usable_num and not usable_den:
        return empty

    n_pairs = min(len(v) for v in (v_num, v_den) if len(v) >= 2)
    u_num_sd = float(np.std(v_num, ddof=1)) if usable_num else 0.0
    u_den_sd = float(np.std(v_den, ddof=1)) if usable_den else 0.0
    u_num_input = (
        blank_input_sigma(u_num_sd, n_pairs, blank_input_mode) if usable_num else 0.0
    )
    u_den_input = (
        blank_input_sigma(u_den_sd, n_pairs, blank_input_mode) if usable_den else 0.0
    )
    if usable_num and usable_den:
        correlation, warning = _compute_correlation(
            v_num, v_den, correlation_method, fixed_r,
        )
    else:
        correlation, warning = 0.0, ""
    dof = float(max(n_pairs - 1, 1))

    if den_corrected_mean == 0.0:
        return BlankUncertaintyResult(
            u_num_sd=u_num_sd,
            u_den_sd=u_den_sd,
            u_num_input=u_num_input,
            u_den_input=u_den_input,
            correlation=correlation,
            degrees_of_freedom=dof,
            num_corrected_mean=num_corrected_mean,
            den_corrected_mean=den_corrected_mean,
            blank_mode="single",
            n_blanks_used=1,
            n_blank_cycles=n_pairs,
            blank_uncertainty_input=blank_input_mode,
            correlation_warning=warning,
            num_weight=num_weight,
            den_weight=den_weight,
        )

    weighted_num_input = u_num_input * max(num_weight, 0.0)
    weighted_den_input = u_den_input * max(den_weight, 0.0)

    u_blank_uncorrelated_abs = u_blank_uncorrelated(
        weighted_num_input,
        weighted_den_input,
        num_corrected_mean,
        den_corrected_mean,
    )
    if correlation_method == "uncorrelated" or correlation == 0.0:
        u_blank_abs = u_blank_uncorrelated_abs
    else:
        u_blank_abs = u_blank_correlated(
            weighted_num_input,
            weighted_den_input,
            num_corrected_mean,
            den_corrected_mean,
            correlation,
        )
    correlation_term_abs2 = u_blank_abs ** 2 - u_blank_uncorrelated_abs ** 2

    return BlankUncertaintyResult(
        u_blank_abs=u_blank_abs,
        u_blank_uncorrelated_abs=u_blank_uncorrelated_abs,
        u_blank_correlation_term_abs2=correlation_term_abs2,
        u_num_sd=u_num_sd,
        u_den_sd=u_den_sd,
        u_num_input=u_num_input,
        u_den_input=u_den_input,
        correlation=correlation,
        degrees_of_freedom=dof,
        num_corrected_mean=num_corrected_mean,
        den_corrected_mean=den_corrected_mean,
        blank_mode="single",
        n_blanks_used=1,
        n_blank_cycles=n_pairs,
        blank_uncertainty_input=blank_input_mode,
        correlation_warning=warning,
        num_weight=num_weight,
        den_weight=den_weight,
    )


def _weighted_channel_mean(*entries: Tuple[float, float]) -> float:
    """Weight-normalised mean of one diagnostic channel SD across blanks.

    Equal halves reproduce the plain average this replaced; a channel only one
    blank supports reports that blank's own value rather than half of it.
    """
    total_weight = sum(max(w, 0.0) for _v, w in entries)
    if total_weight <= 0.0:
        return 0.0
    return float(
        sum(v * max(w, 0.0) for v, w in entries) / total_weight
    )


def _contributing_blank_cycles(
    *entries: Tuple["BlankUncertaintyResult", Tuple[float, float]],
) -> int:
    """Smallest blank cycle count among the blanks that actually contributed."""
    counts = [
        result.n_blank_cycles
        for result, weights in entries
        if max(weights) > 0.0 and result.n_blank_cycles > 0
    ]
    return int(min(counts)) if counts else 0


def _single_blank_dof(v_num: np.ndarray, v_den: np.ndarray) -> float:
    """Degrees of freedom for a single-blank result."""
    n_blank_cycles = min(len(v_num), len(v_den))
    return float(max(n_blank_cycles - 1, 1))


def _filter_blank_spikes(voltages: np.ndarray) -> np.ndarray:
    """Remove blank spikes without narrowing normal blank scatter."""
    arr = np.asarray(voltages, dtype=np.float64)
    if len(arr) < 2:
        return arr

    mask = _compute_spike_keep_mask(arr)
    filtered = arr[mask]
    if len(filtered) < 2:
        return arr

    # Guard against over-filtering when blank levels are close to zero:
    # spike filtering should remove isolated artifacts, not most cycles.
    min_keep = max(2, int(np.ceil(0.7 * len(arr))))
    if len(filtered) < min_keep:
        return arr

    return filtered


def _compute_spike_keep_mask(values: np.ndarray) -> np.ndarray:
    """Return a conservative keep mask for spike filtering."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return np.array([], dtype=bool)

    median_v = float(np.median(arr))
    scale = max(abs(median_v), float(np.median(np.abs(arr))))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.ones(len(arr), dtype=bool)

    return np.abs(arr) <= (BLANK_SPIKE_THRESHOLD * scale)


def _get_blank_voltages(
    blank: Sample,
    isotope: str,
    *,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> np.ndarray:
    """Extract valid blank voltages from a single blank sample."""
    cycle_data = blank.intensities.get(isotope)
    if cycle_data is None:
        return np.array([], dtype=np.float64)

    valid = get_filtered_values(
        np.asarray(cycle_data.values, dtype=np.float64),
        np.asarray(cycle_data.mask, dtype=bool),
        blank.name,
        cycle_ranges=cycle_ranges,
        sample_key=sample_cycle_key(blank),
        filter_method="None",
        filter_threshold=2.0,
    )
    valid = np.asarray(valid, dtype=np.float64)
    if len(valid) < 2:
        return valid

    return _filter_blank_spikes(valid)


def get_paired_blank_voltages(
    blank: Sample,
    num_isotope: str,
    den_isotope: str,
    *,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return paired blank voltages for two isotopes from the same cycles."""
    paired = _get_paired_blank_matrix(
        blank,
        (num_isotope, den_isotope),
        cycle_ranges=cycle_ranges,
    )
    if paired is None:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    return paired[0], paired[1]

def _compute_correlation(
    v_num: np.ndarray,
    v_den: np.ndarray,
    method: str,
    fixed_r: float,
) -> Tuple[float, str]:
    """Compute the blank correlation coefficient and any warning."""
    if method == "uncorrelated":
        return 0.0, ""
    if method == "fixed_value":
        return float(np.clip(fixed_r, -1.0, 1.0)), ""

    n = min(len(v_num), len(v_den))
    if n < MIN_BLANK_CYCLES_FOR_CORRELATION:
        warning = (
            f"Blank has {n} cycles (< {MIN_BLANK_CYCLES_FOR_CORRELATION}); "
            "correlation set to 0 (insufficient data for reliable Pearson r)"
        )
        return 0.0, warning

    corr_matrix = np.corrcoef(v_num[:n], v_den[:n])
    correlation = float(corr_matrix[0, 1])
    if not np.isfinite(correlation):
        return 0.0, "Non-finite correlation; set to 0"

    return correlation, ""


def compute_blank_stats_3var(
    blank: Sample,
    isotopes: Tuple[str, str, str],
    *,
    correlation_method: str = "pearson_from_data",
    fixed_r: float = 0.0,
    blank_uncertainty_input: str = "sd",
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[BlankStats3Var]:
    """Return paired 3-variable blank statistics for Sr blank propagation."""
    blank_input_mode = normalize_blank_uncertainty_input(blank_uncertainty_input)
    paired = _get_paired_blank_matrix(
        blank,
        isotopes,
        cycle_ranges=cycle_ranges,
    )
    if paired is None:
        return None
    n_pairs = int(paired.shape[1])
    means = {
        isotope: float(np.mean(paired[idx]))
        for idx, isotope in enumerate(isotopes)
    }
    sds = {
        isotope: float(np.std(paired[idx], ddof=1))
        for idx, isotope in enumerate(isotopes)
    }

    empirical_corr = _compute_empirical_correlation_matrix(paired)
    applied_corr, warning = _resolve_correlation_matrix_3var(
        empirical_corr,
        n_pairs=n_pairs,
        method=correlation_method,
        fixed_r=fixed_r,
    )
    sd_vector = np.array([sds[isotope] for isotope in isotopes], dtype=np.float64)
    input_sd_vector = blank_input_sigmas(sd_vector, n_pairs, blank_input_mode)
    input_sds = {
        isotope: float(input_sd_vector[idx])
        for idx, isotope in enumerate(isotopes)
    }
    covariance_matrix = np.outer(input_sd_vector, input_sd_vector) * applied_corr

    return BlankStats3Var(
        isotopes=isotopes,
        means=means,
        sds=sds,
        input_sds=input_sds,
        empirical_correlation_matrix=empirical_corr,
        applied_correlation_matrix=applied_corr,
        covariance_matrix=covariance_matrix,
        n_pairs=n_pairs,
        degrees_of_freedom=float(max(n_pairs - 1, 1)),
        blank_uncertainty_input=blank_input_mode,
        warning=warning,
    )


def _get_paired_blank_matrix(
    blank: Sample,
    isotopes: Tuple[str, ...],
    *,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[np.ndarray]:
    """Return cycle-aligned blank voltages for all requested isotopes.

    Pairing is done before spike rejection so correlation/covariance is always
    estimated from the same blank cycles across every isotope in the block.
    """
    cycle_data = []
    n_cycles = None
    for isotope in isotopes:
        cd = blank.intensities.get(isotope)
        if cd is None:
            return None
        cycle_data.append(cd)
        n_iso = len(cd.values)
        if n_cycles is not None and n_cycles != n_iso:
            raise ValueError("Blank channels have incompatible cycle lengths.")
        n_cycles = n_iso

    if n_cycles is None or n_cycles < 2:
        return None

    value_rows = [
        np.asarray(cd.values[:n_cycles], dtype=np.float64)
        for cd in cycle_data
    ]
    mask = np.ones(n_cycles, dtype=bool)
    for values, cd in zip(value_rows, cycle_data):
        mask &= np.asarray(cd.mask[:n_cycles], dtype=bool)
        mask &= np.isfinite(values)

    cycle_range = resolve_cycle_range(
        cycle_ranges,
        sample_name=blank.name,
        sample_key=sample_cycle_key(blank),
    )
    if cycle_range is not None:
        start_idx = max(int(cycle_range[0]) - 1, 0)
        end_idx = min(int(cycle_range[1]), n_cycles)
        range_mask = np.zeros(n_cycles, dtype=bool)
        if end_idx > start_idx:
            range_mask[start_idx:end_idx] = True
        mask &= range_mask

    paired = np.vstack([values[mask] for values in value_rows])
    if paired.shape[1] < 2:
        return None

    keep = np.ones(paired.shape[1], dtype=bool)
    for idx in range(paired.shape[0]):
        keep &= _compute_spike_keep_mask(paired[idx])

    paired_filtered = paired[:, keep]
    if paired_filtered.shape[1] < 2:
        return paired

    min_keep = max(2, int(np.ceil(0.7 * paired.shape[1])))
    if paired_filtered.shape[1] < min_keep:
        return paired

    return paired_filtered


def _compute_empirical_correlation_matrix(paired: np.ndarray) -> np.ndarray:
    """Return a robust empirical correlation matrix for paired blank cycles."""
    n_dim = int(paired.shape[0])
    corr = np.eye(n_dim, dtype=np.float64)
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            sd_i = float(np.std(paired[i], ddof=1))
            sd_j = float(np.std(paired[j], ddof=1))
            if np.isclose(sd_i, 0.0) or np.isclose(sd_j, 0.0):
                value = 0.0
            else:
                value = float(np.corrcoef(paired[i], paired[j])[0, 1])
                if not np.isfinite(value):
                    value = 0.0
            corr[i, j] = value
            corr[j, i] = value
    return corr


def _resolve_correlation_matrix_3var(
    empirical_corr: np.ndarray,
    *,
    n_pairs: int,
    method: str,
    fixed_r: float,
) -> Tuple[np.ndarray, str]:
    """Resolve the applied 3x3 correlation matrix for blank propagation."""
    n_dim = int(empirical_corr.shape[0])
    identity = np.eye(n_dim, dtype=np.float64)

    if method == "uncorrelated":
        return identity, ""

    if method == "fixed_value":
        # Equal-correlation 3x3 matrices require r >= -0.5 to remain PSD.
        applied_r = float(np.clip(fixed_r, -0.5, 1.0))
        corr = np.full((n_dim, n_dim), applied_r, dtype=np.float64)
        np.fill_diagonal(corr, 1.0)
        warning = ""
        if not np.isclose(applied_r, fixed_r):
            warning = (
                "Fixed blank correlation was clipped to -0.5 for the "
                "3-variable Sr blank model to keep the covariance matrix valid."
            )
        return corr, warning

    if n_pairs < MIN_BLANK_CYCLES_FOR_CORRELATION:
        warning = (
            f"Blank has {n_pairs} cycles (< {MIN_BLANK_CYCLES_FOR_CORRELATION}); "
            "3-variable correlation terms set to 0 (insufficient data for reliable Pearson r)"
        )
        return identity, warning

    corr = np.asarray(empirical_corr, dtype=np.float64).copy()
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr, ""

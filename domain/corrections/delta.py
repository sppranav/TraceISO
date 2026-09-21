"""Delta-value calculation for TraceISO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from domain.models import Sample
from domain.filters.outlier import get_filtered_values, sample_cycle_key
from domain.ratio_selection import (
    SkippedStandard,
    StandardProbe,
    find_nearest_usable_standard,
    format_skipped_standards,
    get_ssb_cycle_data,
    is_usable_bracketing_standard,
    select_best_delta_ratio_layer,
)


@dataclass
class DeltaResult:
    """Per-sample delta calculation result for a single ratio."""

    delta: float
    delta_sd: float
    delta_se: float
    delta_2se: float
    n: int
    sample_mean: float
    std_mean: float
    prev_std_name: str = ""
    next_std_name: str = ""
    delta_per_cycle: Optional[np.ndarray] = None


@dataclass(frozen=True)
class ClassicDeltaBracketSide:
    """Structured estimate for one production-selected delta standard."""

    name: str
    mean: float
    standard_error: float
    n: int
    layer_key: str
    observation_id: str = ""


@dataclass(frozen=True)
class ClassicDeltaBracketResolution:
    """One auditable outcome from the production classic-delta bracket search."""

    previous: Optional[ClassicDeltaBracketSide]
    following: Optional[ClassicDeltaBracketSide]
    source_layer: str
    previous_layer: str = ""
    following_layer: str = ""
    skipped: Tuple[SkippedStandard, ...] = ()
    failure_code: str = ""

    @property
    def sides(self) -> Optional[Tuple[ClassicDeltaBracketSide, ClassicDeltaBracketSide]]:
        if self.previous is None or self.following is None:
            return None
        return self.previous, self.following


def resolve_classic_delta_bracket(
    *,
    samples: List[Sample],
    sample: Sample,
    ratio_name: str,
    use_corrected: bool = True,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    sample_index: Optional[int] = None,
) -> ClassicDeltaBracketResolution:
    """Resolve classic-delta sides through the production selection contract.

    This is the single owner of ordering, usable-standard search, ratio-layer
    consistency, skipped-standard provenance and optional runtime cycle ranges.
    Deterministic delta calculation and Engine B replay consume this same
    resolution so they cannot silently select different brackets.
    """
    if use_corrected:
        selected = select_best_delta_ratio_layer(sample, ratio_name)
        source_layer = selected.key if selected is not None else ""
    else:
        source_layer = "ratios" if sample.ratios.get(ratio_name) is not None else ""

    if sample_index is None:
        # Identity is authoritative.  A name/run match is only a compatibility
        # fallback for callers holding an equivalent deserialised object; it
        # must never shadow the actual observation later in the input list.
        sample_index = next(
            (index for index, candidate in enumerate(samples) if candidate is sample),
            None,
        )
        if sample_index is None:
            equivalent_indices = [
                index
                for index, candidate in enumerate(samples)
                if candidate.name == sample.name
                and candidate.run_number == sample.run_number
            ]
            if len(equivalent_indices) == 1:
                sample_index = equivalent_indices[0]
            elif len(equivalent_indices) > 1:
                return ClassicDeltaBracketResolution(
                    None, None, source_layer, failure_code="ambiguous_sample"
                )
    if sample_index is None:
        return ClassicDeltaBracketResolution(
            None, None, source_layer, failure_code="sample_not_found"
        )

    probe = _bracketing_probe(ratio_name, use_corrected)
    previous = find_nearest_usable_standard(
        samples, sample_index, direction="before", probe=probe
    )
    following = find_nearest_usable_standard(
        samples, sample_index, direction="after", probe=probe
    )
    skipped = previous.skipped + following.skipped
    previous_layer = str(previous.layer_key or "")
    following_layer = str(following.layer_key or "")

    if previous.sample is None or following.sample is None:
        return ClassicDeltaBracketResolution(
            None,
            None,
            source_layer,
            previous_layer,
            following_layer,
            skipped,
            "missing_standard",
        )
    if previous.layer_key != source_layer or following.layer_key != source_layer:
        return ClassicDeltaBracketResolution(
            None,
            None,
            source_layer,
            previous_layer,
            following_layer,
            skipped,
            "layer_mismatch",
        )

    def _side(search_result) -> Optional[ClassicDeltaBracketSide]:
        cycle_data = search_result.cycle_data
        if cycle_data is None or search_result.sample is None:
            return None
        values = get_filtered_values(
            cycle_data.values,
            cycle_data.mask,
            search_result.sample.name,
            cycle_ranges=cycle_ranges,
            sample_key=sample_cycle_key(search_result.sample),
            filter_method="None",
            filter_threshold=2.0,
        )
        n = int(len(values))
        if n == 0:
            return None
        standard_error = (
            float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        )
        return ClassicDeltaBracketSide(
            name=str(search_result.name or ""),
            mean=float(np.mean(values)),
            standard_error=standard_error,
            n=n,
            layer_key=str(search_result.layer_key or ""),
            observation_id=str(search_result.sample.observation_id or ""),
        )

    previous_side = _side(previous)
    following_side = _side(following)
    if previous_side is None or following_side is None:
        return ClassicDeltaBracketResolution(
            None,
            None,
            source_layer,
            previous_layer,
            following_layer,
            skipped,
            "empty_standard",
        )
    return ClassicDeltaBracketResolution(
        previous_side,
        following_side,
        source_layer,
        previous_layer,
        following_layer,
        skipped,
    )


def resolve_classic_delta_bracket_sides(
    *,
    sample: Sample,
    ratio_name: str,
    all_samples: List[Sample],
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[Tuple[ClassicDeltaBracketSide, ClassicDeltaBracketSide]]:
    """Return production-selected per-side estimates, or ``None`` if unavailable."""
    return resolve_classic_delta_bracket(
        samples=all_samples,
        sample=sample,
        ratio_name=ratio_name,
        use_corrected=True,
        cycle_ranges=cycle_ranges,
    ).sides


def _measured_se(side: Optional[ClassicDeltaBracketSide]) -> Optional[float]:
    """The bracket side's measured standard error, or None when there is none.

    A side supported by fewer than two accepted cycles has no measured standard
    error.  That is reported as None rather than 0.0, because a zero would assert
    the standard was measured exactly instead of not measured at all (A009).
    """
    if side is None:
        return None
    return side.standard_error if side.n > 1 else None


def calculate_deltas(
    samples: List[Sample],
    ratio_name: str,
    use_corrected: bool = True,
    per_cycle: bool = True,
    reference_value: Optional[float] = None,
    reference_material_name: Optional[str] = None,
) -> List[Sample]:
    """Calculate delta values for every SMP sample."""
    for idx, sample in enumerate(samples):
        if not sample.is_sample:
            if not sample.is_standard and not sample.is_blank:
                sample.warnings.append(
                    f"Delta skipped for {ratio_name}: sample type "
                    f"'{sample.sample_type}' is not included in delta calculations."
                )
            continue

        sample.delta_results.pop(ratio_name, None)
        selected_layer = None
        if reference_value is not None:
            sample_cd = get_ssb_cycle_data(sample, ratio_name)
            source_layer = "ssb"
        elif use_corrected:
            selected_layer = select_best_delta_ratio_layer(sample, ratio_name)
            sample_cd = selected_layer.data if selected_layer is not None else None
            source_layer = selected_layer.key if selected_layer is not None else ""
        else:
            sample_cd = sample.ratios.get(ratio_name)
            source_layer = "ratios"
        if sample_cd is None:
            sample.warnings.append(
                f"Delta skipped for {ratio_name}: "
                + (
                    "certified-reference delta requires a successful SSB result"
                    if reference_value is not None
                    else "sample has no ratio data"
                )
            )
            continue
        valid = sample_cd.valid_values
        if len(valid) == 0:
            sample.warnings.append(
                f"Delta skipped for {ratio_name}: selected {source_layer or 'ratio'} layer "
                "has no valid cycles"
            )
            continue

        all_skipped: Tuple[SkippedStandard, ...] = ()
        if reference_value is not None:
            # SSB-corrected delta: use certified ratio as denominator
            std_avg = reference_value
            _ref_label = f"{reference_material_name} (certified)" if reference_material_name else "(certified)"
            prev_name = _ref_label
            next_name = _ref_label
        else:
            # Classic bracketing delta
            bracket = resolve_classic_delta_bracket(
                samples=samples,
                sample=sample,
                ratio_name=ratio_name,
                use_corrected=use_corrected,
                sample_index=idx,
            )
            if bracket.failure_code == "missing_standard":
                sample.warnings.append(
                    f"Delta skipped for {ratio_name}: missing bracketing standard(s)"
                )
                continue
            if bracket.failure_code == "layer_mismatch":
                sample.warnings.append(
                    f"Delta skipped for {ratio_name}: sample and bracketing standards "
                    f"do not share one ratio layer "
                    f"({source_layer}, {bracket.previous_layer}, {bracket.following_layer})"
                )
                continue
            if bracket.sides is None:
                sample.warnings.append(
                    f"Delta skipped for {ratio_name}: bracketing standard(s) have no valid cycles"
                )
                continue
            previous_side, following_side = bracket.sides
            prev_name, next_name = previous_side.name, following_side.name
            std_avg = (previous_side.mean + following_side.mean) / 2.0
            all_skipped = bracket.skipped

        if not np.isfinite(std_avg) or std_avg <= 0:
            sample.warnings.append(
                f"Delta skipped for {ratio_name}: reference value is "
                f"non-finite or non-positive ({std_avg:.6g})"
            )
            continue

        result = _compute_delta(
            valid, std_avg,
            prev_name=prev_name, next_name=next_name,
            per_cycle=per_cycle,
        )

        if all_skipped:
            sample.warnings.append(
                f"Delta bracket for {ratio_name} widened past unusable "
                f"standard(s): {format_skipped_standards(all_skipped)}"
            )

        sample.delta_results[ratio_name] = {
            "delta": result.delta,
            "delta_sd": result.delta_sd,
            "delta_se": result.delta_se,
            "delta_2se": result.delta_2se,
            "n": result.n,
            "sample_mean": result.sample_mean,
            "std_mean": result.std_mean,
            "prev_std": result.prev_std_name,
            "next_std": result.next_std_name,
            "prev_std_obs": (
                previous_side.observation_id if reference_value is None else ""
            ),
            "next_std_obs": (
                following_side.observation_id if reference_value is None else ""
            ),
            "prev_std_n": previous_side.n if reference_value is None else None,
            "next_std_n": following_side.n if reference_value is None else None,
            # A009: a one-cycle reference has no measured SE.  The payload
            # carries None, never 0.0 — a zero would assert the standard was
            # measured exactly rather than not measured at all.
            "prev_std_se": (
                _measured_se(previous_side) if reference_value is None else None
            ),
            "next_std_se": (
                _measured_se(following_side) if reference_value is None else None
            ),
            "source_layer": source_layer,
            "reference_kind": "certified" if reference_value is not None else "bracketing",
            "skipped_standards": [s.to_payload() for s in all_skipped],
        }
        if result.delta_per_cycle is not None:
            aligned_cycles = np.full(len(sample_cd.values), np.nan, dtype=np.float64)
            valid_indices = np.where(sample_cd.mask)[0]
            n = min(len(valid_indices), len(result.delta_per_cycle))
            aligned_cycles[valid_indices[:n]] = result.delta_per_cycle[:n]
            sample.delta_results[ratio_name]["delta_per_cycle"] = aligned_cycles
            sample.delta_results[ratio_name]["delta_mask"] = sample_cd.mask.copy()

    return samples


def delta_from_values(
    sample_mean: float,
    std_mean: float,
    per_mil: float = 1000.0,
) -> float:
    """Compute a single delta value.

    delta = (sample_mean / std_mean - 1) × per_mil
    """
    if not np.isfinite(std_mean) or std_mean == 0:
        return np.nan
    return (sample_mean / std_mean - 1.0) * per_mil


def delta_per_cycle_array(
    sample_cycles: np.ndarray,
    std_mean: float,
    per_mil: float = 1000.0,
) -> np.ndarray:
    """Compute per-cycle delta array.

    delta_i = (sample_i / std_mean - 1) × per_mil
    """
    if not np.isfinite(std_mean) or std_mean == 0:
        return np.full(len(sample_cycles), np.nan)
    return (sample_cycles / std_mean - 1.0) * per_mil


def _bracketing_probe(
    ratio_name: str, use_corrected: bool,
) -> Callable[[Sample], StandardProbe]:
    """Return a probe testing one candidate standard for delta bracketing."""

    def probe(candidate: Sample) -> StandardProbe:
        from domain.pb_correction_records import hg_blocks_final_value

        # An unavailable requested Hg correction is skipped, never bracketed raw.
        if use_corrected and hg_blocks_final_value(candidate, ratio_name):
            return StandardProbe(False, reason="hg correction unavailable")
        selected = select_best_delta_ratio_layer(candidate, ratio_name) if use_corrected else None
        cd = selected.data if selected is not None else candidate.ratios.get(ratio_name)
        key = selected.key if selected is not None else "ratios"
        if cd is None:
            return StandardProbe(False, reason="missing ratio layer")
        if not is_usable_bracketing_standard(cd):
            return StandardProbe(False, reason="zero valid cycles")
        return StandardProbe(True, layer_key=key, cycle_data=cd)

    return probe


def compute_delta_stats(
    delta_valid: np.ndarray,
) -> "tuple[float, float, float, float, int]":
    """Return (delta, delta_sd, delta_se, delta_2se, n) from per-cycle delta values.

    Shared between the pipeline delta path and runtime_delta so both cannot
    silently diverge.  Returns NaN sentinels for sd/se when n <= 1 (GUM §4.2.1:
    a single-point sample provides no estimate of dispersion).
    """
    n = len(delta_valid)
    delta = float(np.mean(delta_valid)) if n > 0 else float("nan")
    delta_sd = float(np.std(delta_valid, ddof=1)) if n > 1 else float("nan")
    delta_se = delta_sd / np.sqrt(n) if n > 1 else float("nan")
    return delta, delta_sd, delta_se, 2.0 * delta_se, n


#: Retained private alias: this helper is imported by name elsewhere in the
#: tree and by tests. It became public when the CSV writer needed to compute
#: delta statistics over an export-time cycle selection rather than reading
#: the stored full-run fields.
_compute_delta_stats = compute_delta_stats


def _compute_delta(
    sample_valid: np.ndarray,
    std_avg: float,
    prev_name: str,
    next_name: str,
    per_cycle: bool = True,
) -> DeltaResult:
    sample_mean = float(np.mean(sample_valid))

    # Per-cycle deltas are always computed for statistics; return them as an
    # output only when the caller requests it.
    delta_cycles = (sample_valid / std_avg - 1.0) * 1000.0
    delta, delta_sd, delta_se, delta_2se, n = compute_delta_stats(delta_cycles)

    return DeltaResult(
        delta=delta,
        delta_sd=delta_sd,
        delta_se=delta_se,
        delta_2se=delta_2se,
        n=n,
        sample_mean=sample_mean,
        std_mean=std_avg,
        prev_std_name=prev_name,
        next_std_name=next_name,
        delta_per_cycle=delta_cycles if per_cycle else None,
    )

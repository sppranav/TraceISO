"""Extended Engine C: uncertainty of a Pb-standard-calibrated absolute Pb ratio (combined Pb plan C05).

The reported quantity of a sample or independent QC observation is

    Y = X * K,   K = C / B,  B = sum_j w_j S_j        (local SSB, alternating or block average)
                 K = mean_j(C / S_j)                  (session mean of K)

where ``X`` is the mean of the target's Tl-normalized cycles on its active
support, ``S_j`` the mean of calibration standard ``j`` on its frozen calibration
support and ``C`` the accepted Pb reference. Every chain is replayed from
blank-corrected intensities by the same Pb-Tl model processing uses — measured Tl
ratio, Russell factor, optional Hg subtraction on 204Pb, per-cycle IIF — so the
sample, every standard and ``K`` are recomputed together and shared inputs enter
once, jointly (JCGM 100:2008 §5.1.3, §5.2.2).

Each source has one placement (C01 design §4):

* ``u_prec`` — SE of the target cycles, sensitivity 1 on ``lnY``;
* ``u_pb_cal_std_precision`` — SE of each standard mean, ``a_j = dlnY/dlnS_j``;
* ``u_pb_cal_reference`` — ``u(C)/C``; the certificate enters through ``K`` only, so
  ``u_crm`` does not apply and there is no output-level reference term;
* ``u_norm_ref`` — accepted Tl ratio, joint replay; it cancels exactly for a ratio
  without Hg subtraction because ``X`` and every ``S_j`` scale by ``R_Tl^gamma``;
* ``u_interf`` — Hg reference ratio, joint replay through every Hg subtraction;
* ``u_blank`` — every blank observation subtracted from any chain, one shared
  input with its channel covariance and the recorded channel weights.

Cycle noise is inside the two precision terms and is never replayed again.
Residual run-to-run scatter is reported only as a diagnostic (owner decision D3),
and layout mismatch and sample-specific transfer remain explicit, unquantified
coverage limitations. Unknown uncertainty is omitted with a reason, never zero.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from config.crm_schema import standard_uncertainty
from config.reference_materials import get_natural_ratio_record
from config.settings import CustomUncertaintyContributor, ProcessingConfig, UncertaintyConfig
from domain.corrections.interference import hg204_interference_correction
from domain.corrections.mass_bias import apply_iif_correction, calculate_f_factor, calculate_k_factors
from domain.elements.base import ElementConfig
from domain.layer_status import APPLIED
from domain.models import Sample, UncertaintyBudget, UncertaintyContributor
from domain.pb_calibration_records import (
    APPLIED_SESSION_MEAN_K,
    PB_CALIBRATION_BUDGET_ENGINE,
    PbCalibrationRecord,
    canonical_sha256,
    governing_calibration_record,
)
from domain.pb_correction_records import mask_sha256
from domain.uncertainty.contributors import (
    ContributorState,
    SampleContributorApplicability,
    build_custom_contributor_rows,
    inactive_reason_for_state,
    resolve_contributor_state,
)
from domain.uncertainty.pb_calibration_residual import (
    RESIDUAL_ESTIMATED,
    RESIDUAL_LABEL,
    RESIDUAL_REASONS,
    relative_sample_effect_variance,
)
from domain.uncertainty.pb_hg_ssb_propagation import _blank_covariance, _blank_roles, _is_psd, _role_weight
from domain.uncertainty.shared_engine import combine_and_build_budget_shared
from domain.uncertainty.welch_satterthwaite import effective_dof

#: Identity of this GUM model; carried on every row it adds.
PB_CALIBRATED_GUM_METHOD = "engine_c.pb_standard_calibration.gum.v1"

U_PREC = "u_prec"
U_STD_PRECISION = "u_pb_cal_std_precision"
U_REFERENCE = "u_pb_cal_reference"
U_NORM_REF = "u_norm_ref"
U_INTERF = "u_interf"
U_BLANK = "u_blank"
U_RESIDUAL = "u_pb_cal_residual"
U_LAYOUT = "u_pb_cal_layout_mismatch"
U_TRANSFER = "u_pb_cal_sample_transfer"

LAYOUT_LIMITATION_CODE = "calibration_layout_mismatch_unquantified"
TRANSFER_LIMITATION_CODE = "calibration_sample_transfer_unqualified"
EXACT_CANCELLATION_TL_CODE = "exact_cancellation_shared_tl_reference"
SELECTION_LIMITATION_CODE = "pb_calibrated_selection_effect_unquantified"
LINEARIZATION_LIMITATION_CODE = "pb_calibrated_frozen_support_linearization"

BLANK_EVALUATED = "evaluated"
BLANK_NOT_SUBTRACTED = "not_subtracted"
BLANK_INSUFFICIENT = "insufficient_blank_data"

#: Why a calibrated absolute budget or its Monte Carlo cannot be evaluated.
REFUSAL_REASONS: Mapping[str, str] = {
    "calibrated_pb_requires_se_precision": (
        "The calibrated route places sample and standard precision at the level of their means, so it "
        "needs u_prec_mode = SE; SD mode would add the cycle scatter again (owner decision Q-01)."
    ),
    "calibration_not_applied": "The Pb-standard calibration of this ratio is not applied.",
    "calibration_stale": "The Pb-standard calibration is not current for this session.",
    "calibration_chain_unavailable": "The Pb-Tl correction chain of this calibrated result cannot be replayed.",
    "calibration_member_unresolved": "A recorded calibration standard cannot be resolved by its observation identity.",
    "calibration_member_without_se": (
        "A calibration standard has one valid cycle, so its mean has no measured standard error (A009)."
    ),
    "calibration_support_changed": (
        "A calibration standard's cycle support no longer matches the processed calibration; reprocess the session."
    ),
    "calibration_replay_mismatch": "Replaying the recorded calibration does not reproduce the processed values.",
    "runtime_support_mismatch": "The requested cycle support is not a subset of the calibrated cycles.",
    "unresolved_blank_reference": "A blank subtracted from a calibration chain is not in this session.",
    "invalid_blank_covariance": (
        "A blank channel covariance is not positive semidefinite; no independent-channel substitute is used."
    ),
}

REPLAY_PARITY_REL = 1e-9

_REFERENCE = "JCGM 100:2008 §5.1.3, §5.2.2; combined Pb plan C05 (" + PB_CALIBRATED_GUM_METHOD + ")"


class CalibratedModelRefusal(ValueError):
    """The calibrated model cannot be evaluated; ``code`` is a key of :data:`REFUSAL_REASONS`."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        text = REFUSAL_REASONS[code] + (f" {detail}" if detail else "")
        super().__init__(f"{code}: {text}")


@dataclass(frozen=True, eq=False)
class ReplayChain:
    """One observation of the calibrated quantity, replayed on a fixed cycle support."""

    observation_id: str
    label: str
    role: str
    base: Dict[str, np.ndarray]
    mask: np.ndarray
    weight: float
    recorded_mean: Optional[float]
    se: Optional[float]
    n_valid: int


@dataclass(frozen=True, eq=False)
class CalibratedBlankInput:
    """One blank observation subtracted from one or more chains: a single shared input."""

    blank_observation_id: str
    blank_label: str
    channels: Tuple[str, ...]
    mean_vector: np.ndarray
    covariance_matrix: np.ndarray
    n_pairs: int
    #: ``(chain index, per-channel weight)`` for every chain that subtracted this blank.
    weights: Tuple[Tuple[int, Tuple[float, ...]], ...]

    @property
    def isotopes(self) -> Tuple[str, ...]:
        return self.channels


@dataclass(frozen=True, eq=False)
class CalibratedModel:
    """Frozen calibrated Pb-Tl model of one target ratio."""

    sample_name: str
    sample_observation_id: str
    ratio_name: str
    record: PbCalibrationRecord
    applied_mode: str
    reference_inputs: Any
    apply_hg: bool
    target: ReplayChain
    members: Tuple[ReplayChain, ...]
    c_value: float
    u_c: Optional[float]
    c_reason: str
    r_tl: float
    u_r_tl: Optional[float]
    tl_reason: str
    r_hg: Optional[float]
    u_r_hg: Optional[float]
    hg_reason: str
    blank_status: str
    blank_reason: str
    blanks: Tuple[CalibratedBlankInput, ...]
    x_mean: float
    k: float
    y: float
    target_se: Optional[float]

    @property
    def is_session(self) -> bool:
        return self.applied_mode == APPLIED_SESSION_MEAN_K

    @property
    def weights(self) -> np.ndarray:
        return np.array([m.weight for m in self.members], dtype=float)

    def relative_sensitivities(self) -> List[float]:
        """``a_j = dlnY/dlnS_j`` at the recorded standard means."""
        s = np.array([m.recorded_mean for m in self.members], dtype=float)
        if self.is_session:
            inverse = 1.0 / s
            return list(-inverse / float(np.sum(inverse)))
        bracket = float(self.weights @ s)
        return list(-self.weights * s / bracket)


def hg_reference_standard_uncertainty() -> Tuple[Optional[float], str]:
    """Standard uncertainty of the managed 204Hg/202Hg reference, or ``(None, reason)``."""
    record = get_natural_ratio_record("Hg", "204Hg/202Hg")
    if record is None:
        return None, "the 204Hg/202Hg reference record is missing"
    u = standard_uncertainty(record.uncertainty, record.k, record.uncertainty_semantics)
    if u is None or u <= 0.0:
        return None, (
            f"the 204Hg/202Hg reference uncertainty is {record.uncertainty_semantics or 'unassigned'}; "
            "it is omitted, not zero (Q-08)"
        )
    return float(u), ""


def _relative(new: float, reference: float) -> float:
    return abs(new - reference) / max(abs(reference), np.finfo(float).tiny)


def resolve_calibrated_model(
    sample: Sample,
    ratio_name: str,
    *,
    all_samples: Sequence[Sample],
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    processing_config: ProcessingConfig,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    ratio_mask: Optional[np.ndarray] = None,
    include_blanks: bool = True,
    require_current_session: bool = True,
) -> CalibratedModel:
    """Freeze the recorded calibration of one target ratio, verifying nominal replay parity.

    Raises :class:`CalibratedModelRefusal` when any recorded identity, support or
    value cannot be reproduced; nothing is searched again or substituted.
    """
    from domain.pb_standard_calibration import calibration_required_channels, calibration_support
    from domain.uncertainty.engine_external_pb_tl import (
        _get_pb_tl_base_intensities,
        _resolve_pb_tl_ratio_mask,
        _resolve_pb_tl_reference_inputs,
        resolve_pb_tl_normalization_standard_uncertainty,
    )

    record = governing_calibration_record(sample, ratio_name)
    if record is None or record.status != APPLIED:
        raise CalibratedModelRefusal("calibration_not_applied")
    if require_current_session:
        from domain.calibration_dependencies import build_calibration_dependencies
        from domain.pb_standard_calibration import resolve_correction_context

        try:
            from domain.pipeline import ProcessingPipeline

            ratio_definitions = ProcessingPipeline(
                element_config
            )._resolve_ratio_definitions_for_samples(list(all_samples))
            current_dependencies = build_calibration_dependencies(
                all_samples,
                element_symbol=element_config.symbol,
                ratio_definitions=ratio_definitions,
                settings=processing_config,
                cycle_ranges=cycle_ranges,
                correction_context=resolve_correction_context(
                    element_config,
                    processing_config,
                    ratio_definitions=ratio_definitions,
                ),
            )
        except Exception as exc:
            raise CalibratedModelRefusal(
                "calibration_stale", f"Current calibration dependencies could not be resolved: {exc}"
            ) from exc
        current_set = str(current_dependencies.get("calibration_input_digest") or "")
        current_chain = str(
            (current_dependencies.get("sample_chain_digests") or {}).get(sample.observation_id) or ""
        )
        if (
            not current_set
            or current_set != str(record.calibration_input_digest or "")
            or not current_chain
            or current_chain != str(record.sample_chain_digest or "")
        ):
            raise CalibratedModelRefusal(
                "calibration_stale",
                "The current standard windows, roles, materials, blanks, correction inputs, or target chain "
                "do not match the processed calibration. Reprocess before calculating uncertainty.",
            )
    if not getattr(processing_config, "apply_mass_bias_correction", False):
        raise CalibratedModelRefusal("calibration_chain_unavailable", "Tl normalization is not active.")
    try:
        ri = _resolve_pb_tl_reference_inputs(ratio_name, processing_config=processing_config, element_config=element_config)
    except ValueError as exc:
        raise CalibratedModelRefusal("calibration_chain_unavailable", str(exc)) from exc
    pair = (ri.target_numerator, ri.target_denominator)
    tl_pair = (ri.normalization_numerator, ri.normalization_denominator)
    apply_hg = bool(getattr(processing_config, "apply_hg_interference_correction", False)) and "204Pb" in pair
    if apply_hg and (ri.hg204_hg202_natural is None or ri.m202 is None or ri.m204_hg is None):
        raise CalibratedModelRefusal("calibration_chain_unavailable", "The Hg reference ratio is unresolved.")
    channels = set(pair) | set(tl_pair) | ({"202Hg"} if apply_hg else set())

    final = (getattr(sample, "pb_standard_corrected_ratios", None) or {}).get(ratio_name)
    target_mask = _resolve_pb_tl_ratio_mask(sample, ratio_name, cycle_ranges=cycle_ranges)
    if final is None or target_mask is None or len(target_mask) != len(final.mask):
        raise CalibratedModelRefusal("calibration_chain_unavailable", "No calibrated cycle layer.")
    target_mask = np.asarray(target_mask, dtype=bool) & np.asarray(final.mask, dtype=bool)
    if ratio_mask is not None:
        runtime = np.asarray(ratio_mask, dtype=bool)
        if runtime.shape != target_mask.shape:
            raise CalibratedModelRefusal("runtime_support_mismatch", "The runtime mask is not cycle-aligned.")
        target_mask &= runtime
    if not target_mask.any():
        raise CalibratedModelRefusal("runtime_support_mismatch", "No calibrated cycle remains on the requested support.")

    def chain_base(obs: Sample) -> Dict[str, np.ndarray]:
        base = _get_pb_tl_base_intensities(obs)
        missing = sorted(channels - set(base))
        if missing:
            raise CalibratedModelRefusal(
                "calibration_chain_unavailable", f"Observation '{obs.name}' lacks {', '.join(missing)}.",
            )
        return {c: base[c] for c in sorted(channels)}

    target_base = chain_base(sample)
    if any(len(v) != len(target_mask) for v in target_base.values()):
        raise CalibratedModelRefusal("calibration_chain_unavailable", "Target channels are not cycle-aligned.")
    x_cycles = np.asarray(sample.iif_corrected_ratios[ratio_name].values, dtype=float)[target_mask]
    target_se = float(np.std(x_cycles, ddof=1) / math.sqrt(len(x_cycles))) if len(x_cycles) >= 2 else None
    target = ReplayChain(
        observation_id=sample.observation_id, label=sample.name, role="target", base=target_base,
        mask=target_mask, weight=0.0, recorded_mean=None, se=target_se, n_valid=int(target_mask.sum()),
    )

    by_id: Dict[str, List[Sample]] = {}
    for candidate in all_samples:
        by_id.setdefault(candidate.observation_id, []).append(candidate)
    grouped: Dict[str, Dict[str, Any]] = {}
    for member in record.members:
        entry = grouped.setdefault(member.observation_id, {"member": member, "weight": 0.0})
        entry["weight"] += member.weight
    members: List[ReplayChain] = []
    for observation_id, entry in grouped.items():
        member = entry["member"]
        matches = by_id.get(observation_id, [])
        if len(matches) != 1:
            raise CalibratedModelRefusal("calibration_member_unresolved", f"'{member.label}' ({observation_id}).")
        obs = matches[0]
        if member.se is None:
            raise CalibratedModelRefusal("calibration_member_without_se", f"'{member.label}' ({observation_id}).")
        support = calibration_support(
            obs, ratio_name, required_channels=calibration_required_channels(obs, ratio_name, pair, tl_pair),
            window=member.window,
        )
        if support is None or mask_sha256(support.mask) != member.support_mask_sha256:
            raise CalibratedModelRefusal("calibration_support_changed", f"'{member.label}' ({observation_id}).")
        base = chain_base(obs)
        members.append(ReplayChain(
            observation_id=observation_id, label=member.label, role=member.side, base=base,
            mask=np.asarray(support.mask, dtype=bool).copy(), weight=float(entry["weight"]),
            recorded_mean=float(member.s_mean), se=float(member.se), n_valid=int(member.n_valid),
        ))

    reference = dict(record.reference or {})
    c_value = float(reference["value"])
    u_c = standard_uncertainty(reference.get("uncertainty"), reference.get("k"), reference.get("uncertainty_semantics"))
    c_reason = "" if u_c is not None else (
        f"the {reference.get('material', 'reference')} {ratio_name} uncertainty is "
        f"{reference.get('uncertainty_semantics') or 'unassigned'}; it is omitted, not zero"
    )
    try:
        u_r_tl: Optional[float] = float(resolve_pb_tl_normalization_standard_uncertainty(
            uncertainty_config, ri.tl_norm_ratio_name,
        ))
        tl_reason = ""
    except ValueError as exc:
        u_r_tl, tl_reason = None, f"{exc} It is omitted, not zero (Q-07)."
    u_r_hg: Optional[float] = None
    hg_reason = ""
    if apply_hg:
        u_r_hg, hg_reason = hg_reference_standard_uncertainty()

    model = CalibratedModel(
        sample_name=sample.name, sample_observation_id=sample.observation_id, ratio_name=ratio_name,
        record=record, applied_mode=record.applied_mode, reference_inputs=ri, apply_hg=apply_hg,
        target=target, members=tuple(members), c_value=c_value, u_c=u_c, c_reason=c_reason,
        r_tl=float(ri.tl_norm_value), u_r_tl=u_r_tl, tl_reason=tl_reason,
        r_hg=float(ri.hg204_hg202_natural) if apply_hg else None, u_r_hg=u_r_hg, hg_reason=hg_reason,
        blank_status=BLANK_NOT_SUBTRACTED, blank_reason="", blanks=(),
        x_mean=float("nan"), k=float("nan"), y=float("nan"), target_se=target_se,
    )
    nominal = evaluate(model, batch=1)
    for index, chain in enumerate(model.members):
        replayed = float(nominal["s"][0, index])
        if not math.isfinite(replayed) or _relative(replayed, chain.recorded_mean) > REPLAY_PARITY_REL:
            raise CalibratedModelRefusal(
                "calibration_replay_mismatch",
                f"Standard '{chain.label}' replays to {replayed!r}, recorded {chain.recorded_mean!r}.",
            )
    k_nominal, x_nominal = float(nominal["k"][0]), float(nominal["x"][0])
    if not math.isfinite(k_nominal) or _relative(k_nominal, float(record.k_applied)) > REPLAY_PARITY_REL:
        raise CalibratedModelRefusal("calibration_replay_mismatch", f"K replays to {k_nominal!r}, recorded {record.k_applied!r}.")
    stored_x = float(np.mean(np.asarray(final.values, dtype=float)[target_mask])) / float(record.k_applied)
    if not math.isfinite(x_nominal) or _relative(x_nominal, stored_x) > REPLAY_PARITY_REL:
        raise CalibratedModelRefusal("calibration_replay_mismatch", f"The target replays to {x_nominal!r}, stored {stored_x!r}.")

    blank_status, blank_reason, blanks = BLANK_NOT_SUBTRACTED, "", ()
    if include_blanks:
        blank_status, blank_reason, blanks = _resolve_blank_inputs(
            [sample] + [by_id[m.observation_id][0] for m in model.members],
            sorted(channels), by_id, uncertainty_config, processing_config, cycle_ranges,
        )
    object.__setattr__(model, "blank_status", blank_status)
    object.__setattr__(model, "blank_reason", blank_reason)
    object.__setattr__(model, "blanks", blanks)
    object.__setattr__(model, "x_mean", x_nominal)
    object.__setattr__(model, "k", k_nominal)
    object.__setattr__(model, "y", x_nominal * k_nominal)
    return model


def _resolve_blank_inputs(chains, channels, by_id, uncertainty_config, processing_config, cycle_ranges):
    weights: Dict[str, Dict[int, Dict[str, float]]] = {}
    for index, obs in enumerate(chains):
        roles = _blank_roles(obs, processing_config)
        for role, blank_id in roles:
            for channel in channels:
                w = _role_weight(obs, channel, role, len(roles))
                if w > 0.0:
                    per_chain = weights.setdefault(blank_id, {}).setdefault(index, {})
                    per_chain[channel] = per_chain.get(channel, 0.0) + w
    if not weights:
        return BLANK_NOT_SUBTRACTED, "no blank was subtracted from the target or any calibration standard", ()
    inputs: List[CalibratedBlankInput] = []
    for blank_id in sorted(weights):
        matches = by_id.get(blank_id, [])
        if len(matches) != 1:
            raise CalibratedModelRefusal("unresolved_blank_reference", f"Blank observation '{blank_id}'.")
        blank = matches[0]
        used = tuple(sorted({c for per_chain in weights[blank_id].values() for c in per_chain}))
        resolved = _blank_covariance(blank, used, uncertainty_config, cycle_ranges)
        if resolved is None:
            return BLANK_INSUFFICIENT, f"blank '{blank.name}' has fewer than two paired cycles on {', '.join(used)}", ()
        means, covariance, n_pairs, method = resolved
        covariance = np.asarray(covariance, dtype=float)
        if not np.all(np.isfinite(covariance)) or not _is_psd(covariance):
            raise CalibratedModelRefusal("invalid_blank_covariance", f"Blank '{blank.name}' ({method}).")
        inputs.append(CalibratedBlankInput(
            blank_observation_id=blank_id, blank_label=blank.name, channels=used,
            mean_vector=np.asarray(means, dtype=float), covariance_matrix=covariance, n_pairs=int(n_pairs),
            weights=tuple(
                (index, tuple(float(per_chain.get(c, 0.0)) for c in used))
                for index, per_chain in sorted(weights[blank_id].items())
            ),
        ))
    return BLANK_EVALUATED, "", tuple(inputs)


def _chain_means(model: CalibratedModel, chain: ReplayChain, batch: int, r_tl, r_hg, shifts) -> np.ndarray:
    """Tl-normalized means of one chain for ``batch`` input sets, NaN where a supported cycle is not finite."""
    ri = model.reference_inputs
    n = len(chain.mask)

    def channel(name: str) -> np.ndarray:
        values = np.broadcast_to(chain.base[name], (batch, n))
        shift = shifts.get(name)
        return values - shift[:, None] if shift is not None else values

    r_tl_column = r_tl[:, None]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        measured = channel(ri.normalization_numerator) / channel(ri.normalization_denominator)
        numerator = channel(ri.target_numerator)
        denominator = channel(ri.target_denominator)
        if model.apply_hg:
            f = calculate_f_factor(measured, r_tl_column, ri.m_norm_num, ri.m_norm_den)
            pb204 = np.array(channel("204Pb"), dtype=float)
            # The interference at unit reference ratio; the primitive is linear in that ratio.
            _unused, unit = hg204_interference_correction(
                pb204=pb204, hg202=np.array(channel("202Hg"), dtype=float), f_tl=np.asarray(f, dtype=float),
                hg204_hg202_natural=1.0, m202=ri.m202, m204_hg=ri.m204_hg,
            )
            corrected = pb204 - r_hg[:, None] * unit
            if ri.target_numerator == "204Pb":
                numerator = corrected
            else:
                denominator = corrected
        ratio = numerator / denominator
        k_factors = calculate_k_factors(
            normalization_ratio_measured=measured, normalization_ratio_reference=r_tl_column,
            normalization_numerator_mass=ri.m_norm_num, target_numerator_mass=ri.target_num_mass,
            normalization_denominator_mass=ri.m_norm_den, target_denominator_mass=ri.target_den_mass,
        ).target_k
        supported = apply_iif_correction(ratio, k_factors)[:, chain.mask]
        means = supported.mean(axis=1)
    means[~np.all(np.isfinite(supported), axis=1)] = np.nan
    return means


def evaluate(
    model: CalibratedModel,
    *,
    batch: int,
    r_tl: Optional[np.ndarray] = None,
    r_hg: Optional[np.ndarray] = None,
    c: Optional[np.ndarray] = None,
    blank_deltas: Optional[Mapping[str, np.ndarray]] = None,
    target_add: Optional[np.ndarray] = None,
    member_add: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Evaluate ``batch`` input sets through every chain and the calibration functional.

    ``blank_deltas[id]`` is ``(batch, n_channels)``: a change of that blank's mean, which
    lowers the corrected intensity of every chain that subtracted it by weight times
    the change. ``target_add``/``member_add`` are mean-level precision perturbations.
    """
    full = lambda value: np.full(batch, float(value))  # noqa: E731
    r_tl = full(model.r_tl) if r_tl is None else np.asarray(r_tl, dtype=float)
    r_hg = (full(model.r_hg) if model.r_hg is not None else full(0.0)) if r_hg is None else np.asarray(r_hg, dtype=float)
    c = full(model.c_value) if c is None else np.asarray(c, dtype=float)
    chains = (model.target,) + model.members
    shifts: List[Dict[str, np.ndarray]] = [{} for _ in chains]
    for blank in model.blanks:
        deltas = (blank_deltas or {}).get(blank.blank_observation_id)
        if deltas is None:
            continue
        for index, channel_weights in blank.weights:
            for k, channel in enumerate(blank.channels):
                if channel_weights[k]:
                    shifts[index][channel] = shifts[index].get(channel, 0.0) + channel_weights[k] * deltas[:, k]
    x = _chain_means(model, model.target, batch, r_tl, r_hg, shifts[0])
    if target_add is not None:
        x = x + target_add
    s = np.column_stack([
        _chain_means(model, chain, batch, r_tl, r_hg, shifts[i + 1]) for i, chain in enumerate(model.members)
    ])
    if member_add is not None:
        s = s + member_add
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if model.is_session:
            bracket = np.full(batch, np.nan)
            k = np.mean(c[:, None] / s, axis=1)
        else:
            bracket = s @ model.weights
            k = c / bracket
        y = x * k
    return {"x": x, "s": s, "k": k, "y": y, "bracket": bracket}


def calibration_identity_for_record(record: PbCalibrationRecord) -> str:
    """Record-bound identity of a calibrated result, computable from the record alone (freshness checks)."""
    return canonical_sha256({"method": PB_CALIBRATED_GUM_METHOD, "calibration_record_sha256": record.sha256})[:16]


def model_input_digest(model: CalibratedModel, drawn: Mapping[str, Any]) -> str:
    """Digest of every resolved input a Monte Carlo draw depends on."""
    hasher = hashlib.sha256()

    def add(value: object) -> None:
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"\x00")

    for value in (PB_CALIBRATED_GUM_METHOD, model.record.sha256, model.ratio_name, model.applied_mode,
                  model.apply_hg, float(model.r_tl).hex(), float(model.c_value).hex()):
        add(value)
    add(np.ascontiguousarray(model.target.mask, dtype=np.uint8).tobytes().hex())
    for chain in (model.target,) + model.members:
        add(chain.observation_id)
        add(float(chain.weight).hex())
        for name in sorted(chain.base):
            hasher.update(np.ascontiguousarray(chain.base[name], dtype=np.float64).tobytes())
    for blank in model.blanks:
        add(blank.blank_observation_id)
        add("|".join(blank.channels))
        for array in (blank.mean_vector, blank.covariance_matrix):
            hasher.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        add(blank.weights)
    for key in sorted(drawn):
        add(key)
        add(drawn[key])
    return hasher.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# GUM budget
# --------------------------------------------------------------------------- #

def _unavailable(code: str, detail: str, *, output_mode: str, ratio_value: float, n_cycles: int) -> UncertaintyBudget:
    return UncertaintyBudget(
        engine=PB_CALIBRATION_BUDGET_ENGINE,
        output_mode=output_mode,
        budget_scope="unavailable",
        scope_note=f"Unavailable ({code}): {REFUSAL_REASONS[code]}" + (f" {detail}" if detail else ""),
        ratio_value=ratio_value,
        n_cycles=n_cycles,
    )


def _joint_replay_rows(model: CalibratedModel, *, tl: bool, hg: bool, blanks: bool):
    """Batched ±u joint replays: returns ``(y, index)`` with index naming each row."""
    rows: List[Tuple[str, Any, float]] = [("nominal", None, 0.0)]
    if tl:
        rows += [("tl", +1, model.u_r_tl), ("tl", -1, model.u_r_tl)]
    if hg:
        rows += [("hg", +1, model.u_r_hg), ("hg", -1, model.u_r_hg)]
    if blanks:
        for blank in model.blanks:
            for k, _channel in enumerate(blank.channels):
                sigma = math.sqrt(max(float(blank.covariance_matrix[k, k]), 0.0))
                if sigma > 0.0:
                    rows += [(("blank", blank.blank_observation_id, k), +1, sigma),
                             (("blank", blank.blank_observation_id, k), -1, sigma)]
    batch = len(rows)
    r_tl = np.full(batch, model.r_tl)
    r_hg = np.full(batch, model.r_hg if model.r_hg is not None else 0.0)
    deltas = {b.blank_observation_id: np.zeros((batch, len(b.channels))) for b in model.blanks}
    for i, (kind, sign, step) in enumerate(rows):
        if kind == "tl":
            r_tl[i] += sign * step
        elif kind == "hg":
            r_hg[i] += sign * step
        elif isinstance(kind, tuple):
            deltas[kind[1]][i, kind[2]] = sign * step
    y = evaluate(model, batch=batch, r_tl=r_tl, r_hg=r_hg, blank_deltas=deltas)["y"]
    return y, rows


def compute_budget_pb_calibrated(
    sample: Sample,
    ratio_name: str,
    *,
    all_samples: Sequence[Sample],
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    processing_config: ProcessingConfig,
    ratio_values: Optional[np.ndarray] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    custom_contributor_library: Optional[Dict[str, List[CustomUncertaintyContributor]]] = None,
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
) -> UncertaintyBudget:
    """GUM budget of an applied Pb-standard-calibrated absolute ratio."""
    from domain.uncertainty.eligibility import pb_calibration_budget_guard

    output_mode = "absolute_ratio"
    guard = pb_calibration_budget_guard(
        sample, ratio_name, str(uncertainty_config.output_mode or ""), extended_engine_c=True,
    )
    if guard is not None:
        return guard
    final = (getattr(sample, "pb_standard_corrected_ratios", None) or {}).get(ratio_name)
    if ratio_values is None:
        ratio_values = final.valid_values if final is not None else np.array([])
    values = np.asarray(ratio_values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(len(values))
    y_mean = float(np.mean(values)) if n else 0.0
    if n < 2:
        return UncertaintyBudget(
            engine=PB_CALIBRATION_BUDGET_ENGINE, output_mode=output_mode, budget_scope="insufficient_data",
            scope_note=f"Insufficient valid cycles ({n}) for uncertainty computation - need at least 2.",
            ratio_value=y_mean, n_cycles=n,
        )
    if str(getattr(uncertainty_config, "u_prec_mode", "se")).strip().lower() == "sd":
        return _unavailable("calibrated_pb_requires_se_precision", "", output_mode=output_mode, ratio_value=y_mean, n_cycles=n)

    custom_names = {d.name for defs in (custom_contributor_library or {}).values() for d in defs}
    applicability = SampleContributorApplicability.from_sample(
        sample, known_profiles=set(profile_defaults.keys()) if profile_defaults is not None else None,
    )

    def state_of(name: str) -> ContributorState:
        return resolve_contributor_state(
            name=name, uncertainty_config=uncertainty_config, sample=sample, element_symbol=element_config.symbol,
            custom_contributor_names=custom_names, profile_defaults=profile_defaults,
        )

    def gate(name: str, available: bool = True, missing: str = "", not_applicable: str = "") -> Dict[str, Any]:
        state = state_of(name)
        if state == ContributorState.ACTIVE and not_applicable:
            return {"state": ContributorState.NOT_APPLICABLE.value, "inactive_reason": not_applicable}
        if state == ContributorState.ACTIVE and not available:
            return {"state": ContributorState.MISSING_DATA.value,
                    "inactive_reason": missing or inactive_reason_for_state(ContributorState.MISSING_DATA, contributor_name=name)}
        return {"state": state.value,
                "inactive_reason": inactive_reason_for_state(state, contributor_name=name, profile=applicability.profile)}

    try:
        model = resolve_calibrated_model(
            sample, ratio_name, all_samples=all_samples, element_config=element_config,
            uncertainty_config=uncertainty_config, processing_config=processing_config, cycle_ranges=cycle_ranges,
            include_blanks=state_of(U_BLANK) == ContributorState.ACTIVE,
        )
    except CalibratedModelRefusal as refusal:
        return _unavailable(refusal.code, refusal.detail, output_mode=output_mode, ratio_value=y_mean, n_cycles=n)
    if model.target.n_valid != n:
        return _unavailable(
            "runtime_support_mismatch", f"{model.target.n_valid} calibrated cycles on the support, {n} values supplied.",
            output_mode=output_mode, ratio_value=y_mean, n_cycles=n,
        )
    if _relative(model.y, y_mean) > REPLAY_PARITY_REL:
        return _unavailable(
            "calibration_replay_mismatch", f"Replayed Y {model.y!r}, reported mean {y_mean!r}.",
            output_mode=output_mode, ratio_value=y_mean, n_cycles=n,
        )

    scale = abs(y_mean)
    rows: List[UncertaintyContributor] = []

    def row(name, display, rel, type_ab, dof, description, gate_kwargs, **extra):
        rel = 0.0 if rel is None or not math.isfinite(rel) else float(rel)
        rows.append(UncertaintyContributor(
            name=name, display_name=display, value_abs=rel * scale, value_rel_permil=1000.0 * rel,
            type_ab=type_ab, degrees_of_freedom=dof, percentage_contribution=0.0,
            description=f"{description} Reference: {_REFERENCE}.", reference=_REFERENCE, **gate_kwargs, **extra,
        ))

    # P1 target precision (SE of the reported cycles; the scale K cancels in the relative term).
    se_y = float(np.std(values, ddof=1) / math.sqrt(n))
    row(U_PREC, "Sample measurement precision (Type A)", se_y / scale, "A", float(n - 1),
        "Standard error of the mean of the calibrated cycles on the active support; enters the Tl-normalized "
        "mean X once before calibration.", gate(U_PREC, se_y > 0.0, "u_prec is zero on the active support."))

    # P2 standard precision, each recorded standard mean once.
    a = model.relative_sensitivities()
    terms = [(abs(a_j) * m.se / m.recorded_mean, float(m.n_valid - 1)) for a_j, m in zip(a, model.members)]
    u_std_rel = math.sqrt(sum(t * t for t, _ in terms))
    row(U_STD_PRECISION, "Calibration standard precision (Type A)", u_std_rel, "A",
        effective_dof([(t, nu) for t, nu in terms if t > 0.0]) if u_std_rel > 0 else float("inf"),
        "Standard error of each recorded calibration standard mean on its frozen support, through K: "
        "u_rel^2 = sum_j (a_j * se_j / S_j)^2 with a_j = dlnY/dlnS_j = "
        + ", ".join(f"{m.label}: {a_j:.6g}" for a_j, m in zip(a, model.members)) + ".",
        gate(U_STD_PRECISION, u_std_rel > 0.0, "Every calibration standard has zero standard error."))

    # P3 accepted Pb reference, once through K.
    reference = dict(model.record.reference or {})
    row(U_REFERENCE, "Accepted Pb reference value in K (Type B)",
        (model.u_c / model.c_value) if model.u_c is not None else None, "B", float("inf"),
        f"{reference.get('material', '')} {ratio_name} = {model.c_value:.10g}"
        + (" (derived ratio, k = 1)" if reference.get("derived") else "")
        + "; one input shared by every standard's K, relative sensitivity 1 in every mode.",
        gate(U_REFERENCE, model.u_c is not None and model.u_c > 0.0, f"u_pb_cal_reference omitted: {model.c_reason}."))
    row("u_crm", "CRM certified value (Type B)", 0.0, "B", float("inf"),
        "The certificate enters the calibrated result through K only.",
        gate("u_crm", not_applicable=(
            "u_crm does not apply: the accepted reference enters the calibrated result once, through K "
            "(u_pb_cal_reference); an output-level CRM term would count it twice."
        )))

    tl_active = model.apply_hg and model.u_r_tl is not None and model.u_r_tl > 0.0 and state_of(U_NORM_REF) == ContributorState.ACTIVE
    hg_active = model.apply_hg and model.u_r_hg is not None and model.u_r_hg > 0.0 and state_of(U_INTERF) == ContributorState.ACTIVE
    blank_active = model.blank_status == BLANK_EVALUATED and bool(model.blanks)
    y_rows, index = _joint_replay_rows(model, tl=tl_active, hg=hg_active, blanks=blank_active)
    lookup = {(kind, sign): i for i, (kind, sign, _step) in enumerate(index)}

    def half_difference(kind) -> Optional[float]:
        up, down = y_rows[lookup[(kind, +1)]], y_rows[lookup[(kind, -1)]]
        return abs(up - down) / 2.0 if math.isfinite(up) and math.isfinite(down) else None

    # P4 Tl reference, joint through the target and every standard.
    if not model.apply_hg:
        row(U_NORM_REF, "Tl normalization ratio uncertainty (Type B)", 0.0, "B", float("inf"),
            "Accepted Tl ratio shared by the target and every standard chain.",
            gate(U_NORM_REF, not_applicable=(
                f"{EXACT_CANCELLATION_TL_CODE}: without Hg subtraction the Tl reference scales X and every S_j "
                f"by R_Tl^gamma, so it cancels exactly in X*K for {ratio_name}."
            )))
    else:
        u_tl = half_difference("tl") if tl_active else None
        row(U_NORM_REF, "Tl normalization ratio uncertainty (Type B)", (u_tl / scale) if u_tl is not None else None, "B",
            float("inf"),
            "Accepted Tl ratio replayed jointly through the Hg subtraction and IIF of the target and every "
            "standard, with K recomputed; the common scaling cancels and only the Hg-dependent part remains.",
            gate(U_NORM_REF, u_tl is not None,
                 f"u_norm_ref omitted: {model.tl_reason or 'the joint replay was not finite'}"))

    # P5 Hg reference, joint.
    if not model.apply_hg:
        row(U_INTERF, "204Hg interference correction (Type B)", 0.0, "B", float("inf"), "Hg reference ratio.",
            gate(U_INTERF, not_applicable=f"u_interf does not apply: no Hg subtraction enters {ratio_name}."))
    else:
        u_hg = half_difference("hg") if hg_active else None
        row(U_INTERF, "204Hg interference correction (Type B)", (u_hg / scale) if u_hg is not None else None, "B",
            float("inf"),
            "204Hg/202Hg reference ratio replayed jointly through the Hg subtraction of the target and every "
            "standard (unequal Hg/Pb does not cancel), with K recomputed.",
            gate(U_INTERF, u_hg is not None,
                 f"u_interf omitted: {model.hg_reason or 'the joint replay was not finite'}."))

    # P6 blanks, each blank observation once with its channel covariance.
    blank_rel: Optional[float] = None
    blank_dof = float("inf")
    blank_description = "Blank observations subtracted from the target or any standard, each one shared input."
    if blank_active:
        components = []
        parts = []
        for blank in model.blanks:
            gradient = np.zeros(len(blank.channels))
            for k, _channel in enumerate(blank.channels):
                key = ("blank", blank.blank_observation_id, k)
                if (key, +1) in lookup:
                    sigma = math.sqrt(float(blank.covariance_matrix[k, k]))
                    up, down = y_rows[lookup[(key, +1)]], y_rows[lookup[(key, -1)]]
                    gradient[k] = (up - down) / (2.0 * sigma)
            variance = float(gradient @ blank.covariance_matrix @ gradient)
            if not math.isfinite(variance):
                variance = float("nan")
            components.append((math.sqrt(max(variance, 0.0)), float(max(blank.n_pairs - 1, 1)), variance))
            users = ", ".join("target" if i == 0 else model.members[i - 1].label for i, _w in blank.weights)
            parts.append(f"{blank.blank_label} ({', '.join(blank.channels)}; subtracted from {users})")
        if all(math.isfinite(v) for _u, _nu, v in components):
            blank_rel = math.sqrt(sum(max(v, 0.0) for _u, _nu, v in components)) / scale
            positive = [(u, nu) for u, nu, _v in components if u > 0.0]
            blank_dof = effective_dof(positive) if positive else float("inf")
        blank_description += " " + "; ".join(parts) + "."
    if model.blank_status == BLANK_NOT_SUBTRACTED:
        blank_gate = gate(
            U_BLANK,
            not_applicable="u_blank does not apply: no blank was subtracted from the target or calibration standards.",
        )
    else:
        blank_gate = gate(
            U_BLANK, blank_rel is not None,
            f"u_blank omitted: {model.blank_reason or 'the joint blank replay was not finite'}.",
        )
    row(U_BLANK, "Blank correction (Type A)", blank_rel, "A", blank_dof, blank_description, blank_gate)

    # P8 residual repeatability: diagnostic only (owner decision D3; Q-03 for not_resolved).
    diagnostic = dict(model.record.residual_diagnostic or {})
    if not diagnostic:
        residual_text = (
            f"not_resolved (record_predates_residual_diagnostic): {RESIDUAL_REASONS['record_predates_residual_diagnostic']}"
        )
    elif diagnostic.get("status") == RESIDUAL_ESTIMATED:
        variance = relative_sample_effect_variance(diagnostic, a)
        residual_text = (
            f"estimated as {RESIDUAL_LABEL}: tau2 = {diagnostic['tau2']:.6g} from {diagnostic['n_rows']} held-out "
            f"rows of {diagnostic['n_pool']} standards (Satterthwaite nu_res = {diagnostic['nu_res']:.4g}); "
            f"the sample-effect form tau2*(1 + sum a_j^2) would give u_rel = {math.sqrt(variance) * 1000.0:.6g} permil"
        )
    else:
        residual_text = f"not_resolved ({diagnostic.get('reason_code', '')}): {diagnostic.get('reason', '')}"
    residual_reason = (
        f"Residual calibration repeatability is a diagnostic only, {residual_text.rstrip('.')}. The held-out rows are dependent "
        "and no effective degrees-of-freedom model for them is validated (owner decision D3), so it is not in the "
        "combined uncertainty or Monte Carlo; it may include layout mismatch and assumes standards' run-to-run "
        "behaviour transfers to samples (Q-04). Omitted, not zero."
    )
    rows.append(UncertaintyContributor(
        name=U_RESIDUAL, display_name="Residual calibration repeatability (diagnostic)", value_abs=0.0,
        value_rel_permil=0.0, type_ab="A", degrees_of_freedom=float("inf"), percentage_contribution=0.0,
        description=f"{residual_reason} Reference: {_REFERENCE}; ISO 5725-2 variance components.",
        reference=_REFERENCE, state=ContributorState.NO_APPROVED_MODEL.value, inactive_reason=residual_reason,
    ))
    row("u_std_repeatability", "Reference-material repeatability (Type A)", 0.0, "A", float("inf"),
        "Full scatter of standard means.",
        gate("u_std_repeatability", not_applicable=(
            "u_std_repeatability does not apply on the calibrated route: its full scatter already contains the "
            "standard-mean precision placed in u_pb_cal_std_precision; residual scatter is u_pb_cal_residual."
        )))
    row("u_kappa_drift", "Instrumental drift (Type B)", 0.0, "B", float("inf"), "Between-standard drift.",
        gate("u_kappa_drift", not_applicable=(
            "u_kappa_drift does not apply on the calibrated route: drift correction is suppressed and "
            "consecutive-standard scatter is the u_pb_cal_residual information."
        )))

    layout = dict(model.record.layout or {})
    layout_reason = (
        f"{LAYOUT_LIMITATION_CODE}: the calibration functional's layout or drift mismatch at this sample is not "
        f"quantified (Q-21); layout {', '.join(f'{k} = {v}' for k, v in sorted(layout.items())) or 'not recorded'}. "
        "The combined uncertainty does not cover it."
    )
    rows.append(UncertaintyContributor(
        name=U_LAYOUT, display_name="Calibration layout mismatch (not quantified)", value_abs=0.0,
        value_rel_permil=0.0, type_ab="B", degrees_of_freedom=float("inf"), percentage_contribution=0.0,
        description=f"{layout_reason} Reference: {_REFERENCE}; remediation design §5.4.", reference=_REFERENCE,
        state=ContributorState.NO_APPROVED_MODEL.value, inactive_reason=layout_reason,
    ))
    transfer_reason = (
        f"{TRANSFER_LIMITATION_CODE}: calibration against {reference.get('material', 'the reference')} does not by "
        "itself show that sample-specific Pb/Tl fractionation differences are removed (Q-10); no approved "
        "uncertainty model exists. The combined uncertainty does not cover it."
    )
    rows.append(UncertaintyContributor(
        name=U_TRANSFER, display_name="Sample-specific transfer (not qualified)", value_abs=0.0,
        value_rel_permil=0.0, type_ab="B", degrees_of_freedom=float("inf"), percentage_contribution=0.0,
        description=f"{transfer_reason} Reference: {_REFERENCE}.", reference=_REFERENCE,
        state=ContributorState.NO_APPROVED_MODEL.value, inactive_reason=transfer_reason,
    ))

    from domain.uncertainty.sr_sample_values import build_processed_material_contributors

    rows.extend(build_processed_material_contributors(
        sample, uncertainty_config, y_mean, gate,
    ))

    rows.extend(build_custom_contributor_rows(
        sample=sample, element_symbol=element_config.symbol, ratio_mean=y_mean,
        custom_contributor_library=custom_contributor_library or {},
    ))
    budget = combine_and_build_budget_shared(
        engine=PB_CALIBRATION_BUDGET_ENGINE, contributors=rows, ratio_mean=y_mean, n_cycles=n,
        uncertainty_config=uncertainty_config, use_abs_dof=False,
    )
    if budget.budget_scope not in {"unavailable", "insufficient_data"}:
        budget.budget_scope = "limited"
        budget.scope_note = (
            "Limited to implemented components: residual calibration repeatability, "
            "layout mismatch and sample-specific transfer are not quantified (Q-006). "
            "Laboratory coverage and owner acceptance remain open."
        )
    budget.replay_input_digest = (
        f"{calibration_identity_for_record(model.record)}.{model_input_digest(model, {})}"
    )
    budget.coverage_limitations = [
        {
            "code": SELECTION_LIMITATION_CODE,
            "explanation": (
                "The effect of selecting or excluding invalid cycles is not quantified in the combined "
                "uncertainty (Q-26)."
            ),
        },
        {
            "code": LINEARIZATION_LIMITATION_CODE,
            "explanation": (
                "GUM sensitivities use finite perturbations on the accepted frozen cycle support; support "
                "selection and nonlinear boundary changes are not included (Q-46)."
            ),
        },
    ]
    return budget

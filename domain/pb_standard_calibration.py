"""Pb-standard calibration of the Tl-only Pb layer (combined Pb plan C04).

After Pb-Tl external normalization, each Pb ratio ``r`` of a sample or an
independent QC observation is calibrated against explicitly assigned standards
of one accepted reference material ``C_r``:

    K_j   = C / S_j                       S_j = mean of standard j's valid calibration cycles
    local SSB, alternating:   B = (S_prev + S_next) / 2,           K = C / B
    local SSB, block average: B = (B_prev + B_next) / 2,           K = C / B
                              B_side = equal mean of member means, member weight 1/(2 n_side)
    session:                  K = mean_j(C / S_j)                  (never C / mean(S))
    final:                    Y_i = X_i * K
    delta:                    1000 (X_i / B - 1)   or   1000 (X_i * mean_j(1/S_j) - 1)
                              which equals 1000 (Y_i / C - 1) and does not depend on C

``X`` is the Tl-only layer ``iif_corrected_ratios``; it is consumed, never
rewritten. Local brackets use the existing ``ssb_mode`` convention; there is no
average of two K factors, no time interpolation and no session fallback.

One eligibility predicate governs the bracket search, block formation and the
session pool: an explicit ``calibration_standard`` role and the selected
material on a non-blank, non-excluded observation with a Tl-only layer and at
least the configured minimum of valid calibration cycles. Sample type is
provenance only. Search walks the active run order, passes corrected
observations and blanks silently, and records every candidate it tested and did
not use. Any active observation that is not eligible breaks a block.

Calibration cycles (owner decision, 2026-09-11): a standard's support is its
Tl-only mask intersected with its cycle window; a sample's is its Tl-only mask.
On that support a Tl-only value that is non-finite on its channel support, or
that is zero or negative, is excluded in the calibration layer only, with its
reason, count, fraction and a review flag. Nothing is clipped or replaced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from config.reference_materials import get_crm_ratios, get_crm_records
from domain.calibration_dependencies import observation_role, observation_window
from domain.layer_status import APPLIED, NOT_REQUESTED, REVIEW_INVALID_CYCLE_EXCLUSION, UNAVAILABLE
from domain.models import CycleData, Sample
from domain.pb_calibration_records import (
    APPLIED_LOCAL_ALTERNATING,
    APPLIED_LOCAL_BLOCK,
    APPLIED_SESSION_MEAN_K,
    CALIBRATION_NOT_REQUESTED_REASONS,
    CALIBRATION_UNAVAILABLE_REASONS,
    DELTA_ESTIMATORS,
    MODE_SESSION_MEAN_K,
    PB_CALIBRATED_DELTA_RECORD_FAMILY,
    PB_CALIBRATION_QUALITY_KEY,
    PB_CALIBRATION_RECORD_FAMILY,
    PB_CALIBRATION_SCHEMA_NAME,
    PB_CALIBRATION_SCHEMA_VERSION,
    PB_CALIBRATION_SEMANTICS,
    ROLE_CALIBRATION_STANDARD,
    ROLE_NOT_USED,
    ROLE_UNASSIGNED_STANDARD,
    TARGET_ROLES,
    CalibrationMember,
    PbCalibratedDeltaRecord,
    PbCalibrationRecord,
    SkippedObservation,
    canonical_sha256,
)
from domain.pb_correction_records import hg_records, mask_sha256
from domain.ratio_selection import get_processing_ratio_data
from domain.ratio_utils import normalize_ratio_name
from domain.uncertainty.pb_calibration_residual import PoolStandard, calibration_residual_diagnostic

_NOT_REQUESTED_REASON_BY_ROLE = {
    ROLE_CALIBRATION_STANDARD: "calibration_standard_not_self_corrected",
    ROLE_UNASSIGNED_STANDARD: "standard_role_unassigned",
    ROLE_NOT_USED: "not_used",
}


def calibration_requested(element_symbol: str, settings: Any) -> bool:
    """Calibration is requested on the Pb-Tl route: Pb, mass-bias correction and the switch on."""
    cfg = getattr(settings, "pb_standard_calibration", None)
    return bool(
        element_symbol == "Pb"
        and getattr(settings, "apply_mass_bias_correction", False)
        and cfg is not None
        and cfg.enabled
    )


def calibrated_delta_requested(element_symbol: str, settings: Any) -> bool:
    return calibration_requested(element_symbol, settings) and bool(settings.pb_standard_calibration.enable_delta)


def applied_mode_for(settings: Any) -> str:
    cfg = settings.pb_standard_calibration
    if cfg.mode == MODE_SESSION_MEAN_K:
        return APPLIED_SESSION_MEAN_K
    return APPLIED_LOCAL_BLOCK if settings.ssb_mode == "block_average" else APPLIED_LOCAL_ALTERNATING


def minimum_valid_cycles(settings: Any) -> int:
    cfg = settings.pb_standard_calibration
    mode = applied_mode_for(settings)
    if mode == APPLIED_SESSION_MEAN_K:
        return cfg.min_valid_cycles_session
    if mode == APPLIED_LOCAL_BLOCK:
        return cfg.min_valid_cycles_block
    return cfg.min_valid_cycles_alternating


def resolve_calibration_reference(
    element_symbol: str, material_name: Optional[str], ratio_name: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """The accepted reference for one ratio, or ``(None, reason_code)``.

    A directly certified row keeps its stated uncertainty and coverage factor; an
    unassigned uncertainty stays ``None``. A ratio derivable from certified rows
    (e.g. the inverse ``206Pb/204Pb``) is marked ``derived`` with the propagated
    standard uncertainty at k = 1.
    """
    if not material_name:
        return None, "reference_not_selected"
    records = get_crm_records(element_symbol, material_name)
    if not records:
        return None, "reference_unavailable"
    target = normalize_ratio_name(ratio_name)
    base = records[0]
    material_id = str(getattr(base, "material_id", "") or material_name)
    direct = next((r for r in records if normalize_ratio_name(r.ratio_name) == target), None)
    if direct is not None:
        value = float(direct.ratio)
        if not math.isfinite(value) or value <= 0:
            return None, "reference_unavailable"
        unassigned = bool(direct.is_uncertainty_unassigned)
        return {
            "material": material_name,
            "record_id": str(direct.record_id or ""),
            "material_id": str(direct.material_id or material_id),
            "ratio_name": target,
            "value": value,
            "uncertainty": None if unassigned or direct.uncertainty is None else float(direct.uncertainty),
            "k": None if unassigned or direct.k is None else float(direct.k),
            "uncertainty_semantics": str(direct.uncertainty_semantics or ""),
            "derived": False,
        }, ""
    derived = get_crm_ratios(element_symbol, material_name, derive=True).get(target)
    if derived is None:
        return None, "reference_unavailable"
    value, uncertainty, k = derived
    if not math.isfinite(float(value)) or float(value) <= 0:
        return None, "reference_unavailable"
    return {
        "material": material_name,
        "record_id": str(base.record_id or ""),
        "material_id": material_id,
        "ratio_name": target,
        "value": float(value),
        "uncertainty": float(uncertainty),
        "k": float(k),
        "uncertainty_semantics": "standard_uncertainty",
        "derived": True,
    }, ""


def resolve_correction_context(
    element: Any,
    settings: Any,
    *,
    ratio_definitions: Optional[Mapping[str, Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Resolved Tl, Hg and reference inputs, the same values processing uses."""
    from domain.pb_hg_correction import resolve_hg_reference
    from domain.pipeline import ProcessingPipeline

    pipeline = ProcessingPipeline(element)
    try:
        num, den, _m_num, _m_den = pipeline._resolve_normalization_pair(settings)
        tl_pair: Optional[List[str]] = [num, den]
    except ValueError:
        tl_pair = None
    tl_ratio = pipeline._resolve_normalization_ratio_name(settings)
    tl_value = pipeline._resolve_normalization_value(settings, pipeline._resolve_certified_values(settings))
    hg_reference: Dict[str, Any] = {}
    if settings.apply_hg_interference_correction:
        try:
            resolved = resolve_hg_reference()
            hg_reference = {"record_id": resolved["record_id"], "value": resolved["value"]}
        except ValueError as exc:
            hg_reference = {"unresolved": str(exc)}
    cfg = settings.pb_standard_calibration
    references = {}
    for ratio_name in (ratio_definitions or element.default_ratios):
        reference, reason = resolve_calibration_reference(element.symbol, cfg.reference_material, ratio_name)
        references[ratio_name] = reference if reference is not None else {"reason_code": reason}
    from domain.pb_correction_records import PB_HG_CORRECTION_SEMANTICS

    return {
        "tl_pair": tl_pair,
        "references": references,
        "correction": {
            "apply_mass_bias_correction": bool(settings.apply_mass_bias_correction),
            "tl_ratio": tl_ratio,
            "tl_value": None if tl_value is None else float(tl_value),
            "tl_value_override": settings.normalization_value_override,
            "hg_requested": bool(settings.apply_hg_interference_correction),
            "hg_reference": hg_reference,
            "blank_mode": settings.blank_mode,
            "filter_method": settings.filter_method,
            "filter_threshold": float(settings.filter_threshold),
            "semantics": {
                "pb_hg_correction": PB_HG_CORRECTION_SEMANTICS,
                "pb_tl_standard_calibration": PB_CALIBRATION_SEMANTICS,
            },
        },
    }


@dataclass(frozen=True)
class CalibrationSupport:
    """The cycles of one observation's Tl-only ratio that calibration may use."""

    mask: np.ndarray
    support_n_valid: int
    excluded_cycles: Tuple[Tuple[int, str], ...]
    window: Optional[Tuple[int, int]]

    @property
    def n_valid(self) -> int:
        return int(np.sum(self.mask))

    @property
    def excluded_fraction(self) -> Optional[float]:
        return (len(self.excluded_cycles) / self.support_n_valid) if self.support_n_valid else None

    @property
    def review_flags(self) -> Tuple[str, ...]:
        return (REVIEW_INVALID_CYCLE_EXCLUSION,) if self.excluded_cycles else ()


def _source_intensities(sample: Sample) -> Mapping[str, CycleData]:
    return sample.corrected_intensities if sample.corrected_intensities else sample.intensities


def tl_state(sample: Sample, tl_pair: Optional[Sequence[str]]) -> str:
    """``present`` (both Tl channels), ``absent`` (neither) or ``partial``."""
    if not tl_pair:
        return "absent"
    src = _source_intensities(sample)
    present = [iso in src for iso in tl_pair]
    if all(present):
        return "present"
    return "partial" if any(present) else "absent"


def calibration_support(
    sample: Sample,
    ratio_name: str,
    *,
    required_channels: Sequence[str],
    window: Optional[Tuple[int, int]],
) -> Optional[CalibrationSupport]:
    """Calibration-layer support of the Tl-only ratio, or ``None`` when there is no such layer."""
    tl_only = sample.iif_corrected_ratios.get(ratio_name)
    if tl_only is None:
        return None
    values = np.asarray(tl_only.values, dtype=np.float64)
    n = len(values)
    base = np.asarray(tl_only.mask, dtype=bool).copy()
    channel = np.ones(n, dtype=bool)
    processing = get_processing_ratio_data(sample, ratio_name)
    if processing is not None and len(processing.mask) == n:
        channel &= np.asarray(processing.mask, dtype=bool)
    src = _source_intensities(sample)
    for isotope in required_channels:
        cycle_data = src.get(isotope)
        if cycle_data is None or len(cycle_data.values) != n:
            continue
        channel &= np.asarray(cycle_data.mask, dtype=bool) & np.isfinite(cycle_data.values)
    if window is not None:
        in_window = np.zeros(n, dtype=bool)
        in_window[max(0, window[0] - 1):min(n, window[1])] = True
        base &= in_window
        channel &= in_window
    finite = np.isfinite(values)
    nonfinite = channel & ~finite
    with np.errstate(invalid="ignore"):
        nonpositive = base & finite & (values <= 0.0)
    supported = base | nonfinite
    final = base & ~nonpositive
    excluded = sorted(
        [(int(i) + 1, "nonfinite_tl_normalized_value") for i in np.flatnonzero(nonfinite)]
        + [(int(i) + 1, "nonpositive_tl_normalized_value") for i in np.flatnonzero(nonpositive)]
    )
    return CalibrationSupport(
        mask=final, support_n_valid=int(np.sum(supported)), excluded_cycles=tuple(excluded), window=window,
    )


def _required_channels(sample: Sample, ratio_name: str, pair: Tuple[str, str], tl_pair: Optional[Sequence[str]]) -> List[str]:
    required = list(pair) + list(tl_pair or ())
    record = hg_records(sample).get(ratio_name)
    if record is not None and record.status == APPLIED and "204Pb" in pair:
        required.append("202Hg")
    return required


def calibration_required_channels(
    sample: Sample, ratio_name: str, pair: Tuple[str, str], tl_pair: Optional[Sequence[str]],
) -> List[str]:
    """The channels whose support bounds an observation's calibration cycles (same rule as processing)."""
    return _required_channels(sample, ratio_name, pair, tl_pair)


def _mean_of(sample: Sample, ratio_name: str, support: CalibrationSupport) -> float:
    values = np.asarray(sample.iif_corrected_ratios[ratio_name].values, dtype=np.float64)[support.mask]
    return float(np.mean(values)) if len(values) else float("nan")


def eligibility_reason(
    sample: Sample,
    *,
    role: str,
    calibration_config: Any,
    reference: Optional[Mapping[str, Any]],
    support: Optional[CalibrationSupport],
    tl: str,
    minimum: int,
    mean: float,
) -> str:
    """The single calibration eligibility predicate; ``""`` means eligible."""
    if sample.metadata.get("excluded", False):
        return "excluded"
    if sample.is_blank:
        return "blank_not_eligible"
    if role == ROLE_UNASSIGNED_STANDARD or role in TARGET_ROLES and role != "independent_qc":
        return "role_unassigned"
    if role == "independent_qc":
        return "independent_qc"
    if role == ROLE_NOT_USED:
        return "not_used"
    material = (calibration_config.material_assignments or {}).get(sample.observation_id)
    if not material:
        return "material_unassigned"
    if reference is None or material != reference.get("material_id"):
        return "other_material"
    if tl == "absent":
        return "tl_absent"
    if support is None:
        return "tl_failed"
    if support.n_valid < minimum:
        return "insufficient_valid_cycles"
    if not math.isfinite(mean) or mean <= 0:
        return "nonfinite_mean"
    return ""


def _skip(sample: Sample, side: str, reason: str) -> SkippedObservation:
    return SkippedObservation(
        observation_id=sample.observation_id, label=sample.name, sample_type=sample.sample_type,
        run_number=float(sample.run_number), side=side, reason_code=reason,
    )


def _member(sample: Sample, ratio_name: str, support: CalibrationSupport, *, side: str, weight: float, C: float) -> CalibrationMember:
    values = np.asarray(sample.iif_corrected_ratios[ratio_name].values, dtype=np.float64)[support.mask]
    s_mean = float(np.mean(values))
    n_valid = len(values)
    se = float(np.std(values, ddof=1) / np.sqrt(n_valid)) if n_valid >= 2 else None
    return CalibrationMember(
        observation_id=sample.observation_id, label=sample.name, sample_type=sample.sample_type,
        run_number=float(sample.run_number), side=side, s_mean=s_mean, n_valid=n_valid, se=se,
        k_individual=C / s_mean, weight=weight, window=support.window,
        support_n_valid=support.support_n_valid, excluded_cycles=support.excluded_cycles,
        excluded_fraction=support.excluded_fraction, review_flags=support.review_flags,
        support_mask_sha256=mask_sha256(support.mask),
    )


def _walk(active, index, step, roles, eligibility, side):
    skipped: List[SkippedObservation] = []
    j = index + step
    while 0 <= j < len(active):
        candidate = active[j]
        j += step
        if candidate.is_blank or roles[candidate.observation_id] in TARGET_ROLES:
            continue
        reason = eligibility[candidate.observation_id]
        if reason:
            skipped.append(_skip(candidate, side, reason))
            continue
        return candidate, skipped
    return None, skipped


def _form_blocks(active, eligibility) -> List[List[int]]:
    """Maximal runs of consecutive eligible observations in active run order."""
    blocks: List[List[int]] = []
    current: List[int] = []
    for index, sample in enumerate(active):
        if eligibility.get(sample.observation_id) == "":
            current.append(index)
            continue
        if current:
            blocks.append(current)
        current = []
    if current:
        blocks.append(current)
    return blocks


def _block_skipped(active, index, stop, step, roles, eligibility, side):
    skipped = []
    j = index + step
    while 0 <= j < len(active) and j != stop:
        candidate = active[j]
        j += step
        if candidate.is_blank or roles[candidate.observation_id] in TARGET_ROLES:
            continue
        reason = eligibility[candidate.observation_id]
        if reason:
            skipped.append(_skip(candidate, side, reason))
    return skipped


def _fraction(t: float, t_prev: float, t_next: float) -> Optional[float]:
    return (t - t_prev) / (t_next - t_prev) if t_next != t_prev else None


def apply_pb_standard_calibration(
    active: List[Sample],
    *,
    element: Any,
    settings: Any,
    cycle_ranges: Optional[Mapping[str, Any]],
    correction_context: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    ratio_definitions: Mapping[str, Tuple[str, str]],
    warnings: List[str],
) -> Dict[str, Any]:
    """Write calibrated layers and records on ``active`` and return the session evidence."""
    cfg = settings.pb_standard_calibration
    applied_mode = applied_mode_for(settings)
    minimum = minimum_valid_cycles(settings)
    tl_pair = correction_context.get("tl_pair")
    digest = str(dependencies["calibration_input_digest"])
    chain = dict(dependencies["sample_chain_digests"])
    roles = {s.observation_id: observation_role(s, cfg) for s in active if not s.is_blank}
    explicit = any(role == "calibration_standard" for role in cfg.role_assignments.values())
    quality: Dict[str, Any] = {
        "schema_name": PB_CALIBRATION_SCHEMA_NAME,
        "schema_version": PB_CALIBRATION_SCHEMA_VERSION,
        "semantics_version": PB_CALIBRATION_SEMANTICS,
        "requested_mode": cfg.mode,
        "ssb_mode": settings.ssb_mode,
        "applied_mode": applied_mode,
        "reference_material": cfg.reference_material,
        "minimum_valid_cycles": minimum,
        "delta_requested": bool(cfg.enable_delta),
        "delta_precision_statistic": cfg.delta_precision_statistic,
        "calibration_input_digest": digest,
        "sample_chain_digests": chain,
        "dependencies": dependencies["payload"],
        "ratios": {},
    }

    for ratio_name, pair in ratio_definitions.items():
        if not all(str(isotope).endswith("Pb") for isotope in pair):
            continue
        reference, reference_reason = resolve_calibration_reference(element.symbol, cfg.reference_material, ratio_name)
        C = float(reference["value"]) if reference is not None else float("nan")
        supports: Dict[str, Optional[CalibrationSupport]] = {}
        eligibility: Dict[str, str] = {}
        for sample in active:
            if sample.is_blank:
                continue
            obs = sample.observation_id
            is_target = roles[obs] in TARGET_ROLES
            support = calibration_support(
                sample, ratio_name,
                required_channels=_required_channels(sample, ratio_name, pair, tl_pair),
                window=None if is_target else observation_window(sample, cycle_ranges),
            )
            supports[obs] = support
            if not is_target:
                mean = _mean_of(sample, ratio_name, support) if support is not None and support.n_valid else float("nan")
                eligibility[obs] = eligibility_reason(
                    sample, role=roles[obs], calibration_config=cfg, reference=reference, support=support,
                    tl=tl_state(sample, tl_pair), minimum=minimum, mean=mean,
                )
        blocks = _form_blocks(active, eligibility) if applied_mode == APPLIED_LOCAL_BLOCK else []
        pool = [i for i, s in enumerate(active) if eligibility.get(s.observation_id) == ""]
        ratio_quality: Dict[str, Any] = {
            "reference": reference if reference is not None else {"reason_code": reference_reason},
            "eligibility": {obs: (reason or "eligible") for obs, reason in eligibility.items()},
            "n_applied": 0,
            "n_unavailable": 0,
        }
        session_members: Tuple[CalibrationMember, ...] = ()
        session_k = session_mean_inverse = None
        if applied_mode == APPLIED_SESSION_MEAN_K and pool and reference is not None:
            weight = 1.0 / len(pool)
            session_members = tuple(
                _member(active[i], ratio_name, supports[active[i].observation_id], side="session", weight=weight, C=C)
                for i in pool
            )
            session_k = float(np.mean([C / m.s_mean for m in session_members]))
            session_mean_inverse = float(np.mean([1.0 / m.s_mean for m in session_members]))
            ratio_quality["pool"] = {
                "pool_id": canonical_sha256({
                    "ratio_name": ratio_name, "members": [[m.observation_id, m.s_mean] for m in session_members],
                }),
                "members": [m.to_dict() for m in session_members],
                "K_session": session_k,
                "mean_inverse_S": session_mean_inverse,
            }
        if applied_mode == APPLIED_LOCAL_BLOCK:
            ratio_quality["blocks"] = [[active[i].observation_id for i in block] for block in blocks]
        residual_diagnostic: Dict[str, Any] = {}
        if reference is not None and pool:
            # C05 P8: the held-out diagnostic of this functional, on the frozen supports of the
            # same eligible pool, in active run order; blocks are re-indexed to pool positions.
            position = {index: k for k, index in enumerate(pool)}
            pool_standards = []
            for index in pool:
                member = _member(
                    active[index], ratio_name, supports[active[index].observation_id],
                    side="session", weight=1.0, C=C,
                )
                pool_standards.append(PoolStandard(
                    observation_id=member.observation_id, s_mean=member.s_mean, se=member.se, n_valid=member.n_valid,
                ))
            residual_diagnostic = calibration_residual_diagnostic(
                pool_standards, applied_mode,
                [[position[i] for i in block] for block in blocks] if applied_mode == APPLIED_LOCAL_BLOCK else None,
            )
            ratio_quality["residual_diagnostic"] = residual_diagnostic

        for index, sample in enumerate(active):
            if sample.is_blank:
                continue
            obs = sample.observation_id
            role = roles[obs]
            records = sample.correction_records.setdefault(PB_CALIBRATION_RECORD_FAMILY, {})
            base = dict(
                observation_id=obs, ratio_name=ratio_name, role=role, requested_mode=cfg.mode,
                reference=reference or {"reason_code": reference_reason},
                calibration_input_digest=digest, sample_chain_digest=chain.get(obs, ""),
                n_cycles=sample.n_cycles,
            )
            if role not in TARGET_ROLES:
                reason = _NOT_REQUESTED_REASON_BY_ROLE[role]
                records[ratio_name] = PbCalibrationRecord(
                    status=NOT_REQUESTED, reason_code=reason,
                    reason=CALIBRATION_NOT_REQUESTED_REASONS[reason], **base,
                )
                continue

            support = supports[obs]
            support_fields = {}
            if support is not None:
                support_fields = dict(
                    support_n_valid=support.support_n_valid, n_valid=support.n_valid,
                    excluded_cycles=support.excluded_cycles, excluded_fraction=support.excluded_fraction,
                    review_flags=support.review_flags, support_mask_sha256=mask_sha256(support.mask),
                )
            reason = ""
            members: Tuple[CalibrationMember, ...] = ()
            skipped: List[SkippedObservation] = []
            bracket = mean_inverse = k = None
            layout: Dict[str, Any] = {}
            pool_id = ""
            if reference is None:
                reason = reference_reason
            elif tl_state(sample, tl_pair) == "absent":
                reason = "tl_absent"
            elif support is None:
                reason = "tl_failed"
            elif support.n_valid == 0:
                reason = "no_valid_cycles"
            elif not explicit:
                reason = "no_explicit_calibration_assignment"
            elif applied_mode == APPLIED_SESSION_MEAN_K:
                skipped = [
                    _skip(s, "session", eligibility[s.observation_id])
                    for s in active
                    if not s.is_blank and roles[s.observation_id] not in TARGET_ROLES
                    and eligibility[s.observation_id]
                ]
                if not session_members:
                    reason = "no_eligible_standards"
                else:
                    members, k, mean_inverse = session_members, session_k, session_mean_inverse
                    pool_id = ratio_quality["pool"]["pool_id"]
                    runs = [m.run_number for m in members]
                    layout = {
                        "run_number": float(sample.run_number), "pool_run_min": min(runs),
                        "pool_run_max": max(runs), "pool_run_mean": float(np.mean(runs)),
                    }
            elif applied_mode == APPLIED_LOCAL_ALTERNATING:
                prev_sample, prev_skipped = _walk(active, index, -1, roles, eligibility, "prev")
                next_sample, next_skipped = _walk(active, index, +1, roles, eligibility, "next")
                skipped = prev_skipped + next_skipped
                if not pool:
                    reason = "no_eligible_standards"
                elif prev_sample is None or next_sample is None:
                    reason = "missing_bracket_side"
                else:
                    members = (
                        _member(prev_sample, ratio_name, supports[prev_sample.observation_id], side="prev", weight=0.5, C=C),
                        _member(next_sample, ratio_name, supports[next_sample.observation_id], side="next", weight=0.5, C=C),
                    )
                    bracket = (members[0].s_mean + members[1].s_mean) / 2.0
                    k = C / bracket
                    layout = {"bracket_position_fraction": _fraction(
                        float(sample.run_number), members[0].run_number, members[1].run_number,
                    )}
            else:
                prev_block = next((b for b in reversed(blocks) if b[-1] < index), None)
                next_block = next((b for b in blocks if b[0] > index), None)
                skipped = (
                    _block_skipped(active, index, prev_block[-1] if prev_block else -1, -1, roles, eligibility, "prev")
                    + _block_skipped(active, index, next_block[0] if next_block else len(active), +1, roles, eligibility, "next")
                )
                if not pool:
                    reason = "no_eligible_standards"
                elif prev_block is None or next_block is None:
                    reason = "missing_bracket_side"
                else:
                    side_members = []
                    side_means = []
                    for side, block in (("prev", prev_block), ("next", next_block)):
                        weight = 1.0 / (2.0 * len(block))
                        built = [
                            _member(active[i], ratio_name, supports[active[i].observation_id], side=side, weight=weight, C=C)
                            for i in block
                        ]
                        side_members.extend(built)
                        side_means.append(float(np.mean([m.s_mean for m in built])))
                    members = tuple(side_members)
                    bracket = (side_means[0] + side_means[1]) / 2.0
                    k = C / bracket
                    layout = {
                        "bracket_position_fraction": _fraction(
                            float(sample.run_number),
                            float(np.mean([m.run_number for m in members if m.side == "prev"])),
                            float(np.mean([m.run_number for m in members if m.side == "next"])),
                        ),
                        "prev_block_centre_run": float(np.mean([m.run_number for m in members if m.side == "prev"])),
                        "next_block_centre_run": float(np.mean([m.run_number for m in members if m.side == "next"])),
                    }
            if not reason and (k is None or not math.isfinite(k) or k <= 0):
                reason = "nonfinite_calibration_factor"

            if reason:
                ratio_quality["n_unavailable"] += 1
                record = PbCalibrationRecord(
                    status=UNAVAILABLE, reason_code=reason, reason=CALIBRATION_UNAVAILABLE_REASONS[reason],
                    skipped=tuple(skipped), **support_fields, **base,
                )
                records[ratio_name] = record
                warnings.append(
                    f"Sample '{sample.name}': Pb-standard calibration unavailable for {ratio_name} "
                    f"({reason}); no final value is reported in place of the Tl-normalized value."
                )
                if cfg.enable_delta:
                    sample.correction_records.setdefault(PB_CALIBRATED_DELTA_RECORD_FAMILY, {})[ratio_name] = (
                        PbCalibratedDeltaRecord(
                            observation_id=obs, ratio_name=ratio_name, status=UNAVAILABLE,
                            reason_code=reason, reason=CALIBRATION_UNAVAILABLE_REASONS[reason],
                            reference_material=str(cfg.reference_material or ""),
                            reference_record_id=str((reference or {}).get("record_id", "")),
                            precision_statistic=cfg.delta_precision_statistic,
                        )
                    )
                continue

            tl_only = sample.iif_corrected_ratios[ratio_name]
            x_values = np.asarray(tl_only.values, dtype=np.float64)
            with np.errstate(all="ignore"):
                y_values = x_values * k
            sample.pb_standard_corrected_ratios[ratio_name] = CycleData(
                values=y_values, mask=support.mask.copy(),
            )
            record = PbCalibrationRecord(
                status=APPLIED, applied_mode=applied_mode, members=members, skipped=tuple(skipped),
                bracket_estimate=bracket, mean_inverse_standard=mean_inverse, k_applied=k,
                pool_id=pool_id, layout=layout, residual_diagnostic=residual_diagnostic,
                **support_fields, **base,
            )
            records[ratio_name] = record
            ratio_quality["n_applied"] += 1
            if support.excluded_cycles:
                warnings.append(
                    f"Sample '{sample.name}': {len(support.excluded_cycles)} of {support.support_n_valid} "
                    f"cycle(s) excluded from calibrated {ratio_name} as invalid "
                    f"({', '.join(f'{c}:{r}' for c, r in support.excluded_cycles)}); review this result."
                )

            if cfg.enable_delta:
                with np.errstate(all="ignore"):
                    if applied_mode == APPLIED_SESSION_MEAN_K:
                        delta_values = 1000.0 * (x_values * mean_inverse - 1.0)
                        scale = mean_inverse
                    else:
                        delta_values = 1000.0 * (x_values / bracket - 1.0)
                        scale = 1.0 / bracket
                delta_cd = CycleData(values=delta_values, mask=support.mask.copy())
                sample.pb_calibrated_delta_cycles[ratio_name] = delta_cd
                valid = delta_cd.valid_values
                precision = None
                statistic = cfg.delta_precision_statistic
                if statistic != "none" and len(valid) >= 2:
                    sd = float(np.std(valid, ddof=1))
                    precision = sd if statistic == "sd" else sd / math.sqrt(len(valid))
                sample.correction_records.setdefault(PB_CALIBRATED_DELTA_RECORD_FAMILY, {})[ratio_name] = (
                    PbCalibratedDeltaRecord(
                        observation_id=obs, ratio_name=ratio_name, status=APPLIED,
                        applied_mode=applied_mode, estimator=DELTA_ESTIMATORS[applied_mode],
                        scale_factor=scale, calibration_record_sha256=record.sha256,
                        reference_material=str(cfg.reference_material or ""),
                        reference_record_id=str(reference.get("record_id", "")),
                        delta_mean=float(np.mean(valid)), n_valid=len(valid),
                        precision_statistic=statistic, precision_value=precision,
                    )
                )
        quality["ratios"][ratio_name] = ratio_quality
    return quality


def calibrated_runtime_delta_cycles(
    sample: Sample, ratio_name: str, runtime_mask: np.ndarray,
) -> Optional[CycleData]:
    """The stored calibrated delta cycles restricted to a runtime selection; never a new bracket search."""
    stored = sample.pb_calibrated_delta_cycles.get(ratio_name)
    if stored is None:
        return None
    mask = np.asarray(stored.mask, dtype=bool) & np.asarray(runtime_mask, dtype=bool)
    return CycleData(values=stored.values.copy(), mask=mask)

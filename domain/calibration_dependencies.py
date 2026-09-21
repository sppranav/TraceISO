"""Dependency records and digests that decide whether a calibrated Pb result is current.

A Pb-standard calibration depends on more than the stored standard arrays: on the
raw content, masks and cycle windows of every candidate standard, on the blanks
subtracted from each of them (which the stored corrected arrays no longer show),
on the role and material maps and on every correction setting. This module
digests those **current inputs** — never stored corrected arrays — so a later
check can tell whether a processed calibration still describes the session.

* ``calibration_input_digest`` covers the calibration settings, the resolved
  references, the complete role and material maps, the run order and one
  dependency record per candidate (any non-blank observation with a role or
  material assignment, and every standard-typed observation), including the
  candidate's own cycle window. A mismatch makes the whole calibrated set stale.
* ``sample_chain_digests`` cover each corrected observation's own raw inputs and
  its blanks, but not its own window or manual runtime exclusion overlay. A
  mismatch in intrinsic support still makes that observation stale.

Blank assignment is recomputed from the current active sequence with the rule
``apply_blank_correction`` uses (nearest blank before; also after in
``before_and_after``), so a blank edit that re-points an assignment is seen even
though no stored array moved. Channel weights are a deterministic function of the
digested blank content and that rule, so the rule identity stands in for them.
Re-importing raw data mints new observation IDs, which reads as stale.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from domain.filters.outlier import resolve_cycle_range, sample_cycle_key
from domain.models import Sample
from domain.pb_calibration_records import (
    PB_CALIBRATION_QUALITY_KEY,
    PB_CALIBRATION_SEMANTICS,
    ROLE_CALIBRATION_STANDARD,
    ROLE_INDEPENDENT_QC,
    ROLE_NOT_USED,
    ROLE_SAMPLE,
    ROLE_UNASSIGNED_STANDARD,
    TARGET_ROLES,
    canonical_sha256,
)

DEPENDENCY_SCHEMA_NAME = "traceiso.pb_calibration_dependencies"
DEPENDENCY_SCHEMA_VERSION = "1.1"
BLANK_ASSIGNMENT_RULE = "nearest_blank_before_optional_after.v1"

FRESHNESS_CURRENT = "current"
FRESHNESS_STALE = "stale"
FRESHNESS_NOT_APPLICABLE = "not_applicable"

AVAILABILITY_CURRENT = "current"
AVAILABILITY_STALE = "stale"
AVAILABILITY_HISTORICAL_UNVERIFIED = "historical_unverified"
AVAILABILITY_NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class EffectiveCalibrationAvailability:
    """Current-session availability without mutating historical producer records.

    ``freshness=None`` means that only historical evidence was supplied.  Such
    evidence remains inspectable/exportable as historical, but is not certified
    as a current-session final result.
    """

    status: str
    producer_status: str
    current: bool
    reason: str = ""


def effective_calibration_availability(
    sample: Sample, ratio_name: str, freshness: Optional[Mapping[str, Any]],
) -> EffectiveCalibrationAvailability:
    """Combine immutable producer status with authoritative session freshness."""
    from domain.layer_status import APPLIED
    from domain.pb_calibration_records import calibration_final_status, governing_calibration_record

    producer = calibration_final_status(sample, ratio_name)
    if governing_calibration_record(sample, ratio_name) is None:
        return EffectiveCalibrationAvailability(
            AVAILABILITY_NOT_APPLICABLE, producer, True,
        )
    if producer != APPLIED:
        return EffectiveCalibrationAvailability(producer, producer, False)
    if freshness is None:
        return EffectiveCalibrationAvailability(
            AVAILABILITY_HISTORICAL_UNVERIFIED, producer, False,
            "No authoritative current-session calibration context was supplied.",
        )
    if freshness.get("status") == FRESHNESS_STALE:
        set_current = bool(freshness.get("calibration_input_current", False))
        stale_ids = set(freshness.get("stale_observation_ids") or ())
        if not set_current or sample.observation_id in stale_ids:
            return EffectiveCalibrationAvailability(
                AVAILABILITY_STALE, producer, False, str(freshness.get("reason") or "Calibration is stale."),
            )
    return EffectiveCalibrationAvailability(AVAILABILITY_CURRENT, producer, True)

_ROLE_BY_ASSIGNMENT = {
    "calibration_standard": ROLE_CALIBRATION_STANDARD,
    "independent_qc": ROLE_INDEPENDENT_QC,
    "not_used": ROLE_NOT_USED,
}


def observation_role(sample: Sample, calibration_config: Any) -> str:
    """Calibration role of a non-blank observation; the type is used only when no role is set.

    An explicit role is authoritative whatever the sample type. Without one, a
    standard-typed observation is an unassigned standard (never corrected, never
    used) and anything else is a sample.
    """
    assigned = (getattr(calibration_config, "role_assignments", None) or {}).get(sample.observation_id)
    if assigned in _ROLE_BY_ASSIGNMENT:
        return _ROLE_BY_ASSIGNMENT[assigned]
    return ROLE_UNASSIGNED_STANDARD if sample.is_standard else ROLE_SAMPLE


def observation_window(sample: Sample, cycle_ranges: Optional[Mapping[str, Any]]) -> Optional[Tuple[int, int]]:
    """Return a shortened effective window; absent and full-length are equivalent."""
    if not cycle_ranges:
        return None
    window = cycle_ranges.get(sample.observation_id)
    if window is None:
        window = resolve_cycle_range(
            dict(cycle_ranges), sample_name=sample.name, sample_key=sample_cycle_key(sample),
        )
    if window is None:
        return None
    resolved = int(window[0]), int(window[1])
    if resolved[0] <= 1 and resolved[1] >= sample.n_cycles:
        return None
    return resolved


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _intrinsic_channel_mask(sample: Sample, isotope: str, current: np.ndarray) -> np.ndarray:
    """Remove only the controller-owned manual overlay from a raw channel mask."""
    original = (sample.metadata.get("_original_masks") or {}).get("intensities", {}).get(isotope)
    excluded = sample.metadata.get("manual_exclusions") or ()
    if original is None or len(original) != len(current) or not excluded:
        return current
    intrinsic = np.asarray(current, dtype=bool).copy()
    baseline = np.asarray(original, dtype=bool)
    for cycle in excluded:
        index = int(cycle) - 1
        if 0 <= index < len(intrinsic):
            intrinsic[index] = baseline[index]
    return intrinsic


def _channel_payload(
    sample: Sample, channels: Iterable[str], *, ignore_manual_overlay: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for isotope in sorted(set(channels)):
        cycle_data = sample.intensities.get(isotope)
        if cycle_data is None:
            out[isotope] = None
            continue
        mask = np.asarray(cycle_data.mask, dtype=bool)
        if ignore_manual_overlay:
            mask = _intrinsic_channel_mask(sample, isotope, mask)
        out[isotope] = {
            "n": int(len(cycle_data.values)),
            "values_sha256": _digest(np.ascontiguousarray(cycle_data.values, dtype=np.float64).tobytes()),
            "mask_sha256": _digest(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()),
        }
    return out


def recompute_blank_assignment(active: Sequence[Sample], index: int, blank_mode: str) -> Dict[str, Sample]:
    """The blanks the blank correction would subtract from ``active[index]`` now."""
    if blank_mode == "none" or active[index].is_blank:
        return {}
    assigned: Dict[str, Sample] = {}
    for j in range(index - 1, -1, -1):
        if active[j].is_blank:
            assigned["before"] = active[j]
            break
    if blank_mode == "before_and_after":
        for j in range(index + 1, len(active)):
            if active[j].is_blank:
                assigned["after"] = active[j]
                break
    return assigned


def _dependency_record(
    sample: Sample,
    *,
    active: Sequence[Sample],
    channels: Iterable[str],
    blank_mode: str,
    cycle_ranges: Optional[Mapping[str, Any]],
    calibration_config: Any,
    include_own_window: bool,
    include_manual_overlay: bool = True,
) -> Dict[str, Any]:
    index = next((i for i, candidate in enumerate(active) if candidate is sample), None)
    blanks: Dict[str, Any] = {}
    if index is not None:
        for side, blank in recompute_blank_assignment(active, index, blank_mode).items():
            blanks[side] = {
                "observation_id": blank.observation_id,
                "name": blank.name,
                "run_number": float(blank.run_number),
                "channels": _channel_payload(blank, channels),
                "manual_exclusions": sorted(int(c) for c in (blank.metadata.get("manual_exclusions") or ())),
                "window": list(observation_window(blank, cycle_ranges) or ()) or None,
            }
    record: Dict[str, Any] = {
        "observation_id": sample.observation_id,
        "name": sample.name,
        "sample_type": sample.sample_type,
        "run_number": float(sample.run_number),
        "excluded": bool(sample.metadata.get("excluded", False)),
        "role": observation_role(sample, calibration_config),
        "material": (calibration_config.material_assignments or {}).get(sample.observation_id),
        "channels": _channel_payload(
            sample, channels, ignore_manual_overlay=not include_manual_overlay,
        ),
        "blanks": {"blank_mode": blank_mode, "rule": BLANK_ASSIGNMENT_RULE, "assigned": blanks},
    }
    if include_own_window:
        record["window"] = list(observation_window(sample, cycle_ranges) or ()) or None
    if include_manual_overlay:
        record["manual_exclusions"] = sorted(
            int(c) for c in (sample.metadata.get("manual_exclusions") or ())
        )
    return record


def build_calibration_dependencies(
    samples: Sequence[Sample],
    *,
    element_symbol: str,
    ratio_definitions: Mapping[str, Tuple[str, str]],
    settings: Any,
    cycle_ranges: Optional[Mapping[str, Any]],
    correction_context: Mapping[str, Any],
) -> Dict[str, Any]:
    """Digest the current inputs a calibrated Pb result set depends on.

    ``samples`` is the session in processing order, excluded observations
    included. ``correction_context`` carries the resolved references and
    correction settings (``domain.pb_standard_calibration.resolve_correction_context``).
    """
    cfg = settings.pb_standard_calibration
    active = [s for s in samples if not s.metadata.get("excluded", False)]
    channels = {iso for pair in ratio_definitions.values() for iso in pair}
    channels.update(correction_context.get("tl_pair") or ())
    if settings.apply_hg_interference_correction:
        channels.add("202Hg")

    settings_payload = {
        "semantics_version": PB_CALIBRATION_SEMANTICS,
        "element": element_symbol,
        "ratio_definitions": {str(r): list(pair) for r, pair in sorted(ratio_definitions.items())},
        "mode": cfg.mode,
        "ssb_mode": settings.ssb_mode,
        "reference_material": cfg.reference_material,
        "references": correction_context.get("references", {}),
        "minimum_valid_cycles": {
            "alternating": cfg.min_valid_cycles_alternating,
            "block": cfg.min_valid_cycles_block,
            "session": cfg.min_valid_cycles_session,
        },
        "role_assignments": dict(sorted(cfg.role_assignments.items())),
        "material_assignments": dict(sorted(cfg.material_assignments.items())),
        "correction": correction_context.get("correction", {}),
    }
    run_order = [
        [s.observation_id, s.sample_type, bool(s.metadata.get("excluded", False)), float(s.run_number)]
        for s in samples
    ]
    candidates: List[Dict[str, Any]] = []
    sample_chain_digests: Dict[str, str] = {}
    for sample in samples:
        if sample.is_blank:
            continue
        role = observation_role(sample, cfg)
        assigned = (
            sample.observation_id in cfg.role_assignments
            or sample.observation_id in cfg.material_assignments
            or sample.is_standard
        )
        is_target = role in TARGET_ROLES
        common = dict(
            active=active, channels=channels, blank_mode=settings.blank_mode,
            cycle_ranges=cycle_ranges, calibration_config=cfg,
        )
        if assigned:
            # A standard's window selects the cycles its K uses; a corrected
            # observation's own window stays a runtime view.
            candidates.append(_dependency_record(
                sample, include_own_window=not is_target,
                include_manual_overlay=not is_target, **common,
            ))
        if is_target:
            sample_chain_digests[sample.observation_id] = canonical_sha256({
                "correction": settings_payload["correction"],
                "element": element_symbol,
                "record": _dependency_record(
                    sample, include_own_window=False, include_manual_overlay=False, **common,
                ),
            })
    payload = {
        "schema_name": DEPENDENCY_SCHEMA_NAME,
        "schema_version": DEPENDENCY_SCHEMA_VERSION,
        "settings": settings_payload,
        "run_order": run_order,
        "candidates": candidates,
    }
    return {
        "schema_name": DEPENDENCY_SCHEMA_NAME,
        "schema_version": DEPENDENCY_SCHEMA_VERSION,
        "calibration_input_digest": canonical_sha256(payload),
        "sample_chain_digests": sample_chain_digests,
        "payload": payload,
    }


def calibration_freshness(
    quality_metrics: Optional[Mapping[str, Any]],
    samples: Sequence[Sample],
    *,
    element: Any,
    settings: Any,
    cycle_ranges: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Whether a processed calibration still describes the current session.

    Pure: computed from current inputs, never from stored corrected arrays, and
    it mutates nothing, so every consumer can ask it during a render.
    """
    stored = (quality_metrics or {}).get(PB_CALIBRATION_QUALITY_KEY)
    if not isinstance(stored, Mapping):
        return {"status": FRESHNESS_NOT_APPLICABLE, "calibration_input_current": True,
                "stale_observation_ids": [], "reason": ""}
    from domain.pb_standard_calibration import calibration_requested, resolve_correction_context

    stored_chain = dict(stored.get("sample_chain_digests") or {})
    if not calibration_requested(getattr(element, "symbol", ""), settings):
        return {"status": FRESHNESS_STALE, "calibration_input_current": False,
                "stale_observation_ids": sorted(stored_chain),
                "reason": "Pb-standard calibration is no longer requested by the current settings."}
    try:
        from domain.pipeline import ProcessingPipeline

        ratio_definitions = ProcessingPipeline(
            element
        )._resolve_ratio_definitions_for_samples(list(samples))
        context = resolve_correction_context(
            element, settings, ratio_definitions=ratio_definitions,
        )
        current = build_calibration_dependencies(
            samples, element_symbol=element.symbol, ratio_definitions=ratio_definitions,
            settings=settings, cycle_ranges=cycle_ranges, correction_context=context,
        )
    except Exception as exc:  # an input that cannot be resolved cannot be current
        return {"status": FRESHNESS_STALE, "calibration_input_current": False,
                "stale_observation_ids": sorted(stored_chain),
                "reason": f"The calibration inputs could not be resolved: {exc}"}
    set_current = current["calibration_input_digest"] == stored.get("calibration_input_digest")
    if not set_current:
        return {"status": FRESHNESS_STALE, "calibration_input_current": False,
                "stale_observation_ids": sorted(set(stored_chain) | set(current["sample_chain_digests"])),
                "reason": (
                    "A calibration input changed since processing (standard window, exclusions or content, "
                    "an assigned blank, a role or material assignment, or a correction setting)."
                )}
    stale = sorted(
        obs for obs, digest in stored_chain.items()
        if current["sample_chain_digests"].get(obs) != digest
    )
    stale += sorted(set(current["sample_chain_digests"]) - set(stored_chain))
    return {
        "status": FRESHNESS_STALE if stale else FRESHNESS_CURRENT,
        "calibration_input_current": True,
        "stale_observation_ids": stale,
        "reason": (
            "An input of a calibrated observation's own chain (its content, exclusions or blanks) changed."
            if stale else ""
        ),
    }

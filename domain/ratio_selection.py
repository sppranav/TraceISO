"""Helpers for selecting the active ratio layer for a sample.

A sample carries several ratio layers (raw -> blank_corrected -> corrected/iif ->
drift -> ssb). These helpers resolve which layer to use for a given purpose by
walking a priority chain and returning the highest-priority layer that is
present. The two canonical entry points are :func:`get_processing_ratio_data`
(the basis for the processing outlier mask) and :func:`get_best_ratio_data` (the
display/reporting basis).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
import logging

import numpy as np

from domain.layer_status import APPLIED
from domain.layers import cycle_data_equal
from domain.models import CycleData, Sample
from domain.sr_standard_calibration import sr_calibration_unavailable
from domain.pb_calibration_records import calibration_blocks_final_value, governing_calibration_record
from domain.pb_correction_records import governing_hg_record, hg_blocks_final_value

logger = logging.getLogger(__name__)

#: Layer key of the governing Hg interference-corrected ratio. Its presence in a
#: priority chain marks that chain as a final or derived selector, which must
#: enforce the Hg status before falling back to an earlier layer.
INTERFERENCE_LAYER_KEY = "interference"
INTERFERENCE_LAYER_LABEL = "Interference-corrected (Hg)"

#: Sr calibration is a separate layer after internal normalization and before drift.
SR_STANDARD_LAYER_KEY = "sr_standard"
SR_STANDARD_LAYER_LABEL = "Internal normalization + Sr-standard-corrected"

#: Pb calibration enforces its recorded availability before final-layer fallback.
PB_STANDARD_LAYER_KEY = "pb_standard"
PB_STANDARD_LAYER_LABEL = "Tl + Pb-standard-corrected"


def governed_pb_standard_ratio_data(
    sample: Sample, ratio_name: str, *, freshness: Optional[Mapping[str, Any]] = None,
) -> Optional[CycleData]:
    """The calibrated final ratio when an applied calibration record governs it."""
    record = governing_calibration_record(sample, ratio_name)
    if record is None or record.status != APPLIED:
        return None
    if freshness is not None:
        from domain.calibration_dependencies import effective_calibration_availability
        if not effective_calibration_availability(sample, ratio_name, freshness).current:
            return None
    return (getattr(sample, "pb_standard_corrected_ratios", None) or {}).get(ratio_name)


def pb_standard_layer_label(sample: Sample, ratio_name: str) -> str:
    """Return the combined Tl-normalization and standard-correction label."""
    record = governing_calibration_record(sample, ratio_name)
    return record.final_layer_label if record is not None else PB_STANDARD_LAYER_LABEL


def _final_value_blocked(
    sample: Sample, ratio_name: str, priorities, *, freshness: Optional[Mapping[str, Any]] = None,
) -> bool:
    """A governed correction that is requested but unavailable leaves a final chain empty."""
    if SR_STANDARD_LAYER_KEY in priorities and sr_calibration_unavailable(sample, ratio_name):
        return True
    if INTERFERENCE_LAYER_KEY in priorities and hg_blocks_final_value(sample, ratio_name):
        return True
    if PB_STANDARD_LAYER_KEY not in priorities:
        return False
    if freshness is not None:
        from domain.calibration_dependencies import effective_calibration_availability
        availability = effective_calibration_availability(sample, ratio_name, freshness)
        if availability.status != "not_applicable":
            return not availability.current
    return calibration_blocks_final_value(sample, ratio_name)


def governed_interference_ratio_data(sample: Sample, ratio_name: str) -> Optional[CycleData]:
    """The Hg interference-corrected ratio when an applied SSB-route record governs it.

    Pb-Tl intermediates are diagnostic and are never returned here, so they
    cannot become a final, display or export layer.
    """
    record = governing_hg_record(sample, ratio_name)
    if record is None or record.status != APPLIED:
        return None
    return (getattr(sample, "interference_corrected_ratios", None) or {}).get(ratio_name)


def inspect_interference_ratio_data(sample: Sample, ratio_name: str) -> Optional[CycleData]:
    """Labelled inspection accessor for the saved Hg interference-only ratio (both routes)."""
    return (getattr(sample, "interference_corrected_ratios", None) or {}).get(ratio_name)


def intersect_required_channel_masks(sample, ratio_name, channels, required, mask):
    """Intersect aligned required input support with the processing ratio mask.

    A finite value whose own channel excludes it is not an accepted observation.
    Alignment must be established before calling; never shorten a ratio here.
    """
    result = np.asarray(mask, dtype=bool).copy()
    processing = get_processing_ratio_data(sample, ratio_name)
    if processing is not None:
        if len(processing.mask) != len(result):
            raise ValueError("Processing ratio cycle length does not match normalization output")
        result &= processing.mask
    for isotope in required:
        channel = channels[isotope]
        if len(channel.values) != len(result):
            raise ValueError(f"Normalization channel {isotope} has incompatible cycle length")
        result &= channel.mask & np.isfinite(channel.values)
    return result


_LAYERS = {
    SR_STANDARD_LAYER_KEY: lambda s, r: s.sr_standard_corrected_ratios.get(r),
    PB_STANDARD_LAYER_KEY: lambda s, r: governed_pb_standard_ratio_data(s, r),
    "drift": lambda s, r: s.drift_corrected_ratios.get(r) if s.drift_corrected_ratios else None,
    "iif": lambda s, r: s.iif_corrected_ratios.get(r) if s.iif_corrected_ratios else None,
    INTERFERENCE_LAYER_KEY: lambda s, r: governed_interference_ratio_data(s, r),
    "ssb": lambda s, r: get_ssb_cycle_data(s, r),
    "corrected": lambda s, r: s.corrected_ratios.get(r) if s.corrected_ratios else None,
    "blank_corrected": lambda s, r: s.blank_corrected_ratios.get(r) if s.blank_corrected_ratios else None,
    "ratios": lambda s, r: s.ratios.get(r) if s.ratios else None,
}

_LAYER_LABELS = {
    SR_STANDARD_LAYER_KEY: SR_STANDARD_LAYER_LABEL,
    PB_STANDARD_LAYER_KEY: PB_STANDARD_LAYER_LABEL,
    "ssb": "SSB-corrected",
    "drift": "Drift-corrected",
    "iif": "IIF-corrected",
    INTERFERENCE_LAYER_KEY: INTERFERENCE_LAYER_LABEL,
    "corrected": "Corrected",
    "blank_corrected": "Blank-corrected",
    "ratios": "Raw",
}


@dataclass(frozen=True)
class SelectedRatioLayer:
    """A ratio layer selected for export, including its stable label."""

    data: CycleData
    key: str
    label: str


def is_usable_bracketing_standard(cycle_data: Optional[CycleData]) -> bool:
    """Return whether *cycle_data* can bracket a sample.

    A standard is usable as a bracketing point only when its selected ratio
    layer exists AND has at least one valid cycle. A standard whose cycles are
    all masked/excluded (e.g. a failed run) must be skipped by delta and SSB
    bracketing searches rather than treated as a hard stop.
    """
    return cycle_data is not None and cycle_data.n_valid > 0


@dataclass(frozen=True)
class StandardProbe:
    """Usability verdict for one candidate standard in a bracketing search.

    ``reason`` is populated only when ``usable`` is ``False`` and should be
    one of a small, explicit vocabulary (``"missing ratio layer"``,
    ``"zero valid cycles"``, ``"runtime excluded"``, or another explicit
    reason) so provenance stays auditable rather than free-text.
    """

    usable: bool
    layer_key: Optional[str] = None
    cycle_data: Optional[CycleData] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class SkippedStandard:
    """A candidate standard rejected during a bracketing search."""

    name: str
    run_number: float
    reason: str

    def to_payload(self) -> Dict[str, object]:
        return {"name": self.name, "run_number": self.run_number, "reason": self.reason}


@dataclass(frozen=True)
class BracketSearchResult:
    """Outcome of walking outward from a sample for the nearest usable standard.

    Carries both the selected standard (identity, layer, ``CycleData``) and
    every intervening standard that was tested and rejected, so a reviewer can
    audit exactly which candidates were skipped and why without reconstructing
    the run sequence — a numeric run-number gap between the sample and the
    selected standard is not by itself evidence of anything unusual (several
    ordinary samples between standards is normal); only candidates that were
    actually tested and failed ``probe`` appear in ``skipped``.
    """

    sample: Optional[Sample]
    name: Optional[str]
    run_number: Optional[float]
    layer_key: Optional[str]
    cycle_data: Optional[CycleData]
    skipped: Tuple[SkippedStandard, ...] = ()


def find_nearest_usable_standard(
    samples: List[Sample],
    idx: int,
    *,
    direction: str,
    probe: Callable[[Sample], StandardProbe],
) -> BracketSearchResult:
    """Walk outward from *idx* and return the nearest usable bracketing standard.

    Shared by the stored delta, SSB, and runtime-delta bracketing searches so
    all three provably agree on one definition of "usable" and one provenance
    format. ``direction`` is ``"before"`` or ``"after"``; non-standard samples
    are skipped silently (they were never candidates), while standards that
    fail ``probe`` are recorded in the returned ``skipped`` tuple.
    """
    index_range = (
        range(idx - 1, -1, -1) if direction == "before" else range(idx + 1, len(samples))
    )
    skipped: List[SkippedStandard] = []
    for j in index_range:
        candidate = samples[j]
        if not candidate.is_standard:
            continue
        result = probe(candidate)
        if not result.usable:
            skipped.append(
                SkippedStandard(
                    name=candidate.name,
                    run_number=float(candidate.run_number),
                    reason=result.reason or "unusable",
                )
            )
            continue
        return BracketSearchResult(
            sample=candidate,
            name=candidate.name,
            run_number=float(candidate.run_number),
            layer_key=result.layer_key,
            cycle_data=result.cycle_data,
            skipped=tuple(skipped),
        )
    return BracketSearchResult(
        sample=None, name=None, run_number=None, layer_key=None, cycle_data=None,
        skipped=tuple(skipped),
    )


def format_skipped_standards(skipped: Tuple[SkippedStandard, ...]) -> str:
    """Render skipped-standard provenance as one concise, human-readable string."""
    return "; ".join(f"{s.name} (run {s.run_number:g}, {s.reason})" for s in skipped)


def select_summary_export_layer(
    sample: Sample,
    ratio_name: str,
) -> Optional[SelectedRatioLayer]:
    """Select the highest-priority distinct layer for summary exports."""
    if (sr_calibration_unavailable(sample, ratio_name) or hg_blocks_final_value(sample, ratio_name)
            or calibration_blocks_final_value(sample, ratio_name)):
        return None
    raw = sample.ratios.get(ratio_name) if sample.ratios else None

    calibrated = governed_pb_standard_ratio_data(sample, ratio_name)
    if calibrated is not None:
        return SelectedRatioLayer(
            calibrated, PB_STANDARD_LAYER_KEY, pb_standard_layer_label(sample, ratio_name),
        )

    ssb = get_ssb_cycle_data(sample, ratio_name)
    if ssb is not None and not cycle_data_equal(ssb, raw):
        return SelectedRatioLayer(ssb, "ssb", "SSB-corrected")

    drift = (
        sample.drift_corrected_ratios.get(ratio_name)
        if sample.drift_corrected_ratios
        else None
    )
    if drift is not None and not cycle_data_equal(drift, raw):
        return SelectedRatioLayer(drift, "drift", "Drift-corrected")

    sr_calibrated = sample.sr_standard_corrected_ratios.get(ratio_name)
    if sr_calibrated is not None:
        return SelectedRatioLayer(sr_calibrated, SR_STANDARD_LAYER_KEY, SR_STANDARD_LAYER_LABEL)

    iif = (
        sample.iif_corrected_ratios.get(ratio_name)
        if sample.iif_corrected_ratios
        else None
    )
    if iif is not None and not cycle_data_equal(iif, raw):
        return SelectedRatioLayer(iif, "iif", "IIF-corrected")

    interference = governed_interference_ratio_data(sample, ratio_name)
    if interference is not None and not cycle_data_equal(interference, raw):
        return SelectedRatioLayer(
            interference, INTERFERENCE_LAYER_KEY, _LAYER_LABELS[INTERFERENCE_LAYER_KEY],
        )

    blank = (
        sample.blank_corrected_ratios.get(ratio_name)
        if sample.blank_corrected_ratios
        else None
    )
    corrected = (
        sample.corrected_ratios.get(ratio_name)
        if sample.corrected_ratios
        else None
    )
    if corrected is not None:
        # Exact inequality, not a closeness tolerance. A correction the
        # pipeline actually computed is a distinct scientific layer however
        # small it is: a sub-ppm Rb interference correction on 87Sr/86Sr sits
        # far inside any sensible rtol yet moves the reported mean, and this
        # application's context makes sub-ppm differences material. Numerical
        # closeness answers a different question (are these the same array to
        # within noise?) and must not decide which layer is reported.
        if (
            blank is not None
            and not cycle_data_equal(corrected, blank)
            and not cycle_data_equal(corrected, raw)
        ):
            return SelectedRatioLayer(
                corrected,
                "corrected",
                "Interference-corrected",
            )
        if blank is not None and not cycle_data_equal(blank, raw):
            return SelectedRatioLayer(
                blank,
                "blank_corrected",
                "Blank-corrected",
            )
        if not cycle_data_equal(corrected, raw):
            return SelectedRatioLayer(corrected, "corrected", "Corrected")

    if blank is not None and not cycle_data_equal(blank, raw):
        return SelectedRatioLayer(blank, "blank_corrected", "Blank-corrected")
    if raw is not None:
        return SelectedRatioLayer(raw, "ratios", "Raw")
    return None


def select_cycle_export_layer(
    sample: Sample,
    ratio_name: str,
) -> Optional[SelectedRatioLayer]:
    """Select the downstream corrected layer for per-cycle Excel exports."""
    if (sr_calibration_unavailable(sample, ratio_name) or hg_blocks_final_value(sample, ratio_name)
            or calibration_blocks_final_value(sample, ratio_name)):
        return None
    raw = sample.ratios.get(ratio_name) if sample.ratios else None
    calibrated = governed_pb_standard_ratio_data(sample, ratio_name)
    if calibrated is not None:
        return SelectedRatioLayer(
            calibrated, PB_STANDARD_LAYER_KEY, pb_standard_layer_label(sample, ratio_name),
        )
    ssb = get_ssb_cycle_data(sample, ratio_name)
    if ssb is not None and not cycle_data_equal(ssb, raw):
        return SelectedRatioLayer(ssb, "ssb", "SSB-corrected")

    def _from(layer):
        return layer.get(ratio_name) if layer else None

    candidates = (
        ("drift", "Drift-corrected", _from(sample.drift_corrected_ratios)),
        (SR_STANDARD_LAYER_KEY, SR_STANDARD_LAYER_LABEL, _from(sample.sr_standard_corrected_ratios)),
        ("iif", "IIF-corrected", _from(sample.iif_corrected_ratios)),
        (
            INTERFERENCE_LAYER_KEY,
            _LAYER_LABELS[INTERFERENCE_LAYER_KEY],
            governed_interference_ratio_data(sample, ratio_name),
        ),
        ("corrected", "Corrected", _from(sample.corrected_ratios)),
    )
    for key, label, data in candidates:
        if data is not None:
            if key != SR_STANDARD_LAYER_KEY and cycle_data_equal(data, raw):
                return None
            return SelectedRatioLayer(data, key, label)
    return None


def _resolve_layer_by_priority(
    sample: Sample,
    ratio_name: str,
    priorities: List[str],
) -> Optional[CycleData]:
    """Iterate over layer keys in priority order and return the first non-None CycleData."""
    if _final_value_blocked(sample, ratio_name, priorities):
        return None
    for layer in priorities:
        resolver = _LAYERS.get(layer)
        if resolver is not None:
            cd = resolver(sample, ratio_name)
            if cd is not None:
                return cd
    return None


def _select_layer_by_priority(
    sample: Sample,
    ratio_name: str,
    priorities: List[str],
    *,
    freshness: Optional[Mapping[str, Any]] = None,
) -> Optional[SelectedRatioLayer]:
    """Return both the selected data and its stable layer key.

    A chain containing the Hg interference layer is a final or derived chain:
    a requested Hg correction that is unavailable leaves it with no layer
    rather than an uncorrected fallback. The same holds for a requested
    Pb-standard calibration that is unavailable.
    """
    if _final_value_blocked(sample, ratio_name, priorities, freshness=freshness):
        return None
    for layer in priorities:
        resolver = _LAYERS.get(layer)
        if resolver is None:
            continue
        cd = resolver(sample, ratio_name)
        if cd is not None:
            label = (
                pb_standard_layer_label(sample, ratio_name)
                if layer == PB_STANDARD_LAYER_KEY
                else _LAYER_LABELS[layer]
            )
            return SelectedRatioLayer(cd, layer, label)
    return None


def _get_ratio_source(sample: Sample, ratio_name: str) -> Optional[CycleData]:
    """Return the most relevant ratio source for aligned derived payloads."""
    return _resolve_layer_by_priority(sample, ratio_name, ["corrected", "blank_corrected", "ratios"])


def _align_series_to_source(
    values: np.ndarray,
    *,
    source_cd: Optional[CycleData],
    base_mask: Optional[np.ndarray] = None,
    extra_mask: Optional[np.ndarray] = None,
) -> Optional[CycleData]:
    """Align full-length or packed derived-cycle arrays to the source ratio mask."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None

    if source_cd is not None and len(values) == len(source_cd.values):
        mask = (
            np.asarray(base_mask, dtype=bool).copy()
            if base_mask is not None and len(base_mask) == len(values)
            else np.isfinite(values)
        )
        mask &= source_cd.mask
        if extra_mask is not None and len(extra_mask) == len(values):
            mask &= np.asarray(extra_mask, dtype=bool)
        return CycleData(values=values.copy(), mask=mask)

    if source_cd is None:
        mask = (
            np.asarray(base_mask, dtype=bool).copy()
            if base_mask is not None and len(base_mask) == len(values)
            else np.ones(len(values), dtype=bool)
        )
        if extra_mask is not None and len(extra_mask) == len(values):
            mask &= np.asarray(extra_mask, dtype=bool)
        return CycleData(values=values.copy(), mask=mask)

    full_values = np.full(len(source_cd.values), np.nan, dtype=np.float64)
    full_mask = np.zeros(len(source_cd.values), dtype=bool)

    valid_indices = np.where(source_cd.mask)[0]
    n = min(len(valid_indices), len(values))
    if n <= 0:
        return None

    packed_mask = (
        np.asarray(base_mask, dtype=bool)
        if base_mask is not None and len(base_mask) == len(values)
        else np.ones(len(values), dtype=bool)
    )
    full_values[valid_indices[:n]] = values[:n]
    full_mask[valid_indices[:n]] = packed_mask[:n]

    if extra_mask is not None and len(extra_mask) == len(values):
        full_mask[valid_indices[:n]] &= np.asarray(extra_mask[:n], dtype=bool)

    return CycleData(values=full_values, mask=full_mask)


def get_ssb_cycle_data(sample: Sample, ratio_name: str) -> Optional[CycleData]:
    """Build CycleData from an SSB payload when available."""
    if not sample.ssb_results or ratio_name not in sample.ssb_results:
        return None

    ssb_data = sample.ssb_results.get(ratio_name, {})
    cycles = ssb_data.get("ssb_corrected_cycles")
    if cycles is None:
        return None

    return _align_series_to_source(
        np.asarray(cycles, dtype=np.float64),
        source_cd=_get_ratio_source(sample, ratio_name),
        base_mask=ssb_data.get("ssb_mask"),
        extra_mask=ssb_data.get("ssb_outlier_mask"),
    )


def get_delta_cycle_data(sample: Sample, ratio_name: str) -> Optional[CycleData]:
    """Build CycleData from a delta payload when available."""
    if not sample.delta_results or ratio_name not in sample.delta_results:
        return None

    delta_data = sample.delta_results.get(ratio_name, {})
    cycles = delta_data.get("delta_per_cycle")
    if cycles is None:
        return None

    return _align_series_to_source(
        np.asarray(cycles, dtype=np.float64),
        source_cd=get_best_delta_ratio_data(sample, ratio_name),
        base_mask=delta_data.get("delta_mask"),
    )


def select_best_pre_drift_ratio_layer(
    sample: Sample,
    ratio_name: str,
) -> Optional[SelectedRatioLayer]:
    """Select the normalized measured-scale layer that drift must consume."""
    return _select_layer_by_priority(
        sample, ratio_name, [SR_STANDARD_LAYER_KEY, "iif", INTERFERENCE_LAYER_KEY, "corrected", "blank_corrected", "ratios"]
    )


def get_best_pre_drift_ratio_data(sample: Sample, ratio_name: str) -> Optional[CycleData]:
    """Return the highest-priority ratio series available before drift."""
    selected = select_best_pre_drift_ratio_layer(sample, ratio_name)
    return selected.data if selected is not None else None


def select_best_pre_ssb_ratio_layer(
    sample: Sample,
    ratio_name: str,
) -> Optional[SelectedRatioLayer]:
    """Select the measured-scale layer that SSB must consume."""
    return _select_layer_by_priority(
        sample,
        ratio_name,
        ["drift", SR_STANDARD_LAYER_KEY, "iif", INTERFERENCE_LAYER_KEY, "corrected", "blank_corrected", "ratios"],
    )


def get_best_pre_ssb_ratio_data(sample: Sample, ratio_name: str) -> Optional[CycleData]:
    """Return the measured-scale layer that SSB must consume."""
    selected = select_best_pre_ssb_ratio_layer(sample, ratio_name)
    return selected.data if selected is not None else None


def select_best_delta_ratio_layer(
    sample: Sample,
    ratio_name: str,
    *,
    freshness: Optional[Mapping[str, Any]] = None,
) -> Optional[SelectedRatioLayer]:
    """Select the final ratio layer and retain its scale-identifying key."""
    return _select_layer_by_priority(
        sample,
        ratio_name,
        [
            PB_STANDARD_LAYER_KEY, "ssb", "drift", SR_STANDARD_LAYER_KEY, "iif", INTERFERENCE_LAYER_KEY,
            "corrected", "blank_corrected", "ratios",
        ], freshness=freshness,
    )


def get_best_delta_ratio_data(sample: Sample, ratio_name: str) -> Optional[CycleData]:
    """Return the ratio layer that should govern delta calculations."""
    selected = select_best_delta_ratio_layer(sample, ratio_name)
    return selected.data if selected is not None else None


def get_processing_ratio_data(sample: Sample, ratio_name: str) -> Optional[CycleData]:
    """Return the canonical ratio layer used to define the processing mask."""
    return _resolve_layer_by_priority(sample, ratio_name, ["blank_corrected", "corrected", "ratios"])


def select_best_ratio_layer(
    sample: Sample,
    ratio_name: str,
    *,
    freshness: Optional[Mapping[str, Any]] = None,
) -> Optional[SelectedRatioLayer]:
    """Return the exact layer selected for charts and general result display."""
    return _select_layer_by_priority(
        sample,
        ratio_name,
        [
            PB_STANDARD_LAYER_KEY, "ssb", "drift", SR_STANDARD_LAYER_KEY, "iif", INTERFERENCE_LAYER_KEY,
            "corrected", "blank_corrected", "ratios",
        ], freshness=freshness,
    )


def get_best_ratio_data(
    sample: Sample, ratio_name: str, *, freshness: Optional[Mapping[str, Any]] = None,
) -> Optional[CycleData]:
    """Return the highest-priority ratio series available on *sample*."""
    selected = select_best_ratio_layer(sample, ratio_name, freshness=freshness)
    return selected.data if selected is not None else None


def align_cycle_series(
    cycle_inputs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    min_aligned_cycles: int = 4,
) -> Dict[str, np.ndarray]:
    """Return per-series values aligned by shared original cycle indices.

    Filters all input series by their masks and only includes indices where
    all series are finite and masked as True.
    """
    if not cycle_inputs:
        return {}

    lengths = {len(values) for values, _mask in cycle_inputs.values()}
    mask_lengths = {len(mask) for _values, mask in cycle_inputs.values()}
    if len(lengths) != 1 or lengths != mask_lengths:
        logger.warning("Cannot align cycle series with unequal value/mask lengths.")
        return {}
    min_len = next(iter(lengths))
    if min_len <= 0:
        return {}

    shared_mask = np.ones(min_len, dtype=bool)
    for values, mask in cycle_inputs.values():
        values_view = np.asarray(values[:min_len], dtype=np.float64)
        mask_view = np.asarray(mask[:min_len], dtype=bool)
        shared_mask &= mask_view & np.isfinite(values_view)

    if int(np.sum(shared_mask)) < min_aligned_cycles:
        return {}

    return {
        key: np.asarray(values[:min_len], dtype=np.float64)[shared_mask]
        for key, (values, _mask) in cycle_inputs.items()
    }

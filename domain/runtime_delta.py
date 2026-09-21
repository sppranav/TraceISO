"""
Runtime delta helpers.

Recomputes delta statistics from the current session view
(cycle range + active masks) without mutating pipeline outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from config.constants import DEFAULT_OUTLIER_THRESHOLD_SD
from config.settings import ProcessingConfig
from domain.corrections.delta import _compute_delta_stats
from domain.elements.base import ElementConfig
from domain.filters.outlier import apply_filter, resolve_cycle_range, sample_cycle_key
from domain.models import CycleData, Sample
from domain.observation_lookup import lookup_by_observation
from domain.layer_status import APPLIED
from domain.pb_calibration_records import (
    APPLIED_SESSION_MEAN_K,
    calibrated_delta_record,
    governing_calibration_record,
)
from domain.ratio_selection import (
    SelectedRatioLayer,
    SkippedStandard,
    StandardProbe,
    find_nearest_usable_standard,
    get_ssb_cycle_data,
    is_usable_bracketing_standard,
    select_best_delta_ratio_layer,
)

_log = logging.getLogger(__name__)


# (run_number, name, ratio_name, observation_id) — the same identity shape
# as RuntimeUncertaintyKey, for the same reason: names and run numbers both
# repeat, and two such observations carry different delta results.
RuntimeDeltaKey = Tuple[int, str, str, str]


@dataclass
class RuntimeDeltaResult:
    """Runtime delta statistics for one sample/ratio pair."""

    delta: float
    delta_sd: float
    delta_se: float
    delta_2se: float  # display convenience: two standard errors, not a coverage-k result
    n: int
    sample_mean: float
    std_mean: float
    prev_std: str
    next_std: str
    cycle_data: CycleData
    source_layer: str = ""
    reference_kind: str = ""
    skipped_standards: Tuple[SkippedStandard, ...] = ()
    prev_std_obs: str = ""
    next_std_obs: str = ""
    prev_std_n: Optional[int] = None
    next_std_n: Optional[int] = None
    prev_std_se: Optional[float] = None
    next_std_se: Optional[float] = None

    def to_payload(self) -> Dict[str, object]:
        """Convert to the dict shape used in ``sample.delta_results``.

        Does not mutate any stored sample — this is a read-only runtime view;
        ``skipped_standards`` reports this recomputation's own bracket-search
        provenance without writing it back onto the pipeline-time result.
        """
        return {
            "delta": self.delta,
            "delta_sd": self.delta_sd,
            "delta_se": self.delta_se,
            "delta_2se": self.delta_2se,
            "n": self.n,
            "sample_mean": self.sample_mean,
            "std_mean": self.std_mean,
            "prev_std": self.prev_std,
            "next_std": self.next_std,
            "prev_std_obs": self.prev_std_obs,
            "next_std_obs": self.next_std_obs,
            "prev_std_n": self.prev_std_n,
            "next_std_n": self.next_std_n,
            "prev_std_se": self.prev_std_se,
            "next_std_se": self.next_std_se,
            "delta_per_cycle": self.cycle_data.values.copy(),
            "delta_mask": self.cycle_data.mask.copy(),
            "source_layer": self.source_layer,
            "reference_kind": self.reference_kind,
            "skipped_standards": [s.to_payload() for s in self.skipped_standards],
        }


def make_runtime_delta_key(sample: Sample, ratio_name: str) -> RuntimeDeltaKey:
    """Stable key for runtime delta lookups, including the observation identity."""
    return (sample.run_number, sample.name, ratio_name, sample.observation_id)


def lookup_runtime_delta(
    results: Mapping[RuntimeDeltaKey, "RuntimeDeltaResult"],
    sample: Sample,
    ratio_name: str,
) -> Optional["RuntimeDeltaResult"]:
    """Find a sample's runtime delta result, tolerating a genuine legacy key.

    Shares :func:`domain.observation_lookup.lookup_by_observation` with
    :func:`domain.uncertainty.runtime.lookup_runtime_budget`, so a delta result
    and a budget answer the same question about the same session identically.
    """
    return lookup_by_observation(
        results,
        run_number=sample.run_number,
        name=sample.name,
        ratio_name=ratio_name,
        observation_id=sample.observation_id,
    )


def _resolve_runtime_reference_kind(
    ratio_name: str,
    samples: Iterable[Sample],
    *,
    processing_config: Optional[ProcessingConfig],
    element_config: Optional[ElementConfig],
) -> Optional[str]:
    """Resolve one certified/bracketing mode for a ratio across the session."""
    sample_list = list(samples)
    stored_kinds = {
        str(sample.delta_results.get(ratio_name, {}).get("reference_kind") or "")
        .strip()
        .lower()
        for sample in sample_list
        if sample.delta_results.get(ratio_name, {}).get("reference_kind")
    }
    stored_kinds &= {"certified", "bracketing"}
    if len(stored_kinds) > 1:
        _log.warning(
            "Runtime delta omitted for %s: conflicting stored reference kinds %s",
            ratio_name,
            sorted(stored_kinds),
        )
        return None
    if stored_kinds:
        return next(iter(stored_kinds))

    if any(ratio_name in (sample.ssb_results or {}) for sample in sample_list):
        return "certified"

    if processing_config is not None and not processing_config.enable_delta:
        return None
    if (
        processing_config is not None
        and element_config is not None
        and processing_config.enable_ssb
        and element_config.supports_ssb
    ):
        # A configured SSB workflow must fail closed when an individual sample
        # lacks SSB output. IIF output is evidence that the session instead used
        # an internally/externally normalized measured-scale branch.
        has_iif_output = any(
            ratio_name in (sample.iif_corrected_ratios or {})
            for sample in sample_list
        )
        if not has_iif_output:
            return "certified"

    return "bracketing"


def _resolve_runtime_reference_value(
    ratio_name: str,
    *,
    reference_kind: str,
    processing_config: ProcessingConfig,
    element_config: ElementConfig,
) -> Optional[float]:
    """Return the certified reference value for a certified-mode ratio.

    Returns ``None`` when classic bracketing against neighbouring standards should
    be used instead.
    """
    if reference_kind != "certified":
        return None
    cv = element_config.certified_values.get(ratio_name)
    if cv is None:
        return None
    value = float(cv.value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"Invalid certified reference value for runtime delta ratio "
            f"{ratio_name!r}: certified reference must be finite and > 0."
        )
    return value


def _select_delta_source_ratio(
    sample: Sample, ratio_name: str
) -> Optional[SelectedRatioLayer]:
    """Return the active ratio layer and stable key for runtime delta output."""
    return select_best_delta_ratio_layer(sample, ratio_name)


def get_runtime_filtered_cycle_data(
    cycle_data: CycleData,
    sample_name: str,
    *,
    sample_key: Optional[str] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
) -> CycleData:
    """Return full-length cycle data with current range + runtime filter applied."""
    values = np.asarray(cycle_data.values, dtype=np.float64).copy()
    base_mask = np.asarray(cycle_data.mask, dtype=bool).copy()

    if not cycle_ranges:
        return CycleData(values=values, mask=base_mask)

    cycle_range = resolve_cycle_range(
        cycle_ranges,
        sample_name=sample_name,
        sample_key=sample_key,
    )
    if cycle_range is None:
        return CycleData(values=values, mask=base_mask)
    # Match ratio statistics: selecting the whole extent preserves the single
    # processing filter, rather than applying another outlier pass.
    if int(cycle_range[0]) <= 1 and int(cycle_range[1]) >= len(values):
        return CycleData(values=values, mask=base_mask)

    runtime_mask = np.zeros(len(values), dtype=bool)
    start_idx = max(0, cycle_range[0] - 1)
    end_idx = min(len(values), cycle_range[1])
    if start_idx >= end_idx:
        _log.warning(  # item 73: use module-scope logger, avoid hot-path import
            "Cycle range (%s, %s) produces empty selection for %s (%d cycles available)",
            cycle_range[0], cycle_range[1], sample_name, len(values),
        )
        return CycleData(values=values, mask=runtime_mask)

    subset = values[start_idx:end_idx]
    subset_mask = base_mask[start_idx:end_idx]
    candidate_mask = subset_mask & np.isfinite(subset)

    if filter_method == "None":
        runtime_mask[start_idx:end_idx] = candidate_mask
        return CycleData(values=values, mask=runtime_mask)

    finite_vals = subset[candidate_mask]
    if len(finite_vals) < 3:
        runtime_mask[start_idx:end_idx] = candidate_mask
        return CycleData(values=values, mask=runtime_mask)

    result = apply_filter(finite_vals, filter_method, filter_threshold)
    # item 76: vectorised mask remap — avoid a Python-level loop over candidate indices
    kept_subset_mask = np.zeros(len(subset), dtype=bool)
    candidate_indices = np.where(candidate_mask)[0]
    n_filter = min(len(candidate_indices), len(result.mask))
    kept_subset_mask[candidate_indices[:n_filter]] = result.mask[:n_filter]

    runtime_mask[start_idx:end_idx] = kept_subset_mask
    return CycleData(values=values, mask=runtime_mask)


def _runtime_measured_se(values: "np.ndarray") -> Optional[float]:
    """Standard error of a runtime bracket side, or None when unavailable.

    Mirrors the production producer: fewer than two accepted cycles means the
    standard error was not measured, which is reported as None rather than as a
    zero that would claim the standard is exact (A009).
    """
    if values is None or len(values) < 2:
        return None
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _calibrated_runtime_delta(
    sample: Sample,
    ratio_name: str,
    *,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]],
    filter_method: str,
    filter_threshold: float,
    calibration_freshness: Optional[Mapping[str, Any]] = None,
) -> Optional[RuntimeDeltaResult]:
    """Calibrated Pb delta on the current window, from the stored calibration only.

    The per-cycle values are those the calibration wrote,
    ``1000 (X / B - 1)`` or ``1000 (X mean(1/S) - 1)``; the runtime window and
    filter select cycles on the calibrated final layer. No bracket is searched
    again, and nothing here is a combined uncertainty.
    """
    calibration = governing_calibration_record(sample, ratio_name)
    if calibration_freshness is not None:
        from domain.calibration_dependencies import effective_calibration_availability
        if not effective_calibration_availability(sample, ratio_name, calibration_freshness).current:
            return None
    delta_record = calibrated_delta_record(sample, ratio_name)
    stored = sample.pb_calibrated_delta_cycles.get(ratio_name)
    final = sample.pb_standard_corrected_ratios.get(ratio_name)
    if (
        calibration is None or calibration.status != APPLIED
        or delta_record is None or delta_record.status != APPLIED
        or stored is None or final is None
    ):
        return None
    if cycle_ranges and sample.observation_id in cycle_ranges:
        cycle_ranges = {sample_cycle_key(sample): cycle_ranges[sample.observation_id]}
    selected = get_runtime_filtered_cycle_data(
        final, sample.name, sample_key=sample_cycle_key(sample), cycle_ranges=cycle_ranges,
        filter_method=filter_method, filter_threshold=filter_threshold,
    )
    delta_cd = CycleData(values=stored.values.copy(), mask=stored.mask & selected.mask)
    valid = delta_cd.valid_values
    if len(valid) == 0:
        return None
    delta, delta_sd, delta_se, delta_2se, n = _compute_delta_stats(valid)
    denominator = 1.0 / float(delta_record.scale_factor)
    session = calibration.applied_mode == APPLIED_SESSION_MEAN_K
    prev_members = [m for m in calibration.members if m.side == "prev"]
    next_members = [m for m in calibration.members if m.side == "next"]
    return RuntimeDeltaResult(
        delta=delta, delta_sd=delta_sd, delta_se=delta_se, delta_2se=delta_2se, n=n,
        sample_mean=float(np.mean(final.values[delta_cd.mask])),
        std_mean=denominator,
        prev_std="session pool" if session else "+".join(m.label for m in prev_members),
        next_std="session pool" if session else "+".join(m.label for m in next_members),
        cycle_data=delta_cd,
        source_layer="pb_standard",
        reference_kind="pb_standard_calibration",
        prev_std_obs="" if session else "+".join(m.observation_id for m in prev_members),
        next_std_obs="" if session else "+".join(m.observation_id for m in next_members),
    )


def compute_runtime_delta(
    sample: Sample,
    ratio_name: str,
    all_samples: Iterable[Sample],
    *,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
    processing_config: Optional[ProcessingConfig] = None,
    element_config: Optional[ElementConfig] = None,
    reference_kind: Optional[str] = None,
    calibration_freshness: Optional[Mapping[str, Any]] = None,
    _std_cd_cache: Optional[Dict] = None,  # item 75: build-level memoisation dict
) -> Optional[RuntimeDeltaResult]:
    """Compute runtime delta for one sample/ratio from current masks and ranges."""
    if sample.metadata.get("excluded", False):
        return None
    if governing_calibration_record(sample, ratio_name) is not None:
        # A calibrated Pb result never falls back to classic bracketing.
        return _calibrated_runtime_delta(
            sample, ratio_name, cycle_ranges=cycle_ranges,
            filter_method=filter_method, filter_threshold=filter_threshold,
            calibration_freshness=calibration_freshness,
        )
    if not sample.is_sample:
        return None

    ordered = sorted(all_samples, key=lambda s: s.run_number)
    identities = [getattr(s, "observation_id", "") for s in ordered]
    target_id = getattr(sample, "observation_id", "")
    if (not target_id or any(not identity for identity in identities)
            or len(set(identities)) != len(identities)
            or target_id not in identities):
        _log.warning(
            "Runtime delta omitted for %s: missing, duplicate or absent observation "
            "identity; reload observations with unique IDs.", sample.name,
        )
        return None
    idx = identities.index(target_id)
    if cycle_ranges:
        for observation in ordered:
            if observation.observation_id in cycle_ranges:
                continue
            for key in (sample_cycle_key(observation), observation.name):
                if key in cycle_ranges and sum(
                    key in (sample_cycle_key(other), other.name) for other in ordered
                ) > 1:
                    _log.warning(
                        "Runtime delta omitted: ambiguous legacy window %r; "
                        "use observation identity or distinct session cycle keys.", key,
                    )
                    return None

    def ranges_for(observation):
        if cycle_ranges and observation.observation_id in cycle_ranges:
            return {sample_cycle_key(observation): cycle_ranges[observation.observation_id]}
        return cycle_ranges

    if reference_kind is None:
        reference_kind = _resolve_runtime_reference_kind(
            ratio_name,
            ordered,
            processing_config=processing_config,
            element_config=element_config,
        )
    if reference_kind not in {"certified", "bracketing"}:
        return None

    certified_ref: Optional[float] = None
    if (
        reference_kind == "certified"
        and processing_config is not None
        and element_config is not None
    ):
        certified_ref = _resolve_runtime_reference_value(
            ratio_name,
            reference_kind=reference_kind,
            processing_config=processing_config,
            element_config=element_config,
        )

    if reference_kind == "certified":
        if certified_ref is None:
            return None
        sample_source = get_ssb_cycle_data(sample, ratio_name)
        if sample_source is None:
            return None
        source_layer = "ssb"
        sample_cd = get_runtime_filtered_cycle_data(
            sample_source,
            sample.name,
            sample_key=sample_cycle_key(sample),
            cycle_ranges=ranges_for(sample),
            filter_method=filter_method,
            filter_threshold=filter_threshold,
        )
        sample_valid = sample_cd.valid_values
        if len(sample_valid) == 0:
            return None
        std_mean = certified_ref
        crm_name = None
        if processing_config is not None:
            crm_name = processing_config.reference_material or (
                element_config.reference_material if element_config is not None else None
            )
        _ref_label = f"{crm_name} (certified)" if crm_name else "(certified)"
        prev_std_name = _ref_label
        next_std_name = _ref_label
        skipped_standards: Tuple[SkippedStandard, ...] = ()
    else:
        sample_selected = _select_delta_source_ratio(sample, ratio_name)
        if sample_selected is None:
            return None
        sample_source = sample_selected.data
        source_layer = sample_selected.key

        # item 75: use build-level cache for standard cycle data when available.
        # Defined before the bracketing search so the search itself can apply
        # cycle-range/runtime filtering and skip candidates left with zero
        # valid cycles (e.g. a fully-excluded standard) instead of stopping there.
        def _get_std_cd_local(
            std: Sample, selected: SelectedRatioLayer
        ) -> Optional[CycleData]:
            if _std_cd_cache is not None:
                key = (std.observation_id, ratio_name, selected.key)
                if key not in _std_cd_cache:
                    _std_cd_cache[key] = get_runtime_filtered_cycle_data(
                        selected.data, std.name,
                        sample_key=sample_cycle_key(std),
                        cycle_ranges=ranges_for(std),
                        filter_method=filter_method,
                        filter_threshold=filter_threshold,
                    )
                return _std_cd_cache[key]
            # fallback without cache
            return get_runtime_filtered_cycle_data(
                selected.data, std.name,
                sample_key=sample_cycle_key(std),
                cycle_ranges=ranges_for(std),
                filter_method=filter_method,
                filter_threshold=filter_threshold,
            )

        def _probe(candidate: Sample) -> StandardProbe:
            if candidate.metadata.get("excluded", False):
                return StandardProbe(False, reason="runtime excluded")
            selected = _select_delta_source_ratio(candidate, ratio_name)
            if selected is None:
                return StandardProbe(False, reason="missing ratio layer")
            candidate_cd = _get_std_cd_local(candidate, selected)
            if not is_usable_bracketing_standard(candidate_cd):
                return StandardProbe(
                    False, reason="zero valid cycles (after runtime filter)"
                )
            return StandardProbe(True, layer_key=selected.key, cycle_data=candidate_cd)

        prev_result = find_nearest_usable_standard(
            ordered, idx, direction="before", probe=_probe
        )
        next_result = find_nearest_usable_standard(
            ordered, idx, direction="after", probe=_probe
        )

        if prev_result.sample is None or next_result.sample is None:
            return None
        if (
            prev_result.layer_key != source_layer
            or next_result.layer_key != source_layer
        ):
            return None

        prev_cd = prev_result.cycle_data
        next_cd = next_result.cycle_data

        sample_cd = get_runtime_filtered_cycle_data(
            sample_source,
            sample.name,
            sample_key=sample_cycle_key(sample),
            cycle_ranges=ranges_for(sample),
            filter_method=filter_method,
            filter_threshold=filter_threshold,
        )

        sample_valid = sample_cd.valid_values
        prev_valid = prev_cd.valid_values
        next_valid = next_cd.valid_values
        if len(sample_valid) == 0 or len(prev_valid) == 0 or len(next_valid) == 0:
            return None

        prev_mean = float(np.mean(prev_valid))
        next_mean = float(np.mean(next_valid))
        std_mean = (prev_mean + next_mean) / 2.0
        if not np.isfinite(std_mean) or std_mean == 0:
            return None
        prev_std_name = prev_result.name
        next_std_name = next_result.name
        skipped_standards = prev_result.skipped + next_result.skipped

    delta_values = np.full(len(sample_cd.values), np.nan, dtype=np.float64)
    delta_values[sample_cd.mask] = (sample_cd.values[sample_cd.mask] / std_mean - 1.0) * 1000.0
    delta_cd = CycleData(values=delta_values, mask=sample_cd.mask.copy())

    delta_valid = delta_cd.valid_values
    if len(delta_valid) == 0:
        return None

    sample_mean = float(np.mean(sample_valid))
    delta, delta_sd, delta_se, delta_2se, n = _compute_delta_stats(delta_valid)

    return RuntimeDeltaResult(
        delta=delta,
        delta_sd=delta_sd,
        delta_se=delta_se,
        delta_2se=delta_2se,
        n=n,
        sample_mean=sample_mean,
        std_mean=std_mean,
        prev_std=prev_std_name,
        next_std=next_std_name,
        cycle_data=delta_cd,
        source_layer=source_layer,
        reference_kind=reference_kind,
        skipped_standards=skipped_standards,
        prev_std_obs=(
            str(prev_result.sample.observation_id) if reference_kind == "bracketing" else ""
        ),
        next_std_obs=(
            str(next_result.sample.observation_id) if reference_kind == "bracketing" else ""
        ),
        prev_std_n=(len(prev_valid) if reference_kind == "bracketing" else None),
        next_std_n=(len(next_valid) if reference_kind == "bracketing" else None),
        # A009: None, never 0.0, when the side cannot supply a measured SE.
        prev_std_se=(
            _runtime_measured_se(prev_valid) if reference_kind == "bracketing" else None
        ),
        next_std_se=(
            _runtime_measured_se(next_valid) if reference_kind == "bracketing" else None
        ),
    )


def build_runtime_delta_map(
    samples: Iterable[Sample],
    ratio_names: Iterable[str],
    *,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
    processing_config: Optional[ProcessingConfig] = None,
    element_config: Optional[ElementConfig] = None,
    calibration_freshness: Optional[Mapping[str, Any]] = None,
) -> Dict[RuntimeDeltaKey, RuntimeDeltaResult]:
    """Compute runtime delta statistics for many sample/ratio combinations.

    item 75: ``all_samples`` is sorted once here and the sorted list is reused
    across all (sample, ratio) iterations.  Standard cycle data is memoised
    within the build so repeated calls to ``get_runtime_filtered_cycle_data``
    for the same standard are avoided.
    """
    # item 75: sort once and share the sorted list across all iterations
    sample_list = sorted(samples, key=lambda s: s.run_number)
    ratio_name_list = list(ratio_names)

    reference_kind_by_ratio = {
        ratio_name: _resolve_runtime_reference_kind(
            ratio_name,
            sample_list,
            processing_config=processing_config,
            element_config=element_config,
        )
        for ratio_name in ratio_name_list
    }

    # item 75: per-build memoisation of filtered standard cycle data
    # Map-local: windows/filter context is fixed for this build.
    # Keyed by observation identity, ratio and consumed source layer.
    _std_cd_cache: Dict[Tuple[str, str, str], Optional[CycleData]] = {}

    out: Dict[RuntimeDeltaKey, RuntimeDeltaResult] = {}
    for sample in sample_list:
        for ratio_name in ratio_name_list:
            reference_kind = reference_kind_by_ratio[ratio_name]
            if reference_kind is None and governing_calibration_record(sample, ratio_name) is None:
                continue
            result = compute_runtime_delta(
                sample,
                ratio_name,
                sample_list,
                cycle_ranges=cycle_ranges,
                filter_method=filter_method,
                filter_threshold=filter_threshold,
                processing_config=processing_config,
                element_config=element_config,
                reference_kind=reference_kind,
                calibration_freshness=calibration_freshness,
                _std_cd_cache=_std_cd_cache,
            )
            if result is not None:
                out[make_runtime_delta_key(sample, ratio_name)] = result
    return out

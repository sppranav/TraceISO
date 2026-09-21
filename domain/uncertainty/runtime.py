"""
Runtime uncertainty helpers.

Computes uncertainty budgets from the current session view
(cycle range + active masks) without mutating pipeline outputs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from config.reference_materials import get_crm_ratios, library_generation
from config.constants import SR_GEOREM_REFERENCE_MATERIAL
from config.settings import (
    CustomUncertaintyContributor,
    ProcessingConfig,
    UncertaintyConfig,
    is_russell_law_normalization_engine,
)
from domain.corrections.drift import build_drift_x_value_map
from domain.elements.base import CertifiedValue, ElementConfig
from domain.elements.crm_utils import resolve_optional_certified_value
from domain.filters.outlier import get_filtered_values, get_runtime_mask, sample_cycle_key
import numpy as np

from domain.models import Sample, UncertaintyBudget
from domain.observation_lookup import lookup_by_observation
from domain.pb_calibration_records import PB_CALIBRATION_BUDGET_ENGINE, governing_calibration_record
from domain.ratio_selection import get_best_ratio_data
from domain.uncertainty.eligibility import (
    generic_internal_unavailable_budget,
    hg_correction_unavailable_budget,
    missing_tl_normalized_layer_budget,
    pb_calibration_budget_guard,
    uses_generic_internal_normalization,
)
from domain.uncertainty.scope import is_invalid_budget_scope


# (run_number, name, ratio_name, observation_id). The first three fields keep
# the key readable and let a legacy three-part reference be matched as a
# prefix; the observation ID is what actually distinguishes two observations
# that share a name and a run number.
RuntimeUncertaintyKey = Tuple[int, str, str, str]
RuntimeUncertaintyMap = Dict[RuntimeUncertaintyKey, UncertaintyBudget]


# item 79: small LRU cache so repeated render cycles for the same
# (element, rm_name) pair do not re-traverse the CRM library dict.
#
# A017: the library generation is part of the key. An edited certificate keeps
# its element and its material name by design, so those two fields alone
# cannot distinguish before from after — this cache used to keep serving the
# superseded numbers to every runtime budget after a CRM Manager edit and a
# library reload, while the stored pipeline budgets already used the new ones.
@lru_cache(maxsize=32)
def _get_crm_ratios_for_generation(
    element_symbol: str, rm_name: str, generation: int
) -> Dict:
    """Cached wrapper around get_crm_ratios, keyed by library generation."""
    return get_crm_ratios(element_symbol, rm_name, derive=True)


def _get_crm_ratios_cached(element_symbol: str, rm_name: str) -> Dict:
    """Resolve certified ratios for the library that is loaded *now*."""
    return _get_crm_ratios_for_generation(
        element_symbol, rm_name, library_generation()
    )


def make_runtime_uncertainty_key(sample: Sample, ratio_name: str) -> RuntimeUncertaintyKey:
    """Stable key for runtime uncertainty lookups.

    Includes the observation identity: a name and a run number may both repeat
    within one session, and two such observations carry genuinely different
    budgets that must not collapse into one map entry.
    """
    return (sample.run_number, sample.name, ratio_name, sample.observation_id)


def lookup_runtime_budget(
    budgets: Mapping[RuntimeUncertaintyKey, UncertaintyBudget],
    sample: Sample,
    ratio_name: str,
) -> Optional[UncertaintyBudget]:
    """Find a sample's runtime budget, tolerating a genuine legacy key.

    The rule lives in :func:`domain.observation_lookup.lookup_by_observation`
    so the delta map answers identically. An entry that records a *different*
    observation is never this sample's budget, however alone it stands under
    the shared label; a sample absent from the map gets nothing.
    """
    return lookup_by_observation(
        budgets,
        run_number=sample.run_number,
        name=sample.name,
        ratio_name=ratio_name,
        observation_id=sample.observation_id,
    )


def select_best_ratio_data(sample: Sample, ratio_name: str):
    """Pick the highest-priority ratio series available on a sample."""
    return get_best_ratio_data(sample, ratio_name)


def resolve_runtime_drift_model_inputs(
    all_samples: Iterable[Sample],
    processing_config: Optional[ProcessingConfig],
    drift_fit_info: Optional[Dict[str, object]],
    ratio_name: str,
) -> Tuple[
    Optional[Callable[[np.ndarray], np.ndarray]],
    Optional[Callable[[Sample], float]],
]:
    """Rebuild the committed drift model and its position basis for runtime budgets."""
    if processing_config is None or not processing_config.drift.enabled:
        return None, None
    if not drift_fit_info:
        return None, None

    fit_ratio = str(drift_fit_info.get("ratio_name") or "").strip()
    if fit_ratio != ratio_name:
        return None, None

    coeffs_raw = drift_fit_info.get("coeffs")
    if coeffs_raw is None:
        return None, None

    try:
        coeffs = np.asarray(coeffs_raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None, None

    if coeffs.size == 0 or not np.all(np.isfinite(coeffs)):
        return None, None

    x_axis = str(
        drift_fit_info.get("x_axis")
        or processing_config.drift.x_axis
        or "run_number"
    )
    sample_list = list(all_samples)

    # Prefer the frozen coordinate map stored at pipeline time (item 142): this
    # prevents index-mode positions from shifting when samples are excluded after
    # processing.  Fall back to live enumeration only when the map is absent
    # (older fit_info without a frozen map, or run_number / time axes where
    # live lookup is equivalent).
    #
    # The map is keyed by observation identity. The older run-keyed map is
    # still read when that is all a stored fit carries, but only when its run
    # numbers are unique across the active samples: repeated run numbers mean
    # the producer collapsed several distinct coordinates onto one entry, and
    # replaying that map would predict the wrong position for every observation
    # it lost.
    stored_x_by_observation: Dict[str, float] = {
        str(k): float(v)
        for k, v in (drift_fit_info.get("sample_x_by_observation") or {}).items()
    }
    stored_x_by_run: Dict[int, float] = {}
    if not stored_x_by_observation:
        legacy_map = drift_fit_info.get("sample_x_by_run") or {}
        run_numbers = [int(s.run_number) for s in sample_list]
        if legacy_map and len(run_numbers) == len(set(run_numbers)):
            stored_x_by_run = {int(k): float(v) for k, v in legacy_map.items()}

    def _drift_model(x_values: np.ndarray) -> np.ndarray:
        x_array = np.asarray(x_values, dtype=np.float64)
        if drift_fit_info.get("fit_format") == "centered_scaled_v1":
            try:
                x_center = float(drift_fit_info["x_center"])
                x_scale = float(drift_fit_info["x_scale"])
            except (KeyError, TypeError, ValueError):
                return np.full(x_array.shape, np.nan, dtype=np.float64)
            if not np.isfinite(x_center) or not np.isfinite(x_scale) or x_scale == 0.0:
                return np.full(x_array.shape, np.nan, dtype=np.float64)
            x_array = (x_array - x_center) / x_scale
        return np.polyval(coeffs, x_array)

    # The repeatability estimator must know whether the curve was fitted to
    # these same observations and how many parameters were estimated.  Keep
    # the public return shape stable while carrying the frozen producer record
    # on the callable it reconstructs.
    setattr(_drift_model, "_traceiso_fit_info", dict(drift_fit_info))

    if stored_x_by_observation:
        def _position_extractor(sample: Sample) -> float:
            return float(
                stored_x_by_observation.get(
                    str(sample.observation_id), float(sample.run_number)
                )
            )
    elif stored_x_by_run:
        def _position_extractor(sample: Sample) -> float:
            return float(stored_x_by_run.get(int(sample.run_number), float(sample.run_number)))
    else:
        active_samples = [s for s in sample_list if not s.metadata.get("excluded", False)]
        x_value_by_sample = build_drift_x_value_map(
            sample_list,
            x_axis,
            active_samples=active_samples,
        )

        def _position_extractor(sample: Sample) -> float:
            return float(x_value_by_sample.get(id(sample), float(sample.run_number)))

    return _drift_model, _position_extractor


def _resolve_crm_uncertainty(
    element_config: Optional[ElementConfig],
    processing_config: ProcessingConfig,
    ratio_name: str,
) -> Tuple[float, float]:
    """Resolve the same validated authority used by runtime dispatch and MC."""
    cv = _resolve_crm_certified_value(element_config, processing_config, ratio_name)
    return (float(cv.uncertainty), float(cv.k)) if cv is not None else (0.0, 1.0)


def _resolve_crm_certified_value(
    element_config: Optional[ElementConfig],
    processing_config: ProcessingConfig,
    ratio_name: str,
) -> Optional[CertifiedValue]:
    """Resolve one authority; malformed or explicitly unresolved inputs fail closed."""
    if element_config is None:
        return None

    def validated(value, uncertainty, k, source):
        try:
            value, uncertainty, k = float(value), float(uncertainty), float(k)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Selected CRM value/uncertainty/coverage is unassigned for {ratio_name}.") from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"CRM value for {ratio_name!r} must be finite and > 0.")
        if not np.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError(f"CRM uncertainty for {ratio_name!r} must be finite and >= 0.")
        if not np.isfinite(k) or k <= 0:
            raise ValueError(f"CRM coverage factor k for {ratio_name!r} must be finite and > 0, got {k!r}.")
        return CertifiedValue(value=value, uncertainty=uncertainty, k=k, source=source)

    rm_name = processing_config.reference_material or element_config.reference_material
    if rm_name:
        payload = _get_crm_ratios_cached(element_config.symbol, rm_name).get(ratio_name)
        if payload is not None:
            if len(payload) < 2:
                raise ValueError(f"Selected CRM uncertainty is unassigned for {ratio_name}.")
            # Historical two-item tuples declare standard uncertainty. An explicit
            # missing/invalid third item is never replaced with a divisor of one.
            return validated(payload[0], payload[1], payload[2] if len(payload) >= 3 else 1.0, rm_name)
    if processing_config.reference_material:
        raise ValueError(f"Selected CRM {processing_config.reference_material!r} has no resolved value for {ratio_name}.")
    cv = element_config.certified_values.get(ratio_name)
    return validated(cv.value, cv.uncertainty, cv.k, cv.source) if cv is not None else None


def _resolve_ref_value_certified_value(
    element_config: Optional[ElementConfig],
    ratio_name: str,
) -> Optional[CertifiedValue]:
    """Resolve the fixed literature (GeoReM) reference value for Sr's u_ref_value.

    Unlike :func:`_resolve_crm_certified_value`, this is not affected by the
    user's CRM/anchoring selection — it always reads the GeoReM entry.
    """
    if element_config is None or element_config.symbol != "Sr":
        return None
    return resolve_optional_certified_value(
        element_config.symbol, SR_GEOREM_REFERENCE_MATERIAL, ratio_name,
    )


def compute_runtime_budget(
    sample: Sample,
    ratio_name: str,
    *,
    element_config: Optional[ElementConfig],
    processing_config: ProcessingConfig,
    uncertainty_config: Optional[UncertaintyConfig] = None,
    all_samples: Optional[Iterable[Sample]] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    filter_method: Optional[str] = None,
    filter_threshold: Optional[float] = None,
    drift_fit_info: Optional[Dict[str, object]] = None,
    custom_contributor_library: Optional[Dict[str, List[CustomUncertaintyContributor]]] = None,
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
) -> Optional[UncertaintyBudget]:
    """Compute an uncertainty budget from current session context."""
    # Match the pipeline's master gate. A read/export must not recreate budgets
    # the session explicitly disabled; None denotes absence, never valid zero.
    if uncertainty_config is not None and not uncertainty_config.enabled:
        return None
    if sample.is_blank or sample.metadata.get("excluded", False):
        return None

    # A requested Hg correction that is unavailable has no final ratio; say so
    # with the producer's reason instead of returning no budget at all.
    from domain.sr_standard_calibration import sr_calibration_budget_guard
    sr_refusal = sr_calibration_budget_guard(
        sample, ratio_name, uncertainty_config.output_mode if uncertainty_config is not None else "absolute_ratio",
    )
    if sr_refusal is not None:
        return sr_refusal

    hg_refusal = hg_correction_unavailable_budget(
        sample, ratio_name,
        uncertainty_config.output_mode if uncertainty_config is not None else "",
    )
    if hg_refusal is not None:
        return hg_refusal
    # A ratio governed by Pb-standard calibration: an applied absolute result goes
    # to the extended Engine C below; delta and unavailable results are refused
    # here instead of letting Engine C describe the Tl-only value.
    calibration_refusal = pb_calibration_budget_guard(
        sample, ratio_name,
        uncertainty_config.output_mode if uncertainty_config is not None else "absolute_ratio",
        extended_engine_c=True,
    )
    if calibration_refusal is not None:
        return calibration_refusal

    ratio_data = select_best_ratio_data(sample, ratio_name)
    if ratio_data is None:
        return None

    if uncertainty_config is not None:
        u_config = uncertainty_config
    else:
        # Default: infer output_mode from processing_config for backward compat.
        # When include_certified_uncertainty is True, use absolute_ratio so CRM
        # appears in the budget.
        u_config = UncertaintyConfig(
            output_mode=(
                "absolute_ratio" if processing_config.include_certified_uncertainty
                else "delta"
            ),
        )
    from domain.uncertainty.correction_state import (
        runtime_correction_state, synchronize_calculation_config,
    )
    samples_list = [s for s in (all_samples if all_samples is not None else [sample])
                    if not s.metadata.get("excluded", False)]
    processing_config, ssb_applied = runtime_correction_state(
        sample, samples_list, processing_config, element_config,
    )
    u_config = synchronize_calculation_config(
        u_config, processing_config, ssb_applied=ssb_applied,
    )
    # Synchronization may return the same mutable configuration. Validate even
    # when no mode flag changed, as the processing pipeline does on entry.
    u_config.validate()

    # A019: the pipeline refuses to model generic internal normalization, and the
    # runtime view cannot model it either.  Checked before the cycle-count gate
    # so both routes give the same reason rather than one reporting "insufficient
    # data" for a correction that has no engine at any cycle count.
    if uses_generic_internal_normalization(element_config, processing_config):
        return generic_internal_unavailable_budget(
            element_config.symbol, u_config.output_mode,
        )

    method = filter_method if filter_method is not None else processing_config.filter_method
    threshold = processing_config.get_active_filter_threshold(
        method,
        fallback_threshold=filter_threshold,
    )

    runtime_ratio_mask = get_runtime_mask(
        ratio_data.values, ratio_data.mask, sample.name,
        cycle_ranges=cycle_ranges, sample_key=sample_cycle_key(sample),
        filter_method=method, filter_threshold=threshold,
    )
    valid_values = np.asarray(ratio_data.values)[runtime_ratio_mask]

    if len(valid_values) < 2:
        # Return a minimal budget that carries the reason for UI/export
        return UncertaintyBudget(
            budget_scope="insufficient_data",
            scope_note=(
                f"Insufficient valid cycles ({len(valid_values)}) for "
                f"uncertainty computation — need at least 2."
            ),
        )

    if element_config is None:
        return None

    pb_tl_requested = (
        element_config.symbol == "Pb"
        and u_config.resolve_engine(
            element_config.symbol,
            processing_config=processing_config,
        )
        == "pb_tl_external_normalization"
    )
    if pb_tl_requested:
        # A012: the same check Engine C now applies on the stored route, from
        # the one shared builder so both reasons read identically.
        missing_layer = missing_tl_normalized_layer_budget(
            sample, ratio_name, u_config.output_mode,
        )
        if missing_layer is not None:
            return missing_layer

    drift_model, position_extractor = resolve_runtime_drift_model_inputs(
        samples_list,
        processing_config,
        drift_fit_info,
        ratio_name,
    )

    # Resolve CRM certified value
    cv = _resolve_crm_certified_value(element_config, processing_config, ratio_name)
    ref_cv = _resolve_ref_value_certified_value(element_config, ratio_name)

    resolved_engine = u_config.resolve_engine(
        element_config.symbol, processing_config=processing_config
    )

    if is_russell_law_normalization_engine(resolved_engine):
        # Internal-normalization engines (Sr and Pb-Tl)
        from domain.uncertainty.engine_internal_sr import compute_budget_internal

        budget = compute_budget_internal(
            sample,
            ratio_name,
            all_samples=samples_list,
            element_config=element_config,
            uncertainty_config=u_config,
            processing_config=processing_config,
            certified_value=cv,
            ref_certified_value=ref_cv,
            ratio_values=valid_values,
            runtime_ratio_mask=runtime_ratio_mask,
            cycle_ranges=cycle_ranges,
            drift_model=drift_model,
            position_extractor=position_extractor,
            custom_contributor_library=custom_contributor_library,
            profile_defaults=profile_defaults,
        )
        # A001/A019: only a budget the engine actually computed becomes a
        # runtime result.  An engine that refused - unavailable or
        # insufficient_data - keeps its own scope and its own reason; the
        # runtime view cannot supply the input the engine was missing, and
        # relabelling U=0 as "hybrid" would assert an exact zero.
        if budget is not None and not is_invalid_budget_scope(budget):
            original_scope_note = budget.scope_note
            if budget.budget_scope != "limited":
                budget.budget_scope = "hybrid"
            if budget.engine == PB_CALIBRATION_BUDGET_ENGINE:
                budget.scope_note = (
                    "u_prec is cycle-range-aware; standard means, K and the calibration "
                    "support are the processed calibration."
                )
            else:
                budget.scope_note = (
                    "u_prec and u_blank are cycle-range-aware; "
                    "u_std_repeatability is session-level."
                )
            budget.scope_note = (original_scope_note + " " + budget.scope_note).strip()
        return budget

    if governing_calibration_record(sample, ratio_name) is not None:
        # Only the Pb-Tl route runs the extended Engine C; never let Engine B
        # describe a calibrated result.
        return pb_calibration_budget_guard(sample, ratio_name, u_config.output_mode)

    # Engine B — SSB-delta
    from domain.uncertainty.engine_ssb import compute_budget_ssb

    budget = compute_budget_ssb(
        sample,
        ratio_name,
        all_samples=samples_list,
        element_config=element_config,
        uncertainty_config=u_config,
        processing_config=processing_config,
        certified_value=cv,
        ratio_values=valid_values,
        runtime_ratio_mask=runtime_ratio_mask,
        cycle_ranges=cycle_ranges,
        drift_model=drift_model,
        position_extractor=position_extractor,
        custom_contributor_library=custom_contributor_library,
        profile_defaults=profile_defaults,
    )
    if budget is not None and not is_invalid_budget_scope(budget):
        budget.budget_scope = "hybrid"
        budget.scope_note = (
            "u_prec is cycle-range-aware; "
            "u_std and u_std_repeatability are session-level."
        )
    return budget


def build_runtime_uncertainty_map(
    samples: Iterable[Sample],
    ratio_names: Iterable[str],
    *,
    element_config: Optional[ElementConfig],
    processing_config: ProcessingConfig,
    uncertainty_config: Optional[UncertaintyConfig] = None,
    all_session_samples: Optional[Iterable[Sample]] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    filter_method: Optional[str] = None,
    filter_threshold: Optional[float] = None,
    drift_fit_info: Optional[Dict[str, object]] = None,
    custom_contributor_library: Optional[Dict[str, List[CustomUncertaintyContributor]]] = None,
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
) -> RuntimeUncertaintyMap:
    """Compute runtime uncertainty budgets for many sample/ratio combinations.

    This is the uncached domain helper used by tests and non-Streamlit callers.
    Streamlit production views should use
    ``ui.runtime_budget_cache.build_cached_runtime_uncertainty_map``.
    """
    samples_list = list(samples)
    # A072: the ratio iterable is re-read for every sample, so a one-shot
    # generator must be materialized once before the loop.
    ratio_list = list(ratio_names)
    session_list = list(all_session_samples) if all_session_samples is not None else samples_list
    budgets: RuntimeUncertaintyMap = {}
    for sample in samples_list:
        for ratio_name in ratio_list:
            budget = compute_runtime_budget(
                sample,
                ratio_name,
                element_config=element_config,
                processing_config=processing_config,
                uncertainty_config=uncertainty_config,
                all_samples=session_list,
                cycle_ranges=cycle_ranges,
                filter_method=filter_method,
                filter_threshold=filter_threshold,
                drift_fit_info=drift_fit_info,
                custom_contributor_library=custom_contributor_library,
                profile_defaults=profile_defaults,
            )
            if budget is not None:
                budgets[make_runtime_uncertainty_key(sample, ratio_name)] = budget
    return budgets

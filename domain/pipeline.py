"""Unified processing pipeline for TraceISO."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

from config.settings import ProcessingConfig
from config.constants import (
    SR_GEOREM_REFERENCE_MATERIAL,
    SR_REFINEMENT_STABILITY_WARNING_REL,
)
from config.reference_materials import (
    get_crm_ratios,
    get_internal_normalization,
    get_masses_for_isotopes,
    get_natural_ratio,
    require_isotope_mass,
    require_natural_ratio,
)
from domain.corrections.blank import BLANK_CHANNEL_WEIGHTS_KEY
from domain.output_scale import (
    DRIFT_OUTPUT_FACTORS_KEY,
    SR_ANCHOR_RATIO_NAME_KEY,
)
from domain.models import CycleData, Sample, ProcessingResult, UncertaintyBudget
from domain.elements.base import CertifiedValue, DataLayer, ElementConfig, MonitorSpec
from domain.elements.crm_utils import resolve_optional_certified_value
from domain.elements.sr import SR_CORRECTION_ROLES
from domain.filters.outlier import apply_filter
from domain.ratio_utils import (
    element_symbol_from_isotope,
    normalize_ratio_name,
    normalize_ratio_token,
)
from domain.ratio_selection import get_ssb_cycle_data, intersect_required_channel_masks
from domain.sr_normalization_support import require_supported_sr_normalization_ratio
from domain.corrections.blank import apply_blank_correction, calculate_ratios
from domain.corrections.ssb import apply_ssb, apply_ssb_block
from domain.corrections.delta import calculate_deltas
from domain.corrections.mass_bias import (
    calculate_f_factor,
    calculate_k_factors,
    apply_iif_correction,
)
from domain.corrections.interference import (
    hg204_interference_correction,
)
from domain.corrections.sr_chain import (
    RETIRED_SR_SOLVER_METADATA_KEYS,
    SR_CHAIN_INITIALIZATION_PASSES,
    SR_CHAIN_METADATA_KEYS,
    SR_CHAIN_METHOD,
    SR_CHAIN_REFINEMENT_PASSES,
    SrChainResult,
    SrInterferenceTerm,
    refinement_stability,
    run_sr_chain,
    sr_chain_method_for_iterations,
)
from domain.corrections.drift import apply_drift_correction
from domain.pb_correction_records import (
    HG_RECORD_FAMILY,
    PB_HG_CORRECTION_SEMANTICS,
    PB_HG_CORRECTION_STATE_KEY,
    ROUTE_PB_TL,
    ROUTE_SSB,
)
from domain.pb_hg_correction import (
    apply_ssb_hg_correction,
    build_pb_tl_hg_record,
    resolve_hg_reference,
)
from domain.calibration_dependencies import build_calibration_dependencies
from domain.pb_calibration_records import PB_CALIBRATION_QUALITY_KEY, governing_calibration_record
from domain.pb_standard_calibration import (
    apply_pb_standard_calibration,
    calibration_requested,
    resolve_correction_context,
)
from domain.uncertainty.eligibility import (
    generic_internal_unavailable_budget,
    pb_calibration_budget_guard,
    uses_generic_internal_normalization,
)
from domain.layer_status import intrinsic_restorable_mask
from config.settings import UncertaintyConfig, is_russell_law_normalization_engine


# A030: the user's manual cycle exclusions and the run's own processing mask are
# different things with different lifetimes. The exclusion is an input the run
# inherits through the raw masks; the processing mask (outlier filtering,
# non-finite cycles) is an output of the run. The Inspector restores from
# ``_original_masks``, so that snapshot must describe the layers this run
# produced and must widen only the excluded positions.
USER_EXCLUSIONS_METADATA_KEY = "manual_exclusions"
ORIGINAL_MASKS_METADATA_KEY = "_original_masks"

MASK_SNAPSHOT_LAYERS = (
    "ratios",
    "blank_corrected_ratios",
    "corrected_ratios",
    "interference_corrected_ratios",
    "iif_corrected_ratios",
    "drift_corrected_ratios",
    "sr_standard_corrected_ratios",
    "pb_standard_corrected_ratios",
    "pb_calibrated_delta_cycles",
    "intensities",
    "blank_corrected_intensities",
    "corrected_intensities",
    "interference_corrected_intensities",
)


def _require_aligned(left, right) -> None:
    """Refuse, rather than truncate, arrays that were validated as aligned."""
    if len(left) != len(right):
        raise ValueError(
            "Pb-Tl channels were validated as aligned but differ in length; "
            "no cycle array is truncated."
        )


class ProcessingPipeline:
    """Config-driven processing pipeline."""

    def __init__(self, element: ElementConfig) -> None:
        self.element = element
        self._validate_monitor_role_coherence()
        self._validate_mass_bias_law()

    def _validate_monitor_role_coherence(self) -> None:
        """Fail fast if Step-1 monitors disagree with Step-2 correction roles."""
        monitors = self.element.monitors
        roles = dict(SR_CORRECTION_ROLES if self.element.symbol == "Sr" else {})
        roles.update(getattr(self.element, "correction_roles", {}) or {})
        if not monitors or not roles:
            return

        rb_interferent = roles.get("rb_interfering_mass")
        kr86_interferent = roles.get("kr86_interfering_mass")

        def _find_corrected(interferent_label: Optional[str]) -> Optional[str]:
            if interferent_label is None:
                return None
            normalized_interferent = normalize_ratio_token(interferent_label)
            return next(
                (
                    normalize_ratio_token(monitor.corrected_isotope)
                    for monitor in monitors
                    if monitor.family == "f"
                    and normalize_ratio_token(monitor.interfering_isotope)
                    == normalized_interferent
                ),
                None,
            )

        rb_target = _find_corrected(rb_interferent)
        kr86_target = _find_corrected(kr86_interferent)

        kr86_role_target = normalize_ratio_token(
            roles.get("kr86_target")
            or roles.get("normalization_numerator")
            or "86Sr"
        )
        if (
            kr86_target is not None
            and kr86_role_target is not None
            and kr86_target != kr86_role_target
        ):
            raise ValueError(
                f"ElementConfig invariant: Kr-86 monitor (interferent="
                f"'{kr86_interferent}') declares corrected_isotope="
                f"'{kr86_target}' but correction_roles['kr86_target']"
                f"='{kr86_role_target}'. Step 2 corrects '{kr86_role_target}' while Step 1 "
                f"corrects '{kr86_target}' - silently produces wrong results."
            )

        target_num = normalize_ratio_token(roles.get("target_numerator"))
        if rb_target is not None and target_num is not None and rb_target != target_num:
            raise ValueError(
                f"ElementConfig invariant: Rb monitor (interferent="
                f"'{rb_interferent}') declares corrected_isotope="
                f"'{rb_target}' but correction_roles['target_numerator']="
                f"'{target_num}'. Step 2 corrects '{target_num}' while Step 1 "
                f"corrects '{rb_target}' - silently produces wrong results."
            )

    def _validate_mass_bias_law(self) -> None:
        """Warn if mass_bias_law names a law that is not implemented."""
        law = self.element.mass_bias_law
        if law is not None and law != "russell":
            import warnings as _warnings

            _warnings.warn(
                f"ElementConfig.mass_bias_law='{law}' is set on element "
                f"'{self.element.symbol}', but only 'russell' is implemented. "
                f"calculate_f_factor() always uses Russell's exponential law; "
                f"this field is currently advisory only.",
                UserWarning,
                stacklevel=2,
            )

    def process(
        self,
        samples: List[Sample],
        settings: ProcessingConfig,
        uncertainty_config: Optional[UncertaintyConfig] = None,
        *,
        drift_cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
        cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> ProcessingResult:
        """Run the full correction chain on *samples*.

        The order of steps is determined by ``element.correction_steps``.
        Samples marked ``metadata["excluded"] = True`` are skipped.
        ``cycle_ranges`` select standard cycles for Sr and Pb calibration.
        Pb additionally digests its calibration windows for freshness.
        Other steps keep windows as a runtime view.
        """
        settings.validate()
        if uncertainty_config is not None:
            uncertainty_config.validate()
        warnings: List[str] = []

        # Separate excluded samples — they should not be processed
        prepared_samples = [self._prepare_sample_for_processing(s) for s in samples]
        active = [s for s in prepared_samples if not s.metadata.get("excluded", False)]
        excluded = [s for s in prepared_samples if s.metadata.get("excluded", False)]

        # Resolve certified values from selected CRM (or element defaults)
        certified_values = self._resolve_certified_values(settings)
        normalization_value = self._resolve_normalization_value(
            settings, certified_values,
        )


        if "blank" in self.element.correction_steps:
            result = apply_blank_correction(
                active,
                mode=settings.blank_mode,
                element_symbol=self.element.symbol,
            )
            active = result.samples
            warnings.extend(result.warnings)


        ratio_definitions = self._resolve_ratio_definitions_for_samples(active)
        active = calculate_ratios(
            active, ratio_definitions, use_corrected=True,
        )
        for sample in active:
            sample.blank_corrected_ratios = {
                k: v.copy() for k, v in sample.corrected_ratios.items()
            }

        # Apply outlier rejection once on the pre-downstream ratio basis:
        # blank-corrected ratios when available, otherwise corrected or raw.
        if "filter" in self.element.correction_steps:
            active = self._apply_filters(active, settings)


        # Sr is identified by having INTERFERENCE_CORRECTED in its data_layers
        # (the real Sr ElementConfig declares this; test stubs that only have
        # correction_steps=["uncertainty"] intentionally skip this branch).
        pb_tl_applied = False
        iif_applied = False
        use_ssb = False
        hg_route: Optional[str] = None
        ssb_ratios: set[str] = set()
        generic_internal_applied = False
        if DataLayer.IIF_CORRECTED in self.element.data_layers and self.element.symbol == "Sr":
            active = self._sr_correction_loop(
                active,
                settings,
                warnings,
                normalization_value,
            )
            iif_applied = True
            # Session anchoring (Approach 1)
            if settings.sr_session_anchoring and settings.apply_mass_bias_correction:
                active = self._apply_sr_session_anchoring(
                    active, certified_values, warnings,
                    standard_ids=settings.sr_calibration_standard_ids, cycle_ranges=cycle_ranges,
                )
        else:
            # External normalization for Pb using admixed Tl.
            # Only when mass-bias correction is requested AND a normalization
            # ratio is actually configured (via override or element default).
            has_norm_ratio = bool(
                settings.normalization_ratio_override or self.element.normalization_ratio
            )
            # Track whether the Pb–Tl mass-bias correction was actually applied,
            # so the SSB gate below can decide correctly.
            if self.element.symbol == "Pb" and settings.apply_mass_bias_correction and has_norm_ratio:
                # Only route to Pb-Tl correction if the normalization isotopes
                # are actually present in the loaded sample data.
                norm_num_iso, norm_den_iso, _, _ = self._resolve_normalization_pair(settings)
                norm_isotopes_present = any(
                    (not s.is_blank)
                    and norm_num_iso in (s.corrected_intensities or s.intensities)
                    and norm_den_iso in (s.corrected_intensities or s.intensities)
                    for s in active
                )
                if norm_isotopes_present:
                    active = self._pb_tl_correction(
                        active,
                        settings,
                        warnings,
                        normalization_value,
                        ratio_definitions,
                    )
                    iif_applied = True
                    pb_tl_applied = True
                else:
                    warnings.append(
                        f"Pb-Tl correction requested but normalization isotopes "
                        f"{norm_num_iso}/{norm_den_iso} are absent from all samples. "
                        f"Falling back to SSB-delta correction."
                    )
            elif settings.apply_mass_bias_correction and has_norm_ratio:
                active = self._internal_normalization_simple(
                    active,
                    settings,
                    warnings,
                    normalization_value,
                )
                iif_applied = True
                generic_internal_applied = uses_generic_internal_normalization(
                    self.element, settings,
                )

            # Simple elements: canonical filter mask → SSB → delta
            # SSB is used when no Pb–Tl external normalization was applied.
            use_ssb = (
                self.element.supports_ssb
                and settings.enable_ssb
                and not iif_applied
            )

            # Hg interference on 204Pb. Pb-Tl performs its own local subtraction
            # before Tl normalization; ordinary SSB subtracts here, after the
            # canonical filter and before drift and SSB.
            if self.element.symbol == "Pb":
                if pb_tl_applied:
                    hg_route = ROUTE_PB_TL
                elif use_ssb:
                    hg_route = ROUTE_SSB
                if settings.apply_hg_interference_correction:
                    if hg_route == ROUTE_SSB:
                        self._pb_ssb_hg_correction(
                            active, settings, warnings, normalization_value, ratio_definitions,
                        )
                    elif hg_route is None:
                        warnings.append(
                            "Hg interference correction was requested, but neither ordinary "
                            "SSB nor Pb-Tl external normalization is active, so no Hg "
                            "correction was applied."
                        )


        # Pb-standard calibration replaces separate drift fitting on its route.
        # Saved drift settings are kept, never applied, and the suppression is
        # recorded as requested versus effective state.
        cal_requested = calibration_requested(self.element.symbol, settings)
        drift_suppressed = cal_requested and bool(settings.drift.enabled)
        if drift_suppressed:
            warnings.append(
                "Drift correction is enabled in the saved settings but was not applied: separate "
                "drift fitting is disabled while Pb-standard calibration is enabled. The drift "
                "settings are kept unchanged."
            )
        if settings.drift.enabled and settings.drift.ratio_name and not drift_suppressed:
            drift_result = apply_drift_correction(
                samples=active,
                ratio_name=settings.drift.ratio_name,
                method=settings.drift.method,
                degree=settings.drift.degree,
                x_axis=settings.drift.x_axis,
                apply_outlier_filter=settings.drift.apply_outlier_filter,
                outlier_threshold=settings.drift.outlier_threshold,
                outlier_method=settings.drift.outlier_method,
                norm_mode=settings.drift.norm_mode,
                norm_standard=settings.drift.norm_standard,
                cycle_ranges=drift_cycle_ranges,
            )
            active = drift_result.samples
            warnings.extend(drift_result.warnings)
            drift_fit_info = drift_result.fit_info
        else:
            drift_fit_info = {}

        # Pb-standard calibration of the Tl-only layer. Run whenever it is
        # requested, so that a session where Tl normalization did not apply
        # reports each target as unavailable rather than an uncalibrated value.
        calibration_quality: Optional[Dict[str, object]] = None
        if cal_requested:
            if not pb_tl_applied:
                warnings.append(
                    "Pb-standard calibration was requested, but Pb-Tl external normalization was "
                    "not applied, so no calibrated value is reported."
                )
            context = resolve_correction_context(
                self.element, settings, ratio_definitions=ratio_definitions,
            )
            dependencies = build_calibration_dependencies(
                samples,
                element_symbol=self.element.symbol,
                ratio_definitions=ratio_definitions,
                settings=settings,
                cycle_ranges=cycle_ranges,
                correction_context=context,
            )
            calibration_quality = apply_pb_standard_calibration(
                active,
                element=self.element,
                settings=settings,
                cycle_ranges=cycle_ranges,
                correction_context=context,
                dependencies=dependencies,
                ratio_definitions=ratio_definitions,
                warnings=warnings,
            )

        # Supersedes DEC-016: drift remains after internal/external
        # normalization, but precedes SSB so samples and standards share the
        # measured scale before SSB maps samples to the certified scale.
        if use_ssb:
            for ratio_name in self.element.ratio_names:
                cv = certified_values.get(ratio_name)
                if cv is None:
                    continue
                ssb_ratios.add(ratio_name)
                if settings.ssb_mode == "block_average":
                    active = apply_ssb_block(
                        active, ratio_name, cv.value, use_corrected=True,
                    )
                else:
                    active = apply_ssb(
                        active, ratio_name, cv.value, use_corrected=True,
                    )

            if settings.enable_post_ssb_outliers:
                if settings.filter_method == "None" or settings.filter_method is None:
                    warnings.append(
                        "Post-SSB outlier filtering is enabled but no filter method is "
                        "selected (filter_method is None). The post-SSB filter pass was "
                        "skipped. Select a filter method to activate post-SSB filtering."
                    )
                else:
                    active = self._apply_post_ssb_filter(active, settings)


        # SSB-derived ratios use the certified reference. Other final layers
        # use classic bracketing on the same measured scale.
        if cal_requested and settings.enable_delta:
            warnings.append(
                "The classic delta setting is ignored on the Pb-standard calibration route; "
                "calibrated delta values come from the calibration itself."
            )
        if self.element.supports_delta and settings.enable_delta and not cal_requested:
            crm_name = settings.reference_material or self.element.reference_material
            for ratio_name in self.element.ratio_names:
                ref_val = None
                if ratio_name in ssb_ratios:
                    cv = certified_values.get(ratio_name)
                    if cv is not None:
                        ref_val = cv.value
                active = calculate_deltas(
                    active, ratio_name, use_corrected=True,
                    reference_value=ref_val,
                    reference_material_name=crm_name if ref_val is not None else None,
                )


        # Producer evidence is recorded even when budget calculation is disabled.
        # Missing channels within a Pb-Tl session do not turn that route into SSB.
        for sample in active:
            applied_state = {"ssb": bool(ssb_ratios), "pb_tl": bool(pb_tl_applied)}
            if cal_requested:
                applied_state.update({
                    "pb_standard_calibration": (calibration_quality or {}).get("applied_mode"),
                    "pb_standard_calibration_requested": True,
                    "drift_requested": bool(settings.drift.enabled),
                    "drift_effective": False,
                    "drift_suppressed_reason": (
                        "pb_standard_calibration_enabled" if drift_suppressed else None
                    ),
                })
                # A Monte Carlo record of a governed ratio describes one processed
                # calibration; reprocessing makes it stale, so it must not survive.
                for ratio_name in list(sample.mc_results):
                    if governing_calibration_record(sample, ratio_name) is not None:
                        sample.mc_results.pop(ratio_name, None)
            sample.metadata["_applied_correction_state"] = applied_state
            if self.element.symbol == "Pb":
                # R5: the request, the route that received it and the semantics
                # under which it was processed travel with the result.
                sample.metadata[PB_HG_CORRECTION_STATE_KEY] = {
                    "requested": bool(settings.apply_hg_interference_correction),
                    "route": hg_route,
                    "semantics_version": PB_HG_CORRECTION_SEMANTICS,
                }

        if "uncertainty" in self.element.correction_steps:
            if uncertainty_config is not None:
                u_config = uncertainty_config
            else:
                u_config = UncertaintyConfig()
            from domain.uncertainty.correction_state import synchronize_calculation_config
            u_config = synchronize_calculation_config(
                u_config, settings, ssb_applied=bool(ssb_ratios),
            )
            if u_config.enabled:
                if generic_internal_applied:
                    warnings.append(
                        "Uncertainty budget unavailable: generic internal normalization "
                        f"was applied to {self.element.symbol}, but no validated uncertainty "
                        "engine models that correction path."
                    )
                active = self._calculate_uncertainty(
                    active, settings, certified_values,
                    u_config=u_config,
                    pb_tl_applied=pb_tl_applied if self.element.symbol == "Pb" else True,
                    drift_fit_info=drift_fit_info,
                    generic_internal_applied=generic_internal_applied,
                    profile_defaults=profile_defaults,
                    cycle_ranges=cycle_ranges,
                    ratio_definitions=ratio_definitions,
                )

        # Recombine active + excluded, preserving run order
        all_samples = active + excluded
        all_samples.sort(key=lambda s: s.run_number)
        self._rebuild_original_mask_snapshots(all_samples)

        result = ProcessingResult(
            samples=all_samples,
            element_symbol=self.element.symbol,
            warnings=list(dict.fromkeys(warnings)),  # Deduplicate while preserving order
        )

        if drift_fit_info:
            result.quality_metrics["drift_fit_info"] = drift_fit_info
        if calibration_quality is not None:
            result.quality_metrics[PB_CALIBRATION_QUALITY_KEY] = calibration_quality

        return result


    def _rebuild_original_mask_snapshots(self, samples: List[Sample]) -> None:
        """Re-snapshot the pre-exclusion mask of every layer this run produced.

        A run inherits the user's manual exclusions through the raw masks, so the
        corrected and ratio layers it creates are already narrowed by them. The
        snapshot the Inspector restores from must therefore describe *these*
        layers rather than the raw-only snapshot that predates them; without it,
        clearing the exclusions leaves the newly produced layers still excluded.

        Only the manually excluded positions are widened, and only where the
        layer's own value is finite. Every other position keeps the mask the run
        decided on, so an outlier rejection is not silently undone by a user
        clearing an unrelated exclusion.
        """
        for sample in samples:
            excluded_cycles = sample.metadata.get(USER_EXCLUSIONS_METADATA_KEY)
            if not excluded_cycles:
                sample.metadata.pop(ORIGINAL_MASKS_METADATA_KEY, None)
                continue

            indices = [
                cycle - 1 for cycle in excluded_cycles
                if 1 <= cycle <= sample.n_cycles
            ]
            snapshot: Dict[str, Dict[str, np.ndarray]] = {}
            for label in MASK_SNAPSHOT_LAYERS:
                cd_dict = getattr(sample, label, None)
                if not cd_dict:
                    continue
                layer_snapshot = {}
                for key, cd in cd_dict.items():
                    mask = np.asarray(cd.mask, dtype=bool).copy()
                    restorable = intrinsic_restorable_mask(sample, label, key, cd)
                    for index in indices:
                        if index < len(mask):
                            mask[index] = bool(index < len(restorable) and restorable[index])
                    layer_snapshot[key] = mask
                if layer_snapshot:
                    snapshot[label] = layer_snapshot

            sample.metadata[ORIGINAL_MASKS_METADATA_KEY] = snapshot

    def _prepare_sample_for_processing(self, sample: Sample) -> Sample:
        """Start each processing run from raw inputs plus user metadata."""
        out = sample.copy()
        if not out.intensities:
            return out

        out.corrected_intensities = {}
        out.blank_corrected_intensities = {}
        out.corrected_ratios = {}
        out.blank_corrected_ratios = {}
        out.iif_corrected_ratios = {}
        out.sr_standard_corrected_ratios = {}
        out.drift_corrected_ratios = {}
        out.interference_corrected_intensities = {}
        out.interference_corrected_ratios = {}
        out.pb_standard_corrected_ratios = {}
        out.pb_calibrated_delta_cycles = {}
        out.correction_records = {}
        out.ssb_results = {}
        out.delta_results = {}
        out.uncertainty = {}
        out.used_blanks = {}
        out.warnings = []
        for key in (
            "_applied_correction_state",
            PB_HG_CORRECTION_STATE_KEY,
            "_processing_ratio_masks",
            "_sr_correction_invalid",
            "_pre_anchor_ratio",
            "_pre_anchor_ratio_name",
            "_sr_anchor_factor",
            SR_ANCHOR_RATIO_NAME_KEY,
            DRIFT_OUTPUT_FACTORS_KEY,
            BLANK_CHANNEL_WEIGHTS_KEY,
            "_sr_std_session_mean",
            "_sr_calibration_standard_ids",
            "sr_standard_calibration",
            "_k_86_88",
            "_k_87_86",
            ORIGINAL_MASKS_METADATA_KEY,
        ):
            out.metadata.pop(key, None)
        # Sr chain evidence belongs to the run that wrote it, and a retired
        # convergence claim must not outlive the method that made it.
        for key in SR_CHAIN_METADATA_KEYS + RETIRED_SR_SOLVER_METADATA_KEYS:
            out.metadata.pop(key, None)
        return out

    def _resolve_ratio_definitions_for_samples(
        self,
        samples: List[Sample],
    ) -> Dict[str, Tuple[str, str]]:
        """Merge element defaults with user-created ratios backed by intensities."""
        ratio_definitions: Dict[str, Tuple[str, str]] = dict(self.element.default_ratios)

        for sample in samples:
            available_isotopes = set(sample.intensities)
            available_isotopes.update(sample.corrected_intensities)
            for ratio_name in sample.ratios:
                normalized_name = normalize_ratio_name(ratio_name)
                parts = normalized_name.split("/")
                if len(parts) != 2:
                    continue

                numerator, denominator = parts
                if numerator not in available_isotopes or denominator not in available_isotopes:
                    continue

                ratio_definitions.setdefault(
                    normalized_name,
                    (numerator, denominator),
                )

        return ratio_definitions


    def _resolve_certified_values(
        self, settings: ProcessingConfig,
    ) -> Dict[str, CertifiedValue]:
        """Resolve certified values from the selected CRM."""
        crm_name = settings.reference_material or self.element.reference_material
        merged: Dict[str, CertifiedValue] = {} if settings.reference_material else {
            normalize_ratio_name(ratio_name): value
            for ratio_name, value in self.element.certified_values.items()
        }

        if crm_name:
            crm_ratios = get_crm_ratios(self.element.symbol, crm_name)
            if settings.reference_material and not crm_ratios:
                raise ValueError(f"Selected CRM {crm_name!r} cannot be resolved for {self.element.symbol}.")
            for ratio_name, (value, uncertainty, k) in crm_ratios.items():
                merged[normalize_ratio_name(ratio_name)] = CertifiedValue(
                    value=value,
                    uncertainty=uncertainty,
                    k=k,
                    source=crm_name,
                )

        return merged

    def _resolve_normalization_value(
        self,
        settings: ProcessingConfig,
        _certified_values: Dict[str, CertifiedValue],
    ) -> Optional[float]:
        """Resolve normalization ratio value for mass-bias corrections."""
        if settings.normalization_value_override is not None:
            return settings.normalization_value_override

        ratio_name = self._resolve_normalization_ratio_name(settings)
        if ratio_name is None:
            return self.element.normalization_value

        element_ratio = normalize_ratio_name(self.element.normalization_ratio or "")
        if ratio_name == element_ratio and self.element.normalization_value is not None:
            return self.element.normalization_value

        num_iso, den_iso, _m_num, _m_den = self._resolve_normalization_pair(settings)
        norm_element = element_symbol_from_isotope(num_iso)
        den_element = element_symbol_from_isotope(den_iso)
        if norm_element is None or den_element is None or norm_element != den_element:
            return self.element.normalization_value

        managed_internal = get_internal_normalization(norm_element)
        if managed_internal is not None:
            managed_ratio_name, managed_value = managed_internal
            if normalize_ratio_name(managed_ratio_name) == ratio_name:
                return managed_value

        natural_ratio = get_natural_ratio(norm_element, ratio_name)
        if natural_ratio is not None:
            return natural_ratio[0]

        return self.element.normalization_value

    def _resolve_normalization_ratio_name(
        self,
        settings: ProcessingConfig,
    ) -> Optional[str]:
        """Resolve the active normalization ratio name for this session.

        A039/D1: an Sr pair outside the supported envelope is refused here, at
        the one point every Sr correction, sensitivity and replay path reads, so
        production cannot disagree with Kragten or Monte Carlo about which model
        was applied.  A saved override is never coerced to the default.
        """
        ratio_name = settings.normalization_ratio_override or self.element.normalization_ratio
        if ratio_name is None:
            return None
        normalized = normalize_ratio_name(ratio_name)
        require_supported_sr_normalization_ratio(self.element, normalized)
        return normalized if normalized else None

    def _resolve_masses_for_isotopes(self, *isotope_names: str) -> Dict[str, float]:
        """Resolve managed masses for arbitrary isotopes."""
        try:
            return get_masses_for_isotopes(isotope_names)
        except ValueError:
            return {
                normalize_ratio_token(isotope_name): require_isotope_mass(isotope_name)
                for isotope_name in isotope_names
            }

    def _resolve_normalization_pair(
        self,
        settings: ProcessingConfig,
    ) -> Tuple[str, str, float, float]:
        """Resolve the active normalization isotopes and their masses."""
        ratio_name = self._resolve_normalization_ratio_name(settings)
        if ratio_name is None or ratio_name.count("/") != 1:
            raise ValueError("No valid normalization ratio is configured.")

        numerator, denominator = ratio_name.split("/", 1)
        numerator = normalize_ratio_token(numerator)
        denominator = normalize_ratio_token(denominator)
        masses = self._resolve_masses_for_isotopes(numerator, denominator)
        return numerator, denominator, masses[numerator], masses[denominator]

    def _resolve_monitor_inputs(
        self, monitors: Tuple["MonitorSpec", ...]
    ) -> Dict[str, float]:
        """
        Resolve managed masses and natural ratios for a MonitorSpec tuple.

        Returns a flat dict with:
        - isotope masses keyed by isotope name (e.g. ``"85Rb"``)
        - natural ratios keyed by ``MonitorSpec.natural_ratio_key``
        (e.g. ``"87Rb/85Rb"``)
        """
        result: Dict[str, float] = {}
        for spec in monitors:
            if not isinstance(spec, MonitorSpec):
                continue
            for iso in (spec.monitor_isotope, spec.interfering_isotope):
                if iso not in result:
                    result[iso] = require_isotope_mass(iso)
            if spec.natural_ratio_key not in result:
                numerator_iso = spec.natural_ratio_key.split("/", 1)[0]
                element_symbol = element_symbol_from_isotope(numerator_iso)
                if element_symbol is None:
                    raise ValueError(
                        f"Cannot derive element symbol from natural_ratio_key "
                        f"'{spec.natural_ratio_key}' (numerator '{numerator_iso}')."
                    )
                # require_natural_ratio returns (value, uncertainty, k);
                # only the value is needed here.
                result[spec.natural_ratio_key] = require_natural_ratio(
                    element_symbol, spec.natural_ratio_key
                )[0]
        return result

    def _resolve_sr_reference_inputs(
        self,
        settings: ProcessingConfig,
        _warnings: List[str],
        *,
        require_interference: bool,
        require_mass_bias: bool,
        monitor_inputs: Optional[Dict[str, float]] = None,
        enabled_interferents: Optional[set[str]] = None,
    ) -> Dict[str, Optional[float]]:
        """Resolve managed masses and natural ratios for the Sr correction chain."""
        roles = self._resolve_sr_correction_roles(settings)
        _norm_num_iso, _norm_den_iso, m_norm_num, m_norm_den = self._resolve_normalization_pair(
            settings,
        )
        target_den_iso = None
        primary_ratio = self.element.primary_ratio
        if primary_ratio is not None:
            target_ratio_def = self.element.default_ratios.get(primary_ratio)
            if target_ratio_def is not None:
                target_den_iso = target_ratio_def[1]
        inputs: Dict[str, Optional[float]] = {
            "m_norm_num": m_norm_num,
            "m_target_num": require_isotope_mass(roles["target_numerator"]) if (require_interference or require_mass_bias) else None,
            "m_target_den": require_isotope_mass(target_den_iso) if (require_interference or require_mass_bias) and target_den_iso else None,
            "m_norm_den": m_norm_den,
            "m83_kr": None,
            "m84_kr": None,
            "m86_kr": None,
            "m85_rb": None,
            "m87_rb": None,
            "rb87_rb85": None,
            "kr84_kr83": None,
            "kr86_kr83": None,
        }
        if require_interference:
            active_interferents = (
                {
                    spec.interfering_isotope
                    for spec in self.element.monitors
                    if isinstance(spec, MonitorSpec) and spec.family == "f"
                }
                if enabled_interferents is None
                else set(enabled_interferents)
            )
            if active_interferents and monitor_inputs is None:
                raise ValueError(
                    "Sr interference reference inputs require monitor_inputs. "
                    "Resolve ElementConfig.monitors with _resolve_monitor_inputs first."
                )
            monitor_inputs = monitor_inputs or {}

            def _natural_ratio_key_for(interfering_isotope: str) -> str:
                for spec in self.element.monitors:
                    if (
                        isinstance(spec, MonitorSpec)
                        and spec.family == "f"
                        and spec.interfering_isotope == interfering_isotope
                    ):
                        return spec.natural_ratio_key
                raise ValueError(
                    f"Sr correction requires a MonitorSpec for interferent "
                    f"{interfering_isotope!r}."
                )

            try:
                if roles["rb_interfering_mass"] in active_interferents:
                    rb_ratio_key = _natural_ratio_key_for(roles["rb_interfering_mass"])
                    inputs.update({
                        "m85_rb": monitor_inputs[roles["rb_monitor"]],
                        "m87_rb": monitor_inputs[roles["rb_interfering_mass"]],
                        "rb87_rb85": monitor_inputs[rb_ratio_key],
                    })
                if roles["kr84_interfering_mass"] in active_interferents:
                    kr84_ratio_key = _natural_ratio_key_for(roles["kr84_interfering_mass"])
                    inputs.update({
                        "m83_kr": monitor_inputs[roles["kr_monitor"]],
                        "m84_kr": monitor_inputs[roles["kr84_interfering_mass"]],
                        "kr84_kr83": monitor_inputs[kr84_ratio_key],
                    })
                if roles["kr86_interfering_mass"] in active_interferents:
                    kr86_ratio_key = _natural_ratio_key_for(roles["kr86_interfering_mass"])
                    inputs.update({
                        "m83_kr": monitor_inputs[roles["kr_monitor"]],
                        "m86_kr": monitor_inputs[roles["kr86_interfering_mass"]],
                        "kr86_kr83": monitor_inputs[kr86_ratio_key],
                    })
            except KeyError as exc:
                raise ValueError(
                    f"Sr correction requires monitor key {exc.args[0]!r} but "
                    f"it was not produced by _resolve_monitor_inputs. Check "
                    f"the element's MonitorSpec declarations."
                ) from None
        return inputs

    def _resolve_sr_correction_roles(
        self,
        settings: Optional[ProcessingConfig] = None,
    ) -> Dict[str, str]:
        """Return the declared Sr correction isotope roles."""
        roles = dict(SR_CORRECTION_ROLES if self.element.symbol == "Sr" else {})
        roles.update(getattr(self.element, "correction_roles", {}) or {})
        roles["kr86_target"] = normalize_ratio_token(
            roles.get("kr86_target")
            or roles.get("normalization_numerator")
            or "86Sr"
        )
        if settings is not None:
            ratio_name = self._resolve_normalization_ratio_name(settings)
            if ratio_name is not None and ratio_name.count("/") == 1:
                num_iso, den_iso = ratio_name.split("/", 1)
                roles["normalization_numerator"] = normalize_ratio_token(num_iso)
                roles["normalization_denominator"] = normalize_ratio_token(den_iso)
        required = (
            "normalization_numerator",
            "target_numerator",
            "normalization_denominator",
            "rb_monitor",
            "rb_interfering_mass",
            "kr_monitor",
            "kr84_target",
            "kr84_interfering_mass",
            "kr86_interfering_mass",
        )
        missing = [role for role in required if role not in roles]
        if missing:
            raise ValueError(
                "Sr correction roles are incomplete: missing "
                + ", ".join(sorted(missing))
            )
        return roles

    # Sr-specific iterative correction

    def _pb_tl_correction(
        self,
        samples: List[Sample],
        settings: ProcessingConfig,
        warnings: List[str],
        normalization_value: Optional[float],
        ratio_definitions: Mapping[str, Tuple[str, str]],
    ) -> List[Sample]:
        """Pb-Tl external normalization with optional 204Hg interference correction.

        Processing order (GUM-compliant, ref. Woodhead 2002, J. Anal. At. Spectrom.):
          1. blank-corrected intensities
          2. measured Tl ratio → Russell f_tl per cycle (calculate_f_factor)
          3. optional 204Hg correction of a local 204Pb copy monitored on 202Hg
          4. form Pb ratios (using Hg-corrected 204Pb for 204Pb-containing ratios)
          5. K-factor (IIF) calculation from the measured Tl ratio
          6. apply IIF correction → write to iif_corrected_ratios
          7. apply processing ratio masks

        Missing 202Hg with Hg flag enabled: emits a warning and omits all ratios
        that contain 204Pb — does NOT silently produce uncorrected values.

        NOTE: corrected_intensities["204Pb"] is NEVER overwritten.
        """
        if normalization_value is None:
            norm_ratio = (
                self._resolve_normalization_ratio_name(settings)
                or "configured normalization ratio"
            )
            raise ValueError(
                f"External normalization value is required for Pb–Tl correction ({norm_ratio}). "
                "Set it in CRM Library Manager or enter a session override in Session Configuration."
            )

        apply_hg = settings.apply_hg_interference_correction

        norm_ratio_name = self._resolve_normalization_ratio_name(settings)
        norm_num_iso, norm_den_iso, m_norm_num, m_norm_den = self._resolve_normalization_pair(
            settings,
        )

        # Resolve Hg natural ratio and masses once (required only when flag is set)
        hg_natural_ratio: Optional[float] = None
        m_hg202: Optional[float] = None
        m_hg204: Optional[float] = None
        if apply_hg:
            try:
                hg_natural_ratio = require_natural_ratio("Hg", "204Hg/202Hg")[0]
                hg_masses = self._resolve_masses_for_isotopes("202Hg", "204Hg")
                m_hg202 = hg_masses[normalize_ratio_token("202Hg")]
                m_hg204 = hg_masses[normalize_ratio_token("204Hg")]
            except Exception as exc:
                warnings.append(
                    f"Pb Hg correction: could not resolve Hg natural ratio or masses "
                    f"— Hg correction disabled. ({exc})"
                )
                apply_hg = False

        # C02 record evidence only; the arithmetic in this method is unchanged.
        hg_requested = bool(settings.apply_hg_interference_correction)
        hg_reference_payload: Dict[str, object] = {}
        if hg_requested:
            try:
                hg_reference_payload = resolve_hg_reference()
            except ValueError:
                hg_reference_payload = {}

        pb_ratio_definitions = {
            name: pair for name, pair in ratio_definitions.items()
            if all(str(isotope).endswith("Pb") for isotope in pair)
        }

        # Target mass map for each configured Pb ratio
        target_mass_map: Dict[str, Tuple[float, float]] = {}
        for ratio_name, (target_num_iso, target_den_iso) in pb_ratio_definitions.items():
            masses = self._resolve_masses_for_isotopes(target_num_iso, target_den_iso)
            target_mass_map[ratio_name] = (
                masses[normalize_ratio_token(target_num_iso)],
                masses[normalize_ratio_token(target_den_iso)],
            )

        for sample in samples:
            if sample.is_blank:
                continue

            src = sample.corrected_intensities if sample.corrected_intensities else sample.intensities
            if norm_num_iso not in src or norm_den_iso not in src:
                warnings.append(
                    f"Sample '{sample.name}': missing required isotopes "
                    f"{norm_num_iso} and/or {norm_den_iso} for Pb-Tl external normalization "
                    f"({norm_ratio_name or 'configured normalization ratio'})."
                )
                continue

            required_channels = {norm_num_iso, norm_den_iso}
            required_channels.update(
                iso for pair in pb_ratio_definitions.values() for iso in pair
                if iso in src
            )
            if apply_hg and "202Hg" in src:
                required_channels.add("202Hg")
            lengths = {iso: len(src[iso].values) for iso in sorted(required_channels)}
            if len(set(lengths.values())) != 1:
                warnings.append(
                    f"Sample '{sample.name}': incompatible Pb-Tl cycle lengths "
                    f"{lengths}; normalization unavailable, no cycle rows truncated."
                )
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                measured_ratio = src[norm_num_iso].values / src[norm_den_iso].values

            # Per-cycle f_tl (needed by Hg correction primitive)
            f_tl = calculate_f_factor(
                measured_ratio,
                normalization_value,
                m_norm_num,
                m_norm_den,
            )

            # Optional: Hg-correct a local copy of 204Pb (do NOT overwrite original)
            hg_corrected_pb204: Optional[np.ndarray] = None
            hg_correction_attempted = False
            hg_omission_reason = ""
            k_by_ratio: Dict[str, np.ndarray] = {}
            if apply_hg:
                hg_correction_attempted = True
                if "202Hg" not in src:
                    warnings.append(
                        f"Sample '{sample.name}': Hg interference correction requested but "
                        f"202Hg channel is absent — ratios containing 204Pb are omitted."
                    )
                    hg_omission_reason = "hg_monitor_absent"
                    # hg_corrected_pb204 stays None → 204Pb-containing ratios skipped below
                elif "204Pb" not in src:
                    warnings.append(
                        f"Sample '{sample.name}': 204Pb channel is absent — "
                        f"ratios containing 204Pb are omitted."  # item 56
                    )
                    hg_omission_reason = "pb204_absent"
                    # hg_corrected_pb204 stays None → 204Pb-containing ratios skipped below
                else:
                    pb204_vals = src["204Pb"].values.copy()  # local copy only
                    hg202_vals = src["202Hg"].values
                    # Channel lengths were validated above; nothing is truncated.
                    _require_aligned(pb204_vals, hg202_vals)
                    _require_aligned(pb204_vals, f_tl)
                    corrected_pb204, _ = hg204_interference_correction(
                        pb204=pb204_vals,
                        hg202=hg202_vals,
                        f_tl=f_tl,
                        hg204_hg202_natural=hg_natural_ratio,
                        m202=m_hg202,
                        m204_hg=m_hg204,
                    )
                    hg_corrected_pb204 = corrected_pb204
                    # C02: save the local corrected 204Pb as a diagnostic
                    # intermediate with its own explicit channel support.
                    intensity_support = np.ones(len(corrected_pb204), dtype=bool)
                    for iso in ("204Pb", "202Hg", norm_num_iso, norm_den_iso):
                        intensity_support &= src[iso].mask & np.isfinite(src[iso].values)
                    sample.interference_corrected_intensities["204Pb"] = CycleData(
                        values=np.array(corrected_pb204, dtype=float, copy=True),
                        mask=intensity_support,
                    )

            # K-factor and IIF correction for each Pb ratio
            for ratio_name, (target_num_iso, target_den_iso) in pb_ratio_definitions.items():
                uses_pb204 = (target_num_iso == "204Pb" or target_den_iso == "204Pb")

                if uses_pb204 and hg_correction_attempted and hg_corrected_pb204 is None:
                    # 202Hg absent — omit this ratio; do not produce uncorrected value
                    continue

                if uses_pb204 and hg_corrected_pb204 is not None:
                    # Reform ratio from intensities using Hg-corrected 204Pb. The
                    # channel lengths were validated above; nothing is truncated.
                    if target_den_iso == "204Pb":
                        if target_num_iso not in src:
                            continue
                        num_vals = src[target_num_iso].values
                        _require_aligned(num_vals, hg_corrected_pb204)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            ratio_vals = num_vals / hg_corrected_pb204
                    else:  # target_num_iso == "204Pb"
                        if target_den_iso not in src:
                            continue
                        den_vals = src[target_den_iso].values
                        _require_aligned(hg_corrected_pb204, den_vals)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            ratio_vals = hg_corrected_pb204 / den_vals
                    n = len(ratio_vals)
                    cd_orig = sample.corrected_ratios.get(ratio_name)
                    if cd_orig is not None:
                        _require_aligned(cd_orig.mask, ratio_vals)
                    mask = (
                        cd_orig.mask.copy()
                        if cd_orig is not None
                        else np.ones(n, dtype=bool)
                    )
                    cd = CycleData(values=ratio_vals, mask=mask)
                    # C02: save the interference-only ratio before Tl normalization,
                    # on the same channel support the normalized layer uses. It is a
                    # diagnostic intermediate on this route and never governs output.
                    sample.interference_corrected_ratios[ratio_name] = CycleData(
                        values=cd.values.copy(),
                        mask=intersect_required_channel_masks(
                            sample, ratio_name, src,
                            {target_num_iso, target_den_iso, norm_num_iso, norm_den_iso, "202Hg"},
                            cd.mask,
                        ),
                    )
                else:
                    # Use existing blank-corrected ratio (no 204Pb or Hg flag off)
                    cd = sample.corrected_ratios.get(ratio_name)
                    if cd is None:
                        cd = sample.ratios.get(ratio_name)
                    if cd is None:
                        continue

                m_target_num, m_target_den = target_mass_map[ratio_name]
                mb = calculate_k_factors(
                    normalization_ratio_measured=measured_ratio,
                    normalization_ratio_reference=normalization_value,
                    normalization_numerator_mass=m_norm_num,
                    target_numerator_mass=m_target_num,
                    normalization_denominator_mass=m_norm_den,
                    target_denominator_mass=m_target_den,
                )
                if mb.n_valid_cycles == 0:
                    warnings.append(
                        f"Sample '{sample.name}': normalization ratio "
                        f"({norm_ratio_name or 'configured normalization ratio'}) has no valid "
                        f"cycles. IIF correction skipped for {ratio_name}."
                    )
                    continue

                k_by_ratio[ratio_name] = mb.target_k
                n = len(cd.values)
                if n != len(mb.target_k):
                    warnings.append(f"Sample '{sample.name}': incompatible cycle length for {ratio_name}; normalization unavailable.")
                    continue
                if n <= 0:
                    continue
                required = {target_num_iso, target_den_iso, norm_num_iso, norm_den_iso}
                if uses_pb204 and hg_corrected_pb204 is not None:
                    required.add("202Hg")
                sample.iif_corrected_ratios[ratio_name] = CycleData(
                    values=apply_iif_correction(cd.values[:n], mb.target_k[:n]),
                    mask=intersect_required_channel_masks(
                        sample, ratio_name, src, required, cd.mask,
                    ),
                )

            if hg_requested:
                # Diagnostic records mirroring the legacy outcome; they never
                # gate the Tl-normalized result (governs_final=False).
                for ratio_name, pair in pb_ratio_definitions.items():
                    if "204Pb" not in pair:
                        continue
                    reason = hg_omission_reason or ("" if apply_hg else "hg_reference_unresolved")
                    sample.correction_records.setdefault(HG_RECORD_FAMILY, {})[ratio_name] = (
                        build_pb_tl_hg_record(
                            sample=sample,
                            ratio_name=ratio_name,
                            reason_code=reason,
                            hg_reference=hg_reference_payload,
                            tl_ratio_name=norm_ratio_name,
                            tl_value=normalization_value,
                            masses={
                                "202Hg": m_hg202, "204Hg": m_hg204,
                                norm_num_iso: m_norm_num, norm_den_iso: m_norm_den,
                            },
                            intensity_basis=(
                                "corrected_intensities" if sample.corrected_intensities
                                else "intensities"
                            ),
                            corrected_204=hg_corrected_pb204,
                            target_k=k_by_ratio.get(ratio_name),
                        )
                    )

        self._apply_processing_ratio_masks(samples)
        return samples

    def _pb_ssb_hg_correction(
        self,
        samples: List[Sample],
        settings: ProcessingConfig,
        warnings: List[str],
        normalization_value: Optional[float],
        ratio_definitions: Mapping[str, Tuple[str, str]],
    ) -> None:
        """Hg subtraction on 204Pb for ordinary Pb SSB, before drift and SSB.

        The Tl pair is resolved only to decide each measurement's Hg source; it
        does not externally normalize Pb. An unresolvable pair is not treated
        as absent Tl: measurements carrying Tl channels become unavailable.
        """
        try:
            tl_ratio_name = self._resolve_normalization_ratio_name(settings)
            tl_pair: Optional[Tuple[str, str, float, float]] = self._resolve_normalization_pair(settings)
        except ValueError:
            tl_ratio_name, tl_pair = None, None
        apply_ssb_hg_correction(
            samples,
            ratio_definitions=ratio_definitions,
            tl_pair=tl_pair,
            tl_ratio_name=tl_ratio_name,
            tl_reference_value=normalization_value,
            warnings=warnings,
        )

    def _internal_normalization_simple(
        self,
        samples: List[Sample],
        settings: ProcessingConfig,
        warnings: List[str],
        normalization_value: Optional[float],
    ) -> List[Sample]:
        """Apply one-pass Russell-law normalization for non-interference elements."""
        if normalization_value is None:
            norm_ratio = self._resolve_normalization_ratio_name(settings) or "configured normalization ratio"
            raise ValueError(
                f"Internal normalization value is required for mass-bias correction ({norm_ratio}). "
                "Set it in CRM Library Manager or enter a session override in Session Configuration."
            )

        norm_ratio_name = self._resolve_normalization_ratio_name(settings)
        norm_num_iso, norm_den_iso, m_norm_num, m_norm_den = self._resolve_normalization_pair(
            settings,
        )

        target_mass_map: Dict[str, Tuple[float, float]] = {}
        for ratio_name, (target_num_iso, target_den_iso) in self.element.default_ratios.items():
            masses = self._resolve_masses_for_isotopes(target_num_iso, target_den_iso)
            target_mass_map[ratio_name] = (
                masses[normalize_ratio_token(target_num_iso)],
                masses[normalize_ratio_token(target_den_iso)],
            )

        for sample in samples:
            if sample.is_blank:
                continue

            src = sample.corrected_intensities if sample.corrected_intensities else sample.intensities
            if norm_num_iso not in src or norm_den_iso not in src:
                warnings.append(
                    f"Sample '{sample.name}': missing required isotopes "
                    f"{norm_num_iso} and/or {norm_den_iso} for internal normalization "
                    f"({norm_ratio_name or 'configured normalization ratio'})."
                )
                continue

            if src[norm_num_iso].values.shape != src[norm_den_iso].values.shape:
                raise ValueError(f"Sample '{sample.name}': incompatible normalization-channel lengths.")
            norm_mask = src[norm_num_iso].mask & src[norm_den_iso].mask
            with np.errstate(divide="ignore", invalid="ignore"):
                measured_ratio = src[norm_num_iso].values / src[norm_den_iso].values
            norm_mask = norm_mask & np.isfinite(measured_ratio) & (measured_ratio > 0)

            for ratio_name, (_target_num_iso, _target_den_iso) in self.element.default_ratios.items():
                cd = sample.corrected_ratios.get(ratio_name)
                if cd is None:
                    cd = sample.ratios.get(ratio_name)
                if cd is None:
                    continue

                m_target_num, m_target_den = target_mass_map[ratio_name]
                mb = calculate_k_factors(
                    normalization_ratio_measured=measured_ratio,
                    normalization_ratio_reference=normalization_value,
                    normalization_numerator_mass=m_norm_num,
                    target_numerator_mass=m_target_num,
                    normalization_denominator_mass=m_norm_den,
                    target_denominator_mass=m_target_den,
                )
                if mb.n_valid_cycles == 0:
                    warnings.append(
                        f"Sample '{sample.name}': normalization ratio "
                        f"({norm_ratio_name or 'configured normalization ratio'}) has no valid cycles. "
                        f"IIF correction skipped for {ratio_name}."
                    )
                    continue

                if cd.values.shape != mb.target_k.shape:
                    raise ValueError(f"Sample '{sample.name}': incompatible target and normalization lengths for {ratio_name}.")
                corrected_values = apply_iif_correction(cd.values, mb.target_k)
                sample.iif_corrected_ratios[ratio_name] = CycleData(
                    values=corrected_values.copy(),
                    mask=(cd.mask & norm_mask & np.isfinite(corrected_values)).copy(),
                )

        self._apply_processing_ratio_masks(samples)
        return samples

    def _sr_correction_loop(
        self,
        samples: List[Sample],
        settings: ProcessingConfig,
        warnings: List[str],
        normalization_value: Optional[float],
    ) -> List[Sample]:
        """Sr interference and internal-normalization correction.

        An element declaring ``iterations >= 2`` runs the natural-ratio
        initialization plus exactly two K-factor refinements of
        :mod:`domain.corrections.sr_chain` (A025). An element declaring
        ``iterations <= 1`` selects the single-pass model: a natural-abundance
        correction scaled by the Russell exponent of the uncorrected
        normalization ratio. Both methods run through the same shared chain the
        uncertainty replays use, and the selected method is recorded. Every correction starts from the original
        blank-corrected intensities, and the IIF multiplication is applied once,
        per cycle, after the last evaluation.
        """
        roles = self._resolve_sr_correction_roles(settings)
        norm_num_iso = roles["normalization_numerator"]
        target_num_iso = roles["target_numerator"]
        norm_den_iso = roles["normalization_denominator"]
        norm_ratio_name = self._resolve_normalization_ratio_name(settings)
        norm_ratio_label = (
            norm_ratio_name or self.element.normalization_ratio or "normalization ratio"
        )

        if normalization_value is None and (
            settings.apply_mass_bias_correction or settings.apply_interference_correction
        ):
            norm_ratio = norm_ratio_name or self.element.normalization_ratio or "configured normalization ratio"
            raise ValueError(
                f"Internal normalization value is required for Sr corrections ({norm_ratio}). "
                "Set it in CRM Library Manager or enter a session override in Session Configuration."
            )

        if settings.apply_interference_correction and not self.element.monitors:
            raise ValueError(
                f"Sr interference correction requested on element "
                f"'{self.element.symbol}' but ElementConfig.monitors is empty. "
                f"Declare f-family MonitorSpec entries for the Sr "
                f"interferences."
            )

        enabled_monitors: Tuple[MonitorSpec, ...] = (
            tuple(
                spec
                for spec in self.element.monitors
                if (
                    isinstance(spec, MonitorSpec)
                    and spec.family == "f"
                    and settings.is_monitor_enabled(spec.interfering_isotope)
                )
            )
            if settings.apply_interference_correction
            else ()
        )
        enabled_interferents = {
            spec.interfering_isotope for spec in enabled_monitors
        }

        # Resolve monitor inputs once per pipeline run (single source of truth).
        # The Sr-flat dict (sr_reference_inputs) is then derived from it.
        _monitor_inputs: Dict[str, float] = (
            self._resolve_monitor_inputs(enabled_monitors)
            if enabled_monitors
            else {}
        )
        chain_active = bool(
            settings.apply_interference_correction or settings.apply_mass_bias_correction
        )
        sr_reference_inputs = (
            self._resolve_sr_reference_inputs(
                settings,
                warnings,
                require_interference=settings.apply_interference_correction,
                require_mass_bias=settings.apply_mass_bias_correction,
                monitor_inputs=_monitor_inputs,
                enabled_interferents=enabled_interferents,
            )
            if chain_active
            else {}
        )

        # Save original blank-corrected intensities per sample.
        original_intensities: Dict[int, Dict[str, CycleData]] = {}
        for sample in samples:
            if sample.is_blank:
                continue
            src = sample.corrected_intensities if sample.corrected_intensities else sample.intensities
            original_intensities[id(sample)] = {
                k: CycleData(values=v.values.copy(), mask=v.mask.copy())
                for k, v in src.items()
            }

        chain_method = sr_chain_method_for_iterations(self.element.iterations)
        chain_results: Dict[int, SrChainResult] = {}

        for sample in samples:
            if sample.is_blank:
                continue
            if sample.metadata.get("_sr_correction_invalid"):
                continue
            if not sample.corrected_intensities:
                # Work on a copy so no correction ever writes raw intensities.
                sample.corrected_intensities = {
                    k: CycleData(values=v.values.copy(), mask=v.mask.copy())
                    for k, v in sample.intensities.items()
                }
            src = sample.corrected_intensities

            # Need the declared normalization-pair isotopes for K and the
            # declared monitor isotopes for interference.
            if norm_num_iso not in src or norm_den_iso not in src:
                missing = [iso for iso in (norm_num_iso, norm_den_iso) if iso not in src]
                warnings.append(
                    f"Sample '{sample.name}': missing required isotopes {', '.join(missing)} for Sr correction."
                )
                continue

            if settings.apply_interference_correction and enabled_monitors:
                missing_monitors = sorted({
                    spec.monitor_isotope
                    for spec in enabled_monitors
                    if spec.monitor_isotope not in src
                })
                if missing_monitors:
                    missing_text = ", ".join(
                        f"{monitor} is missing" for monitor in missing_monitors
                    )
                    warnings.append(
                        f"Sample '{sample.name}': Sr interference correction skipped "
                        f"for monitors missing from the sample: {missing_text}."
                    )

            if not chain_active:
                continue
            chain_result = self._sr_chain_sample(
                sample,
                src,
                settings,
                warnings,
                method=chain_method,
                roles=roles,
                normalization_value=normalization_value,
                sr_reference_inputs=sr_reference_inputs,
                norm_ratio_label=norm_ratio_label,
            )
            if chain_result is not None and chain_method == SR_CHAIN_METHOD:
                chain_results[id(sample)] = chain_result

        # Recalculate corrected ratios from the final corrected intensities and
        # reapply the canonical pre-downstream processing mask.
        samples = calculate_ratios(
            samples,
            self._resolve_ratio_definitions_for_samples(samples),
            use_corrected=True,
        )
        self._apply_processing_ratio_masks(samples)

        primary_ratio = self.element.primary_ratio or f"{target_num_iso}/{norm_num_iso}"
        canonical = normalize_ratio_name(primary_ratio)

        if settings.apply_mass_bias_correction:
            for sample in samples:
                if sample.is_blank:
                    continue
                if sample.metadata.get("_sr_correction_invalid"):
                    continue
                k_87_86 = sample.metadata.get("_k_87_86")
                if k_87_86 is None:
                    continue

                # item 77: build pre-normalized lookup dicts once per sample
                # to avoid calling normalize_ratio_name inside the inner loops.
                normalized_corrected = {
                    normalize_ratio_name(k): v
                    for k, v in sample.corrected_ratios.items()
                }
                normalized_ratios = {
                    normalize_ratio_name(k): v
                    for k, v in sample.ratios.items()
                }

                # Find the corrected-ratio CycleData for the canonical key.
                cd = normalized_corrected.get(canonical) or normalized_ratios.get(canonical)
                if cd is None:
                    continue

                n = min(len(cd.values), len(k_87_86))
                if n <= 0:
                    continue

                iif_vals = apply_iif_correction(cd.values[:n], k_87_86[:n])
                # Always store under the single canonical key to avoid duplicate
                # entries (item 45).
                sample.iif_corrected_ratios[canonical] = CycleData(
                    values=iif_vals,
                    mask=intersect_required_channel_masks(
                        sample, canonical, original_intensities[id(sample)],
                        {target_num_iso, norm_num_iso, norm_den_iso} | {
                            spec.monitor_isotope for spec in enabled_monitors
                            if spec.corrected_isotope in {target_num_iso, norm_num_iso, norm_den_iso}
                            and spec.monitor_isotope in original_intensities[id(sample)]
                        },
                        cd.mask[:n],
                    ),
                )

        if chain_results:
            ratio_pair = self.element.default_ratios.get(primary_ratio)
            if ratio_pair is None and canonical.count("/") == 1:
                ratio_pair = tuple(canonical.split("/", 1))
            if ratio_pair is not None:
                self._record_sr_refinement_stability(
                    samples,
                    chain_results,
                    settings,
                    warnings,
                    canonical=canonical,
                    numerator=ratio_pair[0],
                    denominator=ratio_pair[1],
                )

        return samples

    def _sr_chain_sample(
        self,
        sample: Sample,
        src: Dict[str, CycleData],
        settings: ProcessingConfig,
        warnings: List[str],
        *,
        method: str,
        roles: Mapping[str, str],
        normalization_value: float,
        sr_reference_inputs: Mapping[str, Optional[float]],
        norm_ratio_label: str,
    ) -> Optional[SrChainResult]:
        """Run the selected Sr method for one sample through the shared chain.

        Writes the final corrected intensities and K-factors. A stage whose
        normalization ratio has no valid cycle marks the sample invalid; no
        earlier stage's result is substituted. The standard method refuses an
        enabled Rb correction whose target channel is missing; the single-pass
        method keeps its established behaviour of skipping that term.
        """
        standard = method == SR_CHAIN_METHOD
        sample.metadata["_sr_chain_method"] = method
        sample.metadata["_sr_chain_initialization_passes"] = SR_CHAIN_INITIALIZATION_PASSES
        sample.metadata["_sr_chain_refinement_passes"] = (
            SR_CHAIN_REFINEMENT_PASSES if standard else 0
        )

        target_num_iso = roles["target_numerator"]
        rb_monitor_iso = roles["rb_monitor"]
        kr_monitor_iso = roles["kr_monitor"]
        terms: List[SrInterferenceTerm] = []

        if settings.is_monitor_enabled(roles["rb_interfering_mass"]) and rb_monitor_iso in src:
            if target_num_iso in src:
                terms.append(
                    SrInterferenceTerm(
                        target=target_num_iso,
                        monitor=rb_monitor_iso,
                        natural_ratio=sr_reference_inputs["rb87_rb85"],
                        m_interferent=sr_reference_inputs["m87_rb"],
                        m_monitor=sr_reference_inputs["m85_rb"],
                    )
                )
            elif standard:
                warnings.append(
                    f"Sample '{sample.name}': missing required Rb target isotope "
                    f"{target_num_iso} for Sr correction."
                )
                sample.metadata["_sr_correction_invalid"] = True
                return None
        if kr_monitor_iso in src:
            if (
                settings.is_monitor_enabled(roles["kr84_interfering_mass"])
                and roles["kr84_target"] in src
            ):
                terms.append(
                    SrInterferenceTerm(
                        target=roles["kr84_target"],
                        monitor=kr_monitor_iso,
                        natural_ratio=sr_reference_inputs["kr84_kr83"],
                        m_interferent=sr_reference_inputs["m84_kr"],
                        m_monitor=sr_reference_inputs["m83_kr"],
                    )
                )
            if (
                settings.is_monitor_enabled(roles["kr86_interfering_mass"])
                and roles["kr86_target"] in src
            ):
                terms.append(
                    SrInterferenceTerm(
                        target=roles["kr86_target"],
                        monitor=kr_monitor_iso,
                        natural_ratio=sr_reference_inputs["kr86_kr83"],
                        m_interferent=sr_reference_inputs["m86_kr"],
                        m_monitor=sr_reference_inputs["m83_kr"],
                    )
                )

        result = run_sr_chain(
            {name: data.values for name, data in src.items()},
            terms=terms,
            normalization_numerator=roles["normalization_numerator"],
            normalization_denominator=roles["normalization_denominator"],
            normalization_reference=normalization_value,
            m_norm_num=sr_reference_inputs["m_norm_num"],
            m_norm_den=sr_reference_inputs["m_norm_den"],
            m_target_num=sr_reference_inputs["m_target_num"],
            m_target_den=sr_reference_inputs["m_target_den"],
            method=method,
        )
        for term in terms:
            src[term.target] = CycleData(
                values=np.array(result.final.intensities[term.target], dtype=float),
                mask=src[term.target].mask.copy(),
            )

        failed = result.failed_stage
        if failed is not None:
            stage_text = (
                "normalization ratio"
                if failed.label in ("initialization", "single_pass")
                else f"{failed.label.replace('_', ' ')} normalization ratio"
            )
            warnings.append(
                f"Sample '{sample.name}': {stage_text} ({norm_ratio_label}) has no "
                f"valid cycles — K-factors are NaN. Sr correction skipped for this "
                f"sample."
            )
            sample.metadata["_sr_correction_invalid"] = True
            return None

        sample.metadata["_k_86_88"] = result.final.mass_bias.normalization_k
        sample.metadata["_k_87_86"] = result.final.mass_bias.target_k
        return result

    @staticmethod
    def _record_sr_refinement_stability(
        samples: List[Sample],
        chain_results: Mapping[int, SrChainResult],
        settings: ProcessingConfig,
        warnings: List[str],
        *,
        canonical: str,
        numerator: str,
        denominator: str,
    ) -> None:
        """Report how far the final ratio moved between the two refinements.

        Summaries cover only cycles inside the reported layer's processing mask.
        A movement above ``SR_REFINEMENT_STABILITY_WARNING_REL`` raises a warning
        and keeps the second-refinement result; it is a numerical diagnostic,
        not an accuracy claim, and is kept out of the uncertainty budget.
        """
        apply_iif = bool(settings.apply_mass_bias_correction)
        for sample in samples:
            result = chain_results.get(id(sample))
            if result is None or sample.metadata.get("_sr_correction_invalid"):
                continue
            final_intensities = result.final.intensities
            if numerator not in final_intensities or denominator not in final_intensities:
                continue
            layer = sample.iif_corrected_ratios if apply_iif else sample.corrected_ratios
            reported = {normalize_ratio_name(k): v for k, v in layer.items()}.get(canonical)
            if reported is None:
                continue

            stability = refinement_stability(
                result,
                numerator=numerator,
                denominator=denominator,
                apply_iif=apply_iif,
                threshold_rel=SR_REFINEMENT_STABILITY_WARNING_REL,
                mask=reported.mask,
            )
            if stability is None:
                continue
            sample.metadata["_sr_refinement_change_abs"] = stability.change_abs
            sample.metadata["_sr_refinement_change_rel"] = stability.change_rel
            sample.metadata["_sr_refinement_n_compared"] = stability.n_compared
            sample.metadata["_sr_refinement_max_abs_change"] = stability.max_abs_change
            sample.metadata["_sr_refinement_max_rel_change"] = stability.max_rel_change
            sample.metadata["_sr_refinement_threshold_rel"] = stability.threshold_rel
            sample.metadata["_sr_refinement_exceeds_threshold"] = stability.exceeds_threshold
            if stability.exceeds_threshold:
                warnings.append(
                    f"Sample '{sample.name}': the final {canonical} moved by "
                    f"{stability.max_rel_change:.3g} relative "
                    f"({stability.max_abs_change:.3g} absolute) between the first "
                    f"and second Sr K-factor refinements, above the "
                    f"{stability.threshold_rel:g} relative stability warning "
                    f"threshold. The second-refinement result is retained. This is "
                    f"a numerical stability diagnostic, not an accuracy or "
                    f"uncertainty statement."
                )

    def _apply_sr_session_anchoring(
        self, samples: List[Sample], certified_values: Dict[str, "CertifiedValue"],
        warnings: List[str], *, standard_ids: List[str],
        cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> List[Sample]:
        """Calibrate on selected standard windows; retain normalized data on refusal."""
        from domain.calibration_dependencies import observation_window

        primary_ratio = self.element.primary_ratio or "87Sr/86Sr"
        selected_ids = set(standard_ids)
        for sample in samples:
            sample.sr_standard_corrected_ratios = {}
            for key in ("_pre_anchor_ratio", "_pre_anchor_ratio_name", "_sr_anchor_factor",
                        SR_ANCHOR_RATIO_NAME_KEY, "_sr_std_session_mean", "_sr_calibration_standard_ids",
                        "sr_standard_calibration"):
                sample.metadata.pop(key, None)

        def unavailable(code, reason):
            warnings.append(f"Sr calibration unavailable: {reason}")
            for sample in samples:
                if sample.is_blank:
                    continue
                sample.metadata["sr_standard_calibration"] = {
                    "method": "sr_explicit_standards.v2", "status": "unavailable",
                    "ratio": primary_ratio, "reason_code": code, "reason": reason,
                    "selected_standard_ids": sorted(selected_ids),
                }
            return samples

        if not selected_ids:
            return unavailable("no_standards_selected", "Select at least one Sr calibration standard and reprocess.")
        candidates = [s for s in samples if s.observation_id in selected_ids]
        if len(candidates) != len(selected_ids) or any(
            not s.is_standard or s.metadata.get("excluded", False) for s in candidates
        ):
            return unavailable("invalid_standard_selection", "The selection contains missing, excluded or non-standard runs. Reselect standards and reprocess.")
        cv = certified_values.get(primary_ratio)
        if cv is None or not np.isfinite(cv.value) or cv.value <= 0:
            return unavailable("reference_unavailable", f"A positive assigned reference value is required for {primary_ratio}.")
        standard_data = {}
        members = []
        for standard in candidates:
            cd = standard.iif_corrected_ratios.get(primary_ratio)
            if cd is None:
                return unavailable("standard_normalization_unavailable", f"Standard {standard.name} (run {standard.run_number}) has no internally normalized {primary_ratio} result.")
            try:
                window = observation_window(standard, cycle_ranges)
            except (TypeError, ValueError, IndexError):
                return unavailable("invalid_standard_window", f"Standard {standard.name} has an invalid cycle window. Select its cycles again and reprocess.")
            mask = cd.mask.copy()
            if window is not None:
                cycles = np.arange(1, len(mask) + 1)
                mask &= (cycles >= window[0]) & (cycles <= window[1])
            selected = CycleData(values=cd.values.copy(), mask=mask)
            if not selected.n_valid or not np.isfinite(selected.mean) or selected.mean <= 0:
                return unavailable("standard_support_unavailable", f"Standard {standard.name} (run {standard.run_number}) has no usable {primary_ratio} mean in its selected cycles.")
            standard_data[standard.observation_id] = selected
            members.append({
                "observation_id": standard.observation_id, "name": standard.name,
                "run_number": standard.run_number, "mean": selected.mean,
                "window": list(window) if window is not None else None,
                "n_valid": selected.n_valid, "accepted_cycles": (np.flatnonzero(mask) + 1).tolist(),
            })
        session_mean = float(np.mean([member["mean"] for member in members]))
        anchor_factor = cv.value / session_mean
        if not np.isfinite(anchor_factor) or anchor_factor <= 0:
            return unavailable("invalid_factor", "The Sr calibration factor is not finite and positive.")
        for sample in samples:
            if sample.is_blank:
                continue
            record = {
                "method": "sr_explicit_standards.v2", "status": "applied",
                "ratio": primary_ratio, "reference_value": float(cv.value),
                "standard_mean": session_mean, "k_factor": float(anchor_factor),
                "standards": [dict(member) for member in members],
            }
            sample.metadata["sr_standard_calibration"] = record
            cd = sample.iif_corrected_ratios.get(primary_ratio)
            if cd is None:
                record.update(status="unavailable", reason_code="sample_normalization_unavailable",
                              reason="This observation has no internally normalized ratio to calibrate.")
                continue
            if sample.is_standard:
                sample.metadata["_pre_anchor_ratio"] = standard_data.get(sample.observation_id, cd).copy()
                sample.metadata["_pre_anchor_ratio_name"] = primary_ratio
            sample.sr_standard_corrected_ratios[primary_ratio] = CycleData(
                values=cd.values * anchor_factor, mask=cd.mask.copy(),
            )
            sample.metadata["_sr_anchor_factor"] = anchor_factor
            sample.metadata[SR_ANCHOR_RATIO_NAME_KEY] = primary_ratio
            sample.metadata["_sr_std_session_mean"] = session_mean
            sample.metadata["_sr_calibration_standard_ids"] = sorted(selected_ids)
        return samples

    def _apply_filters(
        self, samples: List[Sample], settings: ProcessingConfig,
    ) -> List[Sample]:
        """Apply one canonical outlier mask to the processing ratio basis."""
        if settings.filter_method == "None":
            return samples

        for sample in samples:
            ratio_src = self._get_filter_basis_ratio_dict(sample)
            stored_masks: Dict[str, np.ndarray] = {}
            for ratio_name, cd in ratio_src.items():
                result = apply_filter(
                    cd.valid_values,
                    settings.filter_method,
                    settings.get_active_filter_threshold(settings.filter_method),
                )
                # Map filter result back to full-length mask
                valid_indices = np.where(cd.mask)[0]
                full_mask = cd.mask.copy()
                n_result = min(len(valid_indices), len(result.mask))
                full_mask[valid_indices[:n_result]] &= result.mask[:n_result]
                stored_masks[ratio_name] = full_mask

            if stored_masks:
                sample.metadata["_processing_ratio_masks"] = stored_masks
                self._apply_processing_ratio_masks([sample])

        return samples

    def _get_filter_basis_ratio_dict(self, sample: Sample) -> Dict[str, CycleData]:
        """Return the ratio layer used to define the canonical processing mask."""
        if sample.blank_corrected_ratios:
            return sample.blank_corrected_ratios
        if sample.corrected_ratios:
            return sample.corrected_ratios
        return sample.ratios

    def _apply_processing_ratio_masks(self, samples: List[Sample]) -> None:
        """Propagate stored canonical ratio masks to every available ratio layer."""
        ratio_layers = (
            "ratios",
            "blank_corrected_ratios",
            "corrected_ratios",
            "interference_corrected_ratios",
            "iif_corrected_ratios",
            "drift_corrected_ratios",
        )

        for sample in samples:
            stored_masks = sample.metadata.get("_processing_ratio_masks", {})
            if not stored_masks:
                continue

            normalized_layer_entries = []
            for layer_name in ratio_layers:
                cd_dict = getattr(sample, layer_name, None)
                if cd_dict:
                    normalized_layer_entries.extend(
                        (normalize_ratio_name(key), cd)
                        for key, cd in cd_dict.items()
                    )

            for ratio_name, stored_mask in stored_masks.items():
                normalized_ratio_name = normalize_ratio_name(ratio_name)
                base_mask = np.asarray(stored_mask, dtype=bool)
                for normalized_key, cd in normalized_layer_entries:
                    if normalized_key != normalized_ratio_name:
                        continue
                    n = min(len(cd.mask), len(base_mask))
                    if n > 0:
                        cd.mask[:n] = np.asarray(cd.mask[:n], dtype=bool) & base_mask[:n]

    def _apply_post_ssb_filter(
        self, samples: List[Sample], settings: ProcessingConfig,
    ) -> List[Sample]:
        """Apply outlier filter to post-SSB corrected cycles."""
        for sample in samples:
            for ratio_name, ssb_data in sample.ssb_results.items():
                ssb_cd = get_ssb_cycle_data(sample, ratio_name)
                if ssb_cd is not None and ssb_cd.n_valid > 2:
                    result = apply_filter(
                        ssb_cd.valid_values,
                        settings.filter_method,
                        settings.get_active_filter_threshold(
                            settings.filter_method,
                            fallback_threshold=settings.post_ssb_outlier_threshold,
                        ),
                    )
                    post_mask = ssb_cd.mask.copy()
                    valid_indices = np.where(ssb_cd.mask)[0]
                    n_result = min(len(valid_indices), len(result.mask))
                    post_mask[valid_indices[:n_result]] = result.mask[:n_result]
                    ssb_data["ssb_pre_filter_mask"] = np.asarray(
                        ssb_data.get("ssb_mask", ssb_cd.mask), dtype=bool
                    ).copy()
                    # Both keys intentionally store the final effective keep
                    # mask. get_ssb_cycle_data ANDs them idempotently; retain
                    # both serialized names for archive compatibility.
                    ssb_data["ssb_mask"] = post_mask.copy()
                    ssb_data["ssb_outlier_mask"] = post_mask
        return samples


    def _calculate_uncertainty(
        self,
        samples: List[Sample],
        settings: ProcessingConfig,
        certified_values: Optional[Dict[str, CertifiedValue]] = None,
        *,
        u_config: Optional[UncertaintyConfig] = None,
        pb_tl_applied: bool = True,
        drift_fit_info: Optional[Dict[str, object]] = None,
        generic_internal_applied: bool = False,
        profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
        cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
        ratio_definitions: Optional[Mapping[str, Tuple[str, str]]] = None,
    ) -> List[Sample]:
        """Calculate uncertainty budget for every sample and ratio."""
        if certified_values is None:
            certified_values = self.element.certified_values
        if u_config is None:
            u_config = UncertaintyConfig()

        if generic_internal_applied:
            for sample in samples:
                if sample.is_blank:
                    continue
                for ratio_name in self.element.ratio_names:
                    sample.uncertainty[ratio_name] = generic_internal_unavailable_budget(
                        self.element.symbol, u_config.output_mode,
                    )
            return samples

        from config.custom_uncertainty_contributors_loader import load_custom_contributors
        from domain.uncertainty.runtime import resolve_runtime_drift_model_inputs

        custom_contributor_library = load_custom_contributors()

        if self.element.symbol == "Pb" and not pb_tl_applied:
            # Tl isotopes were absent — Pb fell back to SSB-delta; force that engine.
            resolved_engine = "ssb_delta"
        else:
            resolved_engine = u_config.resolve_engine(
                self.element.symbol, processing_config=settings
            )

        budget_ratios = tuple((ratio_definitions or self.element.default_ratios).keys())
        for ratio_name in budget_ratios:
            cv = certified_values.get(ratio_name)
            if settings.reference_material and cv is None:
                from domain.uncertainty.eligibility import required_contributor_unavailable_budget
                for sample in samples:
                    if not sample.is_blank:
                        sample.uncertainty[ratio_name] = required_contributor_unavailable_budget(
                            engine=resolved_engine, output_mode=u_config.output_mode, contributor_name="selected CRM",
                            reason=f"Selected material {settings.reference_material!r} has no value for {ratio_name}.",
                            basis_ratio_value=0.0, n_cycles=0)
                continue
            if cv is None:
                crm_name = settings.reference_material or self.element.reference_material
                cv = resolve_optional_certified_value(
                    self.element.symbol, crm_name, ratio_name,
                )
            ref_cv = (
                resolve_optional_certified_value(
                    self.element.symbol, SR_GEOREM_REFERENCE_MATERIAL, ratio_name,
                )
                if self.element.symbol == "Sr"
                else None
            )
            drift_model, position_extractor = resolve_runtime_drift_model_inputs(
                samples,
                settings,
                drift_fit_info,
                ratio_name,
            )

            if is_russell_law_normalization_engine(resolved_engine):
                # Internal-normalization engines (Sr and Pb-Tl)
                from domain.uncertainty.engine_internal_sr import compute_budget_internal

                for sample in samples:
                    guard = pb_calibration_budget_guard(
                        sample, ratio_name, u_config.output_mode, extended_engine_c=True,
                    )
                    if guard is not None:
                        sample.uncertainty[ratio_name] = guard
                        continue
                    budget = compute_budget_internal(
                        sample,
                        ratio_name,
                        all_samples=samples,
                        element_config=self.element,
                        uncertainty_config=u_config,
                        processing_config=settings,
                        certified_value=cv,
                        ref_certified_value=ref_cv,
                        drift_model=drift_model,
                        position_extractor=position_extractor,
                        custom_contributor_library=custom_contributor_library,
                        profile_defaults=profile_defaults,
                        cycle_ranges=cycle_ranges,
                    )
                    if budget is not None:
                        sample.uncertainty[ratio_name] = budget
            else:
                # Engine B — SSB-delta
                from domain.uncertainty.engine_ssb import compute_budget_ssb

                for sample in samples:
                    guard = pb_calibration_budget_guard(sample, ratio_name, u_config.output_mode)
                    if guard is not None:
                        sample.uncertainty[ratio_name] = guard
                        continue
                    budget = compute_budget_ssb(
                        sample,
                        ratio_name,
                        all_samples=samples,
                        element_config=self.element,
                        uncertainty_config=u_config,
                        processing_config=settings,
                        certified_value=cv,
                        drift_model=drift_model,
                        position_extractor=position_extractor,
                        custom_contributor_library=custom_contributor_library,
                        profile_defaults=profile_defaults,
                    )
                    if budget is not None:
                        sample.uncertainty[ratio_name] = budget

        return samples

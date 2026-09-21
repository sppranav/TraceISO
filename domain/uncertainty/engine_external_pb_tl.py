"""Pb-Tl external-normalization uncertainty budget and replay helpers.

This module handles the Pb element when Tl-normalized mass-bias correction is
active. The deterministic replay helper is the single source of truth for:

- u_norm_ref
- u_interf
- u_blank
- Fixed-size Monte Carlo cross-check
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from config.contributor_names import LABEL_U_PREC

from config.crm_schema import UnassignedUncertaintyError
from config.reference_materials import (
    get_internal_normalization,
    get_natural_ratio_record,
    require_isotope_mass,
    require_natural_ratio,
    require_natural_ratio_relative_uncertainty,
    standard_uncertainty_from_values,
)
from config.settings import CustomUncertaintyContributor, ProcessingConfig, UncertaintyConfig
from domain.corrections.interference import hg204_interference_correction
from domain.corrections.mass_bias import (
    apply_iif_correction,
    calculate_f_factor,
    calculate_k_factors,
)
from domain.elements.base import CertifiedValue, ElementConfig
from domain.filters.outlier import (
    get_filtered_values,
    resolve_cycle_range,
    sample_cycle_key,
)
from domain.models import Sample, UncertaintyBudget, UncertaintyContributor
from domain.pb_calibration_records import governing_calibration_record
from domain.output_scale import (
    INPUT_LAYER_NORMALIZED,
    resolve_output_scale,
    scaled_contribution,
)
from domain.ratio_selection import get_best_pre_drift_ratio_data, get_best_ratio_data
from domain.ratio_utils import normalize_ratio_name, normalize_ratio_token
from domain.uncertainty.blank import (
    BlankUncertaintyResult,
    blank_input_sigmas,
    describe_blank_input_model,
    normalize_blank_uncertainty_input,
    resolve_blank_correction_mode,
    resolve_blank_samples,
    resolve_blank_samples_for_uncertainty,
    resolve_blank_selection,
)
from domain.uncertainty.contributors import (
    ContributorState,
    SampleContributorApplicability,
    build_custom_contributor_rows,
    inactive_reason_for_state,
    not_applicable_reason,
    resolve_contributor_state,
)
from domain.uncertainty.propagation import (
    select_precision_value,
    u_certified_value,
    u_precision,
)
from domain.uncertainty.reprod import compute_reprod
from domain.uncertainty.welch_satterthwaite import (
    coverage_factor as ws_coverage_factor,
    effective_dof,
    effective_dof_from_contributors,
    effective_dof_from_contributors_permil,
)
from domain.uncertainty.eligibility import (
    missing_tl_normalized_layer_budget,
    required_contributor_unavailable_budget,
    unresolved_blank_reference_budget,
)
from domain.uncertainty.shared_engine import combine_and_build_budget_shared


@dataclass(frozen=True)
class PbTlReferenceInputs:
    """Managed reference values required for the Pb-Tl replay chain."""

    ratio_name: str
    tl_norm_ratio_name: str
    tl_norm_value: float
    normalization_numerator: str
    normalization_denominator: str
    target_numerator: str
    target_denominator: str
    m_norm_num: float
    m_norm_den: float
    target_num_mass: float
    target_den_mass: float
    hg204_hg202_natural: Optional[float]
    m202: Optional[float]
    m204_hg: Optional[float]


@dataclass(frozen=True)
class PbTlBlankBlock:
    """One blank-file covariance block for replay-based Pb-Tl propagation."""

    isotopes: Tuple[str, ...]
    mean_vector: np.ndarray
    covariance_matrix: np.ndarray
    n_pairs: int
    degrees_of_freedom: float
    observation_id: str = ""
    channel_weights: Tuple[float, ...] = ()


def compute_budget_pb_tl(
    sample: Sample,
    ratio_name: str,
    *,
    all_samples: List[Sample],
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    processing_config: ProcessingConfig,
    certified_value: Optional[CertifiedValue] = None,
    ratio_values: Optional[np.ndarray] = None,
    ratio_mean: Optional[float] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    drift_model: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    position_extractor: Optional[Callable[[Sample], float]] = None,
    custom_contributor_library: Optional[Dict[str, List[CustomUncertaintyContributor]]] = None,
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
) -> Optional[UncertaintyBudget]:
    """Compute the Pb-Tl external-normalization uncertainty budget."""
    if sample.is_blank:
        return None

    # C05: a ratio whose final layer a Pb-standard calibration governs reports the
    # calibrated measurand, so the extended Engine C owns its budget; the Tl-only
    # budget below never describes it. Standards keep their Tl-only diagnostic.
    if governing_calibration_record(sample, ratio_name) is not None:
        from domain.uncertainty.engine_pb_calibrated import compute_budget_pb_calibrated

        return compute_budget_pb_calibrated(
            sample,
            ratio_name,
            all_samples=all_samples,
            element_config=element_config,
            uncertainty_config=uncertainty_config,
            processing_config=processing_config,
            ratio_values=ratio_values,
            cycle_ranges=cycle_ranges,
            custom_contributor_library=custom_contributor_library,
            profile_defaults=profile_defaults,
        )

    # A012: reaching Engine C means the session routed Pb-Tl normalization.  A
    # sample the correction skipped - absent 205Tl/203Tl, or a 204Pb ratio
    # omitted by the Hg guard - has no normalized layer for this ratio, and its
    # ratio is therefore the uncorrected one.  Checked before the ratio values
    # are read so this route and the runtime route give the same answer.
    missing_layer = missing_tl_normalized_layer_budget(
        sample, ratio_name, uncertainty_config.output_mode,
    )
    if missing_layer is not None:
        return missing_layer

    if (
        uncertainty_config.output_mode == "delta"
        and uncertainty_config.engine == "pb_tl_external_normalization"
        and processing_config.enable_delta
    ):
        raise ValueError("Engine C Pb-Tl uncertainty does not support delta output mode.")

    if ratio_values is None:
        cd = get_best_ratio_data(sample, ratio_name)
        if cd is None:
            return None
        ratio_values = cd.valid_values

    finite_cycle_count = int(np.sum(np.isfinite(ratio_values)))
    if finite_cycle_count < 2:
        return UncertaintyBudget(
            engine="pb_tl_external_normalization",
            output_mode=uncertainty_config.output_mode,
            budget_scope="insufficient_data",
            scope_note=(
                f"Insufficient valid cycles ({finite_cycle_count}) for "
                "uncertainty computation - need at least 2."
            ),
            n_cycles=finite_cycle_count,
        )

    if ratio_mean is None:
        finite = ratio_values[np.isfinite(ratio_values)]
        ratio_mean = float(np.mean(finite)) if len(finite) > 0 else 0.0

    if not np.isfinite(ratio_mean) or ratio_mean == 0.0:
        return None

    contributors: List[UncertaintyContributor] = []

    # Build custom contributor name set for resolver look-up.
    _custom_names = {
        d.name
        for defs in (custom_contributor_library or {}).values()
        for d in defs
    }
    _applicability = SampleContributorApplicability.from_sample(
        sample,
        known_profiles=set(profile_defaults.keys()) if profile_defaults is not None else None,
    )

    def _resolver_state(name: str) -> ContributorState:
        return resolve_contributor_state(
            name=name,
            uncertainty_config=uncertainty_config,
            sample=sample,
            element_symbol=element_config.symbol,
            custom_contributor_names=_custom_names,
            profile_defaults=profile_defaults,
        )

    def _active(name: str) -> bool:
        return _resolver_state(name) == ContributorState.ACTIVE

    def _contributor_gate(
        name: str,
        data_available: bool = True,
        missing_reason: str = "",
        not_applicable_reason: str = "",
    ) -> dict:
        state = _resolver_state(name)
        if state == ContributorState.ACTIVE and not_applicable_reason:
            # Enabled but outside the reported quantity: neither a gap nor a
            # configuration choice, so it must not read as missing data.
            return {
                "is_active": False,
                "state": ContributorState.NOT_APPLICABLE.value,
                "inactive_reason": not_applicable_reason,
            }
        if state == ContributorState.ACTIVE and not data_available:
            state = ContributorState.MISSING_DATA
            if missing_reason:
                return {
                    "is_active": False,
                    "state": state.value,
                    "inactive_reason": missing_reason,
                }
        return {
            "is_active": state == ContributorState.ACTIVE,
            "state": state.value,
            "inactive_reason": inactive_reason_for_state(
                state,
                contributor_name=name,
                profile=_applicability.profile,
            ),
        }

    ratio_mask = _resolve_pb_tl_ratio_mask(
        sample,
        ratio_name,
        cycle_ranges=cycle_ranges,
    )

    # 1. u_prec
    u_prec_se_abs, u_prec_sd_abs, _mean_val = u_precision(ratio_values)
    u_prec_abs, u_prec_mode = select_precision_value(
        getattr(uncertainty_config, "u_prec_mode", "se"),
        u_prec_se_abs,
        u_prec_sd_abs,
    )
    n_cycles = int(np.sum(np.isfinite(ratio_values)))

    # A002 follow-up: see the identical gate in Engine A. An unresolvable blank
    # reference is an unknown contribution; a budget that simply omits it would
    # claim completeness it does not have.
    blank_selection = resolve_blank_selection(sample, all_samples)
    if _active("u_blank") and not blank_selection.is_fully_resolved:
        return unresolved_blank_reference_budget(
            sample,
            blank_selection,
            engine="pb_tl_external_normalization",
            output_mode=uncertainty_config.output_mode,
            basis_ratio_value=ratio_mean,
            n_cycles=n_cycles,
        )

    u_prec_rel_permil = (u_prec_abs / abs(ratio_mean)) * 1000.0 if ratio_mean else 0.0
    u_prec_description = (
        "Within-run standard deviation of per-cycle Tl-normalized ratios."
        if u_prec_mode == "sd"
        else "Standard error of the mean of per-cycle Tl-normalized ratios."
    )
    contributors.append(
        UncertaintyContributor(
            name="u_prec",
            display_name=f"{LABEL_U_PREC} (Type A)",
            value_abs=u_prec_abs,
            value_rel_permil=u_prec_rel_permil,
            type_ab="A",
            degrees_of_freedom=float(max(n_cycles - 1, 1)),
            percentage_contribution=0.0,
            description=u_prec_description,
            **_contributor_gate("u_prec"),
        )
    )

    # 2. u_std_repeatability
    reprod_method = uncertainty_config.resolve_reprod_method(
        enable_ssb=uncertainty_config.enable_ssb,
        enable_delta=uncertainty_config.enable_delta,
    )
    reprod_result = compute_reprod(
        all_samples=all_samples,
        ratio_name=ratio_name,
        uncertainty_config=uncertainty_config,
        element_config=element_config,
        drift_model=drift_model,
        ratio_extractor=(
            get_best_pre_drift_ratio_data
            if reprod_method == "drift_residuals" and drift_model is not None
            else None
        ),
        position_extractor=position_extractor,
    )
    if _active("u_std_repeatability") and reprod_result.status == "unavailable":
        return required_contributor_unavailable_budget(
            engine="pb_tl_external_normalization",
            output_mode=uncertainty_config.output_mode,
            contributor_name="u_std_repeatability",
            reason=reprod_result.unavailable_reason or "repeatability inference is unsupported",
            basis_ratio_value=ratio_mean,
            n_cycles=n_cycles,
        )
    u_reprod_abs = float(reprod_result.u_std_repeatability_abs)
    repeatability_se = str(uncertainty_config.std_repeatability_mode).lower() == "se"
    n_standards = sum(bool(included) for included in reprod_result.std_included)
    if repeatability_se:
        # Same session-mean transfer as Sr: divide once by sqrt(included runs).
        u_reprod_abs /= np.sqrt(max(n_standards, 1))
    from domain.uncertainty.engine_internal_sr import _resolve_reprod_reference_mean

    reference_mean = _resolve_reprod_reference_mean(reprod_result)
    if _active("u_std_repeatability") and (
        not np.isfinite(reference_mean) or reference_mean == 0.0
    ):
        return required_contributor_unavailable_budget(
            engine="pb_tl_external_normalization",
            output_mode=uncertainty_config.output_mode,
            contributor_name="u_std_repeatability",
            reason="No finite, nonzero included-standard mean for relative repeatability.",
            basis_ratio_value=ratio_mean,
            n_cycles=n_cycles,
        )
    # Evaluate relative repeatability on the reference-material basis, then
    # transfer that fraction to the reported sample ratio exactly once.
    u_reprod_rel_permil = (
        u_reprod_abs / abs(reference_mean) * 1000.0
        if reference_mean and u_reprod_abs > 0.0 else 0.0
    )
    u_reprod_abs = u_reprod_rel_permil / 1000.0 * abs(ratio_mean)
    reprod_dof = reprod_result.degrees_of_freedom
    method_labels = {
        "sd_of_means": "SD of standard means",
        "loo_cross_validation": "LOO cross-validation residuals",
        "drift_residuals": "Drift model residuals",
        "robust_mad": "Robust MAD estimator",
    }
    reprod_label = method_labels.get(reprod_result.method, reprod_result.method)
    contributors.append(
        UncertaintyContributor(
            name="u_std_repeatability",
            display_name=f"Reference-material repeatability (Type A, {'SE' if repeatability_se else 'SD'}, {reprod_label})",
            value_abs=u_reprod_abs,
            value_rel_permil=u_reprod_rel_permil,
            type_ab="A",
            degrees_of_freedom=float(reprod_dof) if reprod_dof >= 1 else 1.0,
            percentage_contribution=0.0,
            description=(
                f"Scatter of Tl-normalized Pb standard means via {reprod_label}. "
                + (f"SE of session means: scatter divided by sqrt({max(n_standards, 1)})."
                   if repeatability_se else "Full SD of session means.")
                + f" Relative uncertainty uses the included-standard mean {reference_mean:.10g}; "
                "transferred fractionally to the reported sample ratio. Reference: JCGM 100:2008, section 5.1."
            ),
            **_contributor_gate("u_std_repeatability", u_reprod_abs > 0),
        )
    )

    drift_requested = (
        uncertainty_config.include_kappa_drift
        and _active("u_kappa_drift")
        and reprod_result.method != "drift_residuals"
    )
    drift_has_pairs = (
        reprod_result.drift_deltas is not None
        and len(reprod_result.drift_deltas) > 0
    )
    if drift_requested and not drift_has_pairs:
        budget = required_contributor_unavailable_budget(
            engine="pb_tl_external_normalization", output_mode=uncertainty_config.output_mode,
            contributor_name="u_kappa_drift", reason="Instrumental drift requires at least one eligible consecutive standard pair within a segment.",
            basis_ratio_value=ratio_mean, ratio_value=ratio_mean, n_cycles=n_cycles,
        )
        budget.reprod_result = reprod_result
        return budget

    if drift_requested and drift_has_pairs:
        kd_abs = (reprod_result.kappa_drift_permil / 1000.0) * abs(ratio_mean)
        contributors.append(
            UncertaintyContributor(
                name="u_kappa_drift",
                display_name="Instrumental drift (Type B)",
                value_abs=kd_abs,
                value_rel_permil=reprod_result.kappa_drift_permil,
                type_ab="B",
                degrees_of_freedom=float("inf"),
                percentage_contribution=0.0,
                description=(
                    "Instrumental drift: half the mean absolute eligible consecutive "
                    "relative standard change is adopted as a Type B standard uncertainty "
                    "with infinite degrees of freedom. The configured Monte Carlo PDF "
                    "has this standard deviation (rectangular half-width = sqrt(3) * u). "
                    "Reference: TraceISO owner-selected drift convention, 2026-09-20; "
                    "this assumption does not establish laboratory qualification."
                ),
                **_contributor_gate("u_kappa_drift"),
            )
        )

    # Engine C contributors below replay the Pb-Tl chain, which stops at the
    # Tl-normalized ratio. A committed drift correction then multiplies that
    # ratio by one recorded scalar; d(aR)/dx = a*dR/dx, so each replayed
    # sensitivity carries that factor exactly once.
    replay_output_scale = resolve_output_scale(
        sample, ratio_name, input_layer=INPUT_LAYER_NORMALIZED,
    )

    # 3. u_blank
    try:
        blank_result = _compute_blank_contribution(
            sample=sample,
            ratio_name=ratio_name,
            all_samples=all_samples,
            element_config=element_config,
            uncertainty_config=uncertainty_config,
            ratio_mean=ratio_mean,
            processing_config=processing_config,
            cycle_ranges=cycle_ranges,
        )
    except ValueError as exc:
        if _active("u_blank"):
            return required_contributor_unavailable_budget(engine="pb_tl_external_normalization", output_mode="absolute_ratio",
                contributor_name="u_blank", reason=str(exc), basis_ratio_value=ratio_mean, ratio_value=ratio_mean, n_cycles=n_cycles)
        blank_result = BlankUncertaintyResult(n_blanks_used=0)


    u_blank_abs = scaled_contribution(blank_result.u_blank_abs, replay_output_scale)

    u_blank_missing_reason = ""
    if _active("u_blank") and blank_result.n_blank_cycles < 2:
        ratio_def = element_config.default_ratios.get(ratio_name)
        if ratio_def is None:
            u_blank_missing_reason = f"u_blank ratio {ratio_name} not defined in element configuration."
        else:
            num_isotope, den_isotope = ratio_def
            blank_samples = resolve_blank_samples(sample, all_samples)
            if not blank_samples:
                u_blank_missing_reason = "u_blank missing blank samples in session."
            else:
                u_blank_missing_reason = f"u_blank missing blank voltage data for {num_isotope} or {den_isotope}."

    contributors.append(
        UncertaintyContributor(
            name="u_blank",
            display_name="Blank correction (Type A)",
            value_abs=u_blank_abs,
            value_rel_permil=(
                (u_blank_abs / abs(ratio_mean)) * 1000.0
                if ratio_mean and u_blank_abs > 0
                else 0.0
            ),
            type_ab="A",
            degrees_of_freedom=blank_result.degrees_of_freedom,
            percentage_contribution=0.0,
            description=(
                "Replay-based perturbation of blank-sensitive isotopes through the "
                f"Pb-Tl correction chain ({replay_output_scale.describe()}). "
                + describe_blank_input_model(
                    blank_result.blank_uncertainty_input,
                    blank_result.n_blank_cycles,
                )
            ),
            **_contributor_gate("u_blank", blank_result.n_blank_cycles >= 2, u_blank_missing_reason),
        )
    )

    # 4. u_norm_ref
    try:
        u_norm_ref_abs, u_norm_ref_rel_permil = _compute_normalization_reference_uncertainty(
            sample=sample,
            ratio_name=ratio_name,
            ratio_mean=ratio_mean,
            uncertainty_config=uncertainty_config,
            processing_config=processing_config,
            element_config=element_config,
            cycle_ranges=cycle_ranges,
            ratio_mask=ratio_mask,
        )
    except ValueError as exc:
        if _active("u_norm_ref"):
            return required_contributor_unavailable_budget(engine="pb_tl_external_normalization", output_mode="absolute_ratio",
                contributor_name="u_norm_ref", reason=str(exc), basis_ratio_value=ratio_mean, ratio_value=ratio_mean, n_cycles=n_cycles)
        u_norm_ref_abs, u_norm_ref_rel_permil = 0.0, 0.0


    u_norm_ref_abs = scaled_contribution(u_norm_ref_abs, replay_output_scale)
    u_norm_ref_rel_permil = (
        (u_norm_ref_abs / abs(ratio_mean)) * 1000.0
        if ratio_mean and u_norm_ref_abs > 0 else 0.0
    )

    u_norm_ref_missing_reason = ""
    if _active("u_norm_ref") and u_norm_ref_abs <= 0.0:
        try:
            reference_inputs = _resolve_pb_tl_reference_inputs(
                ratio_name,
                processing_config=processing_config,
                element_config=element_config,
            )
            tl_std = resolve_pb_tl_normalization_standard_uncertainty(
                uncertainty_config,
                reference_inputs.tl_norm_ratio_name,
            )
            if tl_std <= 0.0:
                u_norm_ref_missing_reason = f"u_norm_ref Tl reference standard uncertainty is zero or negative for ratio {reference_inputs.tl_norm_ratio_name}."
        except ValueError as e:
            u_norm_ref_missing_reason = f"u_norm_ref Tl reference configuration error: {str(e)}."

    contributors.append(
        UncertaintyContributor(
            name="u_norm_ref",
            display_name="Tl normalization ratio uncertainty (Type B)",
            value_abs=u_norm_ref_abs,
            value_rel_permil=u_norm_ref_rel_permil,
            type_ab="B",
            degrees_of_freedom=float("inf"),
            percentage_contribution=0.0,
            description=(
                "Replay-based propagation of the accepted Tl normalization-ratio "
                f"uncertainty ({replay_output_scale.describe()})."
            ),
            **_contributor_gate("u_norm_ref", u_norm_ref_abs > 0, u_norm_ref_missing_reason),
        )
    )

    # 5. u_interf
    u_interf_abs, u_interf_rel_permil = _compute_hg_interference_uncertainty(
        sample=sample,
        ratio_name=ratio_name,
        ratio_mean=ratio_mean,
        processing_config=processing_config,
        element_config=element_config,
        cycle_ranges=cycle_ranges,
        ratio_mask=ratio_mask,
    )

    u_interf_abs = scaled_contribution(u_interf_abs, replay_output_scale)
    u_interf_rel_permil = (
        (u_interf_abs / abs(ratio_mean)) * 1000.0
        if ratio_mean and u_interf_abs > 0 else 0.0
    )

    u_interf_missing_reason = ""
    if _active("u_interf") and u_interf_abs <= 0.0:
        if not getattr(processing_config, "apply_hg_interference_correction", False):
            u_interf_missing_reason = "u_interf Hg interference correction is disabled in processing configuration."
        elif "204Pb" not in ratio_name:
            u_interf_missing_reason = f"u_interf not applicable for ratio {ratio_name} (only applicable to 204Pb ratios)."
        else:
            try:
                reference_inputs = _resolve_pb_tl_reference_inputs(
                    ratio_name,
                    processing_config=processing_config,
                    element_config=element_config,
                )
                if reference_inputs.hg204_hg202_natural is None:
                    u_interf_missing_reason = "u_interf 204Hg/202Hg assigned correction ratio not defined in configuration."
                elif _hg_uncertainty_is_unassigned():
                    # Not zero. The correction ratio is applied; its
                    # uncertainty is unknown, so it is omitted from the budget
                    # rather than propagated as an exact quantity.
                    u_interf_missing_reason = (
                        "u_interf omitted: the 204Hg/202Hg assigned correction "
                        "ratio has an unassigned uncertainty. It is not treated "
                        "as zero, and no 204Hg interference uncertainty is "
                        "included in this budget."
                    )
                else:
                    base_intensities = _get_pb_tl_base_intensities(sample)
                    missing_iso = [iso for iso in ["202Hg", "204Pb"] if iso not in base_intensities]
                    if missing_iso:
                        u_interf_missing_reason = f"u_interf missing intensity data for {', '.join(missing_iso)}."
                    else:
                        u_interf_missing_reason = "u_interf Hg interference uncertainty is zero or could not be evaluated."
            except ValueError as e:
                u_interf_missing_reason = f"u_interf reference configuration error: {str(e)}."

    contributors.append(
        UncertaintyContributor(
            name="u_interf",
            display_name="204Hg interference correction (Type B)",
            value_abs=u_interf_abs,
            value_rel_permil=u_interf_rel_permil,
            type_ab="B",
            degrees_of_freedom=float("inf"),
            percentage_contribution=0.0,
            description=(
                "Replay-based propagation of the 204Hg/202Hg assigned "
                "correction-ratio uncertainty through the Pb-Tl correction "
                f"chain ({replay_output_scale.describe()})."
            ),
            **_contributor_gate("u_interf", u_interf_abs > 0, u_interf_missing_reason),
        )
    )

    # 6. u_CRM
    u_crm_abs, u_crm_rel_permil = _compute_crm(
        certified_value=certified_value,
        uncertainty_config=uncertainty_config,
        ratio_mean=ratio_mean,
    )
    contributors.append(
        UncertaintyContributor(
            name="u_crm",
            display_name="CRM certified value (Type B)",
            value_abs=u_crm_abs,
            value_rel_permil=u_crm_rel_permil,
            type_ab="B",
            degrees_of_freedom=float("inf"),
            percentage_contribution=0.0,
            description=(
                "Certified reference material expanded uncertainty divided by k."
            ),
            **_contributor_gate(
                "u_crm",
                u_crm_abs > 0,
                not_applicable_reason=not_applicable_reason(
                    "u_crm", output_mode=uncertainty_config.output_mode,
                ),
            ),
        )
    )

    from domain.uncertainty.sr_sample_values import build_processed_material_contributors

    contributors.extend(build_processed_material_contributors(
        sample, uncertainty_config, ratio_mean, _contributor_gate,
    ))

    # Inject custom contributor rows (element-filtered inside helper).
    custom_rows = build_custom_contributor_rows(
        sample=sample,
        element_symbol=element_config.symbol,
        ratio_mean=ratio_mean,
        custom_contributor_library=custom_contributor_library or {},
    )
    contributors.extend(custom_rows)

    budget = _combine_and_build_budget(
        contributors=contributors,
        ratio_mean=ratio_mean,
        n_cycles=n_cycles,
        uncertainty_config=uncertainty_config,
    )
    budget.reprod_result = reprod_result
    return budget


def _resolve_active_pb_tl_normalization_ratio_name(
    processing_config: ProcessingConfig,
    element_config: ElementConfig,
) -> str:
    ratio_name = processing_config.normalization_ratio_override or element_config.normalization_ratio
    ratio_name = ratio_name or "205Tl/203Tl"
    normalized = normalize_ratio_name(ratio_name)
    if not normalized or normalized.count("/") != 1:
        raise ValueError("A valid Pb Tl normalization ratio is required.")
    return normalized


def _resolve_active_pb_tl_normalization_value(
    processing_config: ProcessingConfig,
    element_config: ElementConfig,
) -> float:
    if processing_config.normalization_value_override is not None:
        return float(processing_config.normalization_value_override)

    if element_config.normalization_value is not None and element_config.normalization_value > 0:
        return float(element_config.normalization_value)

    ratio_name = _resolve_active_pb_tl_normalization_ratio_name(
        processing_config,
        element_config,
    )
    managed = get_internal_normalization("Tl")
    if managed is not None:
        managed_ratio_name, managed_value = managed
        if normalize_ratio_name(managed_ratio_name) == ratio_name:
            return float(managed_value)

    return float(require_natural_ratio("Tl", ratio_name)[0])


def resolve_pb_tl_normalization_standard_uncertainty(
    uncertainty_config: UncertaintyConfig,
    tl_norm_ratio_name: str,
) -> float:
    """Return the 1-sigma uncertainty of the active Tl normalization ratio.

    A positive session value overrides the managed reference-data uncertainty.
    The UI stores this as an absolute standard uncertainty, not expanded U.
    """
    override = float(getattr(uncertainty_config, "pb_tl_norm_ratio_u_abs", 0.0) or 0.0)
    if np.isfinite(override) and override > 0.0:
        if not all(str(getattr(uncertainty_config, field, "") or "").strip() for field in (
            "pb_tl_norm_ratio_u_justification", "pb_tl_norm_ratio_u_source",
        )):
            raise UnassignedUncertaintyError(
                "User-provided Tl standard uncertainty requires a justification and source for the audit trail."
            )
        return override

    record = get_natural_ratio_record("Tl", tl_norm_ratio_name)
    if record is None:
        raise ValueError(
            f"Managed Tl normalization ratio '{tl_norm_ratio_name}' is missing."
        )
    standard = standard_uncertainty_from_values(
        record.uncertainty,
        record.k,
        record.uncertainty_semantics,
    )
    if standard is None:
        raise UnassignedUncertaintyError(
            f"Managed Tl normalization ratio '{tl_norm_ratio_name}' has no "
            "convertible standard uncertainty. Standard uncertainty unavailable from source metadata. "
            "User-provided value required, with justification and source, to include u_norm_ref."
        )
    return float(standard)


def _resolve_pb_tl_reference_inputs(
    ratio_name: str,
    *,
    processing_config: ProcessingConfig,
    element_config: ElementConfig,
) -> PbTlReferenceInputs:
    """Resolve all managed masses and natural ratios for one Pb measurand."""
    ratio_def = element_config.default_ratios.get(ratio_name)
    if ratio_def is None:
        raise ValueError(f"Unsupported Pb ratio '{ratio_name}'.")

    target_numerator = normalize_ratio_token(ratio_def[0])
    target_denominator = normalize_ratio_token(ratio_def[1])
    tl_norm_ratio_name = _resolve_active_pb_tl_normalization_ratio_name(
        processing_config,
        element_config,
    )
    normalization_numerator, normalization_denominator = [
        normalize_ratio_token(token)
        for token in tl_norm_ratio_name.split("/", 1)
    ]
    tl_norm_value = _resolve_active_pb_tl_normalization_value(
        processing_config,
        element_config,
    )

    hg204_hg202_natural: Optional[float] = None
    m202: Optional[float] = None
    m204_hg: Optional[float] = None
    if getattr(processing_config, "apply_hg_interference_correction", False):
        try:
            hg204_hg202_natural = float(require_natural_ratio("Hg", "204Hg/202Hg")[0])
            m202 = float(require_isotope_mass("202Hg"))
            m204_hg = float(require_isotope_mass("204Hg"))
        except ValueError:
            hg204_hg202_natural = None
            m202 = None
            m204_hg = None

    return PbTlReferenceInputs(
        ratio_name=ratio_name,
        tl_norm_ratio_name=tl_norm_ratio_name,
        tl_norm_value=tl_norm_value,
        normalization_numerator=normalization_numerator,
        normalization_denominator=normalization_denominator,
        target_numerator=target_numerator,
        target_denominator=target_denominator,
        m_norm_num=float(require_isotope_mass(normalization_numerator)),
        m_norm_den=float(require_isotope_mass(normalization_denominator)),
        target_num_mass=float(require_isotope_mass(target_numerator)),
        target_den_mass=float(require_isotope_mass(target_denominator)),
        hg204_hg202_natural=hg204_hg202_natural,
        m202=m202,
        m204_hg=m204_hg,
    )


def _get_pb_tl_base_intensities(sample: Sample) -> Dict[str, np.ndarray]:
    """Return the replay source intensities for Pb-Tl calculations.

    Replay starts from the blank-corrected snapshot when available so the Hg
    correction is not double-applied to already corrected 204Pb data.
    """
    src = (
        sample.blank_corrected_intensities
        if sample.blank_corrected_intensities
        else sample.corrected_intensities
        if sample.corrected_intensities
        else sample.intensities
    )
    return {
        isotope: np.asarray(cycle_data.values, dtype=np.float64).copy()
        for isotope, cycle_data in src.items()
    }


def _resolve_pb_tl_ratio_mask(
    sample: Sample,
    ratio_name: str,
    *,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[np.ndarray]:
    """Return the runtime-aware mask for the requested Pb ratio."""
    ratio_cd = get_best_ratio_data(sample, ratio_name)
    if ratio_cd is None:
        return None

    mask = np.asarray(ratio_cd.mask, dtype=bool).copy()
    if not cycle_ranges:
        return mask

    cycle_range = resolve_cycle_range(
        cycle_ranges,
        sample_name=sample.name,
        sample_key=sample_cycle_key(sample),
    )
    if cycle_range is None:
        return mask

    start_idx = max(int(cycle_range[0]) - 1, 0)
    end_idx = min(int(cycle_range[1]), len(mask))
    range_mask = np.zeros(len(mask), dtype=bool)
    if end_idx > start_idx:
        range_mask[start_idx:end_idx] = True
    return mask & range_mask


def _run_pb_tl_correction_chain(
    intensities: Dict[str, np.ndarray],
    normalization_value: float,
    ratio_name: str,
    mask: Optional[np.ndarray],
    *,
    reference_inputs: PbTlReferenceInputs,
    apply_interference: bool = True,
    apply_iif: bool = True,
) -> float:
    """Replay the deterministic Pb-Tl correction chain for one ratio mean."""
    if normalization_value <= 0.0:
        return float("nan")

    corrected = {
        isotope: np.asarray(values, dtype=np.float64).copy()
        for isotope, values in intensities.items()
    }

    required = {
        reference_inputs.normalization_numerator,
        reference_inputs.normalization_denominator,
        reference_inputs.target_numerator,
        reference_inputs.target_denominator,
    }
    if not required.issubset(corrected):
        return float("nan")

    shapes = {corrected[key].shape for key in required}
    if len(shapes) != 1 or any(len(shape) != 1 for shape in shapes):
        return float("nan")
    if mask is not None and np.asarray(mask).shape != corrected[reference_inputs.target_numerator].shape:
        return float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        measured_tl = (
            corrected[reference_inputs.normalization_numerator]
            / corrected[reference_inputs.normalization_denominator]
        )

    if apply_interference and "204Pb" in (
        reference_inputs.target_numerator,
        reference_inputs.target_denominator,
    ):
        if (
            reference_inputs.hg204_hg202_natural is None
            or reference_inputs.m202 is None
            or reference_inputs.m204_hg is None
            or "202Hg" not in corrected
            or "204Pb" not in corrected
        ):
            return float("nan")

        f_tl = calculate_f_factor(
            normalization_ratio_measured=measured_tl,
            normalization_ratio_reference=normalization_value,
            normalization_numerator_mass=reference_inputs.m_norm_num,
            normalization_denominator_mass=reference_inputs.m_norm_den,
        )
        corrected_204pb, _interference = hg204_interference_correction(
            pb204=corrected["204Pb"],
            hg202=corrected["202Hg"],
            f_tl=f_tl,
            hg204_hg202_natural=reference_inputs.hg204_hg202_natural,
            m202=reference_inputs.m202,
            m204_hg=reference_inputs.m204_hg,
        )
        corrected["204Pb"] = corrected_204pb

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_values = (
            corrected[reference_inputs.target_numerator]
            / corrected[reference_inputs.target_denominator]
        )

    if apply_iif:
        mb = calculate_k_factors(
            normalization_ratio_measured=measured_tl,
            normalization_ratio_reference=normalization_value,
            normalization_numerator_mass=reference_inputs.m_norm_num,
            target_numerator_mass=reference_inputs.target_num_mass,
            normalization_denominator_mass=reference_inputs.m_norm_den,
            target_denominator_mass=reference_inputs.target_den_mass,
        )
        if mb.n_valid_cycles == 0:
            return float("nan")
        n = min(len(ratio_values), len(mb.target_k))
        if n <= 0:
            return float("nan")
        ratio_values = apply_iif_correction(ratio_values[:n], mb.target_k[:n])
        active_mask = (
            np.asarray(mask[:n], dtype=bool).copy()
            if mask is not None
            else np.ones(n, dtype=bool)
        )
    else:
        active_mask = (
            np.asarray(mask, dtype=bool).copy()
            if mask is not None
            else np.ones(len(ratio_values), dtype=bool)
        )

    n_active = min(len(ratio_values), len(active_mask))
    if n_active <= 0:
        return float("nan")

    ratio_values = np.asarray(ratio_values[:n_active], dtype=np.float64)
    if len(active_mask) != len(ratio_values) or not np.all(np.isfinite(ratio_values[active_mask])):
        return float("nan")
    active_mask = active_mask.copy()
    if not np.any(active_mask):
        return float("nan")

    return float(np.mean(ratio_values[active_mask]))


def _resolve_blank_perturbation_isotopes(
    base_intensities: Dict[str, np.ndarray],
    ratio_name: str,
    *,
    apply_hg: bool,
) -> Tuple[str, ...]:
    """Return the isotopes whose blank shifts can affect the Pb measurand."""
    ratio_num, ratio_den = [normalize_ratio_token(token) for token in ratio_name.split("/", 1)]
    isotopes = [ratio_num, ratio_den]
    if apply_hg and "204Pb" in isotopes:
        isotopes.append("202Hg")
    isotopes.extend(token for token in ("203Tl", "205Tl") if token in base_intensities)
    return tuple(dict.fromkeys(isotopes))


def _build_blank_matrix(
    blank: Sample,
    isotopes: Sequence[str],
    *,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[np.ndarray]:
    """Return cycle-paired blank voltages for an arbitrary isotope set."""
    src = blank.corrected_intensities if blank.corrected_intensities else blank.intensities
    cycle_data = [src.get(isotope) for isotope in isotopes]
    if any(cd is None for cd in cycle_data):
        return None

    assert all(cd is not None for cd in cycle_data)
    lengths = {len(cd.values) for cd in cycle_data if cd is not None}
    if len(lengths) != 1:
        raise ValueError("Blank channels have incompatible lengths.")
    n_cycles = lengths.pop()
    if n_cycles < 2:
        return None

    mask = np.ones(n_cycles, dtype=bool)
    for cd in cycle_data:
        assert cd is not None
        mask &= np.asarray(cd.mask[:n_cycles], dtype=bool)
        mask &= np.isfinite(np.asarray(cd.values[:n_cycles], dtype=np.float64))

    if cycle_ranges:
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

    if int(np.sum(mask)) < 2:
        return None

    columns = [
        np.asarray(cd.values[:n_cycles], dtype=np.float64)[mask]
        for cd in cycle_data
        if cd is not None
    ]
    return np.column_stack(columns)


def build_pb_tl_blank_blocks(
    sample: Sample,
    all_samples: List[Sample],
    *,
    isotopes: Sequence[str],
    uncertainty_config: UncertaintyConfig,
    processing_config: ProcessingConfig,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    paired_blank_filter: bool = False,
) -> Tuple[PbTlBlankBlock, ...]:
    """Build one or two covariance-aware blank blocks for replay propagation."""
    blank_samples, _selection = resolve_blank_samples_for_uncertainty(sample, all_samples)
    if blank_samples is None:
        return ()
    if not blank_samples:
        blank_samples = [
            blank_sample
            for blank_sample in all_samples
            if blank_sample.is_blank and not blank_sample.metadata.get("excluded", False)
        ]
    if not blank_samples:
        return ()

    blank_mode = resolve_blank_correction_mode(sample, processing_config.blank_mode)
    if blank_mode == "none":
        return ()
    if blank_mode == "before_and_after" and len(blank_samples) >= 2:
        selected_blanks = [blank_samples[0], blank_samples[-1]]
    else:
        selected_blanks = [blank_samples[0]]

    from domain.uncertainty.numerical_domain import validate_covariance_matrix

    blocks: List[PbTlBlankBlock] = []
    method = getattr(uncertainty_config, "blank_correlation_method", "pearson_from_data")
    fixed_r = float(getattr(uncertainty_config, "blank_fixed_r", 0.0))
    blank_uncertainty_input = getattr(uncertainty_config, "blank_uncertainty_input", "sd")
    from domain.uncertainty.blank import resolve_blank_channel_weights
    weights = resolve_blank_channel_weights(sample, isotopes, blank_mode=blank_mode)
    grouped = {}
    selected_roles = [("before" if i == 0 else "after", blank) for i, blank in enumerate(selected_blanks)]
    if any(sample.used_blank_ids.get(role) for role in ("before", "after")):
        selected_roles = []
        for role in (("before", "after") if blank_mode == "before_and_after" else ("before",)):
            identity = sample.used_blank_ids.get(role)
            if identity:
                matches = [b for b in blank_samples if b.observation_id == identity]
                if not matches:
                    raise ValueError(f"Unresolved required blank observation {identity!r}.")
                selected_roles.append((role, matches[0]))
    for role, blank in selected_roles:
        vector = np.array([weights[role].get(iso, 0.0) if weights else 1 / len(selected_roles) for iso in isotopes])
        if blank.observation_id in grouped:
            grouped[blank.observation_id][1][:] += vector
        else:
            grouped[blank.observation_id] = (blank, vector)
    for blank, vector in grouped.values():
        block_isotopes = tuple(iso for iso, weight in zip(isotopes, vector) if weight != 0)
        block_weights = tuple(float(weight) for weight in vector if weight != 0)
        if not block_isotopes:
            continue
        if paired_blank_filter:
            from domain.uncertainty.blank import _get_paired_blank_matrix
            paired = _get_paired_blank_matrix(blank, block_isotopes, cycle_ranges=cycle_ranges)
            matrix = None if paired is None else paired.T
        else:
            matrix = _build_blank_matrix(blank, block_isotopes, cycle_ranges=cycle_ranges)
        if matrix is None:
            raise ValueError(f"Blank {blank.name!r}: insufficient or incompatible required channel support.")

        means = np.asarray(np.mean(matrix, axis=0), dtype=np.float64)
        sds = np.asarray(np.std(matrix, axis=0, ddof=1), dtype=np.float64)
        n_pairs = int(matrix.shape[0])
        input_sds = blank_input_sigmas(sds, n_pairs, blank_uncertainty_input)
        dim = int(matrix.shape[1])
        if dim == 1:
            covariance_matrix = np.array([[float(input_sds[0] ** 2)]], dtype=np.float64)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                empirical_corr = np.corrcoef(matrix, rowvar=False)
            empirical_corr = np.nan_to_num(empirical_corr, nan=0.0)
            empirical_corr = np.clip(empirical_corr, -1.0, 1.0)

            if method == "uncorrelated":
                applied_corr = np.eye(dim, dtype=np.float64)
            elif method == "fixed_value":
                if not np.isfinite(fixed_r) or not -1 / (dim - 1) <= fixed_r <= 1:
                    raise ValueError(f"Fixed correlation is not PSD for {dim} required blank channels.")
                applied_corr = np.full((dim, dim), fixed_r, dtype=np.float64)
                np.fill_diagonal(applied_corr, 1.0)
            else:
                if paired_blank_filter:
                    from domain.uncertainty.blank import _compute_empirical_correlation_matrix, MIN_BLANK_CYCLES_FOR_CORRELATION
                    applied_corr = _compute_empirical_correlation_matrix(matrix.T) if n_pairs >= MIN_BLANK_CYCLES_FOR_CORRELATION else np.eye(dim)
                else:
                    applied_corr = empirical_corr

            covariance_matrix = np.outer(input_sds, input_sds) * applied_corr

        validate_covariance_matrix(means, covariance_matrix)
        blocks.append(
            PbTlBlankBlock(
                isotopes=block_isotopes,
                observation_id=blank.observation_id,
                channel_weights=block_weights,
                mean_vector=means,
                covariance_matrix=np.asarray(covariance_matrix, dtype=np.float64),
                n_pairs=n_pairs,
                degrees_of_freedom=float(max(n_pairs - 1, 1)),
            )
        )

    return tuple(blocks)


def _compute_normalization_reference_uncertainty(
    *,
    sample: Sample,
    ratio_name: str,
    ratio_mean: float,
    uncertainty_config: UncertaintyConfig,
    processing_config: ProcessingConfig,
    element_config: ElementConfig,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    ratio_mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Propagate the accepted Tl normalization value through the replay chain."""
    try:
        reference_inputs = _resolve_pb_tl_reference_inputs(
            ratio_name,
            processing_config=processing_config,
            element_config=element_config,
        )
        tl_std = resolve_pb_tl_normalization_standard_uncertainty(
            uncertainty_config,
            reference_inputs.tl_norm_ratio_name,
        )
    except ValueError as exc:
        raise ValueError("Tl normalization reference uncertainty unavailable.") from exc

    if tl_std <= 0.0:
        return 0.0, 0.0

    base_intensities = _get_pb_tl_base_intensities(sample)
    if ratio_mask is None:
        ratio_mask = _resolve_pb_tl_ratio_mask(
            sample,
            ratio_name,
            cycle_ranges=cycle_ranges,
        )

    norm_plus = reference_inputs.tl_norm_value + tl_std
    norm_minus = reference_inputs.tl_norm_value - tl_std
    if norm_minus <= 0.0:
        raise ValueError("Tl reference perturbation leaves the positive domain.")

    up = _run_pb_tl_correction_chain(
        base_intensities,
        norm_plus,
        ratio_name,
        ratio_mask,
        reference_inputs=reference_inputs,
        apply_interference=getattr(processing_config, "apply_hg_interference_correction", False),
        apply_iif=processing_config.apply_mass_bias_correction,
    )
    down = _run_pb_tl_correction_chain(
        base_intensities,
        norm_minus,
        ratio_name,
        ratio_mask,
        reference_inputs=reference_inputs,
        apply_interference=getattr(processing_config, "apply_hg_interference_correction", False),
        apply_iif=processing_config.apply_mass_bias_correction,
    )
    if not np.isfinite(up) or not np.isfinite(down):
        raise ValueError("Pb-Tl reference perturbation failed on selected support.")

    u_abs = abs(up - down) / 2.0
    u_rel_permil = (u_abs / abs(ratio_mean)) * 1000.0 if ratio_mean else 0.0
    return float(u_abs), float(u_rel_permil)


def _hg_uncertainty_is_unassigned() -> bool:
    """True when the 204Hg/202Hg record exists but carries no uncertainty."""
    record = get_natural_ratio_record("Hg", "204Hg/202Hg")
    return record is not None and record.is_uncertainty_unassigned


def _compute_hg_interference_uncertainty(
    *,
    sample: Sample,
    ratio_name: str,
    ratio_mean: float,
    processing_config: ProcessingConfig,
    element_config: ElementConfig,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    ratio_mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Propagate the 204Hg/202Hg correction-ratio uncertainty through the chain.

    Returns ``(0.0, 0.0)`` when that uncertainty is unassigned. The caller
    reports the omission through the contributor's inactive reason; a zero
    magnitude here is the absence of a term, not a claim that the ratio is
    exact.
    """
    if not getattr(processing_config, "apply_hg_interference_correction", False):
        return 0.0, 0.0
    if "204Pb" not in ratio_name:
        return 0.0, 0.0

    try:
        reference_inputs = _resolve_pb_tl_reference_inputs(
            ratio_name,
            processing_config=processing_config,
            element_config=element_config,
        )
        if reference_inputs.hg204_hg202_natural is None:
            return 0.0, 0.0
        hg_std = (
            reference_inputs.hg204_hg202_natural
            * require_natural_ratio_relative_uncertainty("Hg", "204Hg/202Hg")
        )
    except UnassignedUncertaintyError:
        # Unknown, so it contributes nothing and is reported as omitted.
        return 0.0, 0.0
    except ValueError:
        return 0.0, 0.0

    if hg_std <= 0.0:
        return 0.0, 0.0

    base_intensities = _get_pb_tl_base_intensities(sample)
    if "202Hg" not in base_intensities or "204Pb" not in base_intensities:
        return 0.0, 0.0
    if ratio_mask is None:
        ratio_mask = _resolve_pb_tl_ratio_mask(
            sample,
            ratio_name,
            cycle_ranges=cycle_ranges,
        )

    hg_plus = replace(
        reference_inputs,
        hg204_hg202_natural=reference_inputs.hg204_hg202_natural + hg_std,
    )
    hg_minus_value = reference_inputs.hg204_hg202_natural - hg_std
    if hg_minus_value <= 0.0:
        return 0.0, 0.0
    hg_minus = replace(
        reference_inputs,
        hg204_hg202_natural=hg_minus_value,
    )

    up = _run_pb_tl_correction_chain(
        base_intensities,
        reference_inputs.tl_norm_value,
        ratio_name,
        ratio_mask,
        reference_inputs=hg_plus,
        apply_interference=True,
        apply_iif=processing_config.apply_mass_bias_correction,
    )
    down = _run_pb_tl_correction_chain(
        base_intensities,
        reference_inputs.tl_norm_value,
        ratio_name,
        ratio_mask,
        reference_inputs=hg_minus,
        apply_interference=True,
        apply_iif=processing_config.apply_mass_bias_correction,
    )
    if not np.isfinite(up) or not np.isfinite(down):
        raise ValueError("Pb-Tl reference perturbation failed on selected support.")

    u_abs = abs(up - down) / 2.0
    u_rel_permil = (u_abs / abs(ratio_mean)) * 1000.0 if ratio_mean else 0.0
    return float(u_abs), float(u_rel_permil)


def _apply_uniform_shift(
    intensities: Dict[str, np.ndarray],
    isotope: str,
    shift: float,
) -> Dict[str, np.ndarray]:
    shifted = {
        name: values.copy()
        for name, values in intensities.items()
    }
    if isotope in shifted:
        shifted[isotope] = shifted[isotope] + shift
    return shifted


def _compute_blank_contribution(
    *,
    sample: Sample,
    ratio_name: str,
    all_samples: List[Sample],
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    ratio_mean: float,
    processing_config: ProcessingConfig,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> BlankUncertaintyResult:
    """Compute replay-based blank uncertainty for a Pb-Tl ratio."""
    base_intensities = _get_pb_tl_base_intensities(sample)
    apply_hg = bool(getattr(processing_config, "apply_hg_interference_correction", False))
    isotopes = _resolve_blank_perturbation_isotopes(
        base_intensities,
        ratio_name,
        apply_hg=apply_hg,
    )
    if not isotopes:
        return BlankUncertaintyResult()

    # The configured input model must travel with the result, not just with the
    # arithmetic. ``build_pb_tl_blank_blocks`` already scales the block sigmas
    # by it, but the u_blank contributor describes itself from these fields; if
    # they keep their defaults the budget states "cycle SD" while carrying
    # SD/sqrt(n), which is the S-2 defect.
    blank_input_mode = normalize_blank_uncertainty_input(
        getattr(uncertainty_config, "blank_uncertainty_input", "sd")
    )

    blank_mode = resolve_blank_correction_mode(sample, processing_config.blank_mode)
    blocks = build_pb_tl_blank_blocks(
        sample,
        all_samples,
        isotopes=isotopes,
        uncertainty_config=uncertainty_config,
        processing_config=processing_config,
        cycle_ranges=cycle_ranges,
    )
    if not blocks:
        return BlankUncertaintyResult(
            blank_mode=blank_mode,
            n_blanks_used=0,
            blank_uncertainty_input=blank_input_mode,
        )

    try:
        reference_inputs = _resolve_pb_tl_reference_inputs(
            ratio_name,
            processing_config=processing_config,
            element_config=element_config,
        )
    except ValueError as exc:
        raise ValueError("Pb-Tl blank reference inputs unavailable.") from exc

    ratio_mask = _resolve_pb_tl_ratio_mask(
        sample,
        ratio_name,
        cycle_ranges=cycle_ranges,
    )

    per_block_components: List[Tuple[float, float]] = []
    per_block_uncorrelated_components: List[float] = []
    per_block_results: List[BlankUncertaintyResult] = []
    n_blocks = float(len(blocks))
    for block in blocks:
        sensitivities: List[float] = []
        for index, isotope in enumerate(block.isotopes):
            sigma = float(np.sqrt(max(block.covariance_matrix[index, index], 0.0)))
            if sigma <= 0.0 or isotope not in base_intensities:
                sensitivities.append(0.0)
                continue

            step = max(sigma * 1e-3, np.finfo(float).eps ** (1/3) * max(abs(float(np.mean(base_intensities[isotope]))), sigma))
            blank_plus = _apply_uniform_shift(
                base_intensities,
                isotope,
                -(step * (block.channel_weights[index] if block.channel_weights else 1 / n_blocks)),
            )
            blank_minus = _apply_uniform_shift(
                base_intensities,
                isotope,
                +(step * (block.channel_weights[index] if block.channel_weights else 1 / n_blocks)),
            )
            up = _run_pb_tl_correction_chain(
                blank_plus,
                reference_inputs.tl_norm_value,
                ratio_name,
                ratio_mask,
                reference_inputs=reference_inputs,
                apply_interference=apply_hg,
                apply_iif=processing_config.apply_mass_bias_correction,
            )
            down = _run_pb_tl_correction_chain(
                blank_minus,
                reference_inputs.tl_norm_value,
                ratio_name,
                ratio_mask,
                reference_inputs=reference_inputs,
                apply_interference=apply_hg,
                apply_iif=processing_config.apply_mass_bias_correction,
            )
            if not np.isfinite(up) or not np.isfinite(down):
                raise ValueError("Pb-Tl blank perturbation invalid on selected cycle support.")
            sensitivities.append((up - down) / (2.0 * step))

        jacobian = np.asarray(sensitivities, dtype=np.float64)
        variance = float(jacobian @ block.covariance_matrix @ jacobian.T)
        variance = max(variance, 0.0)
        block_sds = np.sqrt(np.clip(np.diag(block.covariance_matrix), 0.0, None))
        variance_uncorrelated = float(np.sum((jacobian * block_sds) ** 2))
        correlation_term_abs2 = variance - variance_uncorrelated
        u_block = float(np.sqrt(variance))
        u_block_uncorrelated = float(np.sqrt(max(variance_uncorrelated, 0.0)))
        per_block_components.append((u_block, block.degrees_of_freedom))
        per_block_uncorrelated_components.append(u_block_uncorrelated)
        per_block_results.append(
            BlankUncertaintyResult(
                u_blank_abs=u_block,
                u_blank_uncorrelated_abs=u_block_uncorrelated,
                u_blank_correlation_term_abs2=correlation_term_abs2,
                degrees_of_freedom=block.degrees_of_freedom,
                blank_mode=blank_mode,
                n_blanks_used=1,
                n_blank_cycles=block.n_pairs,
                blank_uncertainty_input=blank_input_mode,
                model_dimension=len(block.isotopes),
                correlation_labels=tuple(block.isotopes),
                correlation_matrix=block.covariance_matrix.copy(),
            )
        )

    u_blank_abs = float(np.sqrt(sum(u_i ** 2 for u_i, _nu_i in per_block_components)))
    u_blank_uncorrelated_abs = float(
        np.sqrt(sum(u_i ** 2 for u_i in per_block_uncorrelated_components))
    )
    u_blank_correlation_term_abs2 = u_blank_abs ** 2 - u_blank_uncorrelated_abs ** 2
    blank_dof = effective_dof(per_block_components)
    num_corrected_mean = _get_corrected_intensity_mean(
        sample=sample,
        isotope=isotopes[0],
        cycle_ranges=cycle_ranges,
    )
    den_corrected_mean = _get_corrected_intensity_mean(
        sample=sample,
        isotope=isotopes[1] if len(isotopes) > 1 else isotopes[0],
        cycle_ranges=cycle_ranges,
    )

    u_num_sd = float(np.mean([np.sqrt(max(block.covariance_matrix[0, 0], 0.0)) for block in blocks]))
    u_den_sd = float(
        np.mean(
            [
                np.sqrt(max(block.covariance_matrix[min(1, len(block.isotopes) - 1), min(1, len(block.isotopes) - 1)], 0.0))
                for block in blocks
            ]
        )
    )

    correlation = 0.0
    if len(isotopes) >= 2:
        corr_values = []
        for block in blocks:
            s0 = float(np.sqrt(max(block.covariance_matrix[0, 0], 0.0)))
            s1 = float(np.sqrt(max(block.covariance_matrix[1, 1], 0.0)))
            if s0 > 0 and s1 > 0:
                corr_values.append(float(block.covariance_matrix[0, 1] / (s0 * s1)))
        if corr_values:
            correlation = float(np.mean(corr_values))

    aux_isotope = isotopes[2] if len(isotopes) > 2 else ""
    aux_corrected_mean = (
        _get_corrected_intensity_mean(sample=sample, isotope=aux_isotope, cycle_ranges=cycle_ranges)
        if aux_isotope
        else 0.0
    )
    u_aux_sd = (
        float(np.mean([np.sqrt(max(block.covariance_matrix[2, 2], 0.0)) for block in blocks]))
        if aux_isotope
        else 0.0
    )

    return BlankUncertaintyResult(
        u_blank_abs=u_blank_abs,
        u_blank_uncorrelated_abs=u_blank_uncorrelated_abs,
        u_blank_correlation_term_abs2=u_blank_correlation_term_abs2,
        u_num_sd=u_num_sd,
        u_den_sd=u_den_sd,
        correlation=correlation,
        degrees_of_freedom=blank_dof,
        num_corrected_mean=num_corrected_mean,
        den_corrected_mean=den_corrected_mean,
        blank_mode=blank_mode,
        n_blanks_used=len(blocks),
        # Two blanks are combined here, so report the smaller cycle count: the
        # same convention the 2-variable and 3-variable before/after models use,
        # and the conservative one for an SD/sqrt(n) statement.
        n_blank_cycles=min(block.n_pairs for block in blocks),
        blank_uncertainty_input=blank_input_mode,
        model_dimension=len(isotopes),
        aux_isotope=aux_isotope,
        aux_corrected_mean=aux_corrected_mean,
        u_aux_sd=u_aux_sd,
        correlation_labels=tuple(isotopes),
        correlation_matrix=blocks[0].covariance_matrix.copy() if len(blocks) == 1 else None,
        per_blank_results=per_block_results,
    )


def _get_corrected_intensity_mean(
    *,
    sample: Sample,
    isotope: str,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> float:
    src = sample.corrected_intensities if sample.corrected_intensities else sample.intensities
    cd = src.get(isotope)
    if cd is not None and cd.n_valid > 0:
        valid_vals = get_filtered_values(
            cd.values,
            cd.mask,
            sample.name,
            cycle_ranges=cycle_ranges,
            sample_key=sample_cycle_key(sample),
            filter_method="None",
            filter_threshold=2.0,
        )
        if len(valid_vals) > 0:
            return float(np.mean(valid_vals))
    return 0.0


def _compute_crm(
    *,
    certified_value: Optional[CertifiedValue],
    uncertainty_config: UncertaintyConfig,
    ratio_mean: float,
) -> Tuple[float, float]:
    if uncertainty_config.output_mode == "delta":
        return 0.0, 0.0
    if certified_value is None or certified_value.uncertainty <= 0:
        return 0.0, 0.0
    u_std = u_certified_value(certified_value.uncertainty, certified_value.k)
    u_rel_permil = (u_std / abs(ratio_mean)) * 1000.0 if ratio_mean else 0.0
    return u_std, u_rel_permil


def _combine_and_build_budget(
    *,
    contributors: List[UncertaintyContributor],
    ratio_mean: float,
    n_cycles: int,
    uncertainty_config: UncertaintyConfig,
) -> UncertaintyBudget:
    """Combine Pb-Tl contributors by RSS in relative permil space."""
    return combine_and_build_budget_shared(
        engine="pb_tl_external_normalization",
        contributors=contributors,
        ratio_mean=ratio_mean,
        n_cycles=n_cycles,
        uncertainty_config=uncertainty_config,
        use_abs_dof=False,
    )

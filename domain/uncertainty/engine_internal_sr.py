"""Engine A — Internal normalisation uncertainty budget for Sr."""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, List, Mapping, Optional, Set, Tuple

from dataclasses import replace as _dc_replace

import numpy as np
from config.contributor_names import LABEL_U_PREC

from config.reference_materials import (
    get_internal_normalization,
    get_masses_for_isotopes,
    get_natural_ratio,
    get_natural_ratio_relative_uncertainty,
    require_isotope_mass,
    require_natural_ratio,
    require_natural_ratio_relative_uncertainty,
)
from config.settings import (
    CustomUncertaintyContributor,
    ProcessingConfig,
    UncertaintyConfig,
)
from domain.elements.base import CertifiedValue, ElementConfig
from domain.filters.outlier import (
    get_filtered_values,
    resolve_cycle_range,
    sample_cycle_key,
)
from domain.models import Sample, UncertaintyBudget, UncertaintyContributor
from domain.ratio_utils import normalize_ratio_token as _normalize_ratio_token
from domain.output_scale import (
    INPUT_LAYER_PRE_ANCHOR,
    resolve_output_scale,
    scaled_contribution,
)
from domain.ratio_selection import get_best_ratio_data
from domain.uncertainty.blank import (
    BlankUncertaintyResult,
    _compute_correlation,
    _single_blank_dof,
    blank_input_sigma,
    compute_blank_stats_3var,
    compute_blank_uncertainty,
    describe_blank_input_model,
    get_paired_blank_voltages,
    normalize_blank_uncertainty_input,
    _contributing_blank_cycles,
    _weighted_channel_mean,
    resolve_blank_channel_weights,
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
from domain.uncertainty.kragten import (
    InterferenceUncertainty,
    SrInterferenceReferenceInputs,
    _run_sr_correction_chain,
    compute_interference_uncertainty,
)
from domain.uncertainty.propagation import (
    select_precision_value,
    u_certified_value,
    u_precision,
)
from domain.uncertainty.reprod import ReprodResult, compute_reprod
from domain.uncertainty.sr_sample_values import (
    resolve_sr_digestion_inputs,
    resolve_sr_qc_bias_inputs,
)
from domain.uncertainty.welch_satterthwaite import effective_dof
from domain.uncertainty.sr_chain_identity import replay_sr_chain_method
from domain.sr_normalization_support import (
    require_supported_sr_normalization_ratio,
)
from domain.uncertainty.eligibility import (
    missing_sr_normalization_input_budget,
    required_contributor_unavailable_budget,
    unresolved_blank_reference_budget,
)
from domain.uncertainty.shared_engine import combine_and_build_budget_shared

_LOG = logging.getLogger(__name__)


def _resolve_reprod_reference_mean(reprod_result: ReprodResult) -> float:
    """Return the included-standard mean that defines the repeatability basis."""
    if len(reprod_result.std_means) == 0:
        return 0.0
    included_mask = np.asarray(reprod_result.std_included, dtype=bool)
    std_means = np.asarray(reprod_result.std_means, dtype=np.float64)
    selected = std_means[included_mask]
    selected = selected[np.isfinite(selected)]
    return float(np.mean(selected)) if len(selected) > 0 else 0.0


#: Engine A reports an internally normalised absolute ratio only.
ENGINE_A_SUPPORTED_OUTPUT_MODES = ("absolute_ratio",)


def resolve_engine_a_output_mode(
    uncertainty_config: UncertaintyConfig,
    processing_config: ProcessingConfig,
) -> str:
    """Return the output mode Engine A will report, or reject the request.

    Raises ``ValueError`` when delta output is actually being asked for -
    ``output_mode="delta"`` together with delta processing enabled. Engine A
    has no delta measurand, so producing a budget anyway and labelling it
    "delta" would be worse than failing.

    ``output_mode="delta"`` while delta processing is *disabled* is the stale
    default in :class:`~config.settings.UncertaintyConfig`, which
    ``sync_uncertainty_with_processing`` already normalises upstream. Here it
    is normalised too, so a budget can never carry a mode it did not compute.
    """
    mode = str(getattr(uncertainty_config, "output_mode", "") or "")
    delta_enabled = bool(getattr(processing_config, "enable_delta", False))

    if mode == "delta" and delta_enabled:
        raise ValueError(
            "Engine A (internal normalisation) reports absolute ratios only; "
            "output_mode='delta' is not supported. Delta output requires the "
            "SSB/delta engine. Supported modes: "
            f"{', '.join(ENGINE_A_SUPPORTED_OUTPUT_MODES)}."
        )
    if mode in ENGINE_A_SUPPORTED_OUTPUT_MODES:
        return mode
    return "absolute_ratio"


def compute_budget_internal(
    sample: Sample,
    ratio_name: str,
    *,
    all_samples: List[Sample],
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    processing_config: ProcessingConfig,
    certified_value: Optional[CertifiedValue] = None,
    ref_certified_value: Optional[CertifiedValue] = None,
    ratio_values: Optional[np.ndarray] = None,
    ratio_mean: Optional[float] = None,
    runtime_ratio_mask: Optional[np.ndarray] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    drift_model: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    position_extractor: Optional[Callable[[Sample], float]] = None,
    custom_contributor_library: Optional[Dict[str, List[CustomUncertaintyContributor]]] = None,
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
) -> Optional[UncertaintyBudget]:
    """Compute a full Engine A (internal normalisation) uncertainty budget.

    For Pb, delegates to :func:`domain.uncertainty.engine_external_pb_tl.compute_budget_pb_tl`
    (Engine C) which handles Tl-normalised Pb uncertainty with optional Hg
    interference correction.
    """
    if sample.is_blank:
        return None

    from domain.sr_standard_calibration import sr_calibration_budget_guard
    sr_refusal = sr_calibration_budget_guard(
        sample, ratio_name, uncertainty_config.output_mode if uncertainty_config is not None else "absolute_ratio",
    )
    if sr_refusal is not None:
        return sr_refusal

    if runtime_ratio_mask is not None:
        # Replay helpers consume the selected ratio's support. Isolate this
        # runtime view so session-level standards and stored outputs stay intact.
        sample = sample.copy()
        selected = get_best_ratio_data(sample, ratio_name)
        if selected is not None:
            accepted = np.asarray(runtime_ratio_mask, dtype=bool)
            if accepted.shape != selected.mask.shape:
                raise ValueError("Runtime ratio mask must match the selected cycle series")
            selected.mask &= accepted

    # --- Pb dispatches to the Pb-Tl engine (Engine C) -----------------------
    if element_config.symbol == "Pb":
        from domain.uncertainty.engine_external_pb_tl import compute_budget_pb_tl  # local import avoids circular
        return compute_budget_pb_tl(
            sample=sample,
            ratio_name=ratio_name,
            all_samples=all_samples,
            element_config=element_config,
            uncertainty_config=uncertainty_config,
            processing_config=processing_config,
            certified_value=certified_value,
            ratio_values=ratio_values,
            ratio_mean=ratio_mean,
            cycle_ranges=cycle_ranges,
            drift_model=drift_model,
            position_extractor=position_extractor,
            custom_contributor_library=custom_contributor_library,
            profile_defaults=profile_defaults,
        )

    # --- A-13: reject delta output at the engine boundary --------------------
    #
    # Engine A reports an internally normalised absolute ratio. There is no
    # delta measurand in this model, so a delta request is a misuse, not a
    # formatting preference. The check lives here rather than in the UI
    # because the UI is not the only caller: a script importing this function
    # bypasses every upstream guard.
    #
    # It is deliberately not conditioned on ``uncertainty_config.engine``,
    # which is a user-settable label. Reaching this function *is* the
    # engine-A boundary.
    output_mode = resolve_engine_a_output_mode(
        uncertainty_config, processing_config,
    )

    # A012: session-wide routing to Engine A does not mean every sample was
    # normalized.  _sr_correction_loop skips a sample missing its
    # normalization-pair channels, leaving that sample's ratios uncorrected, so
    # an "internal_normalization" budget would label a correction that never ran.
    missing_normalization = missing_sr_normalization_input_budget(
        sample, ratio_name, element_config, processing_config, output_mode,
    )
    if missing_normalization is not None:
        return missing_normalization

    # --- Extract ratio values ------------------------------------------------
    if ratio_values is None:
        cd = get_best_ratio_data(sample, ratio_name)
        if cd is None:
            return None
        ratio_values = cd.valid_values

    finite_cycle_count = int(np.sum(np.isfinite(ratio_values)))
    if finite_cycle_count < 2:
        return UncertaintyBudget(
            engine="internal_normalization",
            output_mode=output_mode,
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

    # Keep _enabled as an alias so internal helpers that call it directly
    # (e.g. reprod SE branch) still work without modification.
    _enabled = _active

    # u_prec - Sample measurement precision (SE by default, SD when selected)
    u_prec_se_abs, u_prec_sd_abs, mean_val = u_precision(ratio_values)
    u_prec_abs, u_prec_mode = select_precision_value(
        getattr(uncertainty_config, "u_prec_mode", "se"),
        u_prec_se_abs,
        u_prec_sd_abs,
    )
    n_cycles = int(np.sum(np.isfinite(ratio_values)))

    # A002 follow-up: a recorded blank reference that cannot be resolved is an
    # unknown contribution, not a missing one. Reporting the rest of the budget
    # would present an uncertainty computed without a term this configuration
    # requires, so the budget is refused with the reason all three engines share.
    blank_selection = resolve_blank_selection(sample, all_samples)
    if _active("u_blank") and not blank_selection.is_fully_resolved:
        return unresolved_blank_reference_budget(
            sample,
            blank_selection,
            engine="internal_normalization",
            output_mode=output_mode,
            basis_ratio_value=ratio_mean,
            n_cycles=n_cycles,
        )

    u_prec_rel_permil = (u_prec_abs / ratio_mean) * 1000.0 if ratio_mean else 0.0
    u_prec_description = (
        "Within-run standard deviation of per-cycle ratios (Type A)."
        if u_prec_mode == "sd"
        else "Standard error of the mean of per-cycle ratios (Type A)."
    )

    contributors.append(UncertaintyContributor(
        name="u_prec",
        display_name=f"{LABEL_U_PREC} (Type A)",
        value_abs=u_prec_abs,
        value_rel_permil=u_prec_rel_permil,
        type_ab="A",
        degrees_of_freedom=float(max(n_cycles - 1, 1)),
        percentage_contribution=0.0,
        description=u_prec_description,
        **_contributor_gate("u_prec"),
    ))

    # u_std_repeatability — Standard repeatability
    # Engine A uses internal normalisation: LOO and drift-residuals are
    # designed for SSB bracketing and have no meaning here.  Force
    # sd_of_means for Engine A repeatability.
    _reprod_config = _dc_replace(uncertainty_config, reprod_method="sd_of_means")
    reprod_ratio_extractor = (
        _get_pre_anchor_ratio
        if processing_config.sr_session_anchoring
        else _get_iif_or_best_ratio
    )
    reprod_basis_label = (
        "pre-anchor internally-corrected SRM session means"
        if processing_config.sr_session_anchoring
        else "internally-corrected SRM session means"
    )
    bias_ref_basis_label = (
        "pre-anchor included Sr standards"
        if processing_config.sr_session_anchoring
        else "included Sr standards"
    )
    reprod_result = compute_reprod(
        all_samples=all_samples,
        ratio_name=ratio_name,
        uncertainty_config=_reprod_config,
        element_config=element_config,
        ratio_extractor=reprod_ratio_extractor,
        drift_model=drift_model,
        position_extractor=position_extractor,
    )
    if (
        (_enabled("u_std_repeatability") or _enabled("u_std_repeatability_se"))
        and reprod_result.status == "unavailable"
    ):
        return required_contributor_unavailable_budget(
            engine="internal_normalization",
            output_mode=output_mode,
            contributor_name="u_std_repeatability",
            reason=reprod_result.unavailable_reason or "repeatability inference is unsupported",
            basis_ratio_value=ratio_mean,
            n_cycles=n_cycles,
        )

    u_std_repeatability_abs = reprod_result.u_std_repeatability_abs
    reprod_reference_mean = _resolve_reprod_reference_mean(reprod_result)
    # Engine A repeatability is estimated from SRM-session means. Keep the
    # relative display on that same SRM reference basis rather than silently
    # re-normalising it to the current sample mean.
    u_std_repeatability_rel_permil = (
        (u_std_repeatability_abs / reprod_reference_mean) * 1000.0
        if reprod_reference_mean and u_std_repeatability_abs > 0 else 0.0
    )
    reprod_dof = reprod_result.degrees_of_freedom

    # Number of included standards for SE computation
    _n_std_included = int(sum(1 for inc in reprod_result.std_included if inc))
    # SE = SD / sqrt(n) — collapses the standard-session scatter to the mean
    # of n independent SRM runs.
    _n_std_for_se = _n_std_included if _n_std_included > 1 else 1
    u_std_repeatability_se_abs = u_std_repeatability_abs / np.sqrt(_n_std_for_se)
    u_std_repeatability_se_rel_permil = (
        (u_std_repeatability_se_abs / reprod_reference_mean) * 1000.0
        if reprod_reference_mean and u_std_repeatability_se_abs > 0 else 0.0
    )

    _METHOD_LABELS = {
        "sd_of_means": "SD of standard means",
        "loo_cross_validation": "LOO cross-validation residuals",
        "drift_residuals": "Drift model residuals",
        "robust_mad": "Robust MAD estimator",
    }
    reprod_label = _METHOD_LABELS.get(reprod_result.method, reprod_result.method)

    # u_std_repeatability_abs is expressed on the SRM basis (the SD of the
    # session's included SRM standard means), not re-projected onto the current
    # sample ratio. Across the validated Sr sample/SRM range, the basis
    # difference is below 2% of this minor term. Changing that basis requires a
    # separately validated scientific issue; do not silently "correct" it here.
    contributors.append(UncertaintyContributor(
        name="u_std_repeatability",
        display_name=f"Repeatability of SRM (SD, {reprod_label})",
        value_abs=u_std_repeatability_abs,
        value_rel_permil=u_std_repeatability_rel_permil,
        type_ab="A",
        degrees_of_freedom=float(reprod_dof) if reprod_dof >= 1 else 1.0,
        percentage_contribution=0.0,
        description=(
            f"Full SD of {reprod_basis_label} "
            f"via {reprod_label}. "
            "Reported in ‰ on the included SRM-session mean basis; combined "
            "in absolute ratio units in the Engine A RSS step."
        ),
        **_contributor_gate("u_std_repeatability", u_std_repeatability_abs > 0),
    ))

    # u_std_repeatability_se is an alternative SE-based contributor; only
    # append it when explicitly enabled. The two are mutually exclusive —
    # appending the disabled SE entry alongside the SD entry adds a dead
    # zero-value row to every budget.
    if _enabled("u_std_repeatability_se"):
        contributors.append(UncertaintyContributor(
            name="u_std_repeatability_se",
            display_name=f"Repeatability of SRM (SE, n={_n_std_included})",
            value_abs=u_std_repeatability_se_abs,
            value_rel_permil=u_std_repeatability_se_rel_permil,
            type_ab="A",
            degrees_of_freedom=float(_n_std_for_se - 1) if _n_std_for_se > 1 else 1.0,
            percentage_contribution=0.0,
            description=(
                f"SE of SRM session means: SD / sqrt({_n_std_included}). "
                "Reported in ‰ on the included SRM-session mean basis; combined "
                "in absolute ratio units in the Engine A RSS step."
            ),
            **_contributor_gate("u_std_repeatability_se", u_std_repeatability_se_abs > 0),
        ))

    # Optional: kappa_drift as additional Type B contributor
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
            engine="internal_normalization", output_mode=uncertainty_config.output_mode,
            contributor_name="u_kappa_drift", reason="Instrumental drift requires at least one eligible consecutive standard pair within a segment.",
            basis_ratio_value=ratio_mean, ratio_value=ratio_mean, n_cycles=n_cycles,
        )
        budget.reprod_result = reprod_result
        return budget

    if drift_requested and drift_has_pairs:
        kd_abs = (reprod_result.kappa_drift_permil / 1000.0) * ratio_mean
        contributors.append(UncertaintyContributor(
            name="u_kappa_drift",
            display_name="Instrumental drift (Type B)",
            value_abs=kd_abs,
            value_rel_permil=reprod_result.kappa_drift_permil,
            type_ab="B",
            degrees_of_freedom=float('inf'),
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
        ))

    # u_blank — Blank correction
    try:
        blank_result = _compute_blank_contribution(
            sample,
            ratio_name,
            all_samples,
            element_config,
            uncertainty_config,
            ratio_mean,
            processing_config=processing_config,
            cycle_ranges=cycle_ranges,
        )
    except ValueError as exc:
        if _active("u_blank"):
            return required_contributor_unavailable_budget(engine="internal_normalization", output_mode="absolute_ratio",
                contributor_name="u_blank", reason=str(exc), basis_ratio_value=ratio_mean, ratio_value=ratio_mean, n_cycles=n_cycles)
        blank_result = BlankUncertaintyResult(n_blanks_used=0)

    # Engine A contributors below are produced by replaying the Sr chain,
    # which stops at the internally normalized (pre-anchor) ratio. Session
    # anchoring and drift then multiply that ratio by committed scalars, so
    # each replayed sensitivity carries the same product exactly once.
    replay_output_scale = resolve_output_scale(
        sample, ratio_name, input_layer=INPUT_LAYER_PRE_ANCHOR,
    )

    blank_description = (
        f"Sr blank {blank_result.model_dimension}-variable covariance model on {', '.join(blank_result.correlation_labels)}; numerical sensitivities through the exact Sr chain. "
        f"Distinct blank observations assumed independent; repeated observation IDs are shared. ({replay_output_scale.describe()}). "
        + describe_blank_input_model(blank_result.blank_uncertainty_input, blank_result.n_blank_cycles)
        + " Reference: JCGM 100:2008 section 5.2."
    )
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

    contributors.append(UncertaintyContributor(
        name="u_blank",
        display_name="Blank correction (Type A)",
        value_abs=u_blank_abs,
        value_rel_permil=(
            (u_blank_abs / ratio_mean) * 1000.0
            if ratio_mean and u_blank_abs > 0 else 0.0
        ),
        type_ab="A",
        degrees_of_freedom=blank_result.degrees_of_freedom,
        percentage_contribution=0.0,
        description=blank_description,
        **_contributor_gate("u_blank", blank_result.n_blank_cycles >= 2, u_blank_missing_reason),
    ))

    # u_interf — Isobaric interference
    try:
        interf_result = _compute_interference(
            sample,
            ratio_name=ratio_name,
            element_config=element_config,
            processing_config=processing_config,
            cycle_ranges=cycle_ranges,
        )
    except ValueError as exc:
        if _active("u_interf"):
            return required_contributor_unavailable_budget(engine="internal_normalization", output_mode="absolute_ratio",
                contributor_name="u_interf", reason=str(exc), basis_ratio_value=ratio_mean, ratio_value=ratio_mean, n_cycles=n_cycles)
        interf_result = InterferenceUncertainty(0., 0., 0., 0., 0., 0., 0., 0.)

    u_interf_missing_reason = ""
    if _active("u_interf") and interf_result.u_interf_abs <= 0.0:
        if ratio_name != "87Sr/86Sr":
            u_interf_missing_reason = f"u_interf not applicable for ratio {ratio_name}."
        elif not processing_config.apply_interference_correction:
            u_interf_missing_reason = "u_interf interference correction is disabled in processing configuration."
        else:
            src = _get_engine_a_chain_source(sample)
            reference_inputs = _resolve_sr_interference_reference_inputs(
                processing_config,
                element_config,
                sample,
            )
            enabled_interferents = _enabled_sr_interferents(processing_config, element_config)
            if not enabled_interferents:
                u_interf_missing_reason = "u_interf no interferents enabled in configuration."
            else:
                required = ["87Sr", "86Sr", reference_inputs.normalization_numerator, reference_inputs.normalization_denominator]
                if "87Rb" in enabled_interferents:
                    required.append("85Rb")
                if "86Kr" in enabled_interferents:
                    required.append("83Kr")
                missing_iso = [iso for iso in required if iso not in src]
                if missing_iso:
                    u_interf_missing_reason = f"u_interf missing intensity data for {', '.join(missing_iso)}."
                else:
                    u_interf_missing_reason = "u_interf interference uncertainty is zero or could not be evaluated."

    enabled_interferents = _enabled_sr_interferents(processing_config, element_config)
    u_interf_abs = scaled_contribution(
        interf_result.u_interf_abs, replay_output_scale,
    )
    contributors.append(UncertaintyContributor(
        name="u_interf",
        display_name=_sr_interference_display_name(enabled_interferents),
        value_abs=u_interf_abs,
        value_rel_permil=(
            (u_interf_abs / ratio_mean) * 1000.0
            if ratio_mean and u_interf_abs > 0 else 0.0
        ),
        type_ab="B",
        degrees_of_freedom=float('inf'),
        percentage_contribution=0.0,
        description=(
            f"{_sr_interference_description(enabled_interferents)} "
            f"({replay_output_scale.describe()})."
        ),
        **_contributor_gate("u_interf", u_interf_abs > 0, u_interf_missing_reason),
    ))

    # u_norm_ratio -- accepted internal-normalization ratio uncertainty
    try:
        u_norm_ratio_abs, u_norm_ratio_rel_permil = _compute_norm_ratio_uncertainty(
            sample,
            ratio_name=ratio_name,
            ratio_mean=ratio_mean,
            element_config=element_config,
            uncertainty_config=uncertainty_config,
            processing_config=processing_config,
            cycle_ranges=cycle_ranges,
        )
    except ValueError as exc:
        if _active("u_norm_ratio"):
            return required_contributor_unavailable_budget(engine="internal_normalization", output_mode="absolute_ratio",
                contributor_name="u_norm_ratio", reason=str(exc), basis_ratio_value=ratio_mean, ratio_value=ratio_mean, n_cycles=n_cycles)
        u_norm_ratio_abs, u_norm_ratio_rel_permil = 0.0, 0.0
    u_norm_ratio_abs = scaled_contribution(u_norm_ratio_abs, replay_output_scale)
    u_norm_ratio_rel_permil = (
        (u_norm_ratio_abs / ratio_mean) * 1000.0
        if ratio_mean and u_norm_ratio_abs > 0 else 0.0
    )
    contributors.append(UncertaintyContributor(
        name="u_norm_ratio",
        display_name="Normalization ratio reference (Type B)",
        value_abs=u_norm_ratio_abs,
        value_rel_permil=u_norm_ratio_rel_permil,
        type_ab="B",
        degrees_of_freedom=float("inf"),
        percentage_contribution=0.0,
        description=(
            "Accepted internal-normalization ratio standard uncertainty "
            "propagated by numerical perturbation through the Sr correction "
            f"chain ({replay_output_scale.describe()})."
        ),
        **_contributor_gate("u_norm_ratio", u_norm_ratio_abs > 0),
    ))

    # NOTE: IIF correction repeatability was a separate contributor in earlier
    # versions. It has been merged into u_std_repeatability for Engine A:
    # both compute the SD of the Engine A SRM basis (pre-anchor when anchoring
    # is enabled, otherwise the live IIF-corrected standard means).
    # Keeping a separate IIF repeatability term would double-count the same variance.
    # u_std_repeatability is the single reproducibility contributor for Engine A.

    # u_CRM — Certified reference material
    u_crm_abs, u_crm_rel_permil = _compute_crm(
        certified_value, uncertainty_config, ratio_mean,
    )

    contributors.append(UncertaintyContributor(
        name="u_crm",
        display_name="CRM certified value (Type B)",
        value_abs=u_crm_abs,
        value_rel_permil=u_crm_rel_permil,
        type_ab="B",
        degrees_of_freedom=float('inf'),
        percentage_contribution=0.0,
        description="Certified reference material expanded uncertainty / k.",
        **_contributor_gate(
            "u_crm",
            u_crm_abs > 0,
            not_applicable_reason=not_applicable_reason(
                "u_crm", output_mode=uncertainty_config.output_mode,
            ),
        ),
    ))

    # u_ref_value — literature reference value (e.g. GeoReM), Type B.
    # Independent user-selected reference-related term. If the user enables
    # both u_crm and u_ref_value, both are included when data are available.
    u_ref_value_abs, u_ref_value_rel_permil = _compute_ref_value(
        ref_certified_value, uncertainty_config, ratio_mean,
    )

    contributors.append(UncertaintyContributor(
        name="u_ref_value",
        display_name="Reference value uncertainty (literature, Type B)",
        value_abs=u_ref_value_abs,
        value_rel_permil=u_ref_value_rel_permil,
        type_ab="B",
        degrees_of_freedom=float("inf"),
        percentage_contribution=0.0,
        description=(
            "Static Type B uncertainty of the accepted literature reference "
            "ratio (e.g. GeoReM consensus value), independent of session data "
            "or anchoring state."
        ),
        **_contributor_gate("u_ref_value", u_ref_value_abs > 0),
    ))

    # u_bias_ref — standard-session bias (optional Type B)
    u_bias_ref_abs, u_bias_ref_rel_permil, bias_ref_stats = compute_reference_bias_term(
        reprod_result=reprod_result,
        certified_value=certified_value,
        element_config=element_config,
        ratio_name=ratio_name,
        ratio_mean=ratio_mean,
        reference_value_override=getattr(
            uncertainty_config,
            "sr_reference_bias_ref_value",
            None,
        ),
    )
    contributors.append(UncertaintyContributor(
        name="u_bias_ref",
        display_name="Reference bias (\u0394_ref) (Type B, unavailable)",
        value_abs=u_bias_ref_abs,
        value_rel_permil=u_bias_ref_rel_permil,
        type_ab="B",
        degrees_of_freedom=float("inf"),
        percentage_contribution=0.0,
        description=(
            "Observed session mean offset of the "
            f"{bias_ref_basis_label} relative to the accepted reference value. "
            "Diagnostic only: the automatic rectangular model "
            "(u = |Δ_ref| / sqrt(3)) is disabled in this release, so this term "
            "contributes nothing to the combined uncertainty."
        ),
        # The row is still emitted so the budget says explicitly
        # that this term exists and is unavailable, and why. Dropping it would
        # leave a reader unable to tell "not applicable here" from "silently
        # omitted".
        is_active=False,
        state=ContributorState.NO_APPROVED_MODEL.value,
        inactive_reason=AUTOMATIC_REFERENCE_BIAS_DISABLED_REASON,
    ))

    # u_bias_qc — QC bias (optional Type B)
    sr_qc_bias_abs, sr_qc_cert_value = resolve_sr_qc_bias_inputs(
        sample,
        uncertainty_config,
    )
    u_bias_qc_abs, u_bias_qc_rel_permil, _ = compute_qc_bias_term(
        observed_bias_abs=sr_qc_bias_abs,
        ratio_mean=ratio_mean,
        qc_cert_value=sr_qc_cert_value,
    )
    contributors.append(UncertaintyContributor(
        name="u_bias_qc",
        display_name="Bias in processed control sample",
        value_abs=u_bias_qc_abs,
        value_rel_permil=u_bias_qc_rel_permil,
        type_ab="B",
        degrees_of_freedom=float("inf"),
        percentage_contribution=0.0,
        description=(
            "Top-down Type B term from a user-supplied standard uncertainty "
            "for a processed control sample (e.g. AGV-2a), transferred "
            "fractionally using the QC material's certified ratio."
        ),
        **_contributor_gate("u_bias_qc", u_bias_qc_abs > 0),
    ))

    # u_reprod_dig — Between-digestion reproducibility (user-supplied)
    sr_digestion_sd_abs, sr_digestion_ref_value = resolve_sr_digestion_inputs(
        sample,
        uncertainty_config,
    )
    _reprod_dig_abs, _reprod_dig_rel_permil, _ = compute_digestion_reproducibility_term(
        digestion_sd_abs=sr_digestion_sd_abs,
        ratio_mean=ratio_mean,
        digestion_ref_value=sr_digestion_ref_value,
    )
    contributors.append(UncertaintyContributor(
        name="u_reprod_dig",
        display_name="Between-digestion reproducibility (Type B)",
        value_abs=_reprod_dig_abs,
        value_rel_permil=_reprod_dig_rel_permil,
        type_ab="B",
        degrees_of_freedom=float("inf"),
        percentage_contribution=0.0,
        description=(
            "SD of independently processed digestion means, user-supplied. "
            "If a reference ratio is supplied, the SD is treated as a "
            "fractional reproducibility term and transferred to the sample "
            "basis. Treated as Type B (\u03bd=\u221e)."
        ),
        **_contributor_gate("u_reprod_dig", _reprod_dig_abs > 0),
    ))

    # Inject custom contributor rows (element-filtered inside helper).
    custom_rows = build_custom_contributor_rows(
        sample=sample,
        element_symbol=element_config.symbol,
        ratio_mean=ratio_mean,
        custom_contributor_library=custom_contributor_library or {},
    )
    contributors.extend(custom_rows)

    # RSS combination + Welch-Satterthwaite
    budget = _combine_and_build_budget(
        contributors=contributors,
        ratio_mean=ratio_mean,
        n_cycles=n_cycles,
        uncertainty_config=uncertainty_config,
    )

    # Q-005: preserve calibrated ratios without claiming an unqualified joint budget.
    if processing_config.sr_session_anchoring:
        budget = required_contributor_unavailable_budget(
            engine="internal_normalization", output_mode=uncertainty_config.output_mode,
            contributor_name="Sr anchor/reference model",
            reason="Joint Y=C*X/S anchor, standard and reference dependence is unqualified (Q-005); calibrated ratios remain available, combined uncertainty does not.",
            basis_ratio_value=ratio_mean, ratio_value=ratio_mean, n_cycles=n_cycles,
        )

        budget.contributors = contributors

    # Attach reprod metadata for UI display
    budget.reprod_result = reprod_result

    return budget


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_blank_contribution(
    sample: Sample,
    ratio_name: str,
    all_samples: List[Sample],
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    ratio_mean: float,
    *,
    processing_config: ProcessingConfig,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> BlankUncertaintyResult:
    """Compute Engine A blank uncertainty, with optional Sr 3-variable mode."""
    from domain.uncertainty.engine_external_pb_tl import build_pb_tl_blank_blocks
    if resolve_blank_correction_mode(sample, processing_config.blank_mode) == "none":
        return BlankUncertaintyResult(blank_mode="none", n_blanks_used=0)
    if ratio_name != "87Sr/86Sr":
        raise ValueError("Sr chain blank propagation is supported only for 87Sr/86Sr.")
    isotopes = _sr_blank_isotopes(sample, uncertainty_config, processing_config)
    blocks = build_pb_tl_blank_blocks(sample, all_samples, isotopes=isotopes,
        uncertainty_config=uncertainty_config, processing_config=processing_config, cycle_ranges=cycle_ranges, paired_blank_filter=True)
    components = []
    independent = 0.0
    for block in blocks:
        derivatives = []
        for index, iso in enumerate(block.isotopes):
            sigma = float(np.sqrt(block.covariance_matrix[index, index]))
            weight = block.channel_weights[index]
            derivatives.append(weight * _compute_blank_isotope_sensitivity(sample, ratio_name=ratio_name,
                perturbed_isotope=iso, blank_sd=sigma * 1e-3, element_config=element_config,
                processing_config=processing_config, cycle_ranges=cycle_ranges) if sigma else 0.0)
        gradient = np.array(derivatives)
        variance = float(gradient @ block.covariance_matrix @ gradient)
        if not np.isfinite(variance):
            raise ValueError("Nonfinite Sr blank propagation.")
        components.append((float(np.sqrt(max(variance, 0))), block.degrees_of_freedom))
        independent += float(np.sum(gradient**2 * np.diag(block.covariance_matrix)))
    total = float(np.sqrt(sum(u*u for u, _ in components)))
    return BlankUncertaintyResult(u_blank_abs=total, u_blank_uncorrelated_abs=float(np.sqrt(independent)),
        u_blank_correlation_term_abs2=total**2-independent, degrees_of_freedom=effective_dof(components),
        blank_mode=resolve_blank_correction_mode(sample, processing_config.blank_mode), n_blanks_used=len(blocks),
        n_blank_cycles=min((b.n_pairs for b in blocks), default=0), model_dimension=len(isotopes),
        correlation_labels=isotopes, blank_uncertainty_input=uncertainty_config.blank_uncertainty_input)


def _sr_blank_isotopes(sample, uncertainty_config, processing_config):
    source = _get_engine_a_chain_source(sample)
    isotopes = ["87Sr", "86Sr"]
    if uncertainty_config.sr_blank_3var:
        isotopes.append("88Sr")
    if processing_config.is_monitor_enabled("87Rb") and "85Rb" in source:
        isotopes.append("85Rb")
    if processing_config.is_monitor_enabled("86Kr") and "83Kr" in source:
        isotopes.append("83Kr")
    return tuple(isotopes)


def _compute_blank_contribution_2var(
    sample: Sample,
    blank_samples: List[Sample],
    *,
    num_isotope: str,
    den_isotope: str,
    num_corrected_mean: float,
    den_corrected_mean: float,
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    processing_config: ProcessingConfig,
    blank_mode: str,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[BlankUncertaintyResult]:
    """Compute a 2-variable Engine A blank model via chain-based sensitivities."""
    if blank_mode == "none":
        return BlankUncertaintyResult(
            blank_mode="none",
            n_blanks_used=0,
            num_corrected_mean=num_corrected_mean,
            den_corrected_mean=den_corrected_mean,
        )

    if not blank_samples:
        return None

    if blank_mode == "before_and_after" and len(blank_samples) >= 2:
        channel_weights = resolve_blank_channel_weights(
            sample, (num_isotope, den_isotope), blank_mode=blank_mode,
        )
        weights_before = _channel_weight_vector(
            channel_weights, "before", (num_isotope, den_isotope),
        )
        weights_after = _channel_weight_vector(
            channel_weights, "after", (num_isotope, den_isotope),
        )
        result_before = _compute_single_blank_uncertainty_2var(
            sample,
            blank_samples[0],
            num_isotope=num_isotope,
            den_isotope=den_isotope,
            num_corrected_mean=num_corrected_mean,
            den_corrected_mean=den_corrected_mean,
            element_config=element_config,
            uncertainty_config=uncertainty_config,
            processing_config=processing_config,
            cycle_ranges=cycle_ranges,
            channel_weight_vector=weights_before,
        )
        result_after = _compute_single_blank_uncertainty_2var(
            sample,
            blank_samples[-1],
            num_isotope=num_isotope,
            den_isotope=den_isotope,
            num_corrected_mean=num_corrected_mean,
            den_corrected_mean=den_corrected_mean,
            element_config=element_config,
            uncertainty_config=uncertainty_config,
            processing_config=processing_config,
            cycle_ranges=cycle_ranges,
            channel_weight_vector=weights_after,
        )
        if result_before is None or result_after is None:
            return None

        # The producer's per-channel weights are already inside each side's
        # quadrature, so the two independent blanks combine directly.
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

        combined_corr = None
        if (
            result_before.correlation_matrix is not None
            and result_after.correlation_matrix is not None
        ):
            _m_before = np.clip(
                np.asarray(result_before.correlation_matrix, dtype=np.float64),
                -0.9999, 0.9999,
            )
            _m_after = np.clip(
                np.asarray(result_after.correlation_matrix, dtype=np.float64),
                -0.9999, 0.9999,
            )
            combined_corr = np.tanh((np.arctanh(_m_before) + np.arctanh(_m_after)) / 2.0)
            np.fill_diagonal(combined_corr, 1.0)

        warning_parts = []
        if result_before.correlation_warning:
            warning_parts.append(
                f"{blank_samples[0].name}: {result_before.correlation_warning}"
            )
        if result_after.correlation_warning:
            warning_parts.append(
                f"{blank_samples[-1].name}: {result_after.correlation_warning}"
            )

        _r_before = np.clip(result_before.correlation, -0.9999, 0.9999)
        _r_after = np.clip(result_after.correlation, -0.9999, 0.9999)
        _r_combined = float(np.tanh(
            (np.arctanh(_r_before) + np.arctanh(_r_after)) / 2.0
        ))

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
            correlation=_r_combined,
            degrees_of_freedom=dof,
            num_corrected_mean=num_corrected_mean,
            den_corrected_mean=den_corrected_mean,
            blank_mode="before_and_after",
            n_blanks_used=sum(
                1 for w in (weights_before, weights_after) if float(np.max(w)) > 0.0
            ),
            n_blank_cycles=_contributing_blank_cycles(
                (result_before, tuple(weights_before)),
                (result_after, tuple(weights_after)),
            ),
            blank_uncertainty_input=result_before.blank_uncertainty_input,
            correlation_warning=" | ".join(warning_parts),
            correlation_labels=(num_isotope, den_isotope),
            correlation_matrix=combined_corr,
            per_blank_results=[result_before, result_after],
        )

    result = _compute_single_blank_uncertainty_2var(
        sample,
        blank_samples[0],
        num_isotope=num_isotope,
        den_isotope=den_isotope,
        num_corrected_mean=num_corrected_mean,
        den_corrected_mean=den_corrected_mean,
        element_config=element_config,
        uncertainty_config=uncertainty_config,
        processing_config=processing_config,
        cycle_ranges=cycle_ranges,
    )
    if result is None:
        return None
    result.blank_mode = "single"
    result.n_blanks_used = 1
    return result


def _compute_single_blank_uncertainty_2var(
    sample: Sample,
    blank: Sample,
    *,
    num_isotope: str,
    den_isotope: str,
    num_corrected_mean: float,
    den_corrected_mean: float,
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    processing_config: ProcessingConfig,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    channel_weight_vector: Optional[np.ndarray] = None,
) -> Optional[BlankUncertaintyResult]:
    """Compute a single-blank Engine A 2-variable covariance result.

    *channel_weight_vector* holds the coefficient this blank's mean carried in
    each channel's subtraction. It scales the chain sensitivities, leaving the
    Kragten perturbation step at the unweighted blank sigma so the sensitivity
    itself is unchanged by how the bracket was shared.
    """
    v_num, v_den = get_paired_blank_voltages(
        blank,
        num_isotope,
        den_isotope,
        cycle_ranges=cycle_ranges,
    )
    if len(v_num) < 2 or len(v_den) < 2:
        return None

    u_num_sd = float(np.std(v_num, ddof=1))
    u_den_sd = float(np.std(v_den, ddof=1))
    n_pairs = min(len(v_num), len(v_den))
    blank_input_mode = normalize_blank_uncertainty_input(
        getattr(uncertainty_config, "blank_uncertainty_input", "sd")
    )
    u_num_input = blank_input_sigma(u_num_sd, n_pairs, blank_input_mode)
    u_den_input = blank_input_sigma(u_den_sd, n_pairs, blank_input_mode)
    correlation, warning = _compute_correlation(
        v_num,
        v_den,
        uncertainty_config.blank_correlation_method,
        uncertainty_config.blank_fixed_r,
    )
    covariance = float(np.clip(correlation, -1.0, 1.0) * u_num_input * u_den_input)
    covariance_matrix = np.array(
        [
            [u_num_input ** 2, covariance],
            [covariance, u_den_input ** 2],
        ],
        dtype=np.float64,
    )
    c_num = _compute_blank_isotope_sensitivity(
        sample,
        ratio_name="87Sr/86Sr",
        perturbed_isotope=num_isotope,
        blank_sd=u_num_input,
        element_config=element_config,
        processing_config=processing_config,
        cycle_ranges=cycle_ranges,
    )
    c_den = _compute_blank_isotope_sensitivity(
        sample,
        ratio_name="87Sr/86Sr",
        perturbed_isotope=den_isotope,
        blank_sd=u_den_input,
        element_config=element_config,
        processing_config=processing_config,
        cycle_ranges=cycle_ranges,
    )
    sensitivities = np.array([c_num, c_den], dtype=np.float64)
    if channel_weight_vector is not None:
        sensitivities = sensitivities * np.asarray(
            channel_weight_vector, dtype=np.float64,
        )
    variance = float(sensitivities @ covariance_matrix @ sensitivities.T)
    variance_uncorrelated = float(
        np.sum(
            (sensitivities * np.array([u_num_input, u_den_input], dtype=np.float64)) ** 2
        )
    )
    correlation_term_abs2 = variance - variance_uncorrelated

    return BlankUncertaintyResult(
        u_blank_abs=float(np.sqrt(max(variance, 0.0))),
        u_blank_uncorrelated_abs=float(np.sqrt(max(variance_uncorrelated, 0.0))),
        u_blank_correlation_term_abs2=correlation_term_abs2,
        u_num_sd=u_num_sd,
        u_den_sd=u_den_sd,
        u_num_input=u_num_input,
        u_den_input=u_den_input,
        correlation=correlation,
        degrees_of_freedom=_single_blank_dof(v_num, v_den),
        num_corrected_mean=num_corrected_mean,
        den_corrected_mean=den_corrected_mean,
        blank_mode="single",
        n_blanks_used=1,
        n_blank_cycles=n_pairs,
        blank_uncertainty_input=blank_input_mode,
        correlation_warning=warning,
        correlation_labels=(num_isotope, den_isotope),
        correlation_matrix=np.array(
            [
                [1.0, np.clip(correlation, -1.0, 1.0)],
                [np.clip(correlation, -1.0, 1.0), 1.0],
            ],
            dtype=np.float64,
        ),
    )


def _compute_blank_contribution_3var(
    sample: Sample,
    blank_samples: List[Sample],
    *,
    num_isotope: str,
    den_isotope: str,
    aux_isotope: str,
    num_corrected_mean: float,
    den_corrected_mean: float,
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    processing_config: ProcessingConfig,
    blank_mode: str,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[BlankUncertaintyResult]:
    """Compute optional Sr 3-variable blank uncertainty including m/z 88."""
    aux_corrected_mean = _get_corrected_intensity_mean(
        sample,
        aux_isotope,
        cycle_ranges=cycle_ranges,
    )
    if not np.isfinite(aux_corrected_mean) or aux_corrected_mean == 0.0:
        return None

    if blank_mode == "none":
        return BlankUncertaintyResult(
            blank_mode="none",
            n_blanks_used=0,
            num_corrected_mean=num_corrected_mean,
            den_corrected_mean=den_corrected_mean,
            model_dimension=3,
            aux_isotope=aux_isotope,
            aux_corrected_mean=aux_corrected_mean,
            correlation_labels=(num_isotope, den_isotope, aux_isotope),
        )

    if not blank_samples:
        return None

    if blank_mode == "before_and_after" and len(blank_samples) >= 2:
        channel_weights = resolve_blank_channel_weights(
            sample, (num_isotope, den_isotope, aux_isotope), blank_mode=blank_mode,
        )
        weights_before = _channel_weight_vector(
            channel_weights, "before", (num_isotope, den_isotope, aux_isotope),
        )
        weights_after = _channel_weight_vector(
            channel_weights, "after", (num_isotope, den_isotope, aux_isotope),
        )
        result_before = _compute_single_blank_uncertainty_3var(
            sample,
            blank_samples[0],
            num_isotope=num_isotope,
            den_isotope=den_isotope,
            aux_isotope=aux_isotope,
            num_corrected_mean=num_corrected_mean,
            den_corrected_mean=den_corrected_mean,
            aux_corrected_mean=aux_corrected_mean,
            element_config=element_config,
            uncertainty_config=uncertainty_config,
            processing_config=processing_config,
            cycle_ranges=cycle_ranges,
            channel_weight_vector=weights_before,
        )
        result_after = _compute_single_blank_uncertainty_3var(
            sample,
            blank_samples[-1],
            num_isotope=num_isotope,
            den_isotope=den_isotope,
            aux_isotope=aux_isotope,
            num_corrected_mean=num_corrected_mean,
            den_corrected_mean=den_corrected_mean,
            aux_corrected_mean=aux_corrected_mean,
            element_config=element_config,
            uncertainty_config=uncertainty_config,
            processing_config=processing_config,
            cycle_ranges=cycle_ranges,
            channel_weight_vector=weights_after,
        )
        if result_before is None or result_after is None:
            return None

        # The producer's per-channel weights are already inside each side's
        # quadrature, so the two independent blanks combine directly.
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

        combined_corr = None
        if (
            result_before.correlation_matrix is not None
            and result_after.correlation_matrix is not None
        ):
            _m_before = np.clip(
                np.asarray(result_before.correlation_matrix, dtype=np.float64),
                -0.9999, 0.9999,
            )
            _m_after = np.clip(
                np.asarray(result_after.correlation_matrix, dtype=np.float64),
                -0.9999, 0.9999,
            )
            combined_corr = np.tanh((np.arctanh(_m_before) + np.arctanh(_m_after)) / 2.0)
            np.fill_diagonal(combined_corr, 1.0)

        _r_before = np.clip(result_before.correlation, -0.9999, 0.9999)
        _r_after = np.clip(result_after.correlation, -0.9999, 0.9999)
        _r_combined = float(np.tanh(
            (np.arctanh(_r_before) + np.arctanh(_r_after)) / 2.0
        ))

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
            u_aux_sd=_weighted_channel_mean(
                (result_before.u_aux_sd, weights_before[2]),
                (result_after.u_aux_sd, weights_after[2]),
            ),
            u_num_input=_weighted_channel_mean(
                (result_before.u_num_input, weights_before[0]),
                (result_after.u_num_input, weights_after[0]),
            ),
            u_den_input=_weighted_channel_mean(
                (result_before.u_den_input, weights_before[1]),
                (result_after.u_den_input, weights_after[1]),
            ),
            u_aux_input=_weighted_channel_mean(
                (result_before.u_aux_input, weights_before[2]),
                (result_after.u_aux_input, weights_after[2]),
            ),
            correlation=_r_combined,
            degrees_of_freedom=dof,
            num_corrected_mean=num_corrected_mean,
            den_corrected_mean=den_corrected_mean,
            blank_mode="before_and_after",
            n_blanks_used=sum(
                1 for w in (weights_before, weights_after) if float(np.max(w)) > 0.0
            ),
            n_blank_cycles=_contributing_blank_cycles(
                (result_before, tuple(weights_before)),
                (result_after, tuple(weights_after)),
            ),
            blank_uncertainty_input=result_before.blank_uncertainty_input,
            correlation_warning=" | ".join(warning_parts),
            model_dimension=3,
            aux_isotope=aux_isotope,
            aux_corrected_mean=aux_corrected_mean,
            correlation_labels=(num_isotope, den_isotope, aux_isotope),
            correlation_matrix=combined_corr,
            per_blank_results=[result_before, result_after],
        )

    result = _compute_single_blank_uncertainty_3var(
        sample,
        blank_samples[0],
        num_isotope=num_isotope,
        den_isotope=den_isotope,
        aux_isotope=aux_isotope,
        num_corrected_mean=num_corrected_mean,
        den_corrected_mean=den_corrected_mean,
        aux_corrected_mean=aux_corrected_mean,
        element_config=element_config,
        uncertainty_config=uncertainty_config,
        processing_config=processing_config,
        cycle_ranges=cycle_ranges,
    )
    if result is None:
        return None
    result.blank_mode = "single"
    result.n_blanks_used = 1
    return result


def _compute_single_blank_uncertainty_3var(
    sample: Sample,
    blank: Sample,
    *,
    num_isotope: str,
    den_isotope: str,
    aux_isotope: str,
    num_corrected_mean: float,
    den_corrected_mean: float,
    aux_corrected_mean: float,
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    processing_config: ProcessingConfig,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    channel_weight_vector: Optional[np.ndarray] = None,
) -> Optional[BlankUncertaintyResult]:
    """Compute a single-blank Sr 3-variable covariance result.

    *channel_weight_vector* scales the chain sensitivities by the coefficient
    this blank's mean carried in each channel's subtraction; the Kragten
    perturbation step stays at the unweighted blank sigma.
    """
    stats = compute_blank_stats_3var(
        blank,
        (num_isotope, den_isotope, aux_isotope),
        correlation_method=uncertainty_config.blank_correlation_method,
        fixed_r=uncertainty_config.blank_fixed_r,
        blank_uncertainty_input=getattr(
            uncertainty_config,
            "blank_uncertainty_input",
            "sd",
        ),
        cycle_ranges=cycle_ranges,
    )
    if stats is None or not np.isfinite(den_corrected_mean) or den_corrected_mean == 0.0:
        return None

    c_num = _compute_blank_isotope_sensitivity(
        sample,
        ratio_name="87Sr/86Sr",
        perturbed_isotope=num_isotope,
        blank_sd=stats.input_sds[num_isotope],
        element_config=element_config,
        processing_config=processing_config,
        cycle_ranges=cycle_ranges,
    )
    c_den = _compute_blank_isotope_sensitivity(
        sample,
        ratio_name="87Sr/86Sr",
        perturbed_isotope=den_isotope,
        blank_sd=stats.input_sds[den_isotope],
        element_config=element_config,
        processing_config=processing_config,
        cycle_ranges=cycle_ranges,
    )
    c_aux = _compute_blank_isotope_sensitivity(
        sample,
        ratio_name="87Sr/86Sr",
        perturbed_isotope=aux_isotope,
        blank_sd=stats.input_sds[aux_isotope],
        element_config=element_config,
        processing_config=processing_config,
        cycle_ranges=cycle_ranges,
    )
    sensitivities = np.array([c_num, c_den, c_aux], dtype=np.float64)
    if channel_weight_vector is not None:
        sensitivities = sensitivities * np.asarray(
            channel_weight_vector, dtype=np.float64,
        )
    variance = float(
        sensitivities @ stats.covariance_matrix @ sensitivities.T
    )
    sd_vector = np.array(
        [
            stats.input_sds[num_isotope],
            stats.input_sds[den_isotope],
            stats.input_sds[aux_isotope],
        ],
        dtype=np.float64,
    )
    variance_uncorrelated = float(np.sum((sensitivities * sd_vector) ** 2))
    correlation_term_abs2 = variance - variance_uncorrelated

    return BlankUncertaintyResult(
        u_blank_abs=float(np.sqrt(max(variance, 0.0))),
        u_blank_uncorrelated_abs=float(np.sqrt(max(variance_uncorrelated, 0.0))),
        u_blank_correlation_term_abs2=correlation_term_abs2,
        u_num_sd=stats.sds[num_isotope],
        u_den_sd=stats.sds[den_isotope],
        u_aux_sd=stats.sds[aux_isotope],
        u_num_input=stats.input_sds[num_isotope],
        u_den_input=stats.input_sds[den_isotope],
        u_aux_input=stats.input_sds[aux_isotope],
        correlation=float(stats.applied_correlation_matrix[0, 1]),
        degrees_of_freedom=stats.degrees_of_freedom,
        num_corrected_mean=num_corrected_mean,
        den_corrected_mean=den_corrected_mean,
        blank_mode="single",
        n_blanks_used=1,
        n_blank_cycles=stats.n_pairs,
        blank_uncertainty_input=stats.blank_uncertainty_input,
        correlation_warning=stats.warning,
        model_dimension=3,
        aux_isotope=aux_isotope,
        aux_corrected_mean=aux_corrected_mean,
        correlation_labels=stats.isotopes,
        correlation_matrix=stats.applied_correlation_matrix,
    )


def _channel_weight_vector(
    channel_weights: Optional[Dict[str, Dict[str, float]]],
    role: str,
    isotopes: Tuple[str, ...],
) -> np.ndarray:
    """Per-channel weights for one bracketing role, defaulting to equal halves."""
    per_role = (channel_weights or {}).get(role) or {}
    return np.array(
        [float(per_role.get(isotope, 0.5)) for isotope in isotopes],
        dtype=np.float64,
    )


def _get_engine_a_chain_source(sample: Sample):
    """Return the intensity layer used to replay the Sr correction chain."""
    if sample.blank_corrected_intensities:
        return sample.blank_corrected_intensities
    if sample.corrected_intensities:
        return sample.corrected_intensities
    return sample.intensities

def _resolve_active_sr_normalization_pair(
    processing_config: ProcessingConfig,
    element_config: Optional[ElementConfig],
) -> Tuple[str, str, float, float]:
    """Resolve the active Sr normalization pair and its managed masses.

    A039/D1: Engine A and the Monte Carlo chain replay both resolve the pair
    here, so the supported-envelope check lives here too.  An unsupported pair
    raises instead of quietly resolving to masses the replay would then model
    differently from production.
    """
    ratio_name = (
        processing_config.normalization_ratio_override
        or (element_config.normalization_ratio if element_config is not None else None)
        or "86Sr/88Sr"
    )
    normalized = str(ratio_name).strip().replace("\\", "/").replace("_", "/")
    if normalized.count("/") != 1:
        raise ValueError(f"Invalid Sr normalization ratio: {ratio_name!r}")
    if element_config is not None:
        require_supported_sr_normalization_ratio(element_config, normalized)
    numerator, denominator = normalized.split("/", 1)
    numerator = _normalize_ratio_token(numerator)
    denominator = _normalize_ratio_token(denominator)
    masses = get_masses_for_isotopes((numerator, denominator))
    return numerator, denominator, masses[numerator], masses[denominator]


def _resolve_active_sr_normalization_value(
    processing_config: ProcessingConfig,
    element_config: Optional[ElementConfig],
) -> Optional[float]:
    """Resolve the active Sr normalization value for the selected ratio."""
    if processing_config.normalization_value_override is not None:
        return processing_config.normalization_value_override

    numerator, denominator, _m_num, _m_den = _resolve_active_sr_normalization_pair(
        processing_config,
        element_config,
    )
    ratio_name = f"{numerator}/{denominator}"
    element_default_ratio = (
        str(element_config.normalization_ratio).strip().replace("\\", "/").replace("_", "/")
        if element_config is not None and element_config.normalization_ratio is not None
        else None
    )
    if (
        element_config is not None
        and element_default_ratio is not None
        and ratio_name == f"{_normalize_ratio_token(element_default_ratio.split('/', 1)[0])}/{_normalize_ratio_token(element_default_ratio.split('/', 1)[1])}"
        and element_config.normalization_value is not None
    ):
        return element_config.normalization_value

    norm_element = re.match(r"^\d+([A-Za-z]+)$", numerator)
    den_element = re.match(r"^\d+([A-Za-z]+)$", denominator)
    if norm_element is None or den_element is None or norm_element.group(1) != den_element.group(1):
        return element_config.normalization_value if element_config is not None else None

    element_symbol = norm_element.group(1).capitalize()
    internal_norm = get_internal_normalization(element_symbol)
    if internal_norm is not None:
        managed_ratio_name, managed_value = internal_norm
        normalized_managed = str(managed_ratio_name).strip().replace("\\", "/").replace("_", "/")
        if normalized_managed.count("/") == 1:
            managed_num, managed_den = normalized_managed.split("/", 1)
            normalized_managed = (
                f"{_normalize_ratio_token(managed_num)}/{_normalize_ratio_token(managed_den)}"
            )
        if normalized_managed == ratio_name:
            return managed_value

    natural_ratio = get_natural_ratio(element_symbol, ratio_name)
    if natural_ratio is not None:
        return natural_ratio[0]

    return element_config.normalization_value if element_config is not None else None


def _resolve_sr_interference_reference_inputs(
    processing_config: ProcessingConfig,
    element_config: Optional[ElementConfig],
    sample: Optional[Sample] = None,
) -> SrInterferenceReferenceInputs:
    """Resolve managed Sr/Rb/Kr reference data and the Sr method for Engine A replay.

    The method is the one ``sample`` was processed with, checked against the
    active element configuration (see ``sr_chain_identity``); without a sample it
    is the configured method.
    """
    norm_num, norm_den, m_norm_num, m_norm_den = _resolve_active_sr_normalization_pair(
        processing_config,
        element_config,
    )
    return SrInterferenceReferenceInputs(
        rb87_rb85=require_natural_ratio("Rb", "87Rb/85Rb")[0],
        kr84_kr83=require_natural_ratio("Kr", "84Kr/83Kr")[0],
        kr86_kr83=require_natural_ratio("Kr", "86Kr/83Kr")[0],
        # The nominal Rb ratio remains part of the correction. Its uncertainty
        # is optional: an unassigned value omits only the Rb Type-B term.
        u_rel_rb=get_natural_ratio_relative_uncertainty("Rb", "87Rb/85Rb"),
        u_rel_kr84=require_natural_ratio_relative_uncertainty("Kr", "84Kr/83Kr"),
        u_rel_kr86=require_natural_ratio_relative_uncertainty("Kr", "86Kr/83Kr"),
        normalization_numerator=norm_num,
        normalization_denominator=norm_den,
        m_norm_num=m_norm_num,
        m_norm_den=m_norm_den,
        m86_sr=require_isotope_mass("86Sr"),
        m87_sr=require_isotope_mass("87Sr"),
        m88_sr=require_isotope_mass("88Sr"),
        m83_kr=require_isotope_mass("83Kr"),
        m84_kr=require_isotope_mass("84Kr"),
        m86_kr=require_isotope_mass("86Kr"),
        m85_rb=require_isotope_mass("85Rb"),
        m87_rb=require_isotope_mass("87Rb"),
        chain_method=replay_sr_chain_method(element_config, sample),
    )


def _enabled_sr_interferents(
    processing_config: ProcessingConfig,
    element_config: Optional[ElementConfig],
) -> Set[str]:
    """Return effectively enabled Sr interferents for chain replay."""
    if not processing_config.apply_interference_correction or element_config is None:
        return set()
    return {
        spec.interfering_isotope
        for spec in getattr(element_config, "monitors", ()) or ()
        if (
            getattr(spec, "family", "f") == "f"
            and processing_config.is_monitor_enabled(spec.interfering_isotope)
        )
    }


def _sr_interference_component_label(enabled_interferents: Set[str]) -> str:
    """Return the Sr u_interf component label for the 87Sr/86Sr budget."""
    components: List[str] = []
    if "87Rb" in enabled_interferents:
        components.append("Rb")
    if "86Kr" in enabled_interferents:
        components.append("Kr")
    return " + ".join(components)


def _sr_interference_display_name(enabled_interferents: Set[str]) -> str:
    """Return a display name that matches the active 87Sr/86Sr interferents."""
    component_label = _sr_interference_component_label(enabled_interferents)
    if not component_label:
        return "Isobaric interference (Type B)"
    return f"Isobaric interference (Type B, {component_label})"


def _sr_interference_description(enabled_interferents: Set[str]) -> str:
    """Return a u_interf description that does not mention disabled monitors."""
    component_label = _sr_interference_component_label(enabled_interferents)
    if component_label == "Rb":
        ratio_text = "the Rb natural-abundance ratio"
    elif component_label == "Kr":
        ratio_text = "the Kr natural-abundance ratio"
    elif component_label:
        ratio_text = "the Rb and Kr natural-abundance ratios"
    else:
        ratio_text = "enabled Sr isobaric-interference natural-abundance ratios"
    return (
        f"Numerical perturbation of {ratio_text} through the Sr correction chain "
        "(natural-ratio initialization plus two K-factor refinements)."
    )


def _compute_blank_isotope_sensitivity(
    sample: Sample,
    *,
    ratio_name: str,
    perturbed_isotope: str,
    blank_sd: float,
    element_config: Optional[ElementConfig],
    processing_config: ProcessingConfig,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> float:
    """Return signed dR/dB for one Sr blank mass via chain perturbation."""
    if blank_sd == 0.0:
        return 0.0
    if not np.isfinite(blank_sd) or blank_sd < 0:
        raise ValueError("Invalid Sr blank sigma.")

    normalization_value = _resolve_active_sr_normalization_value(
        processing_config,
        element_config,
    )
    if normalization_value is None or normalization_value <= 0:
        raise ValueError("Required Sr blank chain input unavailable.")

    src = _get_engine_a_chain_source(sample)
    reference_inputs = _resolve_sr_interference_reference_inputs(
        processing_config,
        element_config,
        sample,
    )
    enabled_interferents = _enabled_sr_interferents(processing_config, element_config)
    required = [
        "87Sr",
        "86Sr",
        reference_inputs.normalization_numerator,
        reference_inputs.normalization_denominator,
    ]
    if "87Rb" in enabled_interferents and "85Rb" in src:
        required.append("85Rb")
    if "86Kr" in enabled_interferents and "83Kr" in src:
        required.append("83Kr")
    if any(isotope not in src for isotope in required):
        raise ValueError("Required Sr blank chain input unavailable.")

    intensities: Dict[str, np.ndarray] = {}
    replay_isotopes = {
        "87Sr",
        "86Sr",
        "85Rb",
        "83Kr",
        "84Sr",
        reference_inputs.normalization_numerator,
        reference_inputs.normalization_denominator,
    }
    for isotope in replay_isotopes:
        cycle_data = src.get(isotope)
        if cycle_data is not None:
            intensities[isotope] = cycle_data.values.copy()

    if perturbed_isotope not in intensities:
        raise ValueError("Required Sr blank chain input unavailable.")

    mask = _get_runtime_ratio_mask(sample, ratio_name, cycle_ranges=cycle_ranges)

    intensities_up = {
        isotope: values.copy()
        for isotope, values in intensities.items()
    }
    intensities_down = {
        isotope: values.copy()
        for isotope, values in intensities.items()
    }
    intensities_up[perturbed_isotope] = intensities_up[perturbed_isotope] - blank_sd
    intensities_down[perturbed_isotope] = intensities_down[perturbed_isotope] + blank_sd

    common = dict(
        normalization_value=float(normalization_value),
        mask=mask,
        reference_inputs=reference_inputs,
        apply_iif=processing_config.apply_mass_bias_correction,
        apply_interference=processing_config.apply_interference_correction,
        enabled_interferents=enabled_interferents,
    )
    r_up = _run_sr_correction_chain(
        intensities=intensities_up,
        **common,
    )
    r_down = _run_sr_correction_chain(
        intensities=intensities_down,
        **common,
    )
    if (r_up == 0.0 and r_down == 0.0) or not np.isfinite(r_up) or not np.isfinite(r_down):
        raise ValueError("Sr blank perturbation invalid on selected support.")

    return (r_up - r_down) / (2.0 * blank_sd)


def _resolve_norm_ratio_u_abs(
    uncertainty_config: UncertaintyConfig,
    normalization_value: float,
) -> float:
    """Return absolute standard u of the active normalization ratio.

    The visible input is absolute, and an absolute zero is honoured as zero.
    The relative permil field is a legacy input that only old sessions carry;
    it is converted when it is actually *present*, never as a class default, so
    a config the user built today cannot inherit a magnitude nobody entered.
    """
    try:
        u_abs = float(getattr(uncertainty_config, "sr_norm_ratio_u_abs", 0.0) or 0.0)
    except (TypeError, ValueError):
        u_abs = 0.0
    if u_abs > 0.0:
        return u_abs

    legacy_permil = getattr(uncertainty_config, "sr_norm_ratio_u_permil", None)
    if legacy_permil is None:
        return 0.0
    try:
        u_permil = float(legacy_permil)
    except (TypeError, ValueError):
        return 0.0
    if not (u_permil > 0.0):
        return 0.0
    return abs(float(normalization_value)) * u_permil / 1000.0


def _compute_norm_ratio_uncertainty(
    sample: Sample,
    *,
    ratio_name: str,
    ratio_mean: float,
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    processing_config: ProcessingConfig,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Tuple[float, float]:
    """Propagate accepted normalization-ratio uncertainty through the Sr chain."""
    if ratio_name != "87Sr/86Sr" or not processing_config.apply_mass_bias_correction:
        return 0.0, 0.0

    norm_value = _resolve_active_sr_normalization_value(
        processing_config,
        element_config,
    )
    if norm_value is None or norm_value <= 0.0:
        raise ValueError("Sr normalization perturbation invalid on selected support.")

    u_norm_abs = _resolve_norm_ratio_u_abs(uncertainty_config, float(norm_value))
    if u_norm_abs == 0:
        return 0.0, 0.0
    if not np.isfinite(u_norm_abs) or u_norm_abs < 0 or u_norm_abs >= float(norm_value):
        raise ValueError("Sr reference perturbation leaves the positive domain.")

    src = _get_engine_a_chain_source(sample)
    reference_inputs = _resolve_sr_interference_reference_inputs(
        processing_config,
        element_config,
        sample,
    )
    enabled_interferents = _enabled_sr_interferents(processing_config, element_config)
    required = {
        "87Sr",
        "86Sr",
        reference_inputs.normalization_numerator,
        reference_inputs.normalization_denominator,
    }
    if not required.issubset(src.keys()):
        raise ValueError("Required Sr normalization replay channels are unavailable.")

    intensities: Dict[str, np.ndarray] = {}
    replay_isotopes = {
        "87Sr",
        "86Sr",
        "85Rb",
        "83Kr",
        "84Sr",
        reference_inputs.normalization_numerator,
        reference_inputs.normalization_denominator,
    }
    for isotope in replay_isotopes:
        cd = src.get(isotope)
        if cd is not None:
            intensities[isotope] = cd.values.copy()

    mask = _get_runtime_ratio_mask(
        sample,
        ratio_name,
        cycle_ranges=cycle_ranges,
    )
    common = dict(
        intensities=intensities,
        mask=mask,
        reference_inputs=reference_inputs,
        apply_iif=True,
        apply_interference=processing_config.apply_interference_correction,
        enabled_interferents=enabled_interferents,
    )
    r_up = _run_sr_correction_chain(
        normalization_value=float(norm_value) + u_norm_abs,
        **common,
    )
    r_down = _run_sr_correction_chain(
        normalization_value=float(norm_value) - u_norm_abs,
        **common,
    )
    if not np.isfinite(r_up) or not np.isfinite(r_down):
        raise ValueError("Sr normalization perturbation failed on selected support.")

    u_abs = abs(r_up - r_down) / 2.0
    u_rel = (u_abs / ratio_mean) * 1000.0 if ratio_mean and u_abs > 0.0 else 0.0
    return float(u_abs), float(u_rel)


def _compute_interference(
    sample: Sample,
    *,
    ratio_name: str,
    element_config: ElementConfig,
    processing_config: ProcessingConfig,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> InterferenceUncertainty:
    """Compute interference uncertainty via Kragten perturbation."""
    if ratio_name != "87Sr/86Sr":
        return InterferenceUncertainty(
            u_rb_abs=0.0, u_rb_rel_permil=0.0,
            u_kr86_abs=0.0, u_kr86_rel_permil=0.0,
            u_kr84_abs=0.0, u_kr84_rel_permil=0.0,
            u_interf_abs=0.0, u_interf_rel_permil=0.0,
        )

    if not processing_config.apply_interference_correction:
        return InterferenceUncertainty(
            u_rb_abs=0.0, u_rb_rel_permil=0.0,
            u_kr86_abs=0.0, u_kr86_rel_permil=0.0,
            u_kr84_abs=0.0, u_kr84_rel_permil=0.0,
            u_interf_abs=0.0, u_interf_rel_permil=0.0,
        )

    # Resolve normalization value
    norm_value = _resolve_active_sr_normalization_value(
        processing_config,
        element_config,
    )
    if norm_value is None:
        # Cannot compute interference without normalization value
        return InterferenceUncertainty(
            u_rb_abs=0.0, u_rb_rel_permil=0.0,
            u_kr86_abs=0.0, u_kr86_rel_permil=0.0,
            u_kr84_abs=0.0, u_kr84_rel_permil=0.0,
            u_interf_abs=0.0, u_interf_rel_permil=0.0,
        )

    # Extract intensity arrays from blank-corrected (or raw) intensities
    src = _get_engine_a_chain_source(sample)
    reference_inputs = _resolve_sr_interference_reference_inputs(
        processing_config,
        element_config,
        sample,
    )
    enabled_interferents = _enabled_sr_interferents(processing_config, element_config)
    if not enabled_interferents:
        return InterferenceUncertainty(
            u_rb_abs=0.0, u_rb_rel_permil=0.0,
            u_kr86_abs=0.0, u_kr86_rel_permil=0.0,
            u_kr84_abs=0.0, u_kr84_rel_permil=0.0,
            u_interf_abs=0.0, u_interf_rel_permil=0.0,
        )
    required = [
        "87Sr",
        "86Sr",
        reference_inputs.normalization_numerator,
        reference_inputs.normalization_denominator,
    ]
    if "87Rb" in enabled_interferents:
        required.append("85Rb")
    if "86Kr" in enabled_interferents:
        required.append("83Kr")
    for iso in required:
        if iso not in src:
            return InterferenceUncertainty(
                u_rb_abs=0.0, u_rb_rel_permil=0.0,
                u_kr86_abs=0.0, u_kr86_rel_permil=0.0,
                u_kr84_abs=0.0, u_kr84_rel_permil=0.0,
                u_interf_abs=0.0, u_interf_rel_permil=0.0,
            )

    intensities: Dict[str, np.ndarray] = {}
    replay_isotopes = {
        "87Sr",
        "86Sr",
        "85Rb",
        "83Kr",
        "84Sr",
        reference_inputs.normalization_numerator,
        reference_inputs.normalization_denominator,
    }
    for iso in replay_isotopes:
        cd = src.get(iso)
        if cd is not None:
            intensities[iso] = cd.values.copy()

    # Build a mask from the ratio data (use corrected_ratios mask if available)
    mask = _get_runtime_ratio_mask(sample, ratio_name, cycle_ranges=cycle_ranges)
    if any(values.shape != mask.shape for values in intensities.values()):
        raise ValueError("Sr interference replay requires aligned intensity and selected-mask lengths.")

    return compute_interference_uncertainty(
        intensities=intensities,
        normalization_value=norm_value,
        mask=mask,
        reference_inputs=reference_inputs,
        apply_iif=processing_config.apply_mass_bias_correction,
        enabled_interferents=enabled_interferents,
    )


def _get_corrected_intensity_mean(
    sample: Sample,
    isotope: str,
    *,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> float:
    """Get mean blank-corrected intensity for an isotope."""
    src = _get_engine_a_chain_source(sample)
    cd = src.get(isotope)
    if cd is not None and cd.n_valid > 0:
        valid_values = get_filtered_values(
            cd.values,
            cd.mask,
            sample.name,
            cycle_ranges=cycle_ranges,
            sample_key=sample_cycle_key(sample),
            filter_method="None",
            filter_threshold=2.0,
        )
        if len(valid_values) > 0:
            return float(np.mean(valid_values))
    return 0.0


def _get_iif_or_best_ratio(sample: Sample, ratio_name: str):
    """Return IIF-corrected ratio if available, else best available."""
    if sample.iif_corrected_ratios and ratio_name in sample.iif_corrected_ratios:
        return sample.iif_corrected_ratios[ratio_name]
    return get_best_ratio_data(sample, ratio_name)


def _get_pre_anchor_ratio(sample: Sample, ratio_name: str):
    """Return the pre-calibration basis of explicitly selected standards."""
    selected = sample.metadata.get("_sr_calibration_standard_ids")
    if (selected is not None and sample.is_standard
            and sample.metadata.get("_sr_anchor_ratio_name") == ratio_name
            and sample.observation_id not in selected):
        return None
    if sample.metadata.get("_pre_anchor_ratio_name") == ratio_name:
        pre_anchor = sample.metadata.get("_pre_anchor_ratio")
        if pre_anchor is not None:
            return pre_anchor
    return _get_iif_or_best_ratio(sample, ratio_name)


def _get_runtime_ratio_mask(
    sample: Sample,
    ratio_name: str,
    *,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[np.ndarray]:
    """Return a ratio mask aligned to sample cycle arrays."""
    ratio_cd = get_best_ratio_data(sample, ratio_name)
    if ratio_cd is None:
        return None

    mask = ratio_cd.mask.copy()
    cycle_range = (
        resolve_cycle_range(
            cycle_ranges,
            sample_name=sample.name,
            sample_key=sample_cycle_key(sample),
        )
        if cycle_ranges
        else None
    )
    if cycle_range is not None:
        start, end = cycle_range
        start_idx = max(int(start) - 1, 0)
        end_idx = min(int(end), len(mask))
        range_mask = np.zeros(len(mask), dtype=bool)
        if end_idx > start_idx:
            range_mask[start_idx:end_idx] = True
        mask &= range_mask

    return mask


def _compute_crm(
    certified_value: Optional[CertifiedValue],
    uncertainty_config: UncertaintyConfig,
    ratio_mean: float,
) -> Tuple[float, float]:
    """Compute u_CRM.  Returns (u_abs, u_rel_permil).

    Zero for delta mode (cancels in delta calculation).
    """
    if uncertainty_config.output_mode == "delta":
        return 0.0, 0.0

    if certified_value is None or certified_value.uncertainty <= 0:
        return 0.0, 0.0

    u_std = u_certified_value(certified_value.uncertainty, certified_value.k)
    u_rel_permil = (u_std / ratio_mean) * 1000.0 if ratio_mean else 0.0

    return u_std, u_rel_permil


def _compute_ref_value(
    ref_certified_value: Optional[CertifiedValue],
    uncertainty_config: UncertaintyConfig,
    ratio_mean: float,
) -> Tuple[float, float]:
    """Compute u_ref_value.  Returns (u_abs, u_rel_permil).

    Static Type B term for the accepted literature reference ratio (e.g.
    GeoReM), structurally mirroring :func:`_compute_crm`. Zero for delta mode
    for parity, though Sr/Engine A never runs in delta mode in practice.
    """
    if uncertainty_config.output_mode == "delta":
        return 0.0, 0.0

    if ref_certified_value is None or ref_certified_value.uncertainty <= 0:
        return 0.0, 0.0

    u_std = u_certified_value(ref_certified_value.uncertainty, ref_certified_value.k)
    u_rel_permil = (u_std / ratio_mean) * 1000.0 if ratio_mean else 0.0

    return u_std, u_rel_permil


def _resolve_reference_value(
    *,
    certified_value: Optional[CertifiedValue],
    element_config: ElementConfig,
    ratio_name: str,
) -> Optional[float]:
    """Resolve the accepted reference value for the active ratio."""
    if certified_value is not None:
        try:
            value = float(certified_value.value)
            if np.isfinite(value):
                return value
        except (TypeError, ValueError, OverflowError) as exc:
            _LOG.debug("Ignoring invalid certified reference override: %s", exc)
    payload = element_config.certified_values.get(ratio_name)
    if payload is None:
        return None
    try:
        value = float(payload.value)
    except (TypeError, ValueError, OverflowError) as exc:
        _LOG.debug("Ignoring invalid configured certified reference: %s", exc)
        return None
    return value if np.isfinite(value) else None


#: The automatic reference-bias model derived a
#: standard uncertainty from the observed session offset by treating
#: |delta_ref| as the half-width of a rectangular distribution:
#: u = |delta_ref| / sqrt(3). Treating an observed bias that way is a
#: modelling assumption, not a GUM consequence, and no metrologist has
#: approved it for V1. It is therefore disabled: the offset is still computed
#: and displayed as a diagnostic, but it produces no uncertainty contribution.
#: u_bias_qc is a different contributor and stays available - it carries a
#: standard uncertainty the user supplies, rather than one this code invents.
AUTOMATIC_REFERENCE_BIAS_ENABLED = False

AUTOMATIC_REFERENCE_BIAS_DISABLED_REASON = (
    "u_bias_ref is disabled in this release. The automatic model treated the "
    "observed reference offset as rectangular (u = |delta_ref| / sqrt(3)), "
    "which is an unapproved modelling assumption rather than a GUM "
    "consequence. The offset is reported as a diagnostic only and contributes "
    "no uncertainty. To include an observed bias, supply a standard "
    "uncertainty through u_bias_qc."
)


def compute_reference_bias_term(
    *,
    reprod_result: Optional[ReprodResult],
    certified_value: Optional[CertifiedValue],
    element_config: ElementConfig,
    ratio_name: str,
    ratio_mean: float,
    reference_value_override: Optional[float] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """Return the reference-offset diagnostic for Sr Engine A.

    The returned magnitudes are always ``0.0`` while
    :data:`AUTOMATIC_REFERENCE_BIAS_ENABLED` is false. The statistics -
    included-standard count, session mean, reference value and ``delta_ref`` -
    are still computed, because the observed offset is a useful diagnostic.
    What is withdrawn is the claim that it can be turned into a standard
    uncertainty without an approved model.
    """
    try:
        override_value = float(reference_value_override)
    except (TypeError, ValueError):
        override_value = 0.0
    if np.isfinite(override_value) and override_value > 0.0:
        ref_value = override_value
    else:
        ref_value = _resolve_reference_value(
            certified_value=certified_value,
            element_config=element_config,
            ratio_name=ratio_name,
        )
    stats: Dict[str, float] = {
        "n_included": 0.0,
        "session_mean": 0.0,
        "reference_value": float(ref_value) if ref_value is not None else 0.0,
        "delta_ref": 0.0,
    }
    if reprod_result is None or ref_value is None:
        return 0.0, 0.0, stats
    if len(reprod_result.std_means) == 0 or len(reprod_result.std_included) != len(reprod_result.std_means):
        return 0.0, 0.0, stats

    included_mask = np.asarray(reprod_result.std_included, dtype=bool)
    included_means = np.asarray(reprod_result.std_means, dtype=np.float64)[included_mask]
    included_means = included_means[np.isfinite(included_means)]
    if len(included_means) == 0:
        return 0.0, 0.0, stats

    session_mean = float(np.mean(included_means))
    delta_ref = float(session_mean - ref_value)
    if AUTOMATIC_REFERENCE_BIAS_ENABLED:  # pragma: no cover - disabled for V1
        u_abs = abs(delta_ref) / np.sqrt(3.0)
        # Relative figure is expressed on the reference-standard ratio basis
        # (the value the bias was measured against), not the sample mean.
        # Engine A combines in absolute units, so this affects display/export
        # only.
        u_rel_permil = (u_abs / ref_value) * 1000.0 if ref_value else 0.0
    else:
        u_abs = 0.0
        u_rel_permil = 0.0
    stats.update(
        {
            "n_included": float(len(included_means)),
            "session_mean": session_mean,
            "reference_value": float(ref_value),
            "delta_ref": delta_ref,
        }
    )
    return u_abs, u_rel_permil, stats


def compute_qc_bias_term(
    *,
    observed_bias_abs: float,
    ratio_mean: float,
    qc_cert_value: Optional[float] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """Return the optional QC-bias Type B term for Sr Engine A.

    ``observed_bias_abs`` is treated as the user-supplied standard uncertainty
    numerator on the QC material's absolute ratio scale. The user applies any
    distribution conversion before entering it. The certified QC ratio converts
    that numerator to the relative standard uncertainty used in the square-sum
    budget:

        u_rel = u_QC_input / R_QC_cert

    Engine A stores contributors in absolute ratio units, so the same relative
    term is also expressed on the active sample basis:

        u_abs = u_QC_input * (ratio_mean / R_QC_cert)

    Without a positive certified ratio, the contributor is unavailable rather
    than divided by the active sample ratio.
    """
    try:
        qc_input_abs = abs(float(observed_bias_abs))
    except (TypeError, ValueError):
        qc_input_abs = 0.0

    try:
        cert = float(qc_cert_value) if qc_cert_value is not None else 0.0
    except (TypeError, ValueError):
        cert = 0.0
    denominator = cert if np.isfinite(cert) and cert > 0.0 else None

    if qc_input_abs > 0.0 and denominator is not None:
        # Fractional bias of the QC material, transferred to the sample basis.
        u_rel_permil = (qc_input_abs / denominator) * 1000.0
        u_abs = (u_rel_permil / 1000.0) * ratio_mean if ratio_mean else 0.0
    else:
        u_abs = 0.0
        u_rel_permil = 0.0

    return u_abs, u_rel_permil, {
        "delta_qc": qc_input_abs,
        "qc_input_abs": qc_input_abs,
        "qc_cert_value": float(denominator) if denominator is not None else 0.0,
    }


def compute_digestion_reproducibility_term(
    *,
    digestion_sd_abs: float,
    ratio_mean: float,
    digestion_ref_value: Optional[float] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """Return the optional between-digestion reproducibility term for Sr Engine A.

    ``digestion_sd_abs`` is the SD of independently processed digestion means.
    When ``digestion_ref_value > 0`` is supplied, the SD is interpreted on that
    processed material's ratio basis and transferred fractionally to the active
    sample:

        u_abs = (SD_dig / R_dig_ref) x R_sample

    Without a positive reference ratio, the contributor is unavailable rather
    than divided by the active sample ratio.
    """
    try:
        sd_abs = abs(float(digestion_sd_abs))
    except (TypeError, ValueError):
        sd_abs = 0.0

    try:
        ref_value = float(digestion_ref_value) if digestion_ref_value is not None else 0.0
    except (TypeError, ValueError):
        ref_value = 0.0
    denominator = ref_value if np.isfinite(ref_value) and ref_value > 0.0 else None

    if sd_abs > 0.0 and denominator is not None:
        u_rel_permil = (sd_abs / denominator) * 1000.0
        u_abs = (u_rel_permil / 1000.0) * ratio_mean if ratio_mean else 0.0
    else:
        u_abs = 0.0
        u_rel_permil = 0.0

    return u_abs, u_rel_permil, {
        "digestion_sd_abs": sd_abs,
        "digestion_ref_value": float(denominator) if denominator is not None else 0.0,
    }


def _combine_and_build_budget(
    *,
    contributors: List[UncertaintyContributor],
    ratio_mean: float,
    n_cycles: int,
    uncertainty_config: UncertaintyConfig,
) -> UncertaintyBudget:
    """RSS combination, Welch-Satterthwaite, and budget assembly."""
    return combine_and_build_budget_shared(
        engine="internal_normalization",
        contributors=contributors,
        ratio_mean=ratio_mean,
        n_cycles=n_cycles,
        uncertainty_config=uncertainty_config,
        use_abs_dof=True,
    )

"""Engine B — SSB-delta uncertainty budget for Li, B, Mg, Cd, Pb."""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
from config.contributor_names import LABEL_U_K4, LABEL_U_PREC, LABEL_U_STD
from config.settings import (
    CustomUncertaintyContributor,
    DEFAULT_K1_SAMPLE_DECOMPOSITION_PERMIL,
    DEFAULT_K2_MATRIX_SEPARATION_PERMIL,
    DEFAULT_K3_PROCEDURAL_BLANK_PERMIL,
    DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_PERMIL,
    DEFAULT_K6_MATRIX_EFFECTS_PERMIL,
    DEFAULT_K7_RESIDUAL_INTERFERENCES_PERMIL,
    ProcessingConfig,
    UncertaintyConfig,
)
from domain.elements.base import CertifiedValue, ElementConfig
from domain.corrections.delta import (
    ClassicDeltaBracketSide,
    resolve_classic_delta_bracket,
    resolve_classic_delta_bracket_sides,
)
from domain.filters.outlier import get_filtered_values, sample_cycle_key
from domain.models import Sample, UncertaintyBudget, UncertaintyContributor
from domain.output_scale import (
    INPUT_LAYER_BLANK_CORRECTED,
    resolve_output_scale,
    scaled_contribution,
)
from domain.ratio_selection import get_best_pre_drift_ratio_data, get_best_ratio_data
from domain.uncertainty.contributors import (
    ContributorState,
    SampleContributorApplicability,
    build_custom_contributor_rows,
    inactive_reason_for_state,
    not_applicable_reason,
    resolve_contributor_state,
)
from domain.uncertainty.reprod import compute_reprod
from domain.uncertainty.blank import (
    BlankUncertaintyResult,
    compute_blank_uncertainty,
    describe_blank_input_model,
    resolve_blank_correction_mode,
    resolve_blank_channel_weights,
    resolve_blank_samples_for_uncertainty,
    resolve_blank_selection,
)
from domain.uncertainty.propagation import (
    select_precision_value,
    u_certified_value,
    u_precision,
    u_ssb_standards,
)
from domain.uncertainty.eligibility import (
    required_contributor_unavailable_budget,
    unresolved_blank_reference_budget,
)
from domain.uncertainty.eligibility import hg_correction_unavailable_budget
from domain.uncertainty.pb_hg_ssb_propagation import (
    BLANK_INVALID_COVARIANCE,
    compute_pb_hg_ssb_propagation,
    hg_blank_row_values,
    hg_propagation_contributor_rows,
)
from domain.uncertainty.shared_engine import combine_and_build_budget_shared
from domain.uncertainty.welch_satterthwaite import effective_dof


def compute_budget_ssb(
    sample: Sample,
    ratio_name: str,
    *,
    all_samples: List[Sample],
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    processing_config: Optional[ProcessingConfig] = None,
    certified_value: Optional[CertifiedValue] = None,
    ratio_values: Optional[np.ndarray] = None,
    ratio_mean: Optional[float] = None,
    runtime_ratio_mask: Optional[np.ndarray] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    drift_model: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    position_extractor: Optional[Callable[[Sample], float]] = None,
    custom_contributor_library: Optional[Dict[str, List[CustomUncertaintyContributor]]] = None,
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
) -> Optional[UncertaintyBudget]:
    """Compute a full Engine B (SSB-delta) uncertainty budget."""
    if sample.is_blank:
        return None

    hg_refusal = hg_correction_unavailable_budget(
        sample, ratio_name, uncertainty_config.output_mode,
    )
    if hg_refusal is not None:
        return hg_refusal

    if ratio_values is None:
        cd = get_best_ratio_data(sample, ratio_name)
        if cd is None:
            return None
        ratio_values = cd.valid_values
    if runtime_ratio_mask is not None:
        runtime_ratio_mask = np.asarray(runtime_ratio_mask, dtype=bool)
        source = get_best_ratio_data(sample, ratio_name)
        if source is None or len(runtime_ratio_mask) != len(source.mask):
            raise ValueError("Runtime ratio support is not aligned to the stored correction layer")
        if np.any(runtime_ratio_mask & ~np.asarray(source.mask, dtype=bool)):
            raise ValueError("Runtime ratio support cannot reintroduce correction-invalid cycles")

    finite_cycle_count = int(np.sum(np.isfinite(ratio_values)))
    if finite_cycle_count < 2:
        return UncertaintyBudget(
            engine="ssb_delta",
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

    u_prec_se_abs, u_prec_sd_abs, mean_val = u_precision(ratio_values)
    u_prec_abs, u_prec_mode = select_precision_value(
        getattr(uncertainty_config, "u_prec_mode", "se"),
        u_prec_se_abs,
        u_prec_sd_abs,
    )
    n_cycles = int(np.sum(np.isfinite(ratio_values)))
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

    u_std_unavailable_reason = _selected_reference_se_unavailable_reason(
        sample, ratio_name
    )
    if _active("u_std") and u_std_unavailable_reason:
        return required_contributor_unavailable_budget(
            engine="ssb_delta",
            output_mode=uncertainty_config.output_mode,
            contributor_name="u_std",
            reason=u_std_unavailable_reason,
            basis_ratio_value=ratio_mean,
            n_cycles=n_cycles,
        )

    u_std_abs, u_std_rel_permil, u_std_dof = _compute_bracketing_standard_uncertainty(
        sample,
        ratio_name,
        all_samples,
        ratio_mean,
    )
    u_std_missing_reason = ""
    if _active("u_std") and u_std_abs <= 0.0:
        if not _has_bracketing_standard(sample, ratio_name, all_samples):
            u_std_missing_reason = (
                f"u_std found no bracketing standard for ratio {ratio_name} "
                f"(no standard before and/or after this sample)."
            )
        else:
            u_std_missing_reason = "u_std missing bracketing standard uncertainty data (prev_std/next_std)."

    if uncertainty_config.enable_ssb or sample.ssb_results.get(ratio_name) is not None or uncertainty_config.enable_delta:
        contributors.append(UncertaintyContributor(
            name="u_std",
            display_name=f"{LABEL_U_STD} (Type A)",
            value_abs=u_std_abs,
            value_rel_permil=u_std_rel_permil,
            type_ab="A",
            degrees_of_freedom=float(u_std_dof) if u_std_dof >= 1 else 1.0,
            percentage_contribution=0.0,
            description=(
                "Local uncertainty of the bracketing standard mean: "
                "u_std = 0.5 * sqrt(SEM_std1^2 + SEM_std2^2)."
            ),
            **_contributor_gate("u_std", u_std_abs > 0, u_std_missing_reason),
        ))

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
            engine="ssb_delta",
            output_mode=uncertainty_config.output_mode,
            contributor_name="u_std_repeatability",
            reason=reprod_result.unavailable_reason or "repeatability inference is unsupported",
            basis_ratio_value=ratio_mean,
            n_cycles=n_cycles,
        )

    # Reproducibility is evaluated on the standard sequence, so its natural
    # denominator is the mean standard ratio, not the sample basis ratio.
    # Convert it into an equivalent uncertainty on the active basis ratio so
    # RSS/MC can keep treating all contributors as additive output terms.
    u_std_repeatability_rel_permil = float(
        getattr(reprod_result, "u_std_repeatability_rel_permil", 0.0) or 0.0
    )
    u_std_repeatability_abs = (
        (u_std_repeatability_rel_permil / 1000.0) * ratio_mean
        if ratio_mean and u_std_repeatability_rel_permil > 0.0 else 0.0
    )
    reprod_dof = reprod_result.degrees_of_freedom

    _METHOD_LABELS = {
        "sd_of_means": "SD of standard means",
        "loo_cross_validation": "LOO cross-validation residuals",
        "drift_residuals": "Drift model residuals",
        "robust_mad": "Robust MAD estimator",
    }
    reprod_label = _METHOD_LABELS.get(reprod_result.method, reprod_result.method)

    u_reprod_missing_reason = ""
    if _active("u_std_repeatability") and u_std_repeatability_abs <= 0.0:
        u_reprod_missing_reason = "u_std_repeatability missing standard repeatability measurements."

    contributors.append(UncertaintyContributor(
        name="u_std_repeatability",
        display_name=f"Standard repeatability (Type A, {reprod_label})",
        value_abs=u_std_repeatability_abs,
        value_rel_permil=u_std_repeatability_rel_permil,
        type_ab="A",
        degrees_of_freedom=float(reprod_dof) if reprod_dof >= 1 else 1.0,
        percentage_contribution=0.0,
        description=(
            f"Scatter of bracketing standards via {reprod_label}. "
            "Full SD, not SEM — represents session-level instrument scatter."
        ),
        **_contributor_gate("u_std_repeatability", u_std_repeatability_abs > 0, u_reprod_missing_reason),
    ))

    # A002 follow-up: a recorded blank reference that cannot be resolved is not
    # a missing contributor, it is an unknown one. Every other blank the session
    # holds is a different measurement, so there is no number to report here and
    # no honest way to leave the term out of a budget that requires it.
    blank_selection = resolve_blank_selection(sample, all_samples)
    if _active("u_blank") and not blank_selection.is_fully_resolved:
        return unresolved_blank_reference_budget(
            sample,
            blank_selection,
            engine="ssb_delta",
            output_mode=uncertainty_config.output_mode,
            basis_ratio_value=ratio_mean,
            n_cycles=n_cycles,
        )

    # An applied ordinary-SSB Hg correction changes the measurement model: the
    # blank now also enters through 202Hg (and Tl for Tl-assisted chains) and
    # through every bracket member, and the Hg and Tl reference ratios become
    # inputs. Those terms come from one linear propagation of the corrected chain.
    hg_propagation = compute_pb_hg_ssb_propagation(
        sample,
        ratio_name,
        all_samples=all_samples,
        uncertainty_config=uncertainty_config,
        processing_config=processing_config,
        cycle_ranges=cycle_ranges,
        runtime_sample_mask=runtime_ratio_mask,
        classic_delta=(
            uncertainty_config.output_mode == "delta"
            and _uses_classic_delta_reference(
                sample,
                ratio_name,
                uncertainty_config=uncertainty_config,
                processing_config=processing_config,
            )
        ),
    )

    if hg_propagation is not None:
        if _active("u_blank") and hg_propagation.blank_status == BLANK_INVALID_COVARIANCE:
            return required_contributor_unavailable_budget(
                engine="ssb_delta",
                output_mode=uncertainty_config.output_mode,
                contributor_name="u_blank",
                reason=f"{hg_propagation.blank_reason}.",
                basis_ratio_value=ratio_mean,
                n_cycles=n_cycles,
            )
        u_blank_abs, u_blank_rel_permil, u_blank_dof, u_blank_missing_reason = hg_blank_row_values(
            hg_propagation, ratio_mean,
        )
        blank_description = (
            "Blank means of every blank subtracted from the sample or a recorded bracket member, "
            "propagated through the Hg-corrected chain (204Pb, 202Hg, the other isotope and, for "
            "Tl-assisted chains, 203Tl/205Tl) with the within-blank channel covariance; each blank "
            "observation is one shared input. "
            + describe_blank_input_model(getattr(uncertainty_config, "blank_uncertainty_input", "sd"))
            + " Reference: JCGM 100:2008 §5.1.3 and §5.2.2."
        )
    else:
        from domain.uncertainty.joint_blank import ordinary_ssb_blank
        try:
            joint = ordinary_ssb_blank(sample, ratio_name, all_samples, uncertainty_config,
                processing_config, cycle_ranges,
                classic_delta=uncertainty_config.output_mode == "delta" and _uses_classic_delta_reference(sample, ratio_name, uncertainty_config=uncertainty_config, processing_config=processing_config),
                runtime_sample_mask=runtime_ratio_mask)
        except ValueError as exc:
            if _active("u_blank"):
                return required_contributor_unavailable_budget(engine="ssb_delta", output_mode=uncertainty_config.output_mode,
                    contributor_name="u_blank", reason=str(exc), basis_ratio_value=ratio_mean, n_cycles=n_cycles)
            joint = None
        u_blank_abs = joint.u_blank_rel * abs(ratio_mean) if joint else 0.0
        u_blank_rel_permil = joint.u_blank_rel * 1000 if joint else 0.0
        u_blank_dof = joint.u_blank_dof if joint else float("inf")
        u_blank_missing_reason = "" if joint and joint.blank_inputs else "not_applicable:no recorded blank subtraction"
        blank_description = "Joint target/bracket blank propagation by observation identity; distinct observations assumed independent. First-order sensitivities of the mean cycle ratio; SD/SE as configured. Reference: JCGM 100:2008 section 5.2." + describe_blank_input_model(uncertainty_config.blank_uncertainty_input, min((b.n_pairs for b in joint.blank_inputs), default=0) if joint else 0)

    blank_not_applicable = ""
    blank_available = u_blank_abs > 0
    if hg_propagation is None and joint is not None and joint.blank_inputs:
        blank_available = True
    if u_blank_missing_reason.startswith("not_applicable:"):
        blank_not_applicable = u_blank_missing_reason.split(":", 1)[1]
        blank_available = True
        u_blank_missing_reason = ""
    elif hg_propagation is not None and hg_propagation.blank_status == "evaluated":
        # A covariance replay that evaluates to finite zero is a result, not
        # missing data.
        blank_available = True
    contributors.append(UncertaintyContributor(
        name="u_blank",
        display_name="Blank correction (Type A)",
        value_abs=u_blank_abs,
        value_rel_permil=u_blank_rel_permil,
        type_ab="A",
        degrees_of_freedom=u_blank_dof,
        percentage_contribution=0.0,
        description=blank_description,
        **_contributor_gate(
            "u_blank", blank_available, u_blank_missing_reason,
            not_applicable_reason=blank_not_applicable,
        ),
    ))

    def _append_fixed_kappa(
        *,
        name: str,
        display_name: str,
        attr_name: str,
        default_value: float,
        description: str,
    ) -> None:
        raw_value = float(getattr(uncertainty_config, attr_name, default_value))
        if not np.isfinite(raw_value) or raw_value < 0.0:
            raise ValueError(
                f"{attr_name} must be a non-negative finite standard uncertainty, "
                f"got {raw_value!r}."
            )
        rel_permil = raw_value
        abs_value = (rel_permil / 1000.0) * ratio_mean if ratio_mean else 0.0
        contributors.append(UncertaintyContributor(
            name=name,
            display_name=display_name,
            value_abs=abs_value,
            value_rel_permil=rel_permil,
            type_ab="B",
            degrees_of_freedom=float("inf"),
            percentage_contribution=0.0,
            description=description,
            **_contributor_gate(name),
        ))

    _append_fixed_kappa(
        name="u_k1_sample_decomposition",
        display_name="Sample decomposition (k1, Type B)",
        attr_name="k1_sample_decomposition_permil",
        default_value=DEFAULT_K1_SAMPLE_DECOMPOSITION_PERMIL,
        description=(
            "Fixed Type B uncertainty for sample decomposition or digestion, "
            "expressed as a standard uncertainty in permil."
        ),
    )
    _append_fixed_kappa(
        name="u_k2_matrix_separation",
        display_name="Matrix separation (k2, Type B)",
        attr_name="k2_matrix_separation_permil",
        default_value=DEFAULT_K2_MATRIX_SEPARATION_PERMIL,
        description=(
            "Fixed Type B uncertainty for chromatographic matrix separation, "
            "expressed as a standard uncertainty in permil."
        ),
    )
    _append_fixed_kappa(
        name="u_k3_procedural_blank",
        display_name="Procedural blank (k3, Type B)",
        attr_name="k3_procedural_blank_permil",
        default_value=DEFAULT_K3_PROCEDURAL_BLANK_PERMIL,
        description=(
            "Fixed Type B uncertainty for procedural blank effects, "
            "expressed as a standard uncertainty in permil."
        ),
    )
    _append_fixed_kappa(
        name="u_k4_bracketing_standard_heterogeneity",
        display_name=f"{LABEL_U_K4} (k4, Type B)",
        attr_name="k4_bracketing_standard_heterogeneity_permil",
        default_value=DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_PERMIL,
        description=(
            "Fixed Type B uncertainty for heterogeneity of the bracketing "
            "standard solution, expressed as a standard uncertainty in permil."
        ),
    )

    drift_requested = (
        uncertainty_config.include_kappa_drift
        and _active("u_k5_instrumental_drift")
        and reprod_result.method != "drift_residuals"
    )
    drift_has_pairs = (
        reprod_result.drift_deltas is not None
        and len(reprod_result.drift_deltas) > 0
    )
    if drift_requested and not drift_has_pairs:
        budget = required_contributor_unavailable_budget(
            engine="ssb_delta", output_mode=uncertainty_config.output_mode,
            contributor_name="u_k5_instrumental_drift", reason="Instrumental drift requires at least one eligible consecutive standard pair within a segment.",
            basis_ratio_value=ratio_mean, ratio_value=ratio_mean, n_cycles=n_cycles,
        )
        budget.reprod_result = reprod_result
        return budget

    if drift_requested and drift_has_pairs:
        kd_abs = (reprod_result.kappa_drift_permil / 1000.0) * ratio_mean
        contributors.append(UncertaintyContributor(
            name="u_k5_instrumental_drift",
            display_name="Instrumental drift (k5, Type B)",
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
            **_contributor_gate("u_k5_instrumental_drift"),
        ))

    _append_fixed_kappa(
        name="u_k6_matrix_effects",
        display_name="Matrix effects (k6, Type B)",
        attr_name="k6_matrix_effects_permil",
        default_value=DEFAULT_K6_MATRIX_EFFECTS_PERMIL,
        description=(
            "Fixed Type B uncertainty for matrix effects affecting mass "
            "discrimination, expressed as a standard uncertainty in permil."
        ),
    )
    _append_fixed_kappa(
        name="u_k7_residual_interferences",
        display_name="Residual interferences (k7, Type B)",
        attr_name="k7_residual_interferences_permil",
        default_value=DEFAULT_K7_RESIDUAL_INTERFERENCES_PERMIL,
        description=(
            "Fixed Type B uncertainty for residual (uncorrected) isobaric or "
            "polyatomic interferences, expressed as a standard uncertainty in permil."
        ),
    )

    u_crm_abs, u_crm_rel_permil = _compute_crm(
        certified_value, uncertainty_config, ratio_mean,
    )

    u_crm_not_applicable = not_applicable_reason(
        "u_crm", output_mode=uncertainty_config.output_mode,
    )
    u_crm_missing_reason = ""
    if _active("u_crm") and u_crm_abs <= 0.0 and not u_crm_not_applicable:
        if certified_value is None:
            u_crm_missing_reason = f"u_crm certified value not found for ratio {ratio_name}."
        else:
            u_crm_missing_reason = f"u_crm certified value uncertainty is zero or missing for ratio {ratio_name}."

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
            u_crm_missing_reason,
            not_applicable_reason=u_crm_not_applicable,
        ),
    ))

    # Inject custom contributor rows (element-filtered inside helper).
    custom_rows = build_custom_contributor_rows(
        sample=sample,
        element_symbol=element_config.symbol,
        ratio_mean=ratio_mean,
        custom_contributor_library=custom_contributor_library or {},
    )
    contributors.extend(custom_rows)

    delta_reference_value = _resolve_delta_reference_value(
        sample=sample,
        ratio_name=ratio_name,
        all_samples=all_samples,
        uncertainty_config=uncertainty_config,
        processing_config=processing_config,
        certified_value=certified_value,
        element_config=element_config,
        cycle_ranges=cycle_ranges,
    )
    certified_reference_value = _resolve_certified_reference_value(
        certified_value=certified_value,
        element_config=element_config,
        ratio_name=ratio_name,
    )
    anchor_absolute_to_certified = (
        uncertainty_config.output_mode == "absolute_ratio"
        and _uses_classic_delta_reference(
            sample,
            ratio_name,
            uncertainty_config=uncertainty_config,
            processing_config=processing_config,
        )
    )
    if anchor_absolute_to_certified and (
        certified_reference_value is None or delta_reference_value is None
    ):
        missing_parts = []
        if delta_reference_value is None:
            missing_parts.append("delta reference value")
        if certified_reference_value is None:
            missing_parts.append("certified reference value")
        missing = " and ".join(missing_parts)
        return _unavailable_budget(
            engine="ssb_delta",
            output_mode=uncertainty_config.output_mode,
            basis_ratio_value=ratio_mean,
            delta_reference_value=delta_reference_value,
            n_cycles=n_cycles,
            scope_note=(
                "Absolute ratio output is unavailable for delta-only bracketing "
                f"because no finite positive {missing} was found."
            ),
        )

    # A011: delta output requires a delta the producer actually computed.  When
    # bracketing failed - or SSB was requested and did not succeed - there is no
    # reference, and the combiner would otherwise fall back to a scale factor of
    # 1 (a reference equal to the sample itself, which makes delta identically
    # zero) or to a certificate the producer explicitly declined to use.  A
    # missing reference is not repaired; it is reported.
    if uncertainty_config.output_mode == "delta" and uncertainty_config.enable_delta:
        produced = (getattr(sample, "delta_results", {}) or {}).get(ratio_name)
        produced_reference = (
            _finite_positive(produced.get("std_mean"))
            if isinstance(produced, dict)
            else None
        )
        if produced_reference is None:
            return _unavailable_budget(
                engine="ssb_delta",
                output_mode=uncertainty_config.output_mode,
                basis_ratio_value=ratio_mean,
                delta_reference_value=delta_reference_value,
                n_cycles=n_cycles,
                scope_note=(
                    f"Delta output is unavailable because no delta result was "
                    f"produced for '{ratio_name}' on this sample, so the delta "
                    "reference is undefined."
                ),
            )
        if delta_reference_value is None:
            return _unavailable_budget(
                engine="ssb_delta",
                output_mode=uncertainty_config.output_mode,
                basis_ratio_value=ratio_mean,
                delta_reference_value=None,
                n_cycles=n_cycles,
                scope_note=(
                    f"Delta output is unavailable because the delta reference for "
                    f"'{ratio_name}' could not be resolved for the current view."
                ),
            )

    if hg_propagation is not None:
        contributors.extend(hg_propagation_contributor_rows(
            hg_propagation, ratio_mean=ratio_mean, gate=_contributor_gate,
        ))

    budget = combine_and_build_budget_shared(
        engine="ssb_delta",
        contributors=contributors,
        ratio_mean=ratio_mean,
        n_cycles=n_cycles,
        uncertainty_config=uncertainty_config,
        delta_reference_value=delta_reference_value,
        certified_reference_value=certified_reference_value,
        anchor_absolute_to_certified=anchor_absolute_to_certified,
    )

    # Attach reprod metadata for UI display
    budget.reprod_result = reprod_result
    if hg_propagation is None and _active("u_blank"):
        budget.coverage_limitations.append({"code": "ordinary_ssb_blank_linearized",
            "explanation": "Blank uncertainty and MC use first-order joint sensitivities with frozen cycle support and bracket weights; nonlinear blank-chain replay and drift refitting are not included. Distinct blank observation IDs are assumed independent."})
    if hg_propagation is not None:
        budget.coverage_limitations.append({
            "code": "pb_ssb_hg_selection_effect_unquantified",
            "explanation": (
                "The effect of selecting or excluding invalid Hg-corrected cycles is not quantified in the "
                "combined uncertainty (Q-26)."
            ),
        })
        if bool(getattr(getattr(processing_config, "drift", None), "enabled", False)):
            budget.coverage_limitations.append({
                "code": "pb_ssb_hg_drift_refit_not_replayed",
                "explanation": (
                    "The uncertainty replay uses the recorded drift correction and does not refit drift for "
                    "perturbed Hg/blank inputs (Q-34)."
                ),
            })

    return budget


# Internal helpers

def _compute_blank_contribution(
    sample: Sample,
    ratio_name: str,
    all_samples: List[Sample],
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    ratio_mean: float,
    *,
    processing_config: Optional[ProcessingConfig] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> BlankUncertaintyResult:
    """Joint blank uncertainty on the supplied reported-ratio scale."""
    from domain.uncertainty.joint_blank import ordinary_ssb_blank
    joint = ordinary_ssb_blank(sample, ratio_name, all_samples, uncertainty_config,
        processing_config, cycle_ranges)
    return BlankUncertaintyResult(u_blank_abs=abs(ratio_mean)*joint.u_blank_rel,
        degrees_of_freedom=joint.u_blank_dof, n_blanks_used=len(joint.blank_inputs),
        n_blank_cycles=min((b.n_pairs for b in joint.blank_inputs), default=0),
        blank_uncertainty_input=uncertainty_config.blank_uncertainty_input)


def _get_corrected_intensity_mean(
    sample: Sample,
    isotope: str,
    *,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    ratio_name: Optional[str] = None,
) -> float:
    """Mean blank-corrected intensity for an isotope on the accepted support.

    When *ratio_name* is given the channel mask is intersected with that
    ratio's accepted cycles, so a cycle the outlier filter rejected from the
    reported ratio cannot re-enter through the blank sensitivity coefficients.
    A channel whose cycle count does not match the ratio series cannot be
    aligned here and keeps its own mask.
    """
    src = sample.corrected_intensities if sample.corrected_intensities else sample.intensities
    cd = src.get(isotope)
    if cd is not None and cd.n_valid > 0:
        mask = np.asarray(cd.mask, dtype=bool)
        if ratio_name is not None:
            ratio_cd = get_best_ratio_data(sample, ratio_name)
            if ratio_cd is not None and len(ratio_cd.mask) == len(mask):
                mask = mask & np.asarray(ratio_cd.mask, dtype=bool)
        valid_values = get_filtered_values(
            cd.values,
            mask,
            sample.name,
            cycle_ranges=cycle_ranges,
            sample_key=sample_cycle_key(sample),
            filter_method="None",
            filter_threshold=2.0,
        )
        if len(valid_values) > 0:
            return float(np.mean(valid_values))
    return 0.0


def _sample_position_index(sample: Sample, all_samples: List[Sample]) -> int:
    """Return *sample*'s index in *all_samples* (identity, then name+run match)."""
    return next(
        (
            i for i, s in enumerate(all_samples)
            if s is sample
            or (s.name == sample.name and s.run_number == sample.run_number)
        ),
        -1,
    )


def _has_bracketing_standard(
    sample: Sample,
    ratio_name: str,
    all_samples: List[Sample],
) -> bool:
    """True if a usable standard exists both before and after *sample* by position."""
    idx = _sample_position_index(sample, all_samples)
    if idx < 0:
        return False

    def _usable(rng) -> bool:
        for j in rng:
            s = all_samples[j]
            if not s.is_standard or s.metadata.get("excluded", False):
                continue
            cd = get_best_ratio_data(s, ratio_name)
            if cd is not None and cd.n_valid > 1:
                return True
        return False

    return _usable(range(idx - 1, -1, -1)) and _usable(range(idx + 1, len(all_samples)))


def _compute_bracketing_standard_uncertainty(
    sample: Sample,
    ratio_name: str,
    all_samples: List[Sample],
    ratio_mean: float,
) -> Tuple[float, float, float]:
    """Compute basis-equivalent u_std from the SEMs of the bracketing standards."""
    ssb_data = sample.ssb_results.get(ratio_name)
    if ssb_data is None:
        delta_data = sample.delta_results.get(ratio_name, {})
        if delta_data.get("reference_kind") == "bracketing" and (
            "prev_std_n" in delta_data or "next_std_n" in delta_data
        ):
            prev_n = max(int(delta_data.get("prev_std_n", 0) or 0), 0)
            next_n = max(int(delta_data.get("next_std_n", 0) or 0), 0)
            prev_se = float(delta_data.get("prev_std_se", 0.0) or 0.0)
            next_se = float(delta_data.get("next_std_se", 0.0) or 0.0)
            u_std_raw_abs = float(u_ssb_standards(prev_se, next_se))
            reference_mean = _finite_positive(delta_data.get("std_mean"))
            u_std_abs, u_std_rel = _project_reference_uncertainty_to_basis(
                uncertainty_abs=u_std_raw_abs,
                reference_mean=reference_mean,
                ratio_mean=ratio_mean,
            )
            dof = _combined_bracket_dof(
                prev_se=prev_se,
                prev_dof=max(prev_n - 1, 0),
                next_se=next_se,
                next_dof=max(next_n - 1, 0),
            )
            return u_std_abs, u_std_rel, dof

        # No stored SSB bracket for this ratio. This is the normal path in
        # delta-only mode, but it also occurs when an SSB session's stored
        # bracket is not available at recomputation time (runtime/export
        # rebuilds, ratio-filtered sample copies). In both cases u_std is
        # still well-defined: recover it directly from the nearest bracketing
        # standards by position — u_std = 0.5 * sqrt(SEM_prev^2 + SEM_next^2).
        idx = next(
            (
                i for i, s in enumerate(all_samples)
                if s is sample
                or (
                    s.name == sample.name
                    and s.run_number == sample.run_number
                )
            ),
            -1,
        )
        if idx < 0:
            return 0.0, 0.0, 0
        prev_n, prev_se, prev_mean = 0, 0.0, None
        for j in range(idx - 1, -1, -1):
            s = all_samples[j]
            if not s.is_standard or s.metadata.get("excluded", False):
                continue
            cd = get_best_ratio_data(s, ratio_name)
            if cd is not None and cd.n_valid > 1:
                prev_n = cd.n_valid
                prev_se = float(np.std(cd.valid_values, ddof=1) / np.sqrt(prev_n))
                prev_mean = float(np.mean(cd.valid_values))
                break
        next_n, next_se, next_mean = 0, 0.0, None
        for j in range(idx + 1, len(all_samples)):
            s = all_samples[j]
            if not s.is_standard or s.metadata.get("excluded", False):
                continue
            cd = get_best_ratio_data(s, ratio_name)
            if cd is not None and cd.n_valid > 1:
                next_n = cd.n_valid
                next_se = float(np.std(cd.valid_values, ddof=1) / np.sqrt(next_n))
                next_mean = float(np.mean(cd.valid_values))
                break
        if prev_se <= 0 and next_se <= 0:
            return 0.0, 0.0, 0
        u_std_raw_abs = float(u_ssb_standards(prev_se, next_se))
        prev_mean = _finite_positive(prev_mean)
        next_mean = _finite_positive(next_mean)
        reference_mean = (
            _finite_positive((prev_mean + next_mean) / 2.0)
            if prev_mean is not None and next_mean is not None
            else None
        )
        u_std_abs, u_std_rel = _project_reference_uncertainty_to_basis(
            uncertainty_abs=u_std_raw_abs,
            reference_mean=reference_mean,
            ratio_mean=ratio_mean,
        )
        dof = _combined_bracket_dof(
            prev_se=prev_se,
            prev_dof=max(prev_n - 1, 0),
            next_se=next_se,
            next_dof=max(next_n - 1, 0),
        )
        return u_std_abs, u_std_rel, dof

    prev_se = ssb_data.get("prev_std_se", 0.0)
    next_se = ssb_data.get("next_std_se", 0.0)
    if prev_se <= 0 and next_se <= 0:
        return 0.0, 0.0, 0

    u_std_raw_abs = float(u_ssb_standards(prev_se, next_se))
    reference_mean = _resolve_ssb_bracketing_reference_mean(
        ssb_data=ssb_data,
        ratio_name=ratio_name,
        all_samples=all_samples,
    )
    u_std_abs, u_std_rel = _project_reference_uncertainty_to_basis(
        uncertainty_abs=u_std_raw_abs,
        reference_mean=reference_mean,
        ratio_mean=ratio_mean,
    )

    prev_n = 0
    next_n = 0
    if ssb_data.get("ssb_mode") == "block_average":
        try:
            prev_n = max(int(ssb_data.get("prev_n", 0) or 0), 0)
            next_n = max(int(ssb_data.get("next_n", 0) or 0), 0)
        except (TypeError, ValueError):
            prev_n = next_n = 0
        if prev_n == 0:
            prev_label = str(ssb_data.get("prev_std", "") or "")
            prev_n = len([name for name in prev_label.split("+") if name])
        if next_n == 0:
            next_label = str(ssb_data.get("next_std", "") or "")
            next_n = len([name for name in next_label.split("+") if name])
    else:
        # Use the accepted cycle counts the SSB correction recorded for the
        # standards it selected. Re-resolving the bracket by name finds a
        # different observation whenever two standards share a label — for
        # instance an unusable same-named standard sitting between the selected
        # one and the sample — and its n=0 then floors that side's DoF to one.
        prev_n = _producer_side_count(ssb_data, "prev_n")
        next_n = _producer_side_count(ssb_data, "next_n")

        if prev_n is None or next_n is None:
            # Legacy ssb_results without recorded counts: fall back to
            # resolution, preferring the recorded observation identity.
            sample_idx = _sample_index(all_samples, sample)
            if prev_n is None:
                prev_n = _resolved_side_count(
                    all_samples=all_samples,
                    ratio_name=ratio_name,
                    observation_id=ssb_data.get("prev_std_obs"),
                    label=ssb_data.get("prev_std"),
                    sample_idx=sample_idx,
                    before=True,
                )
            if next_n is None:
                next_n = _resolved_side_count(
                    all_samples=all_samples,
                    ratio_name=ratio_name,
                    observation_id=ssb_data.get("next_std_obs"),
                    label=ssb_data.get("next_std"),
                    sample_idx=sample_idx,
                    before=False,
                )

    # DoF: Welch-Satterthwaite for u_std = sqrt(prev_se^2 + next_se^2) / 2.
    # Components entering the RSS are prev_se/2 and next_se/2.
    # W-S is scale-invariant and gives correct nu_eff.
    dof = _combined_bracket_dof(
        prev_se=prev_se,
        prev_dof=max(prev_n - 1, 0),
        next_se=next_se,
        next_dof=max(next_n - 1, 0),
    )

    return u_std_abs, u_std_rel, dof


def _selected_reference_se_unavailable_reason(
    sample: Sample,
    ratio_name: str,
) -> str:
    """Explain why a producer-selected reference cannot supply measured SE."""
    payload = sample.ssb_results.get(ratio_name)
    count_keys = ("prev_n", "next_n")
    if payload is None:
        payload = sample.delta_results.get(ratio_name)
        count_keys = ("prev_std_n", "next_std_n")
    if not isinstance(payload, dict):
        return ""
    reasons = []
    for side, key, label_key, id_key in (
        ("preceding", count_keys[0], "prev_std", "prev_std_obs"),
        ("following", count_keys[1], "next_std", "next_std_obs"),
    ):
        if key not in payload or payload.get(key) is None:
            continue
        try:
            count = int(payload.get(key) or 0)
        except (TypeError, ValueError):
            count = 0
        if count < 2:
            name = str(payload.get(label_key) or "").strip()
            observation = str(payload.get(id_key) or "").strip()
            if name and observation:
                identity = f"{name} ({observation})"
            else:
                identity = name or observation or "unknown"
            reasons.append(
                f"the producer-selected {side} reference {identity} has "
                f"{count} accepted cycle(s); at least two are required for a "
                f"measured standard error, and a farther standard must not be "
                f"substituted for the one the delta actually used"
            )
    return "; ".join(reasons)


def _producer_side_count(ssb_data: Dict[str, object], key: str) -> Optional[int]:
    """Return the accepted cycle count the SSB producer recorded for one side.

    ``None`` means the producer left no count — an ``ssb_results`` payload from
    before the count was recorded — not that the count was zero.
    """
    if key not in ssb_data:
        return None
    raw = ssb_data.get(key)
    if raw is None:
        return None
    try:
        return max(int(raw), 0)
    except (TypeError, ValueError):
        return None


def _resolved_side_count(
    *,
    all_samples: List[Sample],
    ratio_name: str,
    observation_id: object,
    label: object,
    sample_idx: int,
    before: bool,
) -> int:
    """Accepted cycle count for a legacy bracket record, by identity if present."""
    resolved = _resolve_bracketing_sample(
        all_samples=all_samples,
        observation_id=observation_id,
        label=label,
        sample_idx=sample_idx,
        before=before,
    )
    if resolved is None:
        return 0
    cycle_data = get_best_ratio_data(resolved, ratio_name)
    return cycle_data.n_valid if cycle_data is not None else 0


def _combined_bracket_dof(
    *,
    prev_se: float,
    prev_dof: int,
    next_se: float,
    next_dof: int,
) -> float:
    """Combine both non-zero bracket components, flooring data-poor DoF at one."""
    components = []
    if prev_se > 0:
        components.append((prev_se / 2.0, float(max(prev_dof, 1))))
    if next_se > 0:
        components.append((next_se / 2.0, float(max(next_dof, 1))))
    return effective_dof(components) if components else 0.0


def _sample_index(all_samples: List[Sample], sample: Sample) -> int:
    return next(
        (
            i for i, candidate in enumerate(all_samples)
            if candidate is sample
            or (
                candidate.name == sample.name
                and candidate.run_number == sample.run_number
            )
        ),
        -1,
    )


def _resolve_bracketing_sample(
    *,
    all_samples: List[Sample],
    label: object,
    sample_idx: int,
    before: bool,
    observation_id: object = None,
) -> Optional[Sample]:
    """Resolve a recorded bracket, by identity when the producer recorded one.

    The label fallback prefers the side of the active sample, as before. It is
    a legacy path: a label alone cannot distinguish two standards sharing a
    name, which is exactly the case that made the DoF wrong.
    """
    if observation_id:
        wanted = str(observation_id)
        for candidate in all_samples:
            if candidate.observation_id == wanted:
                return candidate

    if label is None:
        return None
    label_text = str(label)

    if sample_idx >= 0:
        if before:
            indices = range(sample_idx - 1, -1, -1)
        else:
            indices = range(sample_idx + 1, len(all_samples))
        for idx in indices:
            candidate = all_samples[idx]
            if candidate.name == label_text:
                return candidate

    for candidate in all_samples:
        if candidate.name == label_text:
            return candidate
    return None


def _project_reference_uncertainty_to_basis(
    *,
    uncertainty_abs: float,
    reference_mean: Optional[float],
    ratio_mean: float,
) -> Tuple[float, float]:
    """Convert a reference-side uncertainty into the active basis-ratio space."""
    if uncertainty_abs <= 0.0:
        return 0.0, 0.0

    denominator = _finite_positive(reference_mean)
    if denominator is None:
        if ratio_mean <= 0.0:
            return 0.0, 0.0
        rel_permil = (uncertainty_abs / ratio_mean) * 1000.0
        return uncertainty_abs, rel_permil

    rel_permil = (uncertainty_abs / denominator) * 1000.0
    basis_abs = (rel_permil / 1000.0) * ratio_mean if ratio_mean > 0.0 else 0.0
    return basis_abs, rel_permil


def _resolve_ssb_bracketing_reference_mean(
    *,
    ssb_data: Dict[str, object],
    ratio_name: str,
    all_samples: List[Sample],
) -> Optional[float]:
    """Resolve the raw bracketing-standard mean used by the SSB correction."""
    prev_mean = _finite_positive(ssb_data.get("prev_std_mean"))
    next_mean = _finite_positive(ssb_data.get("next_std_mean"))

    if prev_mean is None:
        prev_mean = _resolve_standard_mean_by_label(
            label=ssb_data.get("prev_std"),
            observation_id=ssb_data.get("prev_std_obs"),
            ratio_name=ratio_name,
            all_samples=all_samples,
        )
    if next_mean is None:
        next_mean = _resolve_standard_mean_by_label(
            label=ssb_data.get("next_std"),
            observation_id=ssb_data.get("next_std_obs"),
            ratio_name=ratio_name,
            all_samples=all_samples,
        )

    if prev_mean is None or next_mean is None:
        return None
    return _finite_positive((prev_mean + next_mean) / 2.0)


def _resolve_standard_mean_by_label(
    *,
    label: object,
    ratio_name: str,
    all_samples: List[Sample],
    observation_id: object = None,
) -> Optional[float]:
    """Resolve a recorded standard into a mean ratio, by identity when recorded.

    Only reached for a legacy ``ssb_results`` payload that carries no
    ``prev_std_mean`` / ``next_std_mean``. The label fallback still handles a
    block-average label such as ``"A+B"``, which names several observations and
    so has no single identity.
    """
    if observation_id:
        wanted = str(observation_id)
        for sample in all_samples:
            if sample.observation_id != wanted:
                continue
            cd = get_best_ratio_data(sample, ratio_name)
            if cd is None or cd.n_valid == 0:
                return None
            return _finite_positive(float(np.mean(cd.valid_values)))

    names = [name for name in str(label or "").split("+") if name]
    if not names:
        return None

    means: List[float] = []
    for name in names:
        for sample in all_samples:
            if sample.name != name:
                continue
            cd = get_best_ratio_data(sample, ratio_name)
            if cd is None or cd.n_valid == 0:
                break
            means.append(float(np.mean(cd.valid_values)))
            break

    if len(means) != len(names):
        return None
    return _finite_positive(float(np.mean(means)))


def _compute_crm(
    certified_value: Optional[CertifiedValue],
    uncertainty_config: UncertaintyConfig,
    ratio_mean: float,
) -> Tuple[float, float]:
    """Compute u_CRM.  Returns (u_abs, u_rel_permil)."""
    if uncertainty_config.output_mode == "delta":
        return 0.0, 0.0

    if certified_value is None or certified_value.uncertainty <= 0:
        return 0.0, 0.0

    u_std = u_certified_value(certified_value.uncertainty, certified_value.k)
    reference_value = _finite_positive(getattr(certified_value, "value", None))
    denominator = reference_value if reference_value is not None else ratio_mean
    u_rel_permil = (u_std / denominator) * 1000.0 if denominator else 0.0

    return u_std, u_rel_permil


def _finite_positive(value: object) -> Optional[float]:
    """Return a finite positive float, or None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out > 0.0 else None


def _resolve_certified_reference_value(
    *,
    certified_value: Optional[CertifiedValue],
    element_config: ElementConfig,
    ratio_name: str,
) -> Optional[float]:
    cv_ref = _finite_positive(getattr(certified_value, "value", None))
    if cv_ref is not None:
        return cv_ref
    element_cv = element_config.certified_values.get(ratio_name)
    return _finite_positive(getattr(element_cv, "value", None))


def _unavailable_budget(
    *,
    engine: str,
    output_mode: str,
    basis_ratio_value: float,
    delta_reference_value: Optional[float],
    n_cycles: int,
    scope_note: str,
) -> UncertaintyBudget:
    return UncertaintyBudget(
        engine=engine,
        output_mode=output_mode,
        ratio_value=0.0,
        basis_ratio_value=basis_ratio_value,
        delta_reference_value=delta_reference_value,
        n_cycles=n_cycles,
        budget_scope="unavailable",
        scope_note=scope_note,
    )


def _resolve_delta_reference_value(
    *,
    sample: Sample,
    ratio_name: str,
    all_samples: List[Sample],
    uncertainty_config: UncertaintyConfig,
    processing_config: Optional[ProcessingConfig],
    certified_value: Optional[CertifiedValue],
    element_config: ElementConfig,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[float]:
    """Resolve the denominator used by the stored delta calculation."""
    use_classic_reference = _uses_classic_delta_reference(
        sample,
        ratio_name,
        uncertainty_config=uncertainty_config,
        processing_config=processing_config,
    )
    if use_classic_reference:
        bracketing_ref = _resolve_bracketing_delta_reference(
            sample=sample,
            ratio_name=ratio_name,
            all_samples=all_samples,
            cycle_ranges=cycle_ranges,
        )
        if bracketing_ref is not None:
            return bracketing_ref

    delta_data = (getattr(sample, "delta_results", {}) or {}).get(ratio_name, {})
    if isinstance(delta_data, dict):
        runtime_ref = _finite_positive(delta_data.get("std_mean"))
        if runtime_ref is not None:
            return runtime_ref
    if use_classic_reference:
        return None

    cv_ref = _finite_positive(getattr(certified_value, "value", None))
    if cv_ref is not None:
        return cv_ref

    element_cv = element_config.certified_values.get(ratio_name)
    return _finite_positive(getattr(element_cv, "value", None))


def _uses_classic_delta_reference(
    sample: Sample,
    ratio_name: str,
    *,
    uncertainty_config: UncertaintyConfig,
    processing_config: Optional[ProcessingConfig],
) -> bool:
    """Return True when delta denominator should come from bracketing standards."""
    if not uncertainty_config.enable_delta:
        return False
    delta_data = (getattr(sample, "delta_results", {}) or {}).get(ratio_name, {})
    if isinstance(delta_data, dict):
        reference_kind = str(delta_data.get("reference_kind") or "").strip().lower()
        if reference_kind == "bracketing":
            return True
        if reference_kind == "certified":
            return False
    if not uncertainty_config.enable_ssb:
        return True
    # Compatibility for sessions written before reference_kind was persisted:
    # SSB-enabled delta used the certified denominator unless the old pipeline
    # incorrectly switched it because drift was enabled. Never reproduce that
    # known scale-mixing behavior.
    return False


def _resolve_bracketing_delta_reference(
    *,
    sample: Sample,
    ratio_name: str,
    all_samples: List[Sample],
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[float]:
    """Resolve classic delta reference from the active bracketing standard layer."""
    sides = resolve_classic_delta_bracket_sides(
        sample=sample,
        ratio_name=ratio_name,
        all_samples=all_samples,
        cycle_ranges=cycle_ranges,
    )
    if sides is None:
        return None
    prev_side, next_side = sides
    return _finite_positive((prev_side.mean + next_side.mean) / 2.0)

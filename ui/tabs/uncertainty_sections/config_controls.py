"""Configuration and contributor controls for the uncertainty tab."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from typing import Dict, Optional

import numpy as np
import streamlit as st

from config.contributor_names import (
    LABEL_U_K4,
    LABEL_U_PREC,
    LABEL_U_STD,
    canonical_contributor_display_label,
    canonical_contributor_mapping,
    canonical_contributor_name,
)
from config.settings import (
    DEFAULT_K1_SAMPLE_DECOMPOSITION_PERMIL,
    DEFAULT_K2_MATRIX_SEPARATION_PERMIL,
    DEFAULT_K3_PROCEDURAL_BLANK_PERMIL,
    DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_PERMIL,
    DEFAULT_K6_MATRIX_EFFECTS_PERMIL,
    DEFAULT_K7_RESIDUAL_INTERFERENCES_PERMIL,
    DEFAULT_REPROD_DIG_SD,
    SSB_DELTA_BUILTIN_CONTRIBUTORS,
    SSB_DELTA_DEFAULT_ENABLED_CONTRIBUTORS,
    UncertaintyConfig,
)
from ui.formatting import get_k_footnote
from ui.state import get_state
from ui.tabs.uncertainty_sections.common import _chunked

_CONTRIBUTOR_TOGGLE_GROUPS = {
    "ssb_delta": (
        ("Type A evaluation", (
            ("u_prec", "Sample within-run precision"),
            ("u_std", LABEL_U_STD),
            ("u_std_repeatability", "Bracketing-standard repeatability"),
            ("u_blank", "Measurement-blank correction"),
        )),
        ("Other configurable contributors", (
            ("u_crm", "Certified/assigned reference ratio"),
            ("u_k1_sample_decomposition", "Sample digestion (κ₁)"),
            ("u_k2_matrix_separation", "Matrix separation (κ₂)"),
            ("u_k3_procedural_blank", "Procedural blank (κ₃)"),
            ("u_k4_bracketing_standard_heterogeneity", f"{LABEL_U_K4} (κ₄)"),
            ("u_k5_instrumental_drift", "Instrumental mass-bias drift (κ₅)"),
            ("u_k6_matrix_effects", "Matrix effects (κ₆)"),
            ("u_k7_residual_interferences", "Residual interferences (κ₇)"),
        )),
    ),
    "internal_normalization": (
        ("Type A evaluation", (
            ("u_prec", "Within-run precision"),
            ("enable_srm_repeatability", "Reference-material repeatability"),
            ("u_blank", "Blank correction"),
        )),
        ("Type B evaluation", (
            ("u_interf", "Isobaric interference"),
            ("u_crm", "Certified/assigned reference value"),
            ("u_ref_value", "Reference value (literature)"),
            ("u_bias_qc", "Bias of processed QC material"),
            ("u_reprod_dig", "Between-digestion reproducibility"),
            ("u_norm_ratio", "Normalisation-ratio uncertainty"),
        )),
    ),
    "pb_tl": (
        ("Type A evaluation", (
            ("u_prec", LABEL_U_PREC),
            ("u_std_repeatability", "Reference-material repeatability"),
            ("u_blank", "Blank correction"),
        )),
        ("Type B evaluation", (
            ("u_norm_ref", "Tl normalization ratio uncertainty"),
            ("u_interf", "204Hg interference correction"),
            ("u_crm", "CRM certified value"),
            ("u_kappa_drift", "Instrumental drift"),
            ("u_bias_qc", "Bias of processed QC material"),
            ("u_reprod_dig", "Between-digestion reproducibility"),
        )),
    ),
}
_CONTRIBUTOR_DISPLAY_LABELS = {
    name: label
    for _engine_groups in _CONTRIBUTOR_TOGGLE_GROUPS.values()
    for _group_title, _items in _engine_groups
    for name, label in _items
}
_CONTRIBUTOR_DISPLAY_LABELS["u_interf"] = "Isobaric interference"
_ALWAYS_BY_DESIGN = {"u_crm"}
_CONTROL_TOGGLE_NAMES = {
    "enable_srm_repeatability",
}
_PENDING_KAPPA_DEFAULT_ACTION_KEY = "_uc_pending_kappa_default_action"
_KAPPA_WIDGET_EPOCH_KEY = "_uc_kappa_widget_epoch"
_BUDGET_CHECKBOX_BACKUP_KEY = "_uc_budget_checkbox_backup"

_KAPPA_INPUT_SPECS = {
    "k1": {
        "attr": "k1_sample_decomposition_permil",
        "distribution_attr": "k1_sample_decomposition_distribution",
        "contributor_name": "u_k1_sample_decomposition",
        "widget_key": "uc_k1_sample_decomposition",
        "label": "Sample digestion (κ₁, ‰)",
        "step": 0.01,
        "format": "%.3f",
        "fallback": DEFAULT_K1_SAMPLE_DECOMPOSITION_PERMIL,
        "help": "Fixed Type B standard uncertainty for sample decomposition or digestion.",
    },
    "k2": {
        "attr": "k2_matrix_separation_permil",
        "distribution_attr": "k2_matrix_separation_distribution",
        "contributor_name": "u_k2_matrix_separation",
        "widget_key": "uc_k2_matrix_separation",
        "label": "Matrix separation (κ₂, ‰)",
        "step": 0.01,
        "format": "%.3f",
        "fallback": DEFAULT_K2_MATRIX_SEPARATION_PERMIL,
        "help": "Fixed Type B standard uncertainty for matrix separation.",
    },
    "k3": {
        "attr": "k3_procedural_blank_permil",
        "distribution_attr": "k3_procedural_blank_distribution",
        "contributor_name": "u_k3_procedural_blank",
        "widget_key": "uc_k3_procedural_blank",
        "label": "Procedural blank (κ₃, ‰)",
        "step": 0.005,
        "format": "%.3f",
        "fallback": DEFAULT_K3_PROCEDURAL_BLANK_PERMIL,
        "help": "Fixed Type B standard uncertainty for procedural blank effects.",
    },
    "k4": {
        "attr": "k4_bracketing_standard_heterogeneity_permil",
        "distribution_attr": "k4_bracketing_standard_heterogeneity_distribution",
        "contributor_name": "u_k4_bracketing_standard_heterogeneity",
        "widget_key": "uc_k4_bracketing_standard_heterogeneity",
        "label": f"{LABEL_U_K4} (κ₄, ‰)",
        "step": 0.01,
        "format": "%.3f",
        "fallback": DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_PERMIL,
        "help": f"Fixed Type B standard uncertainty for {LABEL_U_K4.lower()}.",
    },
    "k6": {
        "attr": "k6_matrix_effects_permil",
        "distribution_attr": "k6_matrix_effects_distribution",
        "contributor_name": "u_k6_matrix_effects",
        "widget_key": "uc_k6_matrix_effects",
        "label": "Matrix effects (κ₆, ‰)",
        "step": 0.01,
        "format": "%.3f",
        "fallback": DEFAULT_K6_MATRIX_EFFECTS_PERMIL,
        "help": "Fixed Type B standard uncertainty for matrix effects on mass discrimination.",
    },
    "k7": {
        "attr": "k7_residual_interferences_permil",
        "distribution_attr": "k7_residual_interferences_distribution",
        "contributor_name": "u_k7_residual_interferences",
        "widget_key": "uc_k7_residual_interferences",
        "label": "Residual interferences (κ₇, ‰)",
        "step": 0.01,
        "format": "%.3f",
        "fallback": DEFAULT_K7_RESIDUAL_INTERFERENCES_PERMIL,
        "help": "Fixed Type B standard uncertainty for residual interferences.",
    },
}
_SSB_KAPPA_CONTRIBUTOR_NAMES = {
    str(spec["contributor_name"]) for spec in _KAPPA_INPUT_SPECS.values()
} | {"u_k5_instrumental_drift"}


def contributor_display_label(name: str, *, ui_kind: Optional[str] = None) -> str:
    """Return the user-facing label for a contributor/control name."""
    canonical_name = canonical_contributor_name(name)
    if ui_kind == "sr_internal" and canonical_name in {
        "u_std_repeatability",
        "u_std_repeatability_se",
    }:
        return "Reference-material repeatability"
    group_key = "internal_normalization" if ui_kind == "sr_internal" else ui_kind
    if group_key in _CONTRIBUTOR_TOGGLE_GROUPS:
        for _group_title, items in _CONTRIBUTOR_TOGGLE_GROUPS[group_key]:
            for item_name, label in items:
                if canonical_contributor_name(item_name) == canonical_name:
                    return label
    canonical_label = canonical_contributor_display_label(name)
    if canonical_label is not None:
        return canonical_label
    if ui_kind == "pb_tl" and name == "u_interf":
        return "204Hg interference correction"
    if ui_kind == "pb_tl":
        calibrated_labels = {
            "u_pb_cal_std_precision": "Calibration-standard mean precision",
            "u_pb_cal_reference": "Accepted Pb reference through calibration",
            "u_pb_cal_residual": "Residual calibration repeatability (diagnostic)",
            "u_pb_cal_layout_mismatch": "Calibration layout mismatch (not quantified)",
            "u_pb_cal_sample_transfer": "Sample-specific transfer (not qualified)",
        }
        if name in calibrated_labels:
            return calibrated_labels[name]
    return _CONTRIBUTOR_DISPLAY_LABELS.get(name, name)


def _kappa_widget_epoch() -> int:
    """Return the active kappa widget generation."""
    try:
        return int(st.session_state.get(_KAPPA_WIDGET_EPOCH_KEY, 0))
    except (TypeError, ValueError):
        return 0


def _bump_kappa_widget_epoch() -> None:
    st.session_state[_KAPPA_WIDGET_EPOCH_KEY] = _kappa_widget_epoch() + 1


def _versioned_widget_key(base_key: str) -> str:
    epoch = _kappa_widget_epoch()
    return base_key if epoch <= 0 else f"{base_key}__{epoch}"


def _matching_versioned_keys(base_key: str) -> list[str]:
    prefix = f"{base_key}__"
    return [
        str(key)
        for key in list(st.session_state.keys())
        if str(key) == base_key or str(key).startswith(prefix)
    ]


def _render_config_panel(
    current: UncertaintyConfig,
    ui_context=None,
    *,
    expanded: bool = False,
    framed: bool = True,
) -> UncertaintyConfig:
    """Render the uncertainty configuration panel. Returns updated config.

    ``framed=True`` (default) keeps the historical collapsible expander.
    ``framed=False`` renders the same controls inline (no expander) so the
    Configure sub-pane can present them already open.
    """
    from ui.tabs.uncertainty_sections import shared_ui

    state = get_state()
    element_symbol = state.element_symbol or ""
    if not element_symbol and state.element_config is not None:
        element_symbol = getattr(state.element_config, "symbol", "")
    current = _apply_pending_kappa_default_action(current)
    resolved_engine = _resolve_uncertainty_engine(state, current)
    ui_kind = getattr(ui_context, "ui_kind", None) or (
        "pb_tl"
        if resolved_engine == "pb_tl_external_normalization"
        else "sr_internal"
        if resolved_engine == "internal_normalization"
        else "ssb_delta"
    )

    sr_iif_mode = getattr(current, "sr_iif_mode", "A")
    sr_norm_ratio_u_abs = float(
        getattr(current, "sr_norm_ratio_u_abs", 0.0)
    )
    pb_tl_norm_ratio_u_abs = float(
        getattr(current, "pb_tl_norm_ratio_u_abs", 0.0)
    )
    pb_tl_norm_ratio_u_justification = getattr(current, "pb_tl_norm_ratio_u_justification", "")
    pb_tl_norm_ratio_u_source = getattr(current, "pb_tl_norm_ratio_u_source", "")
    sr_blank_3var = bool(getattr(current, "sr_blank_3var", True))
    blank_uncertainty_input = str(
        getattr(current, "blank_uncertainty_input", "sd") or "sd"
    ).lower()
    if blank_uncertainty_input not in {"sd", "se"}:
        blank_uncertainty_input = "sd"
    sr_qc_bias_abs = float(getattr(current, "sr_qc_bias_abs", 0.0))
    sr_qc_cert_value = float(getattr(current, "sr_qc_cert_value", 0.0))
    u_prec_mode = str(getattr(current, "u_prec_mode", "se") or "se").lower()
    std_repeatability_mode = str(getattr(current, "std_repeatability_mode", "sd"))
    control_enabled = dict(getattr(current, "control_enabled", {}))
    u_reprod_dig_sd = float(getattr(current, "u_reprod_dig_sd", DEFAULT_REPROD_DIG_SD))
    u_reprod_dig_ref_value = float(getattr(current, "u_reprod_dig_ref_value", 0.0))
    blank_correlation_method = current.blank_correlation_method
    blank_fixed_r = current.blank_fixed_r

    panel_cm = (
        st.expander("Uncertainty Configuration", expanded=expanded)
        if framed
        else nullcontext()
    )
    with panel_cm:
        col1, col2 = st.columns(2)

        with col1:
            output_options = (
                ["delta", "absolute_ratio"]
                if bool(getattr(current, "enable_delta", False))
                else ["absolute_ratio"]
            )
            if (
                not bool(getattr(current, "enable_delta", False))
                and st.session_state.get("uc_output_mode") == "delta"
            ):
                st.session_state["uc_output_mode"] = "absolute_ratio"
            output_mode = shared_ui.segmented_choice(
                "Output mode",
                output_options,
                format_func=lambda x: {"delta": "Delta", "absolute_ratio": "Absolute ratio"}[x],
                index=(
                    output_options.index(current.output_mode)
                    if current.output_mode in output_options else 0
                ),
                key="uc_output_mode",
            )
            if not bool(getattr(current, "enable_delta", False)):
                st.caption("Delta output is available only when Report delta values is enabled.")

            coverage_method = shared_ui.segmented_choice(
                "Coverage method",
                ["fixed_k", "welch_satterthwaite"],
                format_func=lambda x: {
                    "welch_satterthwaite": "Welch-Satterthwaite",
                    "fixed_k": "Fixed k",
                }[x],
                index=0 if current.coverage_method == "fixed_k" else 1,
                key="uc_coverage_method",
            )

            coverage_k = current.coverage_k
            if coverage_method == "fixed_k":
                coverage_k = st.number_input(
                    "Fixed coverage factor k",
                    min_value=1.0,
                    max_value=4.0,
                    value=current.coverage_k,
                    step=0.1,
                    key="uc_coverage_k",
                )

        st.divider()
        contributor_enabled, control_enabled = _render_budget_contributor_toggles(
            current,
            ui_context=ui_context,
        )
        blank_enabled = contributor_enabled.get(
            "u_blank",
            current.is_contributor_enabled("u_blank", element_symbol=element_symbol),
        )
        if bool(blank_enabled):
            st.divider()
            blank_uncertainty_input = _render_blank_uncertainty_input(
                shared_ui,
                blank_uncertainty_input,
            )
            blank_correlation_method = shared_ui.segmented_choice(
                "Blank correlation",
                ["pearson_from_data", "fixed_value", "uncorrelated"],
                format_func=lambda x: {
                    "pearson_from_data": "Pearson from data",
                    "fixed_value": "Fixed r",
                    "uncorrelated": "Uncorrelated",
                }[x],
                index=["pearson_from_data", "fixed_value", "uncorrelated"].index(
                    blank_correlation_method
                ),
                key="uc_blank_corr",
            )
            if blank_correlation_method == "fixed_value":
                blank_fixed_r = st.number_input(
                    "Fixed blank correlation r",
                    min_value=-1.0,
                    max_value=1.0,
                    value=blank_fixed_r,
                    step=0.01,
                    key="uc_blank_fixed_r",
                )
        u_prec_enabled = contributor_enabled.get(
            "u_prec",
            current.is_contributor_enabled("u_prec", element_symbol=element_symbol),
        )
        if bool(u_prec_enabled):
            st.divider()
            u_prec_mode = _render_u_prec_mode(
                current,
                u_prec_mode=u_prec_mode,
            )
        if ui_kind == "ssb_delta":
            st.divider()
            (
                k1_value,
                k2_value,
                k3_value,
                k4_value,
                k6_value,
                k7_value,
                k1_distribution,
                k2_distribution,
                k3_distribution,
                k4_distribution,
                k6_distribution,
                k7_distribution,
            ) = _render_type_b_k_inputs(current, contributor_enabled)
        else:
            k1_value = float(getattr(current, "k1_sample_decomposition_permil", DEFAULT_K1_SAMPLE_DECOMPOSITION_PERMIL))
            k2_value = float(getattr(current, "k2_matrix_separation_permil", DEFAULT_K2_MATRIX_SEPARATION_PERMIL))
            k3_value = float(getattr(current, "k3_procedural_blank_permil", DEFAULT_K3_PROCEDURAL_BLANK_PERMIL))
            k4_value = float(getattr(
                current,
                "k4_bracketing_standard_heterogeneity_permil",
                DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_PERMIL,
            ))
            k6_value = float(getattr(current, "k6_matrix_effects_permil", DEFAULT_K6_MATRIX_EFFECTS_PERMIL))
            k7_value = float(getattr(current, "k7_residual_interferences_permil", DEFAULT_K7_RESIDUAL_INTERFERENCES_PERMIL))
            k1_distribution = getattr(current, "k1_sample_decomposition_distribution", "rectangular")
            k2_distribution = getattr(current, "k2_matrix_separation_distribution", "normal")
            k3_distribution = getattr(current, "k3_procedural_blank_distribution", "rectangular")
            k4_distribution = getattr(
                current,
                "k4_bracketing_standard_heterogeneity_distribution",
                "rectangular",
            )
            k6_distribution = getattr(current, "k6_matrix_effects_distribution", "normal")
            k7_distribution = getattr(current, "k7_residual_interferences_distribution", "rectangular")
        if ui_kind == "sr_internal":
            # Engine A uses the IIF/K-factor chain for drift; u_kappa_drift is
            # a separate post-correction term that is not defined for internal
            # normalization — see DECISIONS_LOG §uncertainty-engine-a-drift.
            contributor_enabled["u_kappa_drift"] = False
            st.session_state.pop("uc_contributor_u_kappa_drift", None)
            norm_ratio_enabled = contributor_enabled.get(
                "u_norm_ratio",
                current.is_contributor_enabled("u_norm_ratio", element_symbol=element_symbol),
            )
            sr_iif_mode = "B" if norm_ratio_enabled else "A"
            st.divider()
            (
                sr_qc_bias_abs,
                sr_qc_cert_value,
                u_reprod_dig_sd,
                u_reprod_dig_ref_value,
                sr_norm_ratio_u_abs,
            ) = _render_engine_a_contributor_value_inputs(
                current,
                sr_qc_bias_abs=sr_qc_bias_abs,
                sr_qc_cert_value=sr_qc_cert_value,
                u_reprod_dig_sd=u_reprod_dig_sd,
                u_reprod_dig_ref_value=u_reprod_dig_ref_value,
                sr_norm_ratio_u_abs=sr_norm_ratio_u_abs,
                contributor_enabled=contributor_enabled,
                norm_ratio_enabled=bool(norm_ratio_enabled),
                element_config=state.element_config,
            )
            st.divider()
            std_repeatability_mode = _render_engine_a_repeatability_mode(
                current,
                std_repeatability_mode=std_repeatability_mode,
            )
            if bool(blank_enabled):
                st.divider()
                sr_blank_3var = _render_engine_a_blank_controls(
                    current,
                    sr_blank_3var=sr_blank_3var,
                )
            # Map one UI toggle + one mode selector to the two internal
            # contributors used by the runtime engine.
            _srm_rep_enabled = bool(
                control_enabled.get(
                    "enable_srm_repeatability",
                    current.is_control_enabled("enable_srm_repeatability"),
                )
            )
            control_enabled["enable_srm_repeatability"] = _srm_rep_enabled
            contributor_enabled["u_std_repeatability"] = (
                _srm_rep_enabled and std_repeatability_mode == "sd"
            )
            contributor_enabled["u_std_repeatability_se"] = (
                _srm_rep_enabled and std_repeatability_mode == "se"
            )
        elif ui_kind == "pb_tl":
            norm_ref_enabled = bool(
                contributor_enabled.get(
                    "u_norm_ref",
                    current.is_contributor_enabled("u_norm_ref", element_symbol=element_symbol),
                )
            )
            if norm_ref_enabled:
                st.divider()
                pb_tl_norm_ratio_u_abs = _render_pb_tl_norm_ref_input(
                    pb_tl_norm_ratio_u_abs,
                )
                pb_tl_norm_ratio_u_justification = st.text_area(
                    "Tl uncertainty justification", value=pb_tl_norm_ratio_u_justification,
                    key="uc_pb_tl_norm_ratio_u_justification",
                    help="Explain the uncertainty evaluation and why it applies to this Tl ratio and material.",
                )
                pb_tl_norm_ratio_u_source = st.text_input(
                    "Tl uncertainty source", value=pb_tl_norm_ratio_u_source,
                    key="uc_pb_tl_norm_ratio_u_source",
                    help="Publication, certificate, or laboratory evaluation record identifier.",
                )
                if pb_tl_norm_ratio_u_abs > 0 and not (
                    pb_tl_norm_ratio_u_justification.strip() and pb_tl_norm_ratio_u_source.strip()
                ):
                    st.warning("Provide both a justification and source for the Tl standard uncertainty.")


    if ui_kind == "pb_tl":
        calibration_enabled = bool(getattr(
            getattr(getattr(state, "processing_config", None), "pb_standard_calibration", None),
            "enabled", False,
        ))
        if calibration_enabled:
            st.markdown("**Reference-material repeatability mode**")
            st.caption(
                "Pb-standard calibration propagates each calibration standard's mean SE through K. "
                "The SD/SE session-repeatability selector applies only to Tl-only Pb. "
                "Residual calibration repeatability remains a diagnostic outside the combined uncertainty."
            )
        else:
            std_repeatability_mode = _render_engine_a_repeatability_mode(
                current, std_repeatability_mode=std_repeatability_mode, pb_tl=True,
            )
        (sr_qc_bias_abs, sr_qc_cert_value, u_reprod_dig_sd,
         u_reprod_dig_ref_value, sr_norm_ratio_u_abs) = _render_engine_a_contributor_value_inputs(
            current, sr_qc_bias_abs=sr_qc_bias_abs, sr_qc_cert_value=sr_qc_cert_value,
            u_reprod_dig_sd=u_reprod_dig_sd, u_reprod_dig_ref_value=u_reprod_dig_ref_value,
            sr_norm_ratio_u_abs=sr_norm_ratio_u_abs, contributor_enabled=contributor_enabled,
            norm_ratio_enabled=False, element_config=state.element_config,
        )

    if ui_kind == "sr_internal":
        include_kappa_drift = False
    else:
        drift_contributor_name = (
            "u_k5_instrumental_drift" if ui_kind == "ssb_delta" else "u_kappa_drift"
        )
        include_kappa_drift = contributor_enabled.get(
            drift_contributor_name,
            current.include_kappa_drift,
        )

    return replace(
        current,
        blank_correlation_method=blank_correlation_method,
        blank_fixed_r=blank_fixed_r,
        blank_uncertainty_input=blank_uncertainty_input,
        output_mode=output_mode,
        coverage_method=coverage_method,
        coverage_k=coverage_k,
        include_kappa_drift=include_kappa_drift,
        contributor_enabled=contributor_enabled,
        control_enabled=control_enabled,
        k1_sample_decomposition_permil=k1_value,
        k2_matrix_separation_permil=k2_value,
        k3_procedural_blank_permil=k3_value,
        k4_bracketing_standard_heterogeneity_permil=k4_value,
        k6_matrix_effects_permil=k6_value,
        k7_residual_interferences_permil=k7_value,
        k1_sample_decomposition_distribution=k1_distribution,
        k2_matrix_separation_distribution=k2_distribution,
        k3_procedural_blank_distribution=k3_distribution,
        k4_bracketing_standard_heterogeneity_distribution=k4_distribution,
        k6_matrix_effects_distribution=k6_distribution,
        k7_residual_interferences_distribution=k7_distribution,
        sr_iif_mode=sr_iif_mode,
        sr_norm_ratio_u_abs=sr_norm_ratio_u_abs,
        pb_tl_norm_ratio_u_abs=pb_tl_norm_ratio_u_abs,
        pb_tl_norm_ratio_u_justification=pb_tl_norm_ratio_u_justification,
        pb_tl_norm_ratio_u_source=pb_tl_norm_ratio_u_source,
        sr_blank_3var=sr_blank_3var,
        sr_qc_bias_abs=sr_qc_bias_abs,
        sr_qc_cert_value=sr_qc_cert_value,
        std_repeatability_mode=std_repeatability_mode,
        u_prec_mode=u_prec_mode,
        u_reprod_dig_sd=u_reprod_dig_sd,
        u_reprod_dig_ref_value=u_reprod_dig_ref_value,
    )


def _render_budget_contributor_toggles(
    current: UncertaintyConfig,
    ui_context=None,
) -> tuple[Dict[str, bool], Dict[str, bool]]:
    """Render include/exclude checkboxes for contributors and UI controls."""
    state = get_state()
    element_symbol = state.element_symbol or ""
    if not element_symbol and state.element_config is not None:
        element_symbol = getattr(state.element_config, "symbol", "")
    engine = _resolve_uncertainty_engine(state, current)
    ui_kind = getattr(ui_context, "ui_kind", None) or (
        "pb_tl"
        if engine == "pb_tl_external_normalization"
        else "sr_internal"
        if engine == "internal_normalization"
        else "ssb_delta"
    )
    group_key = "internal_normalization" if ui_kind == "sr_internal" else ui_kind
    groups = _CONTRIBUTOR_TOGGLE_GROUPS.get(group_key, _CONTRIBUTOR_TOGGLE_GROUPS["ssb_delta"])
    contributor_enabled = canonical_contributor_mapping(
        getattr(current, "contributor_enabled", {})
    )
    if (
        engine == "ssb_delta"
        and "u_k5_instrumental_drift" not in contributor_enabled
        and "u_kappa_drift" in getattr(current, "contributor_enabled", {})
    ):
        contributor_enabled["u_k5_instrumental_drift"] = bool(
            getattr(current, "contributor_enabled", {})["u_kappa_drift"]
        )
    if engine == "ssb_delta":
        contributor_enabled.pop("u_kappa_drift", None)
    global_enabled_defaults = state.global_contributor_enabled_defaults(element_symbol)
    contributor_enabled.pop("u_std_repeatability_enabled", None)
    contributor_enabled.pop("_sr_norm_ratio_u", None)
    control_enabled = dict(getattr(current, "control_enabled", {}))
    backup_scope = f"{element_symbol or '-'}:{ui_kind}"
    checkbox_backups = st.session_state.setdefault(_BUDGET_CHECKBOX_BACKUP_KEY, {})
    if not isinstance(checkbox_backups, dict):
        checkbox_backups = {}
        st.session_state[_BUDGET_CHECKBOX_BACKUP_KEY] = checkbox_backups
    checkbox_backup = checkbox_backups.setdefault(backup_scope, {})
    if not isinstance(checkbox_backup, dict):
        checkbox_backup = {}
        checkbox_backups[backup_scope] = checkbox_backup
    if ui_kind == "sr_internal":
        checkbox_backup.pop("u_kappa_drift", None)
        contributor_enabled["u_kappa_drift"] = False
        st.session_state.pop("uc_contributor_u_kappa_drift", None)

    st.markdown("**Budget contributors**")
    st.caption(
        "Select the uncertainty contributions evaluated by Type A and Type B methods "
        "for inclusion in the uncertainty budget."
    )

    for title, items in groups:
        st.caption(title)
        for name, label in items:
            base_widget_key = (
                f"uc_control_{name}"
                if name in _CONTROL_TOGGLE_NAMES
                else f"uc_contributor_{name}"
            )
            widget_key = (
                _versioned_widget_key(base_widget_key)
                if ui_kind == "ssb_delta"
                and name in _SSB_KAPPA_CONTRIBUTOR_NAMES
                else base_widget_key
            )
            if name in _CONTROL_TOGGLE_NAMES:
                if name in control_enabled:
                    default = bool(control_enabled[name])
                elif name in global_enabled_defaults:
                    default = bool(global_enabled_defaults[name])
                else:
                    default = bool(current.is_control_enabled(name))
            elif (
                ui_kind != "ssb_delta"
                and name not in contributor_enabled
                and name in global_enabled_defaults
            ):
                default = bool(global_enabled_defaults[name])
            elif ui_kind == "pb_tl" and name == "u_kappa_drift":
                default = bool(getattr(current, "include_kappa_drift", False))
            else:
                default = current.is_contributor_enabled(
                    name, element_symbol=element_symbol
                )
            if name in checkbox_backup:
                default = bool(checkbox_backup[name])
            if widget_key not in st.session_state:
                st.session_state[widget_key] = default
            value = st.checkbox(
                label,
                key=widget_key,
                help=(
                    _sr_contributor_toggle_help(name)
                    if ui_kind == "sr_internal" else None
                ),
            )
            st.caption(
                "Active — included in the configured budget when applicable."
                if value
                else "Inactive — editable details are hidden; their values are retained."
            )
            if name in _CONTROL_TOGGLE_NAMES:
                control_enabled[name] = value
            else:
                contributor_enabled[name] = value
            checkbox_backup[name] = bool(value)

    return contributor_enabled, control_enabled


def reset_ssb_delta_contributor_widget_state() -> None:
    """Clear cached Engine B contributor checkboxes after correction-mode changes."""
    names = set(SSB_DELTA_BUILTIN_CONTRIBUTORS) | {"u_kappa_drift"}
    for name in names:
        base_key = f"uc_contributor_{name}"
        for key in list(st.session_state.keys()):
            text_key = str(key)
            if text_key == base_key or text_key.startswith(f"{base_key}__"):
                st.session_state.pop(key, None)

    backups = st.session_state.get(_BUDGET_CHECKBOX_BACKUP_KEY)
    if isinstance(backups, dict):
        for scope in list(backups.keys()):
            if ":ssb_delta" in str(scope):
                backups.pop(scope, None)


def _render_u_prec_mode(
    current: UncertaintyConfig,
    *,
    u_prec_mode: str,
) -> str:
    """Render the common u_prec SE/SD selector."""
    options = ["se", "sd"]
    selected = str(u_prec_mode or getattr(current, "u_prec_mode", "se") or "se").lower()
    if selected not in options:
        selected = "se"

    st.markdown("**Measurement precision mode**")
    mode = st.radio(
        "u_prec statistic",
        options=options,
        index=options.index(selected),
        format_func=lambda value: {
            "se": "SE of the mean",
            "sd": "SD of cycles",
        }[value],
        horizontal=True,
        key="uc_u_prec_mode",
        help=(
            "SE reports uncertainty of the mean ratio. SD uses full within-run "
            "cycle scatter as a conservative repeatability term."
        ),
    )
    if mode == "sd":
        st.caption("u_prec = cycle SD; this does not shrink with cycle count.")
    else:
        st.caption("u_prec = cycle SD / sqrt(n); this is the default GUM mean uncertainty.")
    return str(mode)


def _default_or_current_contributor_enabled(
    current: UncertaintyConfig,
    name: str,
) -> bool:
    """Return contributor state using engine-aware defaults for the current element."""
    state = get_state()
    element_symbol = state.element_symbol or ""
    if not element_symbol and state.element_config is not None:
        element_symbol = getattr(state.element_config, "symbol", "")
    return current.is_contributor_enabled(name, element_symbol=element_symbol)


def _coverage_footnote_for_config(u_config: UncertaintyConfig) -> str:
    """Return a coverage footnote for the active uncertainty configuration."""
    if u_config.coverage_method == "welch_satterthwaite":
        return get_k_footnote(
            coverage_method="welch_satterthwaite",
            sample_specific=True,
        )
    return get_k_footnote(k=u_config.coverage_k, coverage_method="fixed_k")


def _current_element_symbol() -> str:
    state = get_state()
    element_symbol = state.element_symbol or ""
    if not element_symbol and state.element_config is not None:
        element_symbol = getattr(state.element_config, "symbol", "")
    return element_symbol


def _resolve_kappa_input_default(current: UncertaintyConfig, kappa: str) -> float:
    """Resolve a kappa input default, switching element defaults safely."""
    spec = _KAPPA_INPUT_SPECS[kappa]
    state = get_state()
    element_symbol = _current_element_symbol()

    widget_key = _versioned_widget_key(str(spec["widget_key"]))
    element_key = f"_{widget_key}_default_element"
    element_default = state.resolve_kappa_default(kappa, element_symbol)
    stored_value = float(getattr(current, spec["attr"], element_default))

    if widget_key in st.session_state:
        last_element = st.session_state.get(element_key, element_symbol)
        last_default = state.resolve_kappa_default(kappa, last_element)
        widget_value = float(st.session_state[widget_key])

        if (
            last_element != element_symbol
            and np.isclose(widget_value, last_default)
            and np.isclose(stored_value, widget_value)
        ):
            st.session_state[widget_key] = element_default
            widget_value = element_default

        st.session_state[element_key] = element_symbol
        return widget_value

    st.session_state[element_key] = element_symbol
    if np.isclose(stored_value, getattr(UncertaintyConfig(), spec["attr"])):
        return element_default
    return stored_value


def _resolve_kappa_distribution_default(current: UncertaintyConfig, kappa: str) -> str:
    """Resolve the element distribution for a kappa term."""
    spec = _KAPPA_INPUT_SPECS[kappa]
    state = get_state()
    element_symbol = _current_element_symbol()
    element_key = f"_{_versioned_widget_key(str(spec['widget_key']))}_distribution_default_element"
    element_default = state.resolve_kappa_distribution(kappa, element_symbol)
    stored_value = str(getattr(current, spec["distribution_attr"], element_default) or element_default)

    last_element = st.session_state.get(element_key, element_symbol)
    last_default = state.resolve_kappa_distribution(kappa, last_element)
    st.session_state[element_key] = element_symbol
    factory_default = str(getattr(UncertaintyConfig(), spec["distribution_attr"]))
    if stored_value in {factory_default, last_default}:
        return element_default
    return stored_value


def _resolve_sr_engine_a_input_default(
    current: UncertaintyConfig,
    *,
    spec_key: str,
    attr: str,
    widget_key: str,
    hardcoded_default: float,
) -> float:
    """Resolve a Sr Engine A input default from global_uncertainty_values.json."""
    state = get_state()
    element_symbol = _current_element_symbol()
    stored_value = float(getattr(current, attr, hardcoded_default))

    if widget_key in st.session_state:
        return float(st.session_state[widget_key])

    if element_symbol == "Sr":
        element_default = state.resolve_sr_engine_a_default(spec_key, element_symbol)
        if np.isclose(stored_value, hardcoded_default):
            return float(element_default)
    return stored_value


def _clear_kappa_widget_state() -> None:
    """Clear kappa widget/default sentinels so Streamlit re-pulls disk defaults."""
    for spec in _KAPPA_INPUT_SPECS.values():
        key = str(spec["widget_key"])
        for versioned_key in _matching_versioned_keys(key):
            st.session_state.pop(versioned_key, None)
            st.session_state.pop(f"_{versioned_key}_default_element", None)
            st.session_state.pop(f"_{versioned_key}_distribution_default_element", None)
        contributor_key = f"uc_contributor_{spec['contributor_name']}"
        for versioned_key in _matching_versioned_keys(contributor_key):
            st.session_state.pop(versioned_key, None)
    for contributor_key in (
        "uc_contributor_u_k5_instrumental_drift",
        "uc_contributor_u_kappa_drift",
    ):
        for versioned_key in _matching_versioned_keys(contributor_key):
            st.session_state.pop(versioned_key, None)


def _config_with_element_kappa_defaults(
    current: UncertaintyConfig,
) -> UncertaintyConfig:
    """Return *current* with the active element's disk/cached kappa defaults applied."""
    state = get_state()
    element_symbol = _current_element_symbol()
    contributor_enabled = canonical_contributor_mapping(
        getattr(current, "contributor_enabled", {})
    )
    updates = {}

    for short_name, spec in _KAPPA_INPUT_SPECS.items():
        updates[spec["attr"]] = state.resolve_kappa_default(short_name, element_symbol)
        updates[spec["distribution_attr"]] = state.resolve_kappa_distribution(
            short_name,
            element_symbol,
        )
        contributor_name = str(spec["contributor_name"])
        contributor_enabled[contributor_name] = (
            contributor_name in SSB_DELTA_DEFAULT_ENABLED_CONTRIBUTORS
        )

    updates["kappa_drift_distribution"] = state.resolve_kappa_drift_distribution(
        element_symbol
    )

    contributor_enabled["u_k5_instrumental_drift"] = (
        "u_k5_instrumental_drift" in SSB_DELTA_DEFAULT_ENABLED_CONTRIBUTORS
    )

    return replace(
        current,
        contributor_enabled=contributor_enabled,
        include_kappa_drift=contributor_enabled.get(
            "u_k5_instrumental_drift",
            current.include_kappa_drift,
        ),
        **updates,
    )


def _apply_pending_kappa_default_action(current: UncertaintyConfig) -> UncertaintyConfig:
    """Apply a queued kappa reset/reload before Streamlit creates the widgets."""
    action = st.session_state.pop(_PENDING_KAPPA_DEFAULT_ACTION_KEY, None)
    if action not in {"reset", "reload"}:
        return current

    state = get_state()
    if action == "reload":
        state.refresh_global_uncertainty_values()
        st.session_state.pop("_global_uncertainty_values_warning_shown", None)

    _clear_kappa_widget_state()
    _bump_kappa_widget_epoch()
    updated = _config_with_element_kappa_defaults(current)
    state.uncertainty_config = updated
    return updated


def _render_kappa_number_input(current: UncertaintyConfig, kappa: str) -> float:
    spec = _KAPPA_INPUT_SPECS[kappa]
    return float(
        st.number_input(
            spec["label"],
            min_value=0.0,
            value=_resolve_kappa_input_default(current, kappa),
            step=spec["step"],
            format=spec["format"],
            key=_versioned_widget_key(str(spec["widget_key"])),
            help=spec["help"],
        )
    )


def _render_type_b_k_inputs(
    current: UncertaintyConfig,
    contributor_enabled: Dict[str, bool],
) -> tuple[float, float, float, float, float, float, str, str, str, str, str, str]:
    """Render only active fixed Type B kappa inputs, retaining hidden values."""
    state = get_state()
    st.markdown("**Other configurable contributor values**")
    st.caption(
        "κ₁, κ₂, κ₃, κ₄, κ₆, and κ₇ are user-editable fixed standard "
        "uncertainties in ‰. κ₅ is calculated from the included "
        "standard sequence when instrumental drift is enabled."
    )
    st.caption("Element defaults can be overridden for this session.")
    error = state.global_uncertainty_values_error
    if error and not st.session_state.get("_global_uncertainty_values_warning_shown"):
        st.warning(
            "Could not load global_uncertainty_values.json; using hardcoded "
            f"fallback defaults. Parser error: {error}"
        )
        st.session_state["_global_uncertainty_values_warning_shown"] = True

    names = ("k1", "k2", "k3", "k4", "k6", "k7")
    values = {name: _resolve_kappa_input_default(current, name) for name in names}
    active_names = [
        name
        for name in names
        if contributor_enabled.get(str(_KAPPA_INPUT_SPECS[name]["contributor_name"]), False)
    ]
    if active_names:
        action_col1, action_col2, _ = st.columns([1, 1, 3])
        with action_col1:
            if st.button("Reset to element defaults", key="uc_reset_kappa_defaults"):
                st.session_state[_PENDING_KAPPA_DEFAULT_ACTION_KEY] = "reset"
                st.rerun()
        with action_col2:
            if st.button("Reload defaults from disk", key="uc_reload_global_uc_defaults"):
                st.session_state[_PENDING_KAPPA_DEFAULT_ACTION_KEY] = "reload"
                st.rerun()
        for name in active_names:
            values[name] = _render_kappa_number_input(current, name)
            st.caption("Active fixed Type B contributor; source and model are given in the field help.")
    else:
        st.caption("All editable κ contributors are inactive. Their entered values are retained.")

    return (
        *(float(values[name]) for name in names),
        _resolve_kappa_distribution_default(current, "k1"),
        _resolve_kappa_distribution_default(current, "k2"),
        _resolve_kappa_distribution_default(current, "k3"),
        _resolve_kappa_distribution_default(current, "k4"),
        _resolve_kappa_distribution_default(current, "k6"),
        _resolve_kappa_distribution_default(current, "k7"),
    )


def _resolve_uncertainty_engine(state, u_config: UncertaintyConfig) -> str:
    """Resolve the active uncertainty engine for the current element/state."""
    element_symbol = state.element_symbol or ""
    if not element_symbol and state.element_config is not None:
        element_symbol = getattr(state.element_config, "symbol", "")
    return u_config.resolve_engine(
        element_symbol,
        processing_config=getattr(state, "processing_config", None),
    )


def _render_engine_a_blank_controls(
    current: UncertaintyConfig,
    *,
    sr_blank_3var: bool,
) -> bool:
    """Render Engine A Sr blank-model controls."""
    st.markdown("**Sr blank model**")
    st.caption(
        "Default Engine A blank propagation uses m/z 87 and 86 only. "
        "Enable the 3-variable Sr mode to include m/z 88 via Kragten "
        "perturbation through the correction chain."
    )

    return st.checkbox(
        "Include m/z 88 in Sr blank covariance",
        value=sr_blank_3var,
        key="uc_sr_blank_3var",
        help=(
            "Engine A only. Keeps the current 87/86 blank covariance model when off; "
            "when on, adds the indirect blank effect from 88Sr on the IIF correction."
        ),
    )


def _render_blank_uncertainty_input(shared_ui, current_value: str) -> str:
    """Render the global blank input SD/SE selector."""
    options = ["sd", "se"]
    value = str(current_value or "sd").lower()
    if value not in options:
        value = "sd"

    st.markdown("**Blank uncertainty input**")
    return shared_ui.segmented_choice(
        "Blank uncertainty input",
        options,
        format_func=lambda x: {
            "sd": "SD of blank cycles",
            "se": "SE of blank mean",
        }[x],
        index=options.index(value),
        key="uc_blank_uncertainty_input",
        help=(
            "SD uses cycle-to-cycle blank scatter as the input standard uncertainty. "
            "SE divides that cycle SD by sqrt(n) and propagates uncertainty of the "
            "selected blank mean. The selected blank correlation method is applied "
            "in both modes."
        ),
    )


def _enable_contributor_when_positive(widget_key: str, contributor_name: str) -> None:
    """Enable a contributor checkbox when a linked positive value is entered."""
    try:
        value = float(st.session_state.get(widget_key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return
    if value > 0.0:
        st.session_state[f"uc_contributor_{contributor_name}"] = True


def _sr_contributor_toggle_help(name: str) -> Optional[str]:
    """Return Sr-specific help text for contributor toggles."""
    if name == "u_bias_qc":
        return (
            "Bias of a processed control material: "
            "enter the already-converted absolute uncertainty numerator and "
            "the QC material certified ratio."
        )
    if name == "u_norm_ratio":
        return (
            "Propagate the uncertainty of the accepted normalisation ratio "
            "through the Sr correction chain. If the selected normalisation "
            "value is conventional/exact, keep this off."
        )
    return None


def _engine_a_required_value_warnings(
    *,
    qc_bias_enabled: bool,
    qc_bias_abs: float,
    qc_cert_value: float,
    reprod_dig_enabled: bool,
    reprod_dig_sd: float,
    reprod_dig_ref_value: float,
) -> list[str]:
    """Return warnings for enabled Sr Engine A terms with incomplete inputs."""
    warnings: list[str] = []
    if qc_bias_enabled:
        if float(qc_bias_abs) <= 0.0:
            warnings.append(
                "Enter the QC uncertainty numerator for the processed control sample."
            )
        if float(qc_cert_value) <= 0.0:
            warnings.append(
                "Enter the certified ratio of the QC material; this denominator is required."
            )
    if reprod_dig_enabled:
        if float(reprod_dig_sd) <= 0.0:
            warnings.append(
                "Enter the between-digestion SD from independent processed digestions."
            )
        if float(reprod_dig_ref_value) <= 0.0:
            warnings.append(
                "Enter the reference ratio of the digested material; this denominator is required."
            )
    return warnings


def _render_engine_a_contributor_value_inputs(
    current: UncertaintyConfig,
    *,
    sr_qc_bias_abs: float,
    sr_qc_cert_value: float,
    u_reprod_dig_sd: float,
    u_reprod_dig_ref_value: float,
    sr_norm_ratio_u_abs: float,
    contributor_enabled: Dict[str, bool],
    norm_ratio_enabled: bool,
    element_config=None,
) -> tuple[float, float, float, float, float]:
    """Render Engine A numeric contributor values."""
    state = get_state()

    qc_bias_enabled = bool(contributor_enabled.get("u_bias_qc", False))
    reprod_dig_enabled = bool(contributor_enabled.get("u_reprod_dig", False))

    sr_qc_bias_abs = _resolve_sr_engine_a_input_default(
        current,
        spec_key="u_bias_qc",
        attr="sr_qc_bias_abs",
        widget_key="uc_sr_qc_bias_abs",
        hardcoded_default=0.0,
    )
    u_reprod_dig_sd = _resolve_sr_engine_a_input_default(
        current,
        spec_key="u_reprod_dig",
        attr="u_reprod_dig_sd",
        widget_key="uc_u_reprod_dig_sd",
        hardcoded_default=DEFAULT_REPROD_DIG_SD,
    )
    sr_norm_ratio_u_abs = _resolve_sr_engine_a_input_default(
        current,
        spec_key="enable_sr_norm_ratio_uncertainty",
        attr="sr_norm_ratio_u_abs",
        widget_key="uc_sr_norm_ratio_u_abs",
        hardcoded_default=0.0,
    )

    _qc_bias_val = float(sr_qc_bias_abs)
    _qc_cert_val = float(sr_qc_cert_value)
    _reprod_dig_val = float(u_reprod_dig_sd)
    _reprod_dig_ref_val = float(u_reprod_dig_ref_value)
    _norm_ratio_val = float(sr_norm_ratio_u_abs)

    if not any((qc_bias_enabled, reprod_dig_enabled, norm_ratio_enabled)):
        return (
            float(_qc_bias_val),
            float(_qc_cert_val),
            float(_reprod_dig_val),
            float(_reprod_dig_ref_val),
            float(_norm_ratio_val),
        )

    st.markdown("**Session default contributor values**")
    st.caption(
        "These values are used when a sample has no sample-specific value."
    )

    active_value_fields = [
        name for name, enabled in (
            ("u_bias_qc", qc_bias_enabled),
            ("u_reprod_dig", reprod_dig_enabled),
            ("u_norm_ratio", norm_ratio_enabled),
        )
        if enabled
    ]
    value_columns = iter(st.columns(len(active_value_fields))) if active_value_fields else iter(())

    if qc_bias_enabled:
        bias_col = next(value_columns)
        with bias_col:
            _qc_bias_val = st.number_input(
                "Default QC uncertainty numerator (absolute ratio)",
                value=float(sr_qc_bias_abs),
                min_value=0.0,
                step=0.000001,
                format="%.10f",
                key="uc_sr_qc_bias_abs",
                on_change=_enable_contributor_when_positive,
                args=("uc_sr_qc_bias_abs", "u_bias_qc"),
                help=(
                    "User-supplied absolute uncertainty numerator for the processed "
                    "control material. Apply any distribution conversion yourself "
                    "(for example divide by sqrt(3), by 2, or another justified "
                    "factor) before entering this value."
                ),
            )
            _qc_cert_val = st.number_input(
                "Default certified ratio of QC material",
                value=float(sr_qc_cert_value),
                min_value=0.0,
                step=0.000001,
                format="%.10f",
                key="uc_sr_qc_cert_value",
                help=(
                    "Required denominator for the control material. The QC "
                    "uncertainty is first converted to a relative standard "
                    "uncertainty as u_QC_input / R_QC_cert. TraceISO converts "
                    "that relative term to absolute ratio units only when "
                    "assembling the final ratio budget."
                ),
            )
            for warning in _engine_a_required_value_warnings(
                qc_bias_enabled=True,
                qc_bias_abs=float(_qc_bias_val),
                qc_cert_value=float(_qc_cert_val),
                reprod_dig_enabled=False,
                reprod_dig_sd=0.0,
                reprod_dig_ref_value=0.0,
            ):
                st.warning(warning)
    if reprod_dig_enabled:
        reprod_col = next(value_columns)
        with reprod_col:
            _reprod_dig_val = st.number_input(
                "Default between-digestion SD (absolute ratio)",
                value=float(u_reprod_dig_sd),
                min_value=0.0,
                step=1e-7,
                format="%.10f",
                key="uc_u_reprod_dig_sd",
                on_change=_enable_contributor_when_positive,
                args=("uc_u_reprod_dig_sd", "u_reprod_dig"),
                help=(
                    "Absolute SD of independently processed digestion means. "
                    "If the SD was measured on a reference/control material, "
                    "enter that material's ratio below so the SD is propagated "
                    "fractionally to each sample."
                ),
            )
            _reprod_dig_ref_val = st.number_input(
                "Default reference ratio of digested material",
                value=float(u_reprod_dig_ref_value),
                min_value=0.0,
                step=0.000001,
                format="%.10f",
                key="uc_u_reprod_dig_ref_value",
                help=(
                    "Required denominator for the processed material used to "
                    "estimate between-digestion SD. The term is u = "
                    "SD_dig / R_dig_ref, transferred to the active sample basis."
                ),
            )
            for warning in _engine_a_required_value_warnings(
                qc_bias_enabled=False,
                qc_bias_abs=0.0,
                qc_cert_value=0.0,
                reprod_dig_enabled=True,
                reprod_dig_sd=float(_reprod_dig_val),
                reprod_dig_ref_value=float(_reprod_dig_ref_val),
            ):
                st.warning(warning)
    if norm_ratio_enabled:
        norm_col = next(value_columns)
        with norm_col:
            active_norm_ratio = None
            try:
                from ui.utils import get_active_internal_normalization_ratio

                active_norm_ratio = get_active_internal_normalization_ratio(
                    getattr(state, "processing_config", None),
                    element_config,
                )
            except Exception:
                active_norm_ratio = None
            _norm_label = (
                f"Normalisation-ratio standard uncertainty ({active_norm_ratio}, absolute ratio)"
                if active_norm_ratio else "Normalisation-ratio standard uncertainty (absolute ratio)"
            )
            _norm_ratio_val = float(
                st.number_input(
                    _norm_label,
                    min_value=0.0,
                    value=float(sr_norm_ratio_u_abs),
                    step=1e-8,
                    format="%.10f",
                    key="uc_sr_norm_ratio_u_abs",
                    on_change=_enable_contributor_when_positive,
                    args=("uc_sr_norm_ratio_u_abs", "u_norm_ratio"),
                    help=(
                        "Absolute standard uncertainty of the selected normalization "
                        "ratio. If a certificate gives expanded uncertainty U, enter U/k. "
                        "This term is propagated through the Sr correction chain."
                    ),
                )
            )

    return (
        float(_qc_bias_val),
        float(_qc_cert_val),
        float(_reprod_dig_val),
        float(_reprod_dig_ref_val),
        float(_norm_ratio_val),
    )


def _render_pb_tl_norm_ref_input(pb_tl_norm_ratio_u_abs: float) -> float:
    """Render the Pb-Tl normalization-ratio uncertainty input."""
    state = get_state()
    active_norm_ratio = "205Tl/203Tl"
    try:
        processing_config = getattr(state, "processing_config", None)
        element_config = getattr(state, "element_config", None)
        ratio_name = (
            getattr(processing_config, "normalization_ratio_override", None)
            if processing_config is not None
            else None
        ) or (
            getattr(element_config, "normalization_ratio", None)
            if element_config is not None
            else None
        )
        if ratio_name:
            active_norm_ratio = str(ratio_name).replace("\\", "/")
    except Exception:
        active_norm_ratio = "205Tl/203Tl"

    st.markdown("**Pb-Tl contributor values**")
    st.caption(
        "For SRM 997: Standard uncertainty unavailable from source metadata. User-provided value required."
    )
    return float(
        st.number_input(
            f"Tl normalisation-ratio standard uncertainty ({active_norm_ratio}, absolute ratio)",
            min_value=0.0,
            value=float(pb_tl_norm_ratio_u_abs),
            step=1e-8,
            format="%.10f",
            key="uc_pb_tl_norm_ratio_u_abs",
            on_change=_enable_contributor_when_positive,
            args=("uc_pb_tl_norm_ratio_u_abs", "u_norm_ref"),
            help=(
                "Absolute standard uncertainty of the accepted Tl normalization ratio. "
                "If a certificate gives expanded uncertainty U, enter U/k. "
                "Use an evaluation applicable to the selected material and ratio; "
                "do not transfer uncertainty from another reference material."
            ),
        )
    )


def _render_engine_a_repeatability_mode(
    current: UncertaintyConfig,
    *,
    std_repeatability_mode: str,
    pb_tl: bool = False,
) -> str:
    """Render Engine A SRM repeatability mode controls."""
    st.markdown("**Reference-material repeatability mode**")
    st.caption(
        "Choose how the reference-material session scatter enters the budget. "
        "SD uses the full scatter of reference-material session means. "
        "SE uses the standard error of those means when multiple reference-material runs are available. "
        "Selecting a mode automatically enables the reference-material repeatability "
        "contributor above."
    )
    _mode_idx = 0 if std_repeatability_mode == "sd" else 1

    def _on_sd_se_change() -> None:
        """Ensure SRM repeatability is enabled when the mode is changed."""
        key = "uc_contributor_u_std_repeatability" if pb_tl else "uc_control_enable_srm_repeatability"
        st.session_state[key] = True

    return str(
        st.radio(
            "Repeatability mode",
            options=["sd", "se"],
            format_func=lambda x: {
                "sd": "SD - full scatter of reference-material means",
                "se": "SE - standard error of reference-material means",
            }[x],
            index=_mode_idx,
            key="uc_std_repeatability_mode",
            horizontal=True,
            on_change=_on_sd_se_change,
        )
    )

"""CRM certified value diagnostics for the uncertainty tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from domain.uncertainty.scope import is_invalid_budget_scope, budget_scope_label, budget_scope_note

from config.settings import UncertaintyConfig
from domain.uncertainty.propagation import u_certified_value
from domain.uncertainty.runtime import _resolve_crm_certified_value
from ui.formatting import format_uncertainty
from ui.components.table_utils import style_status_column
from ui.runtime_budget_cache import get_cached_runtime_budget
from ui.tabs.uncertainty_sections.common import (
    _render_compact_summary,
)
from ui.utils import (
    format_sample_display_label,
    get_cycle_ranges,
    get_sample_state_key,
    render_runtime_budget_error,
)


def _render_crm_certified_value_section(
    all_samples: list,
    ratio_name: str,
    u_config: UncertaintyConfig,
    state,
) -> None:
    """Render CRM certified value (Type B) diagnostics for the selected ratio."""
    element_symbol = getattr(state, "element_symbol", "") or ""
    if not element_symbol and getattr(state, "element_config", None) is not None:
        element_symbol = getattr(state.element_config, "symbol", "") or ""

    if not u_config.is_contributor_enabled("u_crm", element_symbol=element_symbol):
        return

    st.divider()
    st.subheader("CRM Certified Value (Type B)")
    st.caption(
        "The certificate value and U/k are common for the selected ratio. "
        "The sample selector only changes the per-sample relative contribution "
        "and variance share shown below."
    )

    candidates = [
        sample
        for sample in all_samples
        if not sample.is_blank and not sample.metadata.get("excluded", False)
    ]
    if not candidates:
        st.info("No active samples available for CRM certified value diagnostics.")
        return

    sample_labels = {
        get_sample_state_key(sample): format_sample_display_label(sample)
        for sample in candidates
    }
    sample_options = [get_sample_state_key(sample) for sample in candidates]
    widget_key = "uncertainty_diag_crm_sample"
    if st.session_state.get(widget_key) not in sample_options:
        st.session_state[widget_key] = sample_options[0]

    selected_key = st.selectbox(
        "Sample for CRM diagnostic",
        options=sample_options,
        format_func=lambda key: sample_labels.get(key, str(key)),
        key=widget_key,
    )
    sample = next(
        (item for item in candidates if get_sample_state_key(item) == selected_key),
        None,
    )
    if sample is None:
        return

    cycle_ranges = get_cycle_ranges(state)
    filter_method = (
        state.processing_config.filter_method if state.processing_config else "None"
    )
    filter_threshold = (
        state.processing_config.get_active_filter_threshold()
        if state.processing_config
        else 2.0
    )
    drift_fit_info = (
        state.result.quality_metrics.get("drift_fit_info")
        if state.has_result and getattr(state.result, "quality_metrics", None)
        else None
    )

    try:
        budget = get_cached_runtime_budget(
            sample,
            ratio_name,
            element_config=state.element_config,
            processing_config=state.processing_config,
            uncertainty_config=u_config,
            all_samples=all_samples,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
            drift_fit_info=drift_fit_info,
            custom_contributor_library=state.custom_contributor_library,
            profile_defaults=getattr(state, "uncertainty_profile_defaults", None),
        )
    except Exception as exc:
        render_runtime_budget_error("CRM certified value", exc)
        return

    if budget is None:
        st.info("No runtime budget is available for the selected sample and ratio.")
        return

    if is_invalid_budget_scope(budget):
        st.info(f"{budget_scope_label(budget)}: {budget_scope_note(budget)}")
        return

    crm_contributor = next(
        (c for c in budget.contributors if c.name == "u_crm"),
        None,
    )
    if crm_contributor is None or not crm_contributor.is_active:
        st.info("The u_CRM contributor is not active for this budget.")
        return

    # Resolve reference material and certified value
    element_config = state.element_config
    processing_config = state.processing_config
    certified_record = _resolve_crm_certified_value(
        element_config, processing_config, ratio_name
    )
    ref_mat_name = (
        processing_config.reference_material
        or (element_config.reference_material if element_config else None)
        or "Unknown"
    )

    certified_value = getattr(budget, "certified_reference_value", None)
    if certified_value is None:
        certified_value = getattr(budget, "delta_reference_value", None)

    # Determine which certified ratio this is
    certified_ratio = "87Sr/86Sr"
    if element_config and hasattr(element_config, "symbol") and element_config.symbol != "Sr":
        # For non-Sr elements, use the selected ratio
        certified_ratio = ratio_name
    cert_u_expanded = (
        float(getattr(certified_record, "uncertainty", 0.0) or 0.0)
        if certified_record is not None else 0.0
    )
    cert_k = (
        float(getattr(certified_record, "k", 0.0) or 0.0)
        if certified_record is not None else 0.0
    )
    cert_u_standard = u_certified_value(cert_u_expanded, cert_k)

    _render_compact_summary(
        [
            ("Selected sample", format_sample_display_label(sample)),
            ("Ratio (reported)", ratio_name),
            ("Certified ratio", certified_ratio),
            ("Reference material", ref_mat_name),
            ("CRM u(x)", format_uncertainty(crm_contributor.value_abs) if crm_contributor.is_active else "0"),
            ("Certificate U", format_uncertainty(cert_u_expanded) if cert_u_expanded > 0 else "N/A"),
            ("Coverage k", f"{cert_k:g}" if cert_k > 0 else "N/A"),
        ]
    )

    # Detailed breakdown
    st.markdown("### Contribution Details")

    rows = [
        {
            "Contributor": crm_contributor.display_name,
            "Status": "ACTIVE" if crm_contributor.is_active else "MISSING",
            "u(x) absolute": format_uncertainty(crm_contributor.value_abs)
            if crm_contributor.is_active
            else "0",
            "u(x) relative (‰)": format_uncertainty(crm_contributor.value_rel_permil, unit="‰")
            if crm_contributor.is_active
            else "0 ‰",
            "Variance share (%)": round(crm_contributor.percentage_contribution, 1),
            "DoF": "∞"
            if crm_contributor.degrees_of_freedom == float("inf")
            else f"{float(crm_contributor.degrees_of_freedom):.1f}",
        }
    ]
    st.dataframe(style_status_column(pd.DataFrame(rows)), width="stretch", hide_index=True)

    # Equation
    st.markdown("### Propagation")
    st.markdown(
        "\n".join(
            [
                "Type B uncertainty from the CRM certificate.",
                f"Reference material: **{ref_mat_name}**",
                f"Certified ratio: **{certified_ratio}**",
                f"Reported ratio: **{ratio_name}**",
                f"Certificate expanded uncertainty U: **{format_uncertainty(cert_u_expanded)}**",
                f"Coverage factor k: **{cert_k:g}**" if cert_k > 0 else "Coverage factor k: **N/A**",
                f"Standard uncertainty u = U/k: **{format_uncertainty(cert_u_standard)}**",
                f"Budget contribution u(x): **{format_uncertainty(crm_contributor.value_abs)}**",
                f"Contribution to u_c: **{format_uncertainty(crm_contributor.value_rel_permil, unit='‰')}**",
            ]
        )
    )

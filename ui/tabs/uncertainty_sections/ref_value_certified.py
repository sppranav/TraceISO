"""Literature reference-value (e.g. GeoReM) diagnostics for the uncertainty tab.

Structurally mirrors ``crm_certified_value.py`` for the ``u_crm`` contributor,
but for the ``u_ref_value`` contributor: a static Type B uncertainty on the
accepted literature reference ratio (e.g. GeoReM consensus value for NIST SRM
987), independent of whichever CRM the user selected for anchoring.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from domain.uncertainty.scope import is_invalid_budget_scope, budget_scope_label, budget_scope_note

from config.constants import SR_GEOREM_REFERENCE_MATERIAL
from config.settings import UncertaintyConfig
from domain.elements.crm_utils import resolve_optional_certified_value
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


def _render_ref_value_certified_section(
    all_samples: list,
    ratio_name: str,
    u_config: UncertaintyConfig,
    state,
) -> None:
    """Render literature reference-value (Type B) diagnostics for the selected ratio."""
    element_symbol = getattr(state, "element_symbol", "") or ""
    if not element_symbol and getattr(state, "element_config", None) is not None:
        element_symbol = getattr(state.element_config, "symbol", "") or ""

    if not u_config.is_contributor_enabled("u_ref_value", element_symbol=element_symbol):
        return

    st.divider()
    st.subheader("Reference Value (Literature, Type B)")
    st.caption(
        "A fixed, literature-sourced standard uncertainty on the accepted "
        "reference ratio (e.g. GeoReM consensus value), read-only and "
        "independent of the selected CRM/anchoring material."
    )

    candidates = [
        sample
        for sample in all_samples
        if not sample.is_blank and not sample.metadata.get("excluded", False)
    ]
    if not candidates:
        st.info("No active samples available for reference-value diagnostics.")
        return

    sample_labels = {
        get_sample_state_key(sample): format_sample_display_label(sample)
        for sample in candidates
    }
    sample_options = [get_sample_state_key(sample) for sample in candidates]
    widget_key = "uncertainty_diag_ref_value_sample"
    if st.session_state.get(widget_key) not in sample_options:
        st.session_state[widget_key] = sample_options[0]

    selected_key = st.selectbox(
        "Sample for reference-value diagnostic",
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
        render_runtime_budget_error("Reference value (literature)", exc)
        return

    if budget is None:
        st.info("No runtime budget is available for the selected sample and ratio.")
        return

    if is_invalid_budget_scope(budget):
        st.info(f"{budget_scope_label(budget)}: {budget_scope_note(budget)}")
        return

    ref_contributor = next(
        (c for c in budget.contributors if c.name == "u_ref_value"),
        None,
    )
    if ref_contributor is None or not ref_contributor.is_active:
        st.info("The u_ref_value contributor is not active for this budget.")
        return

    element_config = state.element_config
    element_symbol = element_config.symbol if element_config else ""
    ref_material_name = SR_GEOREM_REFERENCE_MATERIAL

    certified_record = None
    if element_symbol == "Sr":
        certified_record = resolve_optional_certified_value(
            element_symbol, ref_material_name, ratio_name,
        )
    cert_u_standard = float(getattr(certified_record, "uncertainty", 0.0) or 0.0)
    cert_ratio_value = (
        float(getattr(certified_record, "value", 0.0)) if certified_record else None
    )

    _render_compact_summary(
        [
            ("Selected sample", format_sample_display_label(sample)),
            ("Ratio (reported)", ratio_name),
            ("Reference material", ref_material_name),
            (
                "Certified ratio value",
                f"{cert_ratio_value:.7g}" if cert_ratio_value is not None else "N/A",
            ),
            ("Ref. u(x)", format_uncertainty(ref_contributor.value_abs) if ref_contributor.is_active else "0"),
            ("Standard uncertainty u_c", format_uncertainty(cert_u_standard) if cert_u_standard > 0 else "N/A"),
        ]
    )

    st.markdown("### Contribution Details")

    rows = [
        {
            "Contributor": ref_contributor.display_name,
            "Status": "ACTIVE" if ref_contributor.is_active else "MISSING",
            "u(x) absolute": format_uncertainty(ref_contributor.value_abs)
            if ref_contributor.is_active
            else "0",
            "u(x) relative (‰)": format_uncertainty(ref_contributor.value_rel_permil, unit="‰")
            if ref_contributor.is_active
            else "0 ‰",
            "Variance share (%)": round(ref_contributor.percentage_contribution, 1),
            "DoF": "∞"
            if ref_contributor.degrees_of_freedom == float("inf")
            else f"{float(ref_contributor.degrees_of_freedom):.1f}",
        }
    ]
    st.dataframe(style_status_column(pd.DataFrame(rows)), width="stretch", hide_index=True)

    st.markdown("### Propagation")
    st.markdown(
        "\n".join(
            [
                "Static Type B uncertainty from a literature reference value.",
                f"Reference material: **{ref_material_name}**",
                f"Reported ratio: **{ratio_name}**",
                f"Standard uncertainty u_c (already combined, not expanded): "
                f"**{format_uncertainty(cert_u_standard)}**",
                f"Budget contribution u(x): **{format_uncertainty(ref_contributor.value_abs)}**",
                f"Contribution to u_c: **{format_uncertainty(ref_contributor.value_rel_permil, unit='‰')}**",
            ]
        )
    )

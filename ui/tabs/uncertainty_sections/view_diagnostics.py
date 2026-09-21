"""Diagnostics subview for the redesigned Uncertainty tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from domain.uncertainty.scope import is_invalid_budget_scope, budget_scope_label, budget_scope_note

from config.contributor_names import canonical_contributor_display_label
from config.settings import UncertaintyConfig
from ui.formatting import format_uncertainty
from ui.components.table_utils import style_status_column
from ui.runtime_budget_cache import get_cached_runtime_budget
from ui.tabs.uncertainty_sections import blank as blank_section
from ui.tabs.uncertainty_sections import crm_certified_value as crm_section
from ui.tabs.uncertainty_sections import drift as drift_section
from ui.tabs.uncertainty_sections import precision as precision_section
from ui.tabs.uncertainty_sections import ref_value_certified as ref_value_section
from ui.tabs.uncertainty_sections import repeatability as repeatability_section
from ui.tabs.uncertainty_sections import sr as sr_section
from ui.tabs.uncertainty_sections import shared_ui
from ui.tabs.uncertainty_sections.common import _render_compact_summary
from ui.tabs.uncertainty_sections.ui_context import DiagnosticSection, UncertaintyUiContext
from ui.utils import (
    format_sample_display_label,
    get_cycle_ranges,
    get_sample_state_key,
    render_runtime_budget_error,
)


def _format_dominant_contributor_label(name: str) -> str:
    """Return a canonical dominant label while preserving unrelated keys."""
    return canonical_contributor_display_label(name) or name or "-"


def render_diagnostics_view(
    *,
    samples: list,
    selected_ratio: str,
    context: UncertaintyUiContext,
    u_config: UncertaintyConfig,
    state,
) -> UncertaintyConfig:
    """Render one selected contributor diagnostic at a time."""
    st.subheader("Contributor Diagnostics")

    sections = list(context.diagnostic_sections)
    if not sections:
        st.info("No enabled contributor diagnostics are available for this uncertainty configuration.")
        return u_config

    # Honour a pending "Jump to diagnostic" request from Results before the
    # selectbox is instantiated, and drop any stale selection that no longer
    # maps to a section in the active engine context.
    shared_ui.consume_pending_diagnostic(context)
    section_keys = [section.key for section in sections]
    if st.session_state.get("uncertainty_diagnostic_select") not in section_keys:
        st.session_state["uncertainty_diagnostic_select"] = section_keys[0]

    # Rail-only aggregate: cache-backed, used solely for section share %.
    try:
        aggregate = shared_ui.session_budget_aggregate(
            samples,
            selected_ratio,
            u_config,
            state,
            context,
        )
    except Exception:
        aggregate = None

    rail_col, panel_col = st.columns([1, 3])
    with rail_col:
        selected_key = shared_ui.render_diagnostic_rail(
            sections,
            key="uncertainty_diagnostic_select",
            aggregate=aggregate,
        )
    section = next(section for section in sections if section.key == selected_key)

    with panel_col:
        u_config = _render_selected_diagnostic(
            section=section,
            samples=samples,
            selected_ratio=selected_ratio,
            u_config=u_config,
            state=state,
        )
    return u_config


def _render_selected_diagnostic(
    *,
    section: DiagnosticSection,
    samples: list,
    selected_ratio: str,
    u_config: UncertaintyConfig,
    state,
) -> UncertaintyConfig:
    """Dispatch to the bespoke panel for the selected diagnostic section."""
    if section.key == "precision":
        precision_section._render_measurement_precision_section(
            samples,
            selected_ratio,
            u_config,
            state,
        )
    elif section.key == "repeatability":
        updated = repeatability_section._render_standard_repeatability(
            samples,
            selected_ratio,
            u_config,
            state,
        )
        if updated != state.uncertainty_config:
            state.uncertainty_config = updated
            u_config = updated
    elif section.key == "blank":
        blank_section._render_blank_correction_section(
            samples,
            selected_ratio,
            u_config,
            state,
        )
    elif section.key == "drift":
        updated = drift_section._render_drift_section(
            u_config,
            samples,
            selected_ratio,
            state,
        )
        if updated != state.uncertainty_config:
            state.uncertainty_config = updated
            u_config = updated
    elif section.key == "sr_interference":
        sr_section._render_interference_section(
            samples,
            selected_ratio,
            u_config,
            state,
        )
    elif section.key == "sr_bias":
        sr_section._render_bias_checks_section(
            samples,
            selected_ratio,
            u_config,
            state,
        )
    elif section.key == "crm_certified_value":
        crm_section._render_crm_certified_value_section(
            samples,
            selected_ratio,
            u_config,
            state,
        )
    elif section.key == "ref_value_certified":
        ref_value_section._render_ref_value_certified_section(
            samples,
            selected_ratio,
            u_config,
            state,
        )
    else:
        _render_budget_contributor_snapshot(
            samples=samples,
            selected_ratio=selected_ratio,
            section=section,
            u_config=u_config,
            state=state,
        )

    return u_config


def _render_budget_contributor_snapshot(
    *,
    samples: list,
    selected_ratio: str,
    section: DiagnosticSection,
    u_config: UncertaintyConfig,
    state,
) -> None:
    """Fallback diagnostic for contributors that do not yet have a bespoke panel."""
    candidates = [
        sample
        for sample in samples
        if not sample.is_blank and not sample.metadata.get("excluded", False)
    ]
    if not candidates:
        st.info("No active samples are available for this diagnostic.")
        return

    sample_labels = {
        get_sample_state_key(sample): format_sample_display_label(sample)
        for sample in candidates
    }
    sample_options = [get_sample_state_key(sample) for sample in candidates]
    widget_key = f"uncertainty_diag_sample_{section.key}"
    if st.session_state.get(widget_key) not in sample_options:
        st.session_state[widget_key] = sample_options[0]
    selected_key = st.selectbox(
        "Sample for contributor diagnostic",
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
    filter_method = state.processing_config.filter_method if state.processing_config else "None"
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
            selected_ratio,
            element_config=state.element_config,
            processing_config=state.processing_config,
            uncertainty_config=u_config,
            all_samples=samples,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
            drift_fit_info=drift_fit_info,
            custom_contributor_library=state.custom_contributor_library,
            profile_defaults=getattr(state, "uncertainty_profile_defaults", None),
        )
    except Exception as exc:
        render_runtime_budget_error(section.label, exc)
        return

    if budget is None:
        st.info("No runtime budget is available for the selected sample and ratio.")
        return

    if is_invalid_budget_scope(budget):
        st.info(f"{budget_scope_label(budget)}: {budget_scope_note(budget)}")
        return

    names = set(section.contributor_names)
    rows = [
        contributor
        for contributor in budget.contributors
        if contributor.name in names
    ]
    if not rows:
        st.info("The selected budget does not contain this contributor.")
        return

    _render_compact_summary(
        [
            ("Selected sample", format_sample_display_label(sample)),
            ("Ratio", selected_ratio),
            ("Budget U", format_uncertainty(budget.expanded_abs)),
            (
                "Dominant",
                _format_dominant_contributor_label(budget.dominant_contributor),
            ),
        ]
    )

    display_rows = []
    for contributor in rows:
        display_rows.append(
            {
                "Contributor": contributor.display_name,
                "Status": "ACTIVE" if contributor.is_active else getattr(contributor, "state", "MISSING"),
                "u abs": format_uncertainty(contributor.value_abs) if contributor.is_active else "0",
                "u rel (‰)": format_uncertainty(contributor.value_rel_permil, unit="‰")
                if contributor.is_active
                else "0 ‰",
                "Variance share (%)": round(contributor.percentage_contribution, 1),
                "DoF": "∞"
                if contributor.degrees_of_freedom == float("inf")
                else f"{float(contributor.degrees_of_freedom):.1f}",
            }
        )

    st.dataframe(style_status_column(pd.DataFrame(display_rows)), width="stretch", hide_index=True)

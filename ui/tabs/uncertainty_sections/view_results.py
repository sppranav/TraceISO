"""Results subview for the redesigned Uncertainty tab."""

from __future__ import annotations

import streamlit as st

from config.settings import UncertaintyConfig
from ui.tabs.uncertainty_sections import shared_ui
from ui.tabs.uncertainty_sections import summary_detail as summary_detail_section
from ui.tabs.uncertainty_sections.ui_context import UncertaintyUiContext
from ui.utils import get_filtered_sample_caption


def render_results_view(
    *,
    samples: list,
    selected_ratio: str,
    selected_types: list[str],
    u_config: UncertaintyConfig,
    state,
    context: UncertaintyUiContext | None = None,
) -> None:
    """Render summary and selected-sample budget detail for the active ratio."""
    st.subheader("Budget Review")

    filtered_samples = [
        sample
        for sample in samples
        if sample.sample_type.upper() in selected_types
        and not sample.metadata.get("excluded", False)
    ]
    filtered_samples = summary_detail_section._filter_uncertainty_budget_samples(
        filtered_samples,
        u_config,
        state,
    )

    if not filtered_samples:
        st.warning("No samples match the selected filters.")
        return

    filtered_caption = get_filtered_sample_caption(
        showing=len(filtered_samples),
        total=len(samples),
    )
    if filtered_caption is not None:
        st.caption(filtered_caption)

    if context is not None:
        # Results owns the single aggregate call; it is cache-backed so the
        # summary table below reuses the same per-sample budgets.
        aggregate = shared_ui.session_budget_aggregate(
            filtered_samples,
            selected_ratio,
            u_config,
            state,
            context,
        )
        shared_ui.render_session_metric_strip(aggregate, u_config)

    budget_rows = summary_detail_section._render_summary_table(
        filtered_samples,
        selected_ratio,
        u_config,
        state,
    )

    if budget_rows:
        summary_detail_section._render_detail_selector(
            budget_rows,
            selected_ratio,
            u_config,
            state,
            context=context,
        )

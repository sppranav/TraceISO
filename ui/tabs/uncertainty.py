"""Uncertainty tab for TraceISO."""

from __future__ import annotations

import streamlit as st

from config.settings import sync_uncertainty_with_processing
from ui.components.custom_ratio import get_selected_ratios
from ui.state import get_state
from ui.tabs.uncertainty_sections import shared_ui
from ui.tabs.uncertainty_sections import view_configure
from ui.tabs.uncertainty_sections import view_diagnostics
from ui.tabs.uncertainty_sections import view_results
from ui.tabs.uncertainty_sections.config_controls import (
    reset_ssb_delta_contributor_widget_state,
)
from ui.tabs.uncertainty_sections.ui_context import (
    UncertaintyUiContext,
    built_in_contributors_for_context,
    resolve_ui_context,
)
from ui.utils import (
    format_isotope_label,
    settings_changed_since_processing,
    sticky_type_multiselect,
)


_SUBVIEW_KEY = "uncertainty_subview"
_SUBVIEW_RESET_TOKEN_KEY = "_uncertainty_subview_reset_token"
_SUBVIEWS = ("Configure", "Budget review", "Diagnostics")


def _built_in_contributors_for_engine(
    engine: str,
    u_config,
    state,
) -> list[tuple[str, str]]:
    """Backward-compatible wrapper for tests and older callers."""
    element_symbol = state.element_config.symbol if state.element_config else ""
    ui_kind = (
        "pb_tl"
        if engine == "pb_tl_external_normalization"
        else "sr_internal"
        if engine == "internal_normalization"
        else "ssb_delta"
    )
    context = UncertaintyUiContext(
        element_symbol=element_symbol,
        engine_key=engine,
        ui_kind=ui_kind,
        engine_label=ui_kind,
        ratio_list=(),
        default_ratio="",
        selected_ratio="",
        enabled_contributors=(),
        diagnostic_sections=(),
        reset_token=(),
    )
    return built_in_contributors_for_context(context, u_config, state)


def _sync_uncertainty_config_with_processing(state) -> None:
    """Keep persisted uncertainty settings coherent with processing settings."""
    if not state.processing_config:
        return
    previous = state.uncertainty_config
    synced = sync_uncertainty_with_processing(
        previous, state.processing_config
    )
    if synced is not previous:
        state.uncertainty_config = synced
        if (
            getattr(previous, "engine", "") == "ssb_delta"
            and (
                getattr(previous, "enable_ssb", False) != getattr(synced, "enable_ssb", False)
                or getattr(previous, "enable_delta", False) != getattr(synced, "enable_delta", False)
            )
        ):
            reset_ssb_delta_contributor_widget_state()


def _resolve_ratio_list(state) -> list[str]:
    selected_ratios = get_selected_ratios(state.samples)
    return sorted(selected_ratios)


def _default_ratio_index(state, ratio_list: list[str]) -> int:
    if state.element_config and state.element_config.primary_ratio:
        primary = state.element_config.primary_ratio
        if primary in ratio_list:
            return ratio_list.index(primary)
    return 0


def _reset_stale_subview(context: UncertaintyUiContext) -> None:
    selected = st.session_state.get(_SUBVIEW_KEY, "Configure")
    if selected == "Results":
        selected = "Budget review"
        st.session_state[_SUBVIEW_KEY] = selected

    previous = st.session_state.get(_SUBVIEW_RESET_TOKEN_KEY)
    if previous != context.reset_token:
        st.session_state[_SUBVIEW_RESET_TOKEN_KEY] = context.reset_token
        can_display = selected in _SUBVIEWS and not (
            selected == "Diagnostics" and not context.diagnostic_sections
        )
        if not can_display:
            st.session_state[_SUBVIEW_KEY] = "Configure"


def _format_output_mode(mode: str) -> str:
    return "Delta" if mode == "delta" else "Absolute ratio"


def _format_coverage(u_config) -> str:
    if u_config.coverage_method == "welch_satterthwaite":
        return "Welch-Satterthwaite"
    return f"Fixed k = {u_config.coverage_k:g}"


def _sample_type_options_for_context(
    available_types: list[str],
    context: UncertaintyUiContext,
) -> list[str]:
    """Return sample-type filter options appropriate for the active budget view."""
    if context.ui_kind == "sr_internal":
        return [sample_type for sample_type in available_types if sample_type != "BLK"]
    return list(available_types)


def _default_sample_types_for_context(
    options: list[str],
    context: UncertaintyUiContext,
) -> list[str]:
    """Return the default selected sample types for the active budget view."""
    if context.ui_kind == "sr_internal":
        preferred = [sample_type for sample_type in options if sample_type in {"STD", "SMP"}]
        return preferred or list(options)
    return list(options)


def _sync_uncertainty_type_filter(
    options: list[str],
    context: UncertaintyUiContext,
) -> None:
    """Reset the sticky type filter when the budget context or option set changes."""
    key = "sticky_uncertainty_type_filter"
    scope_key = f"{key}__scope"
    scope = (context.ui_kind, tuple(options))

    if st.session_state.get(scope_key) == scope:
        return

    st.session_state[key] = _default_sample_types_for_context(options, context)
    st.session_state[scope_key] = scope


def _render_header(
    *,
    samples: list,
    ratio_list: list[str],
    context: UncertaintyUiContext,
    u_config,
    state,
) -> tuple[str, list[str]]:
    """Render shared lightweight controls and return selected ratio/types."""
    available_types = sorted(set(sample.sample_type.upper() for sample in samples))
    sample_type_options = _sample_type_options_for_context(available_types, context)
    _sync_uncertainty_type_filter(sample_type_options, context)

    if st.session_state.get("uncertainty_ratio_select") not in ratio_list:
        st.session_state.pop("uncertainty_ratio_select", None)

    col_ratio, col_types, col_context = st.columns([2, 2, 3])
    with col_ratio:
        selected_ratio = st.selectbox(
            "Ratio",
            options=ratio_list,
            index=_default_ratio_index(state, ratio_list),
            format_func=format_isotope_label,
            key="uncertainty_ratio_select",
        )

    with col_types:
        selected_types = sticky_type_multiselect(
            "Budget review sample types",
            options=sample_type_options,
            key="sticky_uncertainty_type_filter",
        )

    with col_context:
        st.markdown(f"**Engine:** {context.engine_label}")
        st.caption(
            f"Output: {_format_output_mode(u_config.output_mode)} | "
            f"Coverage: {_format_coverage(u_config)}"
        )

    return selected_ratio, selected_types


def render_uncertainty_tab() -> None:
    """Render the Uncertainty tab."""
    state = get_state()

    if not state.has_result:
        st.info("Process the data first to see uncertainty budgets.")
        return

    result = state.result
    samples = result.samples
    from ui.components.sr_calibration_panel import render_sr_calibration_notice
    render_sr_calibration_notice(samples)

    if not samples:
        st.warning("No samples in processing result.")
        return

    load_errors = [
        ("custom uncertainty contributors", state.custom_contributor_library_error),
        ("uncertainty profiles", state.uncertainty_profiles_error),
    ]
    for label, error in load_errors:
        if error:
            st.warning(
                f"Could not load {label}; using safe built-in defaults. "
                f"Parser error: {error}"
            )

    if settings_changed_since_processing(state):
        st.warning(
            "Cycle selections, sample exclusions, types, or processing settings "
            "have changed since last processing. "
            "Go to **Session Configuration** and click **Execute Data Reduction** to update results."
        )

    _sync_uncertainty_config_with_processing(state)

    ratio_list = _resolve_ratio_list(state)
    if not ratio_list:
        st.warning("No ratio data available.")
        return

    default_ratio = ratio_list[_default_ratio_index(state, ratio_list)]
    selected_ratio_for_context = st.session_state.get(
        "uncertainty_ratio_select",
        default_ratio,
    )
    if selected_ratio_for_context not in ratio_list:
        selected_ratio_for_context = default_ratio
    context = resolve_ui_context(
        state,
        state.uncertainty_config,
        ratio_list=ratio_list,
        selected_ratio=selected_ratio_for_context,
        samples=samples,
    )
    selected_ratio, selected_types = _render_header(
        samples=samples,
        ratio_list=ratio_list,
        context=context,
        u_config=state.uncertainty_config,
        state=state,
    )

    # Re-resolve after the user-selected ratio is known; this matters for Pb-Tl
    # because a Pb session can have IIF data for some ratios but not others.
    context = resolve_ui_context(
        state,
        state.uncertainty_config,
        ratio_list=ratio_list,
        selected_ratio=selected_ratio,
        samples=samples,
    )
    _reset_stale_subview(context)
    # Apply any queued "Jump to diagnostic" navigation before the nav widget
    # is instantiated (post-widget writes raise StreamlitAPIException).
    shared_ui.consume_pending_subview(_SUBVIEW_KEY, _SUBVIEWS)

    selected_view = shared_ui.render_subview_nav(
        _SUBVIEWS,
        key=_SUBVIEW_KEY,
        default="Configure",
    )

    if selected_view == "Configure":
        updated = view_configure.render_configure_view(
            samples=samples,
            selected_ratio=selected_ratio,
            ratio_list=ratio_list,
            context=context,
            u_config=state.uncertainty_config,
            state=state,
        )
        if updated != state.uncertainty_config:
            state.uncertainty_config = updated
    elif selected_view == "Budget review":
        view_results.render_results_view(
            samples=samples,
            selected_ratio=selected_ratio,
            selected_types=selected_types,
            context=context,
            u_config=state.uncertainty_config,
            state=state,
        )
    elif selected_view == "Diagnostics":
        updated = view_diagnostics.render_diagnostics_view(
            samples=samples,
            selected_ratio=selected_ratio,
            context=context,
            u_config=state.uncertainty_config,
            state=state,
        )
        if updated != state.uncertainty_config:
            state.uncertainty_config = updated

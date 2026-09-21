"""Application shell for TraceISO."""

import streamlit as st
from packaging.version import InvalidVersion, Version

from ui.state import get_state
from ui.theme import get_theme
from ui.sidebar import render_sidebar
from ui.navigation import (
    MAIN_SECTION_KEY,
    MAIN_SECTION_LABELS,
    MAIN_SECTION_WIDGET_KEY,
    SESSION_KEY_LAST_EDITED_SAMPLE_NAME,
    SESSION_KEY_SAMPLE_SEARCH_FILTER,
    SESSION_KEY_THEME_NAME,
    consume_main_section_request,
    is_sticky_widget_key,
)
from ui.utils import finalize_processing_run, get_processing_status_message

_MIN_STREAMLIT = (1, 52, 1)


def _assert_streamlit_version() -> None:
    """item 71: assert Streamlit version meets the minimum required.

    TraceISO relies on sticky-widget suffix rules and other behaviour
    introduced in Streamlit 1.52.  An older version produces cryptic
    widget errors rather than a clear compatibility message.
    """
    import importlib.metadata
    try:
        raw = importlib.metadata.version("streamlit")
    except importlib.metadata.PackageNotFoundError:
        # Cannot determine version — skip assertion
        return
    try:
        actual = Version(raw)
    except InvalidVersion as exc:
        raise RuntimeError(
            f"TraceISO could not parse the installed Streamlit version {raw!r}."
        ) from exc
    minimum = Version(".".join(str(v) for v in _MIN_STREAMLIT))
    if actual < minimum:
        min_str = ".".join(str(v) for v in _MIN_STREAMLIT)
        actual_str = raw
        raise RuntimeError(
            f"TraceISO requires Streamlit >= {min_str} but found {actual_str}. "
            f"Upgrade with: pip install --upgrade streamlit"
        )


def configure_page() -> None:
    """Configure Streamlit page settings. Must be called first."""
    st.set_page_config(
        page_title="TraceISO",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_status_banner(state=None) -> None:
    """Render the single session-status strip below the application title."""
    if state is None:
        state = get_state()
    processing_status = get_processing_status_message(state)
    if processing_status is not None:
        level, msg = processing_status
        context = (
            f"{state.loaded_file or 'No file'} | "
            f"{state.element_symbol or 'Unknown element'} | "
            f"{len(state.samples)} samples"
        )
        msg = f"{context} — {msg}"
        if level == "warning":
            st.warning(msg)
        elif level == "info":
            st.info(msg)
        else:
            st.success(msg)


def render_header() -> None:
    """Render the page header."""
    state = get_state()

    st.title("TraceISO")
    render_status_banner(state)


def _keep_sticky_widgets_alive() -> None:
    """Prevent Streamlit from evicting Results-tab widget state.

    With lazy section rendering, a Results widget is not instantiated on
    runs where another section is active, so Streamlit garbage-collects
    its session-state entry and the widget resets to its default on
    return. Reasserting the value every run marks it as user-set and
    keeps it (the same technique the main-section route uses).
    """
    for key in list(st.session_state.keys()):
        if is_sticky_widget_key(key):
            st.session_state[key] = st.session_state[key]


def render_tabs() -> None:
    """Render only the selected main section (lazy rendering)."""
    _ = get_state()

    _keep_sticky_widgets_alive()
    consume_main_section_request()

    section_renderers = {
        "Session Configuration": _render_overview_tab,
        "Sample Inspector": _render_inspector_tab,
        "Instrumental Drift": _render_drift_tab,
        "Results & Statistics": _render_results_tab,
        "Uncertainty Budgets": _render_uncertainty_tab,
        "Export": _render_export_tab,
    }

    if MAIN_SECTION_KEY not in st.session_state:
        st.session_state[MAIN_SECTION_KEY] = "Session Configuration"
    if st.session_state[MAIN_SECTION_KEY] not in MAIN_SECTION_LABELS:
        st.session_state[MAIN_SECTION_KEY] = "Session Configuration"

    # Keep widget state separate from the persisted route state to avoid
    # unrelated reruns forcing a fallback section.
    if (
        MAIN_SECTION_WIDGET_KEY not in st.session_state
        or st.session_state[MAIN_SECTION_WIDGET_KEY] not in MAIN_SECTION_LABELS
    ):
        st.session_state[MAIN_SECTION_WIDGET_KEY] = st.session_state[MAIN_SECTION_KEY]

    labels = list(section_renderers.keys())
    selected = st.radio(
        "Section",
        options=labels,
        horizontal=True,
        key=MAIN_SECTION_WIDGET_KEY,
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="main-nav-overflow-cue" aria-hidden="true">'
        'Scroll for more sections &rarr;</div>',
        unsafe_allow_html=True,
    )
    st.session_state[MAIN_SECTION_KEY] = selected
    _render_selected_section(st.session_state[MAIN_SECTION_KEY], section_renderers)


def _render_selected_section(selected: str, section_renderers: dict) -> None:
    """Render a single selected section from a renderer map."""
    renderer = section_renderers.get(selected)
    if renderer is not None:
        renderer()


def _commit_sample_classification_changes(updated_samples) -> None:
    """Persist classification edits and invalidate stale processed results."""
    state = get_state()
    state.samples = updated_samples
    state.invalidate_processed_result()


def _render_overview_tab() -> None:
    """Render the Session Configuration tab."""
    from ui.components.run_sequence import render_run_sequence
    from ui.components.sample_editor import (
        apply_batch_type,
        apply_filtered_type,
        clear_sample_editor_state,
        render_batch_type_buttons,
        render_sample_editor,
        render_search_assign,
    )
    from ui.components.settings_panel import render_processing_settings
    from ui.components.workspace_ui import render_compact_metadata, render_workflow_progress, workspace_panel

    state = get_state()

    if not state.has_data:
        st.info("Upload an HDF5 file using the sidebar to get started.")
        return

    st.markdown("### Prepare this analytical session")
    st.caption("Classify samples, configure processing, and execute data reduction.")
    progress_slot = st.empty()

    # Classification and processing share the width; Run sits beneath both.
    # The keyed wrapper lets the theme stack the two panels when narrow.
    with st.container(key="session_layout"):
        classify_col, workflow_col = st.columns(2, gap="medium")
    with classify_col, workspace_panel(
        "Classify samples",
        subtitle="Review the detected sample classifications.",
        eyebrow="1 · Classify",
        key="session_classify",
    ):
        type_counts: dict[str, int] = {}
        for sample in state.samples:
            sample_type = sample.sample_type.upper()
            type_counts[sample_type] = type_counts.get(sample_type, 0) + 1
        render_compact_metadata(
            (
                ("SMP", type_counts.get("SMP", 0)),
                ("STD", type_counts.get("STD", 0)),
                ("BLK", type_counts.get("BLK", 0)),
            )
        )

        # Show feedback for last edited sample (auto-clears when Inspector tab is opened)
        last_edited = st.session_state.get(SESSION_KEY_LAST_EDITED_SAMPLE_NAME)
        if last_edited:
            st.success(f"Updated **{last_edited}**. Go to **Sample Inspector** to view.")

        # Batch buttons
        batch_type = render_batch_type_buttons()
        if batch_type:
            _commit_sample_classification_changes(
                apply_batch_type(state.samples, batch_type)
            )
            clear_sample_editor_state()
            st.rerun()

        # Search & assign
        search_term, assign_type = render_search_assign(state.samples)
        if search_term and assign_type:
            _commit_sample_classification_changes(
                apply_filtered_type(state.samples, search_term, assign_type)
            )
            clear_sample_editor_state()
            st.rerun()

        # Editable table — filtered to match the live search term (if any)
        active_filter = st.session_state.get(
            SESSION_KEY_SAMPLE_SEARCH_FILTER,
            "",
        ).strip()
        with st.form("sample_classification_editor_form", clear_on_submit=False):
            updated_samples, changed = render_sample_editor(
                state.samples,
                filter_term=active_filter,
                height=320,
            )
            col_apply, col_discard = st.columns(2)
            with col_apply:
                apply_table_changes = st.form_submit_button(
                    "Apply Table Changes",
                    type="primary",
                    width="stretch",
                )
            with col_discard:
                discard_table_changes = st.form_submit_button(
                    "Discard Pending Changes",
                    width="stretch",
                )

        if discard_table_changes:
            clear_sample_editor_state()
            st.rerun()

        if apply_table_changes and changed:
            _commit_sample_classification_changes(updated_samples)
            clear_sample_editor_state()
            st.rerun()

    with workflow_col, workspace_panel(
        "Configure processing",
        subtitle="Choose ratios and processing settings for this session.",
        eyebrow="2 · Configure",
        key="session_configure",
    ):
        from ui.components.custom_ratio import (
            render_custom_ratio_creator,
            render_ratio_manager,
        )

        render_ratio_manager(state.samples, key="ratio_mgr_overview")
        new_ratio = render_custom_ratio_creator(
            state.samples,
            key="custom_ratio_overview",
        )
        if new_ratio:
            st.rerun()

        new_config = render_processing_settings(
            state.element_config,
            state.processing_config,
            show_heading=False,
        )
        if new_config != state.processing_config:
            state.processing_config = new_config
            # Re-render from the newly committed configuration so the
            # page-level banner and Run panel always report one status.
            st.rerun()

    processing_status = get_processing_status_message(state)
    status_level, status_message = processing_status or (
        "info",
        "Ready to configure this analytical session.",
    )
    with progress_slot:
        render_workflow_progress(
            ("Classify", "Configure", "Run"),
            completed_through=None,
        )

    with workspace_panel(
        "Run data reduction",
        subtitle="Execute once after reviewing the visible classification and settings.",
        eyebrow="3 · Run",
        key="session_run",
    ):
        if state.has_result and status_level == "warning":
            st.warning(
                "**Results need updating**\n\n"
                "Current settings differ from the last processed configuration."
            )
        else:
            st.caption(status_message)
        if st.button(
            "Execute Data Reduction",
            type="primary",
            width="stretch",
            disabled=not state.has_data,
        ):
            with st.spinner("Processing..."):
                _run_processing()

        if state.has_result and state.warnings:
            with st.expander(f"Warnings ({len(state.warnings)})", expanded=False):
                for warn in state.warnings:
                    st.warning(warn)

    # Retain the existing visualization as secondary content; it is still
    # constructed exactly once and no longer displaces the primary workflow.
    with st.expander("Run Sequence", expanded=False):
        render_run_sequence(state.samples, state.selected_sample_idx)


def _run_processing() -> None:
    """Run the processing pipeline."""
    state = get_state()

    if not state.has_data or not state.element_config:
        st.error("No data loaded or element not detected.")
        return

    try:
        from domain.processing_service import process_samples
        from ui.diagnostics import timed
        from ui.utils import get_cycle_ranges

        result = timed("Data reduction")(process_samples)(
            state.samples,
            state.element_config,
            state.processing_config,
            uncertainty_config=state.uncertainty_config,
            profile_defaults=state.uncertainty_profile_defaults,
            cycle_ranges=get_cycle_ranges(state),
        )
        finalize_processing_run(state, result)
        # Rerun so render_header() sees the completed result from the start of
        # the next render pass (without this, the header banner — rendered before
        # the button — shows the pre-run "Ready to execute" state).
        st.rerun()

    except Exception as e:
        st.error(f"Processing error: {e}")
        if state.dev_mode:
            from ui.diagnostics import record_error
            record_error("ui/app_shell.py", e)
            st.exception(e)


def _render_inspector_tab() -> None:
    """Render the Sample Inspector tab."""
    from ui.tabs.inspector import render_inspector_tab
    render_inspector_tab()


def _render_drift_tab() -> None:
    """Render the Instrumental Drift tab."""
    from ui.tabs.drift import render_drift_tab
    render_drift_tab()


def _render_results_tab() -> None:
    """Render the Results & Statistics tab."""
    from ui.tabs.results import render_results_tab
    render_results_tab()


def _render_uncertainty_tab() -> None:
    """Render the Uncertainty tab."""
    from ui.tabs.uncertainty import render_uncertainty_tab
    render_uncertainty_tab()


def _render_export_tab() -> None:
    """Render the Export tab."""
    from ui.tabs.export import render_export_tab
    render_export_tab()


def run() -> None:
    """Main entry point for the application."""
    _assert_streamlit_version()  # item 71
    configure_page()

    theme_name = st.session_state.get(SESSION_KEY_THEME_NAME, "light")
    theme = get_theme(theme_name)
    theme.inject_css()

    sidebar_error = None
    try:
        render_sidebar()
    except Exception as exc:
        sidebar_error = exc

    # item 63: protect header rendering — an exception in get_processing_status_message
    # or _build_ratio_summary must not produce a raw Streamlit error page.
    try:
        render_header()
    except Exception as _header_exc:
        state = get_state()
        st.error("Header failed to render. Please refresh or contact support.")
        if getattr(state, "dev_mode", False):
            from ui.diagnostics import record_error
            record_error("ui/app_shell.py", _header_exc)
            st.exception(_header_exc)
        else:
            import logging
            logging.getLogger(__name__).exception("render_header() failed: %s", _header_exc)

    if sidebar_error:
        state = get_state()
        st.error("Sidebar failed to render. The main workspace remains available.")
        import logging
        logging.getLogger(__name__).exception(
            "render_sidebar() failed: %s",
            sidebar_error,
            exc_info=(type(sidebar_error), sidebar_error, sidebar_error.__traceback__),
        )
        if getattr(state, "dev_mode", False):
            from ui.diagnostics import record_error
            record_error("ui/app_shell.py", sidebar_error)
            st.exception(sidebar_error)

    # item 63: protect tab rendering — each section renderer is independent;
    # an exception in one section must not take down the whole UI.
    try:
        render_tabs()
    except Exception as _tab_exc:
        state = get_state()
        st.error("Tab rendering failed. Try switching sections or refreshing.")
        if getattr(state, "dev_mode", False):
            from ui.diagnostics import record_error
            record_error("ui/app_shell.py", _tab_exc)
            st.exception(_tab_exc)
        else:
            import logging
            logging.getLogger(__name__).exception("render_tabs() failed: %s", _tab_exc)

    from ui.diagnostics import render_diagnostics
    render_diagnostics(get_state())

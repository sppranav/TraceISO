"""Shared navigation constants for the main TraceISO workspace."""

MAIN_SECTION_KEY = "main_section_nav"
MAIN_SECTION_WIDGET_KEY = "main_section_nav_widget"
MAIN_SECTION_REQUEST_KEY = "_main_section_navigation_request"
SESSION_KEY_PROCESSED_EXCLUSIONS = "_processed_exclusions"
SESSION_KEY_PROCESSED_MANUAL_EXCLUSIONS = "_processed_manual_exclusions"
SESSION_KEY_PROCESSED_TYPES = "_processed_types"
SESSION_KEY_PROCESSED_PROCESSING_CONFIG = "_processed_processing_config"
SESSION_KEY_LAST_EDITED_SAMPLE_NAME = "_last_edited_sample_name"
SESSION_KEY_SAMPLE_SEARCH_FILTER = "sample_search_filter"
SESSION_KEY_THEME_NAME = "theme_name"
SESSION_KEY_SUPPRESS_RELOAD = "_suppress_reload"
MAIN_SECTION_LABELS = [
    "Session Configuration",
    "Sample Inspector",
    "Instrumental Drift",
    "Results & Statistics",
    "Uncertainty Budgets",
    "Export",
]


def request_main_section(section: str, *, session_state=None) -> None:
    """Queue a route change for consumption before the navigation widget exists."""
    if section not in MAIN_SECTION_LABELS:
        raise ValueError(f"Unknown main section {section!r}.")
    if session_state is None:
        import streamlit as st

        session_state = st.session_state
    session_state[MAIN_SECTION_REQUEST_KEY] = section


def consume_main_section_request(*, session_state=None) -> str | None:
    """Apply and return a queued route change before widget construction."""
    if session_state is None:
        import streamlit as st

        session_state = st.session_state
    requested = session_state.pop(MAIN_SECTION_REQUEST_KEY, None)
    if requested not in MAIN_SECTION_LABELS:
        return None
    session_state[MAIN_SECTION_KEY] = requested
    session_state[MAIN_SECTION_WIDGET_KEY] = requested
    return requested

# Widget keys that must survive navigating away from a lazily-rendered section
# opt in with this prefix. Streamlit garbage-collects widget state when the
# widget is absent on a script run; these keys are reasserted every rerun.
STICKY_WIDGET_PREFIXES = ("sticky_",)
# Keys excluded from sticky reassertion: transient downloads, editors, and button widgets.
# Streamlit 1.52+ raises StreamlitValueAssignmentNotAllowedError when
# application code writes a value for button/data-editor widget keys, so those keys
# must never be reasserted by _keep_sticky_widgets_alive.
STICKY_WIDGET_TRANSIENT_SUFFIXES = (
    "_download",
    "_button",
    "_matches",
    "_bulk",
    "_editor",
)


def is_sticky_widget_key(key: object) -> bool:
    """Return True for persistent lazy-rendered widget keys.

    Button-like widgets and data editors expose transient state that Streamlit
    does not allow application code to write back into ``st.session_state``.
    """
    return (
        isinstance(key, str)
        and key.startswith(STICKY_WIDGET_PREFIXES)
        and not key.endswith(STICKY_WIDGET_TRANSIENT_SUFFIXES)
        and "_jump_" not in key
    )


def resolve_subview_selection(
    *,
    key: str,
    options,
    default=None,
    session_state=None,
):
    """Resolve and repair the stored selection for a keyed subview control.

    A102: navigation state is application state, so the owning controller reads
    and repairs it here and hands the result to the presentation component. A
    Deselection restores the last valid view. A renamed or removed option falls
    back to the default. Repairs happen before widget creation because
    Streamlit refuses post-widget writes to a widget key.
    """
    choices = list(options)
    if not choices:
        raise ValueError("Workspace subnavigation requires at least one option")
    if default is None:
        default = choices[0]
    if default not in choices:
        raise ValueError("Workspace subnavigation default must be an option")

    if session_state is None:
        import streamlit as st

        session_state = st.session_state

    last_key = f"_subview_last_selection_{key}"
    current = session_state.get(key)
    if current is None:
        current = session_state.get(last_key)
    if current not in choices:
        current = default
    if session_state.get(key) != current:
        session_state[key] = current
    session_state[last_key] = current
    return current

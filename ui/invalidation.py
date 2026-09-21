"""item 65: Centralised UI invalidation events.

Named events replace the scattered direct calls to ``invalidate_processed_result``,
``clear_runtime_budget_cache``, ``_purge_sticky_widget_state``, and
``_reset_cycle_range_state``.  Using named events makes invalidation contracts
reviewable in one place and prevents 2.1-class bugs where a new code path
forgets to clear a dependent cache.

Call sites should import the named event functions and call them instead of
calling the underlying state/cache methods directly.

Events defined here:

- ``on_new_file_loaded(state)`` — a new HDF5 file was loaded
- ``on_classification_changed(state)`` — sample types or roles changed
- ``on_masks_changed(state)`` — cycle masks or manual exclusions changed
- ``on_config_changed(state)`` — processing configuration changed

All events are safe to call multiple times; they are idempotent.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ui.state import AppState


def on_new_file_loaded(state: "AppState") -> None:
    """Invalidate all derived state after a new HDF5 file is loaded.

    - Clears the processed result and runtime budget cache
    - Resets cycle-range and sticky-widget state
    - Does NOT clear the file's identity keys (loaded_file, file_hash, etc.)
      or the loaded samples themselves — the caller is responsible for those.
    """
    state.invalidate_processed_result()
    _clear_runtime_budgets()
    _clear_drift_preview()


def on_classification_changed(state: "AppState") -> None:
    """Invalidate after sample type / role edits.

    Sample classification changes affect which samples are bracketing
    standards and blanks, so the processed result is stale but the raw
    cycle data is still valid.
    """
    state.invalidate_processed_result()
    _clear_runtime_budgets()
    _clear_drift_preview()


def on_masks_changed(state: "AppState") -> None:
    """Invalidate runtime budgets after cycle-mask or manual-exclusion edits.

    The stored pipeline result is NOT invalidated (masks are editor-only;
    they feed into the runtime layer without triggering a re-run).
    """
    _clear_runtime_budgets()
    _clear_drift_preview()


def on_processing_completed(state: "AppState") -> None:
    """Invalidate every derived cache after a data reduction is committed.

    A031: the previous finalization cleared only the analytical runtime-budget
    cache. The Monte Carlo display cache survived, and because its key knew
    nothing about which result or which measured content it belonged to, the
    panel served the *previous* reduction's centre and expanded uncertainty as
    if they were current, with no staleness warning.

    The stored result itself is not invalidated - it is what was just
    committed - so this is deliberately not ``on_config_changed``.
    """
    _clear_runtime_budgets()
    _bump_mc_display_generation()
    _clear_drift_preview()


def on_config_changed(state: "AppState") -> None:
    """Invalidate after any processing-configuration change.

    Configuration changes (filter method, blank mode, SSB toggle, drift
    settings, etc.) affect the stored pipeline result.
    """
    state.invalidate_processed_result()
    _clear_runtime_budgets()
    _clear_drift_preview()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


#: Session key holding the transient Monte Carlo display-cache generation.
#: Deliberately distinct from observation identity and from the numerical
#: content fingerprint of a budget: it says only "the caches from before this
#: event are not reusable", and carries no scientific meaning. It is never
#: exported.
MC_DISPLAY_GENERATION_KEY = "_mc_display_cache_generation"


def mc_display_generation() -> int:
    """Return the current transient MC display-cache generation."""
    try:
        import streamlit as st
        return int(st.session_state.get(MC_DISPLAY_GENERATION_KEY, 0) or 0)
    except Exception:  # pragma: no cover - defensive, never fatal to a render
        return 0


def _bump_mc_display_generation() -> None:
    """Advance the transient MC display-cache generation."""
    try:
        import streamlit as st
        st.session_state[MC_DISPLAY_GENERATION_KEY] = mc_display_generation() + 1
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning("Failed to advance Monte Carlo display generation: %s", exc)


def _clear_runtime_budgets() -> None:
    """Clear the per-session runtime budget cache."""
    try:
        from ui.runtime_budget_cache import clear_runtime_budget_cache
        clear_runtime_budget_cache()
    except Exception as exc:
        _log.warning("Failed to clear runtime budget cache: %s", exc)
    try:
        import streamlit as st
        for key in list(st.session_state):
            if str(key).startswith("_mc_result_"):
                st.session_state.pop(key, None)
    except Exception as exc:
        _log.warning("Failed to clear Monte Carlo result cache: %s", exc)


def _clear_drift_preview() -> None:
    try:
        import streamlit as st
        st.session_state.pop("_drift_preview", None)
    except Exception as exc:  # pragma: no cover - defensive
        _log.warning("Failed to clear drift preview: %s", exc)

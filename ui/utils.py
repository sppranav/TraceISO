"""Shared UI utilities for session-derived state."""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st

from domain.filters.outlier import sample_cycle_key
from domain.models import Sample
from domain.ratio_utils import normalize_ratio_name
from file_io.sanitize import format_isotope_label, to_superscript
from ui.navigation import (
    SESSION_KEY_PROCESSED_EXCLUSIONS,
    SESSION_KEY_PROCESSED_MANUAL_EXCLUSIONS,
    SESSION_KEY_PROCESSED_TYPES,
    SESSION_KEY_PROCESSED_PROCESSING_CONFIG,
)
from ui.state import AppState, get_state


# Default delta reference standards per element (first CRM in library)
_DELTA_REFERENCES: Dict[str, str] = {
    "Li": "LSVEC",
    "B": "NIST SRM 951",
    "Sr": "NIST SRM 987",
    "Mg": "ERM-AE143",
    "Cd": "BAM-I012",
    "Pb": "NIST SRM 981",
}


def get_plot_config(state: Optional[AppState] = None) -> Dict[str, Any]:
    """Read plot configuration safely inside or outside Streamlit runtime."""
    try:
        session_config = getattr(st, "session_state", {}).get("plot_config")
        if isinstance(session_config, dict):
            return dict(session_config)
    except Exception:
        pass
    try:
        return dict((state or get_state()).plot_config)
    except Exception:
        return {}


def decimals_for_uncertainty(
    errors,
    *,
    default: int = 6,
    min_dec: int = 2,
    max_dec: int = 8,
) -> int:
    """Choose fixed decimals with one displayed digit below the error scale."""
    values = np.asarray(list(errors), dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return int(default)
    decimals = 1 - np.floor(np.log10(np.median(values)))
    return int(np.clip(decimals, min_dec, max_dec))


def blank_correction_active(sample: Sample, state: Optional[AppState] = None) -> bool:
    """Return whether *this sample's run* actually applied a blank correction.

    ``Sample.used_blanks`` is the record of what the run did: the blank correction
    writes it only when a bracketing blank was found and subtracted, and leaves it
    empty for ``blank_mode="none"`` and for samples where no blank was located.

    Two things are deliberately not consulted. ``ProcessingConfig.blank_mode`` is a
    live session field that drifts from the run the moment the user changes it
    without reprocessing — gating on it let an unrelated session change hide a
    produced layer, or claim a correction that never ran. And
    ``blank_corrected_ratios`` is populated unconditionally by the pipeline (in
    ``none`` mode it simply equals the uncorrected layer), so its presence proves
    nothing.

    ``state`` is accepted for call-site compatibility and is intentionally unused:
    the answer belongs to the run, not to the current session view.
    """
    return bool(sample.used_blanks)


def format_name(name: str) -> str:
    """Format isotope/ratio names with HTML superscripts for Plotly.

    Examples:
        '87Sr/86Sr' -> '<sup>87</sup>Sr/<sup>86</sup>Sr'
        '206Pb'     -> '<sup>206</sup>Pb'
    """
    return re.sub(r"(\d+)([a-zA-Z]+)", r"<sup>\1</sup>\2", name)


def get_sample_state_key(sample: Sample) -> str:
    """Return the canonical collision-safe state key for a sample."""
    return sample_cycle_key(sample)


def format_sample_display_label(sample: Sample) -> str:
    """Return a collision-safe, human-readable sample label."""
    name = str(getattr(sample, "name", "") or "")
    run_number = getattr(sample, "run_number", None)
    if run_number is None:
        return name
    return f"#{int(run_number)} - {name}"


def get_filtered_sample_caption(*, showing: int, total: int) -> Optional[str]:
    """Return a caption describing a filtered sample subset, if useful."""
    showing = int(showing)
    total = int(total)
    if total <= 0 or showing >= total:
        return None
    return (
        f"Showing {showing} of {total} samples "
        f"after type filter and exclusions."
    )


def get_active_internal_normalization_ratio(
    processing_config: Any,
    element_config: Any,
) -> Optional[str]:
    """Return the active internal-normalization ratio, if one is configured."""
    if processing_config is None or element_config is None:
        return None
    if not getattr(processing_config, "apply_mass_bias_correction", False):
        return None
    return normalize_ratio_name(
        getattr(processing_config, "normalization_ratio_override", None)
        or getattr(element_config, "normalization_ratio", None)
    )


def _element_from_ratio(ratio_name: str) -> Optional[str]:
    """Extract element symbol from a ratio name like '87Sr/86Sr' -> 'Sr'."""
    match = re.search(r"\d+([A-Z][a-z]?)", ratio_name)
    return match.group(1) if match else None


def format_delta_label(ratio_name: str, reference: Optional[str] = None, *,
                       processing_config: Any = None, element_config: Any = None) -> str:
    """Format delta notation with reference standard for display.

    item 68: prefer ``processing_config.reference_material or
    element_config.reference_material`` over the static ``_DELTA_REFERENCES``
    fallback map so axis labels always reflect the actual session reference.
    """
    formatted = _format_delta_ratio_label(ratio_name)

    if reference is None:
        # item 68: resolve from active config before falling back to static map
        reference = (
            getattr(processing_config, "reference_material", None)
            or getattr(element_config, "reference_material", None)
        )
        if not reference:
            element = _element_from_ratio(ratio_name)
            reference = _DELTA_REFERENCES.get(element, "") if element else ""

    if reference:
        return f"\u03b4{formatted} (\u2030 vs {reference})"
    return f"\u03b4{formatted} (\u2030)"


def format_delta_html(ratio_name: str, reference: Optional[str] = None, *,
                      processing_config: Any = None, element_config: Any = None) -> str:
    """Format delta notation with HTML superscripts for Plotly.

    Returns string like '\u03b4<sup>11</sup>B (\u2030 vs NIST SRM 951)'.

    item 68: resolve reference from active config before falling back to
    ``_DELTA_REFERENCES``.
    """
    formatted = _format_delta_ratio_html(ratio_name)

    if reference is None:
        reference = (
            getattr(processing_config, "reference_material", None)
            or getattr(element_config, "reference_material", None)
        )
        if not reference:
            element = _element_from_ratio(ratio_name)
            reference = _DELTA_REFERENCES.get(element, "") if element else ""

    if reference:
        return f"\u03b4{formatted} (\u2030 vs {reference})"
    return f"\u03b4{formatted} (\u2030)"


def _split_isotope_token(token: str) -> Optional[tuple[str, str]]:
    match = re.fullmatch(r"\s*(\d+)([A-Za-z]+)\s*", token)
    return match.groups() if match else None


def _format_delta_ratio_label(ratio_name: str) -> str:
    """Show both isotope masses, sharing a common element symbol."""
    numerator, separator, denominator = str(ratio_name).partition("/")
    num = _split_isotope_token(numerator)
    den = _split_isotope_token(denominator) if separator else None
    if num and den and num[1] == den[1]:
        return f"{to_superscript(num[0])}/{to_superscript(den[0])}{num[1]}"
    return format_isotope_label(str(ratio_name))


def _format_delta_ratio_html(ratio_name: str) -> str:
    numerator, separator, denominator = str(ratio_name).partition("/")
    num = _split_isotope_token(numerator)
    den = _split_isotope_token(denominator) if separator else None
    if num and den and num[1] == den[1]:
        return f"<sup>{num[0]}</sup>/<sup>{den[0]}</sup>{num[1]}"
    return format_name(str(ratio_name))


# item 69: single session-state key for all per-sample cycle ranges.
# Replaces the pattern of individual ``cycle_range_{sample_name}`` keys
# which would fail silently for sample names starting with "slider_".
_CYCLE_RANGES_STATE_KEY = "_traceiso_cycle_ranges"


def get_cycle_ranges(state: Optional[AppState] = None) -> Dict[str, Tuple[int, int]]:
    """Return current global/per-sample cycle ranges from session state.

    item 69: reads from the structural ``_CYCLE_RANGES_STATE_KEY`` dict first,
    then falls back to the legacy ``cycle_range_<sample_name>`` keys for
    sessions that were initialised before the upgrade.
    """
    state = state or get_state()
    ranges: Dict[str, Tuple[int, int]] = {}

    global_enabled = bool(
        getattr(
            state,
            "global_cycle_range_enabled",
            st.session_state.get("global_cycle_range_enabled", False),
        )
    )
    if global_enabled:
        global_range = getattr(
            state,
            "global_cycle_range",
            st.session_state.get("global_cycle_range"),
        )
        if global_range is not None:
            ranges["__global__"] = tuple(global_range)
        return ranges

    # item 69: structural dict key (new path)
    structural = st.session_state.get(_CYCLE_RANGES_STATE_KEY)
    if isinstance(structural, dict) and structural:
        for sample_key, value in structural.items():
            if isinstance(value, (tuple, list)) and len(value) == 2:
                ranges[sample_key] = (int(value[0]), int(value[1]))
        return ranges

    # Legacy fallback: per-key parsing (pre-69 sessions)
    for key, value in st.session_state.items():
        if key.startswith("cycle_range_") and not key.startswith("cycle_range_slider_"):
            sample_name = key[len("cycle_range_"):]
            if isinstance(value, (tuple, list)) and len(value) == 2:
                ranges[sample_name] = (int(value[0]), int(value[1]))

    return ranges


def sticky_type_multiselect(label: str, *, options: list[str], key: str) -> list[str]:
    """Sample-type ``st.multiselect`` that persists across lazy reruns.

    The keep-alive sweep (``app_shell._keep_sticky_widgets_alive``) reasserts
    ``sticky_`` widget keys as user-set on every run. Passing ``default=`` to a
    widget whose key is already user-set trips Streamlit's
    "created with a default value but also had its value set via the Session
    State API" warning, so we seed the value once and rely on the key instead.
    Persisted types absent from the current dataset are pruned to avoid the
    "default value must exist in options" error when sample types change.
    """
    if not key.startswith("sticky_"):
        raise ValueError("sticky_type_multiselect keys must start with 'sticky_'")
    persisted = st.session_state.get(key)
    if persisted is None:
        st.session_state[key] = list(options)
    else:
        pruned = [t for t in persisted if t in options]
        if pruned != list(persisted):
            st.session_state[key] = pruned or list(options)
    return st.multiselect(label, options=options, key=key)


def clamp_cycle_range(
    cycle_range: Optional[Tuple[int, int]],
    *,
    max_cycles: int,
) -> Tuple[int, int]:
    """Clamp a cycle range to ``[1, max_cycles]`` and normalize ordering."""
    max_cycles = max(int(max_cycles), 1)
    if not isinstance(cycle_range, (tuple, list)) or len(cycle_range) != 2:
        return (1, max_cycles)

    start = max(1, min(int(cycle_range[0]), max_cycles))
    end = max(1, min(int(cycle_range[1]), max_cycles))
    if start > end:
        start, end = end, start
    return (start, end)


def render_runtime_budget_error(section_label: str, exc: Exception) -> None:
    """Show a concise in-tab runtime-budget failure message."""
    st.error(f"{section_label}: unable to compute runtime uncertainty ({exc}).")
    state = get_state()
    if state.dev_mode:
        from ui.diagnostics import record_error
        record_error("ui/utils.py", exc)
        st.exception(exc)


def _freeze_for_snapshot(value: Any) -> Any:
    """Convert nested processing settings into a stable comparable token."""
    if dataclasses.is_dataclass(value):
        return _freeze_for_snapshot(dataclasses.asdict(value))
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_for_snapshot(item))
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_snapshot(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_for_snapshot(item) for item in value))
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        return (
            "ndarray",
            str(arr.dtype),
            tuple(arr.shape),
            tuple(arr.ravel().tolist()),
        )
    if isinstance(value, np.generic):
        return value.item()
    return value


def processing_config_snapshot(config: Any) -> Any:
    """Return a stable freshness token for processing settings."""
    return _freeze_for_snapshot(config)


def manual_exclusions_snapshot(samples: Any) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
    """Return the session-local manual-cycle selection by observation identity."""
    rows = []
    for sample in samples or []:
        metadata = getattr(sample, "metadata", None)
        excluded = metadata.get("manual_exclusions", ()) if isinstance(metadata, dict) else ()
        cycles = tuple(sorted({int(cycle) for cycle in (excluded or ())}))
        rows.append((str(sample.observation_id), cycles))
    return tuple(sorted(rows))


def observation_state_snapshot(samples: Any) -> Tuple[Tuple[str, str, bool], ...]:
    """Return classification/exclusion state keyed by unique observation identity."""
    rows = []
    seen: set[str] = set()
    for sample in samples or []:
        observation_id = str(getattr(sample, "observation_id", "") or "")
        if not observation_id or observation_id in seen:
            # An ambiguous legacy session cannot safely claim freshness.
            return ()
        seen.add(observation_id)
        rows.append(
            (
                observation_id,
                str(getattr(sample, "sample_type", "")),
                bool(getattr(sample, "metadata", {}).get("excluded", False)),
            )
        )
    return tuple(sorted(rows))


def finalize_processing_run(state: AppState, result: Any) -> None:
    """Commit a successful pipeline result and refresh all derived UI state."""
    from ui.invalidation import on_processing_completed

    state.result = result
    state.warnings = list(getattr(result, "warnings", []) or [])
    # A031: the named event clears every derived cache, not just the analytical
    # one. Clearing the runtime budget cache alone left the Monte Carlo display
    # cache holding the previous reduction's numbers.
    on_processing_completed(state)
    observation_snapshot = observation_state_snapshot(getattr(state, "samples", None))
    st.session_state[SESSION_KEY_PROCESSED_EXCLUSIONS] = tuple(
        observation_id for observation_id, _sample_type, excluded in observation_snapshot if excluded
    )
    st.session_state[SESSION_KEY_PROCESSED_MANUAL_EXCLUSIONS] = (
        manual_exclusions_snapshot(getattr(state, "samples", None))
    )
    st.session_state[SESSION_KEY_PROCESSED_TYPES] = {
        observation_id: sample_type
        for observation_id, sample_type, _excluded in observation_snapshot
    }
    st.session_state[SESSION_KEY_PROCESSED_PROCESSING_CONFIG] = (
        processing_config_snapshot(getattr(state, "processing_config", None))
    )


def settings_changed_since_processing(state: Optional[AppState] = None) -> bool:
    """Check whether processing-relevant state changed after processing."""
    state = state or get_state()
    processed_exclusions = st.session_state.get(SESSION_KEY_PROCESSED_EXCLUSIONS)
    processed_manual_exclusions = st.session_state.get(
        SESSION_KEY_PROCESSED_MANUAL_EXCLUSIONS
    )
    processed_types = st.session_state.get(SESSION_KEY_PROCESSED_TYPES)
    processed_config = st.session_state.get(SESSION_KEY_PROCESSED_PROCESSING_CONFIG)

    if processed_exclusions is None or processed_types is None:
        return False
    if not state.samples:
        return False

    if processed_manual_exclusions is None:
        if bool(getattr(state, "has_result", getattr(state, "result", None) is not None)):
            return True
    elif manual_exclusions_snapshot(state.samples) != processed_manual_exclusions:
        # On the calibrated Pb route, target/QC exclusions are an intentional
        # runtime overlay.  The domain dependency check below distinguishes
        # those from standard or upstream edits; only the latter require a run.
        quality = getattr(getattr(state, "result", None), "quality_metrics", None) or {}
        if not quality.get("pb_standard_calibration") or pb_calibration_inputs_stale(state):
            return True

    current_observations = observation_state_snapshot(state.samples)
    if not current_observations:
        return True
    # Old name-keyed snapshots are deliberately expired: duplicate labels cannot
    # be reconstructed without guessing which observation they described.
    if isinstance(processed_exclusions, (set, frozenset)):
        return True
    current_exclusions = tuple(
        observation_id for observation_id, _sample_type, excluded in current_observations if excluded
    )
    if current_exclusions != tuple(processed_exclusions):
        return True

    if any(processed_types.get(observation_id) != sample_type for observation_id, sample_type, _ in current_observations):
        return True

    if processed_config is not None:
        current_config = processing_config_snapshot(
            getattr(state, "processing_config", None)
        )
        if current_config != processed_config:
            return True

    if pb_calibration_inputs_stale(state):
        return True

    return False


def pb_calibration_inputs_stale(state: AppState) -> bool:
    """Whether a processed Pb-standard calibration no longer describes the session.

    Catches what the setting and exclusion snapshots cannot: a calibration
    standard's cycle window, or an edit to a blank a calibration depends on.
    The domain check is pure and reads current inputs only.
    """
    result = getattr(state, "result", None)
    quality = getattr(result, "quality_metrics", None) or {}
    if not quality.get("pb_standard_calibration") or state.element_config is None:
        return False
    from domain.calibration_dependencies import FRESHNESS_STALE

    try:
        freshness = current_pb_calibration_freshness(state)
    except Exception:  # pragma: no cover - a failed check is never reported as current
        return True
    return freshness["status"] == FRESHNESS_STALE


def current_pb_calibration_freshness(state: AppState) -> Optional[Dict[str, Any]]:
    """Authoritative freshness context for current-session consumers."""
    result = getattr(state, "result", None)
    quality = getattr(result, "quality_metrics", None) or {}
    if not quality.get("pb_standard_calibration") or state.element_config is None:
        return None
    from domain.calibration_dependencies import calibration_freshness
    return calibration_freshness(
        quality, state.samples, element=state.element_config,
        settings=state.processing_config, cycle_ranges=get_cycle_ranges(state),
    )


def _pluralize(count: int, singular: str, plural: Optional[str] = None) -> str:
    """Return a simple count label such as '2 standards'."""
    label = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {label}"


def _normalize_sample_type(sample_type: str) -> str:
    """Collapse sample-type aliases to SMP/STD/BLK."""
    stype = (sample_type or "").upper()
    if stype in ("STD", "STANDARD"):
        return "STD"
    if stype in ("BLK", "BLANK"):
        return "BLK"
    return "SMP"


def _format_sample_type_counts(
    counts: Dict[str, int],
    *,
    empty_label: str,
) -> str:
    """Format active sample counts for banners."""
    parts = []
    if counts.get("SMP", 0) > 0:
        parts.append(_pluralize(counts["SMP"], "sample"))
    if counts.get("STD", 0) > 0:
        parts.append(_pluralize(counts["STD"], "standard"))
    if counts.get("BLK", 0) > 0:
        parts.append(_pluralize(counts["BLK"], "blank"))
    if parts:
        return ", ".join(parts)
    return empty_label


def _count_current_sample_types(state: AppState) -> Tuple[Dict[str, int], int]:
    """Count active/excluded sample types from the current session view."""
    counts = {"SMP": 0, "STD": 0, "BLK": 0}
    excluded = 0
    sample_source = state.samples or getattr(state.result, "samples", None) or []

    for sample in sample_source:
        if sample.metadata.get("excluded", False):
            excluded += 1
            continue
        counts[_normalize_sample_type(sample.sample_type)] += 1

    return counts, excluded


def _count_processed_sample_types() -> Tuple[Dict[str, int], int]:
    """Count active/excluded sample types from the last processed snapshot."""
    counts = {"SMP": 0, "STD": 0, "BLK": 0}
    processed_types = st.session_state.get(SESSION_KEY_PROCESSED_TYPES, {}) or {}
    processed_exclusions = st.session_state.get(SESSION_KEY_PROCESSED_EXCLUSIONS, ())
    if processed_exclusions is None:
        processed_exclusions = ()

    for sample_name, sample_type in processed_types.items():
        if sample_name in processed_exclusions:
            continue
        counts[_normalize_sample_type(sample_type)] += 1

    return counts, len(processed_exclusions)


def get_processing_status_message(
    state: Optional[AppState] = None,
) -> Optional[Tuple[str, str]]:
    """Return the current processing status banner kind and message."""
    state = state or get_state()
    if not state.has_data and not state.has_result:
        return None

    current_counts, current_excluded = _count_current_sample_types(state)
    current_summary = _format_sample_type_counts(
        current_counts,
        empty_label="0 active measurements",
    )

    if not state.has_result:
        msg = f"Ready to execute data reduction: {current_summary} loaded."
        if current_excluded:
            msg += f" {_pluralize(current_excluded, 'sample')} currently excluded."
        return "info", msg

    if settings_changed_since_processing(state):
        processed_counts, processed_excluded = _count_processed_sample_types()
        processed_summary = _format_sample_type_counts(
            processed_counts,
            empty_label="0 active measurements",
        )
        msg = (
            f"Data reduction results are stale. "
            f"Last processed snapshot: {processed_summary} processed."
        )
        if processed_excluded > 0:
            msg += f" {_pluralize(processed_excluded, 'sample')} excluded."
        msg += f" Current setup: {current_summary} active."
        if current_excluded:
            msg += f" {_pluralize(current_excluded, 'sample')} currently excluded."
        msg += " Click Execute Data Reduction to update results."
        return "warning", msg

    msg = f"Data reduction complete: {current_summary} processed."
    if current_excluded:
        msg += f" {_pluralize(current_excluded, 'sample')} excluded."
    return "success", msg

"""Instrumental Drift tab for TraceISO."""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional

import numpy as np
import streamlit as st

from config.constants import DEFAULT_OUTLIER_THRESHOLD_SD
from domain.corrections.drift import (
    build_drift_x_value_map,
    detect_outliers_standard_deviation,
    get_time_axis_debug_info,
)
from domain.filters.outlier import get_filtered_values
from domain.models import Sample
from ui.config_plotly import (
    DRIFT_PREVIEW_COMPACT_HEIGHT,
    DRIFT_PREVIEW_HEIGHT,
    PLOTLY_ANNOTATION_FONT_SIZE,
    PLOTLY_AXIS_TITLE_FONT_SIZE,
    PLOTLY_BASE_FONT_SIZE,
)
from ui.components.workspace_ui import (
    render_status_chip,
    render_subview_nav,
    workspace_panel,
)
from ui.navigation import resolve_subview_selection
from ui.state import get_state
from ui.theme import get_theme
from ui.utils import (
    finalize_processing_run,
    format_isotope_label,
    get_cycle_ranges,
    get_sample_state_key,
)

# Session-state key for isolated drift preview (never touches state.result)
_DRIFT_PREVIEW_KEY = "_drift_preview"
_log = logging.getLogger(__name__)


def _drift_preview_source_token(state, ratio_name: str) -> tuple:
    """Bind a preview to the committed result and all scientific UI overlays."""
    from ui.utils import (
        manual_exclusions_snapshot,
        observation_state_snapshot,
        processing_config_snapshot,
    )

    return (
        "traceiso.drift_preview_source.v1",
        id(getattr(state, "result", None)),
        str(getattr(state, "file_hash", "") or ""),
        str(ratio_name),
        processing_config_snapshot(getattr(state, "processing_config", None)),
        observation_state_snapshot(getattr(state, "samples", None)),
        manual_exclusions_snapshot(getattr(state, "samples", None)),
        tuple(sorted(get_cycle_ranges(state).items())),
    )


def _valid_drift_preview(state, ratio_name: str, preview: object) -> bool:
    return bool(
        isinstance(preview, dict)
        and preview.get("source_token") == _drift_preview_source_token(state, ratio_name)
    )

# Drift figures contain relatively sparse data over a wide plotting area. Use
# larger visual elements so both the on-screen view and high-resolution export
# remain readable when scaled to a page or slide.
_DRIFT_TICK_FONT_SIZE = PLOTLY_BASE_FONT_SIZE + 3
_DRIFT_AXIS_TITLE_FONT_SIZE = PLOTLY_AXIS_TITLE_FONT_SIZE + 4
_DRIFT_LEGEND_FONT_SIZE = PLOTLY_BASE_FONT_SIZE + 4
_DRIFT_ANNOTATION_FONT_SIZE = PLOTLY_ANNOTATION_FONT_SIZE + 3
_DRIFT_STANDARD_MARKER_SIZE = 13
_DRIFT_SAMPLE_MARKER_SIZE = 11
_DRIFT_EXPORT_WIDTH = 1200
_DRIFT_EXPORT_HEIGHT = 700

# Presentation-only R² cut-offs for the fit-quality chip and the advisory
# warning. These are display heuristics chosen to make an obviously bad fit
# visible at a glance — they are NOT an acceptance criterion for the drift
# correction and carry no metrological meaning. Nothing downstream branches on
# them; the drift correction itself is applied regardless of R².
_DRIFT_R2_EXCELLENT = 0.99
_DRIFT_R2_ACCEPTABLE = 0.90
_DRIFT_SETTINGS_FIELDS = (
    "method",
    "degree",
    "x_axis",
    "apply_outlier_filter",
    "outlier_threshold",
    "outlier_method",
    "norm_mode",
    "norm_standard",
)


def _clear_drift_preview_callback() -> None:
    """Invalidate the stored preview when any fitting parameter changes."""
    st.session_state.pop(_DRIFT_PREVIEW_KEY, None)


def _get_active_drift_samples(samples: List[Sample]) -> List[Sample]:
    """Return non-excluded samples for drift preview and plotting."""
    return [s for s in samples if not s.metadata.get("excluded", False)]


def _eval_fit(fit_info: Dict, x) -> np.ndarray:
    """Evaluate a stored drift fit in its declared coordinate system."""
    x_values = np.asarray(x, dtype=float)
    if fit_info.get("fit_format") == "centered_scaled_v1":
        center = float(fit_info.get("x_center", 0.0))
        scale = float(fit_info.get("x_scale", 1.0))
        if not np.isfinite(scale) or scale == 0.0:
            raise ValueError("centered/scaled drift fit has an invalid x_scale")
        x_values = (x_values - center) / scale
    return np.polyval(fit_info.get("coeffs", []), x_values)


def _compute_fit_quality_metrics(
    fit_info: Optional[Dict],
) -> Dict[str, Optional[float]]:
    """Derive fit-quality metrics from stored preview metadata."""
    fit_info = fit_info or {}
    method = fit_info.get("method")
    coeffs = fit_info.get("coeffs")
    x_data = np.asarray(fit_info.get("_fit_quality_x_data", fit_info.get("x_data", [])), dtype=float)
    y_data = np.asarray(fit_info.get("_fit_quality_y_data", fit_info.get("y_data", [])), dtype=float)

    metrics: Dict[str, Optional[float]] = {
        "r_squared": None,
        "residual_sd": None,
        "n_points": float(len(x_data)),
        "n_total": float(len(fit_info.get("x_all", [])))
        if fit_info.get("x_all") is not None
        else None,
    }

    if (
        len(x_data) == 0
        or len(y_data) == 0
        or method not in ("linear", "polynomial")
        or not coeffs
    ):
        return metrics

    y_pred = _eval_fit(fit_info, x_data)
    residuals = y_data - y_pred
    metrics["residual_sd"] = (
        float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    )

    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    if ss_tot > 0:
        ss_res = np.sum((y_data - y_pred) ** 2)
        metrics["r_squared"] = float(1 - (ss_res / ss_tot))

    return metrics


def _prepare_fit_info_for_display(
    fit_info: Optional[Dict],
    *,
    permil_mode: bool,
    reference_mean: Optional[float],
) -> tuple[Optional[Dict], Dict[str, list]]:
    """Return display-safe fit metadata plus untouched absolute y-series."""
    if not fit_info:
        return fit_info, {}

    displayed = dict(fit_info)
    displayed["_fit_quality_x_data"] = list(fit_info.get("x_data", []))
    displayed["_fit_quality_y_data"] = list(fit_info.get("y_data", []))
    absolute: Dict[str, list] = {}
    for key in ("y_fit", "y_all", "y_data"):
        if key not in fit_info:
            continue
        values = list(fit_info[key])
        absolute[key] = values
        displayed[key] = list(values)

    if permil_mode and reference_mean is not None and reference_mean != 0.0:
        for key, values in absolute.items():
            displayed[key] = [
                (float(value) / reference_mean - 1.0) * 1000.0 for value in values
            ]

    return displayed, absolute


def _compute_display_residuals(
    *,
    x_data: np.ndarray,
    y_data_absolute: np.ndarray,
    permil_mode: bool,
    reference_mean: Optional[float],
    fit_info: Optional[Dict] = None,
    coeffs=None,
) -> np.ndarray:
    """Compute residuals in absolute units, then convert for display."""
    resolved_fit = fit_info or {"coeffs": coeffs or []}
    residuals = np.asarray(y_data_absolute, dtype=float) - _eval_fit(
        resolved_fit, x_data
    )
    if permil_mode and reference_mean is not None and reference_mean != 0.0:
        residuals = residuals / reference_mean * 1000.0
    return residuals


def _build_fit_quality_warnings(
    fit_info: Optional[Dict],
    *,
    degree: int,
) -> List[str]:
    """Return advisory warnings for poor or fragile drift fits."""
    metrics = _compute_fit_quality_metrics(fit_info)
    warnings: List[str] = []
    n_points = int(metrics["n_points"] or 0)
    r_squared = metrics["r_squared"]

    if n_points and degree >= (n_points - 1):
        warnings.append(
            f"Drift fit is highly parameterized for preview (degree = {degree}, standards used = {n_points}). "
            "Consider reducing the polynomial degree."
        )
    if r_squared is not None and r_squared < _DRIFT_R2_ACCEPTABLE:
        warnings.append(
            f"Drift fit quality is low (R² = {r_squared:.3f}). "
            "Consider reducing the degree or reviewing the standards used."
        )

    return warnings


def _fit_quality_status(fit_info: Optional[Dict]) -> tuple[str, str]:
    """Return the presentation tone and label for a drift fit."""
    metrics = _compute_fit_quality_metrics(fit_info)
    r_squared = metrics["r_squared"]
    if r_squared is None:
        return "neutral", "Fit quality unavailable"
    if r_squared >= _DRIFT_R2_EXCELLENT:
        return "positive", f"Excellent fit · R² = {r_squared:.4f}"
    if r_squared >= _DRIFT_R2_ACCEPTABLE:
        return "warning", f"Acceptable fit · R² = {r_squared:.4f}"
    return "critical", f"Poor fit · R² = {r_squared:.4f}"


def _corrected_standard_mean_stats(
    fit_info: Optional[Dict],
    y_std_after: List[float],
) -> tuple[Optional[float], Optional[float]]:
    """Return mean and SD for corrected standards actually shown in preview."""
    values = np.asarray(y_std_after, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) > 0:
        mean_val = float(np.nanmean(values))
        std_val = float(np.nanstd(values, ddof=1)) if len(values) >= 2 else None
        return mean_val, std_val

    if fit_info and "std_mean" in fit_info:
        return float(fit_info["std_mean"]), None
    return None, None


def _build_drift_settings(
    *,
    method: str,
    degree: int,
    x_axis: str,
    apply_outlier_filter: bool,
    outlier_threshold: float,
    outlier_method: str = "mad",
    norm_mode: str,
    norm_standard: int,
) -> Dict[str, object]:
    """Return a preview-local drift-settings payload."""
    return {
        "method": method.lower(),
        "degree": int(degree),
        "x_axis": x_axis,
        "apply_outlier_filter": bool(apply_outlier_filter),
        "outlier_threshold": float(outlier_threshold),
        "outlier_method": str(outlier_method),
        "norm_mode": norm_mode,
        "norm_standard": int(norm_standard),
    }


def _commit_drift_settings(config, settings: Dict[str, object]) -> None:
    """Copy previewed drift settings into the persisted processing config."""
    for field_name in _DRIFT_SETTINGS_FIELDS:
        if field_name in settings:
            setattr(config, field_name, settings[field_name])


def _sample_identity(sample: Sample) -> tuple[str, int]:
    """Return a stable identity for a sample in preview maps."""
    return sample.name, int(sample.run_number)


def _get_reference_standards(
    samples: List[Sample],
    ratio_name: str,
    *,
    cycle_ranges: Optional[Dict[str, tuple[int, int]]] = None,
    apply_outlier_filter: bool = True,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
    outlier_method: str = "mad",
) -> List[Sample]:
    """Return standards that are eligible to anchor single-standard normalization."""
    from domain.ratio_selection import get_best_pre_drift_ratio_data

    standards = [s for s in _get_active_drift_samples(samples) if s.is_standard]
    if not standards:
        return []

    valid_standards: List[Sample] = []
    standard_means: List[float] = []
    for std in standards:
        ratio_data = get_best_pre_drift_ratio_data(std, ratio_name)
        if ratio_data is None:
            continue
        valid_values = _get_active_ratio_values(
            ratio_data,
            std.name,
            sample_key=get_sample_state_key(std),
            cycle_ranges=cycle_ranges,
        )
        if len(valid_values) == 0:
            continue
        standard_means.append(float(np.nanmean(valid_values)))
        valid_standards.append(std)

    if len(valid_standards) < 3:
        return valid_standards

    if apply_outlier_filter:
        outlier_mask = detect_outliers_standard_deviation(
            np.asarray(standard_means, dtype=np.float64),
            threshold=outlier_threshold,
            method=outlier_method,
        )
        valid_standards = [
            std for std, keep in zip(valid_standards, outlier_mask) if keep
        ]

    return valid_standards


def render_drift_tab() -> None:
    """Render the Instrumental Drift tab."""
    state = get_state()

    if not state.has_result:
        st.info(
            "Process the data first (in Session Configuration) to enable drift correction."
        )
        return

    result = state.result
    samples = result.samples
    from ui.components.sr_calibration_panel import render_sr_calibration_notice
    render_sr_calibration_notice(samples)

    if not samples:
        st.warning("No samples in processing result.")
        return

    drift_enabled = state.processing_config.drift.enabled
    if not drift_enabled:
        st.info(
            "Drift correction is **not applied** to the current results. "
            "You can still fit and review the drift model below; when you are "
            "satisfied, use **Apply drift correction and reprocess** to enable it. "
            "This is a review-only view — nothing here changes the reduced data "
            "until you commit."
        )

    # Ratio selector - use selected ratios from Session Configuration
    from ui.components.custom_ratio import get_selected_ratios

    selected_ratios = get_selected_ratios(state.samples)
    ratio_list = sorted(selected_ratios)
    if not ratio_list:
        st.warning("No ratio data available.")
        return

    # Default to primary ratio
    default_idx = 0
    if state.element_config and state.element_config.primary_ratio:
        primary = state.element_config.primary_ratio
        if primary in ratio_list:
            default_idx = ratio_list.index(primary)

    selected_ratio = st.selectbox(
        "Ratio",
        options=ratio_list,
        index=default_idx,
        format_func=format_isotope_label,
        key="drift_ratio_select",
        help="Ratio used to fit the instrumental drift model from the standard sequence and to preview the corrected run.",
    )

    preview = st.session_state.get(_DRIFT_PREVIEW_KEY)
    if preview and not _valid_drift_preview(state, selected_ratio, preview):
        st.session_state.pop(_DRIFT_PREVIEW_KEY, None)
        preview = None
        st.info("The previous drift preview expired because its source data or settings changed. Run a new preview.")

    # Side-by-side layout: settings beside preview at wide screens, stacks naturally at narrow screens.
    col_settings, col_preview = st.columns([1, 2])
    with col_settings:
        with workspace_panel(
            "Drift correction settings",
            eyebrow="1 · Fit",
            key="drift_settings_panel",
        ):
            _render_settings(state, selected_ratio)
    with col_preview:
        with workspace_panel(
            "Pre / post correction",
            eyebrow="2 · Preview",
            key="drift_preview_panel",
        ):
            plot_data = _render_preview_plot(state, selected_ratio, preview)

    # Details are rendered below the side-by-side layout in a single full-width column.
    if plot_data:
        with workspace_panel(
            "Preview details",
            eyebrow="3 · Review",
            key="drift_details_panel",
        ):
            _render_preview_details(state, selected_ratio, preview, plot_data)


def _render_settings(state, ratio_name: str) -> None:
    """Render drift correction settings."""
    config = state.processing_config.drift
    time_debug = get_time_axis_debug_info(
        state.result.samples if state.has_result else []
    )
    has_time_axis = bool(time_debug.get("available", False))
    x_axis_options = ["Run Number", "Index"]
    x_axis_values = ["run_number", "index"]
    if has_time_axis:
        x_axis_options.append("Time (min)")
        x_axis_values.append("time_minutes")
    x_axis_value = config.x_axis if config.x_axis in x_axis_values else "run_number"
    method_value = (
        config.method if config.method in {"linear", "polynomial"} else "polynomial"
    )

    method = st.selectbox(
        "Method",
        options=["Linear", "Polynomial"],
        index=["linear", "polynomial"].index(method_value),
        key="drift_method",
        on_change=_clear_drift_preview_callback,
        help=(
            "Empirical drift model fitted to standards. "
            "Linear suits near-constant drift, and polynomial captures "
            "smooth session-scale curvature."
        ),
    )

    if method.lower() == "polynomial":
        # Count active standards to cap admissible degree
        _active_stds = [
            s
            for s in (state.result.samples if state.has_result else [])
            if s.is_standard and not s.metadata.get("excluded", False)
        ]
        _n_stds = len(_active_stds)
        # degree must be < n_standards; cap at 5 for UI sanity
        _max_degree = min(5, max(1, _n_stds - 1)) if _n_stds > 1 else 1
        degree = st.number_input(
            "Polynomial Degree",
            min_value=1,
            max_value=_max_degree,
            value=min(config.degree, _max_degree),
            key="drift_degree",
            on_change=_clear_drift_preview_callback,
            help=(
                f"Polynomial order of the drift fit (max {_max_degree} for "
                f"{_n_stds} active standard{'s' if _n_stds != 1 else ''}). "
                f"Increase only when the standards support genuine curvature."
            ),
        )
    else:
        degree = 1
        st.text_input("Polynomial Degree", value="—", disabled=True)

    x_axis = st.selectbox(
        "X-axis",
        options=x_axis_options,
        index=x_axis_values.index(x_axis_value),
        key="drift_x_axis",
        on_change=_clear_drift_preview_callback,
        help=(
            "Use run number, sequence index, or elapsed minutes from the "
            "first sample when timestamps are available in file metadata."
        ),
    )

    if has_time_axis:
        st.caption(
            f"Time-axis debug: available. First sample time = {time_debug.get('first_timestamp', 'unknown')}."
        )
    else:
        missing_samples = time_debug.get("missing_samples", [])
        missing_preview = ", ".join(missing_samples[:3])
        if len(missing_samples) > 3:
            missing_preview += ", ..."
        suffix = f" Missing: {missing_preview}." if missing_preview else ""
        st.caption(
            f"Time-axis debug: unavailable. {time_debug.get('reason', 'Unknown reason.')}{suffix}"
        )

    # Outlier filter toggle (classic standard-deviation rejection on standard means)
    apply_outlier = st.checkbox(
        "Apply standard deviation outlier rejection to standards before fitting",
        value=config.apply_outlier_filter,
        key="drift_outlier_filter",
        on_change=_clear_drift_preview_callback,
        help="Exclude anomalous standard measurements before fitting the empirical drift trend so single bad standards do not anchor the correction.",
    )

    outlier_method = "sd"
    if apply_outlier:
        outlier_threshold = st.slider(
            "Outlier rejection threshold (standard deviations)",
            min_value=1.0,
            max_value=3.0,
            value=config.outlier_threshold,
            step=0.1,
            key="drift_outlier_threshold",
            on_change=_clear_drift_preview_callback,
            help="Threshold (in sigma) applied to the standard means before the drift fit is calculated.",
        )
    else:
        outlier_threshold = DEFAULT_OUTLIER_THRESHOLD_SD

    # Normalization mode
    st.markdown("**Normalization Reference**")
    norm_options = ["Average of all standards", "Selected standard"]
    norm_default = (
        norm_options[0] if config.norm_mode == "average" else norm_options[1]
    )
    norm_mode_label = render_subview_nav(
        "Normalize to",
        norm_options,
        key="drift_norm_mode",
        default=norm_default,
        current=resolve_subview_selection(
            key="drift_norm_mode", options=norm_options, default=norm_default,
        ),
    )
    norm_mode = "average" if norm_mode_label == norm_options[0] else "single"

    norm_standard = 0
    if norm_mode == "single":
        reference_standards = _get_reference_standards(
            state.result.samples if state.has_result else [],
            ratio_name,
            cycle_ranges=get_cycle_ranges(state),
            apply_outlier_filter=apply_outlier,
            outlier_threshold=outlier_threshold,
            outlier_method=outlier_method,
        )
        if reference_standards:
            std_labels = [
                f"{i + 1}. {std.name} (Run {std.run_number})"
                for i, std in enumerate(reference_standards)
            ]
            selected_name = st.session_state.get("drift_norm_standard_name")
            default_std_idx = 0
            if selected_name:
                for idx, std in enumerate(reference_standards):
                    if std.name == selected_name:
                        default_std_idx = idx
                        break
            else:
                default_std_idx = min(config.norm_standard, len(std_labels) - 1)

            selected_std_label = st.selectbox(
                "Reference standard",
                options=std_labels,
                index=default_std_idx,
                key="drift_norm_standard",
                help="Standard used as the normalization anchor when single-standard mode is selected.",
            )
            norm_standard = std_labels.index(selected_std_label)
            st.session_state["drift_norm_standard_name"] = reference_standards[
                norm_standard
            ].name
        else:
            st.caption("_No standards found. Process data first._")

    drift_settings = _build_drift_settings(
        method=method,
        degree=degree,
        x_axis=x_axis_values[x_axis_options.index(x_axis)],
        apply_outlier_filter=apply_outlier,
        outlier_threshold=outlier_threshold,
        outlier_method=outlier_method,
        norm_mode=norm_mode,
        norm_standard=norm_standard,
    )

    # Preview / Clear buttons
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button(
            "Run Preview",
            type="secondary",
            width="stretch",
            key="preview_drift",
        ):
            with st.spinner("Calculating drift correction..."):
                _preview_drift(state, ratio_name, drift_settings)
                st.rerun()
    with btn_col2:
        has_preview = _DRIFT_PREVIEW_KEY in st.session_state
        if st.button(
            "Clear Preview",
            type="secondary",
            width="stretch",
            key="clear_drift_preview",
            disabled=not has_preview,
        ):
            st.session_state.pop(_DRIFT_PREVIEW_KEY, None)
            st.rerun()

    if has_preview:
        render_status_chip(
            "Preview ready · review, then commit",
            tone="warning",
        )
        commit_label = (
            "Commit Drift Correction"
            if config.enabled
            else "Apply drift correction and reprocess"
        )
        if st.button(
            commit_label,
            type="primary",
            width="stretch",
            key="commit_drift_top",
            help=(
                "Re-runs the canonical processing pipeline with this fit. "
                "No round trip through Session Configuration."
            ),
        ):
            _apply_drift(state, ratio_name)


def _preview_drift(state, ratio_name: str, drift_settings: Dict[str, object]) -> None:
    """Preview drift correction in an isolated copy — state.result is UNTOUCHED."""
    from domain.corrections.drift import apply_drift_correction

    cycle_ranges = get_cycle_ranges(state)

    # Deep-copy samples so the domain function's in-place mutations
    # never reach the production result.
    preview_samples = [
        s.copy() for s in _get_active_drift_samples(state.result.samples)
    ]

    result = apply_drift_correction(
        preview_samples,
        ratio_name,
        method=str(drift_settings["method"]),
        degree=int(drift_settings["degree"]),
        x_axis=str(drift_settings["x_axis"]),
        apply_outlier_filter=bool(drift_settings["apply_outlier_filter"]),
        outlier_threshold=float(drift_settings["outlier_threshold"]),
        outlier_method=str(drift_settings.get("outlier_method", "mad")),
        norm_mode=str(drift_settings["norm_mode"]),
        norm_standard=int(drift_settings["norm_standard"]),
        cycle_ranges=cycle_ranges,
    )
    result.fit_info.update(_compute_fit_quality_metrics(result.fit_info))

    # Store x_axis in fit_info so downstream code (summary table) knows
    # which basis was used (fixes H-4).
    result.fit_info["x_axis"] = drift_settings["x_axis"]
    fit_quality_warnings = _build_fit_quality_warnings(
        result.fit_info,
        degree=int(drift_settings["degree"]),
    )

    # Store in isolated session-state key — NOT in state.result
    st.session_state[_DRIFT_PREVIEW_KEY] = {
        "samples": result.samples,
        "fit_info": result.fit_info,
        "warnings": result.warnings,
        "fit_quality_warnings": fit_quality_warnings,
        "ratio_name": ratio_name,
        "settings": dict(drift_settings),
        "source_token": _drift_preview_source_token(state, ratio_name),
    }


def _apply_drift(state, ratio_name: str) -> None:
    """Commit the previewed drift correction by updating config and re-processing."""
    preview = st.session_state.get(_DRIFT_PREVIEW_KEY)
    if preview is None:
        st.error("No drift preview to apply. Run Preview first.")
        return
    if not _valid_drift_preview(state, ratio_name, preview):
        st.session_state.pop(_DRIFT_PREVIEW_KEY, None)
        st.error("This drift preview is no longer current. Run Preview again before committing.")
        return

    candidate_config = copy.deepcopy(state.processing_config)
    config = candidate_config.drift
    settings = preview.get("settings") or {}
    _commit_drift_settings(config, settings)
    config.enabled = True
    config.ratio_name = ratio_name

    if state.element_config is None or state.samples is None:
        st.error("Missing element config or samples.")
        return

    try:
        from domain.processing_service import process_samples
        from ui.diagnostics import timed

        candidate_samples = copy.deepcopy(state.samples)
        with st.spinner("Re-running processing pipeline with drift correction..."):
            result = timed("Data reduction")(process_samples)(
                candidate_samples,
                state.element_config,
                candidate_config,
                uncertainty_config=state.uncertainty_config,
                profile_defaults=state.uncertainty_profile_defaults,
                cycle_ranges=get_cycle_ranges(state),
            )
        state.processing_config = candidate_config
        state.samples = candidate_samples
        finalize_processing_run(state, result)
        st.success("Drift correction committed via the canonical pipeline.")
    except Exception as exc:
        _log.exception("Drift commit processing failed")
        st.error(f"Drift correction could not be committed: {exc}. The previous result and preview were retained; correct the issue and retry.")
        return

    st.session_state.pop(_DRIFT_PREVIEW_KEY, None)
    st.rerun()


def _render_preview_plot(
    state, ratio_name: str, preview: Optional[dict] = None
) -> Optional[dict]:
    """Render combined before/after preview plot. Returns plot data for details rendering."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from domain.ratio_selection import get_best_pre_drift_ratio_data
    from ui.utils import format_name

    config = state.processing_config.drift
    theme = get_theme()
    palette = theme.palette

    with st.popover("Display options", width="stretch"):
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        with col_ctrl1:
            st.markdown("**Traces to show**")
            show_before = st.checkbox("Before", value=True, key="drift_show_before")
            show_after = st.checkbox("After", value=True, key="drift_show_after")
            show_samples = st.checkbox("Samples", value=False, key="drift_show_samples")
            show_fit = st.checkbox("Fit", value=True, key="drift_show_fit")
        with col_ctrl2:
            st.markdown("**Reference bands and lines**")
            show_ci_band = st.checkbox(
                "±2 estimated SE of fitted mean",
                value=True,
                key="drift_show_ci_band",
                help="Show ±2 estimated standard errors of the fitted mean from the stored OLS coefficient covariance. This is not a formal confidence or prediction interval.",
            )
            show_threshold = st.checkbox(
                "Threshold",
                value=False,
                key="drift_show_threshold",
                help="Overlay the standard-deviation rejection band (mean \u00b1 threshold\u00d7SD) on the before-correction standards, so excluded standards visibly fall outside it.",
            )
            show_after_reference = st.checkbox(
                "After reference",
                value=False,
                key="drift_show_after_reference",
                help="Show the corrected-standard mean as a horizontal reference line.",
            )
            show_mean_annotation = st.checkbox(
                "Mean \u00b1 2SD",
                value=False,
                key="drift_show_mean_annotation",
            )
        with col_ctrl3:
            st.markdown("**Display options**")
            show_residuals = st.checkbox(
                "Residuals",
                value=True,
                key="drift_show_residuals",
            )
            show_grid = st.checkbox(
                "Grid",
                value=False,
                key="drift_show_grid",
                help="Show grid lines on the drift and residual plots.",
            )
            permil_mode = st.checkbox(
                "\u2030 Deviation",
                value=False,
                key="drift_y_permil_mode",
            )
            show_error_bars = st.checkbox(
                "Error Bars",
                value=False,
                key="drift_show_error_bars",
                help="Show error bars (2SE or 2SD, selectable below) on each standard's before/after mean.",
            )

            error_bar_stat = "2SE"
            if show_error_bars:
                error_bar_stat = render_subview_nav(
                    "Error bar statistic",
                    ["2SE", "2SD"],
                    key="drift_error_bar_stat",
                    default="2SE",
                    current=resolve_subview_selection(
                        key="drift_error_bar_stat",
                        options=["2SE", "2SD"],
                        default="2SE",
                    ),
                )

    result = state.result
    cycle_ranges = get_cycle_ranges(state)
    if preview and preview.get("ratio_name") != ratio_name:
        st.session_state.pop(_DRIFT_PREVIEW_KEY, None)
        preview = None
    preview_settings = preview.get("settings") if preview else None
    x_axis = (
        str(preview_settings.get("x_axis", config.x_axis))
        if isinstance(preview_settings, dict)
        else config.x_axis
    )

    active_result_samples = _get_active_drift_samples(result.samples)
    standards = [s for s in active_result_samples if s.is_standard]
    # Only include SMP samples (exclude blanks and excluded samples)
    samples = [s for s in active_result_samples if s.is_sample]

    if len(standards) < 3:
        st.warning("Need ≥3 standards to show drift preview.")
        return

    if preview:
        for warn in preview.get("warnings", []):
            st.warning(warn)
        fit_quality_warnings = preview.get("fit_quality_warnings")
        if fit_quality_warnings is None:
            degree = (
                int(preview_settings.get("degree", config.degree))
                if isinstance(preview_settings, dict)
                else config.degree
            )
            fit_quality_warnings = _build_fit_quality_warnings(
                preview.get("fit_info"),
                degree=degree,
            )
        for warn in fit_quality_warnings:
            st.warning(warn)
    # Determine source of "After" data and Fit Info
    # Priority: 1. Active Preview, 2. Current Committed Result (if drift enabled)
    preview_samples = None
    fit_info = None

    if preview:
        preview_samples = preview["samples"]
        fit_info = preview["fit_info"]
    elif config.enabled:
        # Fallback: Show the currently applied drift correction
        current_fit = result.quality_metrics.get("drift_fit_info")

        if current_fit:
            # Verify if this fit info belongs to the current ratio
            fit_ratio = current_fit.get("ratio_name")

            is_match = False
            if fit_ratio:
                is_match = fit_ratio == ratio_name
            elif config.ratio_name == ratio_name:
                # Legacy fallback if fit_info lacks ratio_name
                is_match = True

            if is_match:
                fit_info = current_fit
                preview_samples = active_result_samples

    fit_info, absolute_fit_series = _prepare_fit_info_for_display(
        fit_info,
        permil_mode=False,
        reference_mean=None,
    )
    fit_tone, fit_label = _fit_quality_status(fit_info)
    render_status_chip(fit_label, tone=fit_tone)

    def _error_bar_halfwidth(vals: np.ndarray) -> float:
        """2SD or 2SE half-width for error bars, per the `error_bar_stat` toggle."""
        n = int(np.sum(~np.isnan(vals)))
        if n < 2:
            return 0.0
        sd = float(np.nanstd(vals, ddof=1))
        if error_bar_stat == "2SE":
            return 2.0 * sd / np.sqrt(n)
        return 2.0 * sd

    before_x_map = _build_sample_x_map(active_result_samples, x_axis)

    x_std_before = []
    y_std_before = []
    err_std_before = []

    for std in standards:
        x_val = before_x_map.get(id(std), float(std.run_number))

        ratio_data = get_best_pre_drift_ratio_data(std, ratio_name)

        before_vals = _get_active_ratio_values(
            ratio_data,
            std.name,
            sample_key=get_sample_state_key(std),
            cycle_ranges=cycle_ranges,
        )
        if len(before_vals) > 0:
            x_std_before.append(x_val)
            y_std_before.append(float(np.nanmean(before_vals)))
            err_std_before.append(_error_bar_halfwidth(before_vals))

    x_std_after = []
    y_std_after = []
    err_std_after = []
    x_smp_before = []
    y_smp_before = []
    text_smp_before = []
    x_smp_after = []
    y_smp_after = []
    text_smp_after = []

    for smp in samples:
        x_val = before_x_map.get(id(smp), float(smp.run_number))
        ratio_data = get_best_pre_drift_ratio_data(smp, ratio_name)
        before_vals = _get_active_ratio_values(
            ratio_data,
            smp.name,
            sample_key=get_sample_state_key(smp),
            cycle_ranges=cycle_ranges,
        )
        if len(before_vals) > 0:
            x_smp_before.append(x_val)
            y_smp_before.append(float(np.nanmean(before_vals)))
            text_smp_before.append(smp.name)

    if fit_info and preview_samples:
        after_x_map = _build_sample_x_map(preview_samples, x_axis)
        preview_standards = [s for s in preview_samples if s.is_standard]
        preview_unknowns = [s for s in preview_samples if s.is_sample]
        for std in preview_standards:
            x_val = after_x_map.get(id(std), float(std.run_number))

            if ratio_name in std.drift_corrected_ratios:
                corr_data = std.drift_corrected_ratios[ratio_name]
                after_vals = _get_active_ratio_values(
                    corr_data,
                    std.name,
                    sample_key=get_sample_state_key(std),
                    cycle_ranges=cycle_ranges,
                )
                if len(after_vals) > 0:
                    x_std_after.append(x_val)
                    y_std_after.append(float(np.nanmean(after_vals)))
                    err_std_after.append(_error_bar_halfwidth(after_vals))
        for smp in preview_unknowns:
            x_val = after_x_map.get(id(smp), float(smp.run_number))
            if ratio_name in smp.drift_corrected_ratios:
                corr_data = smp.drift_corrected_ratios[ratio_name]
                after_vals = _get_active_ratio_values(
                    corr_data,
                    smp.name,
                    sample_key=get_sample_state_key(smp),
                    cycle_ranges=cycle_ranges,
                )
                if len(after_vals) > 0:
                    x_smp_after.append(x_val)
                    y_smp_after.append(float(np.nanmean(after_vals)))
                    text_smp_after.append(smp.name)

    # D1: permil-deviation transformation — convert absolute ratios to ‰ deviation
    # from the session mean so the drift profile is visible at readable scale.
    ref_mean: Optional[float] = None
    permil_label_suffix = ""
    if permil_mode:
        ref_mean_raw = fit_info.get("std_mean") if fit_info else None
        if not ref_mean_raw and y_std_before:
            ref_mean_raw = float(np.mean(y_std_before))
        if ref_mean_raw and ref_mean_raw != 0.0:
            ref_mean = float(ref_mean_raw)
            permil_label_suffix = " (‰ dev.)"

            def _pm(arr: list) -> list:
                return [(v / ref_mean - 1.0) * 1000.0 for v in arr]

            def _pm_err(arr: list) -> list:
                # Error bars are magnitudes, not deviations from ref_mean — scale only.
                return [v / ref_mean * 1000.0 for v in arr]

            y_std_before = _pm(y_std_before)
            y_smp_before = _pm(y_smp_before)
            y_std_after = _pm(y_std_after)
            y_smp_after = _pm(y_smp_after)
            err_std_before = _pm_err(err_std_before)
            err_std_after = _pm_err(err_std_after)
            fit_info, absolute_fit_series = _prepare_fit_info_for_display(
                fit_info,
                permil_mode=True,
                reference_mean=ref_mean,
            )

    # D2: detect extrapolation zones — samples whose x-positions fall outside the
    # standards' span and therefore receive polynomial extrapolation, not interpolation.
    x_extrap_low: list = []
    y_extrap_low: list = []
    x_extrap_high: list = []
    y_extrap_high: list = []
    extrap_warnings: list = []
    if fit_info and "coeffs" in fit_info and x_std_before:
        _x_std_min = float(min(x_std_before))
        _x_std_max = float(max(x_std_before))
        _all_smp_x = [v for v in (list(x_smp_before) + list(x_smp_after))]
        _out_low = [v for v in _all_smp_x if v < _x_std_min]
        _out_high = [v for v in _all_smp_x if v > _x_std_max]
        if _out_low:
            extrap_warnings.append(
                f"{len(set(_out_low))} sample(s) before the first standard — "
                "polynomial extrapolation applies."
            )
            _xe = np.linspace(min(_out_low), _x_std_min, 30)
            _ye = _eval_fit(fit_info, _xe)
            if permil_mode and ref_mean:
                _ye = (_ye / ref_mean - 1.0) * 1000.0
            x_extrap_low, y_extrap_low = _xe.tolist(), _ye.tolist()
        if _out_high:
            extrap_warnings.append(
                f"{len(set(_out_high))} sample(s) after the last standard — "
                "polynomial extrapolation applies."
            )
            _xe = np.linspace(_x_std_max, max(_out_high), 30)
            _ye = _eval_fit(fit_info, _xe)
            if permil_mode and ref_mean:
                _ye = (_ye / ref_mean - 1.0) * 1000.0
            x_extrap_high, y_extrap_high = _xe.tolist(), _ye.tolist()

    for _ew in extrap_warnings:
        st.warning(_ew)

    # D6/A054: use the covariance produced by the exact displayed fit.  The
    # renderer must not silently refit a different support selection.
    x_ci: list = []
    y_ci_upper: list = []
    y_ci_lower: list = []
    if (
        fit_info
        and "x_data" in fit_info
        and "y_data" in fit_info
        and "x_fit" in fit_info
        and fit_info.get("coeffs")
    ):
        from domain.corrections.drift import fitted_mean_band

        _xfit_ci, _hws, _band_reason = fitted_mean_band(fit_info)
        if _band_reason:
            if show_ci_band:
                st.info(f"Fitted-mean SE band unavailable: {_band_reason}.")
        else:
            _yfit_arr = np.array(fit_info["y_fit"], dtype=float)
            _hw_display = _hws / ref_mean * 1000.0 if permil_mode and ref_mean else _hws
            x_ci = _xfit_ci.tolist()
            y_ci_upper = (_yfit_arr + _hw_display).tolist()
            y_ci_lower = (_yfit_arr - _hw_display).tolist()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22] if show_residuals else [1.0, 0.001],
        vertical_spacing=0.06,
    )

    std_color = theme.sample_color("STD")
    std_symbol = theme.sample_symbol("STD")
    smp_color = theme.sample_color("SMP")
    smp_symbol = theme.sample_symbol("SMP")

    # Distinguish before (hollow) from after (filled) so the two overlapping
    # marker sets read as a transformation rather than a single cloud of points.
    std_symbol_before = f"{std_symbol}-open"
    smp_symbol_before = f"{smp_symbol}-open"
    std_symbol_after = std_symbol
    smp_symbol_after = smp_symbol

    # D6: use theme palette colours — no hardcoded hex values below this point.
    # Add before correction trace (standards colour for "before")
    if show_before:
        fig.add_trace(
            go.Scatter(
                x=x_std_before,
                y=y_std_before,
                mode="markers",
                marker=dict(
                    color=std_color,
                    size=_DRIFT_STANDARD_MARKER_SIZE,
                    symbol=std_symbol_before,
                    opacity=0.95,
                ),
                name="Standards before correction",
                error_y=dict(
                    type="data",
                    array=err_std_before,
                    visible=show_error_bars,
                    color=std_color,
                    thickness=1.2,
                    width=3,
                ),
                hovertemplate="<b>Standard</b><br>X: %{x}<br>Before: %{y:.6f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        if show_samples and x_smp_before:
            fig.add_trace(
                go.Scatter(
                    x=x_smp_before,
                    y=y_smp_before,
                    text=text_smp_before,
                    mode="markers",
                    marker=dict(
                        color=smp_color,
                        size=_DRIFT_SAMPLE_MARKER_SIZE,
                        symbol=smp_symbol_before,
                        opacity=1.0,
                    ),
                    name="Samples (Before)",
                    hovertemplate="<b>%{text}</b><br>X: %{x}<br>Before: %{y:.6f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # Outlier-rejection band on the before-correction standards (mean ± k×SD),
    # matching the standard-deviation rule used to exclude standards from the fit.
    # Drawn in the same display space as y_std_before, so it tracks ‰-deviation mode.
    if show_threshold and len(y_std_before) >= 2:
        threshold_k = (
            float(preview_settings.get("outlier_threshold", config.outlier_threshold))
            if isinstance(preview_settings, dict)
            else float(config.outlier_threshold)
        )
        _band_vals = np.asarray(y_std_before, dtype=float)
        _band_vals = _band_vals[np.isfinite(_band_vals)]
        if len(_band_vals) >= 2:
            _band_mean = float(np.mean(_band_vals))
            _band_sd = float(np.std(_band_vals, ddof=1))
            if _band_sd > 0:
                for _edge in (
                    _band_mean + threshold_k * _band_sd,
                    _band_mean - threshold_k * _band_sd,
                ):
                    fig.add_hline(
                        y=_edge,
                        line_dash="dashdot",
                        line_color=palette.layer_excluded,
                        line_width=0.9,
                        row=1,
                        col=1,
                    )

    # D3: the confidence band is independently visible from the fitted line.
    if show_ci_band and x_ci and y_ci_upper and y_ci_lower:
        fig.add_trace(
            go.Scatter(
                x=x_ci + x_ci[::-1],
                y=y_ci_upper + y_ci_lower[::-1],
                fill="toself",
                fillcolor=palette.guide_bounds_light,
                line=dict(width=0),
                name="±2 estimated SE of fitted mean",
                showlegend=True,
                hovertemplate="±2 estimated SE of fitted mean<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Add fitted curve + D2 extrapolation zones.
    if show_fit and fit_info and "x_fit" in fit_info and "y_fit" in fit_info:
        fig.add_trace(
            go.Scatter(
                x=fit_info["x_fit"],
                y=fit_info["y_fit"],
                mode="lines",
                line=dict(color=palette.guide_bounds, width=1.6, dash="dash"),
                name="Fitted drift trend",
                hovertemplate="Fit: %{y:.6f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        # D2: extrapolation zones — dashed extension beyond standards' x-span
        if x_extrap_low and y_extrap_low:
            fig.add_trace(
                go.Scatter(
                    x=x_extrap_low,
                    y=y_extrap_low,
                    mode="lines",
                    line=dict(color=palette.guide_bounds, width=1.2, dash="dot"),
                    name="Extrapolation (low)",
                    hovertemplate="Extrapolated: %{y:.6f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        if x_extrap_high and y_extrap_high:
            fig.add_trace(
                go.Scatter(
                    x=x_extrap_high,
                    y=y_extrap_high,
                    mode="lines",
                    line=dict(color=palette.guide_bounds, width=1.2, dash="dot"),
                    name="Extrapolation (high)",
                    hovertemplate="Extrapolated: %{y:.6f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Mark outliers excluded from the fit
        if "x_all" in fit_info and "y_all" in fit_info and "outlier_mask" in fit_info:
            outlier_mask = np.array(fit_info["outlier_mask"])
            if not np.all(outlier_mask):
                x_outliers = np.array(fit_info["x_all"])[~outlier_mask]
                y_outliers = np.array(fit_info["y_all"])[~outlier_mask]
                fig.add_trace(
                    go.Scatter(
                        x=x_outliers,
                        y=y_outliers,
                        mode="markers",
                        marker=dict(
                            color=palette.layer_excluded,
                            size=_DRIFT_STANDARD_MARKER_SIZE,
                            symbol="x",
                            line=dict(width=1.3),
                        ),
                        name="Excluded Standards",
                        hovertemplate="<b>Excluded standard</b><br>X: %{x}<br>Before: %{y:.6f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

    # Add after correction traces (success colour for "after")
    if show_after and x_std_after and y_std_after:
        fig.add_trace(
            go.Scatter(
                x=x_std_after,
                y=y_std_after,
                mode="markers",
                marker=dict(
                    color=std_color,
                    size=_DRIFT_STANDARD_MARKER_SIZE,
                    symbol=std_symbol_after,
                ),
                name="Standards after correction",
                error_y=dict(
                    type="data",
                    array=err_std_after,
                    visible=show_error_bars,
                    color=std_color,
                    thickness=1.2,
                    width=3,
                ),
                hovertemplate="<b>Standard</b><br>X: %{x}<br>After: %{y:.6f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        if show_samples and x_smp_after:
            fig.add_trace(
                go.Scatter(
                    x=x_smp_after,
                    y=y_smp_after,
                    text=text_smp_after,
                    mode="markers",
                    marker=dict(
                        color=smp_color,
                        size=_DRIFT_SAMPLE_MARKER_SIZE,
                        symbol=smp_symbol_after,
                        opacity=1.0,
                    ),
                    name="Samples (After)",
                    hovertemplate="<b>%{text}</b><br>X: %{x}<br>After: %{y:.6f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Add mean line for corrected standards
        mean_val, std_after = _corrected_standard_mean_stats(fit_info, y_std_after)
        if mean_val is not None:
            mean_label = (
                f"Mean \u00b1 2SD: {mean_val:.6f} \u00b1 {2 * std_after:.6f}"
                if std_after is not None
                else f"Mean \u00b1 2SD: {mean_val:.6f}"
            )
            if show_after_reference:
                fig.add_hline(
                    y=mean_val,
                    line_dash="dot",
                    line_color=std_color,
                    line_width=1.1,
                    row=1,
                    col=1,
                )
                # Plotly shapes are not represented in the trace legend on all
                # supported versions, so add a legend-only trace for this line.
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line=dict(color=std_color, width=1.1, dash="dot"),
                        name="After-correction reference mean",
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )
            if show_mean_annotation and std_after is not None:
                fig.add_annotation(
                    x=0.01,
                    y=0.05,
                    xref="x domain",
                    yref="y domain",
                    text=mean_label,
                    showarrow=False,
                    font=dict(
                        size=_DRIFT_ANNOTATION_FONT_SIZE,
                        color=palette.annotation_text_color,
                    ),
                    align="left",
                    bgcolor=palette.annotation_bg,
                    bordercolor=palette.annotation_border,
                    borderwidth=0.8,
                    row=1,
                    col=1,
                )

    if (
        show_residuals
        and fit_info
        and fit_info.get("coeffs")
        and "x_data" in fit_info
        and "y_data" in fit_info
    ):
        x_data = np.array(fit_info.get("x_data", []), dtype=float)
        y_data = np.array(absolute_fit_series.get("y_data", []), dtype=float)
        if len(x_data) and len(y_data):
            residuals = _compute_display_residuals(
                x_data=x_data,
                y_data_absolute=y_data,
                fit_info=fit_info,
                permil_mode=permil_mode,
                reference_mean=ref_mean,
            )
            # D4: color residuals outside ±2σ to highlight leverage points
            _res_sd = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
            _outlier_res = (
                np.abs(residuals) > 2.0 * _res_sd
                if _res_sd > 0
                else np.zeros(len(residuals), dtype=bool)
            )
            _colors_res = [
                palette.layer_excluded if o else smp_color for o in _outlier_res
            ]
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=residuals,
                    mode="markers",
                    marker=dict(
                        color=_colors_res,
                        size=_DRIFT_SAMPLE_MARKER_SIZE,
                        symbol="circle",
                    ),
                    name="Residuals",
                    showlegend=False,
                    hovertemplate=(
                        "<b>Residual</b><br>X: %{x}<br>Residual: %{y:.3f} ‰<extra></extra>"
                        if permil_mode
                        else "<b>Residual</b><br>X: %{x}<br>Residual: %{y:.6e}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )
            fig.add_hline(
                y=0, line_color=palette.figure_ink, line_width=1, row=2, col=1
            )
            if _res_sd > 0:
                fig.add_hline(
                    y=2 * _res_sd,
                    line_color=palette.guide_bounds,
                    line_width=0.8,
                    line_dash="dot",
                    row=2,
                    col=1,
                )
                fig.add_hline(
                    y=-2 * _res_sd,
                    line_color=palette.guide_bounds,
                    line_width=0.8,
                    line_dash="dot",
                    row=2,
                    col=1,
                )

    # Apply the shared theme first so plot-specific controls below (such as
    # the optional grid) take precedence over the profile defaults.
    theme.apply_to_figure(fig, profile="timeseries")

    x_label = _get_x_axis_label(x_axis)

    fig.update_layout(
        height=DRIFT_PREVIEW_HEIGHT if show_residuals else DRIFT_PREVIEW_COMPACT_HEIGHT,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor=palette.annotation_bg,
            font=dict(size=_DRIFT_LEGEND_FONT_SIZE),
        ),
        margin=dict(l=70, r=40, t=70, b=70),
        font=dict(size=_DRIFT_TICK_FONT_SIZE),
    )
    fig.update_xaxes(
        title_text=x_label,
        row=2,
        col=1,
        showgrid=show_grid,
        zeroline=False,
        showline=True,
        tickfont=dict(size=_DRIFT_TICK_FONT_SIZE),
        title_font=dict(size=_DRIFT_AXIS_TITLE_FONT_SIZE),
    )
    fig.update_xaxes(
        showgrid=show_grid,
        zeroline=False,
        showline=True,
        tickfont=dict(size=_DRIFT_TICK_FONT_SIZE),
    )
    fig.update_yaxes(
        title_text=format_name(ratio_name) + permil_label_suffix,
        row=1,
        col=1,
        showgrid=show_grid,
        zeroline=False,
        showline=True,
        tickfont=dict(size=_DRIFT_TICK_FONT_SIZE),
        title_font=dict(size=_DRIFT_AXIS_TITLE_FONT_SIZE),
    )
    fig.update_yaxes(
        title_text="Residual" + permil_label_suffix,
        row=2,
        col=1,
        showgrid=show_grid,
        zeroline=False,
        showline=True,
        tickfont=dict(size=_DRIFT_TICK_FONT_SIZE),
        title_font=dict(size=_DRIFT_AXIS_TITLE_FONT_SIZE),
        visible=show_residuals,
    )
    if not show_residuals:
        fig.update_xaxes(
            row=1,
            col=1,
            title_text=x_label,
            title_font=dict(size=_DRIFT_AXIS_TITLE_FONT_SIZE),
            showticklabels=True,
            tickfont=dict(size=_DRIFT_TICK_FONT_SIZE),
        )

    from ui.config_plotly import get_plotly_config

    st.plotly_chart(
        fig,
        width="stretch",
        key="drift_preview",
        config=get_plotly_config(
            f"traceiso_{ratio_name.replace('/', '-')}_drift",
            width=_DRIFT_EXPORT_WIDTH,
            height=_DRIFT_EXPORT_HEIGHT,
        ),
    )

    # Return plot data to be rendered at full width
    return {
        "fit_info": fit_info,
        "preview_samples": preview_samples,
        "standards": standards,
        "samples": samples,
        "active_result_samples": active_result_samples,
        "cycle_ranges": cycle_ranges,
    }


def _render_preview_details(
    state, ratio_name: str, preview: Optional[dict], data: dict
) -> None:
    """Render full-width details below the preview plot (equation, summary table, statistics, commit buttons)."""
    fit_info = data["fit_info"]
    preview_samples = data["preview_samples"]
    standards = data["standards"]
    samples = data["samples"]
    active_result_samples = data["active_result_samples"]
    cycle_ranges = data["cycle_ranges"]
    config = state.processing_config.drift

    # Lead with the before/after statistics; the model equation is supporting
    # detail in an expander below (U69).
    if fit_info and "coeffs" in fit_info and fit_info["coeffs"]:
        _render_statistics_comparison(fit_info)

        with st.expander("Drift model equation and fit metrics", expanded=False):
            _render_drift_equation(fit_info, config, nested=True)

    st.divider()

    # Show drift correction summary table (pass preview samples for 'after' data)
    if fit_info and preview_samples:
        preview_stds = [s for s in preview_samples if s.is_standard]
        preview_smps = [s for s in preview_samples if s.is_sample]
        _render_drift_summary_table(
            standards,
            samples,
            ratio_name,
            fit_info,
            before_all_samples=active_result_samples,
            preview_standards=preview_stds,
            preview_samples=preview_smps,
            cycle_ranges=cycle_ranges,
        )

    if fit_info:
        if preview:
            st.divider()
            render_status_chip(
                "Preview ready · you have reviewed the fit above",
                tone="warning",
            )
            commit_label = (
                "Commit Drift Correction"
                if config.enabled
                else "Apply drift correction and reprocess"
            )
            if st.button(
                commit_label,
                type="primary",
                width="stretch",
                key="commit_drift_details",
                help=(
                    "Re-runs the canonical processing pipeline with this fit. "
                    "Same action as the button in Drift correction settings — "
                    "placed here so you can commit without scrolling back up."
                ),
            ):
                _apply_drift(state, ratio_name)
        else:
            render_status_chip("Drift correction active", tone="positive")
            st.caption(
                "Statistics shown above reflect the currently applied correction."
            )
    else:
        render_status_chip(
            "Run a preview to review and commit this fit",
            tone="neutral",
        )


def _render_statistics_comparison(fit_info: dict) -> None:
    """Render the statistics comparison metrics."""
    st.subheader("Statistics Comparison")

    if fit_info and "rsd_before" in fit_info and "rsd_after" in fit_info:
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric(
                "RSD (Before)",
                f"{fit_info['rsd_before']:.4f}%",
                help="Relative standard deviation of standards before correction",
            )

        with col_b:
            st.metric(
                "RSD (After)",
                f"{fit_info['rsd_after']:.4f}%",
                delta=f"{fit_info['rsd_after'] - fit_info['rsd_before']:.4f}%",
                delta_color="inverse",
                help="Relative standard deviation after drift correction",
            )

        with col_c:
            rsd_before = fit_info["rsd_before"]
            rsd_after = fit_info["rsd_after"]
            improvement = (
                (rsd_before - rsd_after) / rsd_before * 100 if rsd_before > 0 else 0
            )
            st.metric(
                "Improvement",
                f"{improvement:.1f}%",
                help="Reduction in RSD",
            )
    else:
        st.info("Run preview first to see statistics comparison.")


def _render_drift_equation(fit_info: dict, config, *, nested: bool = False) -> None:
    """Render drift correction equation and R² value.

    ``nested`` is set when this renders inside an ``st.expander`` (U69): the
    centered/scaled z-form is then shown under a caption rather than a second,
    illegal nested expander.
    """
    method = fit_info.get("method", "polynomial")
    coeffs = fit_info.get("coeffs", [])

    if not coeffs:
        return

    is_centered = fit_info.get("fit_format") == "centered_scaled_v1"
    raw_coeffs: Optional[List[float]] = None
    if is_centered:
        center = float(fit_info.get("x_center", 0.0))
        scale = float(fit_info.get("x_scale", 1.0))
        if scale != 0.0:
            raw_coeffs = _expand_centered_coeffs_to_raw_x(coeffs, center, scale)

    if method in ("linear", "polynomial"):
        # Raw-x form is the familiar "y = Ax² + Bx + C" a viewer expects; show it
        # as the primary equation. The z-form (what was actually fit, for
        # numerical stability) is algebraically identical — kept in an expander.
        equation = _format_polynomial_equation_html(
            raw_coeffs if raw_coeffs is not None else coeffs
        )
    else:
        equation = f"{method.capitalize()} fit (no analytical equation)"

    # Read precomputed metrics from _compute_fit_quality_metrics to avoid
    # duplicating the regression; falls back to None for absent metrics.
    metrics = _compute_fit_quality_metrics(fit_info)
    r_squared = metrics["r_squared"]
    residual_sd = metrics["residual_sd"]
    n_used = int(metrics["n_points"]) if metrics["n_points"] else None
    n_total_raw = metrics["n_total"]
    n_total = int(n_total_raw) if n_total_raw is not None else None

    st.caption("**Drift Correction Model**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if r_squared is not None:
            st.metric("R²", f"{r_squared:.6f}")
    with col2:
        if residual_sd is not None:
            st.metric("Residual SD", f"{residual_sd:.6e}")
    with col3:
        if n_used is not None:
            if n_total is not None and n_total >= n_used:
                st.metric("Standards Used", f"{n_used}/{n_total}")
            else:
                st.metric("Standards Used", str(n_used))
    st.markdown(
        (
            "<div style='padding:0.75rem 1rem;border:1px solid rgba(255,255,255,0.1);"
            "border-radius:0.5rem;background:rgba(255,255,255,0.02);font-family:ui-monospace, SFMono-Regular, monospace;'>"
            f"{equation}</div>"
        ),
        unsafe_allow_html=True,
    )

    if is_centered and raw_coeffs is not None:
        z_equation = _format_polynomial_equation_html(coeffs, fit_info=fit_info)
        z_caption = (
            "The polynomial above is algebraically identical to this — z is the "
            "run number, centered and rescaled, used only so the least-squares "
            "solve stays numerically stable. Not physically meaningful on its own."
        )
        z_markdown = (
            "<div style='padding:0.75rem 1rem;border:1px solid rgba(255,255,255,0.1);"
            "border-radius:0.5rem;background:rgba(255,255,255,0.02);"
            "font-family:ui-monospace, SFMono-Regular, monospace;'>"
            f"{z_equation}</div>"
        )
        if nested:
            st.caption("**Internal fit form (centered/scaled z)**")
            st.caption(z_caption)
            st.markdown(z_markdown, unsafe_allow_html=True)
        else:
            with st.expander("Show internal fit form (centered/scaled z)"):
                st.caption(z_caption)
                st.markdown(z_markdown, unsafe_allow_html=True)


def _format_polynomial_equation_html(
    coeffs: List[float], *, fit_info: Optional[Dict] = None
) -> str:
    """Render polynomial equation with superscript powers for display."""
    degree = len(coeffs) - 1
    variable = (
        "z" if fit_info and fit_info.get("fit_format") == "centered_scaled_v1" else "x"
    )
    pieces: List[str] = ["y = "]
    for i, coeff in enumerate(coeffs):
        power = degree - i
        if abs(coeff) < 1e-15:
            continue
        sign = "-" if coeff < 0 else "+"
        coeff_html = _format_scientific_html(abs(coeff))
        if power == 0:
            term = coeff_html
        elif power == 1:
            term = f"{coeff_html}{variable}"
        else:
            term = f"{coeff_html}{variable}<sup>{power}</sup>"
        if len(pieces) == 1:
            pieces.append(f"- {term}" if coeff < 0 else term)
        else:
            pieces.append(f" {sign} {term}")
    equation = "".join(pieces)
    if variable == "z":
        center = _format_scientific_html(float(fit_info.get("x_center", 0.0)))
        scale = _format_scientific_html(float(fit_info.get("x_scale", 1.0)))
        equation += f"; z = (x - {center}) / {scale}"
    return equation


def _format_scientific_html(value: float) -> str:
    """Format coefficient in scientific notation with superscript exponent."""
    mantissa, exponent = f"{value:.6e}".split("e")
    return f"{mantissa}×10<sup>{int(exponent)}</sup>"


def _expand_centered_coeffs_to_raw_x(
    coeffs: List[float], center: float, scale: float
) -> List[float]:
    """Convert coefficients fit in z = (x-center)/scale back into raw-x coefficients.

    Pure change of variable (z is an affine function of x), so the returned
    polynomial is algebraically identical to the z-form — not an approximation.
    Highest-degree-first ordering, matching np.polyfit's convention.
    """
    z_as_x = np.poly1d([1.0 / scale, -center / scale])  # z(x) = x/scale - center/scale
    raw = np.poly1d([0.0])
    degree = len(coeffs) - 1
    for i, c in enumerate(coeffs):
        power = degree - i
        raw = raw + c * (z_as_x**power)
    return list(raw.coefficients)


def _render_drift_summary_table(
    standards,
    samples,
    ratio_name: str,
    fit_info: dict,
    before_all_samples: Optional[List[Sample]] = None,
    preview_standards: Optional[List[Sample]] = None,
    preview_samples: Optional[List[Sample]] = None,
    cycle_ranges: Optional[Dict[str, tuple[int, int]]] = None,
) -> None:
    """Render drift correction summary statistics table.

    'Before' columns come from *standards* / *samples* (committed result).
    'After' columns come from *preview_standards* / *preview_samples* (preview).
    """
    import pandas as pd
    from domain.ratio_selection import get_best_pre_drift_ratio_data

    if not fit_info:
        return

    config_x_axis = fit_info.get("x_axis", "run_number")
    coeffs = fit_info.get("coeffs", [])
    before_map_source = (
        before_all_samples if before_all_samples is not None else (standards + samples)
    )
    before_x_map = _build_sample_x_map(before_map_source, config_x_axis)
    _preview_std_map: Dict[tuple[str, int], Sample] = {}
    _preview_smp_map: Dict[tuple[str, int], Sample] = {}
    if preview_standards:
        _preview_std_map = {_sample_identity(s): s for s in preview_standards}
    if preview_samples:
        _preview_smp_map = {_sample_identity(s): s for s in preview_samples}

    rows = []

    # Helper to get predicted drift value at x position
    def predict_drift(x_val):
        if coeffs:
            return float(_eval_fit(fit_info, x_val))
        return np.nan

    std_count = 0
    for std in standards:
        # Get x value
        x_val = before_x_map.get(
            id(std),
            float(std.run_number if config_x_axis == "run_number" else std_count),
        )
        std_count += 1

        ratio_data = get_best_pre_drift_ratio_data(std, ratio_name)

        before_vals = _get_active_ratio_values(
            ratio_data,
            std.name,
            sample_key=get_sample_state_key(std),
            cycle_ranges=cycle_ranges,
        )
        if len(before_vals) == 0:
            continue

        n = len(before_vals)
        before_mean = float(np.nanmean(before_vals))
        before_sd = float(np.nanstd(before_vals, ddof=1)) if n > 1 else np.nan
        before_se = before_sd / np.sqrt(n) if n > 0 else np.nan
        before_rsd = (before_sd / before_mean * 100) if before_mean != 0 else np.nan

        # Predicted drift and correction factor
        predicted_drift = predict_drift(x_val)
        normalization_target = float(fit_info.get("std_mean", np.nan))
        correction_factor = (
            normalization_target / predicted_drift
            if np.isfinite(normalization_target)
            and np.isfinite(predicted_drift)
            and predicted_drift != 0
            else np.nan
        )

        after_mean = np.nan
        after_sd = np.nan
        after_rsd = np.nan
        p_std = _preview_std_map.get(_sample_identity(std))
        if p_std and ratio_name in p_std.drift_corrected_ratios:
            corr_data = p_std.drift_corrected_ratios[ratio_name]
            after_vals = _get_active_ratio_values(
                corr_data,
                std.name,
                sample_key=get_sample_state_key(std),
                cycle_ranges=cycle_ranges,
            )
            if len(after_vals) > 0:
                after_mean = float(np.nanmean(after_vals))
                after_sd = (
                    float(np.nanstd(after_vals, ddof=1)) if len(after_vals) > 1 else np.nan
                )
                after_rsd = (after_sd / after_mean * 100) if after_mean != 0 else np.nan

        rows.append(
            {
                "Sample": std.name,
                "Type": "STD",
                f"{ratio_name}": f"{before_mean:.6f}",
                "2SD": f"{2 * before_sd:.6f}" if np.isfinite(before_sd) else "Unavailable",
                "2SE": f"{2 * before_se:.6f}" if np.isfinite(before_se) else "Unavailable",
                "n": n,
                "RSD (%)": f"{before_rsd:.4f}" if np.isfinite(before_rsd) else "Unavailable",
                "Predicted Drift": f"{predicted_drift:.6f}"
                if not np.isnan(predicted_drift)
                else "—",
                "Correction Factor": f"{correction_factor:.6f}"
                if not np.isnan(correction_factor)
                else "—",
                f"Drift Corrected {ratio_name}": f"{after_mean:.6f}"
                if not np.isnan(after_mean)
                else "—",
                "Drift Corrected RSD (%)": f"{after_rsd:.4f}"
                if not np.isnan(after_rsd)
                else "—",
            }
        )

    smp_count = 0
    for smp in samples:
        x_val = before_x_map.get(
            id(smp),
            float(smp.run_number if config_x_axis == "run_number" else smp_count),
        )
        smp_count += 1

        ratio_data = get_best_pre_drift_ratio_data(smp, ratio_name)

        before_vals = _get_active_ratio_values(
            ratio_data,
            smp.name,
            sample_key=get_sample_state_key(smp),
            cycle_ranges=cycle_ranges,
        )
        if len(before_vals) == 0:
            continue

        n = len(before_vals)
        before_mean = float(np.nanmean(before_vals))
        before_sd = float(np.nanstd(before_vals, ddof=1)) if n > 1 else np.nan
        before_se = before_sd / np.sqrt(n) if n > 0 else np.nan
        before_rsd = (before_sd / before_mean * 100) if before_mean != 0 else np.nan

        # Predicted drift and correction factor
        predicted_drift = predict_drift(x_val)
        normalization_target = float(fit_info.get("std_mean", np.nan))
        correction_factor = (
            normalization_target / predicted_drift
            if np.isfinite(normalization_target)
            and np.isfinite(predicted_drift)
            and predicted_drift != 0
            else np.nan
        )

        after_mean = np.nan
        after_sd = np.nan
        after_rsd = np.nan
        p_smp = _preview_smp_map.get(_sample_identity(smp))
        if p_smp and ratio_name in p_smp.drift_corrected_ratios:
            corr_data = p_smp.drift_corrected_ratios[ratio_name]
            after_vals = _get_active_ratio_values(
                corr_data,
                smp.name,
                sample_key=get_sample_state_key(smp),
                cycle_ranges=cycle_ranges,
            )
            if len(after_vals) > 0:
                after_mean = float(np.nanmean(after_vals))
                after_sd = (
                    float(np.nanstd(after_vals, ddof=1)) if len(after_vals) > 1 else np.nan
                )
                after_rsd = (after_sd / after_mean * 100) if after_mean != 0 else np.nan

        rows.append(
            {
                "Sample": smp.name,
                "Type": "SMP",
                f"{ratio_name}": f"{before_mean:.6f}",
                "2SD": f"{2 * before_sd:.6f}" if np.isfinite(before_sd) else "Unavailable",
                "2SE": f"{2 * before_se:.6f}" if np.isfinite(before_se) else "Unavailable",
                "n": n,
                "RSD (%)": f"{before_rsd:.4f}" if np.isfinite(before_rsd) else "Unavailable",
                "Predicted Drift": f"{predicted_drift:.6f}"
                if not np.isnan(predicted_drift)
                else "—",
                "Correction Factor": f"{correction_factor:.6f}"
                if not np.isnan(correction_factor)
                else "—",
                f"Drift Corrected {ratio_name}": f"{after_mean:.6f}"
                if not np.isnan(after_mean)
                else "—",
                "Drift Corrected RSD (%)": f"{after_rsd:.4f}"
                if not np.isnan(after_rsd)
                else "—",
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        st.caption("**Drift Correction Summary Table**")
        st.dataframe(df, width="stretch", hide_index=True)


def _build_sample_x_map(samples: List[Sample], x_axis: str) -> Dict[int, float]:
    """Build a per-sample x-coordinate map consistent with drift math."""
    return build_drift_x_value_map(samples, x_axis)


def _get_x_axis_label(x_axis: str) -> str:
    """Return the human-readable x-axis label for drift plots."""
    if x_axis == "run_number":
        return "Run Number"
    if x_axis == "time_minutes":
        return "Elapsed Time (min)"
    return "Index"


def _get_active_ratio_values(
    cycle_data,
    sample_name: str,
    *,
    sample_key: Optional[str] = None,
    cycle_ranges: Optional[Dict[str, tuple[int, int]]] = None,
) -> np.ndarray:
    """Return mask-respecting ratio values within the active Inspector window."""
    if cycle_data is None:
        return np.array([], dtype=np.float64)

    return np.asarray(
        get_filtered_values(
            cycle_data.values,
            cycle_data.mask,
            sample_name,
            cycle_ranges=cycle_ranges,
            sample_key=sample_key,
            filter_method="None",
            filter_threshold=2.0,
        ),
        dtype=np.float64,
    )

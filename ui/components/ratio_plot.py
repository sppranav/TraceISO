"""Ratio plot component for TraceISO."""

from __future__ import annotations

from typing import Optional, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st  # noqa: F401 - retained for test/session-state monkeypatch compatibility

from config.constants import DEFAULT_OUTLIER_THRESHOLD_SD
from domain.filters.outlier import calculate_thresholds, get_filtered_values, get_runtime_mask
from domain.models import Sample, CycleData
from domain.layers import cycle_data_equal
from domain.ratio_selection import get_delta_cycle_data, get_ssb_cycle_data
from config.settings import DisplayConfig
from ui.config_plotly import (
    PLOTLY_ANNOTATION_FONT_SIZE,
    PLOTLY_AXIS_TITLE_FONT_SIZE,
    PLOTLY_BASE_FONT_SIZE,
    PLOTLY_LEGEND_FONT_SIZE,
    PLOTLY_TITLE_FONT_SIZE,
    WIDE_TIMESERIES_MAX_HEIGHT,
)
from ui.theme import get_theme
from ui.state import get_state
from ui.components.plot_perf import (
    get_plot_perf_options,
    resolve_marker_outline_width,
    should_simplify_hover,
)
from ui.components.plot_helpers import add_legend_proxy
from ui.utils import (
    blank_correction_active,
    decimals_for_uncertainty,
    format_name,
    format_delta_html,
    get_plot_config,
)


def _get_plot_config() -> dict:
    return get_plot_config(get_state())


def _blank_correction_active(sample: Sample) -> bool:
    return blank_correction_active(sample, get_state())


def _slice_valid(cd: CycleData, all_cycles: np.ndarray, cycle_range: tuple):
    """Return valid cycles/values from CycleData in the selected range."""
    if cycle_range:
        start_idx, end_idx = cycle_range[0] - 1, cycle_range[1]
        sel_cycles = all_cycles[start_idx:end_idx]
        sel_values = cd.values[start_idx:end_idx]
        sel_mask = cd.mask[start_idx:end_idx]
    else:
        sel_cycles = all_cycles
        sel_values = cd.values
        sel_mask = cd.mask
    return sel_cycles[sel_mask], sel_values[sel_mask]


def _slice_cycle_view(cd: CycleData, all_cycles: np.ndarray, cycle_range: tuple | None):
    """Return cycles, values, and mask in the selected range."""
    if cycle_range:
        start_idx, end_idx = cycle_range[0] - 1, cycle_range[1]
        return (
            all_cycles[start_idx:end_idx],
            cd.values[start_idx:end_idx],
            cd.mask[start_idx:end_idx],
        )
    return all_cycles, cd.values, cd.mask


def _get_runtime_filtered_values(
    sample_name: str,
    cycle_data: CycleData,
    *,
    cycle_range: tuple | None = None,
) -> np.ndarray:
    """Return values using the same active range/filter logic as Results."""
    _, filter_method, filter_threshold, _ = _get_active_filter_config()

    cycle_ranges = {sample_name: cycle_range} if cycle_range else None
    return get_filtered_values(
        cycle_data.values,
        cycle_data.mask,
        sample_name,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
    )


def _slice_runtime_cycle_view(
    sample_name: str,
    cycle_data: CycleData,
    all_cycles: np.ndarray,
    cycle_range: tuple | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a cycle view with runtime re-filtering applied to the visible mask."""
    cycles, values, mask = _slice_cycle_view(cycle_data, all_cycles, cycle_range)

    _, filter_method, filter_threshold, _ = _get_active_filter_config()
    cycle_ranges = {sample_name: cycle_range} if cycle_range else None
    runtime_mask_full = get_runtime_mask(
        cycle_data.values,
        cycle_data.mask,
        sample_name,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
    )
    if cycle_range:
        start_idx, end_idx = cycle_range[0] - 1, cycle_range[1]
        runtime_mask = runtime_mask_full[start_idx:end_idx]
    else:
        runtime_mask = runtime_mask_full
    return cycles, values, runtime_mask


def _get_active_filter_config() -> tuple[object | None, str, float, str]:
    """Return the active processing filter configuration for plotting."""
    try:
        state = get_state()
        processing_config = getattr(state, "processing_config", None)
        filter_method = processing_config.filter_method if processing_config else "None"
        filter_threshold = (
            processing_config.get_active_filter_threshold()
            if processing_config else DEFAULT_OUTLIER_THRESHOLD_SD
        )
        filter_label = (
            processing_config.format_filter_parameter(filter_method)
            if processing_config and hasattr(processing_config, "format_filter_parameter")
            else f"{filter_threshold:g}"
        )
        return processing_config, filter_method, filter_threshold, filter_label
    except Exception:
        return None, "None", DEFAULT_OUTLIER_THRESHOLD_SD, f"{DEFAULT_OUTLIER_THRESHOLD_SD:g}"


def _outlier_filter_legend_labels(filter_method: str, filter_threshold: float) -> tuple[str, str]:
    """Return legend labels that state the active outlier-filter rule exactly."""
    if filter_method == "Standard deviation":
        return (
            "Outlier-filter mean",
            f"Outlier-filter limits (±{filter_threshold:g} SD)",
        )
    if filter_method == "MAD":
        return (
            "Outlier-filter median",
            f"Outlier-filter limits (±{filter_threshold:g} MAD)",
        )
    if filter_method == "IQR":
        return (
            "Outlier-filter median",
            (
                "Outlier-filter limits "
                f"(Q1 − {filter_threshold:g} IQR to Q3 + {filter_threshold:g} IQR)"
            ),
        )
    return "Outlier-filter center", "Outlier-filter limits"


def _normalise_layer_label(label: str) -> str:
    """Apply consistent hyphenated correction-layer naming."""
    normalised = " ".join((label or "").replace("_", " ").replace("-", " ").split()).lower()
    mapping = {
        "raw": "Raw",
        "corrected": "Corrected",
        "blank corrected": "Blank-corrected",
        "interference corrected": "Interference-corrected",
        "interference corrected (hg)": "Interference-corrected (Hg)",
        "iif corrected": "IIF-corrected",
        "ssb corrected": "SSB-corrected",
        "drift corrected": "Drift-corrected",
    }
    return mapping.get(normalised, label)


def _wide_timeseries_height(requested_height: int) -> int:
    """Keep time-series plots in a wide manuscript-friendly aspect range."""
    return min(requested_height, WIDE_TIMESERIES_MAX_HEIGHT)


def _add_excluded_cycle_trace(
    fig: go.Figure,
    cycles: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    *,
    name: str,
    color: str,
    outline_color: str,
    legendgroup: str | None = None,
) -> None:
    """Render excluded cycles explicitly for corrected ratio traces."""
    excluded_cycles = cycles[~mask]
    if len(excluded_cycles) == 0:
        return

    fig.add_trace(
        go.Scatter(
            x=excluded_cycles,
            y=values[~mask],
            mode="markers",
            name=name,
            showlegend=False,
            legendgroup=legendgroup,
            marker=dict(
                color=color,
                size=11,
                symbol="x",
                opacity=0.95,
                line=dict(color=color, width=2.0),
            ),
            hovertemplate=(
                "<b>Excluded cycle</b><br>"
                "Cycle: %{x}<br>"
                "Value: %{y:.6f}<extra></extra>"
            ),
        )
    )


HG_INVALID_CYCLE_MARKER = "hg_invalid_cycle"


def _add_hg_invalid_cycle_markers(
    fig: go.Figure,
    sample: Sample,
    ratio_name: str,
    *,
    cycle_range: tuple | None,
    color: str,
) -> None:
    """Mark cycles the Hg correction excluded as invalid, with the recorded reason.

    Drawn as vertical guides, not at the excluded value: that value is formed
    with a non-physical corrected 204Pb (negative or non-finite), so plotting it
    would rescale the axis away from the accepted cycles, and a non-finite one
    could not be drawn at all. Presentation only: cycles and reasons come from
    the domain record.
    """
    from domain.pb_hg_correction import hg_invalid_cycle_exclusions

    for cycle, reason in hg_invalid_cycle_exclusions(sample, ratio_name):
        if cycle_range and not (cycle_range[0] <= cycle <= cycle_range[1]):
            continue
        fig.add_shape(
            type="line", xref="x", yref="paper", x0=cycle, x1=cycle, y0=0, y1=1,
            line=dict(color=color, width=1.2, dash="dot"), name=HG_INVALID_CYCLE_MARKER,
        )
        fig.add_annotation(
            x=cycle, y=1, xref="x", yref="paper", yanchor="bottom", showarrow=False,
            text="×", font=dict(color=color, size=14), name=HG_INVALID_CYCLE_MARKER,
            hovertext=f"Cycle {cycle} excluded by the Hg correction: {reason}",
        )


def _add_stats_threshold_overlay(
    fig: go.Figure,
    *,
    sample_name: str,
    cycle_data: CycleData,
    cycle_range: tuple | None,
    line_color: str,
    band_color: str,
    show_annotation: bool = True,
) -> None:
    """Draw mean/2SE plus mean±2SD guide lines for a ratio layer."""
    valid_for_threshold = _get_runtime_filtered_values(
        sample_name,
        cycle_data,
        cycle_range=cycle_range,
    )
    if len(valid_for_threshold) <= 1:
        return

    mean_val = np.nanmean(valid_for_threshold)
    std_val = np.nanstd(valid_for_threshold, ddof=1)
    se_val = std_val / np.sqrt(len(valid_for_threshold))
    two_se = 2 * se_val
    palette = get_theme().palette

    if two_se > 0:
        fig.add_hrect(
            y0=mean_val - two_se,
            y1=mean_val + two_se,
            fillcolor=band_color,
            opacity=0.22,
            layer="below",
            line_width=0,
        )
        add_legend_proxy(
            fig,
            name="Mean ± 2SE",
            fillcolor=band_color,
            legendgroup="stats_guides",
        )
    fig.add_hline(
        y=mean_val,
        line_dash="dash",
        line_color=line_color,
        line_width=2,
    )
    add_legend_proxy(
        fig,
        name="Mean",
        line=dict(color=line_color, width=2, dash="dash"),
        legendgroup="stats_guides",
    )
    if std_val > 0:
        fig.add_hline(
            y=mean_val + 2 * std_val,
            line_dash="dot",
            line_color=palette.guide_bounds,
            line_width=1,
            opacity=0.8,
        )
        add_legend_proxy(
            fig,
            name="Mean ± 2SD",
            line=dict(color=palette.guide_bounds, width=1, dash="dot"),
            legendgroup="stats_guides",
        )
        fig.add_hline(
            y=mean_val - 2 * std_val,
            line_dash="dot",
            line_color=palette.guide_bounds,
            line_width=1,
            opacity=0.8,
        )

    if show_annotation:
        dec = decimals_for_uncertainty([two_se])
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.02,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            text=(
                f"Mean \u00b1 2SE: {mean_val:.{dec}f} \u00b1 {two_se:.{dec}f} "
                f"(n = {len(valid_for_threshold)})"
            ),
            font=dict(size=PLOTLY_ANNOTATION_FONT_SIZE, color=palette.annotation_text_color),
            bgcolor=palette.annotation_bg,
            bordercolor=palette.annotation_border,
            borderwidth=1,
        )


def _add_processing_threshold_overlay(
    fig: go.Figure,
    *,
    sample_name: str,
    cycle_data: CycleData,
    all_cycles: np.ndarray,
    cycle_range: tuple | None,
    line_color: str,
    bounds_color: str,
) -> None:
    """Draw the active outlier-filter bounds for the canonical processing layer."""
    _, filter_method, filter_threshold, _ = _get_active_filter_config()
    if filter_method == "None":
        return

    visible_cycles, visible_values, visible_mask = _slice_cycle_view(
        cycle_data, all_cycles, cycle_range
    )
    candidate_values = visible_values[visible_mask & np.isfinite(visible_values)]
    if len(candidate_values) <= 1:
        return

    center, lower, upper = calculate_thresholds(
        candidate_values,
        method=filter_method,
        threshold=filter_threshold,
    )
    fig.add_hline(
        y=center,
        line_dash="dash",
        line_color=line_color,
        line_width=2,
    )
    center_label, bounds_label = _outlier_filter_legend_labels(
        filter_method,
        filter_threshold,
    )
    add_legend_proxy(
        fig,
        name=center_label,
        line=dict(color=line_color, width=2, dash="dash"),
        legendgroup="filter_guides",
    )
    fig.add_hline(
        y=upper,
        line_dash="dot",
        line_color=bounds_color,
        line_width=1,
        opacity=0.8,
    )
    add_legend_proxy(
        fig,
        name=bounds_label,
        line=dict(color=bounds_color, width=1, dash="dot"),
        legendgroup="filter_guides",
    )
    if abs(lower - upper) > 0:
        fig.add_hline(
            y=lower,
            line_dash="dot",
            line_color=bounds_color,
            line_width=1,
            opacity=0.8,
        )


def _add_ratio_stats_annotation(
    fig: go.Figure,
    *,
    sample_name: str,
    cycle_data: CycleData,
    cycle_range: tuple | None,
    decimals: int | None = None,
    statistic: str = "2SD",
) -> None:
    """Draw the compact ratio stats box using runtime-filtered values."""
    valid_values = _get_runtime_filtered_values(
        sample_name,
        cycle_data,
        cycle_range=cycle_range,
    )
    if len(valid_values) <= 1:
        return

    mean_val = np.nanmean(valid_values)
    std_val = np.nanstd(valid_values, ddof=1)
    statistic = "2SE" if statistic == "2SE" else "2SD"
    spread = (
        2 * (std_val / np.sqrt(len(valid_values)))
        if statistic == "2SE"
        else 2 * std_val
    )
    dec = decimals if decimals is not None else decimals_for_uncertainty([spread])
    palette = get_theme().palette

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.005,
        y=0.005,
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        text=(
            f"Mean \u00b1 {statistic}: {mean_val:.{dec}f} \u00b1 {spread:.{dec}f} "
            f"(n = {len(valid_values)})"
        ),
        font=dict(size=PLOTLY_ANNOTATION_FONT_SIZE, color=palette.annotation_text_color),
        bgcolor=palette.annotation_bg,
        bordercolor=palette.annotation_border,
        borderwidth=1,
    )


def _stats_box_padded_y_range(
    fig: go.Figure,
    *,
    lower_fraction: float = 0.20,
    upper_fraction: float = 0.04,
) -> Optional[List[float]]:
    """Return a y-range with a clear lower annotation zone."""
    values: List[float] = []
    for trace in fig.data:
        if getattr(trace, "visible", None) in {False, "legendonly"}:
            continue
        y_values = getattr(trace, "y", None)
        if y_values is None:
            continue
        try:
            numeric = np.asarray(y_values, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        values.extend(numeric[np.isfinite(numeric)].tolist())

    for shape in fig.layout.shapes or ():
        if getattr(shape, "visible", None) is False:
            continue
        yref = str(getattr(shape, "yref", None) or "y")
        if yref not in {"y", "y1"}:
            continue
        for attr in ("y0", "y1"):
            try:
                value = float(getattr(shape, attr))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)

    if not values:
        return None

    y_min = float(min(values))
    y_max = float(max(values))
    span = y_max - y_min
    if not np.isfinite(span) or span <= 0:
        span = max(abs(y_min) * 1e-6, 1e-9)
    return [
        y_min - lower_fraction * span,
        y_max + upper_fraction * span,
    ]


def _build_ssb_cycle_data(sample: Sample, ratio_name: str) -> Optional[CycleData]:
    """Build a CycleData view for SSB-corrected cycles."""
    return get_ssb_cycle_data(sample, ratio_name)


def create_ratio_plot(
    sample: Sample,
    ratios: Optional[List[str]] = None,
    display_config: Optional[DisplayConfig] = None,
    height: int = 350,
    certified_value: Optional[float] = None,
) -> go.Figure:
    """Create a ratio plot for a sample."""
    theme = get_theme()
    palette = theme.palette
    outline_color = palette.figure_axis

    plot_config = _get_plot_config()
    marker_size = plot_config.get("marker_size", 9)
    plot_height = _wide_timeseries_height(plot_config.get("height", height))
    show_grid = plot_config.get("show_grid", True)
    perf_options = get_plot_perf_options(plot_config)

    if display_config is None:
        display_config = DisplayConfig()

    fig = go.Figure()

    if ratios is None:
        ratios = list(sample.ratios.keys()) if sample.ratios else []

    if not ratios:
        fig.add_annotation(
            text="No ratio data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        theme.apply_to_figure(fig, profile="timeseries")
        return fig

    blank_layer_active = _blank_correction_active(sample)

    for ratio_name in ratios:
        raw_ratio = sample.ratios.get(ratio_name) if sample.ratios else None

        # Raw ratios
        if display_config.show_raw and ratio_name in sample.ratios:
            cycle_data = sample.ratios[ratio_name]
            cycles = np.arange(1, len(cycle_data.values) + 1)
            valid_cycles = cycles[cycle_data.mask]
            valid_values = cycle_data.values[cycle_data.mask]
            has_corrected_trace = False
            if display_config.show_corrected:
                if sample.iif_corrected_ratios and ratio_name in sample.iif_corrected_ratios:
                    has_corrected_trace = True
                elif sample.ssb_results and ratio_name in sample.ssb_results:
                    has_corrected_trace = True
                elif blank_layer_active and sample.corrected_ratios and ratio_name in sample.corrected_ratios:
                    has_corrected_trace = True
            n_raw = len(valid_cycles)
            raw_outline = resolve_marker_outline_width(
                n_points=n_raw,
                requested_width=1.0,
                outline_mode=perf_options["outline_mode"],
                outline_max_points=perf_options["outline_max_points"],
            )
            skip_raw_hover = (
                has_corrected_trace
                and should_simplify_hover(
                    n_points=n_raw,
                    dense_hover_simplify=perf_options["dense_hover_simplify"],
                    dense_hover_max_points=perf_options["dense_hover_max_points"],
                )
            )

            if len(valid_cycles) > 0:
                raw_kwargs = dict(
                    x=valid_cycles,
                    y=valid_values,
                    mode="markers",
                    name=f"{format_name(ratio_name)} (Raw)",
                    marker=dict(
                        color=palette.layer_raw,
                        size=marker_size,
                        symbol="circle" if raw_outline == 0 else "circle-open",
                        # Keep raw observations fully legible beside corrected
                        # layers; the hollow symbol and outline carry identity.
                        opacity=0.9,
                        line=dict(width=raw_outline, color=outline_color),
                    ),
                    legendgroup=ratio_name,
                )
                if skip_raw_hover:
                    raw_kwargs["hoverinfo"] = "skip"
                else:
                    raw_kwargs["hovertemplate"] = (
                        f"<b>{format_name(ratio_name)} (Raw)</b><br>"
                        "Cycle: %{x}<br>"
                        "Value: %{y:.6f}<br>"
                        "<extra></extra>"
                    )
                fig.add_trace(go.Scattergl(**raw_kwargs))

            if display_config.show_filtered_raw:
                _add_excluded_cycle_trace(
                    fig,
                    cycles,
                    cycle_data.values,
                    cycle_data.mask,
                    name="Excluded cycles",
                    color=palette.layer_excluded,
                    outline_color=outline_color,
                    legendgroup=ratio_name,
                )

        # Corrected ratios
        if display_config.show_corrected:
            corrected_data = None
            correction_label = "Corrected"

            if sample.iif_corrected_ratios and ratio_name in sample.iif_corrected_ratios:
                corrected_data = sample.iif_corrected_ratios[ratio_name]
                correction_label = "IIF-corrected"
            elif sample.ssb_results and ratio_name in sample.ssb_results:
                corrected_data = _build_ssb_cycle_data(sample, ratio_name)
                correction_label = "SSB-corrected"
            elif blank_layer_active and sample.corrected_ratios and ratio_name in sample.corrected_ratios:
                corrected_data = sample.corrected_ratios[ratio_name]
                if display_config.show_raw and cycle_data_equal(corrected_data, raw_ratio):
                    corrected_data = None

            if corrected_data is not None:
                is_final_layer = correction_label in {"IIF-corrected", "SSB-corrected"}
                corrected_color = palette.layer_final if is_final_layer else palette.layer_blank
                corrected_symbol = "diamond" if is_final_layer else "circle"
                cycles = np.arange(1, len(corrected_data.values) + 1)
                valid_cycles = cycles[corrected_data.mask]
                valid_values = corrected_data.values[corrected_data.mask]
                corr_outline = resolve_marker_outline_width(
                    n_points=len(valid_cycles),
                    requested_width=1.2,
                    outline_mode=perf_options["outline_mode"],
                    outline_max_points=perf_options["outline_max_points"],
                )

                fig.add_trace(
                    go.Scattergl(
                        x=valid_cycles,
                        y=valid_values,
                        mode="markers+lines",
                        name=f"{format_name(ratio_name)} ({correction_label})",
                        marker=dict(
                            color=corrected_color,
                            size=marker_size + 2,
                            symbol=corrected_symbol,
                            line=dict(width=corr_outline, color=outline_color),
                        ),
                        line=dict(color=corrected_color, width=1.5),
                        legendgroup=ratio_name,
                        hovertemplate=(
                            f"<b>{correction_label}</b><br>"
                            "Cycle: %{x}<br>"
                            "Value: %{y:.6f}<br>"
                            "<extra></extra>"
                        ),
                    )
                )

                if display_config.show_filtered_corrected:
                    _add_excluded_cycle_trace(
                        fig,
                        cycles,
                        corrected_data.values,
                        corrected_data.mask,
                        name="Excluded cycles",
                        color=palette.layer_excluded,
                        outline_color=outline_color,
                        legendgroup=ratio_name,
                    )

                # Mean lines (simplified for multi-ratio view)
                if display_config.show_threshold_lines and len(valid_values) > 2:
                    mean_val = np.nanmean(valid_values)
                    fig.add_hline(
                        y=mean_val,
                        line_dash="dash",
                        line_color=corrected_color,
                        line_width=1.5,
                        opacity=0.7,
                    )

    # CRM Line
    if certified_value is not None:
        fig.add_hline(
            y=certified_value,
            line_dash="longdash",
            line_color=palette.figure_ink,
            line_width=2,
            annotation_text="Certified",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=dict(
            text=f"<b>Ratios: {sample.name}</b>",
            font=dict(size=PLOTLY_TITLE_FONT_SIZE),
        ),
        xaxis=dict(title="Measurement Cycle", showgrid=show_grid),
        yaxis=dict(title="Ratio", tickformat=".6f", showgrid=show_grid),
        height=plot_height,
        margin=dict(l=80, r=20, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            traceorder="normal",
            font=dict(size=PLOTLY_LEGEND_FONT_SIZE),
        ),
        hovermode="x unified",
        uirevision="constant",  # Persist zoom state
    )

    theme.apply_to_figure(fig, profile="timeseries")
    if not show_grid:
        fig.update_yaxes(showgrid=False)
    return fig


def create_single_ratio_plot(
    sample: Sample,
    ratio_name: str,
    show_raw: bool = True,
    show_corrected: bool = True,
    show_drift: bool = True,
    show_interference: bool = False,
    show_iif: bool = True,
    show_filtered_raw: bool = False,
    show_filtered_corrected: bool = False,
    show_outliers: bool = True,
    show_threshold: bool = True,
    threshold_basis: str = "processing_mask",
    show_stats_box: bool = True,
    stats_statistic: str = "2SD",
    certified_value: Optional[float] = None,
    show_certified: bool = True,
    iif_label: str = "IIF-corrected",
    height: int = 280,
    cycle_range: tuple = None,
    uirevision: Optional[str] = None,
    staged_cycles: Optional[List[int]] = None,
    show_pb_standard: bool = True,
    show_sr_standard: bool = True,
) -> go.Figure:
    """Create a plot for a single ratio.

    On the Pb-standard calibration route the Tl-normalized and the final
    calibrated layers are separate traces with separate toggles. Hiding a
    trace changes only the figure; the final value is chosen in the domain.
    """
    theme = get_theme()
    palette = theme.palette
    outline_color = palette.figure_axis
    plot_config = _get_plot_config()
    marker_size = plot_config.get("marker_size", 9)
    perf_options = get_plot_perf_options(plot_config)
    # I1: stable uirevision persists zoom/pan across Streamlit reruns while the
    # chart key is unchanged; key changes (Apply/Revert) recreate the chart anyway.
    _effective_uirevision = (
        uirevision
        if uirevision is not None
        else f"{sample.name}_{sample.run_number}:{ratio_name}"
    )

    fig = go.Figure()

    if ratio_name not in sample.ratios:
        fig.add_annotation(
            text=f"No data for {ratio_name}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(height=height)
        theme.apply_to_figure(fig, profile="timeseries")
        return fig

    cycle_data = sample.ratios[ratio_name]
    all_cycles = np.arange(1, len(cycle_data.values) + 1)

    if cycle_range:
        start_idx, end_idx = cycle_range[0] - 1, cycle_range[1]
        cycles = all_cycles[start_idx:end_idx]
    else:
        cycles = all_cycles

    raw_cycles, raw_values, raw_mask = _slice_runtime_cycle_view(
        sample.name,
        cycle_data,
        all_cycles,
        cycle_range,
    )
    raw_valid_cycles = raw_cycles[raw_mask]
    raw_valid_values = raw_values[raw_mask]
    iif_label = _normalise_layer_label(iif_label)
    blank_layer_active = _blank_correction_active(sample)
    raw_ratio = sample.ratios.get(ratio_name) if sample.ratios else None

    has_blank_source = (
        blank_layer_active
        and (
            (sample.blank_corrected_ratios and ratio_name in sample.blank_corrected_ratios)
            or (sample.corrected_ratios and ratio_name in sample.corrected_ratios)
        )
    )
    has_drift_source = bool(
        sample.drift_corrected_ratios and ratio_name in sample.drift_corrected_ratios
    )
    from domain.ratio_selection import INTERFERENCE_LAYER_LABEL, inspect_interference_ratio_data

    # The saved Hg interference-only ratio (ordinary SSB or Pb-Tl). Display only;
    # which layer is final is decided in the domain, not by this trace.
    hg_interference_data = inspect_interference_ratio_data(sample, ratio_name)
    has_interference_source = bool(
        hg_interference_data is not None
        or (sample.corrected_ratios and ratio_name in sample.corrected_ratios)
    )
    has_iif_source = bool(
        (sample.iif_corrected_ratios and ratio_name in sample.iif_corrected_ratios)
        or (sample.ssb_results and ratio_name in sample.ssb_results)
    )

    drift_data = None
    if sample.drift_corrected_ratios and ratio_name in sample.drift_corrected_ratios:
        drift_data = sample.drift_corrected_ratios[ratio_name]

    blank_corr_data = None
    blank_label = "Blank-corrected"
    if blank_layer_active and sample.blank_corrected_ratios and ratio_name in sample.blank_corrected_ratios:
        blank_corr_data = sample.blank_corrected_ratios[ratio_name]
    elif blank_layer_active and sample.corrected_ratios and ratio_name in sample.corrected_ratios:
        # Backward-compatible fallback when no pre-interference snapshot exists.
        blank_corr_data = sample.corrected_ratios[ratio_name]
    if show_raw and cycle_data_equal(blank_corr_data, raw_ratio):
        blank_corr_data = None

    interference_data = None
    interference_label = "Interference-corrected"
    if hg_interference_data is not None:
        interference_data = hg_interference_data
        interference_label = INTERFERENCE_LAYER_LABEL
    elif sample.corrected_ratios and ratio_name in sample.corrected_ratios:
        interference_data = sample.corrected_ratios[ratio_name]
    if show_raw and cycle_data_equal(interference_data, raw_ratio):
        interference_data = None

    canonical_exclusion_layer = None
    canonical_exclusion_group = None
    if blank_corr_data is not None and show_corrected:
        canonical_exclusion_layer = "blank_corrected"
        canonical_exclusion_group = "blank_corrected"
    elif blank_corr_data is None and show_raw:
        canonical_exclusion_layer = "raw"
        canonical_exclusion_group = "raw"

    if show_raw:
        raw_outline = resolve_marker_outline_width(
            n_points=len(raw_valid_cycles),
            requested_width=1.3,
            outline_mode=perf_options["outline_mode"],
            outline_max_points=perf_options["outline_max_points"],
        )
        skip_raw_hover = should_simplify_hover(
            n_points=len(raw_valid_cycles),
            dense_hover_simplify=perf_options["dense_hover_simplify"],
            dense_hover_max_points=perf_options["dense_hover_max_points"],
        )
        has_secondary_trace = (
            (show_drift and has_drift_source)
            or (show_corrected and has_blank_source)
            or (show_interference and has_interference_source)
            or (show_iif and has_iif_source)
        )
        if len(raw_valid_cycles) > 0:
            raw_kwargs = dict(
                x=raw_valid_cycles,
                y=raw_valid_values,
                mode="markers",
                name="Raw",
                legendgroup="raw",
                marker=dict(
                    color=palette.layer_raw,
                    size=max(marker_size - 1, 4),
                    symbol="circle-open",
                    # Raw remains prominent when correction layers are shown;
                    # distinguish it by the hollow symbol, not by fading data.
                    opacity=0.9,
                    line=dict(width=raw_outline, color=outline_color),
                ),
            )
            if skip_raw_hover and has_secondary_trace:
                raw_kwargs["hoverinfo"] = "skip"
            else:
                # I5: diagnostic hover — include per-point σ deviation from window mean
                if len(raw_valid_values) > 1:
                    _wm = float(np.nanmean(raw_valid_values))
                    _ws = float(np.nanstd(raw_valid_values, ddof=1))
                    _dev = (
                        (raw_valid_values - _wm) / _ws
                        if _ws > 0
                        else np.zeros_like(raw_valid_values)
                    )
                else:
                    _dev = np.zeros(len(raw_valid_values))
                raw_kwargs["customdata"] = np.column_stack([_dev])
                raw_kwargs["hovertemplate"] = (
                    f"<b>Raw</b><br>{format_name(ratio_name)}: %{{y:.6f}}<br>"
                    "Cycle: %{x}<br>"
                    "Dev: %{customdata[0]:+.2f}σ<extra></extra>"
                )
            fig.add_trace(
                go.Scatter(**raw_kwargs)
            )

        if show_outliers and canonical_exclusion_layer == "raw":
            _add_excluded_cycle_trace(
                fig,
                raw_cycles,
                raw_values,
                raw_mask,
                name="Excluded cycles (Processing mask)",
                color=palette.layer_excluded,
                outline_color=outline_color,
                legendgroup=canonical_exclusion_group,
            )

    show_drift_trace = show_drift and drift_data is not None
    if show_drift_trace:
        drift_cycles, drift_values, drift_mask = _slice_runtime_cycle_view(
            sample.name,
            drift_data,
            all_cycles,
            cycle_range,
        )
        valid_cycles = drift_cycles[drift_mask]
        valid_values = drift_values[drift_mask]
        if len(valid_values) > 0:
            drift_outline = resolve_marker_outline_width(
                n_points=len(valid_cycles),
                requested_width=1.6,
                outline_mode=perf_options["outline_mode"],
                outline_max_points=perf_options["outline_max_points"],
            )
            fig.add_trace(
                go.Scatter(
                    x=valid_cycles,
                    y=valid_values,
                    mode="markers+lines",
                    name="Drift-corrected",
                    legendgroup="drift_corrected",
                    marker=dict(color=palette.layer_drift, size=marker_size, symbol="square",
                                line=dict(width=drift_outline, color=outline_color)),
                    line=dict(color=palette.layer_drift, width=1.0, dash="dot"),
                )
            )
    if show_corrected and blank_corr_data is not None:
        corr_cycles, corr_values, corr_mask = _slice_runtime_cycle_view(
            sample.name,
            blank_corr_data,
            all_cycles,
            cycle_range,
        )
        valid_cycles = corr_cycles[corr_mask]
        valid_values = corr_values[corr_mask]
        if len(valid_values) > 0:
            corr_outline = resolve_marker_outline_width(
                n_points=len(valid_cycles),
                requested_width=1.6,
                outline_mode=perf_options["outline_mode"],
                outline_max_points=perf_options["outline_max_points"],
            )
            fig.add_trace(
                go.Scatter(
                    x=valid_cycles,
                    y=valid_values,
                    mode="markers+lines",
                    name=blank_label,
                    legendgroup="blank_corrected",
                    marker=dict(color=palette.layer_blank, size=marker_size, symbol="circle",
                                line=dict(width=corr_outline, color=outline_color)),
                    line=dict(color=palette.layer_blank, width=1.0, dash="dot"),
                )
            )
            if show_outliers and canonical_exclusion_layer == "blank_corrected":
                _add_excluded_cycle_trace(
                    fig,
                    corr_cycles,
                    corr_values,
                    corr_mask,
                    name="Excluded cycles (Processing mask)",
                    color=palette.layer_excluded,
                    outline_color=outline_color,
                    legendgroup=canonical_exclusion_group,
                )

    show_interference_trace = (
        show_interference
        and interference_data is not None
        and (
            not show_corrected
            or not cycle_data_equal(blank_corr_data, interference_data)
        )
    )
    if show_interference_trace:
        int_cycles, int_values, int_mask = _slice_runtime_cycle_view(
            sample.name,
            interference_data,
            all_cycles,
            cycle_range,
        )
        valid_cycles = int_cycles[int_mask]
        valid_values = int_values[int_mask]
        if len(valid_values) > 0:
            int_outline = resolve_marker_outline_width(
                n_points=len(valid_cycles),
                requested_width=1.6,
                outline_mode=perf_options["outline_mode"],
                outline_max_points=perf_options["outline_max_points"],
            )
            fig.add_trace(
                go.Scatter(
                    x=valid_cycles,
                    y=valid_values,
                    mode="markers+lines",
                    name=interference_label,
                    legendgroup="interference_corrected",
                    marker=dict(color=palette.layer_interference, size=marker_size, symbol="triangle-up",
                                line=dict(width=int_outline, color=outline_color)),
                    line=dict(color=palette.layer_interference, width=1.0, dash="dot"),
                )
            )
        if show_outliers and interference_data is hg_interference_data:
            _add_hg_invalid_cycle_markers(
                fig, sample, ratio_name, cycle_range=cycle_range, color=palette.layer_excluded,
            )
    iif_data = None
    if sample.iif_corrected_ratios and ratio_name in sample.iif_corrected_ratios:
        iif_data = sample.iif_corrected_ratios[ratio_name]
    elif sample.ssb_results and ratio_name in sample.ssb_results:
        iif_data = _build_ssb_cycle_data(sample, ratio_name)

    from domain.ratio_selection import governed_pb_standard_ratio_data, pb_standard_layer_label

    from domain.ratio_selection import SR_STANDARD_LAYER_LABEL
    sr_standard_data = sample.sr_standard_corrected_ratios.get(ratio_name)
    pb_standard_data = governed_pb_standard_ratio_data(sample, ratio_name)
    # With a calibrated final layer present, the Tl-only trace is an intermediate
    # and gives the final colour to the calibrated trace.
    iif_color = palette.layer_drift if pb_standard_data is not None or sr_standard_data is not None else palette.layer_final
    show_iif_trace = show_iif and iif_data is not None
    if show_iif_trace:
        iif_cycles, iif_values, iif_mask = _slice_runtime_cycle_view(
            sample.name,
            iif_data,
            all_cycles,
            cycle_range,
        )
        valid_cycles = iif_cycles[iif_mask]
        valid_values = iif_values[iif_mask]

        if len(valid_values) > 0:
            iif_outline = resolve_marker_outline_width(
                n_points=len(valid_cycles),
                requested_width=1.6,
                outline_mode=perf_options["outline_mode"],
                outline_max_points=perf_options["outline_max_points"],
            )
            fig.add_trace(
                go.Scatter(
                    x=valid_cycles,
                    y=valid_values,
                    mode="markers+lines",
                    name=iif_label,
                    legendgroup=iif_label.lower().replace("-", "_").replace(" ", "_"),
                    marker=dict(color=iif_color, size=marker_size, symbol="diamond",
                                line=dict(width=iif_outline, color=outline_color)),
                    line=dict(color=iif_color, width=1.0, dash="dot"),
                )
            )
    show_pb_standard_trace = show_pb_standard and pb_standard_data is not None
    if show_pb_standard_trace:
        pb_cycles, pb_values, pb_mask = _slice_runtime_cycle_view(
            sample.name, pb_standard_data, all_cycles, cycle_range,
        )
        if int(np.sum(pb_mask)) > 0:
            pb_label = pb_standard_layer_label(sample, ratio_name)
            fig.add_trace(
                go.Scatter(
                    x=pb_cycles[pb_mask],
                    y=pb_values[pb_mask],
                    mode="markers+lines",
                    name=pb_label,
                    legendgroup="pb_standard_corrected",
                    marker=dict(color=palette.layer_final, size=marker_size, symbol="star",
                                line=dict(width=1.2, color=outline_color)),
                    line=dict(color=palette.layer_final, width=1.2),
                )
            )
    show_sr_standard_trace = show_sr_standard and sr_standard_data is not None
    if show_sr_standard_trace:
        sr_cycles, sr_values, sr_mask = _slice_runtime_cycle_view(
            sample.name, sr_standard_data, all_cycles, cycle_range,
        )
        if int(np.sum(sr_mask)) > 0:
            fig.add_trace(go.Scatter(
                x=sr_cycles[sr_mask], y=sr_values[sr_mask], mode="markers+lines",
                name=SR_STANDARD_LAYER_LABEL, legendgroup="sr_standard_corrected",
                marker=dict(color=palette.layer_final, size=marker_size, symbol="star",
                            line=dict(width=1.2, color=outline_color)),
                line=dict(color=palette.layer_final, width=1.2),
            ))
    _, active_filter_method, _, _ = _get_active_filter_config()
    if show_threshold and active_filter_method != "None":
        guide_shape_start = len(fig.layout.shapes)
        guide_trace_start = len(fig.data)
        if threshold_basis == "iif_like" and iif_data is not None:
            guide_group = iif_label.lower().replace("-", "_").replace(" ", "_")
            _add_stats_threshold_overlay(
                fig,
                sample_name=sample.name,
                cycle_data=iif_data,
                cycle_range=cycle_range,
                line_color=palette.layer_final,
                band_color=palette.layer_final_light,
                show_annotation=False,
            )
        else:
            if canonical_exclusion_layer == "blank_corrected" and blank_corr_data is not None:
                guide_group = "blank_corrected"
                threshold_data = blank_corr_data
                threshold_line_color = palette.layer_blank
            else:
                guide_group = "raw"
                threshold_data = cycle_data
                threshold_line_color = palette.layer_raw

            _add_processing_threshold_overlay(
                fig,
                sample_name=sample.name,
                cycle_data=threshold_data,
                all_cycles=cycles,
                cycle_range=cycle_range,
                line_color=threshold_line_color,
                bounds_color=palette.guide_bounds,
            )

        # Shapes do not follow trace legend clicks. Render these guides as
        # grouped traces on a fixed overlay axis so they still span the viewport.
        owner = next((trace for trace in fig.data[:guide_trace_start]
                      if trace.legendgroup == guide_group), None)
        guide_visible = owner.visible if owner is not None else "legendonly"
        for trace in fig.data[guide_trace_start:]:
            trace.legendgroup = guide_group
            trace.visible = guide_visible
        for shape in fig.layout.shapes[guide_shape_start:]:
            is_band = shape.type == "rect"
            fig.add_trace(go.Scatter(
                x=[0, 1, 1, 0, 0] if is_band else [0, 1],
                y=[shape.y0, shape.y0, shape.y1, shape.y1, shape.y0]
                if is_band else [shape.y0, shape.y1],
                xaxis="x2", mode="lines", line=shape.line.to_plotly_json(),
                fill="toself" if is_band else None,
                fillcolor=shape.fillcolor if is_band else None,
                opacity=shape.opacity, hoverinfo="skip", showlegend=False,
                legendgroup=guide_group, visible=guide_visible,
            ))
        fig.layout.shapes = fig.layout.shapes[:guide_shape_start]
        fig.update_layout(xaxis2=dict(
            overlaying="x", range=[0, 1], visible=False, fixedrange=True,
        ))

    if show_pb_standard_trace:
        stats_data = pb_standard_data
    elif show_drift_trace and drift_data is not None:
        stats_data = drift_data
    elif show_sr_standard_trace:
        stats_data = sr_standard_data
    elif show_iif_trace and iif_data is not None:
        stats_data = iif_data
    elif show_interference_trace and interference_data is not None:
        stats_data = interference_data
    elif show_corrected and blank_corr_data is not None:
        stats_data = blank_corr_data
    else:
        stats_data = cycle_data
    stats_values = _get_runtime_filtered_values(
        sample.name, stats_data, cycle_range=cycle_range,
    )
    stats_statistic = "2SE" if stats_statistic == "2SE" else "2SD"
    if len(stats_values) > 1:
        stats_sd = float(np.nanstd(stats_values, ddof=1))
        stats_spread = (
            2 * stats_sd / np.sqrt(len(stats_values))
            if stats_statistic == "2SE"
            else 2 * stats_sd
        )
        ratio_dec = decimals_for_uncertainty(
            [stats_spread]
        )
    else:
        ratio_dec = 6

    stats_box_visible = show_stats_box and len(stats_values) > 1
    if stats_box_visible:
        _add_ratio_stats_annotation(
            fig,
            sample_name=sample.name,
            cycle_data=stats_data,
            cycle_range=cycle_range,
            decimals=ratio_dec,
            statistic=stats_statistic,
        )

    for trace in fig.data:
        if trace.hovertemplate:
            trace.hovertemplate = trace.hovertemplate.replace(".6f", f".{ratio_dec}f")

    x_max = int(cycles[-1]) if len(cycles) > 0 else 1

    # I4: Shade inactive cycle regions outside the selected window
    if cycle_range and len(all_cycles) > 0:
        total_cycles = int(all_cycles[-1])
        c_start, c_end = cycle_range
        _shade = palette.inactive_fill
        if c_start > 1:
            fig.add_vrect(x0=0.5, x1=c_start - 0.5, fillcolor=_shade, line_width=0, layer="below")
        if c_end < total_cycles:
            fig.add_vrect(x0=c_end + 0.5, x1=total_cycles + 0.5, fillcolor=_shade, line_width=0, layer="below")

    # I2: Staged-but-uncommitted exclusion overlay — hollow red X markers
    if staged_cycles:
        _view_start = cycle_range[0] - 1 if cycle_range else 0
        _view_end = cycle_range[1] if cycle_range else len(cycle_data.values)
        _stg_x: List[int] = []
        _stg_y: List[float] = []
        for _cyc in staged_cycles:
            if _view_start < _cyc <= _view_end:
                _idx = _cyc - 1
                if 0 <= _idx < len(cycle_data.values):
                    _stg_x.append(_cyc)
                    _stg_y.append(float(cycle_data.values[_idx]))
        if _stg_x:
            fig.add_trace(
                go.Scatter(
                    x=_stg_x,
                    y=_stg_y,
                    mode="markers",
                    marker=dict(
                        # Hollow staged markers must not cover the underlying data layer.
                        color="rgba(0,0,0,0)",
                        size=marker_size + 5,
                        symbol="x-open",
                        line=dict(color=palette.layer_excluded, width=2.5),
                    ),
                    name="Staged for exclusion",
                    hovertemplate="<b>Staged</b><br>Cycle: %{x}<br>Value: %{y:.6f}<extra></extra>",
                )
            )

    if certified_value is not None and show_certified:
        fig.add_hline(
            y=certified_value,
            line_dash="longdash",
            line_color=palette.figure_ink,
            line_width=2,
            annotation_text="CRM",
            annotation_position="top right",
            annotation_font_size=PLOTLY_ANNOTATION_FONT_SIZE,
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{format_name(ratio_name)}</b>",
            font=dict(size=PLOTLY_TITLE_FONT_SIZE),
            x=0,
            xanchor="left",
        ),
        xaxis=dict(
            title=dict(
                text="Measurement Cycle",
                font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
            ),
            showgrid=False,
            showline=True,
            mirror=True,
            zeroline=False,
            tickfont=dict(size=PLOTLY_BASE_FONT_SIZE),
            range=[0, x_max + 0.5],
        ),
        yaxis=dict(
            title=dict(
                text=format_name(ratio_name),
                font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
            ),
            tickformat=f".{ratio_dec}f",
            showgrid=False,
            showline=True,
            mirror=True,
            zeroline=False,
            tickfont=dict(size=PLOTLY_BASE_FONT_SIZE),
        ),
        height=height,
        margin=dict(l=120, r=20, t=135, b=85, autoexpand=False),
        showlegend=True,
        hovermode="x unified",
        uirevision=_effective_uirevision,
        font=dict(size=PLOTLY_BASE_FONT_SIZE),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=0,
            entrywidth=0.48,
            entrywidthmode="fraction",
            font=dict(size=PLOTLY_LEGEND_FONT_SIZE),
            traceorder="normal",
            groupclick="togglegroup",
        ),
    )

    theme.apply_to_figure(fig, profile="timeseries")
    if stats_box_visible:
        padded_y_range = _stats_box_padded_y_range(fig)
        if padded_y_range is not None:
            fig.update_yaxes(range=padded_y_range)
    return fig


def create_delta_cycle_plot(
    sample: Sample,
    ratio_name: str,
    show_threshold: bool = False,
    show_stats_box: bool = True,
    stats_statistic: str = "2SD",
    height: int = 280,
    delta_cycle_data: Optional[CycleData] = None,
    delta_summary: Optional[dict] = None,
    reference_name: Optional[str] = None,
    uirevision: Optional[str] = None,
) -> Optional[go.Figure]:
    """Create a cycle-by-cycle delta scatter plot (like intensity plots)."""
    delta_cd = delta_cycle_data or get_delta_cycle_data(sample, ratio_name)
    if delta_cd is None or delta_cd.n_valid == 0:
        return None

    theme = get_theme()
    palette = theme.palette
    outline_color = palette.figure_axis
    plot_config = _get_plot_config()
    perf_options = get_plot_perf_options(plot_config)
    # Delta plot is rendered at ~half width, so the wide-aspect clamp does not
    # apply: honour the requested height directly (taller avoids the long y-axis
    # title overlapping the chart title).
    _effective_uirevision = (
        uirevision
        if uirevision is not None
        else f"{sample.name}_{sample.run_number}:{ratio_name}:delta"
    )

    cycles = np.arange(1, len(delta_cd.values) + 1)
    valid_cycles = cycles[delta_cd.mask]
    valid_values = delta_cd.valid_values
    guide_x = [float(cycles[0]), float(cycles[-1])]
    mean_delta = (
        float(delta_summary["delta"])
        if delta_summary and "delta" in delta_summary
        else float(np.nanmean(valid_values))
    )
    sd_delta = (
        float(delta_summary["delta_sd"])
        if delta_summary and "delta_sd" in delta_summary
        else (float(np.nanstd(valid_values, ddof=1)) if len(valid_values) > 1 else 0.0)
    )
    delta_dec = decimals_for_uncertainty([2 * sd_delta], default=2)
    stats_statistic = "2SE" if stats_statistic == "2SE" else "2SD"
    stats_spread = (
        2 * sd_delta / np.sqrt(len(valid_values))
        if stats_statistic == "2SE"
        else 2 * sd_delta
    )
    stats_dec = decimals_for_uncertainty([stats_spread], default=2)

    marker_size = plot_config.get("marker_size", 9)

    fig = go.Figure()

    # Per-cycle delta scatter
    fig.add_trace(
        go.Scatter(
            x=valid_cycles,
            y=valid_values,
            mode="markers",
            marker=dict(
                color=palette.layer_final,
                size=marker_size,
                line=dict(
                    width=resolve_marker_outline_width(
                        n_points=len(valid_cycles),
                        requested_width=1.2,
                        outline_mode=perf_options["outline_mode"],
                        outline_max_points=perf_options["outline_max_points"],
                    ),
                    color=outline_color,
                ),
            ),
            name=f"{format_delta_html(ratio_name, reference='')} values",
            hovertemplate=(
                f"Cycle: %{{x}}<br>Delta: %{{y:.{delta_dec}f}} ‰<extra></extra>"
            ),
        )
    )

    # Mean line (no inline annotation \u2014 placed separately below)
    fig.add_trace(
        go.Scatter(
            x=guide_x,
            y=[mean_delta, mean_delta],
            mode="lines",
            name="Mean",
            line=dict(color=palette.layer_final, width=1.5, dash="dot"),
            hoverinfo="skip",
        )
    )

    # Stats annotation in lower-left of the data area.
    if show_stats_box:
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=0.02,
            y=0.005,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            text=(
                f"Mean \u00b1 {stats_statistic}: {mean_delta:.{stats_dec}f} "
                f"\u00b1 {stats_spread:.{stats_dec}f}\u2030 (n = {len(valid_values)})"
            ),
            font=dict(size=PLOTLY_ANNOTATION_FONT_SIZE, color=palette.annotation_text_color),
            bgcolor=palette.annotation_bg,
            bordercolor=palette.annotation_border,
            borderwidth=1,
        )

    # Threshold lines (mean +/- 2SD)
    if show_threshold and sd_delta > 0:
        fig.add_hline(
            y=mean_delta + 2 * sd_delta,
            line_dash="dash", line_color=palette.guide_bounds, line_width=1,
        )
        fig.add_hline(
            y=mean_delta - 2 * sd_delta,
            line_dash="dash", line_color=palette.guide_bounds, line_width=1,
        )
        add_legend_proxy(
            fig,
            name="Mean ± 2SD",
            line=dict(color=palette.guide_bounds, width=1, dash="dash"),
            legendgroup="delta_guides",
        )

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Measurement Cycle", font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE)),
            tickfont=dict(size=PLOTLY_BASE_FONT_SIZE),
            showgrid=False,
            showline=True,
            mirror=True,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text=format_delta_html(ratio_name, reference=reference_name),
                font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
                standoff=32,
            ),
            automargin=True,
            tickformat=f".{delta_dec}f",
            tickfont=dict(size=PLOTLY_BASE_FONT_SIZE),
            showgrid=False,
            showline=True,
            mirror=True,
            zeroline=False,
        ),
        height=height,
        margin=dict(l=115, r=20, t=70, b=40),
        showlegend=True,
        hovermode="x unified",
        uirevision=_effective_uirevision,
        font=dict(size=PLOTLY_BASE_FONT_SIZE),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=0,
            entrywidth=0.48,
            entrywidthmode="fraction",
            font=dict(size=PLOTLY_LEGEND_FONT_SIZE),
            traceorder="normal",
        ),
    )

    theme.apply_to_figure(fig, profile="timeseries")
    if show_stats_box:
        padded_y_range = _stats_box_padded_y_range(fig, lower_fraction=0.14)
        if padded_y_range is not None:
            fig.update_yaxes(range=padded_y_range)
    return fig

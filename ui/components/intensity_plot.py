"""Intensity plot component for TraceISO."""

from __future__ import annotations

from typing import Optional, List

import numpy as np
import plotly.graph_objects as go

from domain.models import Sample, CycleData
from domain.corrections.blank import KR_ISOTOPES
from domain.layers import cycle_data_equal
from domain.filters.outlier import get_runtime_mask
from config.constants import DEFAULT_OUTLIER_THRESHOLD_SD
from config.settings import DisplayConfig
from ui.config_plotly import (
    PLOTLY_ANNOTATION_FONT_SIZE,
    PLOTLY_TITLE_FONT_SIZE,
    WIDE_TIMESERIES_MAX_HEIGHT,
)
from ui.theme import get_theme
from ui.utils import blank_correction_active, format_name, get_plot_config, get_sample_state_key
from ui.state import get_state
from ui.components.plot_perf import (
    get_plot_perf_options,
    resolve_marker_outline_width,
    should_simplify_hover,
)


def _get_plot_config() -> dict:
    return get_plot_config(get_state())


def _blank_correction_active(sample: Sample) -> bool:
    return blank_correction_active(sample, get_state())


def _get_active_filter_config() -> tuple[str, float]:
    """Return the active processing filter configuration."""
    try:
        state = get_state()
        processing_config = getattr(state, "processing_config", None)
        if processing_config is None:
            return "None", DEFAULT_OUTLIER_THRESHOLD_SD
        return (
            processing_config.filter_method,
            processing_config.get_active_filter_threshold(),
        )
    except Exception:
        return "None", DEFAULT_OUTLIER_THRESHOLD_SD


def _slice_cycle_view(
    values: np.ndarray,
    mask: np.ndarray,
    cycle_range: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cycles, values, and mask restricted to the selected range."""
    all_cycles = np.arange(1, len(values) + 1)
    if cycle_range:
        start_idx, end_idx = cycle_range[0] - 1, cycle_range[1]
        return (
            all_cycles[start_idx:end_idx],
            values[start_idx:end_idx],
            mask[start_idx:end_idx],
        )
    return all_cycles, values, mask


def _wide_timeseries_height(requested_height: int) -> int:
    """Keep paired cycle plots at a consistent wide aspect ratio."""
    return min(requested_height, WIDE_TIMESERIES_MAX_HEIGHT)


def _get_hg_interference_intensity(sample: Sample, isotope: str) -> Optional[CycleData]:
    """The saved Hg interference-corrected intensity (204Pb), when present."""
    return (getattr(sample, "interference_corrected_intensities", None) or {}).get(isotope)


def _get_interference_corrected_intensity(
    sample: Sample,
    isotope: str,
) -> Optional[CycleData]:
    """Return the post-interference intensity layer when it is distinct."""
    hg_intensity = _get_hg_interference_intensity(sample, isotope)
    if hg_intensity is not None:
        return hg_intensity
    return _get_sr_interference_corrected_intensity(sample, isotope)


def _get_sr_interference_corrected_intensity(
    sample: Sample,
    isotope: str,
) -> Optional[CycleData]:
    """The Sr-chain corrected intensity when it differs from the blank-corrected snapshot."""
    if not sample.corrected_intensities or isotope not in sample.corrected_intensities:
        return None

    corrected = sample.corrected_intensities[isotope]
    blank_snapshot = None
    if sample.blank_corrected_intensities:
        blank_snapshot = sample.blank_corrected_intensities.get(isotope)

    if blank_snapshot is not None:
        return corrected if not cycle_data_equal(corrected, blank_snapshot) else None

    # Fallback for sessions without a preserved blank-corrected snapshot.
    if isotope not in {"84Sr", "86Sr", "87Sr"}:
        return None
    raw = sample.intensities.get(isotope) if sample.intensities else None
    return corrected if raw is None or not cycle_data_equal(corrected, raw) else None


def _get_blank_corrected_intensity(
    sample: Sample,
    isotope: str,
) -> Optional[CycleData]:
    """Return the blank-corrected intensity layer when available."""
    if not _blank_correction_active(sample):
        return None

    raw = sample.intensities.get(isotope) if sample.intensities else None
    if sample.blank_corrected_intensities and isotope in sample.blank_corrected_intensities:
        blank_corrected = sample.blank_corrected_intensities[isotope]
        if isotope in KR_ISOTOPES and cycle_data_equal(blank_corrected, raw):
            return None
        return blank_corrected

    if sample.corrected_intensities and isotope in sample.corrected_intensities:
        if _get_sr_interference_corrected_intensity(sample, isotope) is None:
            corrected = sample.corrected_intensities[isotope]
            if isotope in KR_ISOTOPES and cycle_data_equal(corrected, raw):
                return None
            return corrected
    return None


def _apply_selected_ratio_mask(
    sample: Sample,
    base_mask: np.ndarray,
    selected_ratio: Optional[str] = None,
    *,
    cycle_range: tuple | None = None,
) -> np.ndarray:
    """Return *base_mask* constrained by the selected ratio's accepted cycles."""
    mask = np.asarray(base_mask, dtype=bool).copy()
    if not selected_ratio:
        return mask

    from domain.ratio_selection import get_processing_ratio_data

    ratio_cd = get_processing_ratio_data(sample, selected_ratio)
    if ratio_cd is None:
        return mask

    filter_method, filter_threshold = _get_active_filter_config()
    sample_key = get_sample_state_key(sample)
    cycle_ranges = None
    if cycle_range is not None:
        cycle_ranges = {sample_key: cycle_range, sample.name: cycle_range}
    ratio_mask = get_runtime_mask(
        ratio_cd.values,
        ratio_cd.mask,
        sample.name,
        cycle_ranges=cycle_ranges,
        sample_key=sample_key,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
    )
    n = min(len(mask), len(ratio_mask))
    if n > 0:
        mask[:n] = mask[:n] & ratio_mask[:n]
    return mask


def _add_excluded_cycle_trace(
    fig: go.Figure,
    cycles: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    *,
    name: str,
    color: str,
    outline_color: str,
) -> None:
    """Show masked cycles explicitly so exclusions are visible in Inspector charts."""
    excluded_cycles = cycles[~mask]
    if len(excluded_cycles) == 0:
        return

    excluded_values = values[~mask]
    fig.add_trace(
        go.Scatter(
            x=excluded_cycles,
            y=excluded_values,
            mode="markers",
            name=name,
            showlegend=False,
            marker=dict(
                color=color,
                size=8,
                symbol="x-thin",
                opacity=0.95,
                line=dict(color=color, width=1.2),
            ),
            hovertemplate=(
                "<b>Excluded cycle</b><br>"
                "Cycle: %{x}<br>"
                "Value: %{y:.4e}<extra></extra>"
            ),
        )
    )


def create_intensity_plot(
    sample: Sample,
    isotopes: Optional[List[str]] = None,
    display_config: Optional[DisplayConfig] = None,
    height: int = 350,
) -> go.Figure:
    """Create an intensity plot for a sample."""
    theme = get_theme()
    palette = theme.palette
    outline_color = palette.figure_axis

    plot_config = _get_plot_config()
    marker_size = plot_config.get("marker_size", 9)
    plot_height = plot_config.get("height", height)
    show_grid = plot_config.get("show_grid", True)
    perf_options = get_plot_perf_options(plot_config)

    if display_config is None:
        display_config = DisplayConfig()

    fig = go.Figure()

    # Get isotopes to plot
    if isotopes is None:
        isotopes = list(sample.intensities.keys()) if sample.intensities else []

    if not isotopes:
        fig.add_annotation(
            text="No intensity data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        theme.apply_to_figure(fig, profile="timeseries")
        return fig

    for isotope in isotopes:
        blank_data = _get_blank_corrected_intensity(sample, isotope)
        interference_data = (
            _get_interference_corrected_intensity(sample, isotope)
            if display_config.show_interference_corrected
            else None
        )

        # Raw intensities
        if display_config.show_raw and isotope in sample.intensities:
            cycle_data = sample.intensities[isotope]
            cycles = np.arange(1, len(cycle_data.values) + 1)
            valid_cycles = cycles[cycle_data.mask]
            valid_values = cycle_data.values[cycle_data.mask]
            has_corrected_trace = (
                (display_config.show_corrected and blank_data is not None)
                or interference_data is not None
            )
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
                    name=f"{format_name(isotope)} (Raw)",
                    marker=dict(
                        color=palette.layer_raw,
                        size=marker_size,
                        symbol="circle" if raw_outline == 0 else "circle-open",
                        opacity=0.6 if display_config.show_corrected else 0.9,
                        line=dict(width=raw_outline, color=outline_color),
                    ),
                    legendgroup=isotope,
                    showlegend=True,
                )
                if skip_raw_hover:
                    raw_kwargs["hoverinfo"] = "skip"
                else:
                    raw_kwargs["hovertemplate"] = (
                        f"<b>{format_name(isotope)} (Raw)</b><br>"
                        "Cycle: %{x}<br>"
                        "V: %{y:.2e}<br>"
                        "<extra></extra>"
                    )

                fig.add_trace(go.Scattergl(**raw_kwargs))

            _add_excluded_cycle_trace(
                fig,
                cycles,
                cycle_data.values,
                cycle_data.mask,
                name="Excluded cycles",
                color=palette.layer_excluded,
                outline_color=outline_color,
            )

        # Corrected intensities
        if display_config.show_corrected and blank_data is not None:
            cycle_data = blank_data
            cycles = np.arange(1, len(cycle_data.values) + 1)

            # Filtered data (only valid cycles)
            valid_cycles = cycles[cycle_data.mask]
            valid_values = cycle_data.values[cycle_data.mask]
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
                    name=f"{format_name(isotope)} (Blank-corrected)",
                    marker=dict(
                        color=palette.layer_blank,
                        size=marker_size + 2,
                        symbol="circle",
                        line=dict(width=corr_outline, color=outline_color),
                    ),
                    line=dict(color=palette.layer_blank, width=1.5),
                    legendgroup=isotope,
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{format_name(isotope)} (Blank-corrected)</b><br>"
                        "Cycle: %{x}<br>"
                        "V: %{y:.2e}<br>"
                        "<extra></extra>"
                    ),
                    )
                )

            _add_excluded_cycle_trace(
                fig,
                cycles,
                cycle_data.values,
                cycle_data.mask,
                name="Excluded cycles",
                color=palette.layer_excluded,
                outline_color=outline_color,
            )

        if interference_data is not None:
            cycle_data = interference_data
            cycles = np.arange(1, len(cycle_data.values) + 1)
            valid_cycles = cycles[cycle_data.mask]
            valid_values = cycle_data.values[cycle_data.mask]
            interf_outline = resolve_marker_outline_width(
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
                    name=f"{format_name(isotope)} (Interference-corrected)",
                    marker=dict(
                        color=palette.layer_interference,
                        size=marker_size + 2,
                        symbol="triangle-up",
                        line=dict(width=interf_outline, color=outline_color),
                    ),
                    line=dict(color=palette.layer_interference, width=1.5, dash="dot"),
                    legendgroup=isotope,
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{format_name(isotope)} (Interference-corrected)</b><br>"
                        "Cycle: %{x}<br>"
                        "V: %{y:.2e}<br>"
                        "<extra></extra>"
                    ),
                )
            )

            _add_excluded_cycle_trace(
                fig,
                cycles,
                cycle_data.values,
                cycle_data.mask,
                name="Excluded cycles",
                color=palette.layer_excluded,
                outline_color=outline_color,
            )

    fig.update_layout(
        title=dict(text=f"<b>Intensities: {sample.name}</b>", font=dict(size=PLOTLY_TITLE_FONT_SIZE)),
        xaxis=dict(title="Measurement Cycle", showgrid=show_grid),
        yaxis=dict(title="Intensity (V)", showgrid=show_grid),
        height=plot_height,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            traceorder="normal",
        ),
        hovermode="x unified",
        uirevision="constant",
    )

    theme.apply_to_figure(fig, profile="timeseries")
    if not show_grid:
        fig.update_yaxes(showgrid=False)
    return fig


def create_single_isotope_plot(
    sample: Sample,
    isotope: str,
    show_raw: bool = True,
    show_corrected: bool = True,
    show_interference: bool = False,
    show_threshold: bool = True,
    show_outliers: bool = True,
    height: int = 250,
    cycle_range: tuple = None,
    selected_ratio: Optional[str] = None,
    uirevision: Optional[str] = None,
) -> go.Figure:
    """Create a plot for a single isotope."""
    theme = get_theme()
    palette = theme.palette
    outline_color = palette.figure_axis

    plot_config = _get_plot_config()
    marker_size = plot_config.get("marker_size", 9)
    plot_height = height
    perf_options = get_plot_perf_options(plot_config)
    # I1: stable uirevision persists zoom across reruns
    _effective_uirevision = (
        uirevision
        if uirevision is not None
        else f"{sample.name}_{sample.run_number}:{isotope}"
    )

    fig = go.Figure()

    if isotope not in sample.intensities:
        fig.add_annotation(
            text=f"No data for {isotope}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(height=plot_height)
        theme.apply_to_figure(fig, profile="timeseries")
        return fig

    cycle_data = sample.intensities[isotope]
    raw_mask = _apply_selected_ratio_mask(
        sample,
        cycle_data.mask,
        selected_ratio,
        cycle_range=cycle_range,
    )
    cycles, values, _raw_mask = _slice_cycle_view(
        cycle_data.values,
        raw_mask,
        cycle_range,
    )

    blank_data = _get_blank_corrected_intensity(sample, isotope)
    interference_data = _get_interference_corrected_intensity(sample, isotope) if show_interference else None

    primary_corrected = interference_data if interference_data is not None else blank_data
    show_only_single_trace = sample.is_blank and primary_corrected is not None
    render_raw_trace = show_raw and not show_only_single_trace

    if render_raw_trace:
        raw_valid_cycles = cycles[_raw_mask]
        raw_valid_values = values[_raw_mask]
        raw_outline = resolve_marker_outline_width(
            n_points=len(raw_valid_cycles),
            requested_width=1.4,
            outline_mode=perf_options["outline_mode"],
            outline_max_points=perf_options["outline_max_points"],
        )
        skip_raw_hover = (
            ((blank_data is not None and show_corrected) or interference_data is not None)
            and should_simplify_hover(
                n_points=len(raw_valid_cycles),
                dense_hover_simplify=perf_options["dense_hover_simplify"],
                dense_hover_max_points=perf_options["dense_hover_max_points"],
            )
        )
        if len(raw_valid_cycles) > 0:
            raw_kwargs = dict(
                x=raw_valid_cycles,
                y=raw_valid_values,
                mode="markers",
                name="Raw",
                marker=dict(
                    color=palette.layer_raw,
                    size=marker_size,
                    symbol="circle-open",
                    opacity=0.7,
                    line=dict(width=raw_outline, color=outline_color),
                ),
            )
            if skip_raw_hover:
                raw_kwargs["hoverinfo"] = "skip"
            else:
                raw_kwargs["hovertemplate"] = (
                    f"<b>Raw</b><br>{format_name(isotope)}: %{{y:.2e}}<br>"
                    "Cycle: %{x}<extra></extra>"
                )
            fig.add_trace(go.Scatter(**raw_kwargs))

        if show_outliers:
            _add_excluded_cycle_trace(
                fig,
                cycles,
                values,
                _raw_mask,
                name="Excluded cycles",
                color=palette.layer_excluded,
                outline_color=outline_color,
            )

    if show_corrected and blank_data is not None:
        blank_mask_full = _apply_selected_ratio_mask(
            sample,
            blank_data.mask,
            selected_ratio,
            cycle_range=cycle_range,
        )
        blank_cycles, blank_values, blank_mask = _slice_cycle_view(
            blank_data.values,
            blank_mask_full,
            cycle_range,
        )

        valid_cycles = blank_cycles[blank_mask]
        valid_values = blank_values[blank_mask]

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
                    name="Blank-corrected",
                    marker=dict(
                        color=palette.layer_blank,
                        size=marker_size + 1,
                        symbol="circle",
                        line=dict(width=corr_outline, color=outline_color),
                    ),
                    line=dict(color=palette.layer_blank, width=1.0, dash="dot"),
                    hovertemplate=(
                        f"<b>{format_name(isotope)} (Blank-corrected)</b><br>"
                        "Cycle: %{x}<br>"
                        "V: %{y:.2e}<extra></extra>"
                    ),
                )
            )

            if show_outliers:
                _add_excluded_cycle_trace(
                    fig,
                    blank_cycles,
                    blank_values,
                    blank_mask,
                    name="Excluded cycles",
                    color=palette.layer_excluded,
                    outline_color=outline_color,
                )

    if interference_data is not None:
        hg_trace = interference_data is _get_hg_interference_intensity(sample, isotope)
        interf_label = "Interference-corrected (Hg)" if hg_trace else "Interference-corrected"
        # The Hg-corrected intensity carries its own explicit channel support; it
        # must not change with whichever ratio the Inspector has selected.
        interf_mask_full = (
            np.asarray(interference_data.mask, dtype=bool).copy()
            if hg_trace
            else _apply_selected_ratio_mask(
                sample,
                interference_data.mask,
                selected_ratio,
                cycle_range=cycle_range,
            )
        )
        interf_cycles, interf_values, interf_mask = _slice_cycle_view(
            interference_data.values,
            interf_mask_full,
            cycle_range,
        )

        interf_valid_cycles = interf_cycles[interf_mask]
        interf_valid_values = interf_values[interf_mask]

        if len(interf_valid_values) > 0:
            interf_outline = resolve_marker_outline_width(
                n_points=len(interf_valid_cycles),
                requested_width=1.6,
                outline_mode=perf_options["outline_mode"],
                outline_max_points=perf_options["outline_max_points"],
            )
            fig.add_trace(
                go.Scatter(
                    x=interf_valid_cycles,
                    y=interf_valid_values,
                    mode="markers+lines",
                    name=interf_label,
                    marker=dict(
                        color=palette.layer_interference,
                        size=marker_size + 1,
                        symbol="triangle-up",
                        line=dict(width=interf_outline, color=outline_color),
                    ),
                    line=dict(color=palette.layer_interference, width=1.0, dash="dot"),
                    hovertemplate=(
                        f"<b>{format_name(isotope)} ({interf_label})</b><br>"
                        "Cycle: %{x}<br>"
                        "V: %{y:.2e}<extra></extra>"
                    ),
                )
            )

            if show_outliers:
                _add_excluded_cycle_trace(
                    fig,
                    interf_cycles,
                    interf_values,
                    interf_mask,
                    name="Excluded cycles",
                    color=palette.layer_excluded,
                    outline_color=outline_color,
                )

    x_max = int(cycles[-1]) if len(cycles) > 0 else 1

    # I4: shade the inactive cycle region outside the selected window
    if cycle_range is not None and len(cycles) > 0:
        _total_cyc = int(x_max)
        _c_start, _c_end = cycle_range
        _shade = palette.inactive_fill
        if _c_start > 1:
            fig.add_vrect(x0=0.5, x1=_c_start - 0.5, fillcolor=_shade, line_width=0, layer="below")
        if _c_end < _total_cyc:
            fig.add_vrect(x0=_c_end + 0.5, x1=_total_cyc + 0.5, fillcolor=_shade, line_width=0, layer="below")

    fig.update_layout(
        title=dict(text=f"<b>{format_name(isotope)}</b>", font=dict(size=PLOTLY_TITLE_FONT_SIZE)),
        xaxis=dict(
            title=dict(text="Measurement Cycle"),
            showgrid=False,
            zeroline=False,
            range=[0, x_max + 0.5],
        ),
        yaxis=dict(
            title=dict(text="Intensity (V)"),
            showgrid=False,
            zeroline=False,
        ),
        height=plot_height,
        margin=dict(l=120, r=20, t=135, b=85, autoexpand=False),
        showlegend=(
            render_raw_trace
            or (show_corrected and blank_data is not None)
            or (interference_data is not None)
        ),
        hovermode="x unified",
        uirevision=_effective_uirevision,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=0,
            entrywidth=0.48,
            entrywidthmode="fraction",
            font=dict(size=PLOTLY_ANNOTATION_FONT_SIZE),
            traceorder="normal",
        ),
    )

    theme.apply_to_figure(fig, profile="timeseries")
    return fig

"""Statistical charts component for TraceISO."""

from collections import Counter
from typing import List, Optional, Dict, Tuple

import numpy as np
import plotly.graph_objects as go

from config.constants import DEFAULT_OUTLIER_THRESHOLD_SD
from domain.models import Sample
from domain.filters.outlier import get_filtered_values, sample_cycle_key
from domain.ratio_selection import select_best_ratio_layer
from ui.config_plotly import (
    BALANCED_DISTRIBUTION_MIN_HEIGHT,
    PLOTLY_ANNOTATION_FONT_SIZE,
    PLOTLY_AXIS_TITLE_FONT_SIZE,
    PLOTLY_TITLE_FONT_SIZE,
)
from ui.theme import get_theme
from ui.components.plot_helpers import add_legend_proxy
from ui.utils import decimals_for_uncertainty, format_name, format_sample_display_label


class _SampleMetricsDict(dict):
    """Canonical sample-key mapping with unique-name read compatibility."""

    def __init__(self, *args, unique_name_keys: Optional[Dict[str, str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._unique_name_keys = unique_name_keys or {}

    def _resolve_key(self, key):
        if dict.__contains__(self, key):
            return key
        return self._unique_name_keys.get(key, key)

    def __getitem__(self, key):
        return dict.__getitem__(self, self._resolve_key(key))

    def get(self, key, default=None):
        return dict.get(self, self._resolve_key(key), default)


def _sample_display_labels(samples: List[Sample]) -> Dict[str, str]:
    """Keep category coordinates distinct even for identical display labels."""
    counts = Counter(sample.name for sample in samples)
    labels = [format_sample_display_label(s) if counts[s.name] > 1 else s.name for s in samples]
    repeated = Counter(labels)
    return {s.observation_id: (f"{label} [observation {i+1}]" if repeated[label] > 1 else label)
            for i, (s, label) in enumerate(zip(samples, labels))}


def _new_metrics(samples):
    aliases = {}
    candidates = {}
    for sample in samples:
        for alias in (sample.name, sample_cycle_key(sample)):
            candidates.setdefault(alias, set()).add(sample.observation_id)
    for alias, identities in candidates.items():
        if len(identities) == 1:
            aliases[alias] = next(iter(identities))
    return _SampleMetricsDict(unique_name_keys=aliases)


def _metric_for_sample(metrics: Dict, sample: Sample, samples):
    """Modern maps are authoritative; legacy aliases require unique ownership."""
    if isinstance(metrics, _SampleMetricsDict):
        return dict.get(metrics, sample.observation_id)
    if sample.observation_id in metrics:
        return metrics[sample.observation_id]
    if any('observation_id' in value for value in metrics.values()):
        return None
    for alias in (sample_cycle_key(sample), sample.name):
        matches = [s for s in samples if alias in (sample_cycle_key(s), s.name)]
        if len(matches) == 1 and alias in metrics:
            return metrics[alias]
    return None


def _sample_type_color(theme, sample_type: str) -> str:
    """Return the central colour for a sample type."""
    return theme.sample_color(sample_type)


def _sample_type_symbol(theme, sample_type: str) -> str:
    """Return the canonical marker symbol for a sample type."""
    return theme.sample_symbol(sample_type)


def _sample_type_band_color(theme, sample_type: str) -> str:
    """Return the central translucent band colour for a sample type."""
    stype = str(sample_type or "SMP").upper()
    palette = theme.palette
    if stype == "STD":
        return palette.sample_std_light
    if stype == "BLK":
        return palette.sample_blk_light
    if stype == "QC":
        return palette.sample_qc_light
    return palette.sample_smp_light


def _balanced_chart_height(requested_height: int) -> int:
    """Keep histogram and bar-style figures close to 4:3 proportions."""
    return max(requested_height, BALANCED_DISTRIBUTION_MIN_HEIGHT)


def _empty_annotated_figure(message: str, *, height: int) -> go.Figure:
    """Return a defensive empty-state figure for fully filtered views."""
    theme = get_theme()
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=PLOTLY_ANNOTATION_FONT_SIZE, color=theme.palette.annotation_text_color),
    )
    fig.update_layout(height=height)
    theme.apply_to_figure(fig, profile="overview")
    return fig


def _lookup_uncertainty_budget(uncertainty_overrides: Optional[Dict], sample: Sample, ratio_name: str):
    """Resolve a runtime uncertainty budget for one observation.

    Only the identity-keyed runtime map is consulted. A name-keyed fallback
    used to sit here, and it re-opened by display label exactly the aliasing
    the map key closes: two observations may share a name, and the second would
    have been drawn with the first one's error bar.
    """
    if not uncertainty_overrides:
        return None

    from domain.uncertainty.runtime import lookup_runtime_budget

    return lookup_runtime_budget(uncertainty_overrides, sample, ratio_name)


def _error_mode_metadata(error_mode: str) -> Tuple[str, str]:
    """Return metric key and short label for an error mode."""
    if error_mode == "2SD":
        return "error_2sd", "2 SD"
    if error_mode == "U_reported":
        return "error_expanded", "Reported U"
    return "error_2se", "2 SE"


def _active_error_values(type_data: Dict[str, Dict]) -> np.ndarray:
    """Flatten active error arrays for formatting and disclosure checks."""
    return np.asarray(
        [value for data in type_data.values() for value in data.get("errors", [])],
        dtype=float,
    )


def _certified_line_color(theme) -> str:
    """Use the publication figure ink for certified-reference lines."""
    return theme.palette.figure_ink


def _add_certified_overlay(
    fig: go.Figure,
    *,
    value: Optional[float],
    uncertainty: Optional[float],
    theme,
    coverage_factor_k: float = 2.0,
) -> None:
    """Add a manually supplied reference line and optional uncertainty band."""
    if value is None or not np.isfinite(value):
        return

    line_color = _certified_line_color(theme)
    if uncertainty is not None and np.isfinite(uncertainty) and uncertainty > 0:
        band_color = theme.palette.guide_bounds_light
        fig.add_shape(
            type="rect",
            xref="paper",
            x0=0,
            x1=1,
            yref="y",
            y0=value - uncertainty,
            y1=value + uncertainty,
            fillcolor=band_color,
            line_width=0,
            layer="below",
        )
        add_legend_proxy(
            fig,
            name=_certified_band_legend(coverage_factor_k),
            fillcolor=band_color,
            legendgroup="certified",
        )

    fig.add_hline(
        y=value,
        line_color=line_color,
        line_width=1.2,
    )
    add_legend_proxy(
        fig,
        name="Reference value",
        line=dict(color=line_color, width=1.2),
        legendgroup="certified",
    )


def _certified_band_legend(coverage_factor_k: float) -> str:
    """Describe the uncertainty quantity represented by a certified band."""
    try:
        k_value = float(coverage_factor_k)
    except (TypeError, ValueError, OverflowError):
        k_value = 2.0
    if not np.isfinite(k_value) or k_value <= 0:
        k_value = 2.0
    quantity = "u" if np.isclose(k_value, 1.0) else "U"
    return f"Reference value ± <i>{quantity}</i> (<i>k</i> = {k_value:g})"


def _add_population_bands(fig: go.Figure, type_data: Dict[str, Dict], theme) -> None:
    """Add mean +/- 2SD bands for each sample type."""
    for sample_type, data in type_data.items():
        if sample_type == "BLK":
            continue
        means = np.asarray(data.get("means", []), dtype=float)
        means = means[np.isfinite(means)]
        if len(means) < 2:
            continue
        mean = float(np.mean(means))
        sd = float(np.std(means, ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            continue
        fig.add_hrect(
            y0=mean - 2.0 * sd,
            y1=mean + 2.0 * sd,
            fillcolor=_sample_type_band_color(theme, sample_type),
            line_width=0,
            layer="below",
        )
        add_legend_proxy(
            fig,
            name=f"{sample_type} mean ± 2SD",
            fillcolor=_sample_type_band_color(theme, sample_type),
            legendgroup=f"{sample_type}_population",
        )


def _add_reported_uncertainty_bands(
    fig: go.Figure,
    type_data: Dict[str, Dict],
    theme,
    *,
    yaxis_for_type=None,
    x_axis_mode: str = "category",
    decimals: int = 6,
) -> None:
    """Render per-sample Reported-U as translucent vertical bands."""
    for sample_type, data in type_data.items():
        means = np.asarray(data.get("means", []), dtype=float)
        reported = np.asarray(data.get("reported_errors", []), dtype=float)
        if len(means) == 0 or len(reported) == 0:
            continue
        finite = np.isfinite(means) & np.isfinite(reported) & (reported > 0)
        if not np.any(finite):
            continue
        names = np.asarray(data.get("names", []), dtype=object)[finite]
        runs = np.asarray(data.get("runs", []), dtype=float)[finite]
        x_values = runs.tolist() if x_axis_mode == "run_order" else names.tolist()
        yaxis = yaxis_for_type(sample_type) if yaxis_for_type is not None else "y"
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=(2.0 * reported[finite]).tolist(),
                base=(means[finite] - reported[finite]).tolist(),
                marker=dict(
                    color=_sample_type_band_color(theme, sample_type),
                    # A transparent outline lets the proxy represent fill only.
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                opacity=0.85,
                width=0.58,
                name=f"{sample_type} Reported U",
                legendgroup=f"{sample_type}_reported_u",
                showlegend=True,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    + ("Run: %{x}<br>" if x_axis_mode == "run_order" else "")
                    + f"Reported U band: +/- %{{customdata[0]:.{decimals}f}}<extra></extra>"
                ),
                customdata=np.column_stack([reported[finite], names]),
                yaxis=yaxis,
            )
        )


def _get_valid_values(
    ratio_data,
    sample_name: str,
    sample_key: Optional[str] = None,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
):
    """Get valid values from ratio data, with cycle-range-aware re-filtering."""
    return get_filtered_values(
        ratio_data.values,
        ratio_data.mask,
        sample_name,
        cycle_ranges=cycle_ranges,
        sample_key=sample_key,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
    )


def build_ratio_metrics(
    samples: List[Sample],
    ratio_name: str,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
    uncertainty_overrides: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """Build per-sample ratio metrics in a single pass."""
    sorted_samples = sorted(
        [s for s in samples if not s.metadata.get("excluded", False)],
        key=lambda s: s.run_number,
    )
    display_labels = _sample_display_labels(sorted_samples)
    metrics: Dict[str, Dict] = _new_metrics(sorted_samples)

    for sample in sorted_samples:
        selected_layer = select_best_ratio_layer(sample, ratio_name)
        if selected_layer is None:
            continue
        ratio_data = selected_layer.data

        valid = _get_valid_values(
            ratio_data,
            sample.name,
            sample_key=sample_cycle_key(sample),
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
        )
        if len(valid) == 0:
            continue

        mean_val = float(np.nanmean(valid))
        std_val = float(np.nanstd(valid, ddof=1)) if len(valid) > 1 else np.nan
        se_val = std_val / np.sqrt(len(valid))
        rsd_pct = (std_val / mean_val * 100.0) if mean_val != 0 else np.nan
        budget = _lookup_uncertainty_budget(uncertainty_overrides, sample, ratio_name)
        # Reported-U for the ratio plot must be the absolute *ratio* uncertainty
        # regardless of the Uncertainty tab's output mode. budget.expanded_abs is
        # mode-dependent (it holds the delta uncertainty in delta mode), so derive
        # the ratio-absolute value from the mode-invariant relative figure scaled
        # by the reported ratio value. This reproduces expanded_abs exactly in
        # absolute_ratio mode (anchored or not) and across Engines A/C.
        rel_permil = float(getattr(budget, "expanded_rel_permil", np.nan)) if budget is not None else np.nan
        ratio_value = float(getattr(budget, "ratio_value", np.nan)) if budget is not None else np.nan
        expanded_error = (rel_permil / 1000.0) * ratio_value
        expanded_k = float(getattr(budget, "coverage_factor_k", np.nan)) if budget is not None else np.nan

        key = sample.observation_id
        metrics[key] = {
            "observation_id": sample.observation_id,
            "sample_name": sample.name,
            "sample_key": key,
            "display_label": display_labels[key],
            "sample_type": sample.sample_type.upper(),
            "run_number": sample.run_number,
            "layer_key": selected_layer.key,
            "layer_label": selected_layer.label,
            "mean": mean_val,
            "std": std_val,
            "se": se_val,
            "error_2se": 2 * se_val,
            "error_2sd": 2 * std_val,
            "error_expanded": expanded_error,
            "error_expanded_k": expanded_k,
            "rsd_pct": float(rsd_pct),
            "n": int(len(valid)),
        }

    return metrics


def build_delta_metrics(
    samples: List[Sample],
    ratio_name: str,
    delta_overrides: Optional[Dict] = None,
    uncertainty_overrides: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """Build per-sample delta (permil) metrics in a single pass.

    Mirrors :func:`build_ratio_metrics` but draws delta statistics from the
    runtime delta map (preferred) or the stored ``sample.delta_results``.
    The ``error_expanded`` metric is the full Engine B GUM expanded
    uncertainty of the delta value in permil (``expanded_abs``),
    taken from the runtime uncertainty budget. Only samples with a
    resolvable delta (typically SMP) are included.
    """
    from domain.runtime_delta import lookup_runtime_delta

    sorted_samples = sorted(
        [s for s in samples if not s.metadata.get("excluded", False)],
        key=lambda s: s.run_number,
    )
    display_labels = _sample_display_labels(sorted_samples)
    metrics: Dict[str, Dict] = _new_metrics(sorted_samples)

    for sample in sorted_samples:
        delta = delta_sd = delta_se = np.nan
        n = 0

        override = (
            lookup_runtime_delta(delta_overrides, sample, ratio_name)
            if delta_overrides
            else None
        )
        if override is not None:
            delta = float(override.delta)
            delta_sd = float(override.delta_sd)
            delta_se = float(override.delta_se)
            n = int(override.n)
        elif delta_overrides is None and sample.delta_results and ratio_name in sample.delta_results:
            d = sample.delta_results.get(ratio_name, {})
            if d.get("delta") is None:
                continue
            delta = float(d.get("delta", np.nan))
            delta_sd = float(d.get("delta_sd", np.nan))
            delta_se = float(d.get("delta_se", np.nan))
            n = int(d.get("n", 0) or 0)
        else:
            continue

        if not np.isfinite(delta):
            continue

        budget = _lookup_uncertainty_budget(uncertainty_overrides, sample, ratio_name)
        # ``expanded_abs`` is the expanded uncertainty of the reported delta
        # measurand. ``expanded_rel_permil`` is relative to the basis ratio and
        # omits the sample/reference scale factor when that factor is not one.
        expanded_permil = (
            float(getattr(budget, "expanded_abs", np.nan))
            if budget is not None else np.nan
        )
        expanded_k = (
            float(getattr(budget, "coverage_factor_k", np.nan))
            if budget is not None else np.nan
        )

        key = sample.observation_id
        metrics[key] = {
            "observation_id": sample.observation_id,
            "sample_name": sample.name,
            "sample_key": key,
            "display_label": display_labels[key],
            "sample_type": sample.sample_type.upper(),
            "run_number": sample.run_number,
            "mean": delta,
            "std": delta_sd,
            "se": delta_se,
            "error_2se": 2.0 * delta_se,
            "error_2sd": 2.0 * delta_sd,
            "error_expanded": expanded_permil,
            "error_expanded_k": expanded_k,
            "rsd_pct": np.nan,
            "n": n,
        }

    return metrics


def create_delta_overview_chart(
    samples: List[Sample],
    ratio_name: str,
    show_error_bars: bool = True,
    error_mode: str = "2SE",
    show_reported_uncertainty_band: bool = False,
    show_population_band: bool = False,
    show_zero_line: bool = True,
    certified_delta: Optional[float] = None,
    certified_delta_uncertainty: Optional[float] = None,
    cycle_ranges: Optional[Dict] = None,
    delta_overrides: Optional[Dict] = None,
    uncertainty_overrides: Optional[Dict] = None,
    delta_metrics: Optional[Dict[str, Dict]] = None,
    height: int = 400,
    reference_name: Optional[str] = None,
    x_axis_mode: str = "category",
    certified_coverage_factor: float = 2.0,
) -> go.Figure:
    """Create a cross-sample scatter plot of mean delta values (permil).

    A faithful mirror of :func:`create_ratio_overview_chart`, plotting
    delta instead of the raw ratio with the same sample-type colouring
    and error-bar handling.
    """
    from ui.utils import format_delta_html

    theme = get_theme()
    palette = theme.palette
    if x_axis_mode not in {"category", "run_order"}:
        raise ValueError("x_axis_mode must be 'category' or 'run_order'.")
    use_run_order = x_axis_mode == "run_order"

    if not samples:
        return _empty_annotated_figure("No samples to display.", height=height)

    fig = go.Figure()
    error_key, error_label = _error_mode_metadata(error_mode)

    sorted_samples = sorted(
        [s for s in samples if not s.metadata.get("excluded", False)],
        key=lambda s: s.run_number,
    )
    metrics = delta_metrics or build_delta_metrics(
        samples,
        ratio_name,
        delta_overrides=delta_overrides,
        uncertainty_overrides=uncertainty_overrides,
    )

    if not metrics:
        return _empty_annotated_figure(
            "No delta values available for the current selection.", height=height,
        )

    delta_axis_title = format_delta_html(ratio_name, reference=reference_name)

    type_data: Dict[str, Dict] = {}
    for sample in sorted_samples:
        metric = _metric_for_sample(metrics, sample, sorted_samples)
        if metric is None:
            continue
        st_type = metric["sample_type"]
        if st_type not in type_data:
            type_data[st_type] = {
                "names": [],
                "ids": [],
                "means": [],
                "errors": [],
                "reported_errors": [],
                "runs": [],
                "n": [],
                "k": [],
            }
        type_data[st_type]["ids"].append(sample.observation_id)
        type_data[st_type]["names"].append(metric.get("display_label", metric["sample_name"]))
        type_data[st_type]["means"].append(metric["mean"])
        type_data[st_type]["errors"].append(metric.get(error_key, np.nan))
        type_data[st_type]["reported_errors"].append(metric.get("error_expanded", np.nan))
        type_data[st_type]["runs"].append(metric["run_number"])
        type_data[st_type]["n"].append(metric["n"])
        type_data[st_type]["k"].append(metric.get("error_expanded_k", np.nan))

    category_order = [
        metric.get("display_label", metric["sample_name"])
        for s in sorted_samples
        if (metric := _metric_for_sample(metrics, s, sorted_samples)) is not None
    ]

    if not type_data:
        return _empty_annotated_figure(
            "No delta values available for the current selection.", height=height,
        )

    active_errors = _active_error_values(type_data)
    dec = decimals_for_uncertainty(active_errors, default=4)

    if show_population_band:
        _add_population_bands(fig, type_data, theme)

    if show_reported_uncertainty_band:
        _add_reported_uncertainty_bands(
            fig,
            type_data,
            theme,
            x_axis_mode=x_axis_mode,
            decimals=dec,
        )

    for sample_type, data in type_data.items():
        color = _sample_type_color(theme, sample_type)
        if use_run_order:
            customdata = np.column_stack(
                [data["names"], data["n"], data["errors"], data["k"]]
            )
            hover_heading = "<b>%{customdata[0]}</b><br>Run: %{x}<br>"
            n_index, error_index, k_index = 1, 2, 3
        else:
            customdata = np.column_stack(
                [data["runs"], data["n"], data["errors"], data["k"]]
            )
            hover_heading = "<b>%{x}</b><br>Run: %{customdata[0]}<br>"
            n_index, error_index, k_index = 1, 2, 3
        trace_kwargs = dict(
            x=data["runs"] if use_run_order else data["names"],
            y=data["means"],
            ids=data["ids"],
            mode="markers",
            marker=dict(
                color=color,
                size=9,
                symbol=_sample_type_symbol(theme, sample_type),
                line=dict(color=palette.figure_axis, width=0.8),
            ),
            name=sample_type,
            customdata=customdata,
            hovertemplate=(
                hover_heading
                + f"Delta: %{{y:.{dec}f}} ‰<br>"
                + f"n: %{{customdata[{n_index}]}}<br>"
                + f"{error_label}: +/- %{{customdata[{error_index}]:.{dec}f}} ‰"
                + (f" (k=%{{customdata[{k_index}]:.2f}})<br>" if error_mode == "U_reported" else "<br>")
                + f"Type: {sample_type}<extra></extra>"
            ),
        )
        if show_error_bars and data["errors"]:
            errors = np.asarray(data["errors"], dtype=float)
            if np.any(np.isfinite(errors)):
                trace_kwargs["error_y"] = dict(
                    type="data", array=errors, visible=True, color=color,
                )
        fig.add_trace(go.Scatter(**trace_kwargs))

    if show_zero_line:
        fig.add_hline(
            y=0.0,
            line_dash="dot",
            line_color=palette.figure_ink,
            line_width=1,
        )

    _add_certified_overlay(
        fig,
        value=certified_delta,
        uncertainty=certified_delta_uncertainty,
        theme=theme,
        coverage_factor_k=certified_coverage_factor,
    )

    fig.update_layout(
        xaxis=dict(
            title="Run order" if use_run_order else "",
            type="linear" if use_run_order else "category",
            categoryorder=None if use_run_order else "array",
            categoryarray=None if use_run_order else category_order,
        ),
        yaxis=dict(
            title=dict(text=delta_axis_title),
            tickformat=f".{dec}f",
            showgrid=False,
            zeroline=False,
        ),
        height=height,
        margin=dict(l=80, r=90, t=70, b=110),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
            traceorder="normal",
        ),
    )
    fig.update_xaxes(
        tickangle=0 if use_run_order else -45,
        showgrid=False,
        zeroline=False,
    )
    theme.apply_to_figure(fig, profile="overview")
    return fig


def create_ratio_overview_chart(
    samples: List[Sample],
    ratio_name: str,
    show_error_bars: bool = True,
    error_mode: str = "2SE",
    show_reported_uncertainty_band: bool = False,
    show_population_band: bool = False,
    show_certified: Optional[float] = None,
    certified_uncertainty: Optional[float] = None,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
    ratio_metrics: Optional[Dict[str, Dict]] = None,
    uncertainty_overrides: Optional[Dict] = None,
    height: int = 400,
    x_axis_mode: str = "category",
    certified_coverage_factor: float = 2.0,
) -> go.Figure:
    """Create a cross-sample scatter plot of ratio means."""
    theme = get_theme()
    palette = theme.palette
    if x_axis_mode not in {"category", "run_order"}:
        raise ValueError("x_axis_mode must be 'category' or 'run_order'.")
    use_run_order = x_axis_mode == "run_order"

    if not samples:
        return _empty_annotated_figure("No samples to display.", height=height)

    fig = go.Figure()
    error_key, error_label = _error_mode_metadata(error_mode)

    sorted_samples = sorted(
        [s for s in samples if not s.metadata.get("excluded", False)],
        key=lambda s: s.run_number,
    )
    metrics = ratio_metrics or build_ratio_metrics(
        samples,
        ratio_name,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
        uncertainty_overrides=uncertainty_overrides,
    )

    type_data = {}
    for sample in sorted_samples:
        metric = _metric_for_sample(metrics, sample, sorted_samples)
        if metric is None:
            continue

        st_type = metric["sample_type"]
        if st_type not in type_data:
            type_data[st_type] = {
                "names": [],
                "ids": [],
                "means": [],
                "errors": [],
                "reported_errors": [],
                "runs": [],
                "n": [],
                "k": [],
            }

        type_data[st_type]["ids"].append(sample.observation_id)
        type_data[st_type]["names"].append(metric.get("display_label", metric["sample_name"]))
        type_data[st_type]["means"].append(metric["mean"])
        type_data[st_type]["errors"].append(metric.get(error_key, np.nan))
        type_data[st_type]["reported_errors"].append(metric.get("error_expanded", np.nan))
        type_data[st_type]["runs"].append(metric["run_number"])
        type_data[st_type]["n"].append(metric["n"])
        type_data[st_type]["k"].append(metric.get("error_expanded_k", np.nan))

    category_order = [
        metric.get("display_label", metric["sample_name"])
        for s in sorted_samples
        if (metric := _metric_for_sample(metrics, s, sorted_samples)) is not None
    ]

    if not type_data:
        return _empty_annotated_figure("No samples to display.", height=height)

    # R5: use WebGL (Scattergl) when the number of traces points is large
    _total_points = sum(len(d["names"]) for d in type_data.values())
    _scatter_cls = go.Scattergl if _total_points > 50 else go.Scatter

    has_blank_axis = "BLK" in type_data
    active_errors = _active_error_values(type_data)
    dec = decimals_for_uncertainty(active_errors)

    if show_population_band:
        _add_population_bands(fig, type_data, theme)

    if show_reported_uncertainty_band:
        _add_reported_uncertainty_bands(
            fig,
            type_data,
            theme,
            yaxis_for_type=lambda stype: "y2" if stype == "BLK" and has_blank_axis else "y",
            x_axis_mode=x_axis_mode,
            decimals=dec,
        )

    for sample_type, data in type_data.items():
        color = _sample_type_color(theme, sample_type)

        if use_run_order:
            customdata = np.column_stack(
                [data["names"], data["n"], data["errors"], data["k"]]
            )
            hover_heading = "<b>%{customdata[0]}</b><br>Run: %{x}<br>"
            n_index, error_index, k_index = 1, 2, 3
        else:
            customdata = np.column_stack(
                [data["runs"], data["n"], data["errors"], data["k"]]
            )
            hover_heading = "<b>%{x}</b><br>Run: %{customdata[0]}<br>"
            n_index, error_index, k_index = 1, 2, 3
        trace_kwargs = dict(
            x=data["runs"] if use_run_order else data["names"],
            y=data["means"],
            ids=data["ids"],
            mode="markers",
            marker=dict(
                color=color,
                size=9,
                symbol=_sample_type_symbol(theme, sample_type),
                line=dict(color=palette.figure_axis, width=0.8),
            ),
            name=sample_type,
            yaxis="y2" if sample_type == "BLK" and has_blank_axis else "y",
            customdata=customdata,
            hovertemplate=(
                hover_heading
                + f"Mean: %{{y:.{dec}f}}<br>"
                + f"n: %{{customdata[{n_index}]}}<br>"
                + f"{error_label}: +/- %{{customdata[{error_index}]:.{dec}f}}"
                + (f" (k=%{{customdata[{k_index}]:.2f}})<br>" if error_mode == "U_reported" else "<br>")
                + f"Type: {sample_type}<extra></extra>"
            ),
        )

        if show_error_bars and data["errors"]:
            errors = np.asarray(data["errors"], dtype=float)
            if not np.any(np.isfinite(errors)):
                fig.add_trace(_scatter_cls(**trace_kwargs))
                continue
            trace_kwargs["error_y"] = dict(
                type="data", array=errors, visible=True, color=color,
            )

        fig.add_trace(_scatter_cls(**trace_kwargs))

    _add_certified_overlay(
        fig,
        value=show_certified,
        uncertainty=certified_uncertainty,
        theme=theme,
        coverage_factor_k=certified_coverage_factor,
    )

    fig.update_layout(
        title=dict(text=format_name(ratio_name), font=dict(size=PLOTLY_TITLE_FONT_SIZE)),
        xaxis=dict(
            title="Run order" if use_run_order else "",
            type="linear" if use_run_order else "category",
            categoryorder=None if use_run_order else "array",
            categoryarray=None if use_run_order else category_order,
        ),
        yaxis=dict(
            title=dict(text=format_name(ratio_name)),
            tickformat=f".{dec}f",
            showgrid=False,
            zeroline=False,
        ),
        yaxis2=dict(
            title=dict(
                text=f"{format_name(ratio_name)} — blanks",
                font=dict(color=theme.sample_color("BLK")),
            ),
            tickformat=f".{dec}f",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            visible=has_blank_axis,
            tickfont=dict(color=theme.sample_color("BLK")),
            linecolor=theme.sample_color("BLK"),
        ),
        height=height,
        margin=dict(l=80, r=90, t=70, b=110),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
            traceorder="normal",
        ),
    )
    fig.update_xaxes(
        tickangle=0 if use_run_order else -45,
        showgrid=False,
        zeroline=False,
    )
    theme.apply_to_figure(fig, profile="overview")
    if has_blank_axis:
        blank_axis_color = theme.sample_color("BLK")
        fig.update_layout(
            yaxis2_tickfont_color=blank_axis_color,
            yaxis2_title_font_color=blank_axis_color,
            yaxis2_linecolor=blank_axis_color,
        )
    return fig


def create_rsd_comparison_chart(
    samples: List[Sample],
    ratio_name: str,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
    ratio_metrics: Optional[Dict[str, Dict]] = None,
    height: int = 350,
) -> go.Figure:
    """Create a bar chart comparing RSD across samples."""
    theme = get_theme()
    palette = theme.palette
    fig = go.Figure()
    height = _balanced_chart_height(height)

    metrics = ratio_metrics or build_ratio_metrics(
        samples,
        ratio_name,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
    )
    sorted_samples = sorted(
        [s for s in samples if not s.metadata.get("excluded", False)],
        key=lambda s: s.run_number,
    )

    names = []
    rsds = []
    colors = []
    sample_types = []
    for sample in sorted_samples:
        metric = _metric_for_sample(metrics, sample, sorted_samples)
        if metric is None:
            continue
        if metric["n"] < 2 or not np.isfinite(metric["rsd_pct"]):
            continue
        names.append(metric.get("display_label", sample.name))
        rsds.append(metric["rsd_pct"])
        sample_types.append(metric["sample_type"])
        colors.append(_sample_type_color(theme, metric["sample_type"]))

    if not names:
        return _empty_annotated_figure("No samples with sufficient data to calculate RSD.", height=height)

    has_blank_axis = "BLK" in sample_types
    std_names = [name for name, stype in zip(names, sample_types) if stype == "STD"]
    std_rsds = [rsd for rsd, stype in zip(rsds, sample_types) if stype == "STD"]
    smp_names = [name for name, stype in zip(names, sample_types) if stype == "SMP"]
    smp_rsds = [rsd for rsd, stype in zip(rsds, sample_types) if stype == "SMP"]
    blank_names = [name for name, stype in zip(names, sample_types) if stype == "BLK"]
    blank_rsds = [rsd for rsd, stype in zip(rsds, sample_types) if stype == "BLK"]
    blank_colors = [color for color, stype in zip(colors, sample_types) if stype == "BLK"]
    blank_axis_color = theme.sample_color("BLK")

    if std_names:
        fig.add_trace(go.Bar(
            x=std_names,
            y=std_rsds,
            base=0,
            marker_color=_sample_type_color(theme, "STD"),
            width=0.32,
            yaxis="y",
            hovertemplate="<b>%{x}</b><br>RSD: %{y:.4f}%<extra></extra>",
            showlegend=True,
            name="STD",
        ))
    if smp_names:
        fig.add_trace(go.Bar(
            x=smp_names,
            y=smp_rsds,
            base=0,
            marker_color=_sample_type_color(theme, "SMP"),
            width=0.32,
            yaxis="y",
            hovertemplate="<b>%{x}</b><br>RSD: %{y:.4f}%<extra></extra>",
            showlegend=True,
            name="SMP",
        ))
    if blank_names:
        fig.add_trace(go.Bar(
            x=blank_names,
            y=blank_rsds,
            base=0,
            marker_color=blank_colors,
            width=0.32,
            yaxis="y2" if has_blank_axis else "y",
            hovertemplate="<b>%{x}</b><br>RSD: %{y:.4f}%<extra></extra>",
            showlegend=True,
            name="BLK",
        ))

    # Median line
    median_rsd = np.median(rsds)
    fig.add_hline(
        y=median_rsd,
        line_dash="dash",
        line_color=palette.guide_bounds,
        line_width=1,
    )
    fig.add_annotation(
        text=f"Median: {median_rsd:.4f}%",
        xref="x domain", yref="y domain",
        x=1, y=1,
        xanchor="right", yanchor="top",
        showarrow=False,
        font=dict(size=PLOTLY_ANNOTATION_FONT_SIZE),
    )

    primary_max = max(std_rsds + smp_rsds) if (std_rsds or smp_rsds) else 0.0
    blank_max = max(blank_rsds) if blank_rsds else 0.0
    primary_upper = primary_max * 1.1 if primary_max > 0 else 1.0
    blank_upper = blank_max * 1.1 if blank_max > 0 else 1.0

    fig.update_layout(
        title=dict(text=format_name(ratio_name), font=dict(size=PLOTLY_TITLE_FONT_SIZE)),
        xaxis_title="",
        xaxis=dict(
            categoryorder="array",
            categoryarray=names,
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
            showline=False,
            # The unused secondary axis stays structurally present but invisible.
            linecolor="rgba(0,0,0,0)",
            title_font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
        ),
        yaxis=dict(
            title=dict(text="RSD (%)", font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE)),
            range=[0, primary_upper],
            rangemode="nonnegative",
            showgrid=False,
            showline=False,
            zeroline=False,
            tickfont=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
        ),
        yaxis2=dict(
            title=dict(
                text="Blank RSD (%)",
                font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE, color=blank_axis_color),
            ),
            overlaying="y",
            side="right",
            range=[0, blank_upper],
            rangemode="nonnegative",
            showgrid=False,
            zeroline=True,
            zerolinewidth=1.2,
            tickfont=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE, color=blank_axis_color),
            linecolor=blank_axis_color,
            visible=has_blank_axis,
        ),
        height=height,
        margin=dict(l=70, r=85, t=70, b=95),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
        ),
        bargap=0.22,
    )
    fig.update_xaxes(
        tickangle=-45,
        showgrid=False,
        zeroline=False,
        showline=False,
        tickfont=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
        # The unused secondary axis stays structurally present but invisible.
        linecolor="rgba(0,0,0,0)",
        title_font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
    )

    # Plot-area border: a single rect shape guarantees all four corners meet
    # perfectly, unlike mixing axis showline + hline + shape approaches.
    fig.add_shape(
        type="rect",
        xref="x domain", yref="y domain",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color=palette.figure_axis, width=1.2),
        layer="above",
    )
    theme.apply_to_figure(fig, profile="overview")
    if has_blank_axis:
        fig.update_layout(
            yaxis2_tickfont_color=blank_axis_color,
            yaxis2_title_font_color=blank_axis_color,
            yaxis2_linecolor=blank_axis_color,
        )
    return fig

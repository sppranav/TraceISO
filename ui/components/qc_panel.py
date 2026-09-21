"""QC panel component for TraceISO."""

from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
import streamlit as st

from config.constants import DEFAULT_OUTLIER_THRESHOLD_SD
from domain.filters.outlier import get_filtered_values
from domain.models import Sample, ProcessingResult
from domain.ratio_selection import get_best_ratio_data, select_best_ratio_layer
from ui.config_plotly import (
    PLOTLY_ANNOTATION_FONT_SIZE,
    PLOTLY_AXIS_TITLE_FONT_SIZE,
    PLOTLY_BASE_FONT_SIZE,
)
from ui.formatting import format_value_with_uncertainty
from ui.utils import get_sample_state_key


@dataclass
class QCMetric:
    """A single QC metric with pass/fail status."""
    name: str
    value: Optional[float]
    unit: str
    threshold: Optional[float]
    passed: Optional[bool]
    description: str = ""
    status: str = "assessed"


@dataclass
class QCResult:
    """Collection of QC metrics for a processing result."""
    metrics: List[QCMetric]
    warnings: List[str]
    overall_pass: bool


def _get_runtime_ratio_values(
    sample: Sample,
    ratio_name: str,
    *,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
) -> Optional[np.ndarray]:
    """Return runtime-filtered ratio values for QC calculations."""
    ratio_data = get_best_ratio_data(sample, ratio_name)
    if ratio_data is None:
        return None

    return get_filtered_values(
        ratio_data.values,
        ratio_data.mask,
        sample.name,
        cycle_ranges=cycle_ranges,
        sample_key=get_sample_state_key(sample),
        filter_method=filter_method,
        filter_threshold=filter_threshold,
    )


def _calculate_std_reproducibility(
    samples: List[Sample],
    ratio_name: str,
    *,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
) -> Dict[str, float]:
    """Calculate runtime-aware reproducibility statistics for standards."""
    std_samples = [s for s in samples if s.is_standard]
    if not std_samples:
        return {}

    means = []
    for sample in std_samples:
        valid_values = _get_runtime_ratio_values(
            sample,
            ratio_name,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
        )
        if valid_values is None or len(valid_values) == 0:
            continue
        means.append(float(np.nanmean(valid_values)))

    if not means:
        return {}

    mean_val = float(np.mean(means))
    std_val = float(np.std(means, ddof=1)) if len(means) > 1 else float("nan")
    rsd_pct = (std_val / mean_val * 100.0) if (len(means) > 1 and mean_val != 0) else float("nan")

    return {
        "mean": mean_val,
        "std": std_val,
        "rsd_pct": rsd_pct,
        "n": len(means),
    }


def calculate_qc_metrics(
    result: ProcessingResult,
    ratio_name: Optional[str] = None,
    *,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
    std_repro_threshold: float = 0.1,  # %
    blank_contribution_threshold: Optional[float] = None,  # signed ratio-change criterion, %
    drift_threshold: float = 0.1,  # %
) -> QCResult:
    """Calculate QC metrics for a processing result."""
    samples = [s for s in result.samples if not s.metadata.get("excluded", False)]
    metrics = []
    warnings = []

    if ratio_name is None:
        for s in samples:
            if s.ratios:
                ratio_name = list(s.ratios.keys())[0]
                break

    if ratio_name is None:
        return QCResult(metrics=[], warnings=["No ratio data available"], overall_pass=False)


    std_repro = _calculate_std_reproducibility(
        samples,
        ratio_name,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
    )
    if std_repro:
        rsd = std_repro["rsd_pct"]
        if not np.isfinite(rsd):
            warnings.append(
                f"Standard reproducibility cannot be calculated: only {std_repro['n']} standard(s) available."
            )
        else:
            passed = rsd <= std_repro_threshold
            metrics.append(QCMetric(
                name="Standard Reproducibility",
                value=rsd,
                unit="%",
                threshold=std_repro_threshold,
                passed=passed,
                description=f"External reproducibility of {std_repro['n']} standards",
            ))
            if not passed:
                warnings.append(f"Standard reproducibility ({rsd:.4f}%) exceeds threshold ({std_repro_threshold}%)")


    from domain.qc_blank import blank_qc_diagnostics

    blank_diagnostics = blank_qc_diagnostics(
        samples,
        ratio_name,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
    )
    for diagnostic in blank_diagnostics:
        label = f"{diagnostic.sample_name} (run {diagnostic.run_number})"
        if diagnostic.quantity == "blank_load":
            name = f"Blank load {diagnostic.isotope} · {label}"
            description = (
                "100 × applied weighted blank estimate / measured pre-blank signal "
                "on the accepted ratio support"
            )
            criterion = None
        else:
            name = f"Signed blank ratio change · {label}"
            description = (
                "100 × (ratio before blank subtraction − ratio after blank subtraction) "
                "/ ratio after, using ratio-of-means on common accepted support"
            )
            criterion = blank_contribution_threshold
        assessed = diagnostic.value_percent is not None and criterion is not None
        passed = (
            abs(float(diagnostic.value_percent)) <= float(criterion)
            if assessed else None
        )
        metrics.append(QCMetric(
            name=name,
            value=diagnostic.value_percent,
            unit="%",
            threshold=criterion,
            passed=passed,
            description=description + (f". {diagnostic.reason}" if diagnostic.reason else ""),
            status=(
                "unavailable" if diagnostic.status == "unavailable"
                else "assessed" if assessed else "not_assessed"
            ),
        ))
        if passed is False:
            warnings.append(
                f"{name} magnitude ({abs(float(diagnostic.value_percent)):.4f}%) "
                f"exceeds the configured ratio-change criterion ({criterion}%)."
            )
        elif diagnostic.status == "unavailable":
            warnings.append(f"{name} unavailable: {diagnostic.reason}")


    drift, drift_layer_label, drift_layer_warning = _calculate_drift(
        samples,
        ratio_name,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
    )
    if drift_layer_warning:
        warnings.append(drift_layer_warning)
    if drift is not None:
        passed = abs(drift) <= drift_threshold
        metrics.append(QCMetric(
            name="Session Drift",
            value=drift,
            unit="%",
            threshold=drift_threshold,
            passed=passed,
            description=(
                "Difference between first and last standard "
                f"({drift_layer_label} layer)"
            ),
        ))
        if not passed:
            warnings.append(f"Session drift ({drift:.4f}%) exceeds threshold ({drift_threshold}%)")

    max_std_deviation = _calculate_max_standard_deviation(
        samples,
        ratio_name,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
    )
    if max_std_deviation is not None:
        passed = max_std_deviation <= drift_threshold
        metrics.append(QCMetric(
            name="Max Standard Deviation",
            value=max_std_deviation,
            unit="%",
            threshold=drift_threshold,
            passed=passed,
            description="Maximum absolute standard deviation from the session standard mean",
        ))
        if not passed:
            warnings.append(
                f"Maximum standard deviation from session mean ({max_std_deviation:.4f}%) exceeds threshold ({drift_threshold}%)"
            )


    outlier_count = _count_sample_outliers(
        samples,
        ratio_name,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
    )
    if outlier_count > 0:
        warnings.append(f"{outlier_count} sample(s) have RSD > 3x median RSD")

    # Informational warnings should not make an otherwise passing metric set fail.
    assessed_metrics = [m for m in metrics if m.passed is not None]
    overall_pass = bool(assessed_metrics) and all(bool(m.passed) for m in assessed_metrics)

    return QCResult(metrics=metrics, warnings=warnings, overall_pass=overall_pass)


def _calculate_blank_contribution(
    samples: List[Sample],
    ratio_name: str,
    *,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
) -> Optional[float]:
    """Calculate blank contribution as relative bias on the ratio (%)."""
    blanks = [s for s in samples if s.is_blank]
    non_blanks = [s for s in samples if not s.is_blank]

    if not blanks or not non_blanks:
        return None

    parts = ratio_name.split("/") if "/" in ratio_name else None
    if parts is None or len(parts) != 2:
        return None
    num_iso, den_iso = parts[0], parts[1]

    # Collect mean blank intensities for numerator and denominator
    blank_num_vals, blank_den_vals = [], []
    for sample in blanks:
        if not sample.intensities:
            continue
        if num_iso in sample.intensities and den_iso in sample.intensities:
            nv = get_filtered_values(
                sample.intensities[num_iso].values,
                sample.intensities[num_iso].mask,
                sample.name,
                cycle_ranges=cycle_ranges,
                sample_key=get_sample_state_key(sample),
                filter_method=filter_method,
                filter_threshold=filter_threshold,
            )
            dv = get_filtered_values(
                sample.intensities[den_iso].values,
                sample.intensities[den_iso].mask,
                sample.name,
                cycle_ranges=cycle_ranges,
                sample_key=get_sample_state_key(sample),
                filter_method=filter_method,
                filter_threshold=filter_threshold,
            )
            if len(nv) > 0 and len(dv) > 0:
                blank_num_vals.append(np.nanmean(nv))
                blank_den_vals.append(np.nanmean(dv))

    if not blank_num_vals:
        return None

    avg_blank_num = float(np.mean(blank_num_vals))
    avg_blank_den = float(np.mean(blank_den_vals))

    # Collect mean sample intensities for numerator and denominator
    sample_ratios_with = []
    sample_ratios_without = []
    for sample in non_blanks:
        if not sample.intensities:
            continue
        if num_iso in sample.intensities and den_iso in sample.intensities:
            sn = get_filtered_values(
                sample.intensities[num_iso].values,
                sample.intensities[num_iso].mask,
                sample.name,
                cycle_ranges=cycle_ranges,
                sample_key=get_sample_state_key(sample),
                filter_method=filter_method,
                filter_threshold=filter_threshold,
            )
            sd = get_filtered_values(
                sample.intensities[den_iso].values,
                sample.intensities[den_iso].mask,
                sample.name,
                cycle_ranges=cycle_ranges,
                sample_key=get_sample_state_key(sample),
                filter_method=filter_method,
                filter_threshold=filter_threshold,
            )
            if len(sn) > 0 and len(sd) > 0:
                mean_num = np.nanmean(sn)
                mean_den = np.nanmean(sd)
                if mean_den == 0 or (mean_den - avg_blank_den) == 0:
                    continue
                ratio_without = mean_num / mean_den
                ratio_with = (mean_num - avg_blank_num) / (mean_den - avg_blank_den)
                sample_ratios_with.append(ratio_with)
                sample_ratios_without.append(ratio_without)

    if not sample_ratios_without:
        return None

    avg_with = float(np.mean(sample_ratios_with))
    avg_without = float(np.mean(sample_ratios_without))

    if avg_without == 0:
        return None

    return abs(avg_with - avg_without) / abs(avg_without) * 100


def _calculate_drift(
    samples: List[Sample],
    ratio_name: str,
    *,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
) -> tuple[Optional[float], str, Optional[str]]:
    """Calculate drift between first and last standard of the primary bracketing RM.

    Groups standards by name and uses the group with the most appearances so
    that a secondary RM (labelled STD but chemically different) does not
    produce a spurious drift signal.
    """
    stds = [s for s in samples if s.is_standard]

    if len(stds) < 2:
        return None, "", None

    # Group by sample name; pick the largest group (primary bracketing standard)
    groups: Dict[str, List] = {}
    for s in stds:
        groups.setdefault(s.name, []).append(s)
    primary_group = max(groups.values(), key=len)

    if len(primary_group) < 2:
        return None, "", None

    stds_sorted = sorted(primary_group, key=lambda s: s.run_number)
    first_std = stds_sorted[0]
    last_std = stds_sorted[-1]

    first_layer = select_best_ratio_layer(first_std, ratio_name)
    last_layer = select_best_ratio_layer(last_std, ratio_name)
    if first_layer is None or last_layer is None:
        return None, "", None
    if first_layer.key != last_layer.key:
        return (
            None,
            "",
            "Session drift omitted: first and last standards use different "
            f"ratio layers ({first_layer.label}, {last_layer.label}).",
        )

    def get_mean(sample, selected_layer):
        valid = get_filtered_values(
            selected_layer.data.values,
            selected_layer.data.mask,
            sample.name,
            cycle_ranges=cycle_ranges,
            sample_key=get_sample_state_key(sample),
            filter_method=filter_method,
            filter_threshold=filter_threshold,
        )
        if valid is None:
            return None

        return np.nanmean(valid) if len(valid) > 0 else None

    first_mean = get_mean(first_std, first_layer)
    last_mean = get_mean(last_std, last_layer)

    if first_mean is None or last_mean is None or first_mean == 0:
        return None, first_layer.label, None

    return (
        ((last_mean - first_mean) / first_mean) * 100,
        first_layer.label,
        None,
    )


def _calculate_max_standard_deviation(
    samples: List[Sample],
    ratio_name: str,
    *,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
) -> Optional[float]:
    """Return max absolute standard deviation from the session mean, in percent."""
    stds = [s for s in samples if s.is_standard]
    if len(stds) < 2:
        return None

    means = []
    for std in stds:
        valid = _get_runtime_ratio_values(
            std,
            ratio_name,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
        )
        if valid is not None and len(valid) > 0:
            means.append(float(np.nanmean(valid)))

    if len(means) < 2:
        return None
    session_mean = float(np.nanmean(means))
    if session_mean == 0 or not np.isfinite(session_mean):
        return None
    deviations = np.abs((np.asarray(means, dtype=float) - session_mean) / session_mean) * 100.0
    finite = deviations[np.isfinite(deviations)]
    if len(finite) == 0:
        return None
    return float(np.nanmax(finite))


def _count_sample_outliers(
    samples: List[Sample],
    ratio_name: str,
    *,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
) -> int:
    """Count samples with RSD > 3x median RSD."""
    smps = [s for s in samples if s.is_sample]

    if len(smps) < 3:
        return 0

    rsds = []
    for sample in smps:
        valid = _get_runtime_ratio_values(
            sample,
            ratio_name,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
        )
        if valid is None:
            continue

        if len(valid) > 1:
            mean_val = np.nanmean(valid)
            std_val = np.nanstd(valid, ddof=1)
            if mean_val != 0:
                rsds.append((sample.name, std_val / mean_val * 100))

    if not rsds:
        return 0

    median_rsd = np.median([r[1] for r in rsds])
    outliers = [r for r in rsds if r[1] > 3 * median_rsd]

    return len(outliers)


def render_qc_panel(
    result: ProcessingResult,
    ratio_name: Optional[str] = None,
    compact: bool = False,
    *,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
    std_repro_threshold: float = 0.1,
    blank_contribution_threshold: Optional[float] = None,
    drift_threshold: float = 0.01,
) -> None:
    """Render the QC panel."""
    qc = calculate_qc_metrics(
        result,
        ratio_name,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
        std_repro_threshold=std_repro_threshold,
        blank_contribution_threshold=blank_contribution_threshold,
        drift_threshold=drift_threshold,
    )

    if compact:
        _render_compact_panel(qc)
    else:
        _render_full_panel(qc)


def _render_compact_panel(qc: QCResult) -> None:
    """Render compact QC panel (single line)."""
    if not qc.metrics:
        return

    if qc.overall_pass:
        st.success("QC: PASS")
    else:
        st.error("QC: FAIL")

    for start in range(0, len(qc.metrics), 3):
        row_metrics = qc.metrics[start:start + 3]
        cols = st.columns(len(row_metrics))
        for col, metric in zip(cols, row_metrics):
            with col:
                status = "PASS" if metric.passed is True else "CHECK" if metric.passed is False else metric.status.replace("_", " ").title()
                value = f"{metric.value:.4f} {metric.unit}" if metric.value is not None else "Unavailable"
                st.caption(f"{status} | {metric.name}: {value}")


def _render_full_panel(qc: QCResult) -> None:
    """Render full QC panel with details."""
    # Overall status header
    if qc.overall_pass:
        st.success("All QC checks passed")
    elif qc.warnings:
        st.warning(f"{len(qc.warnings)} QC issue(s) detected")
    else:
        st.info("No QC data available")

    if not qc.metrics:
        return

    # Metrics in wrapped rows
    for start in range(0, len(qc.metrics), 3):
        row_metrics = qc.metrics[start:start + 3]
        cols = st.columns(len(row_metrics))
        for col, metric in zip(cols, row_metrics):
            with col:
                delta_color = "normal" if metric.passed is not False else "inverse"
                value = f"{metric.value:.4f} {metric.unit}" if metric.value is not None else "Unavailable"
                if metric.threshold is None:
                    delta = "Not assessed" if metric.status == "not_assessed" else "Unavailable"
                else:
                    comparator = "≤" if metric.passed is True else ">"
                    delta = f"{comparator} {metric.threshold}"
                st.metric(
                    label=metric.name,
                    value=value,
                    delta=delta,
                    delta_color=delta_color,
                    help=metric.description,
                )

    # Warnings
    if qc.warnings:
        with st.expander(f"QC Warnings ({len(qc.warnings)})", expanded=True):
            for warning in qc.warnings:
                st.warning(warning)


def render_qc_dashboard(
    result: ProcessingResult,
    ratio_name: Optional[str] = None,
    *,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
    std_repro_threshold: float = 0.1,
    blank_contribution_threshold: Optional[float] = None,
    drift_threshold: float = 0.01,
) -> None:
    """Render the full QC dashboard with charts."""
    from ui.theme import get_theme

    theme = get_theme()

    # Header
    render_qc_panel(
        result,
        ratio_name,
        compact=False,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
        std_repro_threshold=std_repro_threshold,
        blank_contribution_threshold=blank_contribution_threshold,
        drift_threshold=drift_threshold,
    )

    st.divider()

    active_samples = [s for s in result.samples if not s.metadata.get("excluded", False)]

    if ratio_name is None:
        for s in active_samples:
            if s.ratios:
                ratio_name = list(s.ratios.keys())[0]
                break

    if ratio_name is None:
        return

    # Charts in columns
    col1, col2 = st.columns(2)

    with col1:
        # Standard reproducibility chart
        _render_std_chart(
            active_samples,
            ratio_name,
            theme,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
        )

    with col2:
        # Drift chart
        _render_drift_chart(
            active_samples,
            ratio_name,
            theme,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
        )


def _render_std_chart(
    samples: List[Sample],
    ratio_name: str,
    theme,
    *,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
) -> None:
    """Render standards reproducibility chart."""
    import plotly.graph_objects as go
    from ui.utils import format_name

    stds = [s for s in samples if s.is_standard]
    if len(stds) < 2:
        st.caption("Not enough standards for reproducibility chart.")
        return

    stds_sorted = sorted(stds, key=lambda s: s.run_number)

    names = []
    means = []
    errors = []

    for std in stds_sorted:
        valid = _get_runtime_ratio_values(
            std,
            ratio_name,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
        )
        if valid is None:
            continue

        if len(valid) > 0:
            names.append(std.name)
            mean_val = np.nanmean(valid)
            std_val = np.nanstd(valid, ddof=1) if len(valid) > 1 else np.nan
            se_val = std_val / np.sqrt(len(valid))
            means.append(mean_val)
            errors.append(2 * se_val)

    if not means:
        return

    overall_mean = float(np.mean(means))
    overall_sd = float(np.std(means, ddof=1)) if len(means) > 1 else np.nan

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=names,
        y=means,
        mode="markers",
        error_y=dict(type="data", array=errors, visible=True),
        marker=dict(color=theme.palette.sample_std, size=10, symbol="diamond"),
        name="Standards",
    ))

    # Mean line
    fig.add_hline(
        y=overall_mean,
        line_dash="dash",
        line_color=theme.palette.sample_std,
        line_width=1.1,
    )

    fig.update_layout(
        title="Standard Reproducibility",
        xaxis_title="",
        yaxis_title=format_name(ratio_name),
        yaxis_tickformat=".6f",
        height=380,
        margin=dict(l=80, r=20, t=60, b=90),
        showlegend=False,
    )
    fig.update_xaxes(
        tickangle=-45,
        showgrid=False,
        zeroline=False,
        showline=True,
        tickfont=dict(size=PLOTLY_BASE_FONT_SIZE),
        title_font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        tickfont=dict(size=PLOTLY_BASE_FONT_SIZE),
        title_font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.02,
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        text=f"Mean \u00b1 2SD: {format_value_with_uncertainty(overall_mean, 2 * overall_sd)}",
        font=dict(size=PLOTLY_ANNOTATION_FONT_SIZE, color=theme.palette.annotation_text_color),
        bgcolor=theme.palette.annotation_bg,
        bordercolor=theme.palette.annotation_border,
        borderwidth=1,
    )
    
    from ui.config_plotly import get_plotly_config
    theme.apply_to_figure(fig, profile="overview")
    st.plotly_chart(fig, width="stretch", key=f"qc_std_chart_{ratio_name}", config=get_plotly_config())


def _render_drift_chart(
    samples: List[Sample],
    ratio_name: str,
    theme,
    *,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
) -> None:
    """Render session drift chart showing all standards over time."""
    import plotly.graph_objects as go
    from ui.utils import format_name

    stds = [s for s in samples if s.is_standard]
    if len(stds) < 2:
        st.caption("Not enough standards for drift chart.")
        return

    stds_sorted = sorted(stds, key=lambda s: s.run_number)

    run_numbers = []
    means = []

    for std in stds_sorted:
        valid = _get_runtime_ratio_values(
            std,
            ratio_name,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
        )
        if valid is None:
            continue

        if len(valid) > 0:
            run_numbers.append(std.run_number)
            means.append(np.nanmean(valid))

    if not means:
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=run_numbers,
        y=means,
        mode="markers+lines",
        marker=dict(color=theme.palette.sample_std, size=8, symbol="diamond"),
        line=dict(color=theme.palette.sample_std, width=1.2, dash="dot"),
        name="Standards",
    ))

    # Linear fit for trend
    if len(means) > 2:
        z = np.polyfit(run_numbers, means, 1)
        p = np.poly1d(z)
        trend_y = p(run_numbers)

        fig.add_trace(go.Scatter(
            x=run_numbers,
            y=trend_y,
            mode="lines",
            line=dict(color=theme.palette.guide_bounds, width=1.1, dash="dash"),
            name="Trend",
        ))

    fig.update_layout(
        title="Session Drift",
        xaxis_title="Run Number",
        yaxis_title=format_name(ratio_name),
        yaxis_tickformat=".6f",
        height=380,
        margin=dict(l=80, r=20, t=60, b=70),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        tickfont=dict(size=PLOTLY_BASE_FONT_SIZE),
        title_font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        tickfont=dict(size=PLOTLY_BASE_FONT_SIZE),
        title_font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
    )
    
    from ui.config_plotly import get_plotly_config
    theme.apply_to_figure(fig, profile="timeseries")
    st.plotly_chart(fig, width="stretch", key=f"qc_drift_chart_{ratio_name}", config=get_plotly_config())

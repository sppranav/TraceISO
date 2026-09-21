"""Drift diagnostics for the uncertainty tab."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import UncertaintyConfig
from domain.models import ReprodResult
from ui.formatting import format_uncertainty
from ui.tabs.uncertainty_sections.common import (
    _render_compact_summary,
    _render_equation_note_block,
)
from ui.tabs.uncertainty_sections.repeatability import (
    _render_break_controls,
    _render_std_include_table,
)
from ui.config_plotly import (
    get_plotly_config,
    PLOTLY_AXIS_TITLE_FONT_SIZE,
    PLOTLY_BASE_FONT_SIZE,
)
from ui.theme import get_theme
from ui.utils import format_name


def _render_drift_section(
    u_config: UncertaintyConfig,
    all_samples: list,
    ratio_name: str,
    state,
) -> UncertaintyConfig:
    """Render Type B instrumental-drift diagnostics for k5."""
    from domain.uncertainty.reprod import compute_reprod

    element_symbol = getattr(state, "element_symbol", "") or ""
    if not element_symbol and getattr(state, "element_config", None) is not None:
        element_symbol = getattr(state.element_config, "symbol", "") or ""
    if not u_config.is_contributor_enabled(
        "u_k5_instrumental_drift",
        element_symbol=element_symbol,
    ):
        return u_config

    st.divider()
    st.subheader("Instrumental Drift (k5)")

    stds = [s for s in all_samples if s.is_standard and not s.metadata.get("excluded", False)]
    if not stds:
        st.info("No standards available for drift diagnostics.")
        return u_config

    drift_config = replace(u_config, include_kappa_drift=True)
    reprod = compute_reprod(
        all_samples=all_samples,
        ratio_name=ratio_name,
        uncertainty_config=drift_config,
        element_config=state.element_config,
    )

    included_count = int(np.sum(np.asarray(reprod.std_included, dtype=bool))) if reprod.std_included else 0
    detail_df, stats = _build_drift_detail_dataframe(reprod)

    included_means = (
        np.asarray(reprod.std_means, dtype=float)[np.asarray(reprod.std_included, dtype=bool)]
        if reprod.std_means is not None and reprod.std_included
        else np.array([], dtype=float)
    )
    ref_mean = float(np.mean(included_means)) if len(included_means) > 0 else 0.0
    if u_config.output_mode == "absolute_ratio":
        kappa_abs = (reprod.kappa_drift_permil / 1000.0) * ref_mean if ref_mean else 0.0
        kappa_value = format_uncertainty(kappa_abs)
        mean_abs_delta_value = (
            format_uncertainty((float(stats["mean_abs_delta"]) / 1000.0) * ref_mean)
            if ref_mean and stats["mean_abs_delta"] is not None
            else "n/a"
        )
    else:
        kappa_value = format_uncertainty(reprod.kappa_drift_permil, unit="‰")
        mean_abs_delta_value = (
            format_uncertainty(float(stats["mean_abs_delta"]), unit="‰")
            if stats["mean_abs_delta"] is not None
            else "n/a"
        )

    status_text = "Active in budget" if u_config.include_kappa_drift else "Preview only"
    _render_compact_summary(
        [
            ("Status", status_text),
            ("Included standards", str(included_count)),
            ("Consecutive pairs", str(int(stats["n_pairs"] or 0))),
            ("k5", kappa_value),
            ("Mean |Δ|", mean_abs_delta_value),
        ]
    )

    if not u_config.include_kappa_drift:
        st.caption(
            "Preview only: enable instrumental drift (k5) in Uncertainty Configuration -> Budget contributors to include this Type B term in the budget."
        )

    if detail_df.empty:
        if included_count < 2:
            st.info("Need at least 2 included standards to estimate drift.")
        else:
            st.info("No within-segment consecutive drift pairs are available.")
    else:
        st.plotly_chart(
            _build_drift_standard_figure(reprod, ratio_name),
            width="stretch",
            config=get_plotly_config(),
        )
        st.plotly_chart(
            _build_drift_figure(reprod, ratio_name),
            width="stretch",
            config=get_plotly_config(),
        )
        st.caption(
            "Only consecutive included standards within the same segment contribute. "
            "The orange guides show +/- mean(|Delta|); k5 = mean(|Delta|) / 2."
        )

    prev_excluded = list(u_config.excluded_standards)
    u_config = _render_std_include_table(
        reprod,
        u_config,
        key_prefix="drift",
        expander_label="Standard Include/Exclude",
    )
    if sorted(u_config.excluded_standards) != sorted(prev_excluded):
        state.uncertainty_config = u_config
        st.rerun()

    with st.expander("Method Equations", expanded=False):
        current_distribution = str(
            getattr(u_config, "kappa_drift_distribution", "normal") or "normal"
        ).lower()
        if current_distribution not in {"normal", "rectangular"}:
            current_distribution = "normal"
        distribution = st.selectbox(
            "Monte Carlo distribution",
            options=["normal", "rectangular"],
            index=0 if current_distribution == "normal" else 1,
            key="drift_k5_distribution",
            help=(
                "k5 is already a standard uncertainty. Rectangular sampling "
                "uses a half-width of sqrt(3) * k5; it does not change the /2 calculation."
            ),
        )
        if distribution != current_distribution:
            u_config = replace(u_config, kappa_drift_distribution=distribution)
            state.uncertainty_config = u_config
        _render_drift_equation_block(u_config)

    with st.expander("Per-Standard Detail", expanded=False):
        per_standard_df = _build_drift_per_standard_dataframe(reprod)
        st.dataframe(per_standard_df, width="stretch", hide_index=True)

    with st.expander("Segment Details & Break Controls", expanded=False):
        _render_drift_segment_details(reprod)
        if str(getattr(u_config, "ssb_mode", "alternating")) == "block_average":
            st.info(
                "Break editing is unavailable in block-average mode because blocks "
                "are currently evaluated as one segment."
            )
        else:
            u_config = _render_break_controls(
                reprod,
                u_config,
                all_samples,
                ratio_name,
                state,
                key_prefix="drift",
            )

    return u_config


def _render_drift_equation_block(u_config: UncertaintyConfig) -> None:
    """Render the k5 equations used for the Type B drift term."""
    lines = [
        r"\Delta_i = \left(\frac{y_{i+1} - y_i}{y_i}\right)\times 1000",
        r"u(k_5) = \frac{\mathrm{mean}\left(\left|\Delta_i\right|\right)}{2}",
    ]
    if u_config.output_mode == "absolute_ratio":
        lines.append(r"u_{k_5} = k_5\times \bar{y}/1000")
    _render_equation_note_block(
        lines,
        title="Instrumental Drift (k5) Equations",
        note=(
            "Here y_i and y_(i+1) are consecutive included standard means within "
            "one segment. The /2 estimator defines the Type B standard uncertainty; "
            "the Monte Carlo distribution is configured separately."
        ),
    )


def _build_drift_detail_dataframe(
    reprod: ReprodResult,
) -> Tuple[pd.DataFrame, Dict[str, float | int | None]]:
    """Build per-pair drift rows from the current included standard sequence."""
    from domain.uncertainty.reprod import _eligible_kappa_drift_pairs

    pairs = _eligible_kappa_drift_pairs(reprod.std_included, reprod.std_segments)
    if not pairs:
        return pd.DataFrame(), {
            "n_pairs": 0,
            "mean_abs_delta": None,
            "max_abs_delta": None,
            "mean_signed_delta": None,
        }

    rows: list[dict[str, object]] = []
    stored_deltas = (
        np.asarray(reprod.drift_deltas, dtype=float)
        if reprod.drift_deltas is not None and len(reprod.drift_deltas) == len(pairs)
        else None
    )
    for pair_offset, (left_idx, right_idx) in enumerate(pairs):
        left_pos = float(reprod.std_positions[left_idx])
        right_pos = float(reprod.std_positions[right_idx])
        left_mean = float(reprod.std_means[left_idx])
        if np.isclose(left_mean, 0.0, rtol=0.0, atol=1e-15):
            continue
        delta = (
            float(stored_deltas[pair_offset])
            if stored_deltas is not None
            else (float(reprod.std_means[right_idx]) - left_mean) / left_mean * 1000.0
        )
        rows.append(
            {
                "From": reprod.std_names[left_idx],
                "To": reprod.std_names[right_idx],
                "Run pair": (
                    f"{int(left_pos) if left_pos.is_integer() else left_pos}"
                    f" -> "
                    f"{int(right_pos) if right_pos.is_integer() else right_pos}"
                ),
                "Pair midpoint": (left_pos + right_pos) / 2.0,
                "Mean (left)": left_mean,
                "Mean (right)": float(reprod.std_means[right_idx]),
                "Δ drift (‰)": delta,
                "|Δ| (‰)": abs(delta),
            }
        )

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        return detail_df, {
            "n_pairs": 0,
            "mean_abs_delta": None,
            "max_abs_delta": None,
            "mean_signed_delta": None,
        }

    abs_vals = detail_df["|Δ| (‰)"].to_numpy(dtype=float)
    signed_vals = detail_df["Δ drift (‰)"].to_numpy(dtype=float)
    return detail_df, {
        "n_pairs": len(detail_df),
        "mean_abs_delta": float(np.mean(abs_vals)),
        "max_abs_delta": float(np.max(abs_vals)),
        "mean_signed_delta": float(np.mean(signed_vals)),
    }


def _build_drift_per_standard_dataframe(reprod: ReprodResult) -> pd.DataFrame:
    """Return one auditable row per standard and its outgoing pair state."""
    included_indices = [
        idx for idx, included in enumerate(reprod.std_included) if included
    ]
    next_included = {
        left: right for left, right in zip(included_indices, included_indices[1:])
    }
    rows: list[dict[str, object]] = []
    for idx, name in enumerate(reprod.std_names):
        included = bool(reprod.std_included[idx])
        segment = int(reprod.std_segments[idx]) if reprod.std_segments else 1
        row: dict[str, object] = {
            "Standard": name,
            "Position": float(reprod.std_positions[idx]),
            "Segment": segment,
            "Mean": float(reprod.std_means[idx]),
            "Included": included,
            "Next included standard": "—",
            "Δ drift (‰)": np.nan,
            "|Δ| (‰)": np.nan,
            "Pair status": "excluded standard" if not included else "final included standard",
        }
        right_idx = next_included.get(idx)
        if included and right_idx is not None:
            row["Next included standard"] = reprod.std_names[right_idx]
            right_segment = (
                int(reprod.std_segments[right_idx]) if reprod.std_segments else 1
            )
            if segment != right_segment:
                row["Pair status"] = "excluded: segment boundary"
            else:
                left_mean = float(reprod.std_means[idx])
                if np.isclose(left_mean, 0.0, rtol=0.0, atol=1e-15):
                    row["Pair status"] = "excluded: near-zero denominator"
                else:
                    delta = (
                        float(reprod.std_means[right_idx]) - left_mean
                    ) / left_mean * 1000.0
                    row["Δ drift (‰)"] = delta
                    row["|Δ| (‰)"] = abs(delta)
                    row["Pair status"] = "included in k5"
        rows.append(row)
    return pd.DataFrame(rows)


def _render_drift_segment_details(reprod: ReprodResult) -> None:
    """Render pair-weighted k5 diagnostics for each segment."""
    detail_df, _ = _build_drift_detail_dataframe(reprod)
    from domain.uncertainty.reprod import _eligible_kappa_drift_pairs

    eligible_pairs = _eligible_kappa_drift_pairs(
        reprod.std_included,
        reprod.std_segments,
    )
    segments = sorted(set(int(value) for value in (reprod.std_segments or [1])))
    if not segments:
        st.info("No segment information is available.")
        return
    columns = st.columns(min(len(segments), 4))
    for offset, segment in enumerate(segments):
        included_count = sum(
            bool(included) and int(reprod.std_segments[idx]) == segment
            for idx, included in enumerate(reprod.std_included)
        ) if reprod.std_segments else sum(bool(value) for value in reprod.std_included)
        segment_rows = pd.DataFrame()
        if not detail_df.empty:
            pair_segments = np.asarray(
                [
                    int(reprod.std_segments[left_idx]) if reprod.std_segments else 1
                    for left_idx, _ in eligible_pairs
                    if not np.isclose(
                        float(reprod.std_means[left_idx]),
                        0.0,
                        rtol=0.0,
                        atol=1e-15,
                    )
                ],
                dtype=int,
            )
            segment_rows = detail_df[pair_segments == segment]
        mean_abs = (
            float(segment_rows["|Δ| (‰)"].mean()) if not segment_rows.empty else 0.0
        )
        with columns[offset % len(columns)]:
            st.markdown(f"**Segment {segment}**")
            st.caption(f"{included_count} included standards")
            st.caption(f"{len(segment_rows)} eligible pairs")
            st.metric("Mean |Δ|", f"{mean_abs:.4g} ‰" if mean_abs else "—")
            st.caption(f"k5 = {mean_abs / 2.0:.4g} ‰" if mean_abs else "k5 = —")
    if len(segments) > 1:
        st.caption("The reported global k5 is pair-weighted across all segments.")


_DRIFT_DETAIL_COLUMN_CONFIG = {
    "Mean (left)": st.column_config.NumberColumn("Mean (left)", format="%.8f"),
    "Mean (right)": st.column_config.NumberColumn("Mean (right)", format="%.8f"),
    "Pair midpoint": st.column_config.NumberColumn("Pair midpoint", format="%.4f"),
    "Δ drift (‰)": st.column_config.NumberColumn("Δ drift (‰)", format="%.4f"),
    "|Δ| (‰)": st.column_config.NumberColumn("|Δ| (‰)", format="%.4f"),
}


def _format_drift_detail_dataframe(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Format the drift detail table for compact display."""
    if detail_df.empty:
        return detail_df
    return detail_df.copy()


def _build_drift_standard_figure(
    reprod: ReprodResult,
    ratio_name: str,
):
    """Build a standards-only context figure for the drift section."""
    import plotly.graph_objects as go

    theme = get_theme()
    palette = theme.palette
    fig = go.Figure()
    included_x: list[float] = []
    included_y: list[float] = []
    included_text: list[str] = []
    excluded_x: list[float] = []
    excluded_y: list[float] = []
    excluded_text: list[str] = []

    for idx, name in enumerate(reprod.std_names):
        x_val = float(reprod.std_positions[idx])
        y_val = float(reprod.std_means[idx])
        hover = f"{name}<br>Mean: {y_val:.8f}<br>Run: {x_val:g}"
        if reprod.std_included[idx]:
            included_x.append(x_val)
            included_y.append(y_val)
            included_text.append(hover)
        else:
            excluded_x.append(x_val)
            excluded_y.append(y_val)
            excluded_text.append(hover)

    if included_x:
        fig.add_trace(
            go.Scatter(
                x=included_x,
                y=included_y,
                mode="markers+lines",
                marker=dict(size=9, color=palette.sample_std, symbol="diamond"),
                line=dict(color=palette.sample_std, width=1.4),
                name="Included standards",
                hovertemplate="%{text}<extra></extra>",
                text=included_text,
            )
        )
    if excluded_x:
        fig.add_trace(
            go.Scatter(
                x=excluded_x,
                y=excluded_y,
                mode="markers",
                marker=dict(size=10, color=palette.layer_excluded, symbol="x"),
                name="Excluded standards",
                hovertemplate="%{text}<extra></extra>",
                text=excluded_text,
            )
        )

    fig.update_layout(
        title="Standard Sequence Used for k5",
        xaxis_title="Run number",
        yaxis_title=format_name(ratio_name),
        height=320,
        margin=dict(l=80, r=20, t=60, b=60),
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
    theme.apply_to_figure(fig, profile="timeseries")
    return fig


def _build_drift_figure(
    reprod: ReprodResult,
    ratio_name: str,
):
    """Build a drift-delta figure from consecutive included standards."""
    import plotly.graph_objects as go

    theme = get_theme()
    palette = theme.palette
    detail_df, stats = _build_drift_detail_dataframe(reprod)
    fig = go.Figure()

    if detail_df.empty:
        fig.update_layout(
            title="Consecutive Standard Drift",
            height=320,
            margin=dict(l=80, r=20, t=60, b=60),
        )
        theme.apply_to_figure(fig, profile="timeseries")
        return fig

    x_vals = detail_df["Pair midpoint"].to_numpy(dtype=float)
    y_vals = detail_df["Δ drift (‰)"].to_numpy(dtype=float)
    hover = [
        f"{row['From']} -> {row['To']}<br>"
        f"Left mean: {float(row['Mean (left)']):.8f}<br>"
        f"Right mean: {float(row['Mean (right)']):.8f}<br>"
        f"Δ drift: {float(row['Δ drift (‰)']):.4f}‰"
        for _, row in detail_df.iterrows()
    ]

    mean_abs_delta = float(stats["mean_abs_delta"]) if stats["mean_abs_delta"] is not None else 0.0
    if mean_abs_delta > 0:
        fig.add_hrect(
            y0=-mean_abs_delta,
            y1=mean_abs_delta,
            fillcolor=palette.guide_bounds_light,
            line_width=0,
            layer="below",
        )
        fig.add_hline(y=mean_abs_delta, line_dash="dot", line_color=palette.guide_bounds, line_width=1.2)
        fig.add_hline(y=-mean_abs_delta, line_dash="dot", line_color=palette.guide_bounds, line_width=1.2)

    fig.add_hline(y=0, line_dash="solid", line_color=palette.figure_ink, line_width=1.2)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers+lines",
            marker=dict(size=9, color=palette.layer_drift, symbol="square"),
            line=dict(color=palette.layer_drift, width=1.4),
            name="Consecutive drift",
            hovertemplate="%{text}<extra></extra>",
            text=hover,
        )
    )

    fig.update_layout(
        title="Consecutive Standard Drift (k5)",
        xaxis_title="Standard-pair midpoint (run number)",
        yaxis_title="Δ drift (‰)",
        height=320,
        margin=dict(l=80, r=20, t=60, b=60),
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

    theme.apply_to_figure(fig, profile="timeseries")
    return fig



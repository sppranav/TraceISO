"""Run sequence chart component for TraceISO."""

import html
from typing import List, Optional

import plotly.graph_objects as go
import streamlit as st

from domain.models import Sample
from ui.theme import get_theme
from ui.config_plotly import get_plotly_config


def render_run_sequence(
    samples: List[Sample],
    selected_idx: Optional[int] = None,
    height: int = 150,
) -> None:
    """Render a run sequence timeline chart."""
    if not samples:
        st.info("No samples to display.")
        return

    theme = get_theme()
    palette = theme.get_palette()
    marker_outline = palette.figure_surface
    x_values = [sample.run_number for sample in samples]
    x_min = min(x_values)
    x_max = max(x_values)

    # Group samples by type
    std_x, std_y, std_names, std_colors, std_status = [], [], [], [], []
    smp_x, smp_y, smp_names, smp_colors, smp_status = [], [], [], [], []
    blk_x, blk_y, blk_names, blk_colors, blk_status = [], [], [], [], []

    for i, sample in enumerate(samples):
        stype = sample.sample_type.upper()
        x = sample.run_number
        y = 1  # All on same horizontal line
        is_excluded = bool(sample.metadata.get("excluded", False))
        marker_color = theme.sample_color(stype, excluded=is_excluded)
        status = "Excluded" if is_excluded else "Active"

        if stype in ("STD", "STANDARD"):
            std_x.append(x)
            std_y.append(y)
            std_names.append(html.escape(sample.name))
            std_colors.append(marker_color)
            std_status.append(status)
        elif stype in ("BLK", "BLANK"):
            blk_x.append(x)
            blk_y.append(y)
            blk_names.append(html.escape(sample.name))
            blk_colors.append(marker_color)
            blk_status.append(status)
        else:
            smp_x.append(x)
            smp_y.append(y)
            smp_names.append(html.escape(sample.name))
            smp_colors.append(marker_color)
            smp_status.append(status)

    fig = go.Figure()

    # Add traces for each type
    if std_x:
        fig.add_trace(
            go.Scatter(
                x=std_x,
                y=std_y,
                mode="markers",
                name="Standards",
                marker=dict(
                    size=14,
                    color=std_colors,
                    symbol=theme.sample_symbol("STD"),
                    line=dict(color=marker_outline, width=1.5),
                ),
                text=std_names,
                customdata=std_status,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Run: %{x}<br>"
                    "Status: %{customdata}<extra>STD</extra>"
                ),
            )
        )

    if smp_x:
        fig.add_trace(
            go.Scatter(
                x=smp_x,
                y=smp_y,
                mode="markers",
                name="Samples",
                marker=dict(
                    size=12,
                    color=smp_colors,
                    symbol=theme.sample_symbol("SMP"),
                    line=dict(color=marker_outline, width=1.5),
                ),
                text=smp_names,
                customdata=smp_status,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Run: %{x}<br>"
                    "Status: %{customdata}<extra>SMP</extra>"
                ),
            )
        )

    if blk_x:
        fig.add_trace(
            go.Scatter(
                x=blk_x,
                y=blk_y,
                mode="markers",
                name="Blanks",
                marker=dict(
                    size=10,
                    color=blk_colors,
                    symbol=theme.sample_symbol("BLK"),
                    line=dict(color=marker_outline, width=1.5),
                ),
                text=blk_names,
                customdata=blk_status,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Run: %{x}<br>"
                    "Status: %{customdata}<extra>BLK</extra>"
                ),
            )
        )

    # Highlight selected sample
    if selected_idx is not None and 0 <= selected_idx < len(samples):
        sel_sample = samples[selected_idx]
        fig.add_vrect(
            x0=sel_sample.run_number - 0.45,
            x1=sel_sample.run_number + 0.45,
            fillcolor=palette.guide_bounds_light,
            line_width=0,
            opacity=0.45,
            layer="below",
        )
        fig.add_trace(
            go.Scatter(
                x=[sel_sample.run_number],
                y=[1],
                mode="markers",
                name="Selected",
                marker=dict(
                    size=20,
                    # Transparent fill preserves the selected sample's semantic color.
                    color="rgba(0,0,0,0)",
                    line=dict(color=palette.figure_ink, width=3),
                ),
                text=[html.escape(sel_sample.name)],
                hovertemplate="<b>%{text}</b> (selected)<extra></extra>",
                showlegend=False,
            )
        )

    fig.add_shape(
        type="line",
        x0=x_min - 0.5,
        x1=x_max + 0.5,
        y0=1,
        y1=1,
        line=dict(color=palette.annotation_border, width=2),
        layer="below",
    )

    # Layout
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=20, t=34, b=30),
        xaxis=dict(
            title="Sequence Number",
            showgrid=False,
            zeroline=False,
            dtick=1 if len(samples) <= 30 else None,
            range=[x_min - 0.5, x_max + 0.5],
        ),
        yaxis=dict(
            visible=False,
            range=[0.82, 1.18],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            traceorder="normal",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        hovermode="closest",
    )

    theme.apply_to_figure(fig, profile="overview")
    st.plotly_chart(
        fig,
        width="stretch",
        key="run_sequence_chart",
        config=get_plotly_config(),
    )


def render_run_sequence_compact(
    samples: List[Sample],
    selected_idx: Optional[int] = None,
) -> None:
    """Render a compact inline run sequence using st.columns."""
    if not samples:
        return

    theme = get_theme()

    # Create a row of colored indicators
    n_cols = min(len(samples), 20)  # Limit to 20 visible
    cols = st.columns(n_cols)

    for i, (col, sample) in enumerate(zip(cols, samples[:n_cols])):
        stype = sample.sample_type.upper()
        is_excluded = bool(sample.metadata.get("excluded", False))
        color = theme.sample_color(stype, excluded=is_excluded)
        marker = f'<span style="color:{color};">&#9679;</span>'

        # Highlight selected
        style = "font-weight: bold;" if i == selected_idx else ""

        with col:
            st.markdown(
                f'<span style="{style}" title="{html.escape(sample.name)}">{marker}</span>',
                unsafe_allow_html=True,
            )

    if len(samples) > n_cols:
        st.caption(f"... and {len(samples) - n_cols} more samples")

"""Small reusable helpers for publication-safe Plotly figures."""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go


def add_legend_proxy(
    fig: go.Figure,
    *,
    name: str,
    line: Optional[dict] = None,
    fillcolor: Optional[str] = None,
    legendgroup: Optional[str] = None,
) -> None:
    """Add a zero-data trace representing a shape in the figure legend."""
    if fillcolor:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=name,
                marker=dict(color=fillcolor, size=12, symbol="square"),
                legendgroup=legendgroup,
                showlegend=True,
                hoverinfo="skip",
            )
        )
        return

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=name,
            line=line or {},
            legendgroup=legendgroup,
            showlegend=True,
            hoverinfo="skip",
        )
    )

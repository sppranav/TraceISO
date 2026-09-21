"""Shared table formatting helpers for the Uncertainty tab diagnostic tables."""

from __future__ import annotations

import pandas as pd

_STATUS_STYLES: dict[str, str] = {
    "ACTIVE": "background-color: rgba(46, 204, 113, 0.18)",
    "OK": "background-color: rgba(46, 204, 113, 0.18)",
    "ELEVATED": "background-color: rgba(241, 196, 15, 0.22)",
    "INTERF": "background-color: rgba(230, 126, 34, 0.22)",
    "HIGH": "background-color: rgba(231, 76, 60, 0.25)",
    "MISSING": "background-color: rgba(231, 76, 60, 0.15)",
    "BY DESIGN": "background-color: rgba(149, 165, 166, 0.15)",
    "INSUFFICIENT": "background-color: rgba(149, 165, 166, 0.20)",
    "UNAVAILABLE": "background-color: rgba(149, 165, 166, 0.20)",
}


def style_status_column(
    df: pd.DataFrame, col: str = "Status"
) -> "pd.io.formats.style.Styler":
    """Return a Styler with background-colour highlighting on *col*."""
    if col not in df.columns:
        return df.style
    return df.style.map(
        lambda v: _STATUS_STYLES.get(str(v).upper(), ""),
        subset=[col],
    )

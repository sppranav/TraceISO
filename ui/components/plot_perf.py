"""
Plot performance helpers for UI chart components.

These helpers intentionally avoid any data downsampling and only adjust
styling/interaction settings for dense traces.
"""

from __future__ import annotations

from typing import Any, Dict


def get_plot_perf_options(plot_config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return normalized performance options from ``plot_config``."""
    cfg = plot_config or {}
    return {
        "outline_mode": cfg.get("outline_mode", "adaptive"),
        "outline_max_points": int(cfg.get("outline_max_points", 2500)),
        "dense_hover_simplify": bool(cfg.get("dense_hover_simplify", True)),
        "dense_hover_max_points": int(cfg.get("dense_hover_max_points", 4000)),
    }


def resolve_marker_outline_width(
    n_points: int,
    requested_width: float,
    outline_mode: str,
    outline_max_points: int,
) -> float:
    """Resolve marker outline width based on configured performance mode."""
    mode = (outline_mode or "adaptive").lower()
    if mode == "off":
        return 0.0
    if mode == "on":
        return float(requested_width)
    # Adaptive mode: keep outlines on moderate traces, disable on dense traces.
    if n_points > max(int(outline_max_points), 0):
        return 0.0
    return float(requested_width)


def should_simplify_hover(
    n_points: int,
    dense_hover_simplify: bool,
    dense_hover_max_points: int,
) -> bool:
    """Return True when dense-trace hover should be simplified."""
    if not dense_hover_simplify:
        return False
    return n_points > max(int(dense_hover_max_points), 0)

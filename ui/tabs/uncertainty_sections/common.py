"""Shared UI helpers for the uncertainty tab."""

from __future__ import annotations

from typing import Optional

import streamlit as st


def _chunked(items, size: int):
    """Yield successive chunks of *items* with at most *size* entries."""
    if size <= 0:
        raise ValueError("Chunk size must be greater than 0.")
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _render_compact_summary(items: list[tuple[str, str]]) -> None:
    """Render compact summary values without metric-card boxes."""
    if not items:
        return

    for row_items in _chunked(items, 4):
        cols = st.columns(len(row_items))
        for col, (label, value) in zip(cols, row_items):
            with col:
                st.caption(label)
                st.markdown(f"**{value}**")


def _render_equation_note_block(
    lines: list[str],
    *,
    title: Optional[str] = None,
    note: Optional[str] = None,
) -> None:
    """Render a consistent equation-style note block outside the coverage expander."""
    if title:
        st.caption(f"**{title}**")
    for line in lines:
        st.latex(line)
    if note:
        st.caption(note)


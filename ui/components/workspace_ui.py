"""Presentation-only primitives for TraceISO workspace tabs.

The helpers in this module format values supplied by callers. They do not read
application state, import domain code, or perform scientific calculations.
"""

from __future__ import annotations

import html
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import streamlit as st


_STATUS_TONES = frozenset({"neutral", "positive", "warning", "critical"})


def _api(streamlit_api: Any | None) -> Any:
    return st if streamlit_api is None else streamlit_api


def render_panel_heading(
    title: str,
    *,
    subtitle: str | None = None,
    eyebrow: str | None = None,
    streamlit_api: Any | None = None,
) -> None:
    """Render a compact heading from already-computed display text."""

    parts = ['<div class="ws-panel-heading">']
    if eyebrow:
        parts.append(f'<div class="ws-panel-eyebrow">{html.escape(eyebrow)}</div>')
    parts.append(f'<div class="ws-panel-title">{html.escape(title)}</div>')
    if subtitle:
        parts.append(f'<div class="ws-panel-subtitle">{html.escape(subtitle)}</div>')
    parts.append("</div>")
    _api(streamlit_api).markdown("".join(parts), unsafe_allow_html=True)


@contextmanager
def workspace_panel(
    title: str,
    *,
    subtitle: str | None = None,
    eyebrow: str | None = None,
    key: str | None = None,
    streamlit_api: Any | None = None,
) -> Iterator[None]:
    """Create a bordered panel and render its heading before caller content."""

    api = _api(streamlit_api)
    kwargs: dict[str, Any] = {"border": True}
    if key is not None:
        kwargs["key"] = key
    with api.container(**kwargs):
        render_panel_heading(
            title,
            subtitle=subtitle,
            eyebrow=eyebrow,
            streamlit_api=api,
        )
        yield


def render_status_chip(
    label: str,
    *,
    tone: str = "neutral",
    streamlit_api: Any | None = None,
) -> None:
    """Render a status label without deriving or recomputing its meaning."""

    if tone not in _STATUS_TONES:
        raise ValueError(f"Unsupported workspace status tone: {tone!r}")
    markup = (
        f'<span class="ws-status-chip" data-tone="{tone}">'
        f'<span class="ws-status-dot"></span>{html.escape(label)}</span>'
    )
    _api(streamlit_api).markdown(markup, unsafe_allow_html=True)


def render_next_step(
    title: str,
    reason: str,
    *,
    action_label: str | None = None,
    action_key: str | None = None,
    disabled: bool = False,
    streamlit_api: Any | None = None,
) -> bool:
    """Render an explanatory next-step state and optional native button."""

    api = _api(streamlit_api)
    with api.container(border=True):
        markup = (
            '<div class="ws-next-step">'
            f'<div class="ws-next-step-title">{html.escape(title)}</div>'
            f'<div class="ws-next-step-reason">{html.escape(reason)}</div>'
            "</div>"
        )
        api.markdown(markup, unsafe_allow_html=True)
        if action_label is not None:
            return bool(
                api.button(
                    action_label,
                    key=action_key,
                    disabled=disabled,
                    width="stretch",
                )
            )
    return False


def render_compact_metadata(
    items: Sequence[tuple[str, object]],
    *,
    streamlit_api: Any | None = None,
) -> None:
    """Render compact label/value metadata supplied by a caller."""

    entries = "".join(
        '<span class="ws-metadata-item">'
        f'<span class="ws-metadata-label">{html.escape(str(label))}</span>'
        f'<span class="ws-metadata-value">{html.escape(str(value))}</span>'
        "</span>"
        for label, value in items
    )
    _api(streamlit_api).markdown(
        f'<div class="ws-metadata">{entries}</div>',
        unsafe_allow_html=True,
    )


def render_group_heading(
    title: str,
    *,
    streamlit_api: Any | None = None,
) -> None:
    """Render a full-width settings-group heading from caller-supplied text."""

    _api(streamlit_api).markdown(
        f'<div class="ws-group-heading">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )


def render_statistics_table(
    columns: Sequence[str],
    rows: Sequence[tuple[str, Sequence[object]]],
    *,
    caption: str | None = None,
    streamlit_api: Any | None = None,
) -> None:
    """Render one aligned table of already-formatted statistics.

    ``columns`` names the row-label column followed by one column per series;
    each row is a label and exactly one cell per series. Cells are escaped
    text in semantic table markup, so alignment never depends on whitespace.
    """

    n_series = len(columns) - 1
    if n_series < 1:
        raise ValueError("A statistics table needs a label column and at least one series")
    head = "".join(f'<th scope="col">{html.escape(str(name))}</th>' for name in columns)
    body = []
    for label, cells in rows:
        if len(cells) != n_series:
            raise ValueError(
                f"Row {label!r} has {len(cells)} cells for {n_series} series"
            )
        values = "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells)
        body.append(f'<tr><th scope="row">{html.escape(str(label))}</th>{values}</tr>')
    caption_markup = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    _api(streamlit_api).markdown(
        '<div class="ws-stats-table-wrap"><table class="ws-stats-table">'
        f"{caption_markup}<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_workflow_progress(
    steps: Sequence[str],
    *,
    completed_through: int | None = None,
    streamlit_api: Any | None = None,
) -> None:
    """Render a compact, non-navigational workflow progress rail.

    ``completed_through`` is a zero-based inclusive index. ``None`` renders
    neutral numbered instructions with no completion claim. The component is
    presentation-only: callers remain responsible for deriving workflow state.
    """

    if not steps:
        raise ValueError("Workflow progress requires at least one step")
    if completed_through is not None and (
        completed_through < -1 or completed_through >= len(steps)
    ):
        raise ValueError("Workflow progress index is outside the supplied steps")

    parts = ['<div class="ws-workflow-progress">']
    for index, label in enumerate(steps):
        if completed_through is None:
            state = "neutral"
        else:
            state = "done" if index <= completed_through else "pending"
        if completed_through is not None and index == min(completed_through + 1, len(steps) - 1):
            state = "current" if completed_through < len(steps) - 1 else "done"
        parts.append(
            f'<div class="ws-workflow-step" data-state="{state}">'
            f'<span class="ws-workflow-number">{index + 1}</span>'
            f'<span>{html.escape(str(label))}</span></div>'
        )
        if index < len(steps) - 1:
            line_state = (
                "neutral"
                if completed_through is None
                else ("done" if index < completed_through else "pending")
            )
            parts.append(
                f'<div class="ws-workflow-line" data-state="{line_state}"></div>'
            )
    parts.append("</div>")
    _api(streamlit_api).markdown("".join(parts), unsafe_allow_html=True)


def render_subview_nav(
    label: str,
    options: Sequence[str],
    *,
    key: str,
    default: str | None = None,
    current: str | None = None,
    streamlit_api: Any | None = None,
) -> str:
    """Render a keyed single-select segmented control with a radio fallback.

    A102: presentation only. ``current`` is the selection the owning controller
    resolved (see :func:`ui.navigation.resolve_subview_selection`); this helper
    reads and writes no application state of its own. ``key`` is passed through
    to the widget, which is Streamlit's own bookkeeping, not a state read here.
    The chosen option is returned to the controller.
    """

    api = _api(streamlit_api)
    choices = list(options)
    if not choices:
        raise ValueError("Workspace subnavigation requires at least one option")
    if default is None:
        default = choices[0]
    if default not in choices:
        raise ValueError("Workspace subnavigation default must be an option")
    fallback = current if current in choices else default

    segmented = getattr(api, "segmented_control", None)
    if callable(segmented):
        selected = segmented(
            label,
            options=choices,
            key=key,
            label_visibility="collapsed",
        )
        return selected if selected in choices else fallback

    return api.radio(
        label,
        options=choices,
        horizontal=True,
        key=key,
        label_visibility="collapsed",
    )

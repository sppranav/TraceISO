"""Shared UI helpers for the Uncertainty tab.

Additive, fallback-safe primitives shared by the Configure / Results /
Diagnostics subviews. Nothing here changes domain math; the only domain
dependency is the read-only runtime-budget cache used for UI aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import streamlit as st

from config.settings import UncertaintyConfig
from domain.uncertainty.runtime import RuntimeUncertaintyMap, lookup_runtime_budget
from ui.components.workspace_ui import render_subview_nav as render_workspace_subview_nav
from ui.navigation import resolve_subview_selection
from ui.runtime_budget_cache import build_cached_runtime_uncertainty_map
from ui.tabs.uncertainty_sections.ui_context import (
    DiagnosticSection,
    UncertaintyUiContext,
)
from ui.utils import get_cycle_ranges, render_runtime_budget_error

# Session keys shared with ui.tabs.uncertainty. Kept here so every subview reads
# and writes the same routing state.
_SUBVIEW_KEY = "uncertainty_subview"
_SUBVIEW_RESET_TOKEN_KEY = "_uncertainty_subview_reset_token"
_SUBVIEWS = ("Configure", "Results", "Diagnostics")

_PENDING_SUBVIEW_KEY = "_uncertainty_pending_subview"
_PENDING_DIAGNOSTIC_CONTRIBUTOR_KEY = "_uncertainty_pending_diagnostic_contributor"
_PENDING_DIAGNOSTIC_SECTION_KEY = "_uncertainty_pending_diagnostic_section"
_DIAGNOSTIC_SELECT_KEY = "uncertainty_diagnostic_select"


# Navigation -----------------------------------------------------------------


def render_subview_nav(
    options: Sequence[str],
    key: str,
    default: Optional[str] = None,
) -> str:
    """Render the top-level subview switcher.

    Uses ``st.segmented_control`` when the installed Streamlit exposes it and
    falls back to ``st.radio`` otherwise. The selection is stored under ``key``
    so :func:`request_uncertainty_navigation` can drive it programmatically and
    so the reset-token logic in ``ui.tabs.uncertainty`` keeps working.
    """
    return render_workspace_subview_nav(
        "Uncertainty view",
        options,
        key=key,
        default=default,
        current=resolve_subview_selection(
            key=key, options=options, default=default, session_state=st.session_state,
        ),
        streamlit_api=st,
    )


def segmented_choice(
    label: str,
    options: Sequence[str],
    *,
    format_func=None,
    index: int = 0,
    key: Optional[str] = None,
    help: Optional[str] = None,
    horizontal: bool = True,
) -> str:
    """A single-select control: ``st.segmented_control`` with radio fallback.

    Preserves the caller's ``key`` so existing session-state dependent logic
    keeps working. Mirrors ``st.radio``'s ``index`` semantics for the segmented
    control's ``default`` while respecting any value already in session state.
    """
    options = list(options)
    default = options[index] if 0 <= index < len(options) else (
        options[0] if options else None
    )

    segmented = getattr(st, "segmented_control", None)
    if callable(segmented):
        # Pre-widget repair (allowed): if a prior deselect left None / a stale
        # value in session state, restore the default before instantiation.
        if key is not None and st.session_state.get(key) not in options:
            if key in st.session_state:
                st.session_state[key] = default
        kwargs = dict(options=options, key=key, help=help)
        if format_func is not None:
            kwargs["format_func"] = format_func
        # Passing default alongside a session-state-backed key raises in
        # Streamlit; only seed the default when the key is absent entirely.
        if key is None or key not in st.session_state:
            kwargs["default"] = default
        selected = segmented(label, **kwargs)
        # Deselect returns None; cannot write session state post-widget, so
        # return the default this run (pre-widget guard fixes the next run).
        return selected if selected in options else default

    return st.radio(
        label,
        options=options,
        format_func=format_func if format_func is not None else (lambda x: x),
        index=index,
        key=key,
        horizontal=horizontal,
        help=help,
    )


def request_uncertainty_navigation(
    target_view: str,
    *,
    contributor: Optional[str] = None,
    section_key: Optional[str] = None,
) -> None:
    """Queue a subview switch (and optional diagnostic preselect) then rerun.

    The target view is written to a **non-widget** pending key, never to the
    subview widget key directly: this helper is called from buttons rendered
    *after* the subview nav widget has already been instantiated this run, and
    Streamlit forbids writing a widget-keyed session value post-instantiation.
    :func:`consume_pending_subview` applies the pending value before the nav
    widget is created on the next run; pending diagnostic contributor/section
    are consumed by :func:`consume_pending_diagnostic`.
    """
    st.session_state[_PENDING_SUBVIEW_KEY] = target_view
    if contributor is not None:
        st.session_state[_PENDING_DIAGNOSTIC_CONTRIBUTOR_KEY] = contributor
    if section_key is not None:
        st.session_state[_PENDING_DIAGNOSTIC_SECTION_KEY] = section_key
    st.rerun()


def consume_pending_subview(key: str, options: Sequence[str]) -> None:
    """Apply a queued subview switch before the nav widget is instantiated.

    Must be called *before* :func:`render_subview_nav` for the top-level
    subview so the write lands pre-widget (allowed). A pending value that is
    not a valid option for this nav is left in place for another nav to
    consume.
    """
    pending = st.session_state.get(_PENDING_SUBVIEW_KEY)
    if pending in options:
        st.session_state.pop(_PENDING_SUBVIEW_KEY, None)
        st.session_state[key] = pending


def diagnostic_section_for_contributor(
    contributor_name: str,
    context: UncertaintyUiContext,
) -> Optional[DiagnosticSection]:
    """Map a contributor name to a diagnostic section in the active context.

    Returns ``None`` when no diagnostic panel covers the contributor so callers
    can omit the jump action rather than route somewhere wrong.
    """
    if not contributor_name:
        return None

    sections = context.diagnostic_sections
    for section in sections:
        if contributor_name in section.contributor_names:
            return section

    # The only alias accepted initially: the Sr SE repeatability row maps onto
    # the standard-repeatability panel.
    if contributor_name == "u_std_repeatability_se":
        for section in sections:
            if "u_std_repeatability" in section.contributor_names:
                return section

    return None


def consume_pending_diagnostic(
    context: UncertaintyUiContext,
) -> Optional[DiagnosticSection]:
    """Resolve a queued jump request to a diagnostic selection.

    Pops the pending contributor/section keys, maps them to one
    :class:`DiagnosticSection` in the current context, and primes the
    diagnostic selectbox/rail. Stale pending values that no longer resolve are
    dropped silently.
    """
    contributor = st.session_state.pop(_PENDING_DIAGNOSTIC_CONTRIBUTOR_KEY, None)
    explicit_section = st.session_state.pop(_PENDING_DIAGNOSTIC_SECTION_KEY, None)

    section: Optional[DiagnosticSection] = None
    if explicit_section is not None:
        section = next(
            (s for s in context.diagnostic_sections if s.key == explicit_section),
            None,
        )
    if section is None and contributor is not None:
        section = diagnostic_section_for_contributor(contributor, context)

    if section is not None:
        st.session_state[_DIAGNOSTIC_SELECT_KEY] = section.key
    return section


# Session aggregate ----------------------------------------------------------


@dataclass(frozen=True)
class SessionBudgetAggregate:
    """Runtime budget map plus derived session metrics.

    Owned by the caller (Results owns it; Diagnostics may build a rail-only
    copy). Threaded down to summary/detail helpers so no subcomponent rebuilds
    the runtime budgets.
    """

    budgets: RuntimeUncertaintyMap
    n_total: int
    n_available: int
    n_insufficient: int
    n_unavailable: int
    n_flagged: int
    n_veff_lt6: int
    u_c_median: float
    U_median: float
    dominant_contributor: Optional[str]
    dominant_share_median: float
    contributor_share_median: dict = field(default_factory=dict)


def session_budget_aggregate(
    samples: Sequence,
    ratio: str,
    u_config: UncertaintyConfig,
    state,
    context: UncertaintyUiContext,
) -> SessionBudgetAggregate:
    """Aggregate runtime budgets across eligible session samples.

    Filters the same eligible samples as Results, ignores invalid scopes for
    medians, and reuses ``summary_detail`` flag behaviour (not synthetic flag
    math). Backed by the shared runtime-budget cache, so calling this does not
    add an extra recompute when the per-sample budgets were already built in the
    same render path.
    """
    # Lazy import avoids a circular import: summary_detail imports shared_ui in
    # later phases for the jump actions.
    from ui.tabs.uncertainty_sections import summary_detail as sd

    eligible = [
        sample
        for sample in samples
        if not getattr(sample, "is_blank", False)
        and not (getattr(sample, "metadata", {}) or {}).get("excluded", False)
    ]
    eligible = sd._filter_uncertainty_budget_samples(eligible, u_config, state)

    empty = SessionBudgetAggregate(
        budgets={},
        n_total=len(eligible),
        n_available=0,
        n_insufficient=0,
        n_unavailable=0,
        n_flagged=0,
        n_veff_lt6=0,
        u_c_median=0.0,
        U_median=0.0,
        dominant_contributor=None,
        dominant_share_median=0.0,
        contributor_share_median={},
    )
    if not eligible:
        return empty

    all_session_samples = list(
        state.result.samples if getattr(state, "has_result", False) else eligible
    )
    cycle_ranges = get_cycle_ranges(state)
    filter_method = (
        state.processing_config.filter_method if state.processing_config else "None"
    )
    filter_threshold = (
        state.processing_config.get_active_filter_threshold()
        if state.processing_config
        else 2.0
    )
    drift_fit_info = (
        state.result.quality_metrics.get("drift_fit_info")
        if getattr(state, "has_result", False)
        and getattr(state.result, "quality_metrics", None)
        else None
    )

    try:
        budgets = build_cached_runtime_uncertainty_map(
            eligible,
            [ratio],
            element_config=state.element_config,
            processing_config=state.processing_config,
            uncertainty_config=u_config,
            all_session_samples=all_session_samples,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
            drift_fit_info=drift_fit_info,
            custom_contributor_library=state.custom_contributor_library,
            profile_defaults=getattr(state, "uncertainty_profile_defaults", None),
        )
    except Exception as exc:
        render_runtime_budget_error("Session uncertainty summary", exc)
        return empty

    is_absolute = u_config.output_mode == "absolute_ratio"
    valid = []
    n_insufficient = 0
    n_unavailable = 0
    for sample in eligible:
        budget = lookup_runtime_budget(budgets, sample, ratio)
        if budget is None:
            continue
        scope = sd._budget_scope(budget)
        if scope == "insufficient_data":
            n_insufficient += 1
            continue
        if scope == "unavailable":
            n_unavailable += 1
            continue
        valid.append(budget)

    if not valid:
        return SessionBudgetAggregate(
            budgets=budgets,
            n_total=len(eligible),
            n_available=0,
            n_insufficient=n_insufficient,
            n_unavailable=n_unavailable,
            n_flagged=0,
            n_veff_lt6=0,
            u_c_median=0.0,
            U_median=0.0,
            dominant_contributor=None,
            dominant_share_median=0.0,
            contributor_share_median={},
        )

    flag_metrics = [
        sd._flag_metric_for_budget(b, is_absolute=is_absolute) for b in valid
    ]
    finite_metrics = [m for m in flag_metrics if np.isfinite(m)]
    median_u = float(np.median(finite_metrics)) if finite_metrics else 0.0

    high_multiple = float(getattr(u_config, "summary_flag_high_multiple", 3.0))
    elevated_multiple = float(getattr(u_config, "summary_flag_elevated_multiple", 2.0))
    n_flagged = 0
    for budget, metric in zip(valid, flag_metrics):
        flag = sd._compute_flag(
            budget,
            median_u,
            flag_metric=metric,
            elevated_multiple=elevated_multiple,
            high_multiple=high_multiple,
        )
        if flag != "OK":
            n_flagged += 1

    uc_values = [float(b.u_combined_abs) for b in valid]
    expanded_values = [float(b.expanded_abs) for b in valid]
    veff_lt6 = sum(
        1
        for b in valid
        if np.isfinite(b.effective_dof) and float(b.effective_dof) < 6.0
    )

    dominant_counts: dict[str, int] = {}
    dominant_shares: list[float] = []
    contributor_shares: dict[str, list[float]] = {}
    for budget in valid:
        name = str(getattr(budget, "dominant_contributor", "") or "")
        if name:
            dominant_counts[name] = dominant_counts.get(name, 0) + 1
        active = [
            c
            for c in budget.contributors
            if c.is_active and c.percentage_contribution > 0
        ]
        if active:
            dominant_shares.append(
                max(c.percentage_contribution for c in active)
            )
        for contributor in active:
            contributor_shares.setdefault(contributor.name, []).append(
                float(contributor.percentage_contribution)
            )

    return _finalize_aggregate(
        budgets=budgets,
        eligible=eligible,
        valid=valid,
        n_insufficient=n_insufficient,
        n_unavailable=n_unavailable,
        n_flagged=n_flagged,
        veff_lt6=veff_lt6,
        uc_values=uc_values,
        expanded_values=expanded_values,
        dominant_counts=dominant_counts,
        dominant_shares=dominant_shares,
        contributor_shares=contributor_shares,
    )


def _finalize_aggregate(
    *,
    budgets,
    eligible,
    valid,
    n_insufficient,
    n_unavailable,
    n_flagged,
    veff_lt6,
    uc_values,
    expanded_values,
    dominant_counts,
    dominant_shares,
    contributor_shares,
) -> "SessionBudgetAggregate":
    dominant_contributor = (
        max(dominant_counts.items(), key=lambda kv: kv[1])[0]
        if dominant_counts
        else None
    )
    contributor_share_median = {
        name: float(np.median(shares))
        for name, shares in contributor_shares.items()
    }
    return SessionBudgetAggregate(
        budgets=budgets,
        n_total=len(eligible),
        n_available=len(valid),
        n_insufficient=n_insufficient,
        n_unavailable=n_unavailable,
        n_flagged=n_flagged,
        n_veff_lt6=veff_lt6,
        u_c_median=float(np.median(uc_values)) if uc_values else 0.0,
        U_median=float(np.median(expanded_values)) if expanded_values else 0.0,
        dominant_contributor=dominant_contributor,
        dominant_share_median=(
            contributor_share_median.get(dominant_contributor, 0.0)
            if dominant_contributor is not None else 0.0
        ),
        contributor_share_median=contributor_share_median,
    )


# Session metric strip / verdict --------------------------------------------


def session_verdict(aggregate: SessionBudgetAggregate) -> tuple[str, str]:
    """Return ``(level, message)`` for the session verdict callout.

    Deterministic rules only. ``level`` is ``"success"`` or ``"warning"``.
    """
    n_invalid = aggregate.n_unavailable + aggregate.n_insufficient
    if aggregate.n_available == 0:
        return (
            "warning",
            "No valid uncertainty budgets for the selected ratio "
            f"({n_invalid} invalid).",
        )
    if aggregate.n_flagged == 0 and n_invalid == 0:
        return (
            "success",
            f"All {aggregate.n_available} budgets within normal session "
            "spread; no invalid budgets.",
        )

    parts: list[str] = []
    if aggregate.n_flagged:
        parts.append(f"{aggregate.n_flagged} flagged budget(s)")
    if aggregate.n_unavailable:
        parts.append(f"{aggregate.n_unavailable} unavailable")
    if aggregate.n_insufficient:
        parts.append(f"{aggregate.n_insufficient} with insufficient data")
    return ("warning", "Review needed: " + ", ".join(parts) + ".")


def render_session_metric_strip(
    aggregate: SessionBudgetAggregate,
    u_config: UncertaintyConfig,
) -> None:
    """Render the session metric strip and verdict callout above the table."""
    from ui.formatting import format_uncertainty
    from ui.tabs.uncertainty_sections.config_controls import (
        contributor_display_label,
    )

    is_absolute = u_config.output_mode == "absolute_ratio"
    uc_unit = "" if is_absolute else "‰"

    dominant_label = "-"
    if aggregate.dominant_contributor:
        dominant_label = contributor_display_label(aggregate.dominant_contributor)

    cols = st.columns(6)
    cols[0].metric(
        "Budgets",
        f"{aggregate.n_available}/{aggregate.n_total}",
    )
    cols[1].metric(
        "Median u_c",
        format_uncertainty(aggregate.u_c_median, unit=uc_unit)
        if aggregate.n_available
        else "-",
    )
    cols[2].metric(
        "Median U",
        format_uncertainty(aggregate.U_median, unit=uc_unit)
        if aggregate.n_available
        else "-",
    )
    cols[3].metric(
        "Dominant",
        dominant_label,
        delta=(
            f"{aggregate.dominant_share_median:.0f}% median share"
            if aggregate.dominant_contributor
            else None
        ),
        delta_color="off",
    )
    cols[4].metric("Flagged", aggregate.n_flagged)
    cols[5].metric("Low ν_eff (<6)", aggregate.n_veff_lt6)

    level, message = session_verdict(aggregate)
    if level == "success":
        st.success(message)
    else:
        st.warning(message)


# Diagnostics rail -----------------------------------------------------------


def diagnostic_section_share(
    section: DiagnosticSection,
    aggregate: Optional[SessionBudgetAggregate],
) -> Optional[float]:
    """Return the section's combined median variance share, or ``None``.

    ``None`` when no aggregate is available or the section advertises no
    contributor names.
    """
    if aggregate is None or not section.contributor_names:
        return None
    total = sum(
        aggregate.contributor_share_median.get(name, 0.0)
        for name in section.contributor_names
    )
    if total <= 0:
        return None
    return min(total, 100.0)


def render_diagnostic_rail(
    sections: Sequence[DiagnosticSection],
    key: str,
    aggregate: Optional[SessionBudgetAggregate] = None,
) -> str:
    """Render the left navigation rail and return the selected section key."""

    def _format(section_key: str) -> str:
        section = next(s for s in sections if s.key == section_key)
        return section.label

    return st.radio(
        "Diagnostic",
        options=[section.key for section in sections],
        format_func=_format,
        key=key,
        label_visibility="collapsed",
    )

"""Repeatability diagnostics for the uncertainty tab."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import UncertaintyConfig, is_russell_law_normalization_engine
from domain.models import ReprodResult
from ui.formatting import format_uncertainty
from ui.tabs.uncertainty_sections.common import (
    _render_compact_summary,
    _render_equation_note_block,
)
from ui.tabs.uncertainty_sections.config_controls import (
    _default_or_current_contributor_enabled,
)
from ui.runtime_budget_cache import get_cached_runtime_budget
from ui.config_plotly import (
    get_plotly_config,
    PLOTLY_AXIS_TITLE_FONT_SIZE,
    PLOTLY_BASE_FONT_SIZE,
)
from ui.theme import get_theme
from ui.utils import format_name, get_cycle_ranges, render_runtime_budget_error


_REPROD_METHOD_OPTIONS = [
    "auto",
    "sd_of_means",
    "loo_cross_validation",
    "robust_mad",
]

_REPROD_METHOD_OPTIONS_WITH_DRIFT = [
    "auto",
    "sd_of_means",
    "loo_cross_validation",
    "drift_residuals",
    "robust_mad",
]

_REPROD_METHOD_LABELS = {
    "auto": "Auto",
    "sd_of_means": "SD of means",
    "loo_cross_validation": "LOO cross-validation",
    "drift_residuals": "Drift residuals",
    "robust_mad": "Robust MAD",
}

# Hollow predictions distinguish modeled values without hiding observations.
_LOO_PREDICTED_FILL = "rgba(0,0,0,0)"

def _state_element_symbol(state) -> str:
    """Return the current element symbol from a UI state-like object."""
    element_symbol = getattr(state, "element_symbol", "") or ""
    if not element_symbol and getattr(state, "element_config", None) is not None:
        element_symbol = getattr(state.element_config, "symbol", "") or ""
    return str(element_symbol)


def _is_standard_repeatability_enabled(
    u_config: UncertaintyConfig,
    state,
) -> bool:
    """Return whether the standard-repeatability diagnostics should render."""
    element_symbol = _state_element_symbol(state)
    engine = (
        u_config.resolve_engine(
            element_symbol,
            processing_config=getattr(state, "processing_config", None),
        )
        if element_symbol
        else getattr(u_config, "engine", "ssb_delta")
    )
    if engine == "internal_normalization":
        return (
            u_config.is_contributor_enabled(
                "u_std_repeatability",
                element_symbol=element_symbol,
            )
            or u_config.is_contributor_enabled(
                "u_std_repeatability_se",
                element_symbol=element_symbol,
            )
        )
    return u_config.is_contributor_enabled(
        "u_std_repeatability",
        element_symbol=element_symbol,
    )


def _resolve_reprod_method_options(*, allow_drift_residuals: bool) -> list[str]:
    """Return the repeatability methods currently valid for the UI."""
    if allow_drift_residuals:
        return list(_REPROD_METHOD_OPTIONS_WITH_DRIFT)
    return list(_REPROD_METHOD_OPTIONS)


def _render_standard_repeatability(
    all_samples: list,
    ratio_name: str,
    u_config: UncertaintyConfig,
    state,
) -> UncertaintyConfig:
    """Render the Standard Repeatability section. Returns (possibly updated) config."""
    if not _is_standard_repeatability_enabled(u_config, state):
        return u_config

    from domain.uncertainty.runtime import resolve_runtime_drift_model_inputs

    st.divider()
    st.subheader("Reference-material repeatability (u_std_repeatability)")

    # Compute a single runtime budget to get the ReprodResult
    stds = [s for s in all_samples if s.is_standard and not s.metadata.get("excluded", False)]
    if not stds:
        st.info("No standards available for repeatability analysis.")
        return u_config

    cycle_ranges = get_cycle_ranges(state)
    filter_method = (
        state.processing_config.filter_method if state.processing_config else "None"
    )
    filter_threshold = (
        state.processing_config.get_active_filter_threshold()
        if state.processing_config
        else 2.0
    )

    # Determine engine: Engine A (internal normalization) locks the reprod
    # strategy to sd_of_means — LOO and drift-residuals are SSB-specific.
    _element_symbol = (
        getattr(state.element_config, "symbol", "") if state.element_config else ""
    )
    _engine = (
        u_config.resolve_engine(
            _element_symbol,
            processing_config=getattr(state, "processing_config", None),
        )
        if _element_symbol else False
    )
    _is_engine_a = _engine == "internal_normalization"
    _is_normalization_engine = (
        is_russell_law_normalization_engine(_engine)
        if isinstance(_engine, str)
        else False
    )
    _drift_fit_info = (
        state.result.quality_metrics.get("drift_fit_info")
        if state.has_result and getattr(state.result, "quality_metrics", None)
        else None
    )
    _drift_residuals_available = False
    if not _is_normalization_engine and state.processing_config is not None:
        _drift_model, _position_extractor = resolve_runtime_drift_model_inputs(
            all_samples,
            state.processing_config,
            _drift_fit_info,
            ratio_name,
        )
        _drift_residuals_available = (
            _drift_model is not None and _position_extractor is not None
        )

    # Read current widget state before computing — avoids one-rerun lag when
    # the user changes method or the drift contributor state in configuration.
    _method_now = st.session_state.get("reprod_method_radio", u_config.reprod_method)
    if _is_normalization_engine:
        _method_now = "sd_of_means"
    elif _method_now == "drift_residuals" and not _drift_residuals_available:
        _method_now = "auto"
    _drift_contributor_name = (
        "u_kappa_drift" if _is_normalization_engine else "u_k5_instrumental_drift"
    )
    _drift_symbol = "kappa_drift" if _is_normalization_engine else "k5"
    _kappa_now = (
        False
        if _is_engine_a
        else _default_or_current_contributor_enabled(
            u_config,
            _drift_contributor_name,
        )
    )
    if _method_now != u_config.reprod_method or _kappa_now != u_config.include_kappa_drift:
        u_config = replace(u_config, reprod_method=_method_now, include_kappa_drift=_kappa_now)

    # Try standards one at a time until we find one that produces a
    # reprod result.  Previously only stds[:1] was tried, so if the
    # first standard lacked the selected ratio the section would fail.
    reprod: Optional[ReprodResult] = None
    reprod_budget = None
    for _std_candidate in stds:
        try:
            _budget = get_cached_runtime_budget(
                _std_candidate,
                ratio_name,
                element_config=state.element_config,
                processing_config=state.processing_config,
                uncertainty_config=u_config,
                all_samples=all_samples,
                cycle_ranges=cycle_ranges,
                filter_method=filter_method,
                filter_threshold=filter_threshold,
                drift_fit_info=_drift_fit_info,
                custom_contributor_library=state.custom_contributor_library,
                profile_defaults=getattr(state, "uncertainty_profile_defaults", None),
            )
        except Exception as exc:
            render_runtime_budget_error("Standard Repeatability", exc)
            return u_config
        if hasattr(_budget, "reprod_result") and _budget.reprod_result is not None:
            reprod = _budget.reprod_result
            reprod_budget = _budget
        if reprod is not None:
            break

    if reprod is None:
        st.info("No repeatability data available for the selected ratio.")
        return u_config

    _engine_a_mode = (
        str(getattr(u_config, "std_repeatability_mode", "sd")).strip().lower()
        if _is_normalization_engine
        else "sd"
    )
    if _engine_a_mode not in {"sd", "se"}:
        _engine_a_mode = "sd"

    if _engine == "pb_tl_external_normalization" and bool(getattr(
        getattr(state.processing_config, "pb_standard_calibration", None), "enabled", False,
    )):
        st.caption(
            "This plot describes Tl-normalized reference-material runs. For calibrated sample "
            "results, calibration-standard mean SE enters through K; this session-scatter "
            "diagnostic is not an additional contribution to the calibrated budget."
        )

    # Normalization workflows take their SD/SE mode from Budget settings.
    if _is_normalization_engine:
        if _engine_a_mode == "se":
            st.caption(
                "Method: SE of standard means (computed as SD/sqrt(n) on the "
                "reference-material session means)."
            )
        else:
            st.caption(
                "Method: SD of reference-material means (full session scatter)."
            )
    else:
        u_config = _render_reprod_method_selector(
            u_config,
            reprod,
            allow_drift_residuals=_drift_residuals_available,
            contributor_name=_drift_contributor_name,
            symbol=_drift_symbol,
        )
        if (
            state.processing_config is not None
            and state.processing_config.drift.enabled
            and state.processing_config.drift.ratio_name == ratio_name
            and not _drift_residuals_available
        ):
            st.caption(
                "Drift residuals require a committed drift fit for this ratio."
            )
    _render_reprod_summary(
        reprod,
        u_config,
        engine_a_mode=_engine_a_mode if _is_normalization_engine else None,
        budget=reprod_budget,
    )

    if reprod.fallback_reason:
        if reprod.requested_method:
            requested_label = _REPROD_METHOD_LABELS.get(
                reprod.requested_method, reprod.requested_method
            )
            st.caption(f"Method requested: {requested_label}")
        st.caption(f"Fallback: {reprod.fallback_reason}")

    # Full-width sequence plot
    _render_sequence_plot(
        reprod,
        ratio_name,
        engine_a_mode=_engine_a_mode if _is_normalization_engine else None,
    )

    # Include/exclude table directly below the plot
    prev_excluded = list(u_config.excluded_standards)
    u_config = _render_std_include_table(reprod, u_config, key_prefix="reprod")
    if sorted(u_config.excluded_standards) != sorted(prev_excluded):
        state.uncertainty_config = u_config
        st.rerun()

    # Secondary information in expanders
    with st.expander("Method equations", expanded=False):
        _render_reprod_equation_block(
            reprod,
            engine_a_mode=_engine_a_mode if _is_normalization_engine else None,
        )

    with st.expander("Per-standard detail", expanded=False):
        _render_reprod_detail_section(reprod)

    with st.expander("Segment details & break controls", expanded=False):
        if reprod.segment_sds:
            _render_segment_cards(reprod)
        u_config = _render_break_controls(
            reprod,
            u_config,
            all_samples,
            ratio_name,
            state,
        )

    return u_config


def _render_reprod_kappa_status(
    u_config: UncertaintyConfig,
    reprod: ReprodResult,
    *,
    contributor_name: str,
    symbol: str,
) -> None:
    """Read-only drift status line used when the method selector is hidden."""
    include_kd = _default_or_current_contributor_enabled(u_config, contributor_name)
    st.caption(
        f"Instrumental drift ({symbol}): controlled in Uncertainty Configuration -> Budget contributors."
    )
    if reprod.kappa_drift_permil > 0:
        if u_config.output_mode == "absolute_ratio":
            _ratio_mean = (
                float(np.mean(reprod.std_means[reprod.std_included]))
                if any(reprod.std_included) else 1.0
            )
            _kd_abs = reprod.kappa_drift_permil / 1000.0 * _ratio_mean
            st.caption(f"{symbol} = {_kd_abs:.8f} (abs)")
        else:
            st.caption(f"{symbol} = {reprod.kappa_drift_permil:.3f} \u2030")
    elif include_kd:
        st.caption(f"{symbol} = 0 (suppressed: drift/SSB active or < 2 standards)")
    else:
        st.caption(f"{symbol} excluded from budget")


def _render_reprod_method_selector(
    u_config: UncertaintyConfig,
    reprod: ReprodResult,
    *,
    allow_drift_residuals: bool,
    contributor_name: str,
    symbol: str,
) -> UncertaintyConfig:
    """Radio buttons for repeatability strategies plus read-only drift status."""
    # Resolve auto for display
    resolved = u_config.resolve_reprod_method(
        enable_ssb=u_config.enable_ssb,
        enable_delta=u_config.enable_delta,
    )
    resolved_label = _REPROD_METHOD_LABELS.get(resolved, resolved)

    def _format_method(x: str) -> str:
        label = _REPROD_METHOD_LABELS.get(x, x)
        if x == "auto":
            return f"{label} \u2192 {resolved_label}"
        return label

    col_method, col_kappa = st.columns([3, 2])

    with col_method:
        method_options = _resolve_reprod_method_options(
            allow_drift_residuals=allow_drift_residuals,
        )
        current_method = (
            u_config.reprod_method
            if u_config.reprod_method in method_options
            else "auto"
        )
        current_idx = (
            method_options.index(current_method)
            if current_method in method_options
            else 0
        )
        method = st.radio(
            "Method",
            options=method_options,
            format_func=_format_method,
            index=current_idx,
            key="reprod_method_radio",
            horizontal=True,
        )
        if method == "auto":
            _reason = "SSB active \u2192 LOO" if u_config.enable_ssb else "SSB off \u2192 SD of means"
            st.caption(f"Auto-selected: {_reason}")

    with col_kappa:
        include_kd = _default_or_current_contributor_enabled(
            u_config,
            contributor_name,
        )
        st.caption(f"Instrumental drift ({symbol})")
        st.caption(
            "Controlled in Uncertainty Configuration -> Budget contributors."
        )
        if reprod.kappa_drift_permil > 0:
            if u_config.output_mode == "absolute_ratio":
                # Convert permil to absolute: abs = permil/1000 * ratio_mean
                _ratio_mean = float(np.mean(reprod.std_means[reprod.std_included])) if any(reprod.std_included) else 1.0
                _kd_abs = reprod.kappa_drift_permil / 1000.0 * _ratio_mean
                st.caption(f"{symbol} = {_kd_abs:.8f} (abs)")
            else:
                st.caption(f"{symbol} = {reprod.kappa_drift_permil:.3f} \u2030")
        elif include_kd:
            st.caption(f"{symbol} = 0 (suppressed: drift/SSB active or < 2 standards)")
        else:
            st.caption("Excluded from budget")

    if method != u_config.reprod_method or include_kd != u_config.include_kappa_drift:
        u_config = replace(u_config, reprod_method=method, include_kappa_drift=include_kd)

    return u_config


def _compute_engine_a_repeatability_display(
    reprod: ReprodResult,
    std_repeatability_mode: str,
    budget=None,
) -> Tuple[float, float, float, str]:
    """Return Engine A display values according to SD/SE mode."""
    mode = str(std_repeatability_mode).strip().lower()
    if mode not in {"sd", "se"}:
        mode = "sd"

    contributor_name = (
        "u_std_repeatability_se" if mode == "se" else "u_std_repeatability"
    )
    # Pb uses one contributor key for both modes; its budget already contains
    # the SD-to-SE conversion, so do not divide its value a second time.
    if getattr(budget, "engine", "") == "pb_tl_external_normalization":
        contributor_name = "u_std_repeatability"
    contributor = _find_active_budget_contributor(budget, contributor_name)
    if contributor is not None:
        method_label = "SE of means" if mode == "se" else "SD of means"
        return (
            float(contributor.value_abs),
            float(contributor.value_rel_permil),
            float(contributor.degrees_of_freedom),
            method_label,
        )

    included_count = int(sum(1 for included in reprod.std_included if included))
    n_for_se = included_count if included_count > 1 else 1
    ratio_basis = float(np.mean(reprod.std_means[reprod.std_included])) if any(reprod.std_included) else 0.0

    if mode == "se":
        u_abs = float(reprod.u_std_repeatability_abs) / float(np.sqrt(n_for_se))
        dof = float(n_for_se - 1) if n_for_se > 1 else float("inf")
        method_label = "SE of means"
    else:
        u_abs = float(reprod.u_std_repeatability_abs)
        dof = float(reprod.degrees_of_freedom)
        method_label = "SD of means"

    u_rel = (u_abs / ratio_basis) * 1000.0 if ratio_basis and u_abs > 0 else 0.0
    return u_abs, u_rel, dof, method_label


def _find_active_budget_contributor(budget, name: str):
    """Return an active contributor from a runtime budget, if present."""
    if budget is None:
        return None
    if hasattr(budget, "_find_contributor"):
        contributor = budget._find_contributor(name)
    else:
        contributor = next(
            (item for item in getattr(budget, "contributors", []) if item.name == name),
            None,
        )
    if contributor is None or not getattr(contributor, "is_active", False):
        return None
    return contributor


def _render_reprod_summary(
    reprod: ReprodResult,
    u_config: UncertaintyConfig,
    *,
    engine_a_mode: Optional[str] = None,
    budget=None,
) -> None:
    """Render the repeatability headline results as compact text."""
    method_used = _REPROD_METHOD_LABELS.get(reprod.method, reprod.method)
    dof_value: float = float(reprod.degrees_of_freedom)
    u_abs_value = float(reprod.u_std_repeatability_abs)
    u_rel_value = float(reprod.u_std_repeatability_rel_permil)

    if engine_a_mode is not None:
        u_abs_value, u_rel_value, dof_value, method_used = _compute_engine_a_repeatability_display(
            reprod,
            engine_a_mode,
            budget,
        )

    if u_config.output_mode == "absolute_ratio":
        u_value = format_uncertainty(u_abs_value)
    else:
        u_value = format_uncertainty(u_rel_value, unit="\u2030")

    dof_text = "∞" if not np.isfinite(dof_value) else str(int(round(dof_value)))

    _render_compact_summary(
        [
            ("u_std_repeatability", u_value),
            ("Degrees of freedom \u03bd", dof_text),
            ("Method used", method_used),
        ]
    )


def _render_sequence_plot(
    reprod: ReprodResult,
    ratio_name: str,
    *,
    engine_a_mode: Optional[str] = None,
) -> None:
    """Render the standard-sequence plot for the active u_std_repeatability method."""
    fig = _build_reprod_sequence_figure(reprod, ratio_name, engine_a_mode=engine_a_mode)
    st.plotly_chart(fig, width="stretch", config=get_plotly_config())
    if reprod.method == "sd_of_means":
        st.caption(
            "Diamonds: included reference-material run means. Black line: their mean. "
            "Darker band: mean ±1 SD; full lighter band: mean ±2 SD. "
            "These bands describe run-to-run scatter, including when SE is selected for the budget; "
            "they are not automatic outlier-rejection limits."
        )


def _render_reprod_equation_block(
    reprod: ReprodResult,
    *,
    engine_a_mode: Optional[str] = None,
) -> None:
    """Render a short method-equation note below the sequence plot."""
    payload = _get_reprod_equation_payload(reprod, engine_a_mode=engine_a_mode)
    if payload is None:
        return

    _render_equation_note_block(
        payload["lines"],
        title=str(payload["title"]),
        note=str(payload["note"]),
    )


def _get_reprod_equation_payload(
    reprod: ReprodResult,
    *,
    engine_a_mode: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    """Return display text for the active repeatability method."""
    if reprod.method == "sd_of_means" and str(engine_a_mode).strip().lower() == "se":
        return {
            "title": "SE of Means Equations",
            "lines": [
                r"\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i",
                r"s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(y_i-\bar{y})^2}",
                r"u_{\mathrm{reprod}} = \mathrm{SE} = \frac{s}{\sqrt{n}}",
            ],
            "note": (
                "For internal normalisation, SD of included SRM means is first "
                "computed, then converted to SE by dividing by sqrt(n), where n is "
                "the number of included SRM runs. The sequence plot still shows the "
                "same SRM means and SD guide bands for context."
            ),
        }
    if reprod.method == "loo_cross_validation":
        return {
            "title": "LOO Cross-Validation Equations",
            "lines": [
                r"\hat{y}_i = y_L + (y_R-y_L)\frac{x_i-x_L}{x_R-x_L}",
                r"r_i = y_i - \hat{y}_i",
                r"u_{\mathrm{reprod}} = s(r_i) = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(r_i-\bar{r})^2}",
            ],
            "note": (
                "For each included standard, the app leaves that standard out and predicts "
                "it from the nearest included standards on the left and right. Here x is "
                "the standard sequence position (currently run number), y is the observed "
                "standard mean, y-hat is the interpolated prediction, and r is the "
                "prediction residual. The first and last included standards use the same "
                "linear formula in extrapolation mode with the two nearest included "
                "standards on the available side. If fewer than 3 included standards "
                "remain, the app falls back to SD of means."
            ),
        }
    if reprod.method == "sd_of_means":
        return {
            "title": "SD of Means Equations",
            "lines": [
                r"\bar{y} = \frac{1}{n}\sum_{i=1}^{n} y_i",
                r"s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(y_i-\bar{y})^2}",
                r"u_{\mathrm{reprod}} = s",
            ],
            "note": (
                "The center line shows the global mean of the included standard means. "
                "The shaded bands show the global mean ± 1 SD and ± 2 SD for visual "
                "context. When segmenting is active, the reported budget value may still "
                "be the segment-pooled SD even though the guide bands use the global "
                "included-standard mean and SD."
            ),
        }
    if reprod.method == "robust_mad":
        return {
            "title": "Robust MAD Equations",
            "lines": [
                r"\tilde{y} = \mathrm{median}(y_i)",
                r"\mathrm{MAD} = \mathrm{median}\left(\left|y_i-\tilde{y}\right|\right)",
                r"u_{\mathrm{reprod}} = 1.4826 \times \mathrm{MAD}",
            ],
            "note": (
                "The center line shows the global median of the included standard means. "
                "The shaded band shows the global median ± scaled MAD, where scaled MAD "
                "equals 1.4826 × MAD. When segmenting is active, the plotted guide band "
                "is still global, while the reported budget may be pooled across segments."
            ),
        }
    return None


def _get_included_standard_values(reprod: ReprodResult) -> np.ndarray:
    """Return included standard means as a float array."""
    if len(reprod.std_means) == 0:
        return np.array([], dtype=float)
    included_mask = np.asarray(reprod.std_included, dtype=bool)
    return np.asarray(reprod.std_means, dtype=float)[included_mask]


def _scaled_mad(values: np.ndarray) -> float:
    """Return 1.4826 × MAD for *values*."""
    if len(values) == 0:
        return 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad


def _add_reprod_reference_guides(fig, reprod: ReprodResult) -> None:
    """Add center lines and guide bands for mean/median-based methods."""
    included_vals = _get_included_standard_values(reprod)
    if len(included_vals) == 0:
        return

    palette = get_theme().palette
    center_line_color = palette.figure_ink

    if reprod.method == "sd_of_means":
        center = float(np.mean(included_vals))
        spread = float(np.std(included_vals, ddof=1)) if len(included_vals) > 1 else 0.0
        if spread > 0:
            fig.add_hrect(
                y0=center - 2.0 * spread,
                y1=center + 2.0 * spread,
                fillcolor=palette.sample_std_light,
                line_width=0,
                layer="below",
            )
            fig.add_hrect(
                y0=center - spread,
                y1=center + spread,
                fillcolor=palette.sample_std_light,
                line_width=0,
                layer="below",
            )
        fig.add_hline(
            y=center,
            line_color=center_line_color,
            line_width=2.0,
        )

    elif reprod.method == "robust_mad":
        center = float(np.median(included_vals))
        spread = _scaled_mad(included_vals)
        if spread > 0:
            fig.add_hrect(
                y0=center - spread,
                y1=center + spread,
                fillcolor=palette.guide_bounds_light,
                line_width=0,
                layer="below",
            )
        fig.add_hline(
            y=center,
            line_color=center_line_color,
            line_width=2.0,
        )


def _build_reprod_sequence_figure(
    reprod: ReprodResult,
    ratio_name: str,
    *,
    engine_a_mode: Optional[str] = None,
):
    """Build the standard-sequence figure for repeatability diagnostics."""
    import plotly.graph_objects as go

    theme = get_theme()
    palette = theme.palette
    fig = go.Figure()

    incl_x, incl_y, incl_text = [], [], []
    excl_x, excl_y, excl_text = [], [], []

    for i, name in enumerate(reprod.std_names):
        pos = float(reprod.std_positions[i])
        mean = float(reprod.std_means[i])
        seg = reprod.std_segments[i] if reprod.std_segments else 1
        hover = f"{name}<br>Mean: {mean:.6f}<br>Segment: {seg}"

        if reprod.std_included[i]:
            incl_x.append(pos)
            incl_y.append(mean)
            incl_text.append(hover)
        else:
            excl_x.append(pos)
            excl_y.append(mean)
            excl_text.append(hover)

    if incl_x:
        fig.add_trace(go.Scatter(
            x=incl_x,
            y=incl_y,
            mode="markers",
            marker=dict(size=12, color=palette.sample_std, symbol="diamond"),
            name="Standards (included)",
            hovertemplate="%{text}<extra></extra>",
            text=incl_text,
        ))

    if excl_x:
        fig.add_trace(go.Scatter(
            x=excl_x,
            y=excl_y,
            mode="markers",
            marker=dict(size=12, color=palette.layer_excluded, symbol="x"),
            name="Standards (excluded)",
            hovertemplate="%{text}<extra></extra>",
            text=excl_text,
        ))

    _add_reprod_reference_guides(fig, reprod)

    if reprod.predicted_values is not None:
        pred = reprod.predicted_values
        pred_x, pred_y, pred_text = [], [], []
        for i in range(len(pred)):
            if not np.isnan(pred[i]):
                pred_x.append(float(reprod.std_positions[i]))
                pred_y.append(float(pred[i]))
                actual = float(reprod.std_means[i])
                residual = actual - float(pred[i])
                pred_text.append(
                    f"{reprod.std_names[i]}<br>"
                    f"Predicted: {float(pred[i]):.6f}<br>"
                    f"Actual: {actual:.6f}<br>"
                    f"Residual: {residual:.6f}"
                )
        if pred_x:
            fig.add_trace(go.Scatter(
                x=pred_x,
                y=pred_y,
                mode="markers",
                marker=dict(
                    size=11,
                    color=_LOO_PREDICTED_FILL,
                    line=dict(color=palette.layer_drift, width=2.3),
                    symbol="diamond",
                ),
                name="LOO predicted",
                hovertemplate="%{text}<extra></extra>",
                text=pred_text,
            ))

    if reprod.std_segments:
        segments = reprod.std_segments
        positions = reprod.std_positions
        for i in range(len(segments) - 1):
            if segments[i] != segments[i + 1]:
                boundary_x = (float(positions[i]) + float(positions[i + 1])) / 2.0
                fig.add_vline(
                    x=boundary_x,
                    line_dash="dash",
                    line_color=palette.guide_bounds,
                    opacity=0.7,
                    annotation_text=f"Seg {segments[i]}|{segments[i+1]}",
                    annotation_position="top",
                )

    method_label = _REPROD_METHOD_LABELS.get(reprod.method, reprod.method)
    if reprod.method == "sd_of_means" and str(engine_a_mode).strip().lower() == "se":
        method_label = "SE of means (SD/sqrt(n))"
    fig.update_layout(
        title=f"Standard Sequence ({method_label})",
        xaxis_title="Sequence Number",
        yaxis_title=format_name(ratio_name),
        height=400,
        margin=dict(l=80, r=20, t=50, b=70),
        legend=dict(
            orientation="v",
            x=0.99,
            y=0.99,
            xanchor="right",
            yanchor="top",
            borderwidth=0,
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


def _build_loo_detail_dataframe(
    reprod: ReprodResult,
) -> Tuple[pd.DataFrame, Dict[str, float | int | None]]:
    """Build per-standard LOO detail rows and residual summary statistics."""
    if reprod.predicted_values is None:
        return pd.DataFrame(), {
            "n_residuals": 0,
            "mean_residual": None,
            "residual_sd": None,
        }

    rows = []
    for i, included in enumerate(reprod.std_included):
        if not included:
            continue

        predicted = float(reprod.predicted_values[i])
        if np.isnan(predicted):
            continue

        sequence = float(reprod.std_positions[i])
        actual = float(reprod.std_means[i])
        residual = actual - predicted
        rows.append(
            {
                "Standard": reprod.std_names[i],
                "Sequence Number": int(sequence) if sequence.is_integer() else sequence,
                "Segment": reprod.std_segments[i] if reprod.std_segments else 1,
                "Actual mean": actual,
                "LOO predicted": predicted,
                "Residual (actual - predicted)": residual,
            }
        )

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        return detail_df, {
            "n_residuals": 0,
            "mean_residual": None,
            "residual_sd": None,
        }

    residuals = detail_df["Residual (actual - predicted)"].to_numpy(dtype=float)
    residual_sd = float(np.std(residuals, ddof=1)) if len(residuals) >= 2 else None
    return detail_df, {
        "n_residuals": len(residuals),
        "mean_residual": float(np.mean(residuals)),
        "residual_sd": residual_sd,
    }


def _iter_included_standard_rows(reprod: ReprodResult) -> list[dict[str, object]]:
    """Return one row per included standard for method-detail tables."""
    rows: list[dict[str, object]] = []
    for i, included in enumerate(reprod.std_included):
        if not included:
            continue

        sequence = float(reprod.std_positions[i])
        rows.append(
            {
                "Standard": reprod.std_names[i],
                "Sequence Number": int(sequence) if sequence.is_integer() else sequence,
                "Segment": reprod.std_segments[i] if reprod.std_segments else 1,
                "Mean": float(reprod.std_means[i]),
            }
        )
    return rows


def _build_sd_of_means_detail_dataframe(
    reprod: ReprodResult,
) -> Tuple[pd.DataFrame, Dict[str, float | int | None]]:
    """Build the SD-of-means detail table from included standards."""
    rows = _iter_included_standard_rows(reprod)
    if not rows:
        return pd.DataFrame(), {
            "n_included": 0,
            "overall_mean": None,
            "spread_value": None,
            "spread_label": "SD of means (ddof=1)",
        }

    detail_df = pd.DataFrame(rows)
    overall_mean = float(np.mean(detail_df["Mean"].to_numpy(dtype=float)))
    unique_segments = sorted(detail_df["Segment"].unique().tolist())

    seg_means = (
        detail_df.groupby("Segment", sort=True)["Mean"]
        .mean()
        .astype(float)
        .to_dict()
    )
    detail_df["Reference mean"] = detail_df["Segment"].map(seg_means)
    detail_df["Deviation (mean - reference)"] = (
        detail_df["Mean"] - detail_df["Reference mean"]
    )

    spread_label = "SD of means (ddof=1)" if len(unique_segments) == 1 else "Segment-pooled SD"
    return detail_df, {
        "n_included": len(detail_df),
        "overall_mean": overall_mean,
        "spread_value": float(reprod.u_std_repeatability_abs),
        "spread_label": spread_label,
    }


def _build_robust_mad_detail_dataframe(
    reprod: ReprodResult,
) -> Tuple[pd.DataFrame, Dict[str, float | int | None]]:
    """Build the robust-MAD detail table from included standards."""
    rows = _iter_included_standard_rows(reprod)
    if not rows:
        return pd.DataFrame(), {
            "n_included": 0,
            "overall_median": None,
            "spread_value": None,
            "spread_label": "1.4826 × MAD",
        }

    detail_df = pd.DataFrame(rows)
    overall_median = float(np.median(detail_df["Mean"].to_numpy(dtype=float)))
    unique_segments = sorted(detail_df["Segment"].unique().tolist())

    seg_medians = (
        detail_df.groupby("Segment", sort=True)["Mean"]
        .median()
        .astype(float)
        .to_dict()
    )
    detail_df["Reference median"] = detail_df["Segment"].map(seg_medians)
    detail_df["|Mean - median|"] = (
        detail_df["Mean"] - detail_df["Reference median"]
    ).abs()

    spread_label = "1.4826 × MAD" if len(unique_segments) == 1 else "Segment-pooled robust SD"
    return detail_df, {
        "n_included": len(detail_df),
        "overall_median": overall_median,
        "spread_value": float(reprod.u_std_repeatability_abs),
        "spread_label": spread_label,
    }


def _format_reprod_detail_dataframe(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Format repeatability detail tables for compact display."""
    if detail_df.empty:
        return detail_df
    return detail_df.copy()


def _reprod_detail_column_config(detail_df: pd.DataFrame) -> dict:
    """NumberColumn config for all numeric columns in a reprod detail table."""
    text_cols = {"Standard", "Sequence Number", "Segment"}
    return {
        col: st.column_config.NumberColumn(col, format="%.8f")
        for col in detail_df.columns
        if col not in text_cols and pd.api.types.is_numeric_dtype(detail_df[col])
    }


def _render_reprod_detail_section(reprod: ReprodResult) -> None:
    """Render the method-specific repeatability detail summary and table."""
    captions: list[str]
    summary_items: list[tuple[str, str]]

    if reprod.method == "loo_cross_validation":
        title = "LOO Cross-validation Detail"
        detail_df, stats = _build_loo_detail_dataframe(reprod)
        if detail_df.empty:
            return
        summary_items = [
            ("Residual count", str(int(stats["n_residuals"] or 0))),
            (
                "Mean residual",
                "n/a" if stats["mean_residual"] is None else f"{float(stats['mean_residual']):.8f}",
            ),
            (
                "Residual SD (ddof=1)",
                "n/a" if stats["residual_sd"] is None else f"{float(stats['residual_sd']):.8f}",
            ),
        ]
        captions = [
            "For LOO, u_std_repeatability is the sample SD of the residuals "
            "(actual - predicted, ddof = 1). "
            "The mean residual is shown only as a bias check.",
        ]
        if stats["residual_sd"] is not None and not np.isclose(stats["residual_sd"], reprod.u_std_repeatability_abs):
            captions.append(
                "The residual SD shown here differs from the reported u_std_repeatability above because "
                "the calculation fell back to another method for the final uncertainty value."
            )
    elif reprod.method == "robust_mad":
        title = "Robust MAD Detail"
        detail_df, stats = _build_robust_mad_detail_dataframe(reprod)
        if detail_df.empty:
            return
        summary_items = [
            ("Included standards", str(int(stats["n_included"] or 0))),
            (
                "Overall median",
                "n/a" if stats["overall_median"] is None else f"{float(stats['overall_median']):.8f}",
            ),
            (
                str(stats["spread_label"]),
                "n/a" if stats["spread_value"] is None else format_uncertainty(float(stats["spread_value"])),
            ),
        ]
        captions = [
            "For Robust MAD, u_std_repeatability is 1.4826 × median(|x - median(x)|).",
        ]
        if detail_df["Segment"].nunique() > 1:
            captions.append(
                "Rows are referenced to the segment median; the reported u_std_repeatability above is pooled across segments."
            )
        if (detail_df["Segment"].value_counts() < 2).any():
            captions.append(
                "Segments with fewer than 2 standards inherit the session-wide robust MAD estimate."
            )
    else:
        title = "SD of Means Detail"
        detail_df, stats = _build_sd_of_means_detail_dataframe(reprod)
        if detail_df.empty:
            return
        summary_items = [
            ("Included standards", str(int(stats["n_included"] or 0))),
            (
                "Overall mean",
                "n/a" if stats["overall_mean"] is None else f"{float(stats['overall_mean']):.8f}",
            ),
            (
                str(stats["spread_label"]),
                "n/a" if stats["spread_value"] is None else format_uncertainty(float(stats["spread_value"])),
            ),
        ]
        captions = [
            "For SD of means, u_std_repeatability is the sample SD of the included standard means (ddof = 1).",
        ]
        if detail_df["Segment"].nunique() > 1:
            captions.append(
                "Rows are referenced to the segment mean; the reported u_std_repeatability above is pooled across segments."
            )
        if (detail_df["Segment"].value_counts() < 2).any():
            captions.append(
                "Segments with fewer than 2 standards inherit the session-wide SD estimate."
            )

    st.markdown(f"**{title}**")
    _render_compact_summary(summary_items)
    for caption in captions:
        st.caption(caption)
    _reprod_df = _format_reprod_detail_dataframe(detail_df)
    st.dataframe(
        _reprod_df,
        width="stretch",
        hide_index=True,
        column_config=_reprod_detail_column_config(_reprod_df),
    )


def _render_segment_cards(reprod: ReprodResult) -> None:
    """Per-segment metric cards showing SD, DoF, N standards."""
    seg_sds = reprod.segment_sds or {}
    seg_dofs = reprod.segment_dofs or {}
    seg_n = reprod.segment_n_stds or {}

    unique_segs = sorted(seg_sds.keys())
    if not unique_segs:
        return

    cols = st.columns(min(len(unique_segs), 4))
    for idx, seg in enumerate(unique_segs):
        col = cols[idx % len(cols)]
        n = seg_n.get(seg, 0)
        sd = seg_sds.get(seg, 0.0)
        dof = seg_dofs.get(seg, 0)

        with col:
            st.markdown(f"**Segment {seg}**")
            if n < 2:
                st.warning(f"N = {n} (need \u2265 2)")
            else:
                st.caption(f"N = {n} standards")
            st.metric("SD", format_uncertainty(sd) if sd > 0 else "\u2014")
            st.caption(f"\u03bd = {dof}")


def _render_std_include_table(
    reprod: ReprodResult,
    u_config: UncertaintyConfig,
    *,
    key_prefix: str = "reprod",
    expander_label: str = "Standard Include/Exclude",
) -> UncertaintyConfig:
    """Compact checkbox table for standard include/exclude."""
    if not reprod.std_names:
        return u_config

    # What each displayed row actually refers to. A row's label is a display
    # name — repeated across observations in alternating mode, and a joined
    # "A+B" in block mode — so an exclusion written from the label alone
    # cannot address one observation, and in block mode addressed nothing at
    # all. The identities travel beside the table and never reach the editor.
    row_identities = _row_exclusion_identities(reprod)

    with st.expander(expander_label, expanded=False):
        rows = []
        for i, name in enumerate(reprod.std_names):
            seg = reprod.std_segments[i] if reprod.std_segments else 1
            mean = float(reprod.std_means[i]) if i < len(reprod.std_means) else 0.0
            rows.append({
                "Include": reprod.std_included[i],
                "Standard": name,
                "Mean": round(mean, 6),
                "Segment": seg,
            })

        df = pd.DataFrame(rows)

        editor_key = f"{key_prefix}_std_editor"
        form_key = f"{key_prefix}_std_include_form"

        with st.form(form_key, clear_on_submit=False):
            st.caption("Edits are staged until you click Apply.")
            edited = st.data_editor(
                df,
                column_config={
                    "Include": st.column_config.CheckboxColumn("Include", default=True),
                    "Standard": st.column_config.TextColumn("Standard", disabled=True),
                    "Mean": st.column_config.NumberColumn("Mean", format="%.6f", disabled=True),
                    "Segment": st.column_config.NumberColumn("Segment", disabled=True),
                },
                width="stretch",
                hide_index=True,
                key=editor_key,
            )
            col_apply, col_discard = st.columns(2)
            with col_apply:
                apply_clicked = st.form_submit_button(
                    "Apply", type="primary", width="stretch",
                )
            with col_discard:
                discard_clicked = st.form_submit_button(
                    "Discard", width="stretch",
                )

        if discard_clicked:
            if editor_key in st.session_state:
                del st.session_state[editor_key]
            st.rerun()

        if apply_clicked:
            new_excluded: List[str] = []
            for position, (_, row) in enumerate(edited.iterrows()):
                if row["Include"]:
                    continue
                if position < len(row_identities) and row_identities[position]:
                    new_excluded.extend(row_identities[position])
                else:
                    # No identities available (a ReprodResult built by an older
                    # caller); fall back to the legacy display-name reference.
                    new_excluded.append(row["Standard"])
            if sorted(new_excluded) != sorted(u_config.excluded_standards):
                u_config = replace(u_config, excluded_standards=new_excluded)

    return u_config


def _row_exclusion_identities(reprod: ReprodResult) -> List[List[str]]:
    """Observation identities behind each row of the include/exclude table."""
    identities = getattr(reprod, "std_identities", None) or []
    return [
        [str(member) for member in (identities[i] if i < len(identities) else []) if member]
        for i in range(len(reprod.std_names))
    ]


def _render_break_controls(
    reprod: ReprodResult,
    u_config: UncertaintyConfig,
    all_samples: list,
    ratio_name: str,
    state,
    *,
    key_prefix: str = "reprod",
) -> UncertaintyConfig:
    """Add break and Reset buttons."""
    from domain.uncertainty.reprod_detection import assign_segments_from_breaks

    if u_config.segment_assignments:
        _, empty_segments, thin_segments = _validate_segment_assignments(
            all_samples,
            u_config.segment_assignments,
            excluded_standards=u_config.excluded_standards,
        )
        if empty_segments:
            st.warning(
                "Current repeatability segmentation contains no included standards in "
                f"{_format_segment_list(empty_segments)}."
            )
        if thin_segments:
            st.warning(
                "Current repeatability segmentation has fewer than 2 included standards in "
                f"{_format_segment_list(thin_segments)}; segment-specific SD estimates will use fallback behavior."
            )

    col_add, col_reset = st.columns(2)

    with col_add:
        pos_input = st.number_input(
            "Break position",
            min_value=0,
            value=0,
            step=1,
            key=f"{key_prefix}_break_position",
        )
        if st.button("+ Add break", key=f"{key_prefix}_add_break"):
            if pos_input > 0:
                current_breaks = _extract_break_positions(u_config.segment_assignments, all_samples)
                if pos_input not in current_breaks:
                    current_breaks.append(pos_input)
                    assignments = assign_segments_from_breaks(all_samples, current_breaks)
                    _, empty_segments, thin_segments = _validate_segment_assignments(
                        all_samples,
                        assignments,
                        excluded_standards=u_config.excluded_standards,
                    )
                    if empty_segments:
                        st.error(
                            "Break rejected because "
                            f"{_format_segment_list(empty_segments)} would contain no included standards."
                        )
                        return u_config
                    if thin_segments:
                        st.warning(
                            "Repeatability segmentation has fewer than 2 included standards in "
                            f"{_format_segment_list(thin_segments)}; segment-specific SD estimates will use fallback behavior."
                        )
                    u_config = replace(u_config, segment_assignments=assignments)
                    state.uncertainty_config = u_config
                    st.rerun()

    with col_reset:
        if st.button("Reset segments", key=f"{key_prefix}_reset_segments"):
            u_config = replace(u_config, segment_assignments={})
            state.uncertainty_config = u_config
            st.rerun()

    return u_config


def _format_segment_list(segments: List[int]) -> str:
    """Format segment ids for concise warnings."""
    if len(segments) == 1:
        return f"segment {segments[0]}"
    return "segments " + ", ".join(str(seg) for seg in segments)


def _resolve_segment_assignment(sample, segment_assignments: Dict[str, int]) -> int:
    """Resolve a sample's segment using identity keys before legacy name keys."""
    from domain.uncertainty.reprod import standard_identity_key

    identity = standard_identity_key(sample)
    return int(segment_assignments.get(identity, segment_assignments.get(sample.name, 1)))


def _validate_segment_assignments(
    all_samples: list,
    segment_assignments: Dict[str, int],
    *,
    excluded_standards: Optional[List[str]] = None,
) -> Tuple[Dict[int, int], List[int], List[int]]:
    """Count included standards per segment and identify empty/thin segments."""
    from domain.uncertainty.reprod import _standard_is_excluded, standard_identity_key

    if not segment_assignments:
        return {}, [], []

    excluded = set(excluded_standards or [])
    expected_segments = {int(seg) for seg in segment_assignments.values()}
    counts = {seg: 0 for seg in expected_segments}
    seen_identities: set[str] = set()

    for sample in all_samples:
        segment = _resolve_segment_assignment(sample, segment_assignments)
        counts.setdefault(segment, 0)
        if not sample.is_standard or sample.metadata.get("excluded", False):
            continue
        identity = standard_identity_key(sample)
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        # Same three reference forms the domain honours, so the segment warning
        # counts what compute_reprod will actually include.
        if _standard_is_excluded(sample, excluded):
            continue
        counts[segment] += 1

    empty_segments = [seg for seg, count in sorted(counts.items()) if count == 0]
    thin_segments = [seg for seg, count in sorted(counts.items()) if count == 1]
    return counts, empty_segments, thin_segments


def _extract_break_positions(
    segment_assignments: Dict[str, int],
    all_samples: list,
) -> List[int]:
    """Infer break positions from current segment assignments."""
    if not segment_assignments:
        return []

    # Sort samples by run_number, find where segment changes
    sorted_samples = sorted(all_samples, key=lambda s: s.run_number)
    breaks = []
    prev_seg = None
    prev_pos = None
    for s in sorted_samples:
        seg = _resolve_segment_assignment(s, segment_assignments)
        if prev_seg is not None and seg != prev_seg and prev_pos is not None:
            breaks.append(prev_pos)
        prev_seg = seg
        prev_pos = s.run_number
    return breaks

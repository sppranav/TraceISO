"""Summary and budget-detail rendering for the uncertainty tab."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import UncertaintyConfig
from domain.models import Sample
from domain.uncertainty.scope import (
    CROSS_RATIO_INDEPENDENCE_NOTE,
    budget_scope as _canonical_budget_scope,
    budget_scope_label as _canonical_budget_scope_label,
    is_invalid_budget_scope as _canonical_is_invalid_budget_scope,
)
from ui.formatting import (
    format_uncertainty,
    format_value_matched,
    format_with_k,
    get_k_footnote,
)
from ui.runtime_budget_cache import build_cached_runtime_uncertainty_map
from ui.tabs.uncertainty_sections.common import _render_compact_summary
from ui.tabs.uncertainty_sections.config_controls import (
    _ALWAYS_BY_DESIGN,
    contributor_display_label,
    _coverage_footnote_for_config,
)
from ui.tabs.uncertainty_sections.equations_mc import (
    _render_equations,
    _render_mc_cross_check,
)
from ui.tabs.uncertainty_sections import shared_ui
from ui.components.table_utils import style_status_column
from ui.tabs.uncertainty_sections.ui_context import UncertaintyUiContext
from ui.config_plotly import get_plotly_config, PLOTLY_BASE_FONT_SIZE, PLOTLY_AXIS_TITLE_FONT_SIZE
from ui.theme import get_theme
from ui.utils import (
    format_isotope_label,
    format_sample_display_label,
    get_cycle_ranges,
    get_sample_state_key,
    render_runtime_budget_error,
)


def _render_summary_table(
    samples: list,
    ratio_name: str,
    u_config: UncertaintyConfig,
    state,
) -> List[Dict]:
    """Build and display the uncertainty summary table. Returns row dicts with _budget."""
    from domain.uncertainty.runtime import lookup_runtime_budget

    st.subheader("Uncertainty Summary")

    display_samples = list(samples)
    cycle_ranges = get_cycle_ranges(state)
    filter_method = (
        state.processing_config.filter_method if state.processing_config else "None"
    )
    filter_threshold = (
        state.processing_config.get_active_filter_threshold()
        if state.processing_config
        else 2.0
    )

    all_session_samples = list(state.result.samples if state.has_result else display_samples)
    median_samples = [
        sample
        for sample in all_session_samples
        if not (getattr(sample, "metadata", {}) or {}).get("excluded", False)
    ]
    median_samples = _filter_uncertainty_budget_samples(
        median_samples,
        u_config,
        state,
    )
    budget_samples = _unique_samples_for_budget(display_samples + median_samples)

    try:
        budgets = build_cached_runtime_uncertainty_map(
            budget_samples,
            [ratio_name],
            element_config=state.element_config,
            processing_config=state.processing_config,
            uncertainty_config=u_config,
            all_session_samples=all_session_samples,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
            drift_fit_info=(
                state.result.quality_metrics.get("drift_fit_info")
                if state.has_result and getattr(state.result, "quality_metrics", None)
                else None
            ),
            custom_contributor_library=state.custom_contributor_library,
            profile_defaults=getattr(state, "uncertainty_profile_defaults", None),
        )
    except Exception as exc:
        render_runtime_budget_error("Uncertainty Summary", exc)
        return []

    if not budgets:
        st.info("No uncertainty budgets could be computed for the selected ratio and samples.")
        return []

    # Build table rows
    rows = []
    is_absolute = u_config.output_mode == "absolute_ratio"
    for sample in display_samples:
        budget = lookup_runtime_budget(budgets, sample, ratio_name)
        if budget is None:
            continue
        rows.append({
            "Sample": sample.name,
            "_sample_label": format_sample_display_label(sample),
            "_sample_key": get_sample_state_key(sample),
            "Type": sample.sample_type.upper(),
            "u_c (permil)": budget.u_combined_rel_permil,
            "U (permil)": budget.expanded_rel_permil,
            "k": budget.coverage_factor_k,
            "Dominant": budget.dominant_contributor,
            "_budget": budget,
            "_sample": sample,
        })

    if not rows:
        st.info("No budgets available for the selected samples.")
        return []

    # Compute flag thresholds
    u_values: List[float] = []
    for sample in median_samples:
        budget = lookup_runtime_budget(budgets, sample, ratio_name)
        if budget is None or _is_invalid_budget_scope(budget):
            continue
        flag_metric = _flag_metric_for_budget(budget, is_absolute=is_absolute)
        if np.isfinite(flag_metric):
            u_values.append(flag_metric)
    median_u = float(np.median(u_values)) if u_values else 0.0

    # Median expanded U (in the displayed unit) for the delta-from-median
    # column. This is informational/sortable only; flag math is unchanged.
    expanded_values = [
        float(b.expanded_abs)
        for s in median_samples
        if (b := lookup_runtime_budget(budgets, s, ratio_name)) is not None
        and not _is_invalid_budget_scope(b)
    ]
    median_expanded_abs = (
        float(np.median(expanded_values)) if expanded_values else 0.0
    )

    high_multiple = float(getattr(u_config, "summary_flag_high_multiple", 3.0))
    elevated_multiple = float(getattr(u_config, "summary_flag_elevated_multiple", 2.0))

    df, meta = _build_summary_display_dataframe(
        rows,
        ratio_name=ratio_name,
        is_absolute=is_absolute,
        median_u=median_u,
        median_expanded_abs=median_expanded_abs,
        elevated_multiple=elevated_multiple,
        high_multiple=high_multiple,
    )

    drop_cols = list(meta["collapsed_columns"])
    display_df = df.drop(columns=drop_cols) if drop_cols else df

    numeric_cols = [c for c in meta["numeric_columns"] if c in display_df.columns]
    styler = display_df.style.map(_flag_cell_style, subset=["Flag"]).map(
        _type_cell_style, subset=["Type"]
    )
    column_config = {
        col: st.column_config.NumberColumn(col, format=meta["formats"].get(col, "%.4g"))
        for col in numeric_cols
    }
    st.dataframe(
        styler,
        width="stretch",
        hide_index=True,
        key="uncertainty_summary_table",
        column_config=column_config,
    )

    for note in meta["collapsed_captions"]:
        st.caption(note)

    # Legend + footnote
    flag_metric_label = "expanded relative U" if is_absolute else "expanded delta U"
    st.caption(
        f"Flag basis: {flag_metric_label} median over eligible session samples. "
        "OK = Normal | "
        f"ELEVATED = Elevated ({getattr(u_config, 'summary_flag_elevated_multiple', 2.0):g}x-"
        f"{getattr(u_config, 'summary_flag_high_multiple', 3.0):g}x session median) | "
        "INTERF = Interference-dominated | "
        f"HIGH = Anomalous (> {getattr(u_config, 'summary_flag_high_multiple', 3.0):g}x session median)"
        " | INSUFFICIENT = Fewer than 2 valid cycles"
        " | UNAVAILABLE = Required reference data missing"
    )
    st.caption(_coverage_footnote_for_config(u_config))
    return rows


def _unique_samples_for_budget(samples: List[Sample]) -> List[Sample]:
    """Return samples de-duplicated by stable runtime identity."""
    unique: List[Sample] = []
    seen: set[tuple] = set()
    for sample in samples:
        key = (getattr(sample, "run_number", None), getattr(sample, "name", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def _budget_scope(budget) -> str:
    return _canonical_budget_scope(budget)


def _is_invalid_budget_scope(budget) -> bool:
    return _canonical_is_invalid_budget_scope(budget)


def _budget_scope_label(budget) -> str:
    label = _canonical_budget_scope_label(budget)
    return label.capitalize() if label else "Unavailable"


def _budget_scope_flag(budget) -> str:
    scope = _budget_scope(budget)
    if scope == "insufficient_data":
        return "INSUFFICIENT"
    if scope == "unavailable":
        return "UNAVAILABLE"
    return "UNAVAILABLE"


def _measurand_value(budget, *, is_absolute: bool) -> float:
    """Return the reported measurand: corrected ratio, or delta in permil.

    Mirrors the delta derivation used in equations_mc._delta_display_values:
    delta = (basis_ratio / delta_reference - 1) * 1000.
    """
    if _is_invalid_budget_scope(budget):
        return float("nan")

    if is_absolute:
        value = float(getattr(budget, "ratio_value", 0.0) or 0.0)
        return value if np.isfinite(value) and value != 0.0 else float("nan")

    def _as_float(name):
        try:
            out = float(getattr(budget, name, None))
        except (TypeError, ValueError):
            return None
        return out if np.isfinite(out) else None

    ratio_mean = _as_float("basis_ratio_value") or _as_float("ratio_value")
    ref_value = _as_float("delta_reference_value")
    scale = _as_float("delta_scale_factor")
    if scale is None or scale <= 0.0:
        if ratio_mean is None or ref_value is None or ref_value == 0.0:
            return float("nan")
        scale = ratio_mean / ref_value
    return (scale - 1.0) * 1000.0


def _flag_metric_for_budget(budget, *, is_absolute: bool) -> float:
    """Return the scalar used for summary flag comparisons."""
    return float(
        getattr(
            budget,
            "expanded_rel_permil" if is_absolute else "expanded_abs",
            0.0,
        )
        or 0.0
    )


def _compute_flag(
    budget,
    median_u: float,
    *,
    flag_metric: Optional[float] = None,
    elevated_multiple: float = 2.0,
    high_multiple: float = 3.0,
) -> str:
    """Assign a color flag based on the budget characteristics."""
    if _is_invalid_budget_scope(budget):
        return _budget_scope_flag(budget)

    u = float(budget.expanded_rel_permil if flag_metric is None else flag_metric)

    # Red: anomalously large
    if median_u > 0 and u > high_multiple * median_u:
        return "HIGH"

    # Orange: interference-dominated
    dominant = str(getattr(budget, "dominant_contributor", "") or "").lower()
    if "interf" in dominant or "rb" in dominant or "kr" in dominant:
        return "INTERF"

    # Amber: elevated (between 2x and 3x median) but not otherwise flagged
    if median_u > 0 and u > elevated_multiple * median_u:
        return "ELEVATED"

    # Green: normal (U < 2x session median)
    return "OK"


_FLAG_BG = {
    "OK": "background-color: rgba(46, 204, 113, 0.18)",
    "ELEVATED": "background-color: rgba(241, 196, 15, 0.22)",
    "INTERF": "background-color: rgba(230, 126, 34, 0.22)",
    "HIGH": "background-color: rgba(231, 76, 60, 0.25)",
    "INSUFFICIENT": "background-color: rgba(149, 165, 166, 0.20)",
    "UNAVAILABLE": "background-color: rgba(149, 165, 166, 0.20)",
}
_TYPE_FG = {
    "SMP": "color: #4A90D9",
    "STD": "color: #9B59B6",
    "QC": "color: #16A085",
    "BLK": "color: #7F8C8D",
}


def _flag_cell_style(value) -> str:
    """pandas Styler hook: colour the Flag cell by status."""
    return _FLAG_BG.get(str(value), "")


def _type_cell_style(value) -> str:
    """pandas Styler hook: tint the sample-type text."""
    return _TYPE_FG.get(str(value).upper(), "")


def _build_summary_display_dataframe(
    rows: List[Dict],
    *,
    ratio_name: str,
    is_absolute: bool,
    median_u: float,
    median_expanded_abs: float,
    elevated_multiple: float,
    high_multiple: float,
) -> tuple[pd.DataFrame, Dict]:
    """Build the numeric, sortable summary display DataFrame.

    Numeric columns keep float dtype (``NaN`` for invalid scopes) so the
    Streamlit table sorts numerically. The human-readable status lives in the
    ``Flag`` column only. The internal ``rows`` contract is untouched.
    """
    uc_col = "u_c" if is_absolute else "u(δ) ‰"
    u_col = "U" if is_absolute else "U(δ) ‰"
    delta_col = "Δ vs median"
    ratio_label = format_isotope_label(ratio_name)
    value_col = ratio_label if is_absolute else f"δ {ratio_label} ‰"
    numeric_columns = [value_col, uc_col, u_col, delta_col, "k", "ν_eff"]

    records: List[Dict] = []
    valid_k: List[float] = []
    valid_veff: List[float] = []
    fixed_coverage: List[bool] = []
    for row in rows:
        budget = row["_budget"]
        invalid_scope = _is_invalid_budget_scope(budget)
        flag_metric = _flag_metric_for_budget(budget, is_absolute=is_absolute)
        flag = _compute_flag(
            budget,
            median_u,
            flag_metric=flag_metric,
            elevated_multiple=elevated_multiple,
            high_multiple=high_multiple,
        )
        record = {
            "Sample": row["_sample_label"],
            "Type": row["Type"],
        }
        record[value_col] = _measurand_value(budget, is_absolute=is_absolute)
        if invalid_scope:
            record[uc_col] = float("nan")
            record[u_col] = float("nan")
            record[delta_col] = float("nan")
            record["k"] = float("nan")
            record["ν_eff"] = float("nan")
            record["Dominant"] = "-"
        else:
            expanded_abs = float(budget.expanded_abs)
            record[uc_col] = float(budget.u_combined_abs)
            record[u_col] = expanded_abs
            record[delta_col] = expanded_abs - median_expanded_abs
            record["k"] = float(row["k"])
            fixed_k = getattr(budget, "coverage_method", "unknown") == "fixed_k"
            record["ν_eff"] = float("nan") if fixed_k else float(budget.effective_dof)
            record["Dominant"] = _format_dominant(row["Dominant"], budget=budget)
            valid_k.append(float(row["k"]))
            valid_veff.append(record["ν_eff"])
            fixed_coverage.append(fixed_k)
        record["Flag"] = flag
        records.append(record)

    column_order = [
        "Sample",
        "Type",
        value_col,
        uc_col,
        u_col,
        delta_col,
        "k",
        "ν_eff",
        "Dominant",
        "Flag",
    ]
    df = pd.DataFrame(records, columns=column_order)
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    collapsed_columns: List[str] = []
    collapsed_captions: List[str] = []
    if valid_k and len(set(round(v, 6) for v in valid_k)) == 1:
        collapsed_columns.append("k")
        collapsed_captions.append(
            f"Coverage factor k = {valid_k[0]:g} for all listed samples."
        )
    if fixed_coverage and all(fixed_coverage):
        collapsed_columns.append("ν_eff")
        collapsed_captions.append(
            "Effective degrees of freedom are not calculated in fixed-k mode."
        )
    elif valid_veff and all(v == float("inf") for v in valid_veff):
        collapsed_columns.append("ν_eff")
        collapsed_captions.append(
            "Recorded effective degrees of freedom are infinite "
            "for all listed samples."
        )

    meta = {
        "uc_col": uc_col,
        "u_col": u_col,
        "delta_col": delta_col,
        "numeric_columns": numeric_columns,
        "collapsed_columns": collapsed_columns,
        "collapsed_captions": collapsed_captions,
        "formats": {
            value_col: "%.8g" if is_absolute else "%+.4f",
            uc_col: "%.4g",
            u_col: "%.4g",
            delta_col: "%+.3g",
            "k": "%.2f",
            "ν_eff": "%.1f",
        },
    }
    return df, meta


def _format_effective_dof_value(value: float, *, coverage_method: str = "unknown") -> str:
    """Format effective degrees of freedom for summary display."""
    if coverage_method == "fixed_k":
        return "Not calculated (fixed k)"
    if value == float("inf"):
        return "\u221e"
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _format_dominant(name: str, *, budget=None) -> str:
    """Format a contributor name for display."""
    if not name:
        return "-"
    if name == "u_std_repeatability":
        method = str(
            getattr(getattr(budget, "reprod_result", None), "method", "") or ""
        )
        method_labels = {
            "sd_of_means": "SD",
            "loo_cross_validation": "LOO cross-validation",
            "drift_residuals": "drift residuals",
            "robust_mad": "robust MAD",
        }
        return f"Repeatability of SRM ({method_labels.get(method, 'SD')})"
    if name == "u_std_repeatability_se":
        return "SRM repeatability (SE)"
    label = contributor_display_label(name)
    if label != name:
        return label
    if name == "u_k1_sample_decomposition":
        return "Sample decomposition (k1)"
    if name in {"u_k2", "u_k2_matrix_separation"}:
        return "Matrix separation (k2)"
    if name in {"u_k3", "u_k3_procedural_blank"}:
        return "Procedural blank (k3)"
    if name in {"u_k4", "u_k4_bracketing_standard_heterogeneity"}:
        return "Bracketing-standard heterogeneity (k4)"
    if name == "u_k5_instrumental_drift":
        return "Instrumental drift (k5)"
    if name == "u_k6_matrix_effects":
        return "Matrix effects (k6)"
    if name in {"u_k7", "u_k7_residual_interferences"}:
        return "Residual interferences (k7)"
    if name == "u_bias_ref":
        return "Reference bias (\u0394_ref)"
    return name.replace("_", " ").replace("u ", "").strip().title()


def _inactive_by_design_names(u_config: Optional[UncertaintyConfig]) -> set[str]:
    """Return contributor names that are inactive by design under current config."""
    names = set(_ALWAYS_BY_DESIGN)
    if u_config is not None:
        names.update(u_config.disabled_contributor_names())
    return names


def _get_budget_contributor_status(
    contributor,
    *,
    inactive_by_design: Optional[set[str]] = None,
) -> str:
    """Classify a contributor for the budget detail table."""
    if contributor.is_active:
        return "ACTIVE"
    if contributor.name in (inactive_by_design or set()):
        return "BY DESIGN"
    return "MISSING"


# Detail view

def _render_detail_selector(
    rows: List[Dict],
    ratio_name: str,
    u_config: UncertaintyConfig,
    state,
    *,
    context: Optional[UncertaintyUiContext] = None,
) -> None:
    """Sample selector + detail view dispatch."""
    st.divider()
    st.subheader("Budget Detail")

    detail_rows = _filter_budget_detail_rows(rows, u_config, state)
    if not detail_rows:
        st.info("No eligible samples are available for budget detail in the current uncertainty mode.")
        return

    option_keys = [r.get("_sample_key", str(r["Sample"])) for r in detail_rows]
    labels = {
        r.get("_sample_key", str(r["Sample"])): r.get("_sample_label", str(r["Sample"]))
        for r in detail_rows
    }
    widget_key = "uncertainty_detail_sample"
    if st.session_state.get(widget_key) not in option_keys:
        st.session_state[widget_key] = option_keys[0]
    selected = st.selectbox(
        "Select sample for detail view",
        options=option_keys,
        format_func=lambda key: labels.get(key, str(key)),
        key=widget_key,
    )

    row = next((r for r in detail_rows if r.get("_sample_key", str(r["Sample"])) == selected), None)
    if row is None:
        return

    budget = row["_budget"]
    sample_obj = row["_sample"]
    _render_detail_view(
        budget,
        row.get("_sample_label", str(row["Sample"])),
        sample_obj,
        ratio_name,
        u_config,
        state,
        context=context,
    )


def _filter_uncertainty_budget_samples(
    samples: List[Sample],
    u_config: UncertaintyConfig,
    state,
) -> List[Sample]:
    """Return the samples allowed in uncertainty summary/detail views."""
    if not _is_ssb_delta_budget_mode(u_config, state):
        return samples

    return [
        sample for sample in samples
        if not sample.is_standard and not sample.is_blank
    ]


def _is_ssb_delta_budget_mode(
    u_config: UncertaintyConfig,
    state,
) -> bool:
    """Return True when budget views should show only SSB/delta unknown samples."""
    element_symbol = getattr(state, "element_symbol", "") or ""
    if not element_symbol and getattr(state, "element_config", None) is not None:
        element_symbol = getattr(state.element_config, "symbol", "")

    processing_config = getattr(state, "processing_config", None)
    resolve_engine = getattr(u_config, "resolve_engine", None)
    engine = (
        resolve_engine(element_symbol, processing_config=processing_config)
        if callable(resolve_engine)
        else getattr(u_config, "engine", "")
    )
    return (
        engine == "ssb_delta"
        and bool(getattr(u_config, "enable_ssb", False))
    )


def _filter_budget_detail_rows(
    rows: List[Dict],
    u_config: UncertaintyConfig,
    state,
) -> List[Dict]:
    """Return rows allowed in the Budget Detail selector."""
    if not _is_ssb_delta_budget_mode(u_config, state):
        return rows

    return [
        row for row in rows
        if not getattr(row.get("_sample"), "is_standard", False)
        and not getattr(row.get("_sample"), "is_blank", False)
    ]


def _render_detail_view(
    budget,
    sample_name: str,
    sample=None,
    ratio_name: str = "",
    u_config=None,
    state=None,
    *,
    context: Optional[UncertaintyUiContext] = None,
) -> None:
    """Render the full detail view for a single sample's uncertainty budget."""

    contributors = budget.contributors
    if _is_invalid_budget_scope(budget):
        _render_compact_summary(
            [
                ("Sample", sample_name),
                ("Status", _budget_scope_label(budget)),
            ]
        )
        if getattr(budget, "scope_note", ""):
            if _budget_scope(budget) == "insufficient_data":
                st.warning(budget.scope_note)
            else:
                st.error(budget.scope_note)
        return

    inactive = [c for c in contributors if not c.is_active]

    # Split inactive contributors into "by design" (expected to be zero
    # under current settings) vs "missing data" (genuinely incomplete).
    _BY_DESIGN = {
        "u_crm",   # zero in delta mode; cancels in delta calculation
    }
    if u_config is not None:
        _BY_DESIGN.update(u_config.disabled_contributor_names())
        # Also treat contributors that are off by engine default (never explicitly
        # toggled by the user) as "by design" to avoid false "Budget incomplete"
        # warnings on first render before the config panel has been visited.
        _el_sym = ""
        if state is not None:
            _el_sym = getattr(state, "element_symbol", "") or ""
            if not _el_sym and getattr(state, "element_config", None) is not None:
                _el_sym = getattr(state.element_config, "symbol", "")
        for _c in contributors:
            if not _c.is_active and _c.name not in _BY_DESIGN:
                if not u_config.is_contributor_enabled(_c.name, element_symbol=_el_sym):
                    _BY_DESIGN.add(_c.name)
    by_design = [c for c in inactive if c.name in _BY_DESIGN]

    user_disabled_names: set[str] = set()
    if u_config is not None:
        user_disabled_names.update(u_config.disabled_contributor_names())
    visible_by_design = [c for c in by_design if c.name not in user_disabled_names]
    is_absolute = u_config.output_mode == "absolute_ratio" if u_config else False
    crm_inactive_in_delta = (
        not is_absolute and any(c.name == "u_crm" for c in visible_by_design)
    )
    if crm_inactive_in_delta:
        st.info(
            "Certified/assigned reference-ratio uncertainty is inactive in "
            "delta-value reporting."
        )
    remaining_by_design = [
        c for c in visible_by_design
        if not (crm_inactive_in_delta and c.name == "u_crm")
    ]
    if remaining_by_design:
        st.info(
            f"{len(remaining_by_design)} contributor(s) inactive under the current settings: "
            + ", ".join(c.display_name for c in remaining_by_design)
            + "."
        )

    if is_absolute:
        summary_items = [
            ("Result", format_value_matched(budget.ratio_value, budget.u_combined_abs)),
            ("Combined standard uncertainty u_c", format_uncertainty(budget.u_combined_abs)),
            (
                f"Expanded uncertainty U (k = {budget.coverage_factor_k:.2f})",
                format_uncertainty(budget.expanded_abs),
            ),
            ("Effective DoF", _format_effective_dof_value(budget.effective_dof, coverage_method=budget.coverage_method)),
        ]
    else:
        delta_value = _measurand_value(budget, is_absolute=False)
        delta_reference = getattr(budget, "delta_reference_value", None)
        result_label = "Delta result"
        if delta_reference is not None and np.isfinite(float(delta_reference)):
            result_label = f"Delta result (reference = {float(delta_reference):.8g})"
        summary_items = [
            (
                result_label,
                f"{format_value_matched(delta_value, budget.u_combined_abs)} ‰",
            ),
            ("Combined standard uncertainty u_c", format_uncertainty(budget.u_combined_abs, unit="‰")),
            (
                f"Expanded uncertainty U (k = {budget.coverage_factor_k:.2f})",
                format_uncertainty(budget.expanded_abs, unit="‰"),
            ),
            ("Effective DoF", _format_effective_dof_value(budget.effective_dof, coverage_method=budget.coverage_method)),
        ]
    _render_compact_summary(summary_items)
    if getattr(budget, "scope_note", ""):
        st.info(budget.scope_note)

    st.markdown("**Variance Share**")
    ui_kind = getattr(context, "ui_kind", None)
    _render_tornado_chart(
        contributors,
        ratio_name=ratio_name,
        sample_name=sample_name,
        ui_kind=ui_kind,
    )

    st.markdown("**Budget Table**")
    _render_budget_table(
        contributors,
        is_absolute=is_absolute,
        inactive_by_design=_BY_DESIGN,
        ui_kind=ui_kind,
        coverage_footnote=(
            _coverage_footnote_for_config(u_config)
            if u_config is not None
            else get_k_footnote(k=budget.coverage_factor_k)
        ),
    )

    _render_diagnostic_jump_actions(contributors, context)

    # Equations in expander
    with st.expander("Equations & Coverage", expanded=False):
        _render_equations(budget, is_absolute=is_absolute)
        st.caption(CROSS_RATIO_INDEPENDENCE_NOTE)

    # Copyable budget debug dump (Dev mode) — exposes the exact inputs behind
    # each contributor (cycle range, filter, derived cycle counts) so app
    # numbers can be compared against external scripts/notebooks.
    if state is not None and getattr(state, "dev_mode", False):
        with st.expander("Budget Debug Values (Dev mode)", expanded=False):
            st.caption(
                "Copy this block to compare the exact GUM budget inputs "
                "(cycle ranges, filter, per-contributor cycle counts) with an "
                "external script or notebook."
            )
            try:
                report = build_budget_debug_report(
                    budget,
                    sample_name=sample_name,
                    sample=sample,
                    ratio_name=ratio_name,
                    u_config=u_config,
                    state=state,
                )
            except Exception as exc:  # pragma: no cover - diagnostic only
                st.caption(f"Budget debug unavailable: {exc}")
            else:
                from ui.diagnostics import capture_debug_report
                capture_debug_report("GUM", report)
                st.code(report, language=None)

    # Fixed-size Monte Carlo cross-check (collapsible)
    if sample is not None and ratio_name and u_config is not None and state is not None:
        _render_mc_cross_check(budget, sample, ratio_name, u_config, state)


def build_budget_debug_report(
    budget,
    *,
    sample_name: str,
    sample=None,
    ratio_name: str = "",
    u_config=None,
    state=None,
) -> str:
    """Return a copyable text dump of the exact GUM budget inputs in use.

    The numbers shown are the *runtime* budget the Budget Detail view displays
    (cycle-range and filter aware).  For each active Type-A contributor the
    effective cycle count is derived as ``n = round(dof + 1)`` — this is the
    fastest way to see, e.g., that ``u_prec`` used 64 cycles while ``u_blank``
    used the full 70 because an outlier filter or cycle window trimmed the
    sample ratio.
    """

    def _fmt(value: object) -> str:
        if value is None:
            return "None"
        if isinstance(value, float):
            if value == float("inf"):
                return "inf"
            if value != value:  # NaN
                return "nan"
            if value == 0.0:
                return "0"
            # Fixed-point (no scientific notation), shortest string that
            # round-trips — e.g. 0.00000625076 instead of 6.25076e-06.
            return np.format_float_positional(value, unique=True, trim="-")
        return str(value)

    def _derived_n(contributor) -> str:
        if str(contributor.type_ab).upper() != "A":
            return "-"
        dof = float(contributor.degrees_of_freedom)
        if not np.isfinite(dof):
            return "-"
        return str(int(round(dof + 1.0)))

    engine_config = getattr(u_config, "engine", "") or "" if u_config is not None else ""
    engine_resolved = engine_config
    if u_config is not None:
        _elem_sym = ""
        if state is not None:
            _elem_sym = getattr(state, "element_symbol", "") or ""
            if not _elem_sym and getattr(state, "element_config", None) is not None:
                _elem_sym = getattr(state.element_config, "symbol", "")
        try:
            engine_resolved = u_config.resolve_engine(
                _elem_sym,
                processing_config=getattr(state, "processing_config", None),
            )
        except Exception:
            engine_resolved = engine_config

    lines = [
        "# Budget debug report",
        f"sample = {sample_name}",
        f"ratio = {ratio_name}",
        # engine_config is the raw selector; for Sr it resolves to
        # internal_normalization even though the stored value reads ssb_delta.
        f"engine_config = {engine_config}",
        f"engine_resolved = {engine_resolved}",
        f"output_mode = {getattr(budget, 'output_mode', '') or (u_config.output_mode if u_config else '')}",
        f"budget_scope = {getattr(budget, 'budget_scope', '')}",
        f"ratio_value = {_fmt(getattr(budget, 'ratio_value', None))}",
        f"u_combined_abs = {_fmt(budget.u_combined_abs)}",
        f"u_combined_rel_permil = {_fmt(budget.u_combined_rel_permil)}",
        f"expanded_abs = {_fmt(budget.expanded_abs)}",
        f"coverage_factor_k = {_fmt(budget.coverage_factor_k)}",
        f"effective_dof = {_fmt(budget.effective_dof)}",
        f"dominant_contributor = {getattr(budget, 'dominant_contributor', '') or '-'}",
    ]

    # Runtime context: cycle range + filter (explains why n differs per term).
    if state is not None and sample is not None:
        from domain.filters.outlier import resolve_cycle_range
        from domain.ratio_selection import get_best_ratio_data

        cycle_ranges = get_cycle_ranges(state)
        cycle_range = resolve_cycle_range(
            cycle_ranges,
            sample_name=sample.name,
            sample_key=get_sample_state_key(sample),
        )
        pcfg = getattr(state, "processing_config", None)
        lines.extend(
            [
                f"cycle_range = {_fmt(cycle_range)}",
                f"filter_method = {_fmt(getattr(pcfg, 'filter_method', None))}",
                f"filter_threshold = {_fmt(getattr(pcfg, 'get_active_filter_threshold', lambda: None)() if pcfg else None)}",
            ]
        )
        ratio_cd = get_best_ratio_data(sample, ratio_name)
        if ratio_cd is not None:
            lines.append(f"best_ratio_layer_n_valid = {_fmt(int(ratio_cd.n_valid))}")
            lines.append(f"best_ratio_layer_mean = {_fmt(float(ratio_cd.mean))}")

    reprod = getattr(budget, "reprod_result", None)
    if reprod is not None:
        try:
            n_std = int(sum(1 for inc in reprod.std_included if inc))
        except Exception:
            n_std = None
        lines.append(f"reprod.method = {getattr(reprod, 'method', '-')}")
        lines.append(f"reprod.n_standards = {_fmt(n_std)}")

    lines.append("# contributors (n derived from dof for Type A: n = dof + 1)")
    for c in budget.contributors:
        lines.append(
            f"contributor.{c.name} = "
            f"status:{'ACTIVE' if c.is_active else (c.state or 'INACTIVE')} "
            f"type:{c.type_ab} "
            f"abs:{_fmt(float(c.value_abs))} "
            f"permil:{_fmt(float(c.value_rel_permil))} "
            f"share_pct:{_fmt(float(c.percentage_contribution))} "
            f"dof:{_fmt(float(c.degrees_of_freedom))} "
            f"n:{_derived_n(c)}"
        )

    return "\n".join(lines)

def _render_budget_table(
    contributors: list,
    *,
    is_absolute: bool = False,
    inactive_by_design: Optional[set[str]] = None,
    coverage_footnote: Optional[str] = None,
    ui_kind: Optional[str] = None,
) -> None:
    """Budget breakdown table (JCGM 100:2008 notation)."""
    rows = []
    for c in contributors:
        status = _get_budget_contributor_status(
            c,
            inactive_by_design=inactive_by_design,
        )
        # Hide contributors that are inactive by design (user-unchecked or
        # structurally zero); they add no information to the printed table.
        if status == "BY DESIGN":
            continue

        # DoF display
        if c.degrees_of_freedom == float('inf'):
            dof_str = "\u221e"
        else:
            dof_str = str(int(c.degrees_of_freedom))

        # Classification: full text
        classification = f"Type {c.type_ab}" if c.type_ab else ""

        source_label = contributor_display_label(c.name, ui_kind=ui_kind)
        if source_label == c.name:
            source_label = _strip_type_suffix(c.display_name)
        d = {
            "Status": status,
            "Source": source_label,
            "Classification": classification,
        }
        if is_absolute:
            d["Standard uncertainty contribution u\u1d62(y)"] = format_uncertainty(c.value_abs) if c.is_active else "0"
            d["Relative contribution u\u1d62(y) (\u2030)"] = format_uncertainty(c.value_rel_permil, unit="\u2030") if c.is_active else "0 \u2030"
        else:
            d["Standard uncertainty contribution u\u1d62(y) (\u2030)"] = format_uncertainty(c.value_rel_permil, unit="\u2030") if c.is_active else "0 \u2030"
        d["Variance share (%)"] = round(c.percentage_contribution, 1)
        d["Degrees of freedom \u03bd\u1d62"] = dof_str
        rows.append(d)

    df = pd.DataFrame(rows)
    st.dataframe(style_status_column(df), width="stretch", hide_index=True)

    # Caption for contributor status.
    st.caption(
        "Status:\u2002ACTIVE\u2009\u2014\u2009contributor evaluated and included.\u2002"
        "MISSING\u2009\u2014\u2009contributor inactive (insufficient data; U is a lower bound).\u2002"
        "BY DESIGN\u2009\u2014\u2009contributor is zero under current settings "
        "(e.g. u_crm cancels in \u03b4-mode)."
    )
    st.caption(
        "Contributors are displayed in relative \u2030 for readability. "
        "RSS combination is performed in absolute ratio units."
    )
    st.caption(coverage_footnote or get_k_footnote())


_TYPE_ONLY_RE = re.compile(r"\s*\(Type [AB]\)$")       # "… (Type A)" at end
_TYPE_LEADING_RE = re.compile(r"\(Type [AB],\s*")      # "(Type A, rest)"
_TYPE_TRAILING_RE = re.compile(r",\s*Type [AB]\)")     # "(k2, Type B)"
_K_LABEL_RE = re.compile(r"\s*\(k\d\)$")              # "(k1)"–"(k6)" at end


def _strip_type_suffix(name: str) -> str:
    """Remove 'Type A/B' and kN labels from contributor names — redundant in the chart."""
    name = _TYPE_ONLY_RE.sub("", name)
    name = _TYPE_LEADING_RE.sub("(", name)
    name = _TYPE_TRAILING_RE.sub(")", name)
    name = _K_LABEL_RE.sub("", name)
    return name.strip()


def _variance_chart_label(
    contributor,
    *,
    ui_kind: Optional[str] = None,
) -> str:
    """Return a concise contributor label for the variance-share chart."""
    label = contributor_display_label(contributor.name, ui_kind=ui_kind)
    if label != contributor.name:
        return label
    return _strip_type_suffix(contributor.display_name)


def _contributor_type_color(type_ab: str, palette=None) -> str:
    palette = palette or get_theme().palette
    index = 0 if type_ab == "A" else 1
    return palette.contributor_sequence[index]


def _format_variance_share(percentage: float) -> str:
    """Format a variance share without hiding small non-zero values."""
    percentage = float(percentage)
    if percentage == 0.0:
        return "0.0%"
    if not np.isfinite(percentage):
        return "NaN%"

    magnitude = int(np.floor(np.log10(abs(percentage))))
    decimal_places = max(1, 2 - magnitude - 1)
    if decimal_places > 6:
        return f"{percentage:.1e}%"
    return f"{percentage:.{decimal_places}f}%"


def _render_tornado_chart(
    contributors: list,
    ratio_name: str = "",
    sample_name: str = "",
    ui_kind: Optional[str] = None,
) -> None:
    """Variance-share view: detailed per-contributor tornado chart."""
    active = [c for c in contributors if c.is_active and c.percentage_contribution > 0]
    if not active:
        st.info("No active contributors to chart.")
        return

    # One globally sorted list, largest share first.
    active = sorted(
        active,
        key=lambda c: c.percentage_contribution,
        reverse=True,
    )

    _render_detailed_variance_chart(
        active,
        ratio_name=ratio_name,
        sample_name=sample_name,
        ui_kind=ui_kind,
    )


def _render_compact_variance_bar(active: list) -> None:
    """Single stacked horizontal bar; one segment per contributor."""
    import plotly.graph_objects as go

    theme = get_theme()
    palette = theme.palette
    fig = go.Figure()
    for c in active:
        pct = float(c.percentage_contribution)
        pct_label = _format_variance_share(pct)
        label = _variance_chart_label(c)
        fig.add_trace(go.Bar(
            x=[pct],
            y=["Variance"],
            orientation="h",
            marker_color=_contributor_type_color(c.type_ab, palette),
            marker_line=dict(color=palette.annotation_border, width=1),
            text=[f"{label}: {pct_label}"],
            textposition="inside",
            insidetextanchor="middle",
            customdata=[pct_label],
            hovertemplate=f"{label}<br>%{{customdata}}<extra></extra>",
            showlegend=False,
        ))
    # Dummy legend entries for Type A / Type B.
    for label, color in (
        ("Type A", palette.contributor_sequence[0]),
        ("Type B", palette.contributor_sequence[1]),
    ):
        fig.add_trace(go.Bar(
            x=[None], y=["Variance"], orientation="h",
            marker_color=color, name=label, showlegend=True,
        ))

    fig.update_layout(
        title=dict(text=""),
        barmode="stack",
        height=130,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(title="Contribution to combined variance (%)", range=[0, 100]),
        yaxis=dict(showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor=palette.figure_grid, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    theme.apply_to_figure(fig, profile="diagnostic")
    st.plotly_chart(fig, width="stretch", config=get_plotly_config())


def _render_detailed_variance_chart(
    active: list,
    ratio_name: str = "",
    sample_name: str = "",
    ui_kind: Optional[str] = None,
) -> None:
    """Tornado bars with the largest contributor at the top."""
    import plotly.graph_objects as go

    theme = get_theme()
    palette = theme.palette
    # Plotly renders categorical bars bottom-to-top, so reverse the
    # descending list to place the largest contributor at the top.
    ordered = list(reversed(active))
    names = [_variance_chart_label(c, ui_kind=ui_kind) for c in ordered]
    shares = [float(c.percentage_contribution) for c in ordered]
    share_labels = [_format_variance_share(p) for p in shares]
    colors = [_contributor_type_color(c.type_ab, palette) for c in ordered]

    max_pct = max(shares)
    left_margin = min(320, max(180, max(len(n) for n in names) * 10))
    chart_font_size = PLOTLY_BASE_FONT_SIZE + 2
    chart_height = max(360, 84 * len(active))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=shares,
        y=names,
        orientation="h",
        marker_color=colors,
        text=share_labels,
        textposition="outside",
        textfont=dict(color=palette.figure_ink, size=chart_font_size),
        customdata=share_labels,
        showlegend=False,
        hovertemplate="%{y}<br>%{customdata}<extra></extra>",
    ))
    # Dummy legend traces for the Type A / Type B colour key.
    for label, color in (
        ("Type A", palette.contributor_sequence[0]),
        ("Type B", palette.contributor_sequence[1]),
    ):
        fig.add_trace(go.Bar(
            x=[None], y=[names[0]], orientation="h",
            marker_color=color, name=label, showlegend=True,
        ))

    fig.update_layout(
        title=dict(text=""),
        xaxis_title="Contribution to combined variance (%)",
        yaxis_title="",
        height=chart_height,
        margin=dict(l=left_margin, r=60, t=50, b=70),
        xaxis=dict(
            range=[0, max_pct * 1.3],
            title_font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE + 2),
            tickfont=dict(size=chart_font_size),
        ),
        yaxis=dict(tickfont=dict(size=chart_font_size)),
        font=dict(size=chart_font_size),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="left",
            x=0,
            font=dict(size=chart_font_size),
        ),
        barmode="overlay",
    )
    theme.apply_to_figure(fig, profile="diagnostic")
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, mirror=True)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, mirror=True)
    parts = [p for p in ("traceiso", sample_name, ratio_name) if p]
    export_filename = re.sub(r"[/\\]", "_", "_".join(parts))
    export_config = get_plotly_config(export_filename)
    export_config["toImageButtonOptions"].update({
        "width": 900,
        "height": max(560, chart_height + 200),
    })
    if export_config["toImageButtonOptions"]["format"] == "png":
        export_config["toImageButtonOptions"]["scale"] = 3
    # Use the available width: long contributor labels need the room, and the
    # figure height already scales with the number of bars.
    st.plotly_chart(fig, width="stretch", config=export_config)


def _render_diagnostic_jump_actions(
    contributors: list,
    context: Optional[UncertaintyUiContext],
) -> None:
    """Render compact buttons that jump to a contributor's diagnostic panel."""
    if context is None:
        return

    active = [
        c for c in contributors
        if c.is_active and c.percentage_contribution > 0
    ]
    targets = []
    for c in active:
        section = shared_ui.diagnostic_section_for_contributor(c.name, context)
        if section is not None:
            targets.append(c)
    if not targets:
        return

    st.markdown("**Jump to diagnostic**")
    n_cols = min(len(targets), 3)
    cols = st.columns(n_cols)
    for i, c in enumerate(targets):
        label = (
            f"{contributor_display_label(c.name, ui_kind=context.ui_kind)} "
            f"- {c.percentage_contribution:.1f}%"
        )
        btn_key = f"uncertainty_jump_{c.name}"
        if cols[i % n_cols].button(label, key=btn_key, width="stretch"):
            # Pop the button key before rerun; Streamlit forbids writing
            # button-widget keys in session_state on subsequent renders.
            st.session_state.pop(btn_key, None)
            shared_ui.request_uncertainty_navigation(
                "Diagnostics",
                contributor=c.name,
            )


def _format_absolute_ratio_with_k2(budget) -> str:
    """Compatibility helper for absolute ratio +/- U shown with fixed k=2."""
    return format_with_k(
        float(budget.ratio_value),
        float(budget.u_combined_abs) * 2.0,
        2.0,
    )


def _get_budget_contributor_value_rel_permil(budget, name: str) -> float:
    """Return a contributor's relative uncertainty in permil."""
    contributor = next(
        (c for c in budget.contributors if c.name == name and c.is_active),
        None,
    )
    return float(contributor.value_rel_permil) if contributor is not None else 0.0

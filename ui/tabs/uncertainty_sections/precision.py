"""Measurement-precision diagnostics for the uncertainty tab."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

from domain.uncertainty.scope import is_invalid_budget_scope, budget_scope_label, budget_scope_note

from config.settings import UncertaintyConfig, is_russell_law_normalization_engine
from ui.runtime_budget_cache import build_cached_runtime_uncertainty_map
from ui.formatting import format_uncertainty
from ui.tabs.uncertainty_sections.common import (
    _render_compact_summary,
    _render_equation_note_block,
)
from ui.utils import (
    format_isotope_label,
    format_sample_display_label,
    get_cycle_ranges,
    get_sample_state_key,
    render_runtime_budget_error,
)

def _render_measurement_precision_section(
    all_samples: list,
    ratio_name: str,
    u_config: UncertaintyConfig,
    state,
) -> None:
    """Render local measurement-precision diagnostics for the selected ratio."""
    from domain.uncertainty.runtime import lookup_runtime_budget

    _element_symbol = getattr(state, "element_symbol", "") or ""
    if not _element_symbol and getattr(state, "element_config", None) is not None:
        _element_symbol = getattr(state.element_config, "symbol", "") or ""
    resolved_engine = (
        u_config.resolve_engine(
            _element_symbol,
            processing_config=getattr(state, "processing_config", None),
        )
        if _element_symbol else ""
    )
    is_normalization_engine = is_russell_law_normalization_engine(resolved_engine)
    u_prec_mode = str(getattr(u_config, "u_prec_mode", "se") or "se").lower()
    if u_prec_mode not in {"se", "sd"}:
        u_prec_mode = "se"
    show_u_prec = u_config.is_contributor_enabled(
        "u_prec",
        element_symbol=_element_symbol,
    )
    show_u_std = (
        not is_normalization_engine
        and u_config.is_contributor_enabled("u_std", element_symbol=_element_symbol)
    )
    if not show_u_prec and not show_u_std:
        return

    st.divider()
    if is_normalization_engine or (show_u_prec and not show_u_std):
        title = "Sample measurement precision (u_prec)"
    elif show_u_std and not show_u_prec:
        title = "Bracketing-standard precision (u_std)"
    else:
        title = "Measurement Precision (u_prec + u_std)"
    st.subheader(title)

    # Engine A: include standards (each has its own u_prec = SE of its cycles).
    # Engine B: analyte measurements only — u_std comes from the SSB bracket.
    active_measurements = [
        sample for sample in all_samples
        if (
            not sample.is_blank
            and not sample.metadata.get("excluded", False)
            and (is_normalization_engine or not sample.is_standard)
        )
    ]
    if not active_measurements:
        st.info("No active samples available for measurement-precision diagnostics.")
        return

    cycle_ranges = get_cycle_ranges(state)
    filter_method = (
        state.processing_config.filter_method if state.processing_config else "None"
    )
    filter_threshold = (
        state.processing_config.get_active_filter_threshold()
        if state.processing_config
        else 2.0
    )

    try:
        budgets = build_cached_runtime_uncertainty_map(
            active_measurements,
            [ratio_name],
            element_config=state.element_config,
            processing_config=state.processing_config,
            uncertainty_config=u_config,
            all_session_samples=all_samples,
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
        render_runtime_budget_error("Measurement Precision", exc)
        return

    budget_by_sample = {}
    budgeted_samples = []
    for sample in active_measurements:
        budget = lookup_runtime_budget(budgets, sample, ratio_name)
        if budget is not None:
            budget_by_sample[get_sample_state_key(sample)] = budget
            budgeted_samples.append(sample)

    if not budgeted_samples:
        st.info("No measurement-precision budgets are available for the selected ratio.")
        return

    sample_labels = {
        get_sample_state_key(sample): format_sample_display_label(sample)
        for sample in budgeted_samples
    }
    sample_options = [get_sample_state_key(sample) for sample in budgeted_samples]
    widget_key = "uncertainty_precision_sample"
    if st.session_state.get(widget_key) not in sample_options:
        st.session_state[widget_key] = sample_options[0]
    selected_sample_key = st.selectbox(
        "Sample for precision diagnostics",
        options=sample_options,
        format_func=lambda key: sample_labels.get(key, str(key)),
        key=widget_key,
    )
    selected_sample = next(
        (sample for sample in budgeted_samples if get_sample_state_key(sample) == selected_sample_key),
        None,
    )
    if selected_sample is None:
        st.info("Select a sample to inspect its measurement precision.")
        return

    budget = budget_by_sample[selected_sample_key]
    if is_invalid_budget_scope(budget):
        st.info(f"{budget_scope_label(budget)}: {budget_scope_note(budget)}")
        return

    detail_df = _build_measurement_precision_detail_dataframe(
        budgeted_samples,
        budget_by_sample,
        ratio_name,
        include_standards=is_normalization_engine,
        u_prec_mode=u_prec_mode,
    )
    if detail_df.empty:
        st.info("No per-sample measurement precision values could be computed for the selected ratio.")
        return

    selected_row = detail_df[detail_df["_sample_key"] == selected_sample_key]
    if selected_row.empty:
        st.info("The selected sample does not have a measurement-precision entry for this ratio.")
        return
    selected_row = selected_row.iloc[0]

    is_absolute = u_config.output_mode == "absolute_ratio"
    if is_absolute:
        u_prec_value = format_uncertainty(float(selected_row["u_prec"]))
    else:
        u_prec_value = format_uncertainty(float(selected_row["u_prec (‰)"]), unit="‰")

    _sel_budget = budget_by_sample.get(selected_sample_key)
    _ratio_val = float(_sel_budget.ratio_value) if _sel_budget and getattr(_sel_budget, "ratio_value", None) else None
    _ratio_display = (
        f"{format_isotope_label(ratio_name)} = {_ratio_val:.8f}"
        if _ratio_val
        else format_isotope_label(ratio_name)
    )
    summary_items = [
        ("Selected sample", format_sample_display_label(selected_sample)),
        ("Ratio", _ratio_display),
    ]
    if show_u_prec:
        summary_items.extend(
            [
                ("Valid cycles", str(int(selected_row["n"]))),
                ("Precision mode", "SD of cycles" if u_prec_mode == "sd" else "SE of mean"),
                ("u_prec", u_prec_value),
            ]
        )
    if not is_normalization_engine and show_u_std:
        if is_absolute:
            u_std_value = format_uncertainty(float(selected_row["u_std"]))
            u_local_value = format_uncertainty(float(selected_row["u_local_precision"]))
        else:
            u_std_value = format_uncertainty(float(selected_row["u_std (‰)"]), unit="‰")
            u_local_value = format_uncertainty(float(selected_row["u_local_precision (‰)"]), unit="‰")
        summary_items.append(("u_std", u_std_value))
        if show_u_prec:
            summary_items.append(("Combined precision", u_local_value))
    overview_tab, detail_tab = st.tabs(["Overview", "Detail"])

    with overview_tab:
        col_summary, col_eq = st.columns([3, 2], gap="large")
        with col_summary:
            _render_compact_summary(summary_items)
            if is_normalization_engine:
                normalization_label = (
                    "Pb–Tl external normalization"
                    if resolved_engine == "pb_tl_external_normalization"
                    else "Internal normalization"
                )
                if u_prec_mode == "sd":
                    st.caption(
                        f"{normalization_label}: u_prec is the within-run SD of the "
                        f"IIF-corrected {ratio_name} cycles."
                    )
                else:
                    st.caption(
                        f"{normalization_label}: u_prec is the within-run SE of the "
                        f"IIF-corrected {ratio_name} cycles."
                    )
            elif show_u_std:
                bracket_label = str(selected_row["Bracketing standards"])
                if bracket_label and bracket_label != "-":
                    st.caption(
                        f"Bracketing standards for {format_sample_display_label(selected_sample)}: {bracket_label}"
                    )
                else:
                    st.caption(
                        f"No bracketing standard is attached to {format_sample_display_label(selected_sample)} for this ratio, so u_std = 0."
                    )
                if show_u_prec:
                    stat_label = "SD" if u_prec_mode == "sd" else "SE"
                    st.caption(
                        f"This section combines the sample-cycle {stat_label} (u_prec) with the "
                        "uncertainty of the two bracketing standard means (u_std)."
                    )
                else:
                    st.caption(
                        "This section reports the uncertainty of the two bracketing "
                        "standard means (u_std)."
                    )
        with col_eq:
            u_prec_equation = (
                r"u_{\mathrm{prec}} = s_{\mathrm{smp}}"
                if u_prec_mode == "sd"
                else r"u_{\mathrm{prec}} = s_{\mathrm{smp}} / \sqrt{n}"
            )
            if is_normalization_engine:
                _render_equation_note_block(
                    [u_prec_equation],
                )
            elif show_u_prec and show_u_std:
                _render_equation_note_block(
                    [
                        r"u_{\mathrm{precision}}^2 = u_{\mathrm{prec}}^2 + u_{\mathrm{std}}^2",
                        u_prec_equation,
                        r"u_{\mathrm{std}} = 0.5 \times \sqrt{\mathrm{SEM}_{\mathrm{std1}}^2 + \mathrm{SEM}_{\mathrm{std2}}^2}",
                    ],
                )
            elif show_u_prec:
                _render_equation_note_block(
                    [u_prec_equation],
                )
            elif show_u_std:
                _render_equation_note_block(
                    [r"u_{\mathrm{std}} = 0.5 \times \sqrt{\mathrm{SEM}_{\mathrm{std1}}^2 + \mathrm{SEM}_{\mathrm{std2}}^2}"],
                )

    with detail_tab:
        _prec_col = "u_prec" if is_absolute else "u_prec (‰)"
        _local_col = "u_local_precision" if is_absolute else "u_local_precision (‰)"
        _stat_col = _prec_col if is_normalization_engine else _local_col
        if show_u_std and not show_u_prec:
            _stat_col = "u_std" if is_absolute else "u_std (‰)"
        elif show_u_prec and not show_u_std:
            _stat_col = "u_prec" if is_absolute else "u_prec (‰)"
        median_prec = float(detail_df[_stat_col].median())
        max_prec = float(detail_df[_stat_col].max())
        if is_normalization_engine or (show_u_prec and not show_u_std):
            median_label = "Median u_prec"
            max_label = "Max u_prec"
        elif show_u_std and not show_u_prec:
            median_label = "Median u_std"
            max_label = "Max u_std"
        else:
            median_label = "Median combined precision"
            max_label = "Max combined precision"
        _render_compact_summary(
            [
                ("Entries in table", str(len(detail_df))),
                (
                    median_label,
                    format_uncertainty(median_prec)
                    if is_absolute
                    else format_uncertainty(median_prec, unit="‰"),
                ),
                (
                    max_label,
                    format_uncertainty(max_prec)
                    if is_absolute
                    else format_uncertainty(max_prec, unit="‰"),
                ),
            ]
        )
        _col_cfg = {
            "Value": st.column_config.TextColumn(format_isotope_label(ratio_name)),
            "u_prec": st.column_config.NumberColumn("u_prec", format="%.8f"),
            "cycle SD": st.column_config.NumberColumn("cycle SD", format="%.8f"),
            "cycle SE": st.column_config.NumberColumn("cycle SE", format="%.8f"),
            "u_std": st.column_config.NumberColumn("u_std", format="%.8f"),
            "u_local_precision": st.column_config.NumberColumn("u_local_precision", format="%.8f"),
            "u_prec (‰)": st.column_config.NumberColumn("u_prec (‰)", format="%.4f"),
            "u_std (‰)": st.column_config.NumberColumn("u_std (‰)", format="%.4f"),
            "u_local_precision (‰)": st.column_config.NumberColumn("u_local_precision (‰)", format="%.4f"),
            "ν_prec": st.column_config.TextColumn("ν"),
            "ν_std": st.column_config.TextColumn("ν_std"),
            "ν_local": st.column_config.TextColumn("ν_combined"),
        }
        st.dataframe(
            _format_measurement_precision_detail_dataframe(
                detail_df,
                is_engine_a=is_normalization_engine,
                show_u_prec=show_u_prec,
                show_u_std=show_u_std,
            ),
            width="stretch",
            hide_index=True,
            column_config=_col_cfg,
        )


def _extract_measurement_precision_metrics(budget) -> Dict[str, float]:
    """Return u_prec, u_std, and the RSS-combined local precision from a budget."""

    def _get_contributor(name: str):
        if budget is None or not hasattr(budget, "_find_contributor"):
            return None
        return budget._find_contributor(name)

    prec = _get_contributor("u_prec")
    u_std = _get_contributor("u_std")

    prec_active = bool(getattr(prec, "is_active", False)) if prec is not None else False
    std_active = bool(getattr(u_std, "is_active", False)) if u_std is not None else False

    u_prec_abs = float(prec.value_abs) if prec_active else 0.0
    u_prec_rel = float(prec.value_rel_permil) if prec_active else 0.0
    nu_prec = float(prec.degrees_of_freedom) if prec_active else 0.0

    u_std_abs = float(u_std.value_abs) if std_active else 0.0
    u_std_rel = float(u_std.value_rel_permil) if std_active else 0.0
    nu_std = float(u_std.degrees_of_freedom) if std_active else 0.0

    u_local_abs = float(np.sqrt(u_prec_abs ** 2 + u_std_abs ** 2))
    u_local_rel = float(np.sqrt(u_prec_rel ** 2 + u_std_rel ** 2))
    nu_local = _combine_local_precision_dof(
        u_prec_abs, nu_prec,
        u_std_abs, nu_std,
    )

    return {
        "u_prec": u_prec_abs,
        "u_prec_rel_permil": u_prec_rel,
        "nu_prec": nu_prec,
        "u_std": u_std_abs,
        "u_std_rel_permil": u_std_rel,
        "nu_std": nu_std,
        "u_local_precision": u_local_abs,
        "u_local_precision_rel_permil": u_local_rel,
        "nu_local_precision": nu_local,
        "n_cycles": int(getattr(budget, "n_cycles", 0)),
    }


def _combine_local_precision_dof(
    u_prec_abs: float,
    nu_prec: float,
    u_std_abs: float,
    nu_std: float,
) -> float:
    """Welch-Satterthwaite combination for the local u_prec + u_std term."""
    u_local_abs = float(np.sqrt(u_prec_abs ** 2 + u_std_abs ** 2))
    if u_local_abs <= 0:
        return 0.0

    denom = 0.0
    for u_i, nu_i in ((u_prec_abs, nu_prec), (u_std_abs, nu_std)):
        if u_i <= 0:
            continue
        if nu_i == float("inf"):
            continue
        if nu_i <= 0:
            continue
        denom += (u_i ** 4) / nu_i

    if denom <= 0:
        return float("inf")

    return (u_local_abs ** 4) / denom


def _build_measurement_precision_detail_dataframe(
    samples: list,
    budget_by_sample: Dict[str, object],
    ratio_name: str,
    *,
    include_standards: bool = False,
    u_prec_mode: str = "se",
) -> pd.DataFrame:
    """Build the per-sample measurement-precision detail table."""
    rows = []
    u_prec_mode = str(u_prec_mode or "se").lower()
    if u_prec_mode not in {"se", "sd"}:
        u_prec_mode = "se"

    for sample in samples:
        if sample.is_blank:
            continue
        if sample.is_standard and not include_standards:
            continue
        sample_key = get_sample_state_key(sample)
        budget = budget_by_sample.get(sample_key) or budget_by_sample.get(sample.name)
        if budget is None or is_invalid_budget_scope(budget):
            continue

        metrics = _extract_measurement_precision_metrics(budget)
        n_cycles = int(metrics["n_cycles"])
        sqrt_n = float(np.sqrt(n_cycles)) if n_cycles > 0 else 0.0
        if u_prec_mode == "sd":
            cycle_sd = float(metrics["u_prec"])
            cycle_se = cycle_sd / sqrt_n if sqrt_n > 0 else 0.0
        else:
            cycle_se = float(metrics["u_prec"])
            cycle_sd = cycle_se * sqrt_n if sqrt_n > 0 else 0.0
        ssb_data = sample.ssb_results.get(ratio_name, {})
        delta_data = sample.delta_results.get(ratio_name, {})
        bracket_label = "-"
        _bracket_src = ssb_data if ssb_data else delta_data
        if _bracket_src:
            prev_std = str(_bracket_src.get("prev_std", "") or "")
            next_std = str(_bracket_src.get("next_std", "") or "")
            if prev_std or next_std:
                bracket_label = f"{prev_std} -> {next_std}".strip()

        rows.append(
            {
                "Sample": sample.name,
                "Run": getattr(sample, "run_number", 0),
                "Type": sample.sample_type.upper(),
                "_sample_key": sample_key,
                "_run_number": getattr(sample, "run_number", 0),
                "Value": float(budget.ratio_value) if getattr(budget, "ratio_value", None) else 0.0,
                "n": n_cycles,
                "u_prec mode": u_prec_mode.upper(),
                "cycle SD": cycle_sd,
                "cycle SE": cycle_se,
                "Bracketing standards": bracket_label,
                "u_prec": metrics["u_prec"],
                "u_prec (‰)": metrics["u_prec_rel_permil"],
                "ν_prec": metrics["nu_prec"],
                "u_std": metrics["u_std"],
                "u_std (‰)": metrics["u_std_rel_permil"],
                "ν_std": metrics["nu_std"],
                "u_local_precision": metrics["u_local_precision"],
                "u_local_precision (‰)": metrics["u_local_precision_rel_permil"],
                "ν_local": metrics["nu_local_precision"],
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("_run_number")
        .drop(columns=["_run_number"])
        .reset_index(drop=True)
    )


def _format_measurement_precision_detail_dataframe(
    detail_df: pd.DataFrame,
    *,
    is_engine_a: bool = False,
    show_u_prec: bool = True,
    show_u_std: bool = True,
) -> pd.DataFrame:
    """Format the measurement-precision detail table for display."""
    display_df = detail_df.copy()
    if "_sample_key" in display_df.columns:
        display_df = display_df.drop(columns=["_sample_key"])
    if "Value" in display_df.columns:
        display_df["Value"] = display_df["Value"].map(
            lambda v: f"{float(v):.8f}" if float(v) != 0.0 else "—"
        )
    for column in ("ν_prec", "ν_std", "ν_local"):
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda value: "∞" if value == float("inf") else str(int(round(float(value))))
            )
    if is_engine_a:
        drop_cols = [
            c for c in (
                "Bracketing standards",
                "u_std", "u_std (‰)", "ν_std",
                "u_local_precision", "u_local_precision (‰)", "ν_local",
            )
            if c in display_df.columns
        ]
        display_df = display_df.drop(columns=drop_cols)
    else:
        drop_cols = []
        if not show_u_prec:
            drop_cols.extend(["u_prec mode", "cycle SD", "cycle SE"])
            drop_cols.extend(["n", "u_prec", "u_prec (‰)", "ν_prec"])
            drop_cols.extend(
                c for c in display_df.columns
                if c.startswith("u_prec") or c.endswith("_prec")
            )
        if not show_u_std:
            drop_cols.extend([
                "Bracketing standards",
                "u_std", "u_std (‰)", "ν_std",
                "u_local_precision", "u_local_precision (‰)", "ν_local",
            ])
            drop_cols.extend(
                c for c in display_df.columns
                if c.startswith("u_std")
                or c.endswith("_std")
                or c.startswith("u_local_precision")
                or c.endswith("_local")
            )
        elif not show_u_prec:
            drop_cols.extend(["u_local_precision", "u_local_precision (‰)", "ν_local"])
            drop_cols.extend(
                c for c in display_df.columns
                if c.startswith("u_local_precision") or c.endswith("_local")
            )
        display_df = display_df.drop(
            columns=[c for c in drop_cols if c in display_df.columns]
        )
    return display_df

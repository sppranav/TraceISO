"""Summary table component for TraceISO."""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import streamlit as st
from config.constants import DEFAULT_OUTLIER_THRESHOLD_SD
from file_io.sanitize import round_uncertainty_up, sanitize_spreadsheet_text
from file_io.summary_writer import build_summary_excel as _build_summary_excel
from domain.models import ProcessingResult, Sample
from domain.ratio_selection import get_ssb_cycle_data
from domain.runtime_delta import (
    RuntimeDeltaResult,
    lookup_runtime_delta,
)
from domain.uncertainty.runtime import (
    RuntimeUncertaintyMap,
    lookup_runtime_budget,
)
from domain.uncertainty.scope import (
    budget_scope_label,
    is_invalid_budget_scope,
)
from ui.formatting import format_uncertainty


@dataclass
class TableColumn:
    """Definition of a summary table column."""

    key: str
    label: str
    format: str = ".6f"
    description: str = ""


#: Column order is the default reading order (U51): measurement, then the
#: uncertainty with its coverage factor, then review context, then detailed
#: precision metrics and intermediate correction stages. Visible labels carry
#: their unit (U47) and the notation 2 SD, 2 SE, u_c, U, Coverage k (U48); the
#: coverage factor lives in its own column rather than in every label.
AVAILABLE_COLUMNS = [
    # Measurement
    TableColumn("mean", "Measured Mean", ".6f", "Measured ratio mean of the accepted cycles, before SSB normalization"),
    TableColumn("reported_value", "Reported value", ".6f", "Measurand to which U and u_c apply"),
    TableColumn("reported_unit", "Reported unit", "", "Unit of the reported value, U and u_c"),
    TableColumn("delta", "Delta (\u2030)", ".2f", "Delta value versus the bracketing standard, in permil"),
    # Uncertainty
    TableColumn(
        "delta_unc",
        "Delta U (\u2030)",
        ".2f",
        "Expanded GUM uncertainty U of the delta, in permil, at the coverage factor in Coverage k",
    ),
    TableColumn(
        "u_expanded",
        "U",
        ".6f",
        "Expanded uncertainty U in the reported output unit (ratio units, or permil for "
        "delta output), at the coverage factor in Coverage k",
    ),
    TableColumn("k_coverage", "Coverage k", ".2f", "Coverage factor k of U (fixed or Welch-Satterthwaite)"),
    TableColumn(
        "u_exp_permil",
        "Relative U (\u2030)",
        ".4f",
        "Expanded uncertainty U relative to the ratio, in permil; not a delta-unit value",
    ),
    TableColumn(
        "u_combined",
        "u_c",
        ".6f",
        "Combined standard uncertainty u_c in the reported output unit",
    ),
    TableColumn(
        "u_c_permil",
        "Relative u_c (\u2030)",
        ".4f",
        "Combined standard uncertainty u_c relative to the ratio, in permil; not a delta-unit value",
    ),
    # Review context
    TableColumn("n_valid", "Accepted cycles", "d", "Cycles accepted by the active window, mask and filter"),
    TableColumn("n_total", "Total cycles", "d", "All recorded cycles for the ratio"),
    TableColumn(
        "budget_status",
        "Uncertainty status",
        "",
        "Availability of the uncertainty budget",
    ),
    TableColumn("dominant", "Dominant contributor", "", "Largest contributor to the uncertainty budget"),
    # Detailed precision
    TableColumn("2sd", "2 SD", ".6f", "Two standard deviations of the accepted cycles (ddof = 1)"),
    TableColumn("2se", "2 SE", ".6f", "Two standard errors of the mean of the accepted cycles"),
    TableColumn("rsd", "RSD (%)", ".4f", "Relative standard deviation, in percent"),
    TableColumn("rse", "RSE (%)", ".4f", "Relative standard error, in percent"),
    TableColumn("delta_2se", "Delta 2 SE (\u2030)", ".2f", "Two standard errors of the delta cycles, in permil"),
    TableColumn("delta_2sd", "Delta 2 SD (\u2030)", ".2f", "Two standard deviations of the delta cycles, in permil"),
    # Intermediate correction stages
    TableColumn("iif_corrected", "IIF Corrected", ".6f", "IIF-corrected mean ratio"),
    TableColumn("ssb_corrected", "SSB Corrected", ".6f", "SSB-corrected mean ratio"),
    TableColumn("drift_corrected", "Drift-Corrected", ".6f", "Drift-corrected mean ratio"),
    TableColumn("ssb_k", "SSB K-factor", ".6f", "SSB correction K-factor (not the coverage factor)"),
]

CORE_STATS_COLUMNS = {"mean", "2sd", "2se", "rsd", "rse", "n_valid", "n_total"}

#: Selectable but not shown until chosen (U51).
DETAIL_COLUMNS = frozenset({
    "u_combined", "u_c_permil", "n_total", "dominant",
    "2sd", "2se", "rsd", "rse", "delta_2se", "delta_2sd",
    "iif_corrected", "ssb_corrected", "drift_corrected", "ssb_k",
})
_UNCERTAINTY_KEYS = frozenset({
    "delta_unc", "u_expanded", "k_coverage", "u_exp_permil", "u_combined", "u_c_permil",
})

METRIC_SUFFIX = {col.key: col.label for col in AVAILABLE_COLUMNS}

SUMMARY_NOTATION_NOTE = (
    "Measured Mean describes the measured ratio layer; it can differ from the "
    "Reported value to which U and u_c apply. "
    "U is the expanded uncertainty at the factor in Coverage k; u_c is the combined "
    "standard uncertainty. U and u_c are in the reported unit (ratio units, or \u2030 "
    "for delta output); Relative values are \u2030 of the ratio."
)

METRIC_FORMAT = {
    "reported_value": "{:.6f}",
    "reported_unit": "{}",
    "mean": "{:.6f}",
    "2sd": "{:.6f}",
    "2se": "{:.6f}",
    "rsd": "{:.4f}",
    "rse": "{:.4f}",
    "n_valid": "{:.0f}",
    "n_total": "{:.0f}",
    "iif_corrected": "{:.6f}",
    "ssb_corrected": "{:.6f}",
    "drift_corrected": "{:.6f}",
    "delta": "{:.2f}",
    "delta_2se": "{:.2f}",
    "delta_2sd": "{:.2f}",
    "delta_unc": "{:.2f}",
    "ssb_k": "{:.6f}",
    "u_combined": "{:.6f}",
    "u_expanded": "{:.6f}",
    "u_c_permil": "{:.4f}",
    "u_exp_permil": "{:.4f}",
    "k_coverage": "{:.2f}",
    "dominant": "{}",
    "budget_status": "{}",
}

UNCERTAINTY_METRICS = {
    "delta_unc",
    "u_combined",
    "u_expanded",
    "u_c_permil",
    "u_exp_permil",
}


def get_column_options() -> Dict[str, str]:
    """Get column key -> label mapping for UI selection."""
    return {col.key: col.label for col in AVAILABLE_COLUMNS}


def _ordered_columns(columns: Set[str]) -> List[str]:
    return [col.key for col in AVAILABLE_COLUMNS if col.key in columns]


def default_summary_columns(available: Set[str]) -> List[str]:
    """Default columns: measurement, U with its coverage factor, and review context (U51).

    Detailed precision and intermediate stages stay selectable. When no
    uncertainty budget is available a mean alone would carry no dispersion,
    so 2 SE (and Delta 2 SE) stand in.
    """
    keys = [key for key in _ordered_columns(set(available)) if key not in DETAIL_COLUMNS]
    if not set(available) & _UNCERTAINTY_KEYS:
        keys += [key for key in ("2se", "delta_2se") if key in available]
    return _ordered_columns(set(keys))


def _ratio_metric_column_name(ratio_name: str, metric_key: str) -> str:
    # Table and spreadsheet headers should remain portable and readable in
    # fonts that do not render Unicode superscript digits consistently.
    ratio_label = ratio_name
    suffix = METRIC_SUFFIX[metric_key]
    return ratio_label if not suffix else f"{ratio_label} {suffix}"


def _sanitize_summary_export_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a spreadsheet-safe copy of the visible summary data."""
    out = df.copy()
    # Inspect values rather than selecting the legacy ``object`` dtype.  This
    # also covers pandas' dedicated string dtype and avoids the pandas 3/4
    # transition warning about implicit string selection.
    for col_name in out.columns:
        out[col_name] = out[col_name].map(
            lambda value: sanitize_spreadsheet_text(value)
            if isinstance(value, str)
            else value
        )
    return out


def _round_summary_uncertainties_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with uncertainty columns rounded upward per GUM 7.2.6."""
    out = df.copy()
    uncertainty_suffixes = tuple(
        f" {METRIC_SUFFIX[key]}" for key in UNCERTAINTY_METRICS
    )
    for col_name in out.columns:
        if not str(col_name).endswith(uncertainty_suffixes):
            continue
        out[col_name] = out[col_name].map(
            lambda value: value
            if pd.isna(value)
            else round_uncertainty_up(float(value))
        )
    return out


def _format_summary_metric_value(metric_key: str, value: object) -> str:
    """Format one visible summary value using its metric-specific rule."""
    if pd.isna(value):
        return "-"
    if metric_key in UNCERTAINTY_METRICS:
        return format_uncertainty(float(value))
    return METRIC_FORMAT[metric_key].format(value)


def build_summary_csv(df: pd.DataFrame) -> bytes:
    """Build an Excel-friendly UTF-8 CSV containing the visible summary table."""
    export_df = _sanitize_summary_export_dataframe(
        _round_summary_uncertainties_for_export(df)
    )
    return export_df.to_csv(index=False).encode("utf-8-sig")


def build_summary_excel(df: pd.DataFrame) -> bytes:
    """Build a compact Excel workbook containing the visible summary table."""
    export_df = _sanitize_summary_export_dataframe(
        _round_summary_uncertainties_for_export(df)
    )
    return _build_summary_excel(export_df)


def _get_summary_ratio_data(sample: Sample, ratio_name: str):
    """Return the ratio layer used for the plain summary-stat columns.

    The unqualified ratio column should describe measured ratio statistics after
    ordinary processing and filtering. SSB-corrected cycles are intentionally
    excluded here because they have their own explicit summary-table column.
    """
    if sample.blank_corrected_ratios and ratio_name in sample.blank_corrected_ratios:
        return sample.blank_corrected_ratios[ratio_name]
    if sample.corrected_ratios and ratio_name in sample.corrected_ratios:
        return sample.corrected_ratios[ratio_name]
    if sample.ratios and ratio_name in sample.ratios:
        return sample.ratios[ratio_name]
    return None


def _resolve_delta_metrics(
    sample: Sample,
    ratio_name: str,
    delta_overrides: Optional[Dict[tuple, RuntimeDeltaResult]] = None,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return effective delta mean, 2SE, and 2SD for one sample/ratio pair."""
    if delta_overrides is not None:
        override = lookup_runtime_delta(delta_overrides, sample, ratio_name)
        if override is not None:
            return override.delta, override.delta_2se, 2.0 * override.delta_sd
        return None, None, None

    if not sample.delta_results:
        return None, None, None

    delta_data = sample.delta_results.get(ratio_name, {})
    delta_sd = delta_data.get("delta_sd")
    delta_2sd = 2.0 * delta_sd if delta_sd is not None else None
    delta_2se = delta_data.get("delta_2se")
    if delta_2se is None and delta_data.get("delta_se") is not None:
        delta_2se = 2.0 * delta_data["delta_se"]
    return delta_data.get("delta"), delta_2se, delta_2sd


def _resolve_uncertainty_budget(
    sample: Sample,
    ratio_name: str,
    uncertainty_overrides: Optional[RuntimeUncertaintyMap] = None,
):
    """Return the runtime budget when available, otherwise the stored budget."""
    budget = None
    if uncertainty_overrides is not None:
        budget = lookup_runtime_budget(uncertainty_overrides, sample, ratio_name)
    if budget is None and sample.uncertainty and ratio_name in sample.uncertainty:
        budget = sample.uncertainty[ratio_name]
    return budget


def _resolve_delta_reported_uncertainty(budget) -> Optional[float]:
    """Return expanded delta uncertainty in permil, only for SSB-delta budgets.

    ``output_mode`` is engine-gated in shared_engine.py: only engine_ssb (the
    Li/B/Mg/Cd/Pb SSB-delta workflow) can report "delta" here, so this also
    keeps the column off Sr/Pb-Tl tables where no delta is computed.
    """
    if budget is None or is_invalid_budget_scope(budget):
        return None
    if getattr(budget, "output_mode", "") != "delta":
        return None
    value = getattr(budget, "expanded_abs", None)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def infer_available_columns(
    samples: List[Sample],
    ratio_names: List[str],
    uncertainty_overrides: Optional[RuntimeUncertaintyMap] = None,
    delta_overrides: Optional[Dict[tuple, RuntimeDeltaResult]] = None,
) -> Set[str]:
    """Infer which optional metrics are actually available in current data."""
    available = set(CORE_STATS_COLUMNS)

    for sample in samples:
        for ratio_name in ratio_names:
            if sample.iif_corrected_ratios and ratio_name in sample.iif_corrected_ratios:
                available.add("iif_corrected")

            if sample.ssb_results and ratio_name in sample.ssb_results:
                ssb_data = sample.ssb_results.get(ratio_name, {})
                cycles = ssb_data.get("ssb_corrected_cycles")
                if cycles is not None and len(cycles) > 0:
                    available.add("ssb_corrected")
                if "k_factor" in ssb_data:
                    available.add("ssb_k")

            if sample.drift_corrected_ratios and ratio_name in sample.drift_corrected_ratios:
                available.add("drift_corrected")

            has_runtime_delta = (
                delta_overrides is not None
                and lookup_runtime_delta(delta_overrides, sample, ratio_name) is not None
            )
            delta_value, delta_2se, delta_2sd = _resolve_delta_metrics(
                sample, ratio_name, delta_overrides
            )
            if has_runtime_delta or delta_value is not None:
                available.add("delta")
            if has_runtime_delta or delta_2se is not None:
                available.add("delta_2se")
            if has_runtime_delta or delta_2sd is not None:
                available.add("delta_2sd")

            has_runtime_unc = (
                uncertainty_overrides is not None
                and lookup_runtime_budget(
                    uncertainty_overrides, sample, ratio_name
                ) is not None
            )
            budget = _resolve_uncertainty_budget(
                sample, ratio_name, uncertainty_overrides
            )
            if budget is not None and is_invalid_budget_scope(budget):
                available.add("budget_status")
                continue
            if _resolve_delta_reported_uncertainty(budget) is not None:
                available.add("delta_unc")
            if has_runtime_unc or budget is not None:
                available.update({"reported_value", "reported_unit"})
                available.add("u_combined")
                available.add("u_expanded")
                available.add("u_c_permil")
                available.add("u_exp_permil")
                available.add("k_coverage")
                available.add("dominant")

    return available


def render_column_selector(
    default_columns: Optional[List[str]] = None,
    key: str = "summary_columns",
    available_columns: Optional[Set[str]] = None,
) -> Set[str]:
    """Render checkboxes for column selection and return selected keys."""
    candidates = AVAILABLE_COLUMNS
    if available_columns is not None:
        candidates = [c for c in AVAILABLE_COLUMNS if c.key in available_columns]

    if default_columns is None:
        default_columns = default_summary_columns({c.key for c in candidates})
    else:
        default_columns = [k for k in default_columns if any(c.key == k for c in candidates)]

    st.caption("Select metrics to display:")
    cols = st.columns(4)
    selected = set()

    for i, col_def in enumerate(candidates):
        with cols[i % 4]:
            is_selected = st.checkbox(
                col_def.label,
                value=col_def.key in default_columns,
                key=f"{key}_{col_def.key}",
                help=col_def.description,
            )
            if is_selected:
                selected.add(col_def.key)

    return selected


def _format_dominant_contributor(budget) -> str:
    """Format dominant contributor name, with warning if budget is incomplete."""
    from config.contributor_names import canonical_contributor_display_label

    name = budget.dominant_contributor
    if not name:
        text = "-"
    else:
        text = canonical_contributor_display_label(name) or (
            name.replace("_", " ").replace("u ", "").strip().title()
        )
    # Check for incomplete budget (any inactive contributor)
    if hasattr(budget, "contributors") and budget.contributors:
        inactive = [c for c in budget.contributors if not c.is_active]
        if inactive:
            text += " \u26a0\ufe0f"
    return text


def _prepare_display_dataframe(display_df: pd.DataFrame) -> pd.DataFrame:
    """Escape user-controlled text before rendering HTML."""
    out = display_df.copy()
    for col_name in out.columns:
        if col_name == "Type":
            continue
        out[col_name] = out[col_name].apply(
            lambda val: html.escape(val, quote=True) if isinstance(val, str) else val
        )

    type_badges = {
        "STD": '<span class="summary-badge summary-std">STD</span>',
        "SMP": '<span class="summary-badge summary-smp">SMP</span>',
        "BLK": '<span class="summary-badge summary-blk">BLK</span>',
    }
    if "Type" in out.columns:
        out["Type"] = out["Type"].map(
            lambda val: type_badges.get(str(val).upper(), html.escape(str(val), quote=True))
        )

    if "Sample" in out.columns:
        # Sample column is truncated with ellipsis (fixed sticky width); title
        # keeps the full name reachable on hover.
        out["Sample"] = out["Sample"].apply(lambda val: f'<span title="{val}">{val}</span>')

    return out


def build_summary_dataframe(
    samples: List[Sample],
    ratio_names: List[str],
    columns: Set[str],
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
    uncertainty_overrides: Optional[RuntimeUncertaintyMap] = None,
    delta_overrides: Optional[Dict[tuple, RuntimeDeltaResult]] = None,
) -> pd.DataFrame:
    """Build a summary DataFrame for multiple ratios."""
    from domain.filters.outlier import get_filtered_values, sample_cycle_key

    ordered_metric_keys = _ordered_columns(columns)
    ratio_metric_columns: List[str] = []
    for ratio_name in ratio_names:
        for metric_key in ordered_metric_keys:
            ratio_metric_columns.append(_ratio_metric_column_name(ratio_name, metric_key))

    rows = []
    for sample in samples:
        row = {
            "Sample": sample.name,
            "Type": sample.sample_type.upper(),
            "Run Number": sample.run_number,
        }
        for col_name in ratio_metric_columns:
            row[col_name] = np.nan

        for ratio_name in ratio_names:
            ratio_data = _get_summary_ratio_data(sample, ratio_name)
            if ratio_data is None:
                continue

            valid_values = get_filtered_values(
                ratio_data.values,
                ratio_data.mask,
                sample.name,
                sample_key=sample_cycle_key(sample),
                cycle_ranges=cycle_ranges,
                filter_method=filter_method,
                filter_threshold=filter_threshold,
            )
            n_valid = len(valid_values)
            n_total = len(ratio_data.values)
            for metric_key, count in (("n_valid", n_valid), ("n_total", n_total)):
                if metric_key in columns:
                    row[_ratio_metric_column_name(ratio_name, metric_key)] = count

            if n_valid > 0:
                mean_val = float(np.nanmean(valid_values))
                std_val = float(np.nanstd(valid_values, ddof=1)) if n_valid > 1 else float("nan")
                se_val = std_val / np.sqrt(n_valid) if n_valid > 1 else float("nan")
                rsd_pct = (std_val / mean_val * 100.0) if mean_val != 0 else np.nan
                rse_pct = (se_val / mean_val * 100.0) if mean_val != 0 else np.nan

                metric_values = {
                    "mean": mean_val,
                    "2sd": 2 * std_val,
                    "2se": 2 * se_val,
                    "rsd": rsd_pct,
                    "rse": rse_pct,
                    "n_valid": n_valid,
                    "n_total": n_total,
                }
                for metric_key, metric_val in metric_values.items():
                    if metric_key in columns:
                        row[_ratio_metric_column_name(ratio_name, metric_key)] = metric_val

            if sample.iif_corrected_ratios and "iif_corrected" in columns:
                if ratio_name in sample.iif_corrected_ratios:
                    iif_data = sample.iif_corrected_ratios[ratio_name]
                    iif_values = get_filtered_values(
                        iif_data.values,
                        iif_data.mask,
                        sample.name,
                        sample_key=sample_cycle_key(sample),
                        cycle_ranges=cycle_ranges,
                        filter_method=filter_method,
                        filter_threshold=filter_threshold,
                    )
                    if len(iif_values) > 0:
                        row[_ratio_metric_column_name(ratio_name, "iif_corrected")] = float(
                            np.nanmean(iif_values)
                        )

            if sample.ssb_results and "ssb_corrected" in columns:
                ssb_cd = get_ssb_cycle_data(sample, ratio_name)
                if ssb_cd is not None:
                    ssb_values = get_filtered_values(
                        ssb_cd.values,
                        ssb_cd.mask,
                        sample.name,
                        sample_key=sample_cycle_key(sample),
                        cycle_ranges=cycle_ranges,
                        filter_method=filter_method,
                        filter_threshold=filter_threshold,
                    )
                    if len(ssb_values) > 0:
                        row[_ratio_metric_column_name(ratio_name, "ssb_corrected")] = float(np.nanmean(ssb_values))

            if sample.drift_corrected_ratios and "drift_corrected" in columns:
                if ratio_name in sample.drift_corrected_ratios:
                    drift_data = sample.drift_corrected_ratios[ratio_name]
                    drift_vals = get_filtered_values(
                        drift_data.values,
                        drift_data.mask,
                        sample.name,
                        sample_key=sample_cycle_key(sample),
                        cycle_ranges=cycle_ranges,
                        filter_method=filter_method,
                        filter_threshold=filter_threshold,
                    )
                    if len(drift_vals) > 0:
                        row[_ratio_metric_column_name(ratio_name, "drift_corrected")] = float(np.nanmean(drift_vals))

            delta_value, delta_2se, delta_2sd = _resolve_delta_metrics(
                sample, ratio_name, delta_overrides
            )
            if "delta" in columns and delta_value is not None:
                row[_ratio_metric_column_name(ratio_name, "delta")] = delta_value
            if "delta_2se" in columns and delta_2se is not None:
                row[_ratio_metric_column_name(ratio_name, "delta_2se")] = delta_2se
            if "delta_2sd" in columns and delta_2sd is not None:
                row[_ratio_metric_column_name(ratio_name, "delta_2sd")] = delta_2sd

            if sample.ssb_results and "ssb_k" in columns:
                ssb_data = sample.ssb_results.get(ratio_name, {})
                if ssb_data and "k_factor" in ssb_data:
                    row[_ratio_metric_column_name(ratio_name, "ssb_k")] = ssb_data["k_factor"]

            budget = _resolve_uncertainty_budget(
                sample, ratio_name, uncertainty_overrides
            )
            if budget is not None:
                invalid_scope = is_invalid_budget_scope(budget)
                if "budget_status" in columns:
                    row[_ratio_metric_column_name(ratio_name, "budget_status")] = (
                        budget_scope_label(budget) if invalid_scope else "AVAILABLE"
                    )
                if not invalid_scope:
                    if "reported_value" in columns:
                        row[_ratio_metric_column_name(ratio_name, "reported_value")] = budget.reported_measurand_value()
                    if "reported_unit" in columns:
                        row[_ratio_metric_column_name(ratio_name, "reported_unit")] = (
                            "‰" if budget.output_mode == "delta" else "ratio"
                        )
                    delta_unc = _resolve_delta_reported_uncertainty(budget)
                    if "delta_unc" in columns and delta_unc is not None:
                        row[_ratio_metric_column_name(ratio_name, "delta_unc")] = delta_unc
                    if "u_combined" in columns:
                        row[_ratio_metric_column_name(ratio_name, "u_combined")] = budget.u_combined_abs
                    if "u_expanded" in columns:
                        row[_ratio_metric_column_name(ratio_name, "u_expanded")] = budget.expanded_abs
                    if "u_c_permil" in columns:
                        row[_ratio_metric_column_name(ratio_name, "u_c_permil")] = budget.u_combined_rel_permil
                    if "u_exp_permil" in columns:
                        row[_ratio_metric_column_name(ratio_name, "u_exp_permil")] = budget.expanded_rel_permil
                    if "k_coverage" in columns:
                        row[_ratio_metric_column_name(ratio_name, "k_coverage")] = budget.coverage_factor_k
                    if "dominant" in columns:
                        dominant_text = _format_dominant_contributor(budget)
                        row[_ratio_metric_column_name(ratio_name, "dominant")] = dominant_text

        rows.append(row)

    fixed_cols = ["Sample", "Type", "Run Number"]
    ordered_cols = fixed_cols + ratio_metric_columns
    df = pd.DataFrame(rows)
    return df.reindex(columns=ordered_cols)


def render_summary_table(
    result: ProcessingResult,
    ratio_name: Optional[str] = None,
    ratio_names: Optional[List[str]] = None,
    columns: Optional[Set[str]] = None,
    filter_types: Optional[List[str]] = None,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
    key: str = "summary_table",
    uncertainty_overrides: Optional[RuntimeUncertaintyMap] = None,
    delta_overrides: Optional[Dict[tuple, RuntimeDeltaResult]] = None,
) -> pd.DataFrame:
    """Render the summary table with formatting."""
    samples = [s for s in result.samples if not s.metadata.get("excluded", False)]

    if filter_types:
        samples = [s for s in samples if s.sample_type.upper() in filter_types]

    if not samples:
        st.info("No samples to display.")
        return pd.DataFrame()

    if ratio_names is None:
        if ratio_name is not None:
            ratio_names = [ratio_name]
        else:
            first_ratio = None
            for sample in samples:
                if sample.ratios:
                    first_ratio = list(sample.ratios.keys())[0]
                    break
            ratio_names = [first_ratio] if first_ratio else []

    ratio_names = [r for r in ratio_names if r]
    if not ratio_names:
        st.warning("No ratio data available.")
        return pd.DataFrame()

    sample_names = [s.name for s in samples]
    col_search, col_filter = st.columns([2, 3])
    with col_search:
        query = st.text_input(
            "Search sample names",
            value="",
            placeholder="Type part of a sample name...",
            key=f"{key}_sample_search",
        ).strip().lower()

    matching_names = [name for name in sample_names if query in name.lower()] if query else sample_names

    with col_filter:
        selected_names = st.multiselect(
            "Filter samples",
            options=matching_names,
            default=matching_names,
            key=f"{key}_sample_filter",
        )

    samples = [s for s in samples if s.name in selected_names and (query in s.name.lower() if query else True)]
    if not samples:
        st.info("No samples match the current name filters.")
        return pd.DataFrame()

    available_columns = infer_available_columns(
        samples,
        ratio_names,
        uncertainty_overrides=uncertainty_overrides,
        delta_overrides=delta_overrides,
    )
    if columns is None:
        columns = set(default_summary_columns(available_columns))
    else:
        columns = columns & available_columns
        if not columns:
            columns = set(default_summary_columns(available_columns))

    df = build_summary_dataframe(
        samples,
        ratio_names,
        columns,
        cycle_ranges,
        filter_method,
        filter_threshold,
        uncertainty_overrides=uncertainty_overrides,
        delta_overrides=delta_overrides,
    )

    ordered_metric_keys = _ordered_columns(columns)
    metric_by_column: Dict[str, str] = {}
    header_html: Dict[str, str] = {}
    for ratio_name in ratio_names:
        for metric_key in ordered_metric_keys:
            col_name = _ratio_metric_column_name(ratio_name, metric_key)
            if col_name in df.columns:
                metric_by_column[col_name] = metric_key
                # Ratio on the first line, metric on the second (U49).
                header_html[col_name] = (
                    f'<span class="summary-ratio">{html.escape(ratio_name, quote=True)}</span>'
                    f"<br>{html.escape(METRIC_SUFFIX[metric_key], quote=True)}"
                )

    display_df = df.copy()
    if "Run Number" in display_df.columns:
        display_df["Run Number"] = display_df["Run Number"].apply(
            lambda val: "-" if pd.isna(val) else "{:.0f}".format(val)
        )
    for col_name, metric_key in metric_by_column.items():
        fmt = METRIC_FORMAT[metric_key]
        if fmt == "{}":
            display_df[col_name] = display_df[col_name].fillna("-").astype(str)
        elif fmt == "{:.0f}":
            display_df[col_name] = display_df[col_name].apply(
                lambda val, fmt=fmt: "-" if pd.isna(val) else fmt.format(val)
            )
        else:
            display_df[col_name] = display_df[col_name].apply(
                lambda val, key=metric_key: _format_summary_metric_value(key, val)
            )

    display_df = _prepare_display_dataframe(display_df).rename(columns=header_html)
    n_metrics = len(ordered_metric_keys)
    group_starts = [4 + index * n_metrics for index in range(len(ratio_names))] if n_metrics else []

    st.markdown(_summary_table_css(group_starts), unsafe_allow_html=True)
    table_html = display_df.to_html(
        index=False,
        escape=False,
        classes=["summary-table"],
        border=0,
    )
    st.markdown(f'<div class="summary-table-wrap">{table_html}</div>', unsafe_allow_html=True)
    if metric_by_column and set(ordered_metric_keys) & _UNCERTAINTY_KEYS:
        st.caption(SUMMARY_NOTATION_NOTE)

    file_ratio_part = "multi_ratio" if len(ratio_names) > 1 else ratio_names[0]
    export_df = _sanitize_summary_export_dataframe(df)
    safe_ratio_part = file_ratio_part.replace("/", "_")
    csv = build_summary_csv(export_df)
    excel = build_summary_excel(export_df)
    csv_col, excel_col, spacer_col = st.columns([1, 1, 4])
    with csv_col:
        st.download_button(
            "Download CSV",
            csv,
            file_name=f"summary_{safe_ratio_part}.csv",
            mime="text/csv",
            key=f"{key}_download_csv",
            width="stretch",
        )
    with excel_col:
        st.download_button(
            "Download Excel",
            excel,
            file_name=f"summary_{safe_ratio_part}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key}_download_excel",
            width="stretch",
        )
    return df


def _summary_table_css(group_start_columns: List[int]) -> str:
    """Summary table styles: headings wrap, values never do (U49).

    ``group_start_columns`` are the 1-based column positions where each ratio's
    block begins; a rule on those columns separates the ratio groups.
    """
    group_rule = ""
    if group_start_columns:
        selectors = ",\n        ".join(
            f".summary-table th:nth-child({index}), .summary-table td:nth-child({index})"
            for index in group_start_columns
        )
        group_rule = f"""
        {selectors} {{
            border-left: 2px solid rgba(128,128,128,0.35);
        }}"""
    return """
        <style>
        .summary-table-wrap {
            overflow-x: auto;
            border: 1px solid rgba(128,128,128,0.2);
            border-radius: 8px;
        }
        .summary-table {
            border-collapse: separate;
            border-spacing: 0;
            min-width: max-content;
            width: 100%;
            font-size: var(--t-size-body, 14px);
            font-family: var(--t-sans, Arial, Helvetica, sans-serif);
            font-variant-numeric: tabular-nums;
        }
        .summary-table th,
        .summary-table td {
            padding: 0.45rem 0.7rem;
            border-bottom: 1px solid rgba(128,128,128,0.15);
            background: var(--background-color, inherit);
            color: inherit;
        }
        .summary-table td {
            white-space: nowrap;
        }
        .summary-table td:nth-child(n+4) {
            text-align: right;
        }
        .summary-table th {
            position: sticky;
            top: 0;
            z-index: 4;
            white-space: normal;
            vertical-align: bottom;
            text-align: left;
            min-width: 5.5rem;
            max-width: 10rem;
            font-family: var(--t-sans, Arial, Helvetica, sans-serif) !important;
            font-size: var(--t-size-control, 14px);
            font-weight: 700;
            letter-spacing: 0;
            line-height: 1.25;
            background: var(--t-bg-2, inherit);
        }
        .summary-ratio {
            font-weight: 600;
            font-size: var(--t-size-supporting, 13px);
        }
        .summary-table th:nth-child(1),
        .summary-table td:nth-child(1) {
            position: sticky;
            left: 0;
            z-index: 3;
            width: 180px;
            max-width: 180px;
            overflow: hidden;
            text-overflow: ellipsis;
            box-shadow: 1px 0 0 rgba(128,128,128,0.15);
            background: var(--background-color, inherit);
        }
        .summary-table th:nth-child(2),
        .summary-table td:nth-child(2) {
            position: sticky;
            left: 180px;
            z-index: 3;
            width: 90px;
            max-width: 90px;
            overflow: hidden;
            text-overflow: ellipsis;
            box-shadow: 1px 0 0 rgba(128,128,128,0.15);
            background: var(--background-color, inherit);
        }
        .summary-table th:nth-child(3),
        .summary-table td:nth-child(3) {
            position: sticky;
            left: 270px;
            z-index: 3;
            width: 80px;
            max-width: 80px;
            overflow: hidden;
            text-overflow: ellipsis;
            box-shadow: 1px 0 0 rgba(128,128,128,0.15);
            background: var(--background-color, inherit);
        }
        .summary-table th:nth-child(1),
        .summary-table th:nth-child(2),
        .summary-table th:nth-child(3) {
            z-index: 5;
            background: var(--t-bg-2, inherit);
        }
        .summary-table tbody tr:nth-child(odd) td {
            background: var(--background-color, inherit);
        }
        .summary-table tbody tr:nth-child(even) td {
            background: var(--t-bg-2, inherit);
        }
        .summary-badge {
            display: inline-block;
            padding: 0.1rem 0.45rem;
            border-radius: 999px;
            font-family: var(--t-mono, SFMono-Regular, Consolas, monospace);
            font-size: var(--t-size-supporting, 13px);
            font-weight: 600;
        }
        .summary-std { background: var(--std-soft); color: var(--t-ink); border: 1px solid var(--std); }
        .summary-smp { background: var(--smp-soft); color: var(--t-ink); border: 1px solid var(--smp); }
        .summary-blk { background: var(--blk-soft); color: var(--t-ink); border: 1px solid var(--blk); }""" + group_rule + """
        </style>
        """

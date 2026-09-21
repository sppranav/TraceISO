"""CSV export for TraceISO."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config.constants import DEFAULT_OUTLIER_THRESHOLD_SD
from domain.corrections.delta import compute_delta_stats
from domain.filters.outlier import get_runtime_mask, sample_cycle_key
from domain.models import CycleData, ProcessingResult, Sample
from domain.ratio_selection import (
    get_delta_cycle_data,
    get_ssb_cycle_data,
    select_summary_export_layer,
)
from domain.statistics import calculate_cycle_statistics
from domain.uncertainty.mc_result import (
    MCResultRecord,
    has_mc_results,
    log_mc_results_exported,
    mc_record_freshness_for_budget,
)
from domain.uncertainty.scope import (
    budget_scope_label,
    budget_scope_note,
    canonical_uncertainty_scope_json,
    is_invalid_budget_scope,
    uncertainty_scope_payload,
)
from file_io.sanitize import (
    build_budget_note as _build_budget_note,
    format_contributor_name,
    sanitize_spreadsheet_text,
    summarize_budget_state,
)
from domain.provenance import canonical_provenance_json, normalize_provenance
from file_io.report_formatting import report_uncertainty


# All column keys for a full CSV export.
_ALL_COLUMNS: Set[str] = {
    "mean",
    "2sd",
    "2se",
    "rsd",
    "rse",
    "n_valid",
    "n_total",
    "ssb_corrected",
    "delta",
    "delta_unc",
    "ssb_k",
    "u_combined",
    "u_expanded",
    "u_c_permil",
    "u_exp_permil",
    "k_coverage",
    "dominant",
    "budget_note",
    "budget_state",
}

_BASE_COLUMNS = ["Sample", "Type", "Run Number", "Ratio", "Layer"]

#: CSV variants (U64). The complete audit CSV repeats the canonical provenance
#: and uncertainty-scope JSON, and each Monte Carlo canonical record, on every
#: row, so any single row is self-describing. The compact CSV carries the same
#: rows, scientific columns and values, and replaces those repeated payloads
#: with short identity columns; its canonical evidence lives in the complete CSV
#: or the JSON export.
CSV_VARIANT_COMPLETE = "complete"
CSV_VARIANT_COMPACT = "compact"
CSV_VARIANTS = (CSV_VARIANT_COMPLETE, CSV_VARIANT_COMPACT)
_PROVENANCE_JSON_COLUMN = "TraceISO Provenance (JSON)"
_SCOPE_JSON_COLUMN = "TraceISO Uncertainty Scope (JSON)"
_MC_RECORD_JSON_COLUMN = "MC Canonical Record (JSON)"

#: Format token used in the ``engine_b_mc.exported`` lifecycle record. It is a
#: fixed word, never a filename or a path.
_MC_EXPORT_FORMAT = "csv"

#: Header for the combined standard uncertainty column. JCGM 100 section
#: 7.2.2 warns that presenting u_c as an apparent k = 1 expanded interval can
#: be misunderstood, so the label names the quantity the column carries. It is
#: defined once because the header appears in the column map, the row builder
#: and the column-order list, and those three drifted apart before.
_U_COMBINED_COLUMN = "u_c"

#: Header for the expanded uncertainty column. It is deliberately neutral.
#: The column previously read ``U(k=2)``, but ``coverage_method`` may be
#: ``welch_satterthwaite``, in which case the engine's coverage factor is
#: derived per budget from the effective degrees of freedom and is routinely
#: not 2 -- and ``coverage_k`` itself is configurable. A fixed label on a
#: variable quantity contradicted the ``k`` column standing beside it. The
#: actual coverage factor for every row is in that ``k`` column, which is
#: emitted whenever this one is (JCGM 100 section 6.3.1: an expanded
#: uncertainty is meaningless without its coverage factor).
_U_EXPANDED_COLUMN = "U"

#: Monte Carlo cross-check columns, in output order.
#:
#: These carry the canonical full-precision machine values — they are the same
#: numbers the JSON, Excel and HDF5 exports write, deliberately *not* run
#: through ``round_uncertainty_up``, because a CSV is a machine-readable
#: interchange format and presentation rounding is applied at display time.
#:
#: The labels follow the GUM convention this phase adopts: ``u_c`` names a
#: combined standard uncertainty, and a coverage interval is never reported
#: without its probability and method. The Monte Carlo dispersion column is
#: named for what it is — a sample standard deviation at ``ddof=1``.
#:
#: **Reading these columns back.** The text written here is the exact double —
#: every value's ``repr`` appears verbatim in the file. Pandas, however, does
#: not read it back at full precision by default: its C parser truncates to
#: roughly 13 significant digits unless ``float_precision="round_trip"`` is
#: passed. Use ``pd.read_csv(path, float_precision="round_trip")`` when the
#: full canonical precision matters. The file is exact either way; only the
#: reader's default is lossy.
#:
#: Note also that the GUM budget columns ``u_c`` and ``U`` are *not* MC
#: columns. ``u_c`` carries ``budget.u_combined_abs``, the combined standard
#: uncertainty, and ``U`` carries ``budget.expanded_abs``, the expanded
#: uncertainty at the coverage factor in the ``k`` column of the same row. The absolute column was previously headed ``U(k=1)``,
#: which presented a combined standard uncertainty as though it were an
#: expanded uncertainty -- the misreading JCGM 100 section 7.2.2 warns about.
#: It was renamed to match the permil column ``u_c (permil)`` and the MC
#: block, so every reported standard uncertainty now uses one label. Readers
#: of the old header must map ``U(k=1)`` to ``u_c``; the values are unchanged.
#:
#: The whole block is emitted only when at least one durable record exists, so
#: an export from a session that never ran Monte Carlo is unchanged.
_MC_COLUMNS: List[str] = [
    "MC Semantics",
    "MC Result Space",
    # Freshness is an export-time relationship between the stored result and
    # the budget in the same row, not a property of the stored record. It is
    # carried here because a reader must never have to assume that a Monte
    # Carlo result still corresponds to the data beside it.
    "MC Result Freshness",
    "MC Execution ID",
    "MC Requested Draws",
    "MC Completed Draws",
    "MC Seed",
    "MC Mean",
    "MC SD (ddof=1)",
    "MC Interval Lower",
    "MC Interval Upper",
    "MC Coverage Probability",
    "MC Interval Method",
    "MC u_c (GUM)",
    "MC U (GUM expanded)",
    "MC Coverage Factor k",
    # Exact, complete schema payload. The flattened columns above remain
    # convenient for analysis; this column makes CSV a lossless interchange
    # format for RNG/config/input/software and transform provenance too.
    "MC Canonical Record (JSON)",
]


def export_to_csv(
    result: ProcessingResult,
    *,
    ratio_name: Optional[str] = None,
    ratio_names: Optional[List[str]] = None,
    include_uncertainty: bool = True,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
    uncertainty_config: Optional[object] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    variant: str = CSV_VARIANT_COMPLETE,
) -> str:
    """Export a ProcessingResult summary table as a CSV string."""
    if variant not in CSV_VARIANTS:
        raise ValueError(f"Unknown CSV variant {variant!r}; expected one of {CSV_VARIANTS}")
    samples = result.samples
    if not samples:
        return ""

    resolved_ratio_names = _resolve_ratio_names(samples, ratio_name, ratio_names)
    if not resolved_ratio_names:
        return ""

    columns = set(_ALL_COLUMNS)
    if not include_uncertainty:
        columns -= {
            "u_combined",
            "u_expanded",
            "u_c_permil",
            "u_exp_permil",
            "k_coverage",
            "dominant",
            "budget_note",
            "budget_state",
        }

    include_mc = include_uncertainty and has_mc_results(samples)

    df = _build_dataframe(
        samples,
        resolved_ratio_names,
        columns,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
        include_mc=include_mc,
        uncertainty_config=uncertainty_config,
    )
    has_budgets = include_uncertainty and any(sample.uncertainty for sample in samples)
    if variant == CSV_VARIANT_COMPACT:
        df = df.drop(columns=[_MC_RECORD_JSON_COLUMN], errors="ignore")
        if provenance is not None:
            digest = normalize_provenance(provenance).get("effective_configuration_sha256")
            if digest:
                df["Effective configuration SHA-256"] = sanitize_spreadsheet_text(str(digest))
        if has_budgets:
            scope = uncertainty_scope_payload()
            df["Uncertainty scope schema"] = f"{scope['schema_name']} v{scope['schema_version']}"
    else:
        if provenance is not None:
            canonical = canonical_provenance_json(normalize_provenance(provenance))
            df[_PROVENANCE_JSON_COLUMN] = canonical
        if has_budgets:
            df[_SCOPE_JSON_COLUMN] = canonical_uncertainty_scope_json()
    if include_mc:
        log_mc_results_exported(samples, export_format=_MC_EXPORT_FORMAT)
    try:
        return df.to_csv(index=False)
    except Exception as exc:
        raise ValueError(
            f"CSV export failed for {len(samples)} sample(s) and "
            f"{len(resolved_ratio_names)} ratio(s): {exc}"
        ) from exc


def _build_dataframe(
    samples: List[Sample],
    ratio_names: List[str],
    columns: Set[str],
    *,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
    include_mc: bool = False,
    uncertainty_config: Optional[object] = None,
) -> pd.DataFrame:
    """Build a long-form summary DataFrame."""
    rows = []

    for sample in samples:
        for ratio_name in ratio_names:
            row: Dict[str, object] = {
                "Sample": sanitize_spreadsheet_text(sample.name),
                "Type": sanitize_spreadsheet_text(sample.sample_type.upper()),
                "Run Number": sample.run_number,
                "Ratio": sanitize_spreadsheet_text(ratio_name),
                "Layer": "-",
                "Observation ID": sanitize_spreadsheet_text(sample.observation_id),
            }

            _optional_cols = {
                "mean": "Mean",
                "2sd": "2SD",
                "2se": "2SE",
                "rsd": "RSD%",
                "rse": "RSE%",
                "n_valid": "n",
                "n_total": "N",
                "ssb_corrected": "SSB Corrected",
                "delta": "Delta",
                "delta_unc": "Delta 2SE (‰)",
                "ssb_k": "K-factor",
                "u_combined": _U_COMBINED_COLUMN,
                "u_expanded": _U_EXPANDED_COLUMN,
                "u_c_permil": "u_c (permil)",
                "u_exp_permil": "U (permil)",
                "k_coverage": "k",
                "dominant": "Dominant",
                "budget_note": "Budget Note",
            }
            for col_key, col_label in _optional_cols.items():
                if col_key in columns:
                    row[col_label] = np.nan

            ratio_data, ratio_layer = _get_export_ratio_data(sample, ratio_name)
            if ratio_data is None:
                from domain.sr_standard_calibration import sr_calibration_message
                refusal = sr_calibration_message(sample, ratio_name)
                if refusal:
                    row["Layer"] = "Sr calibration unavailable"
                    row["Budget Note"] = sanitize_spreadsheet_text(refusal)
                rows.append(row)
                continue

            row["Layer"] = sanitize_spreadsheet_text(ratio_layer)

            # One prepared selection for the whole row. The window and
            # filter arguments used to reach the main statistics only, so a
            # row could promise n = 2 while its SSB and delta columns still
            # described the full run. Downstream layers inherit the mask
            # defined on the reported ratio layer, exactly as the pipeline's
            # canonical mask policy requires.
            selection = get_runtime_mask(
                ratio_data.values,
                ratio_data.mask,
                sample.name,
                cycle_ranges=cycle_ranges,
                sample_key=sample_cycle_key(sample),
                filter_method=filter_method,
                filter_threshold=filter_threshold,
            )
            valid = ratio_data.values[selection]
            n_valid = len(valid)
            n_total = len(ratio_data.values)
            if "n_valid" in columns:
                row["n"] = n_valid
            if "n_total" in columns:
                row["N"] = n_total

            if n_valid > 0:
                stats = calculate_cycle_statistics(valid)
                mean_val = stats.mean
                std_val = stats.sd
                se_val = stats.se
                rsd_pct = stats.rsd_percent
                rse_pct = stats.rse_percent

                if "mean" in columns:
                    row["Mean"] = mean_val
                if "2sd" in columns:
                    row["2SD"] = 2 * std_val if np.isfinite(std_val) else np.nan
                if "2se" in columns:
                    row["2SE"] = 2 * se_val if np.isfinite(se_val) else np.nan
                if "rsd" in columns:
                    row["RSD%"] = rsd_pct
                if "rse" in columns:
                    row["RSE%"] = rse_pct
                if "n_valid" in columns:
                    row["n"] = stats.n
                if "n_total" in columns:
                    row["N"] = n_total

            # SSB corrected (nested: ssb_results[ratio_name]["ssb_corrected_cycles"]).
            if sample.ssb_results and "ssb_corrected" in columns:
                ssb_cd = get_ssb_cycle_data(sample, ratio_name)
                ssb_values = _apply_selection(ssb_cd, selection)
                if ssb_values is not None and len(ssb_values) > 0:
                    row["SSB Corrected"] = float(np.mean(ssb_values))

            # Delta (nested: delta_results[ratio_name]["delta"]).
            if sample.delta_results and "delta" in columns:
                delta_data = sample.delta_results.get(ratio_name, {})
                delta_values = _apply_selection(
                    get_delta_cycle_data(sample, ratio_name), selection
                )
                if delta_values is not None and len(delta_values) > 0:
                    # Recomputed over the prepared selection rather than read
                    # from the stored full-run fields. With no window and no
                    # filter this reproduces the stored values exactly: the
                    # pipeline's own delta statistics are the mean and the
                    # ddof=1 standard error of these same per-cycle values.
                    delta, _sd, _se, delta_2se, _n = compute_delta_stats(delta_values)
                    row["Delta"] = delta
                    if "delta_unc" in columns:
                        row["Delta 2SE (‰)"] = delta_2se
                elif delta_data:
                    if "delta" in delta_data:
                        row["Delta"] = delta_data["delta"]
                    if "delta_unc" in columns and "delta_2se" in delta_data:
                        row["Delta 2SE (‰)"] = delta_data["delta_2se"]

            # K-factor (nested: ssb_results[ratio_name]["k_factor"]).
            if sample.ssb_results and "ssb_k" in columns:
                ssb_data = sample.ssb_results.get(ratio_name, {})
                if ssb_data and "k_factor" in ssb_data:
                    row["K-factor"] = ssb_data["k_factor"]

            # Uncertainty - rounded UP per GUM section 7.2.6 (ceiling, not nearest).
            if sample.uncertainty and ratio_name in sample.uncertainty:
                budget = sample.uncertainty[ratio_name]
                invalid_scope = is_invalid_budget_scope(budget)
                if invalid_scope:
                    if "budget_note" in columns:
                        row["Budget Note"] = budget_scope_note(budget)
                    if "budget_state" in columns:
                        row["State"] = budget_scope_label(budget)
                else:
                    if "u_combined" in columns:
                        row[_U_COMBINED_COLUMN] = report_uncertainty(
                            budget.u_combined_abs
                        )
                    if "u_expanded" in columns:
                        row[_U_EXPANDED_COLUMN] = report_uncertainty(budget.expanded_abs)
                    if "u_c_permil" in columns:
                        row["u_c (permil)"] = report_uncertainty(budget.u_combined_rel_permil)
                    if "u_exp_permil" in columns:
                        row["U (permil)"] = report_uncertainty(budget.expanded_rel_permil)
                    if "k_coverage" in columns:
                        row["k"] = budget.coverage_factor_k
                    if "dominant" in columns:
                        name = budget.dominant_contributor
                        row["Dominant"] = format_contributor_name(name)
                    if "budget_note" in columns:
                        row["Budget Note"] = _build_budget_note(budget)
                if "budget_state" in columns and not invalid_scope:
                    row["State"] = summarize_budget_state(budget)

            _add_calibrated_delta_columns(row, sample, ratio_name, selection, columns)

            if include_mc:
                _add_mc_columns(
                    row,
                    sample,
                    ratio_name,
                    uncertainty_config=uncertainty_config,
                )

            for key, value in list(row.items()):
                if isinstance(value, str):
                    row[key] = sanitize_spreadsheet_text(value)

            rows.append(row)

    return pd.DataFrame(rows).reindex(
        columns=_ordered_output_columns(columns, include_mc=include_mc)
    )


def _add_mc_columns(
    row: Dict[str, object],
    sample: Sample,
    ratio_name: str,
    *,
    uncertainty_config: Optional[object] = None,
) -> None:
    """Fill the Monte Carlo block for one sample/ratio row.

    Reads the durable record only; it never runs, rebuilds or mutates a
    result. Every row carries the MC columns so the table stays rectangular,
    with ``NaN`` where a sample/ratio has no completed run.
    """
    for column in _MC_COLUMNS:
        row[column] = np.nan

    record: Optional[MCResultRecord] = (
        getattr(sample, "mc_results", None) or {}
    ).get(ratio_name)
    if record is None:
        return

    row["MC Semantics"] = record.semantics_version
    row["MC Result Space"] = record.effective_result_space or record.result_space
    row["MC Result Freshness"] = mc_record_freshness_for_budget(
        record,
        (getattr(sample, "uncertainty", None) or {}).get(ratio_name),
        uncertainty_config,
        sample=sample,
    )
    row["MC Execution ID"] = record.execution_id
    row["MC Requested Draws"] = record.requested_draws
    row["MC Completed Draws"] = record.completed_draws
    if record.seed is not None:
        # Written as text, not as a number. A seed is an identity, not a
        # measured quantity: a numeric column would be float64 here (the
        # other rows hold NaN) and would silently corrupt any seed at or
        # above 2**53. Text is exact for every seed.
        row["MC Seed"] = str(record.seed)
    if record.mc_mean is not None:
        row["MC Mean"] = record.mc_mean
    if record.mc_std is not None:
        row["MC SD (ddof=1)"] = record.mc_std
    if record.mc_lower is not None:
        row["MC Interval Lower"] = record.mc_lower
    if record.mc_upper is not None:
        row["MC Interval Upper"] = record.mc_upper
    row["MC Coverage Probability"] = record.coverage_probability
    row["MC Interval Method"] = (
        f"{record.interval_convention}/{record.percentile_method}"
    )
    if record.gum_u_c is not None:
        row["MC u_c (GUM)"] = record.gum_u_c
    if record.gum_u_expanded is not None:
        row["MC U (GUM expanded)"] = record.gum_u_expanded
    if record.gum_coverage_factor_k is not None:
        row["MC Coverage Factor k"] = record.gum_coverage_factor_k
    row["MC Canonical Record (JSON)"] = json.dumps(
        record.to_dict(), allow_nan=False, separators=(",", ":")
    )


def _add_calibrated_delta_columns(
    row: Dict[str, object],
    sample: Sample,
    ratio_name: str,
    selection: np.ndarray,
    columns: Set[str],
) -> None:
    """Report a Pb-standard-calibrated delta and its deliberately absent uncertainty.

    The value is the mean of the calibration's own per-cycle delta over the
    row's selection. ``Delta 2SE (‰)`` is filled only when the calibration was
    asked for the SE precision statistic; it is precision, not uncertainty. The
    combined uncertainty is written as ``Not calculated``, never as a number.
    """
    from domain.pb_calibration_records import calibrated_delta_record

    record = calibrated_delta_record(sample, ratio_name)
    if record is None or "delta" not in columns:
        return
    # A runtime-export copy can also carry a legacy-shaped delta payload. Clear
    # that column before applying the calibrated reporting preference so
    # ``none`` and ``sd`` can never leak a legacy 2SE value.
    if "delta_unc" in columns:
        row["Delta 2SE (‰)"] = np.nan
    note = (
        f"Calibrated delta combined uncertainty: Not calculated ({record.uncertainty_reason_code})."
    )
    if record.status == "applied":
        values = _apply_selection(sample.pb_calibrated_delta_cycles.get(ratio_name), selection)
        if values is not None and len(values) > 0:
            delta, _sd, se, _delta_2se, n = compute_delta_stats(values)
            row["Delta"] = delta
            if "delta_unc" in columns and record.precision_statistic == "se" and n >= 2:
                row["Delta 2SE (‰)"] = 2.0 * se
    else:
        note = f"Calibrated delta unavailable ({record.reason_code}). {note}"
    if "budget_note" in columns:
        existing = row.get("Budget Note")
        existing_text = existing if isinstance(existing, str) else ""
        row["Budget Note"] = f"{existing_text} {note}".strip()


def _apply_selection(
    cycle_data: Optional[CycleData], selection: np.ndarray
) -> Optional[np.ndarray]:
    """Restrict a derived per-cycle series to the row's prepared selection.

    Returns ``None`` when the series is absent or is not aligned to the
    reported ratio layer, so a length mismatch degrades to the previous
    behaviour instead of silently exporting a mismatched subset.
    """
    if cycle_data is None:
        return None
    values = np.asarray(cycle_data.values, dtype=np.float64)
    if len(values) != len(selection):
        return None
    keep = selection & np.asarray(cycle_data.mask, dtype=bool)
    return values[keep]


def _ordered_output_columns(
    columns: Set[str], *, include_mc: bool = False
) -> List[str]:
    """Return CSV columns in a stable, human-readable order."""
    ordered = list(_BASE_COLUMNS) + ["Observation ID"]
    optional = [
        ("mean", "Mean"),
        ("2sd", "2SD"),
        ("2se", "2SE"),
        ("rsd", "RSD%"),
        ("rse", "RSE%"),
        ("n_valid", "n"),
        ("n_total", "N"),
        ("ssb_corrected", "SSB Corrected"),
        ("delta", "Delta"),
        ("delta_unc", "Delta 2SE (‰)"),
        ("ssb_k", "K-factor"),
        ("u_combined", _U_COMBINED_COLUMN),
        ("u_expanded", _U_EXPANDED_COLUMN),
        ("u_c_permil", "u_c (permil)"),
        ("u_exp_permil", "U (permil)"),
        ("k_coverage", "k"),
        ("dominant", "Dominant"),
        ("budget_note", "Budget Note"),
        ("budget_state", "State"),
    ]
    for key, label in optional:
        if key in columns:
            ordered.append(label)
    if include_mc:
        ordered.extend(_MC_COLUMNS)
    return ordered


def _resolve_ratio_names(
    samples: List[Sample],
    ratio_name: Optional[str],
    ratio_names: Optional[List[str]],
) -> List[str]:
    """Resolve the final ratio list for export."""
    if ratio_names is not None:
        return list(dict.fromkeys(ratio_names))
    if ratio_name:
        return [ratio_name]
    return _collect_ratio_names(samples)


def _collect_ratio_names(samples: Iterable[Sample]) -> List[str]:
    """Collect all ratio names referenced across samples."""
    ratio_names: Set[str] = set()
    for sample in samples:
        ratio_names.update(sample.ratios.keys())
        ratio_names.update(sample.blank_corrected_ratios.keys())
        ratio_names.update(sample.corrected_ratios.keys())
        ratio_names.update(sample.iif_corrected_ratios.keys())
        ratio_names.update(sample.drift_corrected_ratios.keys())
    return sorted(ratio_names)


def _get_export_ratio_data(sample: Sample, ratio_name: str) -> Tuple[Optional[CycleData], str]:
    """Return the exported ratio layer and its user-facing label."""
    selected = select_summary_export_layer(sample, ratio_name)
    if selected is None:
        return None, "-"
    return selected.data, selected.label

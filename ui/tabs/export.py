"""
Export tab for TraceISO.

Provides full-report download (Excel / JSON / CSV) with configurable
content options, a preview panel, and a quick CSV summary export.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import replace
from collections import OrderedDict
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import streamlit as st

from config.constants import APP_VERSION
from domain.corrections.blank import _compute_ratios
from domain.models import CycleData, ProcessingResult, Sample
from domain.ratio_utils import normalize_ratio_token
from domain.uncertainty.scope import CROSS_RATIO_INDEPENDENCE_NOTE
from ui.state import get_state
from ui.utils import format_isotope_label
from ui.utils import current_pb_calibration_freshness, get_cycle_ranges, settings_changed_since_processing
from ui.components.workspace_ui import render_next_step, render_subview_nav
from ui.navigation import request_main_section, resolve_subview_selection


_log = logging.getLogger(__name__)
_RUNTIME_EXPORT_CACHE_KEY = "_runtime_export_result_cache_sr_v2"
_EXPORT_PAYLOAD_CACHE_KEY = "_export_payload_cache_evidence_v2"
_MAX_EXPORT_CACHE_SIZE = 8
_EXPORT_OPTION_MEMORY_KEY = "_export_option_memory"
_PREPARED_EXPORT_KEY = "_prepared_export_evidence_v2"


def _remembered_export_option(key: str, default):
    """Retain conditional export controls when Streamlit evicts their widgets."""
    session = getattr(st, "session_state", {})
    memory = session.get(_EXPORT_OPTION_MEMORY_KEY, {})
    return memory.get(key, default) if isinstance(memory, dict) else default


def _remember_export_option(key: str, value) -> None:
    session = getattr(st, "session_state", None)
    if session is None:
        return
    memory = session.setdefault(_EXPORT_OPTION_MEMORY_KEY, {})
    if isinstance(memory, dict):
        memory[key] = value


def _has_selected_ratio_selection(state) -> bool:
    """Return whether the user has explicitly initialized ratio selection."""
    return bool(
        getattr(
            state,
            "has_selected_ratio_selection",
            st.session_state.get("selected_ratios") is not None,
        )
    )


def render_export_tab() -> None:
    """Render the Export tab."""
    state = get_state()

    if not state.has_result:
        if render_next_step(
            "Export is not available yet",
            "Configure the session and execute data reduction before exporting.",
            action_label="Open Session Configuration",
            action_key="export_open_session_configuration",
        ):
            request_main_section("Session Configuration")
            st.rerun()
        return

    from ui.components.sr_calibration_panel import render_sr_calibration_notice
    render_sr_calibration_notice(state.result.samples)

    if settings_changed_since_processing(state):
        st.warning(
            "Cycle selections, sample exclusions, types, or processing settings changed "
            "after the last processing run. "
            "Click **Execute Data Reduction** in Session Configuration before exporting."
        )
        return

    st.subheader("Export")

    export_format = _render_format_selector()
    st.divider()

    # Keyed so the theme can stack options above the preview when the
    # workspace is narrow (U34).
    with st.container(key="export_options_layout"):
        col_options, col_preview = st.columns([1, 1.2])
    with col_options:
        options = _render_content_options(export_format)
        if export_format == "csv":
            options.update(_render_csv_scope_options(state.result))
    with col_preview:
        _render_export_preview(export_format, options, state.result)

    st.divider()

    _render_download_button(export_format, options)

def _render_format_selector() -> str:
    """Render the Excel, JSON, and CSV format selector."""
    options = ["Excel (.xlsx)", "JSON (.json)", "CSV (.csv)"]

    # Defensive migration for returning-session stale values (e.g. raw ID values like 'excel')
    if "export_format_radio" in st.session_state:
        val = st.session_state["export_format_radio"]
        id_to_label = {
            "excel": "Excel (.xlsx)",
            "json": "JSON (.json)",
            "csv": "CSV (.csv)",
        }
        if val in id_to_label:
            st.session_state["export_format_radio"] = id_to_label[val]
        elif val not in options:
            st.session_state["export_format_radio"] = "Excel (.xlsx)"

    fmt = render_subview_nav(
        "Export Format",
        options=options,
        key="export_format_radio",
        default="Excel (.xlsx)",
        current=resolve_subview_selection(
            key="export_format_radio", options=options, default="Excel (.xlsx)",
        ),
    )

    mapping = {
        "Excel (.xlsx)": "excel",
        "JSON (.json)": "json",
        "CSV (.csv)": "csv",
    }
    # `fmt` is always one of `options` — the migration block above coerces any
    # stale session-state value back into the label set, and render_subview_nav
    # falls back to `default` when a segmented control deselects to None. The
    # fallback here is therefore unreachable in practice; it exists so a future
    # label rename degrades to the default format instead of a KeyError.
    return mapping.get(fmt, "excel")


def _render_content_options(export_format: str) -> dict:
    """Render content toggle checkboxes."""
    state = get_state()
    ssb_report = export_format == "excel" and _is_ssb_excel_report(
        getattr(state, "element_config", None),
        getattr(state, "processing_config", None),
        getattr(state, "uncertainty_config", None),
    )
    sr_report = export_format == "excel" and _is_sr_excel_report(
        getattr(state, "element_config", None), getattr(state, "processing_config", None),
        getattr(state, "uncertainty_config", None),
    )
    pb_report = export_format == "excel" and _is_pb_tl_excel_report(
        getattr(state, "element_config", None), getattr(state, "processing_config", None),
        getattr(state, "uncertainty_config", None),
    )
    if ssb_report or sr_report or pb_report:
        from file_io.excel_writer import EXCEL_LAYOUT_SSB_REPORT, EXCEL_LAYOUT_SR_REPORT, EXCEL_LAYOUT_PB_REPORT

        with st.expander("Content Options", expanded=True):
            st.caption("Always included: Summary, Results. Saved MC evidence is included when present.")
            detail = st.checkbox(
                "Include uncertainty detail",
                value=True,
                key="sticky_export_pb_detail" if pb_report else "sticky_export_sr_detail" if sr_report else "sticky_export_ssb_detail",
            )
            cycles = st.checkbox(
                "Include cycle data" if pb_report else "Cycle_Data workbook view",
                value=False,
                key="sticky_export_pb_cycles" if pb_report else "sticky_export_sr_cycles" if sr_report else "sticky_export_ssb_cycles",
            )
        return {
            "include_raw": False,
            "include_corrected": True,
            "include_uncertainty": True,
            "include_parameters": True,
            "include_cycle_data": cycles,
            "raw_layout": "stacked",
            "include_cover_summary": True,
            "include_results_final": True,
            "include_uncertainty_budgets": False,
            "include_uncertainty_budget_wide": detail,
            "include_raw_sheet": False,
            "excel_layout": EXCEL_LAYOUT_PB_REPORT if pb_report else EXCEL_LAYOUT_SR_REPORT if sr_report else EXCEL_LAYOUT_SSB_REPORT,
            "contributor_profile_scope": "used",
            "csv_variant": "complete",
        }
    with st.expander("Content Options", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            include_corrected = st.checkbox(
                "Corrected ratios",
                value=True,
                key="export_include_corrected",
                disabled=(export_format == "csv"),
                help="Excel/JSON only. CSV exports the selected ratios in long format with the source layer labeled.",
            )
            include_uncertainty = st.checkbox(
                "Uncertainty budgets",
                value=True,
                key="export_include_uncertainty",
                help=CROSS_RATIO_INDEPENDENCE_NOTE,
            )
            include_raw = st.checkbox(
                (
                    "Cycle_Data workbook view"
                    if export_format == "excel"
                    else "Raw ratio summaries"
                ),
                value=False,
                key="export_include_ratio_statistics",
                disabled=(export_format == "csv"),
                help=(
                    "Excel: enables the Cycle_Data route, containing per-cycle arrays "
                    "and a statistics block; use the dependent sheet option below to "
                    "include or omit it. JSON: includes raw ratio summaries."
                ),
            )

        with col2:
            include_parameters = st.checkbox(
                "Processing parameters",
                value=True,
                key="export_include_parameters",
                disabled=(export_format != "excel"),
                help="Excel only.",
            )
            include_cycle_data = st.checkbox(
                "Per-sample audit cycle sheets (large)",
                value=False,
                key="export_include_cycles",
                disabled=(export_format == "csv"),
                help=(
                    "Include one audit-oriented per-sample sheet with measured values "
                    "and accepted/excluded status for every series. Blank observations "
                    "are not included; intensities are in volts (V). CSV does not "
                    "include cycle data."
                ),
            )

    from file_io.csv_writer import CSV_VARIANT_COMPACT, CSV_VARIANT_COMPLETE
    from file_io.excel_writer import (
        EXCEL_LAYOUT_ANALYST_REPORT,
        EXCEL_LAYOUT_DIAGNOSTIC,
        PROFILE_SCOPE_LIBRARY,
        PROFILE_SCOPE_USED,
    )

    include_cover_summary = True
    include_results_final = True
    include_raw_sheet = True
    include_uncertainty_budgets = True
    include_uncertainty_budget_wide = True
    include_profile_library = False
    raw_layout_choice = "One stacked sheet"
    layout_choice = _LAYOUT_LABELS[0]
    csv_variant_choice = _CSV_VARIANT_LABELS[0]

    if export_format == "csv":
        # U64, designed with U73: the CSV view holds every CSV choice. U73 adds
        # the "All selected ratios / One ratio" scope beside this selector and
        # retires the separate Quick CSV panel.
        csv_variant_choice = st.radio(
            "CSV content",
            _CSV_VARIANT_LABELS,
            horizontal=True,
            # sticky_: kept across section changes by app_shell's keep-alive sweep.
            key="sticky_export_csv_variant",
            help=(
                "Complete audit CSV: every row repeats the canonical provenance, "
                "uncertainty-scope and Monte Carlo record JSON. Compact CSV: the same "
                "rows and scientific columns, with those repeated payloads replaced by "
                "short identity columns."
            ),
        )

    if export_format == "excel":
        layout_choice = st.radio(
            "Workbook layout",
            _LAYOUT_LABELS,
            horizontal=True,
            key="sticky_export_excel_layout",
            help=(
                "Diagnostic keeps the existing workbook, led by the Results sheet. "
                "Analyst report adds a Reported Results sheet first: one row per "
                "observation, ratio and output mode with its U and coverage factor. "
                "Results stays available as an optional diagnostic sheet."
            ),
        )
        with st.expander("Advanced Excel sheet options", expanded=False):
            # One vertical checklist: analyst description first, the unchanged
            # worksheet name in parentheses. Widget keys are export contracts.
            st.caption("Worksheet names in parentheses are unchanged in the workbook.")
            include_cover_summary = st.checkbox(
                "Summary cover sheet (Summary)",
                value=_remembered_export_option("export_sheet_cover", True),
                key="export_sheet_cover",
            )
            include_results_final = st.checkbox(
                "Results table (Results)",
                value=_remembered_export_option("export_sheet_results", True),
                key="export_sheet_results",
                help=(
                    "Statistics for every processing stage, one row per sample. In the "
                    "analyst report layout this is an optional diagnostic sheet."
                ),
            )
            # U54: two views of the same budgets, named for their shape.
            include_uncertainty_budgets = st.checkbox(
                "Contributor-level budgets, one row per contributor (Uncertainty_Budgets)",
                value=_remembered_export_option("export_sheet_uncertainty_budgets", True),
                key="export_sheet_uncertainty_budgets",
                disabled=not include_uncertainty,
                help=(
                    "Every contributor for each sample and ratio with its state, "
                    "reference and applicability, plus the Sample Applicability and "
                    "Contributor Profiles sheets (requires Uncertainty budgets)."
                ),
            )
            include_uncertainty_budget_wide = st.checkbox(
                "Wide analyst budget view, optional (Budget_Detail)",
                value=_remembered_export_option("export_sheet_uncertainty_budget_wide", True),
                key="export_sheet_uncertainty_budget_wide",
                disabled=not include_uncertainty,
                help=(
                    "One row per sample with active contributors as columns; SSB/delta "
                    "budgets only. A compact view of the same budgets that does not "
                    "replace the contributor-level sheet."
                ),
            )
            include_profile_library = st.checkbox(
                "Complete contributor profile library, for audit (Contributor Profiles)",
                value=False,
                key="sticky_export_sheet_profile_library",
                disabled=not (include_uncertainty and include_uncertainty_budgets),
                help=(
                    "Off: Contributor Profiles lists only the profiles the exported "
                    "observations use. On: every profile in the library."
                ),
            )
            include_raw_sheet = st.checkbox(
                "Per-cycle data sheet (Cycle_Data)",
                value=_remembered_export_option("export_sheet_raw", True),
                key="export_sheet_raw",
                disabled=not include_raw,
                help="Requires Cycle_Data workbook view; the emitted worksheet name remains Cycle_Data.",
            )

            if include_raw and include_raw_sheet:
                st.divider()
                raw_layout_choice = st.radio(
                    "Cycle Data layout",
                    ["One stacked sheet", "One sheet per sample"],
                    index=(
                        1
                        if _remembered_export_option("export_raw_layout", "One stacked sheet")
                        == "One sheet per sample"
                        else 0
                    ),
                    horizontal=True,
                    key="export_raw_layout",
                    help='"Stacked" puts all samples in a single Cycle_Data sheet; '
                    '"Per sample" creates a separate sheet for each sample.',
                )

        for key, value in (
            ("export_sheet_cover", include_cover_summary),
            ("export_sheet_results", include_results_final),
            ("export_sheet_uncertainty_budgets", include_uncertainty_budgets),
            ("export_sheet_uncertainty_budget_wide", include_uncertainty_budget_wide),
            ("export_sheet_raw", include_raw_sheet),
            ("export_raw_layout", raw_layout_choice),
        ):
            _remember_export_option(key, value)

    return {
        "include_raw": include_raw,
        "include_corrected": include_corrected,
        "include_uncertainty": include_uncertainty,
        "include_parameters": include_parameters,
        "include_cycle_data": include_cycle_data,
        "raw_layout": "stacked"
        if raw_layout_choice == "One stacked sheet"
        else "per_sample",
        "include_cover_summary": include_cover_summary,
        "include_results_final": include_results_final,
        "include_uncertainty_budgets": include_uncertainty_budgets,
        "include_uncertainty_budget_wide": include_uncertainty_budget_wide,
        "include_raw_sheet": include_raw_sheet,
        "excel_layout": (
            EXCEL_LAYOUT_ANALYST_REPORT
            if layout_choice == _LAYOUT_LABELS[1]
            else EXCEL_LAYOUT_DIAGNOSTIC
        ),
        "contributor_profile_scope": (
            PROFILE_SCOPE_LIBRARY if include_profile_library else PROFILE_SCOPE_USED
        ),
        "csv_variant": (
            CSV_VARIANT_COMPACT
            if csv_variant_choice == _CSV_VARIANT_LABELS[1]
            else CSV_VARIANT_COMPLETE
        ),
    }


_LAYOUT_LABELS = ["Diagnostic (Results first)", "Analyst report (Reported Results first)"]
_CSV_VARIANT_LABELS = ["Complete audit CSV", "Compact CSV"]
_CSV_SCOPE_LABELS = ["All selected ratios", "One ratio"]


def _render_csv_scope_options(result: ProcessingResult) -> dict:
    """Render the CSV row scope and conditional stable ratio selection."""
    scope_choice = st.radio(
        "CSV scope",
        _CSV_SCOPE_LABELS,
        horizontal=True,
        key="sticky_export_csv_scope",
    )
    selected_ratio = None
    if scope_choice == _CSV_SCOPE_LABELS[1]:
        ratio_list = sorted(_collect_export_ratio_names(result))
        if st.session_state.get("sticky_export_csv_ratio") not in ratio_list:
            st.session_state.pop("sticky_export_csv_ratio", None)
        selected_ratio = st.selectbox(
            "Ratio for CSV",
            options=ratio_list,
            index=0 if ratio_list else None,
            format_func=format_isotope_label,
            key="sticky_export_csv_ratio",
            disabled=not ratio_list,
        )
    return {
        "csv_scope": "one_ratio" if scope_choice == _CSV_SCOPE_LABELS[1] else "all_selected",
        "csv_ratio": selected_ratio,
    }


def _render_export_preview(
    export_format: str,
    options: dict,
    result=None,
) -> None:
    """Show a summary of what will be exported."""
    state = get_state()
    result = result or state.result
    if export_format == "excel" and options.get("excel_layout") == "traceiso.excel_layout.pb_report.v1":
        try:
            _require_current_pb_calibration(state)
            result = _build_runtime_export_result(state.result)
            from file_io.pb_excel_writer import validate_pb_report
            validate_pb_report(result, state.processing_config, state.uncertainty_config,
                               current_pb_calibration_freshness(state))
        except (RuntimeError, ValueError) as exc:
            st.warning(str(exc))
            return
    if export_format == "excel" and options.get("excel_layout") == "traceiso.excel_layout.sr_report.v1":
        try:
            _require_current_sr_calibration(result)
            result = _build_runtime_export_result(result)
        except RuntimeError as exc:
            st.warning(str(exc))
            return
    counts = result.count_by_type
    selected_ratios = _get_selected_export_ratios()
    ratio_names = _resolve_export_ratio_names(result, selected_ratios)
    if _has_selected_ratio_selection(state):
        ratio_count_label = f"{len(ratio_names)} selected"
    else:
        ratio_count_label = f"{len(ratio_names)} available"

    count_str = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))

    st.caption("**Preview**")

    if export_format == "excel":
        from file_io.excel_writer import (
            EXCEL_LAYOUT_ANALYST_REPORT,
            EXCEL_LAYOUT_DIAGNOSTIC,
            EXCEL_LAYOUT_SSB_REPORT,
            PROFILE_SCOPE_LIBRARY,
            excel_sheet_manifest,
        )

        layout = options.get("excel_layout", EXCEL_LAYOUT_DIAGNOSTIC)
        sheets = excel_sheet_manifest(
            result,
            include_raw=options.get("include_raw", False),
            include_uncertainty=options.get("include_uncertainty", True),
            include_cycle_data=options.get("include_cycle_data", False),
            raw_layout=options.get("raw_layout", "stacked"),
            include_cover_summary=options.get("include_cover_summary", True),
            include_results_final=options.get("include_results_final", True),
            include_raw_sheet=options.get("include_raw_sheet", True),
            include_uncertainty_budgets=options.get("include_uncertainty_budgets", True),
            include_uncertainty_budget_wide=options.get("include_uncertainty_budget_wide", True),
            provenance_present=True,
            layout=layout,
            **({"processing_config": state.processing_config,
                "uncertainty_config": state.uncertainty_config,
                "calibration_freshness": current_pb_calibration_freshness(state)}
               if layout == "traceiso.excel_layout.pb_report.v1" else {}),
        )
        sheets = [name for name in sheets if name not in {
            "Provenance", "Uncertainty Scope", "Long Payloads (only when required)"
        }]
        cycle_sheets = [
            name
            for name in sheets
            if name == "Cycle_Data"
            or name.startswith("Cycle_")
            or name.startswith("Cycles_")
        ]
        lines = [
            "- **Format:** Excel (.xlsx)",
            "- **Layout:** "
            + (
                "Scientific report"
                if layout in {EXCEL_LAYOUT_SSB_REPORT, "traceiso.excel_layout.sr_report.v1", "traceiso.excel_layout.pb_report.v1"}
                else (
                    "Analyst report — Reported Results first (standards, samples and QC; blanks excluded)"
                    if layout == EXCEL_LAYOUT_ANALYST_REPORT
                    else "Diagnostic — Results first"
                )
            ),
            f"- **Worksheets:** {', '.join(sheets)}",
            f"- **Samples:** {len(result.samples)}",
            f"- **Ratios:** {ratio_count_label}",
            "- **Cycle-resolved data:** "
            + ("Included via " + ", ".join(cycle_sheets) if cycle_sheets else "Not included"),
        ]
        if any(name.startswith("Cycles_") for name in sheets):
            lines.append(
                "- **Per-sample audit sheets:** non-blank observations only; intensities in volts (V)"
            )
        if "Contributor Profiles" in sheets:
            lines.append(
                "- **Contributor Profiles:** "
                + (
                    "complete library"
                    if options.get("contributor_profile_scope") == PROFILE_SCOPE_LIBRARY
                    else "profiles used by the exported observations"
                )
            )
        st.markdown("\n".join(lines))
        if layout == "traceiso.excel_layout.pb_report.v1":
            from file_io.pb_excel_writer import _contexts
            reasons = sorted({reason for layer, budget, reason in _contexts(result, state.processing_config).values() if reason})
            for reason in reasons:
                st.caption(reason)

    elif export_format == "json":
        sections = [
            "export_info",
            "element",
            "sample_counts",
            "warnings",
            "quality_metrics",
            "samples",
        ]
        ratio_layers = []
        if options["include_raw"]:
            ratio_layers.append("raw")
        if options["include_corrected"]:
            ratio_layers.append("blank-corrected, corrected, iif, and drift-corrected")
        ratio_text = ", ".join(ratio_layers) if ratio_layers else "none"
        st.markdown(
            f"- **Format:** JSON (.json)\n"
            f"- **Ratio summaries:** {ratio_text}\n"
            f"- **Per-cycle data:** {'included' if options['include_cycle_data'] else 'not included'}\n"
            f"- **Ratios:** {ratio_count_label}\n"
            f"- **Samples:** {len(result.samples)} ({count_str})"
        )
        with st.expander("JSON schema details", expanded=False):
            st.caption(f"Top-level keys: {', '.join(sections)}")

    elif export_format == "csv":
        compact = options.get("csv_variant") == "compact"
        content = (
            "compact — canonical JSON payloads replaced by identity columns"
            if compact
            else "complete audit — canonical provenance, scope and MC record JSON on every row"
        )
        scope = (
            f"one ratio ({format_isotope_label(options['csv_ratio'])})"
            if options.get("csv_scope") == "one_ratio" and options.get("csv_ratio")
            else "all selected ratios"
        )
        st.markdown(
            f"- **Format:** CSV (.csv)\n"
            f"- **Content:** {content}\n"
            f"- **Scope:** {scope}\n"
            f"- **Rows:** one per sample and ratio\n"
            f"- **Layer:** best available per ratio "
            f"(SSB > drift > IIF > interference/blank-corrected > raw)\n"
            f"- **Ratios:** {ratio_count_label}\n"
            f"- **Samples:** {len(result.samples)} ({count_str})"
        )

def _has_selected_excel_content(options: dict) -> bool:
    """Return whether the current Excel options would produce a real sheet."""
    return any(
        (
            options.get("excel_layout") == "traceiso.excel_layout.ssb_report.v1",
            options.get("excel_layout") == "traceiso.excel_layout.sr_report.v1",
            options.get("excel_layout") == "traceiso.excel_layout.pb_report.v1",
            options.get("excel_layout") == "traceiso.excel_layout.analyst_report.v1",
            options.get("include_cover_summary", True),
            options.get("include_results_final", True),
            options.get("include_uncertainty", True)
            and options.get("include_uncertainty_budgets", True),
            options.get("include_uncertainty", True)
            and options.get("include_uncertainty_budget_wide", True),
            options.get("include_raw", True) and options.get("include_raw_sheet", True),
            options.get("include_cycle_data", False),
        )
    )


def _render_download_button(
    export_format: str,
    options: dict,
    result: Optional[ProcessingResult] = None,
) -> None:
    """Prepare on request, then offer bytes bound to the current dependencies."""
    state = get_state()
    filename = _generate_filename(export_format)
    try:
        _require_current_pb_calibration(state)
    except RuntimeError as exc:
        st.session_state.pop(_PREPARED_EXPORT_KEY, None)
        st.warning(str(exc))
        return
    if export_format == "excel" and not _has_selected_excel_content(options):
        st.warning("Select at least one Excel sheet to export.")
        return

    from ui.runtime_budget_cache import _freeze_for_cache, _processing_config_token
    request_token = (
        _runtime_export_input_token(state, state.result),
        export_format,
        _freeze_for_cache(options),
        _processing_config_token(state.processing_config),
    )
    prepared = st.session_state.get(_PREPARED_EXPORT_KEY)
    if not isinstance(prepared, dict) or prepared.get("token") != request_token:
        st.session_state.pop(_PREPARED_EXPORT_KEY, None)
        prepared = None

    _col1, col2, _col3 = st.columns([1, 2, 1])
    with col2:
        if export_format == "csv":
            one_ratio = options.get("csv_scope") == "one_ratio"
            if one_ratio and not options.get("csv_ratio"):
                st.warning("No ratio is available for the one-ratio CSV.")
                return
            compact = options.get("csv_variant") == "compact"
            selected_ratio = options.get("csv_ratio")
            suffix = "_compact" if compact else ""
            if one_ratio and selected_ratio:
                suffix = f"_{selected_ratio.replace('/', '_')}{suffix}"
            filename = _generate_filename("csv", suffix=suffix)

        if st.button(
            "Prepare report",
            type="primary" if prepared is None else "secondary",
            width="stretch",
            key="prepare_export_button",
        ):
            if settings_changed_since_processing(state):
                st.error("The processing state changed. Reprocess before preparing this report.")
                st.session_state.pop(_PREPARED_EXPORT_KEY, None)
                return
            try:
                with st.spinner("Preparing report..."):
                    runtime_result = _build_runtime_export_result(state.result)
                    cache_format = (
                        "quick_csv"
                        if export_format == "csv" and options.get("csv_scope") == "one_ratio"
                        else export_format
                    )
                    data = _get_cached_export_payload(
                        runtime_result,
                        export_format=cache_format,
                        options=options,
                        processing_config=state.processing_config,
                    )
                prepared = {"token": request_token, "data": data, "filename": filename}
                st.session_state[_PREPARED_EXPORT_KEY] = prepared
            except Exception as exc:
                _log.exception("Export preparation failed")
                st.error(f"Report preparation failed: {exc}. Adjust the options and retry.")
                return

        if prepared is not None:
            mime = {
                "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "json": "application/json",
                "csv": "text/csv",
            }[export_format]
            label = {"excel": "Download Excel Report", "json": "Download JSON Report", "csv": "Download CSV"}[export_format]
            st.success("Report ready for download.")
            st.download_button(
                label,
                data=prepared["data"],
                file_name=prepared["filename"],
                mime=mime,
                type="primary",
                width="stretch",
                key=f"download_{export_format}",
                on_click="ignore",
            )

def _render_quick_export(
    options: dict,
    result: Optional[ProcessingResult] = None,
) -> None:
    """Render quick per-ratio CSV summary download."""
    state = get_state()
    result = result or _build_runtime_export_result(state.result)

    st.caption("**Quick CSV export**")

    ratio_list = sorted(_collect_export_ratio_names(result))
    if not ratio_list:
        st.caption("No ratio data available.")
        return

    col1, col2 = st.columns([2, 2])

    with col1:
        if st.session_state.get("quick_export_ratio") not in ratio_list:
            st.session_state["quick_export_ratio"] = (
                ratio_list[0] if ratio_list else None
            )
        selected_ratio = st.selectbox(
            "Ratio",
            options=ratio_list,
            format_func=format_isotope_label,
            key="quick_export_ratio",
        )

    with col2:
        csv_data = _get_cached_export_payload(
            result,
            export_format="quick_csv",
            options={
                "selected_ratio": selected_ratio,
                "include_uncertainty": options["include_uncertainty"],
            },
            processing_config=state.processing_config,
        )
        safe_name = selected_ratio.replace("/", "_")
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name=f"{safe_name}_summary.csv",
            mime="text/csv",
            key="quick_export_download",
        )


def _generate_excel(result, options: dict, processing_config) -> BytesIO:
    """Generate Excel BytesIO."""
    from file_io.excel_writer import export_to_excel

    state = get_state()
    _require_current_pb_calibration(state)
    if _is_sr_excel_report(getattr(state, "element_config", None), processing_config,
                          getattr(state, "uncertainty_config", None)):
        from file_io.excel_writer import EXCEL_LAYOUT_SR_REPORT
        options = {**options, "excel_layout": EXCEL_LAYOUT_SR_REPORT}
    if _is_ssb_excel_report(getattr(state, "element_config", None), processing_config,
                           getattr(state, "uncertainty_config", None)):
        from file_io.excel_writer import EXCEL_LAYOUT_SSB_REPORT
        options = {**options, "excel_layout": EXCEL_LAYOUT_SSB_REPORT}
    if _is_pb_tl_excel_report(getattr(state, "element_config", None), processing_config,
                             getattr(state, "uncertainty_config", None)):
        from file_io.excel_writer import EXCEL_LAYOUT_PB_REPORT
        options = {**options, "excel_layout": EXCEL_LAYOUT_PB_REPORT}
    kwargs = {
        "include_raw": options["include_raw"],
        "include_corrected": options["include_corrected"],
        "include_uncertainty": options["include_uncertainty"],
        "include_cycle_data": options["include_cycle_data"],
        "raw_layout": options.get("raw_layout", "stacked"),
        "loaded_filename": state.loaded_file,
        "include_cover_summary": options.get("include_cover_summary", True),
        "include_results_final": options.get("include_results_final", True),
        "include_uncertainty_budgets": options.get("include_uncertainty_budgets", True),
        "include_uncertainty_budget_wide": options.get(
            "include_uncertainty_budget_wide", True
        ),
        "include_raw_sheet": options.get("include_raw_sheet", True),
        "corrected_label": "Corrected",
        "contributor_profiles": getattr(state, "uncertainty_profiles", None),
        "provenance": _build_provenance(result),
        "include_metadata_sheets": False,
    }
    if "excel_layout" in options:
        kwargs["layout"] = options["excel_layout"]
    if options.get("excel_layout") == "traceiso.excel_layout.pb_report.v1":
        from file_io.pb_excel_writer import PbReportOptions
        kwargs["pb_report_options"] = PbReportOptions(
            options.get("include_uncertainty_budget_wide", True), options.get("include_cycle_data", False))
        kwargs["element_config"] = getattr(state, "element_config", None)
        kwargs["processing_config"] = processing_config
        kwargs["calibration_freshness"] = current_pb_calibration_freshness(state)
        if options.get("include_cycle_data", False):
            kwargs["cycle_result"] = _build_pb_cycle_export_result(state.result)
    if options.get("excel_layout") == "traceiso.excel_layout.sr_report.v1":
        from file_io.sr_excel_writer import SrReportOptions

        _require_current_sr_calibration(getattr(state, "result", result))
        kwargs["sr_report_options"] = SrReportOptions(
            include_uncertainty_detail=options.get("include_uncertainty_budget_wide", True),
            include_cycle_data=options.get("include_cycle_data", False),
        )
        kwargs["element_config"] = getattr(state, "element_config", None)
        kwargs["processing_config"] = processing_config
        if options.get("include_cycle_data", False):
            kwargs["cycle_result"] = _build_sr_cycle_export_result(state.result)
    if options.get("excel_layout") == "traceiso.excel_layout.ssb_report.v1":
        from file_io.ssb_excel_writer import SSBReportOptions

        kwargs["ssb_report_options"] = SSBReportOptions(
            include_uncertainty_detail=options.get(
                "include_uncertainty_budget_wide", True
            ),
            include_cycle_data=options.get("include_cycle_data", False),
        )
        # Preserve an unfiltered copied measurement view for the rectangular
        # Cycle_Data sheet. Runtime report-ratio filtering must not discard raw
        # channels or observations from this archive view.
        if options.get("include_cycle_data", False):
            cycle_result = _build_runtime_export_result_uncached(
                state.result, all_ratios=True,
            )
            kwargs["cycle_result"] = cycle_result
    if "contributor_profile_scope" in options:
        kwargs["contributor_profile_scope"] = options["contributor_profile_scope"]
    if options["include_parameters"]:
        kwargs["processing_config"] = processing_config

    # Pass uncertainty config for provenance metadata
    uncertainty_config = getattr(state, "uncertainty_config", None)
    if uncertainty_config is not None:
        kwargs["uncertainty_config"] = uncertainty_config

    return export_to_excel(result, **kwargs)


def _bounded_session_cache(key: str) -> OrderedDict:
    """Return a bounded ordered session cache."""
    cache = st.session_state.get(key)
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict(cache or {})
        st.session_state[key] = cache
    return cache


def _cache_store(cache: OrderedDict, key, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _MAX_EXPORT_CACHE_SIZE:
        cache.popitem(last=False)


def _get_cached_export_payload(
    result: ProcessingResult,
    *,
    export_format: str,
    options: dict,
    processing_config,
):
    """Return serialized export data without rebuilding unchanged payloads."""
    from ui.runtime_budget_cache import _freeze_for_cache, _processing_config_token

    _require_current_pb_calibration(get_state())
    cache = _bounded_session_cache(_EXPORT_PAYLOAD_CACHE_KEY)
    cache_key = (
        _runtime_export_input_token(get_state(), result),
        export_format,
        _freeze_for_cache(options),
        _processing_config_token(processing_config),
    )
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]

    if export_format == "excel":
        generated = _generate_excel(result, options, processing_config)
        payload = generated.getvalue() if isinstance(generated, BytesIO) else generated
    elif export_format == "json":
        payload = _generate_json(result, options)
    elif export_format == "csv":
        payload = _generate_csv(result, options)
    elif export_format == "quick_csv":
        payload = _generate_csv_for_ratio(
            result,
            options["csv_ratio"],
            include_uncertainty=options["include_uncertainty"],
            variant=options.get("csv_variant", "complete"),
        )
    else:
        raise ValueError(f"Unsupported export format: {export_format!r}")

    _cache_store(cache, cache_key, payload)
    return payload


def _reference_evidence(element_symbol: str, element_config, processing_config):
    """Return (record IDs, resolved contents) for every reference in force.

    A033: a record ID names which certificate an analysis used, not what that
    certificate said, and a managed certificate is editable under an unchanged
    ID. The resolved numbers travel beside the IDs so an edited value moves the
    digest. A032/A033: when no reference material is selected the pipeline
    resolves the element default, so provenance resolves it too instead of
    writing an empty list.
    """
    from config.constants import SR_GEOREM_REFERENCE_MATERIAL
    from config.reference_materials import get_crm_records, resolved_reference_snapshot

    contents: Dict[str, Any] = {}
    crm_ids: list = []
    if not element_symbol or element_symbol == "unknown":
        return crm_ids, contents

    selected = (
        getattr(processing_config, "reference_material", None)
        or getattr(element_config, "reference_material", None)
        or ""
    )
    roles = [("selected", str(selected))] if selected else []
    if element_symbol == "Sr":
        roles.append(("sr_ref_value", SR_GEOREM_REFERENCE_MATERIAL))

    for role, name in roles:
        if not name:
            continue
        snapshot = resolved_reference_snapshot(element_symbol, name)
        contents[role] = {
            "material_name": name,
            "rows": [list(row) for row in snapshot],
        }
        if role == "selected":
            crm_ids = [
                item.record_id
                for item in get_crm_records(element_symbol, name)
                if item.record_id
            ]
    return crm_ids, contents


def _finite_or_label(value) -> Any:
    """JSON-safe degrees of freedom: infinity is a label, not a number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == float("inf"):
        return "inf"
    if number != number:
        return "nan"
    return number


def _custom_contributor_evidence(library) -> Dict[str, Any]:
    """Serialize the user-defined contributor definitions in force.

    A032: these arrive through the custom-contributor library, not through
    either configuration dataclass, and they carry a magnitude, a type, a
    degrees-of-freedom value and a probability distribution that all change
    the reported number.
    """
    out: Dict[str, Any] = {}
    for element, definitions in sorted((library or {}).items(), key=lambda kv: str(kv[0])):
        rows = []
        for item in definitions:
            rows.append({
                "name": getattr(item, "name", ""),
                "display_name": getattr(item, "display_name", ""),
                "element_symbol": getattr(item, "element_symbol", ""),
                "u_rel_permil": float(getattr(item, "u_rel_permil", 0.0) or 0.0),
                "type_ab": getattr(item, "type_ab", ""),
                "degrees_of_freedom": _finite_or_label(
                    getattr(item, "degrees_of_freedom", float("inf"))
                ),
                "distribution": getattr(item, "distribution", ""),
                "reference": getattr(item, "reference", ""),
                "enabled": bool(getattr(item, "enabled", True)),
            })
        out[str(element)] = sorted(rows, key=lambda row: str(row["name"]))
    return out


def _sample_applicability_evidence(samples) -> Dict[str, Any]:
    """Per-observation contributor applicability, as the engines resolve it.

    A032: the assigned profile, explicit per-contributor overrides, opted-in
    custom terms and any sample-specific numerical inputs all live in
    ``sample.metadata`` and all move the reported uncertainty. Keyed by
    observation ID, because a name and a run number both repeat.
    """
    from domain.uncertainty.contributors import SampleContributorApplicability

    out: Dict[str, Any] = {}
    for sample in samples or []:
        raw = (getattr(sample, "metadata", {}) or {}).get("uncertainty_contributors")
        applicability = SampleContributorApplicability.from_sample(sample)
        entry: Dict[str, Any] = {
            "name": getattr(sample, "name", ""),
            "run_number": int(getattr(sample, "run_number", 0) or 0),
            "profile": applicability.profile,
            "overrides": dict(sorted(applicability.overrides.items())),
            "custom_enabled": sorted(applicability.custom_enabled),
        }
        # Sample-specific numerical inputs (for example the Sr QC-bias and
        # digestion values) live in the same metadata block and are consumed
        # directly by the engines.
        if isinstance(raw, dict):
            for key, value in sorted(raw.items()):
                if key in {"profile", "source", "overrides", "custom_enabled", "note"}:
                    continue
                entry[str(key)] = value
        out[str(getattr(sample, "observation_id", "") or entry["name"])] = entry
    return out


def _build_provenance(result: Optional[ProcessingResult] = None) -> Dict:
    """Build the one typed provenance object shared by every exporter."""
    from config.effective_configuration import build_effective_configuration
    from config.settings import ProcessingConfig, UncertaintyConfig
    from domain.provenance import AnalysisProvenance, dependency_versions, utc_now
    from domain.uncertainty.contributors import normalize_profile_defaults

    state = get_state()
    processing_config = getattr(state, "processing_config", None) or ProcessingConfig()
    uncertainty_config = getattr(state, "uncertainty_config", None) or UncertaintyConfig(enabled=False)
    element_config = getattr(state, "element_config", None)
    element_symbol = getattr(element_config, "symbol", None) or getattr(result, "element_symbol", "") or "unknown"
    crm_ids, reference_contents = _reference_evidence(
        element_symbol, element_config, processing_config
    )
    samples = result.samples if result is not None else (getattr(state, "samples", None) or [])
    cycle_windows = get_cycle_ranges(state)
    rejection_settings = {
        "filter_method": processing_config.filter_method,
        "filter_threshold": processing_config.filter_threshold,
        "post_ssb_enabled": processing_config.enable_post_ssb_outliers,
        "post_ssb_threshold": processing_config.post_ssb_outlier_threshold,
    }
    from config.scientific_identity import scientific_configuration
    effective = build_effective_configuration(
        workflow_id="interactive-session",
        element=element_symbol,
        processing_config=processing_config,
        uncertainty_config=uncertainty_config,
        crm_record_ids=tuple(sorted(crm_ids)),
        reference_contents=reference_contents,
        scientific_inputs={
            "runtime": scientific_configuration(element_config),
            "stored_processing": (getattr(result, "quality_metrics", {}) or {}).get(
                "processing_scientific_identity", {"status": "unknown_legacy"}),
        },
        sample_overrides=_sample_applicability_evidence(samples),
        custom_contributor_definitions=_custom_contributor_evidence(
            getattr(state, "custom_contributor_library", None)
        ),
        profile_defaults={
            str(name): dict(sorted(defaults.items()))
            for name, defaults in normalize_profile_defaults(
                getattr(state, "uncertainty_profile_defaults", None)
            ).items()
        },
        cycle_windows=cycle_windows,
        rejection_settings=rejection_settings,
    )
    masks = {}
    for sample in samples:
        layer = sample.corrected_ratios or sample.ratios
        masks[str(sample.observation_id)] = {
            ratio: data.mask.astype(bool).tolist()
            for ratio, data in sorted(layer.items())
        }
    from config.software_identity import software_identity
    implementation = software_identity()
    return AnalysisProvenance(
        created_utc=utc_now(),
        software_version=APP_VERSION,
        software_commit=implementation['commit'],
        software_dirty=implementation['dirty'],
        software_identity=implementation,
        source_filename=getattr(state, "loaded_file", None) or "",
        input_sha256=getattr(state, "file_hash", None) or "",
        effective_configuration_sha256=effective.sha256,
        effective_configuration=effective.to_dict(),
        crm_record_ids=tuple(sorted(crm_ids)),
        dependencies=dependency_versions(),
        rng={"bit_generator": "PCG64", "seed": None},
        cycle_masks=masks,
        cycle_windows=cycle_windows,
        rejection_settings=rejection_settings,
        warnings=tuple(result.warnings if result is not None else ()),
    ).to_dict()


def _generate_json(result, options: dict) -> str:
    """Generate JSON string."""
    from file_io.json_writer import export_to_json

    return export_to_json(
        result,
        include_raw=options["include_raw"],
        include_corrected=options["include_corrected"],
        include_uncertainty=options["include_uncertainty"],
        include_cycle_data=options["include_cycle_data"],
        provenance=_build_provenance(result),
        uncertainty_config=getattr(get_state(), "uncertainty_config", None),
    )


def _generate_csv(result, options: dict) -> bytes:
    """Generate CSV from a copy whose runtime masks are already folded in."""
    from file_io.csv_writer import export_to_csv

    selected_ratios = _get_selected_export_ratios()
    ratio_names = _resolve_export_ratio_names(result, selected_ratios)
    csv_text = export_to_csv(
        result,
        ratio_names=ratio_names,
        include_uncertainty=options["include_uncertainty"],
        uncertainty_config=getattr(get_state(), "uncertainty_config", None),
        provenance=_build_provenance(result),
        variant=options.get("csv_variant", "complete"),
    )
    return csv_text.encode("utf-8-sig")


def _generate_csv_for_ratio(
    result,
    ratio_name: str,
    *,
    include_uncertainty: bool = True,
    variant: str = "complete",
) -> bytes:
    """Generate one-ratio CSV from the prepared runtime-mask copy."""
    from file_io.csv_writer import export_to_csv

    csv_text = export_to_csv(
        result,
        ratio_name=ratio_name,
        include_uncertainty=include_uncertainty,
        uncertainty_config=getattr(get_state(), "uncertainty_config", None),
        provenance=_build_provenance(result),
        variant=variant,
    )
    return csv_text.encode("utf-8-sig")


def _generate_filename(export_format: str, suffix: str = "") -> str:
    """Generate a filename like 'TraceISO_Sr_20260206_143025.xlsx'."""
    state = get_state()
    element = state.element_symbol or "export"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ext_map = {"excel": "xlsx", "json": "json", "csv": "csv"}
    ext = ext_map.get(export_format, "txt")

    return f"TraceISO_{element}_{timestamp}{suffix}.{ext}"


def _get_cycle_ranges_for_export() -> Dict[str, Tuple[int, int]]:
    """Read active cycle ranges from session state (same semantics as Results tab)."""
    return get_cycle_ranges(get_state())


_RUNTIME_MASK_LAYER_NAMES = (
    "intensities",
    "corrected_intensities",
    "blank_corrected_intensities",
    "ratios",
    "corrected_ratios",
    "blank_corrected_ratios",
    "iif_corrected_ratios",
    "sr_standard_corrected_ratios",
    "drift_corrected_ratios",
    "interference_corrected_intensities",
    "interference_corrected_ratios",
    "pb_standard_corrected_ratios",
    "pb_calibrated_delta_cycles",
)


def _fold_runtime_masks_into_export_samples(
    samples: List[Sample],
    *,
    cycle_ranges: Dict[str, Tuple[int, int]],
    filter_method: str,
    filter_threshold: float,
) -> None:
    """Apply the active report window/filter once to copied sample layers."""
    from domain.filters.outlier import get_runtime_mask, sample_cycle_key
    from domain.ratio_selection import get_ssb_cycle_data

    for sample in samples:
        sample_key = sample_cycle_key(sample)
        ssb_runtime_masks: Dict[str, np.ndarray] = {}
        for ratio_name in sample.ssb_results or {}:
            ssb_cd = get_ssb_cycle_data(sample, ratio_name)
            if ssb_cd is None:
                continue
            ssb_runtime_masks[ratio_name] = get_runtime_mask(
                ssb_cd.values,
                ssb_cd.mask,
                sample.name,
                cycle_ranges=cycle_ranges,
                sample_key=sample_key,
                filter_method=filter_method,
                filter_threshold=filter_threshold,
            )

        for layer_name in _RUNTIME_MASK_LAYER_NAMES:
            mapping = getattr(sample, layer_name, None)
            if not isinstance(mapping, dict):
                continue
            for cycle_data in mapping.values():
                cycle_data.mask = get_runtime_mask(
                    cycle_data.values,
                    cycle_data.mask,
                    sample.name,
                    cycle_ranges=cycle_ranges,
                    sample_key=sample_key,
                    filter_method=filter_method,
                    filter_threshold=filter_threshold,
                )

        for ratio_name, runtime_mask in ssb_runtime_masks.items():
            payload = sample.ssb_results.get(ratio_name)
            if not isinstance(payload, dict):
                continue
            payload["ssb_mask"] = runtime_mask.copy()
            payload["ssb_outlier_mask"] = runtime_mask.copy()


def _collect_export_ratio_names(result: ProcessingResult) -> set:
    """Collect all ratio names referenced across processed samples."""
    ratio_names = set()
    for sample in result.samples:
        ratio_names.update(sample.ratios.keys())
        ratio_names.update(sample.blank_corrected_ratios.keys())
        ratio_names.update(sample.corrected_ratios.keys())
        ratio_names.update(sample.iif_corrected_ratios.keys())
        ratio_names.update(sample.drift_corrected_ratios.keys())
        if result.element_symbol in {"Sr", "Pb"}:
            ratio_names.update(sample.sr_standard_corrected_ratios)
            ratio_names.update(sample.interference_corrected_ratios)
        if result.element_symbol == "Pb":
            ratio_names.update(sample.pb_standard_corrected_ratios)
            ratio_names.update(sample.pb_calibrated_delta_cycles)
    return ratio_names


def _get_selected_export_ratios() -> Set[str]:
    """Get active ratio selection from the ratio manager."""
    state = get_state()
    if not state.samples:
        return set()
    from ui.components.custom_ratio import get_selected_ratios

    return set(get_selected_ratios(state.samples))


def _resolve_export_ratio_names(
    result: ProcessingResult,
    selected_ratios: Set[str],
) -> List[str]:
    """Resolve final export ratio list from selection + available data."""
    available = sorted(_collect_export_ratio_names(result))
    if not _has_selected_ratio_selection(get_state()):
        return available
    if not selected_ratios:
        return []
    available_set = set(available)
    return [r for r in sorted(selected_ratios) if r in available_set]


def _resolve_isotope_key(
    token: str, intensity_map: Dict[str, CycleData]
) -> Optional[str]:
    """Resolve a token to an existing intensity key."""
    if token in intensity_map:
        return token
    normalized_token = normalize_ratio_token(token)
    for key in intensity_map.keys():
        if normalize_ratio_token(key) == normalized_token:
            return key
    return None


def _compute_ratio_cycle_data(
    intensity_map: Dict[str, CycleData],
    ratio_name: str,
) -> Optional[CycleData]:
    """Build CycleData for ratio_name from an intensity dictionary."""
    if ratio_name.count("/") != 1:
        return None

    numerator_raw, denominator_raw = [p.strip() for p in ratio_name.split("/", 1)]
    numerator_key = _resolve_isotope_key(numerator_raw, intensity_map)
    denominator_key = _resolve_isotope_key(denominator_raw, intensity_map)
    if numerator_key is None or denominator_key is None:
        return None

    computed = _compute_ratios(
        intensity_map,
        {ratio_name: (numerator_key, denominator_key)},
    )
    return computed.get(ratio_name)


def _apply_filter_mask_to_ratio(
    sample: Sample,
    ratio_name: str,
    *,
    filter_method: str,
    filter_threshold: float,
) -> None:
    """Apply configured outlier filter to a corrected ratio mask."""
    if filter_method == "None":
        return

    cd = sample.corrected_ratios.get(ratio_name)
    if cd is None:
        return

    from domain.filters.outlier import apply_filter

    result = apply_filter(cd.valid_values, filter_method, filter_threshold)
    valid_indices = np.where(cd.mask)[0]
    blank_cd = sample.blank_corrected_ratios.get(ratio_name)
    for i, vi in enumerate(valid_indices):
        if i >= len(result.mask):
            continue
        cd.mask[vi] = cd.mask[vi] & result.mask[i]
        if blank_cd is not None and vi < len(blank_cd.mask):
            blank_cd.mask[vi] = blank_cd.mask[vi] & result.mask[i]


def _blank_ratio_snapshot_available(sample: Sample) -> bool:
    """Return whether *sample* has a real blank-corrected ratio source."""
    return bool(sample.blank_corrected_intensities) or bool(
        sample.blank_corrected_ratios
    )


def _ensure_selected_ratios_present(
    result: ProcessingResult,
    selected_ratios: Set[str],
    *,
    processing_config,
) -> None:
    """Populate missing selected ratios from intensity layers."""
    if not selected_ratios:
        return

    filter_method = getattr(processing_config, "filter_method", "None")
    filter_threshold = processing_config.get_active_filter_threshold(
        getattr(processing_config, "filter_method", "None"),
        fallback_threshold=getattr(processing_config, "filter_threshold", 2.0),
    )

    for sample in result.samples:
        for ratio_name in selected_ratios:
            derived_layers = []
            created_corrected = False
            if ratio_name not in sample.ratios:
                raw_cd = _compute_ratio_cycle_data(sample.intensities, ratio_name)
                if raw_cd is not None:
                    sample.ratios[ratio_name] = raw_cd
                    derived_layers.append("raw")

            if (
                ratio_name not in sample.blank_corrected_ratios
                and _blank_ratio_snapshot_available(sample)
            ):
                blank_cd = _compute_ratio_cycle_data(
                    sample.blank_corrected_intensities, ratio_name
                )
                if blank_cd is not None:
                    sample.blank_corrected_ratios[ratio_name] = blank_cd
                    derived_layers.append("blank_corrected")

            if ratio_name not in sample.corrected_ratios:
                corr_cd = _compute_ratio_cycle_data(
                    sample.corrected_intensities, ratio_name
                )
                if corr_cd is not None:
                    sample.corrected_ratios[ratio_name] = corr_cd
                    created_corrected = True
                    derived_layers.append("corrected")

            if (
                ratio_name not in sample.blank_corrected_ratios
                and ratio_name in sample.corrected_ratios
                and not _blank_ratio_snapshot_available(sample)
            ):
                sample.blank_corrected_ratios[ratio_name] = sample.corrected_ratios[
                    ratio_name
                ].copy()
                derived_layers.append("blank_corrected_from_corrected")

            if created_corrected:
                _apply_filter_mask_to_ratio(
                    sample,
                    ratio_name,
                    filter_method=filter_method,
                    filter_threshold=filter_threshold,
                )
            if derived_layers:
                sample.metadata.setdefault("_export_time_ratio_derivations", {})[
                    ratio_name
                ] = derived_layers


def _filter_result_to_ratios(
    result: ProcessingResult, ratio_names: List[str]
) -> ProcessingResult:
    """Keep only selected ratios across all ratio-keyed sample structures."""
    ratio_set = set(ratio_names)
    for sample in result.samples:
        sample.ratios = {k: v for k, v in sample.ratios.items() if k in ratio_set}
        sample.blank_corrected_ratios = {
            k: v for k, v in sample.blank_corrected_ratios.items() if k in ratio_set
        }
        sample.corrected_ratios = {
            k: v for k, v in sample.corrected_ratios.items() if k in ratio_set
        }
        sample.iif_corrected_ratios = {
            k: v for k, v in sample.iif_corrected_ratios.items() if k in ratio_set
        }
        sample.sr_standard_corrected_ratios = {
            k: v for k, v in sample.sr_standard_corrected_ratios.items() if k in ratio_set
        }
        if result.element_symbol in {"Sr", "Pb"}:
            sample.interference_corrected_ratios = {
                k: v for k, v in sample.interference_corrected_ratios.items() if k in ratio_set
            }
        if result.element_symbol == "Pb":
            sample.pb_standard_corrected_ratios = {k: v for k, v in sample.pb_standard_corrected_ratios.items() if k in ratio_set}
            sample.pb_calibrated_delta_cycles = {k: v for k, v in sample.pb_calibrated_delta_cycles.items() if k in ratio_set}
        sample.drift_corrected_ratios = {
            k: v for k, v in sample.drift_corrected_ratios.items() if k in ratio_set
        }
        sample.ssb_results = {
            k: v for k, v in sample.ssb_results.items() if k in ratio_set
        }
        sample.delta_results = {
            k: v for k, v in sample.delta_results.items() if k in ratio_set
        }
        sample.uncertainty = {
            k: v for k, v in sample.uncertainty.items() if k in ratio_set
        }
        sample.mc_results = {
            k: v for k, v in (sample.mc_results or {}).items() if k in ratio_set
        }
    return result


def _runtime_export_input_token(
    state,
    result: ProcessingResult,
):
    """Return a content-sensitive token for prepared runtime export data."""
    from config.scientific_identity import digest, scientific_configuration
    from config.software_identity import software_identity
    from ui.runtime_budget_cache import (
        _custom_contributor_token,
        _freeze_for_cache,
        _processing_config_token,
        _profile_defaults_token,
        _session_samples_token,
        _uncertainty_config_token,
    )

    return (
        digest(software_identity()),
        result.element_symbol,
        digest(result.quality_metrics),
        digest([(s.observation_id, s.uncertainty) for s in result.samples]),
        digest([(s.observation_id, s.warnings) for s in result.samples]),
        getattr(state, "loaded_file", None),
        getattr(state, "file_hash", None),
        _session_samples_token(result.samples),
        # MC records are mutable session membership even though each record is
        # frozen. A new run must invalidate previously prepared downloads.
        digest([(s.observation_id, s.mc_results) for s in result.samples]),
        tuple(sorted(_get_selected_export_ratios())),
        digest(scientific_configuration(getattr(state, "element_config", None))),
        _processing_config_token(getattr(state, "processing_config", None)),
        _uncertainty_config_token(getattr(state, "uncertainty_config", None)),
        _freeze_for_cache(_get_cycle_ranges_for_export()),
        _freeze_for_cache((result.quality_metrics or {}).get("drift_fit_info") or {}),
        _custom_contributor_token(getattr(state, "custom_contributor_library", None)),
        _profile_defaults_token(getattr(state, "uncertainty_profile_defaults", None)),
        # Sr applied-state/method metadata changes reportability and Summary
        # even when the stored cycle arrays themselves have not changed.
        _freeze_for_cache([
            (s.observation_id, s.metadata.get("sr_standard_calibration"),
             s.metadata.get("_sr_chain_method"), s.warnings)
            for s in result.samples
        ]) if result.element_symbol == "Sr" else None,
        tuple(result.warnings),
        _freeze_for_cache(result.quality_metrics) if result.element_symbol == "Pb" else None,
        _freeze_for_cache([(s.observation_id, s.metadata, s.warnings) for s in result.samples]) if result.element_symbol == "Pb" else None,
        _freeze_for_cache(current_pb_calibration_freshness(state)) if result.element_symbol == "Pb" else None,
    )


def _build_runtime_export_result(
    result: ProcessingResult,
) -> ProcessingResult:
    """Return cached runtime export data for the current session window."""
    state = get_state()
    _require_current_pb_calibration(state)
    cache = _bounded_session_cache(_RUNTIME_EXPORT_CACHE_KEY)
    cache_key = _runtime_export_input_token(state, result)
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return copy.deepcopy(cache[cache_key])

    prepared = _build_runtime_export_result_uncached(result)
    _cache_store(cache, cache_key, copy.deepcopy(prepared))
    return copy.deepcopy(prepared)


def _build_runtime_export_result_uncached(
    result: ProcessingResult,
    *, all_ratios: bool = False,
) -> ProcessingResult:
    """Build copied export data with uncertainty mapped to the current window."""
    state = get_state()
    freshness = current_pb_calibration_freshness(state)
    if freshness and freshness.get("status") == "stale":
        raise RuntimeError(
            "Current-session export refused: Pb-standard calibration is stale; rerun data reduction."
        )

    from ui.runtime_budget_cache import build_cached_runtime_uncertainty_map
    from domain.uncertainty.runtime import lookup_runtime_budget
    from domain.runtime_delta import (
        build_runtime_delta_map,
        make_runtime_delta_key,
    )

    out = ProcessingResult(
        samples=[s.copy() for s in result.samples],
        element_symbol=result.element_symbol,
        quality_metrics=copy.deepcopy(result.quality_metrics),
        warnings=list(result.warnings),
    )

    selected_ratios = _get_selected_export_ratios()
    if all_ratios:
        selected_ratios = set(_collect_export_ratio_names(out)) | selected_ratios
    _ensure_selected_ratios_present(
        out,
        selected_ratios,
        processing_config=state.processing_config,
    )
    ratio_names = sorted(selected_ratios) if all_ratios else _resolve_export_ratio_names(out, selected_ratios)
    # Preserve normalization/calibration dependencies until runtime consumers finish.
    if result.element_symbol != "Pb":
        _filter_result_to_ratios(out, ratio_names)
    if not ratio_names:
        return _filter_result_to_ratios(out, ratio_names)

    if not state.element_config or not state.processing_config:
        return _filter_result_to_ratios(out, ratio_names)

    cycle_ranges = _get_cycle_ranges_for_export()
    uncertainty_config = getattr(state, "uncertainty_config", None)
    runtime_map = build_cached_runtime_uncertainty_map(
        out.samples,
        ratio_names,
        element_config=state.element_config,
        processing_config=state.processing_config,
        uncertainty_config=uncertainty_config,
        all_session_samples=out.samples,
        cycle_ranges=cycle_ranges,
        filter_method=state.processing_config.filter_method,
        filter_threshold=state.processing_config.get_active_filter_threshold(),
        drift_fit_info=out.quality_metrics.get("drift_fit_info"),
        custom_contributor_library=state.custom_contributor_library,
        profile_defaults=getattr(state, "uncertainty_profile_defaults", None),
    )
    runtime_delta_map = {}
    from domain.pb_standard_calibration import calibrated_delta_requested

    if state.element_config.supports_delta and (
        state.processing_config.enable_delta
        or calibrated_delta_requested(state.element_config.symbol, state.processing_config)
    ):
        runtime_delta_map = build_runtime_delta_map(
            out.samples,
            ratio_names,
            cycle_ranges=cycle_ranges,
            filter_method=state.processing_config.filter_method,
            filter_threshold=state.processing_config.get_active_filter_threshold(),
            processing_config=state.processing_config,
            element_config=state.element_config,
            calibration_freshness=current_pb_calibration_freshness(state),
        )

    # In the SSB/delta workflow standards are bracketing references, not
    # analyte unknowns, so they carry no reportable uncertainty budget. Mirror
    # the Uncertainty tab (summary_detail._filter_uncertainty_budget_samples)
    # so the exported budgets match what the user sees on screen.
    exclude_standards_from_budget = _is_ssb_delta_budget_mode(
        state.element_config, state.processing_config, uncertainty_config
    )

    stored_delta = {s.observation_id: copy.deepcopy(s.delta_results) for s in out.samples} if all_ratios else {}
    for sample in out.samples:
        sample.uncertainty = {}
        sample.delta_results = {}

    for sample in out.samples:
        skip_budget = exclude_standards_from_budget and (
            sample.is_standard or sample.is_blank
        )
        for ratio_name in ratio_names:
            if not skip_budget:
                budget = lookup_runtime_budget(runtime_map, sample, ratio_name)
                if budget is not None:
                    record = sample.mc_results.get(ratio_name)
                    if record is not None:
                        from ui.tabs.uncertainty_sections.equations_mc import resolve_live_mc_digest
                        original = next(s for s in result.samples if s.observation_id == sample.observation_id)
                        # Resolve before export filtering/window materialization;
                        # those transformations are reporting operations, not edits.
                        budget._current_mc_input_digests = {
                            record.input_digest: resolve_live_mc_digest(
                                record, original, ratio_name, state, budget,
                            )
                        }
                    sample.uncertainty[ratio_name] = budget
            delta_result = runtime_delta_map.get(
                make_runtime_delta_key(sample, ratio_name)
            )
            if delta_result is not None:
                sample.delta_results[ratio_name] = delta_result.to_payload()
                if all_ratios:
                    _retain_recorded_delta_values(
                        sample.delta_results[ratio_name],
                        stored_delta.get(sample.observation_id, {}).get(ratio_name, {}),
                    )
                from domain.pb_calibration_records import (
                    PB_CALIBRATED_DELTA_RECORD_FAMILY, calibrated_delta_record,
                )
                record = calibrated_delta_record(sample, ratio_name)
                if record is not None and record.status == "applied":
                    statistic = state.processing_config.pb_standard_calibration.delta_precision_statistic
                    precision = {
                        "none": None,
                        "sd": delta_result.delta_sd if delta_result.n >= 2 else None,
                        "se": delta_result.delta_se if delta_result.n >= 2 else None,
                    }[statistic]
                    updated = replace(
                        record, delta_mean=delta_result.delta, n_valid=delta_result.n,
                        precision_statistic=statistic, precision_value=precision,
                    )
                    sample.correction_records[PB_CALIBRATED_DELTA_RECORD_FAMILY][ratio_name] = updated

    _fold_runtime_masks_into_export_samples(
        out.samples,
        cycle_ranges=cycle_ranges,
        filter_method=state.processing_config.filter_method,
        filter_threshold=state.processing_config.get_active_filter_threshold(),
    )

    return _filter_result_to_ratios(out, ratio_names)


def _retain_recorded_delta_values(runtime, stored) -> None:
    """Keep recorded excluded values only when their reference and layer still match."""
    if not stored or runtime.get("std_mean") != stored.get("std_mean") or runtime.get("source_layer") != stored.get("source_layer"):
        return
    current = runtime.get("delta_per_cycle")
    original = stored.get("delta_per_cycle")
    if current is None or original is None:
        return
    current = np.asarray(current, dtype=float).copy()
    original = np.asarray(original, dtype=float)
    if current.shape != original.shape:
        return
    missing = ~np.isfinite(current) & np.isfinite(original)
    current[missing] = original[missing]
    runtime["delta_per_cycle"] = current
    # The runtime delta_mask is intentionally unchanged.


def _require_current_sr_calibration(result: ProcessingResult) -> None:
    """Refuse an Sr calibration whose selected standard support has changed."""
    from domain.calibration_dependencies import observation_window
    from domain.sr_standard_calibration import sr_calibration_record

    windows = _get_cycle_ranges_for_export()
    observations = {s.observation_id: s for s in result.samples}
    checked = set()
    for sample in result.samples:
        record = sr_calibration_record(sample)
        if record.get("status") != "applied":
            continue
        ratio = record.get("ratio")
        for member in record.get("standards", []):
            identity = member.get("observation_id")
            if (identity, ratio) in checked:
                continue
            checked.add((identity, ratio))
            standard = observations.get(identity)
            cd = standard.iif_corrected_ratios.get(ratio) if standard else None
            stale = cd is None
            if cd is not None:
                window = observation_window(standard, windows)
                mask = cd.mask.copy()
                if window is not None:
                    indices = np.arange(1, cd.n_total + 1)
                    mask &= (indices >= window[0]) & (indices <= window[1])
                current = CycleData(cd.values.copy(), mask)
                stale = ((np.flatnonzero(mask) + 1).tolist() != member.get("accepted_cycles")
                         or not np.isclose(current.mean, member.get("mean", np.nan), rtol=1e-12, atol=0))
            if stale:
                raise RuntimeError("Current-session export refused: Sr-standard calibration is stale; rerun data reduction.")


def _build_sr_cycle_export_result(result: ProcessingResult) -> ProcessingResult:
    """Copy every recorded series, folding runtime masks without deriving ratios."""
    state = get_state()
    _require_current_sr_calibration(result)
    out = ProcessingResult(
        samples=[s.copy() for s in result.samples], element_symbol=result.element_symbol,
        quality_metrics=copy.deepcopy(result.quality_metrics), warnings=list(result.warnings),
    )
    cfg = state.processing_config
    _fold_runtime_masks_into_export_samples(
        out.samples, cycle_ranges=_get_cycle_ranges_for_export(),
        filter_method=cfg.filter_method, filter_threshold=cfg.get_active_filter_threshold(),
    )
    return out


def _require_current_pb_calibration(state):
    cfg = getattr(state, "processing_config", None)
    if (getattr(getattr(state, "element_config", None), "symbol", None) == "Pb"
            and getattr(getattr(cfg, "pb_standard_calibration", None), "enabled", False)
            and not getattr(cfg, "apply_mass_bias_correction", False)):
        raise RuntimeError("Pb-standard calibration requires external Tl normalization; correct the settings and reprocess.")
    freshness = current_pb_calibration_freshness(state)
    if freshness and freshness.get("status") == "stale":
        raise RuntimeError("Current-session export refused: Pb-standard calibration is stale; rerun data reduction.")


def _build_pb_cycle_export_result(result):
    """Full-scope recorded arrays with independent runtime acceptance masks."""
    state = get_state()
    _require_current_pb_calibration(state)
    out = ProcessingResult(samples=[s.copy() for s in result.samples], element_symbol=result.element_symbol,
                           quality_metrics=copy.deepcopy(result.quality_metrics), warnings=list(result.warnings))
    cfg = state.processing_config
    _fold_runtime_masks_into_export_samples(out.samples, cycle_ranges=_get_cycle_ranges_for_export(),
        filter_method=cfg.filter_method, filter_threshold=cfg.get_active_filter_threshold())
    return out


def _is_pb_tl_excel_report(element_config, processing_config, uncertainty_config) -> bool:
    if element_config is None or processing_config is None or uncertainty_config is None:
        return False
    resolve = getattr(uncertainty_config, "resolve_engine", None)
    return (element_config.symbol == "Pb" and bool(processing_config.apply_mass_bias_correction)
            and callable(resolve) and resolve("Pb", processing_config=processing_config) == "pb_tl_external_normalization")


def _is_sr_excel_report(element_config, processing_config, uncertainty_config) -> bool:
    if element_config is None or processing_config is None or uncertainty_config is None:
        return False
    resolve = getattr(uncertainty_config, "resolve_engine", None)
    return (element_config.symbol == "Sr"
            and bool(getattr(processing_config, "apply_mass_bias_correction", False))
            and callable(resolve)
            and resolve("Sr", processing_config=processing_config) == "internal_normalization")


def _is_ssb_excel_report(element_config, processing_config, uncertainty_config) -> bool:
    """Route from processing settings, not a potentially unsynchronised budget flag."""
    if element_config is None or processing_config is None or uncertainty_config is None:
        return False
    resolve = getattr(uncertainty_config, "resolve_engine", None)
    engine = (resolve(element_config.symbol, processing_config=processing_config)
              if callable(resolve) else getattr(uncertainty_config, "engine", ""))
    return engine == "ssb_delta" and (
        bool(processing_config.enable_ssb)
        or bool(processing_config.enable_delta)
        or element_config.symbol in {"Li", "B"}
    )


def _is_ssb_delta_budget_mode(
    element_config,
    processing_config,
    uncertainty_config,
) -> bool:
    """True when uncertainty budgets should exclude standards/blanks.

    Mirrors ``summary_detail._is_ssb_delta_budget_mode``: the SSB/delta engine
    with bracketing active, where standards are references rather than unknowns.
    """
    if element_config is None or uncertainty_config is None:
        return False
    symbol = getattr(element_config, "symbol", "") or ""
    resolve_engine = getattr(uncertainty_config, "resolve_engine", None)
    engine = (
        resolve_engine(symbol, processing_config=processing_config)
        if callable(resolve_engine)
        else getattr(uncertainty_config, "engine", "")
    )
    return engine == "ssb_delta" and bool(
        getattr(uncertainty_config, "enable_ssb", False)
    )


def _settings_changed_since_processing(state) -> bool:
    """Compatibility wrapper for the shared settings-change helper."""
    return settings_changed_since_processing(state)

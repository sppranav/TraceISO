"""Excel export for TraceISO."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

try:  # openpyxl >= 3.1
    from openpyxl.cell.rich_text import CellRichText, TextBlock
    from openpyxl.cell.text import InlineFont
except ImportError:  # pragma: no cover - compatibility with stale environments
    CellRichText = None  # type: ignore[assignment,misc]
    TextBlock = None  # type: ignore[assignment,misc]
    InlineFont = None  # type: ignore[assignment,misc]

from config.constants import APP_VERSION
from domain.layers import cycle_data_equal
from domain.models import CycleData, ProcessingResult, Sample, UncertaintyBudget, UncertaintyContributor
from domain.ratio_selection import (
    get_ssb_cycle_data,
    select_best_pre_drift_ratio_layer,
    select_summary_export_layer,
)
from domain.statistics import calculate_cycle_statistics
from domain.uncertainty.contributors import (
    BUILTIN_PROFILES,
    ENGINE_SSB_CONTRIBUTOR_NAMES,
    ContributorProfile,
    SampleContributorApplicability,
)
from domain.uncertainty.mc_result import (
    MC_RESULT_SCHEMA_NAME,
    MC_RESULT_SCHEMA_VERSION,
    DRAW_ARRAY_POLICY,
    has_mc_results,
    iter_mc_results,
    log_mc_results_exported,
    mc_record_freshness_for_budget,
)
from domain.uncertainty.scope import (
    CROSS_RATIO_INDEPENDENCE_NOTE,
    budget_scope_label,
    budget_scope_note,
    canonical_uncertainty_scope_json,
    is_invalid_budget_scope,
    uncertainty_scope_payload,
)
from file_io.sanitize import (
    build_budget_note,
    format_contributor_name,
    format_isotope_label,
    format_ratio_token,
    safe_float,
    literal_excel_text,
    summarize_budget_state,
)
from domain.provenance import canonical_provenance_json, normalize_provenance
from file_io.report_formatting import report_uncertainty

# Provenance constant — bump when the uncertainty framework changes materially.
UNCERTAINTY_FRAMEWORK_VERSION = "TraceISO GUM v1.1 — independent ratio budgets"

#: Format token used in the ``engine_b_mc.exported`` lifecycle record. It is a
#: fixed word, never a filename or a path.
_MC_EXPORT_FORMAT = "excel"

#: Worksheet title for the durable Monte Carlo cross-check records.
MC_CROSS_CHECK_SHEET = "MC Cross-Check"

#: Workbook layout identifiers (U52). The diagnostic layout is the default and
#: is the workbook as it has always been. The analyst report is opt-in: it adds
#: the ``Reported Results`` sheet, one row per observation x ratio x output
#: mode, and keeps the diagnostic ``Results`` sheet as an optional companion.
EXCEL_LAYOUT_DIAGNOSTIC = "traceiso.excel_layout.diagnostic.v1"
EXCEL_LAYOUT_ANALYST_REPORT = "traceiso.excel_layout.analyst_report.v1"
EXCEL_LAYOUT_SSB_REPORT = "traceiso.excel_layout.ssb_report.v1"
EXCEL_LAYOUT_SR_REPORT = "traceiso.excel_layout.sr_report.v1"
EXCEL_LAYOUT_PB_REPORT = "traceiso.excel_layout.pb_report.v1"
EXCEL_LAYOUTS = (
    EXCEL_LAYOUT_DIAGNOSTIC,
    EXCEL_LAYOUT_ANALYST_REPORT,
    EXCEL_LAYOUT_SSB_REPORT,
    EXCEL_LAYOUT_SR_REPORT,
    EXCEL_LAYOUT_PB_REPORT,
)
REPORTED_RESULTS_SHEET = "Reported Results"

#: Contributor Profiles scope (U55). A routine report lists the profiles the
#: exported observations actually resolve to; the complete library is an audit
#: choice.
PROFILE_SCOPE_USED = "used"
PROFILE_SCOPE_LIBRARY = "library"
PROFILE_SCOPES = (PROFILE_SCOPE_USED, PROFILE_SCOPE_LIBRARY)


# Public API


def export_to_excel(
    result: ProcessingResult,
    *,
    include_raw: bool = True,
    include_corrected: bool = True,
    include_uncertainty: bool = True,
    include_cycle_data: bool = False,
    processing_config: Optional[object] = None,
    uncertainty_config: Optional[object] = None,
    loaded_filename: Optional[str] = None,
    raw_layout: str = "stacked",   # "stacked" | "per_sample"
    include_cover_summary: bool = True,
    include_results_final: bool = True,
    include_raw_sheet: bool = True,
    include_uncertainty_budgets: bool = True,
    include_uncertainty_budget_wide: bool = True,
    corrected_label: str = "Corrected",
    contributor_profiles: Optional[Mapping[str, ContributorProfile]] = None,
    provenance: Optional[Mapping[str, object]] = None,
    include_metadata_sheets: bool = True,
    layout: str = EXCEL_LAYOUT_DIAGNOSTIC,
    contributor_profile_scope: str = PROFILE_SCOPE_USED,
    ssb_report_options: Optional[object] = None,
    cycle_result: Optional[ProcessingResult] = None,
    sr_report_options: Optional[object] = None,
    element_config: Optional[object] = None,
    pb_report_options: Optional[object] = None,
    calibration_freshness: Optional[Mapping[str, object]] = None,
) -> BytesIO:
    """Export a :class:`ProcessingResult` to an Excel workbook in memory."""
    if layout not in EXCEL_LAYOUTS:
        raise ValueError(f"Unknown Excel layout {layout!r}; expected one of {EXCEL_LAYOUTS}")
    if contributor_profile_scope not in PROFILE_SCOPES:
        raise ValueError(
            f"Unknown contributor profile scope {contributor_profile_scope!r}; "
            f"expected one of {PROFILE_SCOPES}"
        )
    if layout == EXCEL_LAYOUT_PB_REPORT:
        from file_io.pb_excel_writer import resolve_pb_options, export_pb_report

        return export_pb_report(
            result, options=resolve_pb_options(pb_report_options,
                detail=include_uncertainty_budget_wide, cycles=include_cycle_data),
            cycle_result=cycle_result, processing_config=processing_config,
            uncertainty_config=uncertainty_config, loaded_filename=loaded_filename,
            element_config=element_config, calibration_freshness=calibration_freshness,
            provenance=provenance,
            include_metadata_sheets=include_metadata_sheets,
        )
    if layout == EXCEL_LAYOUT_SR_REPORT:
        from file_io.sr_excel_writer import resolve_sr_options, export_sr_report

        return export_sr_report(
            result,
            options=resolve_sr_options(sr_report_options, detail=include_uncertainty_budget_wide,
                                       cycles=include_cycle_data),
            cycle_result=cycle_result, processing_config=processing_config,
            uncertainty_config=uncertainty_config, loaded_filename=loaded_filename,
            element_config=element_config,
            provenance=provenance,
            include_metadata_sheets=include_metadata_sheets,
        )
    if layout == EXCEL_LAYOUT_SSB_REPORT:
        from file_io.ssb_excel_writer import SSBReportOptions, export_ssb_report

        options = ssb_report_options
        if options is None:
            options = SSBReportOptions(
                include_uncertainty_detail=include_uncertainty_budget_wide,
                include_cycle_data=include_cycle_data,
            )
        if not isinstance(options, SSBReportOptions):
            raise TypeError("ssb_report_options must be an SSBReportOptions instance")
        return export_ssb_report(
            result,
            options=options,
            cycle_result=cycle_result,
            processing_config=processing_config,
            uncertainty_config=uncertainty_config,
            loaded_filename=loaded_filename,
            provenance=provenance,
            include_metadata_sheets=include_metadata_sheets,
        )
    wb = Workbook()
    wb._traceiso_include_metadata_sheets = include_metadata_sheets
    has_sheet = False
    ratio_names = _get_ratio_names(result)
    emits_profiles = bool(
        include_uncertainty and include_uncertainty_budgets and _has_uncertainty(result)
    )

    if include_cover_summary:
        _create_cover_summary_sheet(
            wb, result, processing_config, loaded_filename,
            uncertainty_config=uncertainty_config,
            include_uncertainty=include_uncertainty,
            layout=layout,
            profile_scope=contributor_profile_scope if emits_profiles else None,
        )
        has_sheet = True
    else:
        wb.remove(wb.active)

    if layout == EXCEL_LAYOUT_ANALYST_REPORT:
        _create_reported_results_sheet(
            wb,
            result,
            include_uncertainty=include_uncertainty,
            ratio_names=ratio_names,
        )
        has_sheet = True

    if include_results_final:
        _create_results_final_sheet(
            wb,
            result,
            include_raw,
            include_corrected,
            include_uncertainty,
            corrected_label=corrected_label,
            ratio_names=ratio_names,
        )
        has_sheet = True

    if include_uncertainty and include_uncertainty_budgets and _has_uncertainty(result):
        _create_uncertainty_budgets_sheet(
            wb, result,
            uncertainty_config=uncertainty_config,
            ratio_names=ratio_names,
        )
        _create_applicability_sheet(wb, result)
        _create_contributor_profiles_sheet(
            wb,
            contributor_profiles or BUILTIN_PROFILES,
            samples=result.samples,
            scope=contributor_profile_scope,
        )
        has_sheet = True

    if include_uncertainty and include_uncertainty_budget_wide and _has_uncertainty(result):
        before = len(wb.worksheets)
        _create_uncertainty_budget_wide_sheet(wb, result, ratio_names=ratio_names)
        has_sheet = has_sheet or len(wb.worksheets) > before

    if include_uncertainty and has_mc_results(result.samples):
        _create_mc_cross_check_sheet(
            wb, result, uncertainty_config=uncertainty_config
        )
        log_mc_results_exported(result.samples, export_format=_MC_EXPORT_FORMAT)
        has_sheet = True

    if include_raw and include_raw_sheet:
        _create_raw_data_summary_sheet(
            wb,
            result,
            include_corrected,
            raw_layout,
            corrected_label=corrected_label,
            ratio_names=ratio_names,
        )
        has_sheet = True

    if include_cycle_data:
        for sample in result.samples:
            if not sample.is_blank:
                _create_cycle_sheet(wb, sample)
                has_sheet = True

    if include_metadata_sheets and provenance is not None:
        _create_canonical_provenance_sheet(wb, provenance)
        has_sheet = True

    if include_metadata_sheets and include_uncertainty and _has_uncertainty(result):
        _create_uncertainty_scope_sheet(wb)
        has_sheet = True

    if not has_sheet:
        ws = wb.active if wb.worksheets else wb.create_sheet("Export")
        ws.title = "Export"
        ws["A1"] = "No presentation sheets selected."
    finalize_workbook_evidence(wb, result, provenance=provenance, uncertainty_config=uncertainty_config)
    _validate_workbook_sheet_names(wb)
    buf = BytesIO()
    try:
        wb.save(buf)
    except Exception as exc:
        raise ValueError(
            f"Excel export failed while saving {len(wb.worksheets)} worksheet(s): {exc}"
        ) from exc
    buf.seek(0)
    return buf


EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_COLUMNS = 16_384


def finalize_workbook_evidence(wb, result, *, provenance=None, uncertainty_config=None):
    """Persist canonical evidence independently of optional presentation sheets."""
    if getattr(wb, '_traceiso_include_metadata_sheets', True) and 'Provenance' not in wb.sheetnames:
        _create_canonical_provenance_sheet(wb, provenance if provenance is not None else {
            'identity_status': 'export_provenance_not_recorded',
            'processing_scientific_identity': result.quality_metrics.get('processing_scientific_identity'),
        })
    if getattr(wb, '_traceiso_include_metadata_sheets', True) and 'Uncertainty Scope' not in wb.sheetnames:
        _create_uncertainty_scope_sheet(wb)
    if has_mc_results(result.samples) and MC_CROSS_CHECK_SHEET not in wb.sheetnames:
        _create_mc_cross_check_sheet(wb, result, uncertainty_config=uncertainty_config)
        log_mc_results_exported(result.samples, export_format=_MC_EXPORT_FORMAT)
    # openpyxl can save columns beyond Excel's XFD boundary. Check every sheet,
    # including canonical evidence, before emitting an unreadable workbook.
    for ws in wb:
        if ws.max_row > EXCEL_MAX_ROWS or ws.max_column > EXCEL_MAX_COLUMNS:
            raise ValueError(
                f"Worksheet {ws.title!r} dimensions {ws.max_row} rows x "
                f"{ws.max_column} columns exceed Excel limits "
                f"({EXCEL_MAX_ROWS} x {EXCEL_MAX_COLUMNS}); reduce the export selection."
            )
    # XLSX has a native text cell type: never turn untrusted strings into formulas.
    for ws in wb:
        for row in ws:
            for cell in row:
                if cell.data_type == 'f':
                    cell.data_type = 's'


def workbook_evidence_names(result):
    return ['Provenance', 'Uncertainty Scope'] + ([MC_CROSS_CHECK_SHEET] if has_mc_results(result.samples) else [])


def excel_sheet_manifest(
    result: ProcessingResult,
    *,
    include_raw: bool = True,
    include_uncertainty: bool = True,
    include_cycle_data: bool = False,
    raw_layout: str = "stacked",
    include_cover_summary: bool = True,
    include_results_final: bool = True,
    include_raw_sheet: bool = True,
    include_uncertainty_budgets: bool = True,
    include_uncertainty_budget_wide: bool = True,
    provenance_present: bool = False,
    layout: str = EXCEL_LAYOUT_DIAGNOSTIC,
    sr_report_options: Optional[object] = None,
    pb_report_options: Optional[object] = None,
    processing_config: Optional[object] = None,
    uncertainty_config: Optional[object] = None,
    calibration_freshness: Optional[Mapping[str, object]] = None,
) -> List[str]:
    """Describe worksheets the writer is eligible to emit for these inputs."""
    if layout not in EXCEL_LAYOUTS:
        raise ValueError(f"Unknown Excel layout {layout!r}; expected one of {EXCEL_LAYOUTS}")
    if layout == EXCEL_LAYOUT_PB_REPORT:
        from file_io.pb_excel_writer import resolve_pb_options, pb_has_detail, pb_sheet_names, validate_pb_report

        validate_pb_report(result, processing_config, uncertainty_config, calibration_freshness)
        return pb_sheet_names(
            resolve_pb_options(pb_report_options, detail=include_uncertainty_budget_wide,
                               cycles=include_cycle_data),
            has_detail=pb_has_detail(result, processing_config=processing_config),
        ) + workbook_evidence_names(result) + ['Long Payloads (only when required)']
    if layout == EXCEL_LAYOUT_SR_REPORT:
        from file_io.sr_excel_writer import resolve_sr_options, sr_has_detail, sr_sheet_names

        return sr_sheet_names(
            resolve_sr_options(sr_report_options, detail=include_uncertainty_budget_wide,
                               cycles=include_cycle_data),
            has_detail=sr_has_detail(result),
        ) + workbook_evidence_names(result) + ['Long Payloads (only when required)']
    if layout == EXCEL_LAYOUT_SSB_REPORT:
        from file_io.ssb_excel_writer import SSBReportOptions, ssb_has_detail, ssb_sheet_names

        return ssb_sheet_names(
            SSBReportOptions(
                include_uncertainty_detail=include_uncertainty_budget_wide,
                include_cycle_data=include_cycle_data,
            ),
            has_detail=ssb_has_detail(result),
        ) + workbook_evidence_names(result) + ['Long Payloads (only when required)']
    names: List[str] = []
    if include_cover_summary:
        names.append("Summary")
    if layout == EXCEL_LAYOUT_ANALYST_REPORT:
        names.append(REPORTED_RESULTS_SHEET)
    if include_results_final:
        names.append("Results")
    has_uncertainty = _has_uncertainty(result)
    if include_uncertainty and include_uncertainty_budgets and has_uncertainty:
        names.extend(("Uncertainty_Budgets", "Sample Applicability", "Contributor Profiles"))
    if include_uncertainty and include_uncertainty_budget_wide and _has_wide_budget_content(result):
        names.append("Budget_Detail")
    if include_uncertainty and has_mc_results(result.samples):
        names.append(MC_CROSS_CHECK_SHEET)
    if include_raw and include_raw_sheet:
        if raw_layout == "per_sample":
            names.extend(_preview_unique_cycle_sheet_names(result.samples, "Cycle_"))
        else:
            names.append("Cycle_Data")
    if include_cycle_data:
        names.extend(
            _preview_unique_cycle_sheet_names(
                [sample for sample in result.samples if not sample.is_blank],
                "Cycles_",
                existing=names,
            )
        )
    if provenance_present:
        names.append("Provenance")
    if include_uncertainty and has_uncertainty:
        names.append("Uncertainty Scope")
    if not names:
        names.append('Export')
    for name in workbook_evidence_names(result):
        if name not in names: names.append(name)
    names.append("Long Payloads (only when required)")
    return names or ["Export"]


def _preview_unique_cycle_sheet_names(
    samples: List[Sample],
    prefix: str,
    *,
    existing: Optional[List[str]] = None,
) -> List[str]:
    wb = Workbook()
    wb.remove(wb.active)
    for name in existing or []:
        if " (only when required)" not in name:
            wb.create_sheet(str(name)[:31])
    generated: List[str] = []
    for sample in samples:
        sheet_name = _build_unique_sheet_name(wb, f"{prefix}{sample.name}", fallback="Cycles")
        wb.create_sheet(sheet_name)
        generated.append(sheet_name)
    return generated


# Sheet builders


# Long canonical payloads


#: openpyxl silently truncates any cell string to this length
#: (``Cell.check_string`` does ``value[:32767]``). A canonical JSON payload
#: cut at an arbitrary character stops being JSON, and the workbook's stated
#: full-precision authority goes with it: the 16-significant-digit numeric
#: columns cannot reconstruct the exact record on their own.
_EXCEL_MAX_CELL_CHARS = 32767

#: Versioned overflow representation. A payload that does not fit one cell is
#: written to this sheet as ordered chunks with the metadata needed to
#: reassemble and verify it. Payloads that do fit stay inline unchanged, so
#: existing readers of short exports see no difference.
_LONG_PAYLOAD_SHEET = "Long Payloads"
_LONG_PAYLOAD_SCHEMA = "traceiso.excel_long_payload"
_LONG_PAYLOAD_SCHEMA_VERSION = "1.0"
_LONG_PAYLOAD_CHUNK_CHARS = 32000
_LONG_PAYLOAD_HEADERS = [
    "Schema",
    "Payload ID",
    "Chunk Index",
    "Chunk Count",
    "Total Characters",
    "SHA-256",
    "Chunk Text",
]
#: Marker opening every pointer cell, so a reader can tell a pointer from a
#: payload without guessing.
_LONG_PAYLOAD_POINTER_PREFIX = (
    f"{_LONG_PAYLOAD_SCHEMA} v{_LONG_PAYLOAD_SCHEMA_VERSION} | "
)


def _long_payload_pointer(payload_id: str, chunks: int, total: int, digest: str) -> str:
    return (
        f"{_LONG_PAYLOAD_POINTER_PREFIX}sheet={_LONG_PAYLOAD_SHEET} | "
        f"payload_id={payload_id} | chunks={chunks} | characters={total} | "
        f"sha256={digest}"
    )


def _store_long_payload(wb: Workbook, payload_id: str, text: str) -> str:
    """Return the cell value for *text*, chunking it when it cannot fit.

    Short payloads are returned unchanged and stay in their original cell.
    A long payload is split into ordered chunks on the overflow sheet and the
    original cell receives a pointer carrying the chunk count, the exact
    character count and the SHA-256 of the whole payload, so a reader can
    reassemble it and prove the reassembly is complete.
    """
    if len(text) <= _EXCEL_MAX_CELL_CHARS:
        return text

    if not getattr(wb, "_traceiso_include_metadata_sheets", True):
        return "Oversized metadata omitted from Excel; use JSON export for the complete record."

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunks = [
        text[i : i + _LONG_PAYLOAD_CHUNK_CHARS]
        for i in range(0, len(text), _LONG_PAYLOAD_CHUNK_CHARS)
    ]

    if _LONG_PAYLOAD_SHEET in wb.sheetnames:
        ws = wb[_LONG_PAYLOAD_SHEET]
    else:
        ws = wb.create_sheet(_LONG_PAYLOAD_SHEET)
        for col, header in enumerate(_LONG_PAYLOAD_HEADERS, 1):
            _apply_header_style(ws.cell(row=1, column=col, value=header))
        ws.column_dimensions["A"].width = 34
        ws.column_dimensions["B"].width = 28
        ws.column_dimensions["G"].width = 120

    schema_label = f"{_LONG_PAYLOAD_SCHEMA} v{_LONG_PAYLOAD_SCHEMA_VERSION}"
    for index, chunk in enumerate(chunks, 1):
        row = ws.max_row + 1
        for col, value in enumerate(
            (schema_label, payload_id, index, len(chunks), len(text), digest),
            1,
        ):
            ws.cell(row=row, column=col, value=value)
        cell = ws.cell(row=row, column=len(_LONG_PAYLOAD_HEADERS), value=chunk)
        # A chunk boundary can land before "=", "+" or "@". Forcing the string
        # type keeps the character verbatim instead of letting the spreadsheet
        # read it as a formula, which would break byte-exact reassembly.
        cell.data_type = "s"

    return _long_payload_pointer(payload_id, len(chunks), len(text), digest)


def reassemble_long_payload(wb: Workbook, cell_value: Optional[str]) -> Optional[str]:
    """Return the full payload behind a cell written by :func:`_store_long_payload`.

    A cell that is not a pointer is returned unchanged, so a caller can apply
    this to every canonical-payload cell without first testing its length.
    Raises :class:`ValueError` when the chunks are missing, incomplete or do
    not hash to the digest recorded in the pointer.
    """
    if not isinstance(cell_value, str) or not cell_value.startswith(
        _LONG_PAYLOAD_POINTER_PREFIX
    ):
        return cell_value

    fields = {}
    for part in cell_value.split(" | ")[1:]:
        key, _, value = part.partition("=")
        fields[key.strip()] = value.strip()
    payload_id = fields.get("payload_id", "")
    expected_chunks = int(fields.get("chunks", "0"))
    expected_chars = int(fields.get("characters", "0"))
    expected_digest = fields.get("sha256", "")

    if _LONG_PAYLOAD_SHEET not in wb.sheetnames:
        raise ValueError(f"Overflow sheet {_LONG_PAYLOAD_SHEET!r} is missing.")
    ws = wb[_LONG_PAYLOAD_SHEET]

    parts: Dict[int, str] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[1] != payload_id:
            continue
        parts[int(row[2])] = row[6] or ""

    if sorted(parts) != list(range(1, expected_chunks + 1)):
        raise ValueError(
            f"Payload {payload_id!r} is incomplete: expected {expected_chunks} "
            f"chunk(s), found {sorted(parts)}."
        )
    text = "".join(parts[index] for index in range(1, expected_chunks + 1))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if len(text) != expected_chars or digest != expected_digest:
        raise ValueError(
            f"Payload {payload_id!r} failed verification: "
            f"{len(text)} characters / {digest} does not match the recorded "
            f"{expected_chars} characters / {expected_digest}."
        )
    return text


def _create_canonical_provenance_sheet(
    wb: Workbook, provenance: Mapping[str, object]
) -> None:
    """Store the exact canonical object used by JSON, CSV, and HDF5."""
    ws = wb.create_sheet("Provenance")
    ws.append(["Schema", "Canonical JSON"])
    payload = normalize_provenance(provenance)
    # normalize_provenance() deliberately passes an unversioned caller-supplied
    # mapping through unchanged, for read compatibility with pre-1.0 metadata.
    # Such a payload has no schema keys, so they are read defensively and the
    # label states what the record actually is rather than borrowing the
    # canonical schema's name and version.
    schema_name = payload.get("schema_name")
    schema_version = payload.get("schema_version")
    if schema_name and schema_version:
        schema_label = f"{schema_name} v{schema_version}"
    else:
        schema_label = "(unversioned caller-supplied provenance metadata)"
    ws.append([
        schema_label,
        _store_long_payload(wb, "provenance", canonical_provenance_json(payload)),
    ])
    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 120
    _write_provenance_field_summary(ws, payload, schema_label)


def _write_provenance_field_summary(
    ws, payload: Mapping[str, object], schema_label: str
) -> None:
    """Readable field/value digest beside the canonical cell (U62).

    Columns A:B keep the exact canonical object, chunked through the long
    payload sheet when it does not fit a cell; nothing here replaces it. The
    summary reads only keys that are present, so an unversioned caller-supplied
    mapping gets a shorter summary rather than invented fields.
    """
    rows: List[Tuple[str, object]] = [("Schema", schema_label)]

    def add(label: str, value: object) -> None:
        if value is None or value == "":
            return
        rows.append((label, value))

    add("Created (UTC)", payload.get("created_utc"))
    add("Software version", payload.get("software_version"))
    add("Software commit", payload.get("software_commit"))
    if "software_dirty" in payload:
        dirty = payload.get("software_dirty")
        add("Uncommitted changes", "Yes" if dirty is True else "No" if dirty is False else "Unknown")
    add("Source file", payload.get("source_filename"))
    add("Input SHA-256", payload.get("input_sha256"))
    add("Effective configuration SHA-256", payload.get("effective_configuration_sha256"))
    if "crm_record_ids" in payload:
        add("CRM record IDs", ", ".join(str(i) for i in payload.get("crm_record_ids") or []) or "(none)")
    rejection = payload.get("rejection_settings")
    if isinstance(rejection, Mapping):
        for key, value in sorted(rejection.items()):
            add(f"Rejection: {key}", value if isinstance(value, (int, float, str)) else json.dumps(value))
    windows = payload.get("cycle_windows")
    if isinstance(windows, Mapping):
        add("Cycle windows set", len(windows))
    rng = payload.get("rng")
    if isinstance(rng, Mapping):
        add("RNG", ", ".join(f"{key}={value}" for key, value in sorted(rng.items())))
    dependencies = payload.get("dependencies")
    if isinstance(dependencies, Mapping):
        add("Dependencies", ", ".join(f"{key} {value}" for key, value in sorted(dependencies.items())))
    if "warnings" in payload:
        add("Warnings recorded", len(payload.get("warnings") or []))
    rows.append(("Authority", "Column B holds the canonical JSON; this summary is for reading only."))

    _apply_header_style(ws.cell(row=1, column=4, value="Field"))
    _apply_header_style(ws.cell(row=1, column=5, value="Value"))
    for index, (label, value) in enumerate(rows, 2):
        _apply_data_style(ws.cell(row=index, column=4, value=literal_excel_text(label)))
        cell_value = literal_excel_text(value) if isinstance(value, str) else value
        _apply_data_style(
            ws.cell(row=index, column=5, value=cell_value),
            kind="wrap" if isinstance(cell_value, str) else "general",
        )
    ws.column_dimensions["D"].width = 32
    ws.column_dimensions["E"].width = 70
    ws.freeze_panes = "A2"
    _apply_print_settings(ws, landscape=True)


def _create_uncertainty_scope_sheet(wb: Workbook) -> None:
    """Store the exact versioned scope object used by all four exporters."""
    ws = wb.create_sheet("Uncertainty Scope")
    ws.append(["Schema", "Canonical JSON"])
    payload = uncertainty_scope_payload()
    ws.append([
        f"{payload['schema_name']} v{payload['schema_version']}",
        literal_excel_text(canonical_uncertainty_scope_json()),
    ])
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 120


def _create_cover_summary_sheet(
    wb: Workbook,
    result: ProcessingResult,
    config: Optional[object],
    loaded_filename: Optional[str],
    *,
    uncertainty_config: Optional[object] = None,
    include_uncertainty: bool = True,
    layout: str = EXCEL_LAYOUT_DIAGNOSTIC,
    profile_scope: Optional[str] = None,
) -> None:
    """Sheet 1 — Summary.

    Consequential warnings and reporting limitations follow the identifying
    header directly (U53); counts, reference material and detailed settings
    come after them.
    """
    ws = wb.active
    ws.title = "Summary"
    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 40
    ws.column_dimensions["C"].width = 22

    row = 1
    ws.cell(row=row, column=1, value="TraceISO — Export Report").font = _FONT_BOLD_TITLE
    row += 2

    _kv(ws, row, "App Version", f"TraceISO {APP_VERSION}"); row += 1
    _kv(ws, row, "Export Date", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")); row += 1
    _kv(ws, row, "Input File", literal_excel_text(loaded_filename or "(unknown)")); row += 1
    _kv(ws, row, "Element", result.element_symbol or "Unknown"); row += 1
    _kv(ws, row, "Workbook Layout", layout); row += 1
    row += 1

    crm_rows, crm_warnings = _get_crm_info(config, result.element_symbol or "")
    row = _write_warnings_and_limitations(
        ws, row, result, crm_warnings, include_uncertainty=include_uncertainty
    )
    row += 1

    _section(ws, row, "Sample Counts"); row += 1
    counts = result.count_by_type
    for stype, n in sorted(counts.items()):
        _kv(ws, row, f"  {stype}", n); row += 1
    _kv(ws, row, "  Total", len(result.samples)); row += 2

    # CRM certified values
    if crm_rows:
        _section(ws, row, "CRM / Reference Material"); row += 1
        for lbl, col in [("CRM Name", 1), ("Certified Ratio", 2), ("Uncertainty (k=1)", 3)]:
            c = ws.cell(row=row, column=col, value=lbl)
            c.font = _FONT_BOLD
        row += 1
        for crm_name, ratio_name, ratio_val, unc in crm_rows:
            ws.cell(row=row, column=1, value=literal_excel_text(crm_name))
            ws.cell(row=row, column=2, value=literal_excel_text(f"{ratio_name} = {ratio_val:.8g}"))
            ws.cell(row=row, column=3, value=unc)
            row += 1
        row += 1

    # Processing settings
    if config is not None:
        _section(ws, row, "Processing Settings"); row += 1
        for label, value in _extract_config_params(
            config, result.element_symbol or "", include_uncertainty=include_uncertainty
        ):
            _kv(ws, row, f"  {label}", value); row += 1
        row += 1

    # Uncertainty provenance (omitted entirely when uncertainty export is disabled)
    if include_uncertainty:
        row = _write_uncertainty_provenance(
            ws, row, result, uncertainty_config, profile_scope=profile_scope
        )
        row += 1
    _apply_print_settings(ws, landscape=False, title_rows=None, fit_width=True)


def _write_warnings_and_limitations(
    ws,
    start_row: int,
    result: ProcessingResult,
    crm_warnings: List[str],
    *,
    include_uncertainty: bool,
) -> int:
    """Write processing/CRM warnings and reporting limitations; return the next row.

    Only statements that already exist are gathered here: pipeline and CRM
    warnings, budgets whose scope makes them unreportable, and the canonical
    cross-ratio scope note. With uncertainty export disabled no uncertainty
    statement is written at all.
    """
    row = start_row
    _section(ws, row, "Warnings and Reporting Limitations"); row += 1
    warnings = [str(w) for w in result.warnings] + [str(w) for w in crm_warnings]
    lines: List[Tuple[str, bool]] = [(text, True) for text in warnings]
    if not warnings:
        lines.append(("No processing warnings recorded.", False))
    if include_uncertainty:
        unreportable: Counter = Counter()
        for sample in result.samples:
            for budget in (sample.uncertainty or {}).values():
                if budget is not None and is_invalid_budget_scope(budget):
                    unreportable[budget_scope_label(budget)] += 1
        for label, count in sorted(unreportable.items()):
            lines.append((
                f"{count} ratio budget(s) cannot be reported ({label}); "
                "see Budget State and Budget Note.",
                True,
            ))
        if _has_uncertainty(result):
            lines.append((CROSS_RATIO_INDEPENDENCE_NOTE, False))
    for text, consequential in lines:
        cell = ws.cell(row=row, column=1, value=literal_excel_text(text))
        if consequential:
            cell.font = Font(color="CC0000")
        row += 1
    return row


def _write_uncertainty_provenance(
    ws,
    start_row: int,
    result: ProcessingResult,
    uncertainty_config: Optional[object],
    *,
    profile_scope: Optional[str] = None,
) -> int:
    """Write an Uncertainty Provenance section into the Summary sheet."""
    # Detect engine / output_mode from first available budget
    engine = ""
    output_mode = ""
    for sample in result.samples:
        for budget in sample.uncertainty.values():
            engine = engine or budget.engine
            output_mode = output_mode or budget.output_mode
            if engine and output_mode:
                break
        if engine and output_mode:
            break

    if not engine and uncertainty_config is None:
        return start_row  # nothing to write

    row = start_row
    _section(ws, row, "Uncertainty Provenance"); row += 1
    _kv(ws, row, "  Framework Version", UNCERTAINTY_FRAMEWORK_VERSION); row += 1
    _kv(ws, row, "  Engine", engine or "(none)"); row += 1
    _kv(ws, row, "  Output Mode", output_mode or "(none)"); row += 1

    coverage_method = getattr(uncertainty_config, "coverage_method", None) if uncertainty_config else None
    _kv(ws, row, "  Coverage Method", coverage_method or "welch_satterthwaite"); row += 1
    coverage_k = getattr(uncertainty_config, "coverage_k", None) if uncertainty_config else None
    if coverage_k is not None:
        _kv(ws, row, "  Fixed k (if applicable)", coverage_k); row += 1
    if profile_scope is not None:
        scope_text = (
            "Profiles used by the exported observations"
            if profile_scope == PROFILE_SCOPE_USED
            else "Complete contributor profile library (audit)"
        )
        _kv(ws, row, "  Contributor Profiles Sheet", scope_text); row += 1
    return row


def _create_results_final_sheet(
    wb: Workbook,
    result: ProcessingResult,
    include_raw: bool,
    include_corrected: bool,
    include_uncertainty: bool,
    *,
    corrected_label: str = "Corrected",
    ratio_names: Optional[List[str]] = None,
) -> None:
    """Sheet 2 — Results: explicit ratio+layer columns."""
    ws = wb.create_sheet("Results")

    ratio_names = ratio_names if ratio_names is not None else _get_ratio_names(result)
    has_raw = include_raw
    has_blank_corrected = include_corrected and _has_blank_corrected(
        result, dedupe_against_raw=include_raw
    )
    has_corrected = include_corrected and _has_corrected(
        result, dedupe_against_raw=include_raw
    )
    has_ssb = include_corrected and _has_ssb(result)
    has_drift = include_corrected and _has_drift(result)
    has_delta = include_corrected and _has_delta(result)
    has_calibrated_delta = include_corrected and any(
        bool(getattr(sample, "pb_calibrated_delta_cycles", None))
        for sample in result.samples
    )
    has_unc = include_uncertainty and _has_uncertainty(result)

    fixed_headers = ["Sample", "Type", "Run Number", "Total cycles"]
    col = 1
    for label in fixed_headers:
        cell = ws.cell(row=1, column=col, value=label)
        _apply_header_style(cell)
        col += 1

    data_columns: List[Tuple[str, str]] = []
    for ratio_name in ratio_names:
        ratio_label = format_isotope_label(ratio_name)
        data_columns.append((ratio_name, f"Accepted cycles {ratio_label}"))
        if has_raw:
            data_columns += [
                (ratio_name, f"Raw {ratio_label}"),
                (ratio_name, f"Raw {ratio_label} SD"),
                (ratio_name, f"Raw {ratio_label} SE"),
                (ratio_name, f"Raw {ratio_label} 2SE"),
                (ratio_name, f"Raw {ratio_label} RSD%"),
            ]
        if has_blank_corrected:
            data_columns += [
                (ratio_name, f"Blank-corrected {ratio_label}"),
                (ratio_name, f"Blank-corrected {ratio_label} SD"),
                (ratio_name, f"Blank-corrected {ratio_label} SE"),
                (ratio_name, f"Blank-corrected {ratio_label} 2SE"),
                (ratio_name, f"Blank-corrected {ratio_label} RSD%"),
            ]
        if has_corrected:
            data_columns += [
                (ratio_name, f"{corrected_label} {ratio_label}"),
                (ratio_name, f"{corrected_label} {ratio_label} SD"),
                (ratio_name, f"{corrected_label} {ratio_label} SE"),
                (ratio_name, f"{corrected_label} {ratio_label} 2SE"),
                (ratio_name, f"{corrected_label} {ratio_label} RSD%"),
            ]
        if has_ssb:
            data_columns += [
                (ratio_name, f"SSB {ratio_label}"),
                (ratio_name, f"SSB {ratio_label} k"),
            ]
        if has_drift:
            data_columns += [
                (ratio_name, f"Drift corrected {ratio_label}"),
                (ratio_name, f"Drift corrected {ratio_label} SD"),
                (ratio_name, f"Drift corrected {ratio_label} SE"),
                (ratio_name, f"Drift corrected {ratio_label} 2SE"),
            ]
        if has_delta:
            data_columns += [
                (ratio_name, f"Delta {ratio_label} (‰)"),
                (ratio_name, f"Delta {ratio_label} SD"),
                (ratio_name, f"Delta {ratio_label} SE"),
                (ratio_name, f"Delta {ratio_label} 2SE"),
            ]
        if has_calibrated_delta:
            data_columns += [
                (ratio_name, f"Calibrated delta {ratio_label} (‰)"),
                (ratio_name, f"Calibrated delta precision statistic {ratio_label}"),
                (ratio_name, f"Calibrated delta precision {ratio_label} (‰)"),
                (ratio_name, f"Calibrated delta combined uncertainty {ratio_label}"),
            ]
        if has_unc:
            data_columns += [
                (ratio_name, f"Reported value {ratio_label}"),
                (ratio_name, f"Output mode {ratio_label}"),
                (ratio_name, f"Unit {ratio_label}"),
                (ratio_name, f"Source layer {ratio_label}"),
                (ratio_name, f"Reference value {ratio_label}"),
                # U is in the adjacent Unit (delta permil or ratio); the two
                # relative columns are permil of the ratio, a different quantity.
                (ratio_name, f"U {ratio_label}"),
                (ratio_name, f"Relative u_c (\u2030) {ratio_label}"),
                (ratio_name, f"Relative U (\u2030) {ratio_label}"),
                (ratio_name, f"k {ratio_label}"),
                (ratio_name, f"Dominant {ratio_label}"),
                (ratio_name, f"Budget Note {ratio_label}"),
                (ratio_name, f"Budget State {ratio_label}"),
            ]

    for _, header in data_columns:
        cell = ws.cell(row=1, column=col, value=header)
        _apply_header_style(cell)
        col += 1

    metric_count_by_ratio = len(data_columns) // max(1, len(ratio_names)) if ratio_names else 0


    # Data rows (start at row 2)
    for r_row, sample in enumerate(result.samples, 2):
        col = 1
        ws.cell(row=r_row, column=col, value=literal_excel_text(sample.name));         col += 1
        ws.cell(row=r_row, column=col, value=literal_excel_text(sample.sample_type));  col += 1
        ws.cell(row=r_row, column=col, value=sample.run_number);   col += 1
        ws.cell(row=r_row, column=col, value=sample.n_cycles);     col += 1

        alt = r_row % 2 == 0
        for c in range(1, 5):
            _apply_data_style(ws.cell(row=r_row, column=c), alt)

        if metric_count_by_ratio > 0:
            for rname in ratio_names:
                selected_report_layer = select_summary_export_layer(sample, rname)
                row_data: List[object] = [
                    selected_report_layer.data.n_valid
                    if selected_report_layer is not None
                    else None
                ]
                if has_raw:
                    raw_vals = _cd_stats(sample.ratios.get(rname))
                    row_data += [
                        raw_vals.get("mean"), raw_vals.get("sd"), raw_vals.get("se"),
                        raw_vals.get("2se"), raw_vals.get("rsd"),
                    ]
                if has_blank_corrected:
                    blank_vals = _cd_stats(sample.blank_corrected_ratios.get(rname))
                    row_data += [
                        blank_vals.get("mean"), blank_vals.get("sd"), blank_vals.get("se"),
                        blank_vals.get("2se"), blank_vals.get("rsd"),
                    ]
                if has_corrected:
                    corr_vals = _cd_stats(
                        _get_best_corrected(
                            sample, rname, dedupe_against_raw=include_raw
                        )
                    )
                    row_data += [
                        corr_vals.get("mean"), corr_vals.get("sd"), corr_vals.get("se"),
                        corr_vals.get("2se"), corr_vals.get("rsd"),
                    ]
                if has_ssb:
                    row_data += [_get_ssb_mean(sample, rname), _get_ssb_k(sample, rname)]
                if has_drift:
                    drift_vals = _cd_stats(sample.drift_corrected_ratios.get(rname))
                    row_data += [drift_vals.get("mean"), drift_vals.get("sd"),
                                 drift_vals.get("se"), drift_vals.get("2se")]
                if has_delta:
                    d = _get_delta_stats(sample, rname) or {}
                    row_data += [d.get("delta"), d.get("delta_sd"), d.get("delta_se"), d.get("delta_2se")]
                if has_calibrated_delta:
                    from domain.pb_calibration_records import calibrated_delta_record

                    record = calibrated_delta_record(sample, rname)
                    cycles = (getattr(sample, "pb_calibrated_delta_cycles", None) or {}).get(rname)
                    delta_stats = _cd_stats(cycles)
                    if record is not None and record.status == "applied":
                        precision = (
                            delta_stats.get(record.precision_statistic)
                            if record.precision_statistic in {"sd", "se"}
                            else None
                        )
                        row_data += [
                            delta_stats.get("mean"),
                            record.precision_label or "None",
                            precision,
                            f"Not calculated ({record.uncertainty_reason_code})",
                        ]
                    elif record is not None:
                        row_data += [
                            None,
                            record.precision_label or "None",
                            None,
                            f"Not calculated ({record.reason_code or record.uncertainty_reason_code})",
                        ]
                    else:
                        row_data += [None, None, None, None]
                if has_unc:
                    budget = sample.uncertainty.get(rname)
                    invalid_scope = bool(
                        budget is not None and is_invalid_budget_scope(budget)
                    )
                    if budget is not None and not invalid_scope:
                        row_data += list(
                            _reported_measurand_context(
                                sample, rname, budget, selected_report_layer
                            )
                        )
                        row_data += [report_uncertainty(budget.expanded_abs)]
                        row_data += [report_uncertainty(budget.u_combined_rel_permil)]
                        row_data += [report_uncertainty(budget.expanded_rel_permil)]
                        row_data += [budget.coverage_factor_k]
                        dom = budget.dominant_contributor
                        dom_text = format_contributor_name(dom)
                        note = build_budget_note(budget)
                        state_text = summarize_budget_state(budget)
                    elif invalid_scope:
                        row_data += [None, budget.output_mode or "", None, None, None]
                        row_data += [None, None, None, None]
                        dom_text = None
                        note = budget_scope_note(budget)
                        state_text = budget_scope_label(budget)
                    else:
                        row_data += [None, None, None, None, None]
                        row_data += [None, None, None, None]
                        dom_text = None
                        note = None
                        state_text = None
                    row_data += [dom_text, note, state_text]

                for s_idx, val in enumerate(row_data):
                    if isinstance(val, str):
                        cell_val = val
                    elif isinstance(val, (int, np.integer)) and not isinstance(val, (bool, np.bool_)):
                        # Accepted-cycle counts stay integers; as floats they
                        # picked up the 0.000000 style and read "20.000000".
                        cell_val = int(val)
                    else:
                        cell_val = safe_float(val)
                    cell = ws.cell(row=r_row, column=col + s_idx, value=cell_val)
                    _apply_data_style(cell, alt)
                if has_unc:
                    # The uncertainty block is the last 12 cells of each ratio
                    # block: value, output, unit, layer, reference, U,
                    # relative u_c, relative U, k, dominant, note, state.
                    block_end = col + metric_count_by_ratio
                    _coordinate_value_with_uncertainty(
                        ws.cell(row=r_row, column=block_end - 12),
                        ws.cell(row=r_row, column=block_end - 7),
                    )
                    for offset in (6, 5):
                        _format_as(ws.cell(row=r_row, column=block_end - offset), "reported_uncertainty")
                    _format_as(ws.cell(row=r_row, column=block_end - 4), "k")
                col += metric_count_by_ratio

    # Column widths + freeze
    identity_col = len(fixed_headers) + len(data_columns) + 1
    _apply_header_style(ws.cell(row=1, column=identity_col, value="Observation ID"))
    for row, sample in enumerate(result.samples, 2):
        _apply_data_style(
            ws.cell(row=row, column=identity_col, value=literal_excel_text(sample.observation_id)),
            row % 2 == 0,
        )
    ws.column_dimensions[get_column_letter(identity_col)].width = 40
    ws.column_dimensions["A"].width = 24
    for i in range(2, len(fixed_headers) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 11
    for i in range(len(fixed_headers) + 1, len(fixed_headers) + len(data_columns) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 16
    # Sample, Type and Run Number stay visible while scrolling across ratios (U57).
    ws.freeze_panes = "D2"
    ws.row_dimensions[1].height = 45
    _apply_table_filter(ws, last_row=len(result.samples) + 1, last_col=identity_col)
    _apply_print_settings(ws, landscape=True, title_cols="A:C")


def _output_label(output_mode: Optional[str]) -> Optional[str]:
    if not output_mode:
        return None
    return "Delta" if output_mode == "delta" else "Absolute ratio"


def _reported_measurand_context(
    sample: Sample,
    ratio_name: str,
    budget: UncertaintyBudget,
    selected_report_layer,
) -> Tuple[object, str, str, str, object]:
    """Return (reported value, output mode, unit, source layer, reference) for a valid budget.

    The value is the measurand the budget's U belongs to: delta in permil for
    delta output, never the ratio kept in ``ratio_value``. Results and Reported
    Results both call this, so the pairing cannot drift between the two sheets.
    """
    if (budget.output_mode or "absolute_ratio") == "delta":
        source_basis = (
            sample.delta_results.get(ratio_name, {}).get("source_layer", "")
            if sample.delta_results
            else ""
        )
        source_layer = f"delta ({source_basis})" if source_basis else "delta"
        return (
            budget.reported_measurand_value(), "Delta", "‰",
            source_layer, budget.delta_reference_value,
        )
    source_layer = selected_report_layer.key if selected_report_layer is not None else ""
    return (
        budget.reported_measurand_value(), "Absolute ratio", "ratio",
        source_layer, budget.certified_reference_value,
    )


_REPORTED_RESULTS_COLUMNS: List[Tuple[str, str, int]] = [
    # (header, cell role, width). Identity -> measurement -> uncertainty ->
    # review context, the same priority as the on-screen summary table (U51).
    ("Sample", "text", 24),
    ("Type", "text", 9),
    ("Run Number", "count", 12),
    ("Ratio", "text", 13),
    ("Output mode", "text", 14),
    ("Reported value", "value", 15),
    ("Unit", "text", 7),
    ("U", "reported_uncertainty", 12),
    ("Coverage k", "k", 9),
    ("u_c", "reported_uncertainty", 12),
    ("Relative U (‰)", "reported_uncertainty", 12),
    ("Accepted cycles", "count", 10),
    ("Source layer", "text", 22),
    ("Reference value", "value", 15),
    ("Budget State", "text", 30),
    ("Dominant contributor", "wrap", 28),
    ("Budget Note", "wrap", 50),
    ("Observation ID", "text", 40),
]


def _create_reported_results_sheet(
    wb: Workbook,
    result: ProcessingResult,
    *,
    include_uncertainty: bool,
    ratio_names: Optional[List[str]] = None,
) -> None:
    """Reported Results — the analyst report's first results table (U52).

    One row per observation x ratio x output mode for every standard, sample
    and QC observation with data for the ratio. Blanks are correction inputs,
    not reported measurands, and remain on the diagnostic sheets. A row whose
    budget is missing or out of scope is still written, with its value and U
    left empty and Budget State saying why: a reported value here is only ever
    the measurand its U belongs to. Values and layers come from the same
    selectors as the Results sheet; nothing is recomputed.
    """
    ws = wb.create_sheet(REPORTED_RESULTS_SHEET)
    ratio_names = ratio_names if ratio_names is not None else _get_ratio_names(result)
    for c_idx, (header, _role, width) in enumerate(_REPORTED_RESULTS_COLUMNS, 1):
        _apply_header_style(ws.cell(row=1, column=c_idx, value=header))
        ws.column_dimensions[get_column_letter(c_idx)].width = width
    value_col = 1 + [h for h, _r, _w in _REPORTED_RESULTS_COLUMNS].index("Reported value")
    u_col = 1 + [h for h, _r, _w in _REPORTED_RESULTS_COLUMNS].index("U")

    row = 2
    for sample in result.samples:
        if sample.is_blank:
            continue
        for rname in ratio_names:
            layer = select_summary_export_layer(sample, rname)
            budget = sample.uncertainty.get(rname) if include_uncertainty else None
            from domain.sr_standard_calibration import sr_calibration_message
            sr_refusal = sr_calibration_message(sample, rname)
            if layer is None and budget is None and not sr_refusal:
                continue
            value = output = unit = source = reference = None
            u_expanded = k = u_combined = relative_u = dominant = note = None
            if sr_refusal:
                state = "Sr calibration unavailable"
                note = sr_refusal
            elif budget is None:
                state = (
                    "No uncertainty budget for this ratio"
                    if include_uncertainty
                    else "Uncertainty not included in this export"
                )
            elif is_invalid_budget_scope(budget):
                output = _output_label(budget.output_mode)
                state = budget_scope_label(budget)
                note = budget_scope_note(budget)
            else:
                value, output, unit, source, reference = _reported_measurand_context(
                    sample, rname, budget, layer
                )
                u_expanded = report_uncertainty(budget.expanded_abs)
                k = budget.coverage_factor_k
                u_combined = report_uncertainty(budget.u_combined_abs)
                relative_u = report_uncertainty(budget.expanded_rel_permil)
                dominant = format_contributor_name(budget.dominant_contributor)
                state = summarize_budget_state(budget)
                note = build_budget_note(budget)
            values = [
                sample.name, sample.sample_type, sample.run_number, rname, output,
                safe_float(value), unit, safe_float(u_expanded), safe_float(k),
                safe_float(u_combined), safe_float(relative_u),
                int(layer.data.n_valid) if layer is not None else None,
                source, safe_float(reference), state, dominant, note,
                sample.observation_id,
            ]
            alt = row % 2 == 0
            for c_idx, (cell_value, (_h, role, _w)) in enumerate(
                zip(values, _REPORTED_RESULTS_COLUMNS), 1
            ):
                if isinstance(cell_value, str):
                    cell_value = literal_excel_text(cell_value)
                _apply_data_style(ws.cell(row=row, column=c_idx, value=cell_value), alt, role)
            _coordinate_value_with_uncertainty(
                ws.cell(row=row, column=value_col), ws.cell(row=row, column=u_col)
            )
            row += 1

    ws.freeze_panes = "E2"
    ws.row_dimensions[1].height = 32
    _apply_table_filter(ws, last_row=row - 1, last_col=len(_REPORTED_RESULTS_COLUMNS))
    _apply_print_settings(ws, landscape=True, title_cols="A:D")


def _contributor_status(contributor: UncertaintyContributor) -> str:
    """Map UncertaintyContributor.state to a human-readable export string."""
    state = getattr(contributor, "state", "") or ""
    if state == "BY_SAMPLE_DESIGN":
        return "BY SAMPLE DESIGN"
    if state == "BY_GLOBAL_DESIGN":
        return "BY GLOBAL DESIGN"
    if state == "NOT_APPLICABLE":
        return "NOT APPLICABLE"
    if state == "MISSING_DATA":
        return "MISSING DATA"
    if getattr(contributor, "is_active", False):
        return "ACTIVE"
    return "MISSING DATA"


def _create_uncertainty_budgets_sheet(
    wb: Workbook,
    result: ProcessingResult,
    *,
    uncertainty_config: Optional[object] = None,
    ratio_names: Optional[List[str]] = None,
) -> None:
    """Uncertainty_Budgets sheet — one row per sample x ratio x contributor."""
    ws = wb.create_sheet("Uncertainty_Budgets")

    headers = [
        "Sample", "Type", "Run Number", "Ratio",
        "Contributor", "Display Name", "u (ratio units)", "Relative u (\u2030)",
        "Type A/B", "DoF", "Variance Share %", "Active",
        "State", "Inactive Reason", "Reference",
        # Summary columns \u2014 full precision floats (not GUM-rounded; use CSV export for rounded values).
        # u_c/U are in the budget's output unit (delta permil for delta output);
        # the relative columns are permil of the ratio; contributor u is in ratio units.
        "u_c (output unit, full prec)", "Relative u_c (\u2030, full prec)",
        "U (output unit, full prec)", "Relative U (\u2030, full prec)",
        "k", "Eff. DoF", "Dominant", "Engine", "Output Mode",
        "Report Ratio", "Basis Ratio", "Delta Reference", "Certified Reference",
        "Absolute Scale", "Delta Scale", "Notes",
    ]
    roles = [
        "text", "text", "count", "text", "text", "text", "uncertainty", "uncertainty",
        "text", "dof", "percent", "text", "text", "wrap", "wrap",
        "value", "value", "value", "value", "k", "dof", "text", "text", "text",
        "value", "value", "value", "value", "value", "value", "text",
    ]
    for c_idx, h in enumerate(headers, 1):
        _apply_header_style(ws.cell(row=1, column=c_idx, value=h))

    ratio_names = ratio_names if ratio_names is not None else _get_ratio_names(result)
    row = 2

    for sample in result.samples:
        for ratio_name in ratio_names:
            budget = sample.uncertainty.get(ratio_name)
            if budget is None:
                continue

            invalid_scope = is_invalid_budget_scope(budget)
            note = budget_scope_note(budget) if invalid_scope else build_budget_note(budget)
            if invalid_scope:
                summary_vals = [
                    None, None, None, None, None, None, "",
                    budget.engine or "",
                    budget.output_mode or "",
                    None, None, None, None, None, None,
                    note,
                ]
            else:
                summary_vals = [
                    safe_float(budget.u_combined_abs),
                    safe_float(budget.u_combined_rel_permil),
                    safe_float(budget.expanded_abs),
                    safe_float(budget.expanded_rel_permil),
                    safe_float(budget.coverage_factor_k),
                    safe_float(budget.effective_dof),
                    format_contributor_name(budget.dominant_contributor or ""),
                    budget.engine or "",
                    budget.output_mode or "",
                    safe_float(getattr(budget, "ratio_value", None)),
                    safe_float(getattr(budget, "basis_ratio_value", None)),
                    safe_float(getattr(budget, "delta_reference_value", None)),
                    safe_float(getattr(budget, "certified_reference_value", None)),
                    safe_float(getattr(budget, "absolute_scale_factor", None)),
                    safe_float(getattr(budget, "delta_scale_factor", None)),
                    note,
                ]

            contributors = budget.contributors if budget.contributors else []
            if not contributors:
                # Write a single summary row even when no contributors list
                sample_vals = [
                    literal_excel_text(sample.name),
                    literal_excel_text(sample.sample_type),
                    sample.run_number,
                    literal_excel_text(ratio_name),
                    "", "", None, None, "", None, None, "",
                    budget_scope_label(budget) if invalid_scope else "",
                    note if invalid_scope else "", "",
                ]
                all_vals = sample_vals + summary_vals
                alt = row % 2 == 0
                for c_idx, val in enumerate(all_vals, 1):
                    cell = ws.cell(row=row, column=c_idx, value=val)
                    _apply_data_style(cell, alt, roles[c_idx - 1])
                row += 1
                continue

            for c_i, contrib in enumerate(contributors):
                alt = row % 2 == 0
                # Disabled contributors do not enter the budget, so their
                # numeric cells are left blank: showing a computed u/DoF next to
                # "Active = No" wrongly implies the term contributed. The row is
                # retained as an audit record (Active/State/Inactive Reason).
                is_active = contrib.is_active and not invalid_scope
                dof_value = (
                    (safe_float(contrib.degrees_of_freedom)
                     if contrib.degrees_of_freedom != float('inf') else "\u221e")
                    if is_active else None
                )
                sample_vals = [
                    literal_excel_text(sample.name),
                    literal_excel_text(sample.sample_type),
                    sample.run_number,
                    literal_excel_text(ratio_name),
                    literal_excel_text(contrib.name),
                    literal_excel_text(contrib.display_name),
                    safe_float(contrib.value_abs) if is_active else None,
                    safe_float(contrib.value_rel_permil) if is_active else None,
                    contrib.type_ab,
                    dof_value,
                    safe_float(contrib.percentage_contribution) if is_active else None,
                    "Yes" if is_active else "No",
                    budget_scope_label(budget) if invalid_scope else _contributor_status(contrib),
                    note if invalid_scope else (getattr(contrib, "inactive_reason", "") or ""),
                    literal_excel_text(getattr(contrib, "reference", "") or ""),
                ]
                # Budget totals repeat on every contributor row, so a filtered
                # view (U58) never separates a contributor from its budget.
                all_vals = sample_vals + summary_vals
                for c_idx, val in enumerate(all_vals, 1):
                    cell = ws.cell(row=row, column=c_idx, value=val)
                    _apply_data_style(cell, alt, roles[c_idx - 1])
                row += 1

    # Column widths (A–O: contributor fields; P–AE: budget summary)
    widths = {
        "A": 22, "B": 9, "C": 12, "D": 13, "E": 22, "F": 26,
        "G": 13, "H": 13, "I": 9, "J": 8, "K": 10, "L": 8,
        "M": 18, "N": 30, "O": 36,
        "P": 14, "Q": 14, "R": 14, "S": 14, "T": 8, "U": 10,
        "V": 22, "W": 14, "X": 12, "Y": 13, "Z": 13, "AA": 13,
        "AB": 13, "AC": 13, "AD": 13, "AE": 40,
    }
    for col_letter, w in widths.items():
        ws.column_dimensions[col_letter].width = w
    # Sample, Type, Run Number and Ratio stay visible while scrolling (U57).
    ws.freeze_panes = "E2"
    ws.row_dimensions[1].height = 45
    _apply_table_filter(ws, last_row=row - 1, last_col=len(headers))
    _apply_print_settings(ws, landscape=True, title_cols="A:D")


_WIDE_CONTRIBUTOR_HEADERS = {
    "u_prec": "Sample measurement\nprecision",
    "u_std": "Bracketing-standard\nprecision",
    "u_std_repeatability": "Standard\nrepeatability",
    "u_blank": "Blank\ncorrection",
    "u_k1_sample_decomposition": "Sample\ndecomposition (κ{1})",
    "u_k2_matrix_separation": "Matrix\nseparation (κ{2})",
    "u_k3_procedural_blank": "Procedural\nblank (κ{3})",
    "u_k4_bracketing_standard_heterogeneity": "Bracketing-standard\nheterogeneity (κ{4})",
    "u_k5_instrumental_drift": "Instrumental\ndrift (κ{5})",
    "u_k6_matrix_effects": "Matrix\neffects (κ{6})",
    "u_k7_residual_interferences": "Residual\ninterferences (κ{7})",
    "u_crm": "CRM certified\nvalue",
}
_WIDE_SUBSCRIPT = {
    "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇",
}


def _rich_kappa_header(text: str):
    """Return a rich κ header, with a deterministic Unicode fallback."""
    plain = text
    for number, subscript in _WIDE_SUBSCRIPT.items():
        plain = plain.replace(f"κ{{{number}}}", f"κ{subscript}")
    if "κ" not in text or CellRichText is None or TextBlock is None or InlineFont is None:
        return plain

    bold = InlineFont(b=True)
    bold_italic = InlineFont(b=True, i=True)
    blocks = []
    buffer = ""
    index = 0
    while index < len(text):
        if text[index] != "κ":
            buffer += text[index]
            index += 1
            continue
        if buffer:
            blocks.append(TextBlock(bold, buffer))
            buffer = ""
        blocks.append(TextBlock(bold_italic, "κ"))
        if index + 2 < len(text) and text[index + 1] == "{":
            end = text.find("}", index + 2)
            if end != -1:
                number = text[index + 2:end]
                buffer += _WIDE_SUBSCRIPT.get(number, number)
                index = end + 1
                continue
        index += 1
    if buffer:
        blocks.append(TextBlock(bold, buffer))
    return CellRichText(blocks)


def _wide_budget(sample: Sample, ratio_name: str) -> Optional[UncertaintyBudget]:
    budget = (sample.uncertainty or {}).get(ratio_name)
    if budget is None or (budget.engine or "") != "ssb_delta":
        return None
    return None if is_invalid_budget_scope(budget) else budget


def _wide_contributor_columns(rows) -> List[str]:
    present = {
        contributor.name
        for _sample, budget in rows
        for contributor in (budget.contributors or [])
        if contributor.is_active
    }
    ordered = [name for name in ENGINE_SSB_CONTRIBUTOR_NAMES if name in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _wide_contributor_header(name: str, rows) -> str:
    if name in _WIDE_CONTRIBUTOR_HEADERS:
        return _WIDE_CONTRIBUTOR_HEADERS[name]
    for _sample, budget in rows:
        for contributor in budget.contributors or []:
            if contributor.name == name and contributor.display_name:
                return literal_excel_text(contributor.display_name)
    return literal_excel_text(format_contributor_name(name))


def _finite_positive(value: object) -> Optional[float]:
    converted = safe_float(value)
    return converted if converted is not None and converted > 0.0 else None


def _wide_absolute_amount_ratio(budget: UncertaintyBudget) -> Optional[float]:
    """Return the displayed amount ratio without recomputing uncertainty."""
    if (budget.output_mode or "") == "absolute_ratio":
        return _finite_positive(budget.ratio_value)
    basis = _finite_positive(budget.basis_ratio_value)
    certified = _finite_positive(budget.certified_reference_value)
    delta_reference = _finite_positive(budget.delta_reference_value)
    if basis is None or certified is None or delta_reference is None:
        return None
    return basis * certified / delta_reference


def _wide_amount_ratio_header(ratio_name: str) -> str:
    if "/" not in ratio_name:
        return f"n({format_ratio_token(ratio_name)})"
    numerator, denominator = ratio_name.split("/", 1)
    return f"n({format_ratio_token(numerator)})/n({format_ratio_token(denominator)})"


def _wide_row_values(
    sample: Sample,
    ratio_name: str,
    budget: UncertaintyBudget,
    contributor_columns: List[str],
    *,
    delta_mode: bool,
) -> Dict[str, object]:
    ssb = (sample.ssb_results or {}).get(ratio_name, {}) or {}
    cycle_stats = _cd_stats(get_ssb_cycle_data(sample, ratio_name))
    previous_mean = safe_float(ssb.get("prev_std_mean"))
    next_mean = safe_float(ssb.get("next_std_mean"))
    mean_standard = (
        (previous_mean + next_mean) / 2.0
        if previous_mean is not None and next_mean is not None
        else None
    )
    amount_ratio = _wide_absolute_amount_ratio(budget)
    combined_relative = safe_float(budget.u_combined_rel_permil)
    expanded_relative = safe_float(budget.expanded_rel_permil)
    by_name = {contributor.name: contributor for contributor in budget.contributors or []}

    values: Dict[str, object] = {
        "sample": literal_excel_text(sample.name),
        "type": literal_excel_text(sample.sample_type),
        "run": sample.run_number,
        "ratio": safe_float(budget.basis_ratio_value),
        "ratio_se": cycle_stats.get("se"),
        "std1": previous_mean,
        "std1_se": safe_float(ssb.get("prev_std_se")),
        "std2": next_mean,
        "std2_se": safe_float(ssb.get("next_std_se")),
        "mean_std": mean_standard,
        "u_c": combined_relative,
        "U": expanded_relative,
        "k": safe_float(budget.coverage_factor_k),
        "nn": amount_ratio,
        "u_c_abs": (
            combined_relative / 1000.0 * amount_ratio
            if combined_relative is not None and amount_ratio is not None
            else None
        ),
        "U_abs": (
            expanded_relative / 1000.0 * amount_ratio
            if expanded_relative is not None and amount_ratio is not None
            else None
        ),
    }
    if delta_mode:
        values["delta"] = safe_float(
            ((sample.delta_results or {}).get(ratio_name, {}) or {}).get("delta")
        )
    for name in contributor_columns:
        contributor = by_name.get(name)
        values[f"contributor::{name}"] = (
            safe_float(contributor.value_rel_permil)
            if contributor is not None and contributor.is_active
            else None
        )
    return values


def _create_uncertainty_budget_wide_sheet(
    wb: Workbook,
    result: ProcessingResult,
    *,
    ratio_names: Optional[List[str]] = None,
) -> None:
    """Create one analyst-oriented per-sample Engine B budget worksheet."""
    samples = [sample for sample in result.samples if not sample.is_standard and not sample.is_blank]
    ratio_names = ratio_names if ratio_names is not None else _get_ratio_names(result)
    blocks = []
    for ratio_name in ratio_names:
        rows = [
            (sample, budget)
            for sample in samples
            if (budget := _wide_budget(sample, ratio_name)) is not None
        ]
        if rows:
            blocks.append((ratio_name, rows))
    if not blocks:
        return

    sheet_name = _build_unique_sheet_name(
        wb,
        "Budget_Detail",
        fallback="Budget_per_sample",
    )
    ws = wb.create_sheet(sheet_name)
    ws.sheet_view.showGridLines = False
    banner_fill = PatternFill("solid", fgColor="2E75B6")
    row_number = 1
    first_header_row = None
    max_columns = 0

    for ratio_name, rows in blocks:
        delta_mode = any((budget.output_mode or "") == "delta" for _sample, budget in rows)
        contributor_columns = _wide_contributor_columns(rows)
        ratio_label = format_isotope_label(ratio_name)
        # (key, header, cell role). Roles pick a magnitude-aware display format
        # per cell (U60); a small uncertainty is never shown as 0.0000.
        columns: List[Tuple[str, object, str]] = [
            ("sample", "Sample", "text"),
            ("type", "Type", "text"),
            ("run", "Run", "count"),
        ]
        if delta_mode:
            columns.append(("delta", "δ (‰)", "value"))
        # The sample column is the budget's basis ratio (SSB-corrected when the
        # sample was bracketed) while the standards are pre-SSB means, so each
        # heading names its layer and bracket role (U56: no repeated "SE").
        bracketed = any(
            ((sample.ssb_results or {}).get(ratio_name, {}) or {}).get("prev_std_mean") is not None
            for sample, _budget in rows
        )
        sample_layer = "SSB-corrected" if bracketed else "budget basis"
        columns.extend([
            ("ratio", f"{ratio_label} sample ({sample_layer})", "value"),
            ("ratio_se", "Sample SE", "uncertainty"),
            ("std1", f"{ratio_label} preceding standard (before SSB)", "value"),
            ("std1_se", "Preceding standard SE", "uncertainty"),
            ("std2", f"{ratio_label} following standard (before SSB)", "value"),
            ("std2_se", "Following standard SE", "uncertainty"),
            ("mean_std", "Mean standard (before SSB)", "value"),
        ])
        for name in contributor_columns:
            columns.append((
                f"contributor::{name}",
                _rich_kappa_header(_wide_contributor_header(name, rows)),
                "uncertainty",
            ))
        columns.extend([
            ("u_c", "Relative u_c (‰)", "uncertainty"),
            ("U", "Relative U (‰)", "uncertainty"),
            ("k", "k", "k"),
            ("nn", _wide_amount_ratio_header(ratio_name), "value"),
            ("u_c_abs", "u_c (ratio units)", "uncertainty"),
            ("U_abs", "U (ratio units)", "uncertainty"),
        ])
        max_columns = max(max_columns, len(columns))

        banner_text = literal_excel_text(f"Ratio:  {ratio_label}")
        banner = ws.cell(row=row_number, column=1, value=banner_text)
        ws.merge_cells(
            start_row=row_number,
            start_column=1,
            end_row=row_number,
            end_column=len(columns),
        )
        banner.font = Font(bold=True, color="FFFFFF")
        banner.fill = banner_fill
        banner.alignment = Alignment(horizontal="left", vertical="center")
        row_number += 1

        for column_index, (_key, header, _number_format) in enumerate(columns, 1):
            cell = ws.cell(row=row_number, column=column_index, value=header)
            _apply_header_style(cell)
        ws.row_dimensions[row_number].height = 50
        if first_header_row is None:
            first_header_row = row_number
        row_number += 1

        for sample, budget in rows:
            values = _wide_row_values(
                sample,
                ratio_name,
                budget,
                contributor_columns,
                delta_mode=delta_mode,
            )
            alternate = row_number % 2 == 0
            for column_index, (key, _header, role) in enumerate(columns, 1):
                value = values.get(key)
                if isinstance(value, str):
                    cell_value = value
                elif key == "run" and isinstance(value, (int, np.integer)):
                    cell_value = int(value)
                else:
                    cell_value = safe_float(value)
                cell = ws.cell(row=row_number, column=column_index, value=cell_value)
                _apply_data_style(cell, alternate, role)
            row_number += 1
        row_number += 2

    for column_index in range(1, max_columns + 1):
        ws.column_dimensions[get_column_letter(column_index)].width = 15
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 8
    ws.column_dimensions["C"].width = 7
    if first_header_row is not None:
        # Identity columns freeze with the first block's header (U57). The
        # sheet holds one banner-and-header block per ratio, so it is not one
        # rectangular range and carries no AutoFilter.
        ws.freeze_panes = f"D{first_header_row + 1}"
        _apply_print_settings(
            ws,
            landscape=True,
            title_rows=f"{first_header_row}:{first_header_row}",
            title_cols="A:C",
        )


def _create_applicability_sheet(
    wb: Workbook,
    result: ProcessingResult,
) -> None:
    """Sample Applicability sheet: one row per sample with applicability metadata."""
    ws = wb.create_sheet("Sample Applicability")
    # The per-sample "values" block only carries Sr-specific contributor values
    # (entered via the Sr-only sample-contributor editor). Omit the column for
    # every other element so a Li/B/Mg/Cd/Pb session does not show a stray,
    # always-empty "Sr Values" column.
    include_sr_values = result.element_symbol == "Sr"
    headers = [
        "Sample",
        "Run Number",
        "Profile",
        "Source",
        "Overrides",
        "Custom Enabled",
    ]
    if include_sr_values:
        headers.append("Sr Values")
    headers.append("Note")
    for c_idx, h in enumerate(headers, 1):
        _apply_header_style(ws.cell(row=1, column=c_idx, value=h))

    row = 2
    for sample in result.samples:
        uc = (sample.metadata or {}).get("uncertainty_contributors") or {}
        # An unassigned sample still runs under the profile the engines resolve,
        # so a blank Profile cell misstated what the budget used.
        profile = uc.get("profile") or SampleContributorApplicability.from_sample(sample).profile
        source = uc.get("source") or ("user_assigned" if uc else "default")
        overrides = uc.get("overrides", {})
        custom_enabled = uc.get("custom_enabled", [])
        values = uc.get("values", {})
        note = uc.get("note", "")

        overrides_str = "; ".join(
            f"{k}={'ON' if v else 'OFF'}" for k, v in sorted(overrides.items())
        )
        custom_str = ", ".join(sorted(custom_enabled))
        value_parts = []
        if isinstance(values, dict):
            for contributor_name, block in sorted(values.items()):
                if not isinstance(block, dict):
                    continue
                fields = ", ".join(
                    f"{field}={value:g}"
                    for field, value in sorted(block.items())
                    if isinstance(value, (int, float))
                )
                if fields:
                    value_parts.append(f"{contributor_name}: {fields}")
        values_str = "; ".join(value_parts)

        vals = [
            literal_excel_text(sample.name),
            sample.run_number,
            literal_excel_text(profile),
            literal_excel_text(source),
            literal_excel_text(overrides_str),
            literal_excel_text(custom_str),
        ]
        if include_sr_values:
            vals.append(literal_excel_text(values_str))
        vals.append(literal_excel_text(note))
        alt = row % 2 == 0
        for c_idx, val in enumerate(vals, 1):
            cell = ws.cell(row=row, column=c_idx, value=val)
            _apply_data_style(cell, alt, "count" if c_idx == 2 else ("wrap" if c_idx > 4 else "text"))
        row += 1

    widths = [24, 10, 28, 16, 50, 40]
    if include_sr_values:
        widths.append(60)  # Sr Values
    widths.append(40)  # Note
    for col_idx, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = w
    ws.freeze_panes = "C2"
    ws.row_dimensions[1].height = 25
    _apply_table_filter(ws, last_row=row - 1, last_col=len(headers))
    _apply_print_settings(ws, landscape=True, title_cols="A:B")


def _create_mc_cross_check_sheet(
    wb: Workbook,
    result: ProcessingResult,
    *,
    uncertainty_config: Optional[object] = None,
) -> None:
    """Monte Carlo cross-check sheet: one row per durable sample/ratio record.

    Values are written as canonical full-precision numbers, exactly as the
    JSON, CSV and HDF5 exports write them. Presentation rounding is a display
    concern and is applied by the UI, not baked into an export cell.

    Header wording follows the GUM convention this phase adopts: ``u_c`` names
    a combined standard uncertainty, and the expanded uncertainty column is
    accompanied by its coverage factor, while the Monte Carlo interval is
    accompanied by its coverage probability, convention and percentile method.
    No draw array is written: see :data:`DRAW_ARRAY_POLICY`.
    """
    ws = wb.create_sheet(MC_CROSS_CHECK_SHEET)

    # Interpretation first (U63): identity, result space and freshness, the GUM
    # and Monte Carlo values with their interval convention and moment status,
    # then warnings and scope. Run configuration, software identifiers and
    # digests follow as supporting columns. Headers are unchanged; only their
    # order moved. The sheet reports diagnostics side by side and states no
    # verdict or threshold.
    headers: List[str] = []
    rows_written = 0
    for sample, ratio_name, record in iter_mc_results(result.samples):
        row = 2 + rows_written
        placements = "; ".join(
            f"{item.name}={item.placement}"
            f"{f' [{item.distribution}]' if item.distribution else ''}"
            for item in record.contributors
        )
        freshness = mc_record_freshness_for_budget(
            record,
            (getattr(sample, "uncertainty", None) or {}).get(ratio_name),
            uncertainty_config,
            sample=sample,
        )
        cells: List[Tuple[str, object, Optional[str]]] = [
            ("Sample", sample.name, "text"),
            ("Run Number", sample.run_number, "count"),
            ("Ratio", ratio_name, "text"),
            ("Result Space", record.effective_result_space or record.result_space, "text"),
            ("Result Freshness", freshness, "text"),
            ("Output Mode", record.output_mode, "text"),
            ("GUM Centre", safe_float(record.gum_center), "value"),
            ("u_c (GUM combined standard uncertainty)", safe_float(record.gum_u_c), "uncertainty"),
            ("U (GUM expanded uncertainty)", safe_float(record.gum_u_expanded), "uncertainty"),
            ("Coverage Factor k", safe_float(record.gum_coverage_factor_k), "k"),
            ("MC Mean", safe_float(record.mc_mean), "value"),
            ("MC SD (ddof=1)", safe_float(record.mc_std), "uncertainty"),
            ("MC Interval Lower", safe_float(record.mc_lower), "value"),
            ("MC Interval Upper", safe_float(record.mc_upper), "value"),
            ("Coverage Probability", safe_float(record.coverage_probability), "probability"),
            ("Interval Convention", record.interval_convention, "text"),
            ("Percentile Method", record.percentile_method, "text"),
            ("Moment Status", record.moment_status, "text"),
            ("Warnings", " | ".join(record.warnings), "wrap"),
            ("Scope Note", record.scope_note, "wrap"),
            ("Engine", record.engine, "text"),
            ("Semantics", record.semantics_label, "text"),
            ("Semantics Version", record.semantics_version, "text"),
            ("Iteration Return Space", record.iteration_return_space, "text"),
            ("Post-Loop Transform Applied", "Yes" if record.post_loop_transform_applied else "No", "text"),
            ("Bracket Mode", record.bracket_mode, "text"),
            ("Preceding Standard", record.prev_std_label, "text"),
            ("Following Standard", record.next_std_label, "text"),
            ("Requested Draws", record.requested_draws, "count"),
            ("Completed Draws", record.completed_draws, "count"),
            ("Draws Dropped", record.n_dropped, "count"),
            ("Blank Input Model", record.blank_uncertainty_input, "text"),
            ("Blank Placement", record.blank_placement, "text"),
            ("Contributor Placements", placements, "text"),
            ("Observation ID", sample.observation_id, "text"),
            ("Execution ID", record.execution_id, "text"),
            # Text, not a number. openpyxl writes numeric cells with "%.16g",
            # which silently corrupts any seed at or above 2**53; a seed is an
            # identity, so an approximate one is worse than useless.
            ("Seed", "" if record.seed is None else str(record.seed), "text"),
            ("Bit Generator", record.bit_generator, "text"),
            ("Config Digest", record.config_digest, "text"),
            ("Input Digest", record.input_digest, "text"),
            ("App Version", record.app_version, "text"),
            ("NumPy Version", record.numpy_version, "text"),
            ("Python Version", record.python_version, "text"),
            ("Platform", record.os_platform, "text"),
            ("Schema", f"{record.schema_name} v{record.schema_version}", "text"),
            # openpyxl writes numeric cells with "%.16g", so a spreadsheet cell
            # cannot always hold the seventeenth significant digit of a double.
            # The canonical payload is therefore carried verbatim as text, and
            # it — not the numeric column — is the full-precision value. A
            # record larger than one cell is written to the versioned overflow
            # sheet and this cell carries the pointer; see
            # reassemble_long_payload().
            (
                "Canonical Record (JSON)",
                _store_long_payload(
                    wb,
                    f"mc_record_r{row}",
                    json.dumps(record.to_dict(), allow_nan=False),
                ),
                None,
            ),
        ]
        headers = [header for header, _value, _role in cells]
        alt = row % 2 == 0
        for c_idx, (header, value, role) in enumerate(cells, 1):
            if isinstance(value, str) and header != "Canonical Record (JSON)":
                value = literal_excel_text(value)
            _apply_data_style(ws.cell(row=row, column=c_idx, value=value), alt, role)
        rows_written += 1

    for c_idx, header in enumerate(headers, 1):
        _apply_header_style(ws.cell(row=1, column=c_idx, value=header))
    row = 2 + rows_written

    note_row = row + 1
    ws.cell(row=note_row, column=1, value=DRAW_ARRAY_POLICY)
    ws.cell(
        row=note_row + 1,
        column=1,
        value=(
            "Numeric cells are unrounded machine values written to 16 "
            "significant digits, the spreadsheet writer's limit; display "
            "rounding is applied separately in the application. The "
            "\"Canonical Record (JSON)\" column carries the exact "
            "full-precision values."
        ),
    )
    ws.cell(
        row=note_row + 2,
        column=1,
        value=(
            "Result Freshness compares each stored result against the "
            "uncertainty configuration and the analytical budget that were "
            "active when this file was written. \"current\" means both match. "
            "\"stale_inputs\" means the measured data changed after the run "
            "(for example a cycle mask or outlier-filter edit) and the result "
            "has not been recomputed. \"stale_configuration\" means the "
            "settings changed. \"unknown\" means freshness could not be "
            "established and the result is unverified. Only \"current\" "
            "results should be reported."
        ),
    )
    ws.cell(
        row=note_row + 3,
        column=1,
        value=(
            f"Record schema: {MC_RESULT_SCHEMA_NAME} v{MC_RESULT_SCHEMA_VERSION}."
        ),
    )

    for c_idx, header in enumerate(headers, 1):
        ws.column_dimensions[get_column_letter(c_idx)].width = max(
            14, min(46, len(header) + 4)
        )
    ws.freeze_panes = "D2"
    ws.row_dimensions[1].height = 45
    if headers:
        _apply_table_filter(ws, last_row=row - 1, last_col=len(headers))
    _apply_print_settings(ws, landscape=True, title_cols="A:C")
    # Notes sit below the data, so the print area must include them too.


def _create_contributor_profiles_sheet(
    wb: Workbook,
    profiles: Mapping[str, ContributorProfile],
    *,
    samples: Optional[List[Sample]] = None,
    scope: str = PROFILE_SCOPE_USED,
) -> None:
    """Contributor Profiles sheet.

    By default only the profiles the exported observations resolve to are
    listed; ``scope="library"`` lists the complete supplied library for audit
    (U55). Resolution uses the same applicability rule as the engines and the
    Sample Applicability sheet. A resolved profile missing from the supplied
    library is still listed, marked as such, rather than silently dropped.
    """
    ws = wb.create_sheet("Contributor Profiles")
    headers = [
        "Profile",
        "Display Name",
        "Built In",
        "Description",
        "Defaults",
        "Custom Enabled Defaults",
        "Observations Using Profile",
    ]
    for c_idx, h in enumerate(headers, 1):
        _apply_header_style(ws.cell(row=1, column=c_idx, value=h))

    usage: Counter = Counter(
        SampleContributorApplicability.from_sample(sample).profile
        for sample in (samples or [])
    )
    if scope == PROFILE_SCOPE_LIBRARY:
        names = list(profiles)
    else:
        names = [name for name in profiles if usage.get(name)]
    missing = sorted(name for name in usage if name not in profiles)

    row = 2
    for name in sorted(names, key=lambda item: (not profiles[item].builtin, item)):
        profile = profiles[name]
        defaults = json.dumps(dict(profile.defaults), sort_keys=True)
        custom_enabled = ", ".join(sorted(profile.custom_enabled_defaults))
        vals = [
            literal_excel_text(name),
            literal_excel_text(profile.display_name),
            "Yes" if profile.builtin else "No",
            literal_excel_text(profile.description),
            literal_excel_text(defaults),
            literal_excel_text(custom_enabled),
            usage.get(name, 0),
        ]
        _write_profile_row(ws, row, vals)
        row += 1
    for name in missing:
        vals = [
            literal_excel_text(name), "", "",
            "Profile not found in the supplied library", "", "", usage[name],
        ]
        _write_profile_row(ws, row, vals)
        row += 1

    widths = {"A": 28, "B": 28, "C": 9, "D": 50, "E": 40, "F": 30, "G": 12}
    for col_letter, w in widths.items():
        ws.column_dimensions[col_letter].width = w
    ws.freeze_panes = "B2"
    ws.row_dimensions[1].height = 30
    _apply_table_filter(ws, last_row=row - 1, last_col=len(headers))
    _apply_print_settings(ws, landscape=True, title_cols="A:A")


def _write_profile_row(ws, row: int, vals: List[object]) -> None:
    alt = row % 2 == 0
    for c_idx, val in enumerate(vals, 1):
        role = "count" if c_idx == 7 else ("wrap" if c_idx in (4, 5, 6) else "text")
        _apply_data_style(ws.cell(row=row, column=c_idx, value=val), alt, role)


def _create_raw_data_summary_sheet(
    wb: Workbook,
    result: ProcessingResult,
    include_corrected: bool,
    raw_layout: str,
    *,
    corrected_label: str = "Corrected",
    ratio_names: Optional[List[str]] = None,
) -> None:
    """Sheet 3+ — Cycle_Data."""
    if raw_layout == "per_sample":
        _write_raw_per_sample(
            wb,
            result,
            include_corrected,
            corrected_label=corrected_label,
            ratio_names=ratio_names,
        )
    else:
        _write_raw_stacked(
            wb,
            result,
            include_corrected,
            corrected_label=corrected_label,
            ratio_names=ratio_names,
        )


def _collect_cycle_columns(
    sample: Sample,
    ratio_names: List[str],
    include_corrected: bool,
    *,
    corrected_label: str,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Collect per-cycle columns for raw summary blocks."""
    cols: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for iso in sorted(sample.intensities.keys()):
        cd = sample.intensities[iso]
        cols.append((f"Intensity {format_isotope_label(iso)} (V)", cd.values, cd.mask))

    for ratio_name in ratio_names:
        if include_corrected:
            blank_cd = sample.blank_corrected_ratios.get(ratio_name)
            if blank_cd is not None:
                cols.append(
                    (
                        f"Blank-corrected {format_isotope_label(ratio_name)}",
                        blank_cd.values,
                        blank_cd.mask,
                    )
                )
            # The corrected branch of this sheet writes no raw ratio column,
            # so there is no emitted duplicate to suppress (A074).
            cd = _get_best_corrected(sample, ratio_name, dedupe_against_raw=False)
            # Exact inequality: a real correction stays in the export
            # however small it is (A035).
            if cd is None or cycle_data_equal(cd, blank_cd):
                continue
            prefix = corrected_label
        else:
            cd = sample.ratios.get(ratio_name)
            prefix = "Raw"
        if cd is None:
            continue
        cols.append((f"{prefix} {format_isotope_label(ratio_name)}", cd.values, cd.mask))
    return cols


def _write_sample_cycle_block(
    ws,
    start_row: int,
    sample: Sample,
    ratio_names: List[str],
    include_corrected: bool,
    *,
    corrected_label: str,
    columns: Optional[List[Tuple[str, np.ndarray, np.ndarray]]] = None,
) -> int:
    """Write per-cycle data table for one sample and return next row index."""
    if columns is None:
        columns = _collect_cycle_columns(
            sample,
            ratio_names,
            include_corrected,
            corrected_label=corrected_label,
        )
    if not columns:
        ws.cell(row=start_row, column=1, value="No cycle data available")
        return start_row + 2

    n_cycles = max(len(values) for _, values, _ in columns)
    headers = ["Cycle"]
    for name, _, _ in columns:
        headers.extend((name, f"{name} — status"))

    for c_idx, header in enumerate(headers, 1):
        _apply_header_style(ws.cell(row=start_row, column=c_idx, value=header))

    row = start_row + 1
    for i in range(n_cycles):
        cycle_cell = ws.cell(row=row, column=1, value=i + 1)
        _apply_data_style(cycle_cell, i % 2 == 1)
        for series_idx, (_, values, mask) in enumerate(columns):
            value_col = 2 + (series_idx * 2)
            status_col = value_col + 1
            has_value = i < len(values)
            accepted = has_value and i < len(mask) and bool(mask[i])
            value = safe_float(values[i]) if has_value else None
            status = "Accepted" if accepted else ("Excluded" if has_value else "Not recorded")
            value_cell = ws.cell(row=row, column=value_col, value=value)
            status_cell = ws.cell(row=row, column=status_col, value=status)
            _apply_data_style(value_cell, i % 2 == 1)
            _apply_data_style(status_cell, i % 2 == 1)
        row += 1

    for c_idx in range(1, len(headers) + 1):
        ws.column_dimensions[get_column_letter(c_idx)].width = 18
    ws.column_dimensions["A"].width = 8
    return row


def _write_sample_stats_block(
    ws,
    start_row: int,
    sample: Sample,
    ratio_names: List[str],
    include_corrected: bool,
    *,
    corrected_label: str,
) -> int:
    """Write per-sample summary stats table and return next row index."""
    title = (
        "Summary Statistics (selected corrected ratio layers)"
        if include_corrected
        else "Summary Statistics (Raw ratios)"
    )
    ws.cell(row=start_row, column=1, value=title).font = _FONT_BOLD_SECTION
    start_row += 1

    stats_headers = ["Ratio", "n", "Mean", "SD", "SE", "RSD%"]
    for c_idx, header in enumerate(stats_headers, 1):
        _apply_header_style(ws.cell(row=start_row, column=c_idx, value=header))
    row = start_row + 1

    row_idx = 0
    for ratio_name in ratio_names:
        entries: List[Tuple[str, Optional[CycleData]]] = []
        if include_corrected:
            blank_cd = sample.blank_corrected_ratios.get(ratio_name)
            best_cd = _get_best_corrected(
                sample, ratio_name, dedupe_against_raw=False
            )
            if blank_cd is not None:
                entries.append((f"Blank-corrected {format_isotope_label(ratio_name)}", blank_cd))
            if best_cd is not None and not cycle_data_equal(best_cd, blank_cd):
                entries.append((f"{corrected_label} {format_isotope_label(ratio_name)}", best_cd))
        else:
            entries.append((format_isotope_label(ratio_name), sample.ratios.get(ratio_name)))

        for label, cd in entries:
            if cd is None:
                continue
            s = _cd_stats(cd)
            vals = [
                literal_excel_text(label),
                s.get("n"),
                safe_float(s.get("mean")),
                safe_float(s.get("sd")),
                safe_float(s.get("se")),
                safe_float(s.get("rsd")),
            ]
            for c_idx, val in enumerate(vals, 1):
                cell = ws.cell(row=row, column=c_idx, value=val)
                _apply_data_style(cell, row_idx % 2 == 1)
            row += 1
            row_idx += 1

    ws.column_dimensions["A"].width = 28
    for col in ("B", "C", "D", "E", "F"):
        ws.column_dimensions[col].width = 14
    return row


def _write_raw_stacked(
    wb: Workbook,
    result: ProcessingResult,
    include_corrected: bool,
    *,
    corrected_label: str = "Corrected",
    ratio_names: Optional[List[str]] = None,
) -> None:
    ws = wb.create_sheet("Cycle_Data")
    ratio_names = ratio_names if ratio_names is not None else _get_ratio_names(result)

    current_row = 1
    for sample in result.samples:
        label = literal_excel_text(
            f"{sample.name}  [{sample.sample_type}]  Run Number {sample.run_number}"
        )
        lc = ws.cell(row=current_row, column=1, value=label)
        lc.font = _FONT_BOLD_LEGEND
        lc.fill = PatternFill(start_color="CFE2F3", end_color="CFE2F3", fill_type="solid")
        columns = _collect_cycle_columns(
            sample,
            ratio_names,
            include_corrected,
            corrected_label=corrected_label,
        )
        cols_count = (2 * len(columns)) + 1
        ws.merge_cells(start_row=current_row, start_column=1, end_row=current_row, end_column=max(cols_count, 10))
        current_row += 1

        current_row = _write_sample_cycle_block(
            ws,
            current_row,
            sample,
            ratio_names,
            include_corrected,
            corrected_label=corrected_label,
            columns=columns,
        )
        current_row += 1
        current_row = _write_sample_stats_block(
            ws,
            current_row,
            sample,
            ratio_names,
            include_corrected,
            corrected_label=corrected_label,
        )
        current_row += 2


def _write_raw_per_sample(
    wb: Workbook,
    result: ProcessingResult,
    include_corrected: bool,
    *,
    corrected_label: str = "Corrected",
    ratio_names: Optional[List[str]] = None,
) -> None:
    ratio_names = ratio_names if ratio_names is not None else _get_ratio_names(result)

    for sample in result.samples:
        sname = _build_unique_sheet_name(
            wb,
            f"Cycle_{sample.name}",
            fallback="Cycle_Sample",
        )
        ws = wb.create_sheet(sname)
        label = literal_excel_text(
            f"{sample.name}  [{sample.sample_type}]  Run Number {sample.run_number}"
        )
        lc = ws.cell(row=1, column=1, value=label)
        lc.font = _FONT_BOLD_LEGEND
        lc.fill = PatternFill(start_color="CFE2F3", end_color="CFE2F3", fill_type="solid")
        columns = _collect_cycle_columns(
            sample,
            ratio_names,
            include_corrected,
            corrected_label=corrected_label,
        )
        cols_count = (2 * len(columns)) + 1
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(cols_count, 10))
        next_row = _write_sample_cycle_block(
            ws,
            2,
            sample,
            ratio_names,
            include_corrected,
            corrected_label=corrected_label,
            columns=columns,
        )
        _write_sample_stats_block(
            ws,
            next_row + 1,
            sample,
            ratio_names,
            include_corrected,
            corrected_label=corrected_label,
        )


def _create_cycle_sheet(wb: Workbook, sample: Sample) -> None:
    """Optional per-sample cycle-level sheet with per-series masked values."""
    sname = _build_unique_sheet_name(wb, f"Cycles_{sample.name}", fallback="Cycles")
    ws = wb.create_sheet(sname)

    columns: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Intensities are Neptune detector signals in volts (U65).
    for iso, cd in sample.intensities.items():
        columns[f"Raw {iso} (V)"] = (cd.values, cd.mask)
    for iso, cd in sample.corrected_intensities.items():
        columns[f"Corr {iso} (V)"] = (cd.values, cd.mask)
    for rname, cd in sample.ratios.items():
        columns[f"Raw {rname}"] = (cd.values, cd.mask)
    for rname, cd in sample.blank_corrected_ratios.items():
        columns[f"Blank {rname}"] = (cd.values, cd.mask)
    for rname, cd in sample.corrected_ratios.items():
        blank_cd = sample.blank_corrected_ratios.get(rname)
        if blank_cd is not None and cycle_data_equal(cd, blank_cd):
            continue
        columns[f"Corr {rname}"] = (cd.values, cd.mask)
    for rname, cd in sample.iif_corrected_ratios.items():
        columns[f"IIF {rname}"] = (cd.values, cd.mask)
    for rname, cd in sample.sr_standard_corrected_ratios.items():
        columns[f"Sr-standard calibrated {rname}"] = (cd.values, cd.mask)
    for rname, cd in sample.drift_corrected_ratios.items():
        columns[f"Drift {rname}"] = (cd.values, cd.mask)

    if not columns:
        ws["A1"] = "No cycle data available"
        return

    n_cycles = max(len(values) for values, _ in columns.values())
    headers = ["Cycle"]
    for name in columns:
        headers.extend((name, f"{name} — status"))
    for c_idx, h in enumerate(headers, 1):
        _apply_header_style(ws.cell(row=1, column=c_idx, value=h))

    for i in range(n_cycles):
        row = i + 2
        ws.cell(row=row, column=1, value=i + 1)
        for series_idx, (_, payload) in enumerate(columns.items()):
            arr, mask = payload
            value_col = 2 + (series_idx * 2)
            status_col = value_col + 1
            has_value = i < len(arr)
            accepted = has_value and i < len(mask) and bool(mask[i])
            ws.cell(row=row, column=value_col, value=safe_float(arr[i]) if has_value else None)
            ws.cell(
                row=row,
                column=status_col,
                value="Accepted" if accepted else ("Excluded" if has_value else "Not recorded"),
            )

    for c_idx in range(1, len(headers) + 1):
        ws.column_dimensions[get_column_letter(c_idx)].width = 14


# Data extraction helpers


def _get_best_corrected(
    sample: Sample, rname: str, *, dedupe_against_raw: bool = True
) -> Optional[CycleData]:
    """Return the measured-scale corrected layer before drift and SSB.

    ``dedupe_against_raw`` suppresses a corrected layer that is byte-identical
    to the raw layer. That is only ever right when the raw layer is actually
    written to the same sheet: with ``include_raw=False`` it removed the only
    copy of the data the caller asked for, and a no-op correction is an
    ordinary reachable state rather than malformed input.
    """
    selected = select_best_pre_drift_ratio_layer(sample, rname)
    if selected is None or selected.key == "ratios":
        return None
    if dedupe_against_raw:
        raw = sample.ratios.get(rname) if sample.ratios else None
        if cycle_data_equal(selected.data, raw):
            return None
    return selected.data


def _cd_stats(cd: Optional[CycleData]) -> Dict:
    """Stats dict from a CycleData, or empty dict if None/empty."""
    if cd is None or cd.n_valid == 0:
        return {}
    stats = calculate_cycle_statistics(cd.valid_values)
    return {
        "mean": stats.mean,
        "sd": stats.sd,
        "se": stats.se,
        "2se": 2.0 * stats.se,
        "rsd": stats.rsd_percent,
        "n": stats.n,
    }


def _get_ssb_mean(sample: Sample, rname: str) -> Optional[float]:
    ssb_cd = get_ssb_cycle_data(sample, rname)
    if ssb_cd is None or ssb_cd.n_valid == 0:
        return None
    return float(np.mean(ssb_cd.valid_values))


def _get_ssb_k(sample: Sample, rname: str) -> Optional[float]:
    ssb = sample.ssb_results.get(rname, {})
    k = ssb.get("k_factor")
    return float(k) if k is not None else None


def _get_delta_stats(sample: Sample, rname: str) -> Optional[Dict]:
    return sample.delta_results.get(rname) or None
# Excel sheet name helpers.
_INVALID_SHEET_CHARS = set("[]:*?/\\")


def _build_unique_sheet_name(wb: Workbook, base_name: str, *, fallback: str) -> str:
    """Return a workbook-unique sheet name (<=31 chars, invalid chars replaced)."""
    cleaned = "".join("_" if ch in _INVALID_SHEET_CHARS else ch for ch in (base_name or ""))
    cleaned = cleaned.strip().strip("'")
    if not cleaned or cleaned.lower() == "history":
        cleaned = fallback

    cleaned = cleaned[:31].strip("'")
    if not cleaned:
        cleaned = fallback[:31]

    if cleaned not in wb.sheetnames and cleaned.lower() != "history":
        return cleaned

    idx = 1
    while True:
        suffix = f"_{idx}"
        keep = max(1, 31 - len(suffix))
        candidate = f"{cleaned[:keep].strip(chr(39))}{suffix}"
        if candidate not in wb.sheetnames and candidate.lower() != "history":
            return candidate
        idx += 1


def _validate_workbook_sheet_names(wb: Workbook) -> None:
    """Fail before saving if an invalid Excel sheet name slipped through."""
    for name in wb.sheetnames:
        if (
            len(name) > 31 
            or any(ch in name for ch in _INVALID_SHEET_CHARS) 
            or name.startswith("'") 
            or name.endswith("'") 
            or name.lower() == "history"
        ):
            raise ValueError(
                f"Invalid Excel sheet name generated: {name!r}. "
                "Sheet names must be <= 31 characters, cannot start/end with a single quote, "
                "cannot be 'History', and cannot contain []:*?/\\."
            )


# Feature-presence helpers


def _get_ratio_names(result: ProcessingResult) -> List[str]:
    names: set = set()
    for s in result.samples:
        names.update(s.ratios.keys())
        names.update(s.blank_corrected_ratios.keys())
        names.update(s.corrected_ratios.keys())
        names.update(s.iif_corrected_ratios.keys())
        names.update(s.drift_corrected_ratios.keys())
        names.update(getattr(s, "interference_corrected_ratios", {}).keys())
        names.update(getattr(s, "pb_standard_corrected_ratios", {}).keys())
        names.update(getattr(s, "pb_calibrated_delta_cycles", {}).keys())
    return sorted(names)


def _has_blank_corrected(
    result: ProcessingResult, *, dedupe_against_raw: bool = True
) -> bool:
    for sample in result.samples:
        for ratio_name, blank_cd in sample.blank_corrected_ratios.items():
            if not dedupe_against_raw:
                return True
            raw_cd = sample.ratios.get(ratio_name)
            if not cycle_data_equal(blank_cd, raw_cd):
                return True
    return False


def _has_corrected(
    result: ProcessingResult, *, dedupe_against_raw: bool = True
) -> bool:
    for sample in result.samples:
        ratio_names = set(sample.ratios)
        ratio_names.update(sample.blank_corrected_ratios)
        ratio_names.update(sample.corrected_ratios)
        ratio_names.update(sample.iif_corrected_ratios)
        for ratio_name in ratio_names:
            if _get_best_corrected(
                sample, ratio_name, dedupe_against_raw=dedupe_against_raw
            ) is not None:
                return True
    return False


def _has_ssb(result: ProcessingResult) -> bool:
    return any(bool(s.ssb_results) for s in result.samples)


def _has_drift(result: ProcessingResult) -> bool:
    return any(bool(s.drift_corrected_ratios) for s in result.samples)


def _has_delta(result: ProcessingResult) -> bool:
    return any(bool(s.delta_results) for s in result.samples)


def _has_uncertainty(result: ProcessingResult) -> bool:
    return any(bool(s.uncertainty) for s in result.samples)


def _has_wide_budget_content(result: ProcessingResult) -> bool:
    return any(
        _wide_budget(sample, ratio_name) is not None
        for sample in result.samples
        if not sample.is_standard and not sample.is_blank
        for ratio_name in _get_ratio_names(result)
    )


# CRM info helper


def _get_crm_info(
    config: Optional[object],
    element_symbol: str,
) -> Tuple[List[Tuple[str, str, float, float]], List[str]]:
    """Return (crm_rows, warnings)."""
    try:
        from config.reference_materials import (
            get_all_certified_ratios,
            standard_uncertainty_from_values,
        )
        rm_name: Optional[str] = getattr(config, "reference_material", None) if config else None
        ratios = get_all_certified_ratios(element_symbol, crm_name=rm_name)
        rows = []
        warnings = []
        for ratio_name, crm in ratios.items():
            standard_uncertainty = standard_uncertainty_from_values(
                crm.uncertainty,
                crm.k,
            )
            if standard_uncertainty is None:
                warnings.append(
                    f"CRM {crm.name} ratio {ratio_name} has invalid uncertainty/k metadata."
                )
                continue
            rows.append((crm.name, ratio_name, crm.ratio, standard_uncertainty))
        return rows, warnings
    except (KeyError, ValueError, LookupError, AttributeError, TypeError) as exc:
        # item 59: narrow catch to expected lookup / data exceptions; let
        # programming errors (AttributeError from wrong API, etc.) surface only
        # if they are NOT already in this list — importing errors always surface.
        msg = (
            f"CRM lookup failed for element '{element_symbol}': {exc}. "
            f"CRM reference data omitted from export."
        )
        logging.getLogger(__name__).warning(msg)
        return [], [msg]


# ProcessingConfig extraction


def _extract_config_params(
    config: object,
    element_symbol: str = "",
    *,
    include_uncertainty: bool = True,
) -> List[tuple]:
    element_symbol = (element_symbol or "").strip()
    label_map = {
        "blank_mode": "Blank Correction Mode",
        "filter_method": "Outlier Filter Method",
        "filter_threshold": "Filter Threshold",
        "enable_ssb": "SSB Correction",
        "enable_delta": "Delta Calculation",
        "apply_interference_correction": "Interference Correction (Sr)",
        "interference_monitors_enabled": "Enabled Interference Monitors",
        "apply_mass_bias_correction": "Mass Bias Correction (Sr)",
        "reference_material": "Reference Material",
        "sr_session_anchoring": "Calibrate against Sr standard",
        "sr_calibration_standard_ids": "Sr calibration standard observation IDs",
        "coverage_factor": "Coverage Factor (k)",
        "include_certified_uncertainty": "Include CRM Uncertainty",
        "data_preference": "Data Preference",
        "global_cycle_range": "Global Cycle Range",
    }
    if element_symbol == "Pb":
        label_map["apply_mass_bias_correction"] = "Mass Bias Correction (Pb-Tl)"
        label_map["apply_hg_interference_correction"] = "Hg Interference Correction"

    sr_only_fields = {
        "apply_interference_correction",
        "interference_monitors_enabled",
        "sr_session_anchoring",
        "sr_calibration_standard_ids",
    }
    pb_only_fields = {"apply_hg_interference_correction"}
    internal_norm_fields = {"apply_mass_bias_correction", "normalization_ratio_override"}
    # Uncertainty-only settings: hidden when uncertainty export is disabled so the
    # workbook carries no indication of the uncertainty framework.
    uncertainty_only_fields = {"coverage_factor", "include_certified_uncertainty"}

    def should_include_field(name: str) -> bool:
        if name in uncertainty_only_fields and not include_uncertainty:
            return False
        if name in sr_only_fields:
            return element_symbol == "Sr"
        if name in pb_only_fields:
            return element_symbol == "Pb"
        if name in internal_norm_fields:
            return element_symbol in {"Sr", "Pb"}
        return True

    from dataclasses import fields as dc_fields, is_dataclass

    def format_config_value(value):
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if value is None:
            return "(default)"
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True, default=str)
        return value

    params = []
    if is_dataclass(config) and not isinstance(config, type):
        for f in dc_fields(config):
            if f.name in ("normalization_value_override", "subtract_kr_blank"):
                continue
            if not should_include_field(f.name):
                continue
            value = getattr(config, f.name)
            if f.name == "drift" and is_dataclass(value):
                for drift_field in dc_fields(value):
                    drift_value = getattr(value, drift_field.name)
                    params.append(
                        (
                            f"Drift - {drift_field.name.replace('_', ' ').title()}",
                            format_config_value(drift_value),
                        )
                    )
                continue
            if f.name == "pb_standard_calibration" and is_dataclass(value):
                # Pb-only nested settings, flattened like drift so every value
                # is a cell; assignment maps are written as canonical JSON.
                if element_symbol != "Pb":
                    continue
                for calibration_field in dc_fields(value):
                    params.append(
                        (
                            "Pb-standard calibration - "
                            f"{calibration_field.name.replace('_', ' ').title()}",
                            format_config_value(getattr(value, calibration_field.name)),
                        )
                    )
                continue
            label = label_map.get(f.name, f.name.replace("_", " ").title())
            if f.name == "interference_monitors_enabled":
                disabled = [
                    str(k) for k, enabled in sorted(value.items()) if not bool(enabled)
                ]
                value = "All declared monitors" if not disabled else f"Disabled: {', '.join(disabled)}"
            else:
                value = format_config_value(value)
            params.append((label, value))
    else:
        for key, value in vars(config).items():
            if not key.startswith("_"):
                if not should_include_field(key):
                    continue
                label = label_map.get(key, key.replace("_", " ").title())
                params.append((label, str(value)))
    return params


# Cell styling helpers


_THIN        = Side(style="thin")
_BORDER      = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
_HEADER_FILL = PatternFill(start_color="2B5EA7", end_color="2B5EA7", fill_type="solid")
_ALT_FILL    = PatternFill(start_color="EEF3FA", end_color="EEF3FA", fill_type="solid")
_SECTION_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

# item 86: hoisted reusable Font objects to avoid repeated allocation in sheet builders
_FONT_BOLD_TITLE = Font(bold=True, size=14, color="1F3864")
_FONT_BOLD       = Font(bold=True)
_FONT_BOLD_SECTION = Font(bold=True, color="1F3864")
_FONT_BOLD_LEGEND  = Font(bold=True, size=11, color="1F3864")


_ALIGN_HEADER    = Alignment(horizontal="left", vertical="center", wrap_text=True)
_ALIGN_TEXT      = Alignment(horizontal="left", vertical="top")
_ALIGN_TEXT_WRAP = Alignment(horizontal="left", vertical="top", wrap_text=True)
_ALIGN_NUMBER    = Alignment(horizontal="right", vertical="top")

#: Significant digits shown per cell role (U60). Display only: the stored
#: value is never rounded or turned into text. ``reported_uncertainty`` cells
#: already hold the ceiling-rounded two-significant-figure value, so they show
#: exactly those digits; unrounded uncertainties show four.
_ROLE_SIGNIFICANT_DIGITS = {"reported_uncertainty": 2, "uncertainty": 4}
_DEFAULT_SIGNIFICANT_DIGITS = 7
_ROLE_FIXED_FORMATS = {
    "count": "0", "k": "0.00", "dof": "0.0", "percent": "0.0", "probability": "0.000",
    "general": "General",
}
#: Unrounded uncertainties switch to scientific notation below 1e-3, so a column
#: of standard errors near 1e-5 does not alternate between fixed and scientific.
_SCIENTIFIC_BELOW = {"uncertainty": 1e-3}


def _apply_header_style(cell) -> None:
    cell.font      = Font(bold=True, color="FFFFFF")
    cell.fill      = _HEADER_FILL
    cell.alignment = _ALIGN_HEADER
    cell.border    = _BORDER


def _apply_data_style(cell, alternate: bool = False, kind: Optional[str] = None) -> None:
    """Style one data cell by role: text left, numbers right (U59), numeric formats by quantity (U60).

    ``kind`` names the cell's role. ``wrap`` wraps descriptive text; any other
    text stays on one line. Without a role a float gets seven significant
    digits and an integer a plain integer format.
    """
    if alternate:
        cell.fill = _ALT_FILL
    cell.border = _BORDER
    value = cell.value
    if isinstance(value, (bool, np.bool_)):
        cell.alignment = _ALIGN_TEXT
    elif isinstance(value, (int, float, np.integer, np.floating)):
        cell.alignment = _ALIGN_NUMBER
        cell.number_format = _number_format(
            kind, float(value), integer=isinstance(value, (int, np.integer))
        )
    elif value == "∞":
        # An infinite degrees of freedom has no numeric cell representation;
        # it stays text but aligns with the numbers beside it.
        cell.alignment = _ALIGN_NUMBER
    else:
        cell.alignment = _ALIGN_TEXT_WRAP if kind == "wrap" else _ALIGN_TEXT


def _number_format(kind: Optional[str], value: float, *, integer: bool = False) -> str:
    if kind in _ROLE_FIXED_FORMATS:
        return _ROLE_FIXED_FORMATS[kind]
    if integer and kind is None:
        return "0"
    digits = _ROLE_SIGNIFICANT_DIGITS.get(kind or "", _DEFAULT_SIGNIFICANT_DIGITS)
    return _significant_format(
        abs(value), digits, scientific_below=_SCIENTIFIC_BELOW.get(kind or "", 1e-5)
    )


def _significant_format(magnitude: float, digits: int, *, scientific_below: float = 1e-5) -> str:
    """Excel format showing ``digits`` significant figures for this magnitude."""
    if magnitude == 0 or not math.isfinite(magnitude):
        return "0"
    if magnitude < scientific_below or magnitude >= 1e7:
        return "0." + "0" * (digits - 1) + "E+00"
    decimals = digits - 1 - math.floor(math.log10(magnitude))
    decimals = max(0, min(10, decimals))
    return "0." + "0" * decimals if decimals else "0"


def _format_as(cell, kind: str) -> None:
    if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
        cell.number_format = _number_format(kind, float(cell.value))


def _coordinate_value_with_uncertainty(value_cell, u_cell) -> None:
    """Show a reported value to the last decimal place of its reported U (U60).

    U is already ceiling-rounded to two significant figures, so both cells
    show the same decimal places. Without a positive finite U in fixed
    notation, the value keeps its own significant-figure format.
    """
    u = u_cell.value
    if not isinstance(u, (int, float)) or isinstance(u, bool):
        return
    u_format = _number_format("reported_uncertainty", float(u))
    u_cell.number_format = u_format
    if "E" in u_format or u <= 0:
        return
    if isinstance(value_cell.value, (int, float)) and not isinstance(value_cell.value, bool):
        value_cell.number_format = u_format


def _apply_table_filter(ws, *, last_row: int, last_col: int, header_row: int = 1) -> None:
    """AutoFilter over one rectangular header-plus-data range (U58)."""
    if last_row > header_row and last_col >= 1:
        ws.auto_filter.ref = f"A{header_row}:{get_column_letter(last_col)}{last_row}"


def _apply_print_settings(
    ws,
    *,
    landscape: bool = True,
    title_rows: Optional[str] = "1:1",
    title_cols: Optional[str] = None,
    fit_width: bool = False,
) -> None:
    """Deliberate print setup (U61).

    Header rows, and identity columns where given, repeat on every page. Only
    the Summary is scaled to the page width; a wide dataset is never shrunk
    onto one unreadable page.
    """
    ws.page_setup.orientation = "landscape" if landscape else "portrait"
    if title_rows:
        ws.print_title_rows = title_rows
    if title_cols:
        ws.print_title_cols = title_cols
    ws.print_area = f"A1:{get_column_letter(max(1, ws.max_column))}{max(1, ws.max_row)}"
    if fit_width:
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0


def _section(ws, row: int, title: str) -> None:
    cell      = ws.cell(row=row, column=1, value=title)
    cell.font = Font(bold=True, color="1F3864")
    cell.fill = _SECTION_FILL


def _kv(ws, row: int, key: str, value) -> None:
    ws.cell(row=row, column=1, value=literal_excel_text(key))
    ws.cell(row=row, column=2, value=literal_excel_text(value))

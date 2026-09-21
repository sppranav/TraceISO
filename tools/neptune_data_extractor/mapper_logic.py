import pandas as pd
import numpy as np
import h5py
import copy
import json
import logging
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections.abc import Mapping
from dataclasses import dataclass, field
from config.input_units import parse_token, UNIT_CONTRACT, LEGACY_CONVENTION

logger = logging.getLogger(__name__)
MAX_NAME_PATTERN_LENGTH = 500

# A mapped column whose values are more than this fraction NaN after numeric
# extraction is almost always a mis-mapping (wrong column or header row): the
# explicit missing markers become NaN. Surfaced as a pre-flight warning.
NAN_FRACTION_THRESHOLD = 0.5
# Lower post-hoc threshold recorded in HDF5 provenance for audit visibility.
NAN_FRACTION_PROVENANCE_THRESHOLD = 0.05
ISOTOPE_NAME_PATTERN = re.compile(r"^\d+[A-Z][a-z]?$")
DISAMBIGUATED_COLUMN_PATTERN = re.compile(r"^(?P<name>.+?) \[col (?P<index>\d+)\]$")

# Numeric/channel-unit parsing is centralized in config.input_units.parse_token.

# --- Supported clock formats for a mapped time column ----------------------
#
# Neptune writes milliseconds after a fourth colon ('12:00:01:250'), which is
# not a strptime format, so it is rewritten to '12:00:01.250' first. Whole
# seconds ('12:00:01') are equally ordinary and are parsed by the second
# format. Both are wall-clock times of day; the extracted Time dataset holds
# seconds elapsed from the first parseable cycle, one entry per data row, NaN
# where a cell could not be parsed, so time stays aligned with the intensities.
NEPTUNE_MILLISECOND_COLON = re.compile(r"^(\d{1,2}:\d{2}:\d{2}):(\d+)$")
SUPPORTED_TIME_FORMATS = ("%H:%M:%S.%f", "%H:%M:%S")
_EMPTY_CELL_TOKENS = frozenset({"", "nan", "nat", "none", "null", "na", "n/a"})
# How many offending cells a rejection message names before it summarises.
_MAX_REPORTED_BAD_CELLS = 3


class CellValueError(ValueError):
    """A mapped cell contains digits but does not match the numeric grammar.

    Carries the file, source column label and raw row index of the first
    offending cells so the user can find them in the mapper grid instead of
    hunting for a silently truncated value in the converted HDF5.
    """

    def __init__(self, message: str, *, fname: str, column: str, rows: List[int]):
        super().__init__(message)
        self.fname = fname
        self.column = column
        self.rows = list(rows)


def disambiguate_header_values(header_values: List[Any]) -> List[str]:
    """Return source column labels decorated with visible grid indices.

    The extractor template is keyed by source column name, so visible column
    indices make every mapping auditable and remove ambiguity when Neptune
    exports repeat cup labels. Blank/NaN headers are left unchanged so the UI
    can keep hiding them.
    """
    raw = [str(value).strip() for value in header_values]
    return [
        f"{name} [col {idx}]" if name and name.lower() != "nan" else name
        for idx, name in enumerate(raw)
    ]


def base_column_name(label: str) -> str:
    match = DISAMBIGUATED_COLUMN_PATTERN.fullmatch(str(label).strip())
    return match.group("name") if match else str(label).strip()


def _base_column_name(label: str) -> str:
    return base_column_name(label)


class ColumnMappingError(ValueError):
    """A template's source column cannot be resolved to one column of a file."""

    def __init__(self, message: str, *, fname: str, source: str, matches: List[str]):
        super().__init__(message)
        self.fname = fname
        self.source = source
        self.matches = list(matches)


class AmbiguousColumnError(ColumnMappingError):
    """A bare source name matches more than one column of the file."""


class ColumnIdentityError(ColumnMappingError):
    """An index-pinned source name does not match the header at that index."""


def _resolve_data_column_name(source_name: str, columns: List[str], fname: str) -> Optional[str]:
    """Resolve a template source column to one unambiguous DataFrame column.

    A mapping written by the current mapper is pinned by grid index — the
    ``Name [col N]`` label — and that pin is validated here: if column ``N``
    of this file carries a different header, the layout is not the one the
    template was built against and resolution fails rather than searching for
    the name elsewhere.

    A legacy mapping carries only a bare header name. When a Neptune export
    repeats a cup label, that name identifies two different columns, and no
    rule can recover which one the author meant — so extraction stops and asks
    for the ``[col N]`` form. Silently taking the first occurrence is what
    turned a `11B/10B` of 4 into 40.
    """
    source = str(source_name).strip()
    if source in columns:
        return source

    pinned = DISAMBIGUATED_COLUMN_PATTERN.fullmatch(source)
    if pinned:
        index = int(pinned.group("index"))
        if 0 <= index < len(columns):
            raise ColumnIdentityError(
                f"Mapped column {source!r} does not match {fname}: column "
                f"{index} of this file is "
                f"{_base_column_name(columns[index])!r}. Re-map the column "
                "against this file's layout.",
                fname=fname,
                source=source,
                matches=[columns[index]],
            )
        return None

    matches = [col for col in columns if _base_column_name(col) == source]
    if not matches:
        return None
    if len(matches) > 1:
        raise AmbiguousColumnError(
            f"Source column {source!r} appears {len(matches)} times in "
            f"{fname} ({', '.join(matches)}), and this mapping names it "
            "without a column index, so which column it means is undecidable. "
            "Re-map it in the mapper using the '[col N]' label of the column "
            "you measured. Extraction will not guess: the repeated cups can "
            "hold different channels.",
            fname=fname,
            source=source,
            matches=matches,
        )
    return matches[0]


def parse_numeric_column(
    raw_series: pd.Series,
    *,
    fname: str,
    column_label: str,
    target_name: str,
    ratio: Optional[bool] = None,
    evidence: Optional[dict] = None,
) -> np.ndarray:
    """Parse a mapped column into floats under the documented numeric grammar.

    Declared voltage units are normalized to V. Ratio channels are dimensionless.
    Explicit missing markers become NaN; arbitrary text and unsupported units
    raise CellValueError with row/channel context without exposing cell contents.
    """
    values: List[float] = []
    bad_rows: List[int] = []
    bad_samples: List[str] = []
    for row_index, raw in raw_series.items():
        try:
            token = parse_token(raw, ratio=("/" in target_name or "_" in target_name) if ratio is None else ratio)
        except ValueError as exc:
            values.append(float("nan"))
            bad_rows.append(int(row_index))
            if len(bad_samples) < _MAX_REPORTED_BAD_CELLS:
                bad_samples.append(f"row {int(row_index)}: {exc}")
            continue
        values.append(token.value)
        if evidence is not None:
            evidence.setdefault(target_name, []).append({
                "row": int(row_index), "source_unit": token.source_unit,
                "canonical_unit": token.canonical_unit, "scale": token.scale,
                "status": token.status,
            })

    if bad_rows:
        detail = "; ".join(bad_samples)
        if len(bad_rows) > len(bad_samples):
            detail += f"; and {len(bad_rows) - len(bad_samples)} more"
        raise CellValueError(
            f"Column '{column_label}' (mapped to '{target_name}') in {fname} "
            f"contains {len(bad_rows)} cell(s) that are not supported numeric "
            f"values — {detail}. Supported: an optional single-letter Neptune "
            "flag, then a number using '.' as the decimal separator with no "
            "digit grouping, then an optional supported channel unit (for example "
            "'X100.5', '-1.25e-3', '101.5 mV'). Detector counts, unknown units, decimal-comma and grouped "
            "numbers are not converted, because reading only the leading "
            "digits would silently report a different value.",
            fname=fname,
            column=str(column_label),
            rows=bad_rows,
        )

    return np.asarray(values, dtype=float)


def parse_time_column(
    raw_column: pd.Series,
    *,
    fname: str,
    column_label: str,
) -> Tuple[np.ndarray, List[str]]:
    """Parse a mapped time column into elapsed seconds, with alignment kept.

    Returns ``(seconds, warnings)``. A clock column is recognised from its first
    non-empty cell and parsed with :data:`SUPPORTED_TIME_FORMATS`, so ordinary
    ``HH:MM:SS`` timestamps work as well as Neptune's ``HH:MM:SS:fff``; the
    result is seconds relative to the first parseable cycle. The array always
    has one entry per data row — NaN for a cell that could not be parsed — so
    ``time_data[i]`` belongs to data row ``i``. An entirely unparseable clock
    column yields an empty array and a warning naming the column, rather than
    the silent empty array that a whole-second file used to produce.
    """
    warnings: List[str] = []
    text = raw_column.astype(str).str.strip()
    non_empty = text[~text.str.lower().isin(_EMPTY_CELL_TOKENS)]
    if non_empty.empty:
        warnings.append(
            f"Time column '{column_label}' is empty; no Time dataset written."
        )
        return np.array([]), warnings

    if ":" not in non_empty.iloc[0]:
        return (
            parse_numeric_column(
                raw_column, fname=fname, column_label=column_label, target_name="Time"
            ),
            warnings,
        )

    clean = text.str.replace(NEPTUNE_MILLISECOND_COLON, r"\1.\2", regex=True)
    parsed = pd.to_datetime(clean, format=SUPPORTED_TIME_FORMATS[0], errors="coerce")
    for fmt in SUPPORTED_TIME_FORMATS[1:]:
        retry = parsed.isna()
        if not retry.any():
            break
        parsed.loc[retry] = pd.to_datetime(clean[retry], format=fmt, errors="coerce")

    unparsed = parsed.isna() & ~text.str.lower().isin(_EMPTY_CELL_TOKENS)
    if not parsed.notna().any():
        warnings.append(
            f"Time column '{column_label}' in {fname}: none of "
            f"{int(unparsed.sum())} value(s) match a supported time format "
            f"({', '.join(SUPPORTED_TIME_FORMATS)}); no Time dataset written."
        )
        return np.array([]), warnings

    if unparsed.any():
        offending = [
            f"row {int(idx)}: {text.loc[idx]!r}"
            for idx in unparsed[unparsed].index[:_MAX_REPORTED_BAD_CELLS]
        ]
        warnings.append(
            f"Time column '{column_label}' in {fname}: "
            f"{int(unparsed.sum())} value(s) could not be parsed and are "
            f"recorded as NaN — {'; '.join(offending)}."
        )

    first_valid = parsed.dropna().iloc[0]
    return (parsed - first_valid).dt.total_seconds().to_numpy(), warnings


def normalize_column_mapping(mapping: Any) -> Dict[str, str]:
    """Normalize legacy column-mapping payloads to ``{source: target}`` dicts.

    Any ``Mapping`` is accepted, not only ``dict``, because a frozen template
    carries its mappings as read-only proxies.
    """
    if isinstance(mapping, Mapping):
        return {
            str(source): str(target)
            for source, target in mapping.items()
            if source is not None and target is not None
        }
    if isinstance(mapping, list):
        return {
            str(col): str(col)
            for col in mapping
            if col is not None
        }
    return {}


def normalize_template_schema(template_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *template_data* with normalized mapping schema."""
    normalized = dict(template_data or {})
    normalized["isotope_cols"] = normalize_column_mapping(
        normalized.get("isotope_cols", {})
    )
    normalized["ratio_cols"] = normalize_column_mapping(
        normalized.get("ratio_cols", {})
    )
    return normalized


def freeze_template_data(template_data: Any) -> Mapping[str, Any]:
    """Return a normalized, deeply-copied, read-only snapshot of a template.

    A batch is one analytical decision: every file in it must be extracted
    under the mapping that was validated when it started. The main window
    hands the worker its live mapping dictionaries, and a control such as
    Clear Mappings in Selection edits those dictionaries in place — so an edit
    made while a batch runs used to reach the remaining files and drop an
    isotope channel from them alone, with no error and no reader warning.

    The snapshot is taken once at the batch boundary. Nested mappings are
    copied and wrapped in :class:`~types.MappingProxyType`, so a later
    in-place edit of the caller's dictionaries cannot be observed here, and an
    accidental write through the snapshot raises instead of succeeding.
    """
    normalized = normalize_template_schema(_thaw(template_data or {}))
    frozen: Dict[str, Any] = {}
    for key, value in normalized.items():
        frozen[key] = MappingProxyType(dict(value)) if isinstance(value, Mapping) else value
    return MappingProxyType(frozen)


def _thaw(value: Any) -> Any:
    """Deep-copy *value* into plain built-ins, so freezing is idempotent.

    ``copy.deepcopy`` cannot copy a ``mappingproxy``, and an already-frozen
    template is a legitimate input — the window freezes what it hands the
    worker, the worker freezes what it received, and ``process_batch`` freezes
    again for callers that bypass both.
    """
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def template_mapping_summary(template_data: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the mapping fields that decide what a batch extracts.

    Recorded in the output's provenance so the converted file states the
    mapping it was actually produced under, not the mapping the window happens
    to hold when someone reads it later.
    """
    return {
        "header_row_idx": template_data.get("header_row_idx"),
        "footer_row_idx": template_data.get("footer_row_idx"),
        "footer_marker": template_data.get("footer_marker"),
        "time_col": template_data.get("time_col"),
        "cycle_col": template_data.get("cycle_col"),
        "isotope_cols": dict(template_data.get("isotope_cols", {})),
        "ratio_cols": dict(template_data.get("ratio_cols", {})),
        "metadata_cells": dict(template_data.get("metadata_cells", {})),
        "isotope_system": template_data.get("isotope_system"),
        "instrument": template_data.get("instrument"),
        "name_pattern": template_data.get("name_pattern"),
    }


def _coerce_run_number(value: Any, *, field_name: str, fname: str) -> int:
    """Extract a numeric run number or default to ``0`` with a warning."""
    text = str(value).strip()
    digits = re.search(r"\d+", text)
    if digits:
        return int(digits.group())
    logger.warning(
        "Could not extract a numeric run number from %r in field %r of %s. "
        "Defaulting to 0.",
        value,
        field_name,
        fname,
    )
    return 0

@dataclass
class ExtractedSample:
    """Represents a fully extracted sample ready for HDF5 serialization.

    ``warnings`` holds non-fatal extraction findings — an unparseable time
    cell, a mapped column absent from this file. They are reported by
    pre-flight and recorded in the output's provenance, so a degraded but
    completed extraction is visible rather than inferred from a missing
    dataset.
    """
    sample_name: str
    metadata: Dict[str, Any]
    data: pd.DataFrame
    time_data: np.ndarray
    cycle_data: np.ndarray
    warnings: List[str] = field(default_factory=list)
    unit_evidence: Dict[str, Any] = field(default_factory=dict)

def load_file_as_df(path) -> pd.DataFrame:
    """Loads a data file (.csv, .txt, .exp, .xls, .xlsx) into a Pandas DataFrame."""
    try:
        import hashlib
        from io import BytesIO, StringIO
        source_bytes = Path(path).read_bytes()
        ext = Path(path).suffix.lower()
        if ext in [".xls", ".xlsx"]:
            frame = pd.read_excel(BytesIO(source_bytes), header=None)
        else:
            # Sniff separator
            text = source_bytes.decode('utf-8-sig', errors='strict')
            head = text[:1024]
            if '\t' in head: sep = '\t'
            elif ',' in head: sep = ','
            else: sep = None
            frame = pd.read_csv(StringIO(text), sep=sep, header=None, engine='python')
        frame.attrs['source_sha256'] = hashlib.sha256(source_bytes).hexdigest()
        frame.attrs['source_sha1'] = hashlib.sha1(source_bytes).hexdigest()
        return frame
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        raise ValueError(f"Failed to load {path}: {e}") from e


def extract_metadata_from_df(df_raw: pd.DataFrame, fname: str, template_data: Dict):
    """
    Extracts metadata from a loaded DataFrame based on the mapping template.
    Returns a tuple of (final_sample_name, metadata_dictionary).
    """
    stem = Path(fname).stem
    meta_map = template_data.get("metadata_cells", {})
    name_pattern = template_data.get("name_pattern", r".*?(\d{2,}_.+)$")
    

    sample_name = stem
    if name_pattern:
        if len(name_pattern) > MAX_NAME_PATTERN_LENGTH:
            logger.warning(
                "Filename pattern is %d characters long (max %d). "
                "Pattern will not be applied for %s. Shorten the pattern "
                "in the template or mapper UI.",
                len(name_pattern),
                MAX_NAME_PATTERN_LENGTH,
                fname,
            )
        else:
            try:
                compiled = re.compile(name_pattern)
                m = compiled.search(stem)
                if m:
                    if len(m.groups()) > 0:
                        sample_name = m.group(1)
                    else:
                        sample_name = m.group(0)
            except Exception as e:
                logger.warning(f"Invalid regex pattern {name_pattern}: {e}")

    extracted_meta = {"Filename": sample_name}
    

    for cell_key, field_name in meta_map.items():
        try:
            r, c = map(int, cell_key.split(","))
            if r < len(df_raw) and c < len(df_raw.columns):
                val = str(df_raw.iloc[r, c]).strip()
                
                # Neptune Name Cleaner
                if field_name in ("Filename", "Sample Name"):
                     clean_name = Path(val).stem
                     m_val = re.search(r'.*?(\d{2,}_.+)$', clean_name)
                     val = m_val.group(1) if m_val else clean_name
                
                extracted_meta[field_name] = val
                
                if field_name == "Sample Name":
                    sample_name = val
                
                if field_name in ("Filename", "Sample Name"):
                    run_match = re.search(r'^(\d+)_', val)
                    if run_match and "Run number" not in extracted_meta:
                        extracted_meta["Run number"] = int(run_match.group(1))

                field_normalized = field_name.lower().replace(" ", "")
                if field_normalized in ("runnumber", "run#", "run"):
                    extracted_meta["Run number"] = _coerce_run_number(
                        val,
                        field_name=field_name,
                        fname=fname,
                    )
                        
        except Exception as e:
            logger.warning(f"Failed to extract metadata {field_name} from {fname}: {e}")

    raw_type = None
    if "Sample Type" in extracted_meta: raw_type = extracted_meta.pop("Sample Type")
    elif "SampleType" in extracted_meta: raw_type = extracted_meta.pop("SampleType")
    elif "Type" in extracted_meta: raw_type = extracted_meta["Type"]

    if raw_type:
        # Strip any leading label like "Sample Type:", "Sample ID:", "Type:", etc.
        clean_raw = re.sub(
            r'^\s*(Sample\s*(Type|ID)|SampleType|Type)\s*:?\s*',
            '', str(raw_type), flags=re.IGNORECASE
        ).strip()
        # Match most-specific first: BLK/Blank before STD/Standard before SMP/Sample
        # so "Sample ID: Blank" correctly resolves to BLK, not SMP.
        type_match = re.search(
            r'\b(BLK|Blank|STD|Standard|SMP|Sample)\b',
            clean_raw, re.IGNORECASE
        )
        if type_match:
            token = type_match.group(1).upper()
            if token in ("BLANK", "BLK"):      clean_type = "BLK"
            elif token in ("STANDARD", "STD"): clean_type = "STD"
            else:                               clean_type = "SMP"
            extracted_meta["Type"] = clean_type
        else:
            # Fallback: infer from filename if cell value gives no clear type
            fname_upper = Path(fname).stem.upper()
            if re.search(r'(^|[_\-\s])BL(AN)?K($|[_\-\s])|_BLK($|[_\-\s])', fname_upper):
                extracted_meta['Type'] = 'BLK'
            elif re.search(r'(^|[_\-\s])STD($|[_\-\s])|STANDARD', fname_upper):
                extracted_meta['Type'] = 'STD'
            else:
                extracted_meta['Type'] = 'SMP'
    else:
        # No cell mapped — try filename-based inference
        fname_upper = Path(fname).stem.upper()
        if re.search(r'(^|[_\-\s])BL(AN)?K($|[_\-\s])|_BLK($|[_\-\s])', fname_upper):
            extracted_meta['Type'] = 'BLK'
        elif re.search(r'(^|[_\-\s])STD($|[_\-\s])|STANDARD', fname_upper):
            extracted_meta['Type'] = 'STD'
        # else: leave Type absent — user did not map it and filename gives no hint


    return sample_name, extracted_meta

def extract_sample_data(fpath: str, template_data: Dict) -> ExtractedSample:
    """
    Extracts metadata and tabular data from a file according to the mapping template.
    Does not perform any disk writing.
    Returns an ExtractedSample object or raises an Exception.
    """
    fname = Path(fpath).name
    template_data = normalize_template_schema(template_data)
    
    header_idx = template_data.get("header_row_idx", 0)
    time_col_name = template_data.get("time_col")
    cycle_col_name = template_data.get("cycle_col")
    iso_cols = template_data.get("isotope_cols", {})
    ratio_cols = template_data.get("ratio_cols", {})
    footer_marker = template_data.get("footer_marker", "***")

    df_raw = load_file_as_df(fpath)
    if df_raw.empty:
        raise ValueError(f"File {fname} is empty or could not be loaded.")
        
    sample_name, extracted_meta = extract_metadata_from_df(df_raw, fname, template_data)
    extracted_meta['source_sha256'] = df_raw.attrs.get('source_sha256', '')
    extracted_meta['source_sha1_legacy'] = df_raw.attrs.get('source_sha1', '')


    if header_idx >= len(df_raw):
        raise ValueError(f"Header row {header_idx} is out of bounds for file {fname}.")

    header_values = df_raw.iloc[header_idx].astype(str).tolist()
    column_labels = disambiguate_header_values(header_values)

    df_data = df_raw.iloc[header_idx+1:].copy()
    df_data.columns = column_labels

    # Footer slicing
    footer_row_idx = template_data.get("footer_row_idx")
    if footer_row_idx is not None:
         nrows = footer_row_idx - header_idx - 1
         if nrows > 0:
             df_data = df_data.head(nrows)
    elif footer_marker:
        first_col = df_data.iloc[:, 0].astype(str)
        mask = first_col.str.startswith(footer_marker, na=False)
        indices = np.where(mask)[0]
        if len(indices) > 0:
            df_data = df_data.iloc[:indices[0]]


    # The dictionary values in iso_cols / ratio_cols are the target names
    # Ensure they exist in columns
    
    final_columns = {}
    extraction_warnings: List[str] = []
    unit_evidence = {}

    # Clean Isotope Columns
    for col_name, target_name in iso_cols.items():
        resolved_col_name = _resolve_data_column_name(
            col_name, list(df_data.columns), fname
        )
        if resolved_col_name is not None:
            final_columns[target_name] = parse_numeric_column(
                df_data[resolved_col_name],
                fname=fname,
                column_label=resolved_col_name,
                target_name=target_name,
                ratio=False, evidence=unit_evidence,
            )
        else:
            message = (
                f"Isotope column '{col_name}' (mapped to '{target_name}') is not "
                f"present in {fname}; that channel is missing from the output."
            )
            logger.warning(message)
            extraction_warnings.append(message)

    # Clean Ratio Columns
    for col_name, target_name in ratio_cols.items():
        resolved_col_name = _resolve_data_column_name(
            col_name, list(df_data.columns), fname
        )
        if resolved_col_name is not None:
            final_columns[target_name] = parse_numeric_column(
                df_data[resolved_col_name],
                fname=fname,
                column_label=resolved_col_name,
                target_name=target_name,
                ratio=True, evidence=unit_evidence,
            )
        else:
            message = (
                f"Ratio column '{col_name}' (mapped to '{target_name}') is not "
                f"present in {fname}; that channel is missing from the output."
            )
            logger.warning(message)
            extraction_warnings.append(message)

    df_final = pd.DataFrame(final_columns)


    t_vals = np.array([])
    c_vals = np.array([])
    
    resolved_time_col = (
        _resolve_data_column_name(time_col_name, list(df_data.columns), fname)
        if time_col_name
        else None
    )
    if resolved_time_col is not None:
        t_vals, time_warnings = parse_time_column(
            df_data[resolved_time_col],
            fname=fname,
            column_label=resolved_time_col,
        )
        extraction_warnings.extend(time_warnings)
    elif time_col_name:
        message = (
            f"Time column '{time_col_name}' is not present in {fname}; "
            "no Time dataset written."
        )
        logger.warning(message)
        extraction_warnings.append(message)

    resolved_cycle_col = (
        _resolve_data_column_name(cycle_col_name, list(df_data.columns), fname)
        if cycle_col_name
        else None
    )
    if resolved_cycle_col is not None:
        c_vals = pd.to_numeric(df_data[resolved_cycle_col], errors='coerce').values

    return ExtractedSample(
        sample_name=sample_name,
        metadata=extracted_meta,
        data=df_final,
        time_data=t_vals,
        cycle_data=c_vals,
        warnings=extraction_warnings,
        unit_evidence=unit_evidence,
    )


def write_sample_to_hdf5(f: h5py.File, sample: ExtractedSample, template_data: Dict):
    """Write one ExtractedSample to an open HDF5 file, all of it or none of it.

    A sample group is only a converted sample once every mapped channel is in
    it. If any part of the write fails, the group is unlinked before the error
    propagates — otherwise the reader, which counts any group containing an
    ``intensities`` group, would load a half-written sample as a real one with
    a channel silently missing, exactly the failure mode A088 fixed on the
    mapping side.
    """
    grp_name = sample.sample_name
    dup_count = 1
    while grp_name in f:
        grp_name = f"{sample.sample_name}_{dup_count}"
        dup_count += 1

    template_data = normalize_template_schema(template_data)
    metadata_to_write = dict(sample.metadata)
    if template_data.get("include_audit_metadata", True):
        metadata_to_write['channel_units'] = {
            'schema': UNIT_CONTRACT, 'bare_number_convention': LEGACY_CONVENTION,
            'channels': sample.unit_evidence,
        }
    else:
        for key in ("channel_units", "extractor_source_id", "source_sha256", "source_sha1_legacy"):
            metadata_to_write.pop(key, None)
    if "Run number" in metadata_to_write and not isinstance(metadata_to_write["Run number"], int):
        metadata_to_write["Run number"] = _coerce_run_number(
            metadata_to_write["Run number"],
            field_name="Run number",
            fname=grp_name,
        )

    grp = f.create_group(grp_name)
    try:
        grp.create_dataset("metadata", data=json.dumps(metadata_to_write))

        if "instrument" in template_data and template_data["instrument"]:
            grp.attrs["instrument"] = template_data["instrument"]

        if "Date" in sample.metadata:
            grp.attrs["Date"] = sample.metadata["Date"]

        int_grp = grp.create_group("intensities")

        # Write Time & Cycle
        if len(sample.time_data) > 0:
            int_grp.create_dataset("Time", data=sample.time_data)
        if len(sample.cycle_data) > 0:
            int_grp.create_dataset("Cycle", data=sample.cycle_data)

        # Write Isotopes
        iso_cols = template_data.get("isotope_cols", {})

        for _, target_name in iso_cols.items():
            if target_name in sample.data.columns:
                arr = sample.data[target_name].values
                dataset = int_grp.create_dataset(target_name.strip(), data=arr)
                dataset.attrs["units"] = "V"
                dataset.attrs["unit_contract"] = UNIT_CONTRACT

        # Write Ratios
        ratio_cols = template_data.get("ratio_cols", {})

        if ratio_cols:
            rat_grp = grp.create_group("ratios")
            for _, target_name in ratio_cols.items():
                if target_name in sample.data.columns:
                    arr = sample.data[target_name].values
                    dataset = rat_grp.create_dataset(target_name.strip().replace("/", "_"), data=arr)
                    dataset.attrs["units"] = "dimensionless"
                    dataset.attrs["unit_contract"] = UNIT_CONTRACT
    except BaseException:
        try:
            if grp_name in f:
                del f[grp_name]
        except Exception as cleanup_exc:  # noqa: BLE001 - never mask the real error
            logger.error(
                "Could not remove the incomplete sample group %r: %s",
                grp_name,
                cleanup_exc,
            )
        raise


def column_nan_fractions(sample: ExtractedSample) -> Dict[str, float]:
    """Return the fraction of NaN values per extracted data column.

    After numeric extraction, un-parseable cells become NaN. A high fraction is
    the classic symptom of a column mapped to the wrong source column or a wrong
    header row, which would otherwise produce a silently NaN-filled HDF5.
    """
    fractions: Dict[str, float] = {}
    n = len(sample.data)
    for col in sample.data.columns:
        fractions[col] = 1.0 if n == 0 else float(sample.data[col].isna().mean())
    return fractions


def high_nan_columns(
    sample: ExtractedSample, threshold: float = NAN_FRACTION_THRESHOLD
) -> Dict[str, float]:
    """Return only the columns whose NaN fraction exceeds *threshold*."""
    return {
        col: frac
        for col, frac in column_nan_fractions(sample).items()
        if frac > threshold
    }


def _load_element_config(isotope_system: str):
    """Return the TraceISO ``ElementConfig`` for *isotope_system*, or ``None``.

    Returns ``None`` when the system is not one of the supported elements or the
    domain/config layer cannot be imported, so the mapper still works for
    unsupported systems (e.g. Nd/Hf/U) and when run outside the full project.
    The import is lazy to keep ``mapper_logic`` usable standalone.
    """
    try:
        from domain.elements.registry import get_element
    except Exception as exc:  # noqa: BLE001 - domain layer optional for the tool
        logger.warning("Could not import TraceISO element registry: %s", exc)
        return None
    try:
        return get_element(isotope_system)
    except Exception as exc:  # noqa: BLE001 - unknown system / CRM lookup failure
        logger.warning("Could not load element config for %s: %s", isotope_system, exc)
        return None


def validate_template_names(template_data: Dict) -> List[str]:
    """Warn when mapped target isotope/ratio names are invalid.

    Catches typos such as ``Sr87`` instead of ``87Sr`` at mapping time rather
    than letting them surface as a missing channel inside TraceISO. Isotope
    channels from other elements (for example ``117Sn`` measured alongside Cd)
    are valid monitor/interference data and are accepted when they use canonical
    ``<mass><element>`` notation. Ratio names are validated against
    ``domain.elements.registry`` (the same source the app uses).
    Unknown/unsupported systems are skipped silently.
    """
    template_data = normalize_template_schema(template_data)
    system = str(template_data.get("isotope_system", "")).strip()
    cfg = _load_element_config(system)
    if cfg is None:
        return []

    iso_expected = set(cfg.isotopes)
    ratio_expected = set(cfg.ratio_names)
    for aliases in cfg.ratio_name_aliases.values():
        ratio_expected.update(aliases)

    warnings: List[str] = []
    for target in sorted(set(template_data.get("isotope_cols", {}).values())):
        if target not in iso_expected and not ISOTOPE_NAME_PATTERN.fullmatch(target):
            warnings.append(
                f"Isotope target '{target}' is not a valid isotope name. "
                "Use canonical <mass><element> notation, for example "
                f"'114{system}' or '117Sn'."
            )
    for target in sorted(set(template_data.get("ratio_cols", {}).values())):
        if target not in ratio_expected:
            warnings.append(
                f"Ratio target '{target}' is not a recognised {system} ratio. "
                f"Expected one of: {', '.join(cfg.ratio_names)}."
            )
    return warnings


# --- Batch completion status, written to the output and read by TraceISO ---
#
# The normal output filename the user chose is a promise that the file is the
# run they asked for. It is therefore published only when the batch reaches the
# end of its file list. A batch that stops early still keeps what it converted,
# because those samples are real measurements — but under an unmistakable
# ".PARTIAL" name and carrying the status below, which file_io/hdf5_reader
# surfaces on load as both a warning and HDF5LoadResult.completion_status.
BATCH_STATUS_ATTR = "completion_status"
BATCH_COMPLETED = "completed"
BATCH_COMPLETED_WITH_ERRORS = "completed_with_errors"
BATCH_CANCELLED = "cancelled"
BATCH_FAILED = "failed"
PARTIAL_OUTPUT_MARKER = ".PARTIAL"
STAGING_OUTPUT_SUFFIX = ".staging"


def staging_output_path(output_h5: Any) -> Path:
    """Return the path a batch writes to before it is allowed a final name."""
    import uuid
    return Path(str(output_h5) + "." + uuid.uuid4().hex + STAGING_OUTPUT_SUFFIX)


def partial_output_path(output_h5: Any) -> Path:
    """Return the unmistakable name a batch that stopped early is kept under."""
    final = Path(output_h5)
    import uuid
    return final.with_name(f"{final.stem}{PARTIAL_OUTPUT_MARKER}.{uuid.uuid4().hex}{final.suffix}")


class BatchOutputError(RuntimeError):
    """The output file itself could not be written, or could not be published.

    Distinct from a per-file extraction failure: an unreadable source file is
    a property of that file and the batch carries on without it, but a failure
    of the *output* means nothing further can be vouched for. ``retained_path``
    names where the converted data actually is, so the caller never has to
    guess and never reports a filename that does not exist.
    """

    def __init__(self, message: str, *, retained_path: Any):
        super().__init__(message)
        self.retained_path = str(retained_path)


def _publish_batch_output(
    staging_path: Path,
    destination: Path,
    *,
    status: Optional[str] = None,
    **counts: Any,
) -> Path:
    """Move the staged output onto *destination*, optionally stamping status.

    ``status`` is only passed on the failure path, where the write block was
    torn down before it could stamp anything; the file is reopened for append
    and stamped best effort, because a file too damaged to reopen is exactly
    the case where losing the bytes would be worst.

    Raises :class:`BatchOutputError` if the move fails, naming the staging path
    the data is still at. A publication failure means the file the caller asked
    for does not exist, and returning quietly would let a `completed` outcome
    assert a filename that is not there.
    """
    import os

    if not staging_path.exists():
        raise BatchOutputError("Owned staging output is missing", retained_path=staging_path)
    if status is not None:
        try:
            with h5py.File(staging_path, "a") as f:
                _write_batch_status(f, status=status, **counts)
        except Exception as exc:  # noqa: BLE001 - keep the bytes, report the cause
            logger.error(
                "Could not stamp %s on the partial output %s: %s",
                status,
                staging_path,
                exc,
            )
    try:
        # Atomic no-clobber publication on the destination filesystem. Even an
        # uncooperative competing writer cannot be replaced by this operation.
        os.link(staging_path, destination)
    except OSError as exc:
        logger.error(
            "Could not publish the extractor output as %s: %s. "
            "The converted data remains at %s.",
            destination,
            exc,
            staging_path,
        )
        raise BatchOutputError(
            f"The converted data could not be saved as {destination}: {exc}. "
            f"It is retained at {staging_path} — move or rename that file to "
            "recover the batch.",
            retained_path=staging_path,
        ) from exc
    try:
        staging_path.unlink()
    except OSError:
        logger.warning("Published output retained an additional owned staging link: %s", staging_path)
    return destination


def _batch_counts(outcome: "BatchOutcome", file_list: List[str]) -> Dict[str, Any]:
    """Return the status and file counts, with every intended file accounted for.

    ``intended == converted + failed + unprocessed`` holds for all four
    endings. A file is *failed* if it was attempted and produced no sample, and
    *unprocessed* only if it was never reached — the distinction a reader needs
    to tell "this file is bad" from "we never got to it".
    """
    intended = len(file_list)
    converted = len(outcome.processed)
    failed = len(outcome.failed)
    unprocessed = len(outcome.unprocessed)
    if converted + failed + unprocessed != intended:
        # Belt and braces: never publish counts that do not add up. Anything
        # unaccounted for was not converted and not attempted.
        unprocessed = max(intended - converted - failed, 0)
    return {
        "status": outcome.completion_status,
        "intended": intended,
        "converted": converted,
        "failed": failed,
        "unprocessed": unprocessed,
    }


def _write_batch_status(
    f: h5py.File,
    *,
    status: str,
    intended: int,
    converted: int,
    failed: int,
    unprocessed: int,
) -> None:
    """Stamp the batch's completion state and file counts on the root group."""
    f.attrs[BATCH_STATUS_ATTR] = status
    f.attrs["files_intended"] = int(intended)
    f.attrs["files_converted"] = int(converted)
    f.attrs["files_failed"] = int(failed)
    f.attrs["files_unprocessed"] = int(unprocessed)


@dataclass
class BatchOutcome:
    """Structured result of a batch conversion run.

    A flat list of error strings cannot distinguish "this file failed" from
    "the user cancelled the batch" — collapsing both into one list caused the
    UI to name-match cancellation markers against files (finding none) and
    mark every unprocessed file VALID, plus show a misleading "Done with
    Errors" dialog for a plain cancellation.
    """

    errors: List[str] = field(default_factory=list)
    cancelled: bool = False
    processed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    unprocessed: List[str] = field(default_factory=list)
    sample_count: int = 0
    output_path: str = ""
    # One of BATCH_COMPLETED / BATCH_COMPLETED_WITH_ERRORS / BATCH_CANCELLED /
    # BATCH_FAILED — the same value stamped on the written file, so a caller
    # need not reopen the output to learn whether it is the whole run.
    completion_status: str = ""
    # Stamped by BatchWorker with the run token active when the worker was
    # started, so a UI slot can detect and ignore a stale/superseded run's
    # completion signal rather than acting on it (default 0 for callers that
    # invoke process_batch() directly, outside the worker/run-token scheme).
    run_token: int = 0


def process_batch(
    file_list: List[str],
    template_data: Dict,
    output_h5: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> BatchOutcome:
    """Extract and write a batch of files to a single TraceISO HDF5.

    Single source of truth for the conversion path — the Qt ``BatchWorker``
    delegates here. ``progress_callback(percent, filename)`` is called before
    each file; ``cancel_check()`` is polled before and after each file and, when it
    returns ``True``, conversion stops with ``BatchOutcome.cancelled = True``
    and the remaining files reported as ``unprocessed`` rather than errored.

    Returns a :class:`BatchOutcome`. **Failures of the output are separated
    from failures of a source file.** An unreadable or malformed source file is
    a property of that file: it is recorded in ``errors``/``failed`` and the
    batch carries on. A failure while writing the output, or while publishing
    it onto its final name, raises :class:`BatchOutputError` — the half-written
    sample group is unlinked first, so it can never be read back as a converted
    sample, and ``retained_path`` names where the data actually is.

    Counts always reconcile: ``intended == converted + failed + unprocessed``
    for all four endings. The file in flight when a batch aborts is counted as
    failed, not as never attempted.

    **The file name the user chose is published only on completion.** The batch
    writes to a uniquely owned staging path and is published at the final name when it
    reaches the end of its file list — errors on individual files included,
    since those are reported and the batch itself ran to the end. A batch that
    stops early keeps what it converted, because those samples are real
    measurements, but under ``<stem>.PARTIAL.<unique-id><suffix>`` and carrying
    ``completion_status`` plus the intended/converted/failed/unprocessed
    counts, which ``file_io.hdf5_reader.load_hdf5`` surfaces to whoever opens
    it later. ``BatchOutcome.output_path`` always names the file that exists.

    A ``.PARTIAL`` file from an earlier stopped run is deliberately **not**
    deleted by a later successful run to the same target: it holds converted
    measurements this function did not produce and is not ours to discard.

    The template is frozen on entry (:func:`freeze_template_data`), so every
    file in the batch is extracted under the mapping this call started with
    even if the caller edits its own dictionaries meanwhile.
    """
    import hashlib
    import datetime

    template_data = freeze_template_data(template_data)
    file_list = list(file_list)
    destination_path = Path(output_h5)
    if destination_path.exists() or destination_path.is_symlink():
        raise ValueError("Choose a new output file; existing destinations are never replaced.")
    if any(Path(p).resolve() == destination_path.resolve() for p in file_list):
        raise ValueError("Output must not alias a source file")
    output_h5 = str(destination_path.resolve())
    staging_path = staging_output_path(output_h5)
    outcome = BatchOutcome(output_path=str(output_h5))
    successful_samples = 0
    # Index of the file currently being converted, or None between files, so an
    # abort can classify it as failed rather than losing it from the counts.
    in_flight_index: Optional[int] = None
    # Provenance accumulators
    _prov_warnings: List[str] = []
    _prov_nan_fractions: Dict[str, Dict[str, float]] = {}
    _prov_source_hashes: Dict[str, str] = {}
    sources = []
    owned_staging = False
    from config.upstream_provenance import seal
    from config.software_identity import software_identity
    extractor_identity = software_identity()

    def write_upstream(handle):
        if not template_data.get("include_audit_metadata", True):
            return
        handle.attrs['upstream_provenance_json'] = seal({
            'sources': sources, 'template': _thaw(template_data),
            'bare_number_convention': LEGACY_CONVENTION,
            'software': extractor_identity, 'warnings': _prov_warnings,
        })

    try:
        with h5py.File(staging_path, "x") as f:
            owned_staging = True
            f.attrs["schema_version"] = "traceiso.universal.v1"
            f.attrs["created_by"] = "Neptune Data Extractor (PyQt)"
            f.attrs["metadata_mode"] = (
                "full_audit" if template_data.get("include_audit_metadata", True) else "minimal"
            )
            f.attrs["isotope_system"] = template_data.get("isotope_system", "Sr")
            if template_data.get("instrument"):
                f.attrs["instrument"] = template_data["instrument"]

            total = len(file_list) or 1
            for i, fpath in enumerate(file_list):
                if cancel_check is not None and cancel_check():
                    outcome.cancelled = True
                    outcome.unprocessed.extend(file_list[i:])
                    _prov_warnings.append(
                        f"Batch cancelled by user after {i}/{len(file_list)} file(s)."
                    )
                    break
                fname = Path(fpath).name
                if progress_callback is not None:
                    progress_callback(int((i / total) * 100), fname)
                in_flight_index = i
                try:
                    sample_data = extract_sample_data(fpath, template_data)
                    source_id = f"source-{i+1:06d}"
                    sample_data.metadata['extractor_source_id'] = source_id
                    source_record = {'source_id': source_id, 'ordinal': i+1,
                                    'filename': fname, 'sha256': sample_data.metadata['source_sha256'],
                                    'sha1_legacy': sample_data.metadata['source_sha1_legacy'],
                                    'units': sample_data.unit_evidence}
                    _prov_source_hashes[source_id] = sample_data.metadata['source_sha1_legacy']
                    for message in sample_data.warnings:
                        _prov_warnings.append(f"{fname}: {message}")
                    # Per-column NaN fraction for provenance audit trail
                    if sample_data.data is not None and not sample_data.data.empty:
                        _nan_fracs = column_nan_fractions(sample_data)
                        _high_nan = [
                            f"{c}={v:.0%}"
                            for c, v in _nan_fracs.items()
                            if v > NAN_FRACTION_PROVENANCE_THRESHOLD
                        ]
                        if _high_nan:
                            _prov_warnings.append(f"{fname}: high NaN fraction — {', '.join(_high_nan)}")
                        _prov_nan_fractions[fname] = _nan_fracs
                    # A failure inside write_sample_to_hdf5 is a failure of the
                    # OUTPUT, not of this source file: the group is rolled back
                    # and the batch aborts, because nothing further about the
                    # file can be vouched for. Reporting it as "Failed: b1.txt"
                    # and carrying on blamed the source and published the
                    # chosen filename over a suspect artifact.
                    try:
                        sources.append(source_record)
                        # Seal the accumulated lineage once at batch completion.
                        # Replacing this growing variable-length attribute per
                        # sample leaves old HDF5 allocations behind (quadratic
                        # file growth). The abort handler seals recovery files.
                        write_sample_to_hdf5(f, sample_data, template_data)
                    except Exception as storage_exc:
                        raise BatchOutputError(
                            f"Writing {fname} into {staging_path} failed: "
                            f"{storage_exc}. The batch was stopped and the "
                            "incomplete sample was removed from the output.",
                            retained_path=staging_path,
                        ) from storage_exc
                    successful_samples += 1
                    outcome.processed.append(fpath)
                except BatchOutputError:
                    raise
                except Exception as e:
                    err_msg = f"Failed: {fname}: {e}"
                    logger.error(err_msg)
                    outcome.errors.append(err_msg)
                    outcome.failed.append(fpath)
                    _prov_warnings.append(err_msg)
                in_flight_index = None

                # Finish the current sample atomically, then honor a request
                # received during extraction/write, including the last file.
                if cancel_check is not None and cancel_check():
                    outcome.cancelled = True
                    outcome.unprocessed.extend(file_list[i + 1:])
                    _prov_warnings.append(
                        f"Batch cancelled by user after {i + 1}/{len(file_list)} file(s)."
                    )
                    break

            f.attrs["sample_count"] = successful_samples
            if outcome.cancelled:
                outcome.completion_status = BATCH_CANCELLED
            elif outcome.errors:
                outcome.completion_status = BATCH_COMPLETED_WITH_ERRORS
            else:
                outcome.completion_status = BATCH_COMPLETED
            _write_batch_status(f, **_batch_counts(outcome, file_list))
            # Write accumulated provenance after all files are processed
            if template_data.get("include_audit_metadata", True):
                f.attrs["provenance_json"] = json.dumps({
                    "writer": "neptune_data_extractor",
                    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "warnings": _prov_warnings,
                    "nan_fractions": _prov_nan_fractions,
                    "source_sha1": _prov_source_hashes,
                    "mapping": template_mapping_summary(template_data),
                })
            write_upstream(f)
    except BaseException as abort_exc:
        if not owned_staging:
            raise
        # The output file is unusable as "the run the user asked for", but
        # whatever was written is still measurement data. Keep it under the
        # partial name, stamped failed if the file can still be opened, and
        # let the caller see the original error.
        outcome.completion_status = BATCH_FAILED
        # The file that was in flight was attempted and did not convert, so it
        # is failed, not unattempted; everything after it was never reached.
        # Without this the counts silently lost every file past the abort.
        if in_flight_index is not None:
            aborted = file_list[in_flight_index]
            if aborted not in outcome.failed:
                outcome.failed.append(aborted)
                outcome.errors.append(
                    f"Failed: {Path(aborted).name}: batch aborted while "
                    f"converting it ({abort_exc})."
                )
        outcome.unprocessed = list(
            file_list[len(outcome.processed) + len(outcome.failed):]
        )
        outcome.sample_count = len(outcome.processed)
        try:
            with h5py.File(staging_path, "a") as recovery:
                write_upstream(recovery)
        except Exception as provenance_exc:
            # Preserve the original output failure and the recoverable bytes
            # even if the same storage failure prevents recording lineage.
            logger.error(
                "Could not record provenance in recovery output %s: %s",
                staging_path, provenance_exc,
            )
        try:
            outcome.output_path = str(
                _publish_batch_output(
                    staging_path,
                    partial_output_path(output_h5),
                    # completion_status is already BATCH_FAILED, so the counts
                    # helper carries the status through with it.
                    **_batch_counts(outcome, file_list),
                )
            )
        except BatchOutputError as publish_exc:
            # Two failures at once. The abort is the cause and must surface;
            # where the bytes ended up still has to be reported.
            outcome.output_path = publish_exc.retained_path
            logger.error("%s", publish_exc)
        if isinstance(abort_exc, BatchOutputError):
            abort_exc.retained_path = outcome.output_path
            abort_exc.args = (f"{abort_exc.args[0]} Retained output: {outcome.output_path}",)
        else:
            abort_exc.add_note(f"Retained output: {outcome.output_path}")
        raise

    destination = (
        partial_output_path(output_h5) if outcome.cancelled else Path(output_h5)
    )
    outcome.output_path = str(_publish_batch_output(staging_path, destination))
    outcome.sample_count = successful_samples

    if progress_callback is not None:
        if outcome.cancelled:
            progress_callback(
                int((len(outcome.processed) / total) * 100), "Cancelled"
            )
        else:
            progress_callback(100, "Done")

    return outcome


def pre_flight_validation(
    file_list: List[str],
    template_data: Dict,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> List[str]:
    """Validate a batch before conversion, writing nothing.

    Surfaces three classes of issue as human-readable messages:

    * extraction failures (raised exceptions),
    * mapped columns that come out mostly NaN (mis-mapping / wrong header row),
    * target isotope/ratio names unknown to the selected isotope system.

    Single source of truth for the validation path — the Qt ``ValidationWorker``
    delegates here. Returns a flat list of messages (empty == clean). The
    template is frozen on entry, so validation and the conversion it gates
    cannot be reasoning about different mappings.
    """
    template_data = freeze_template_data(template_data)
    file_list = list(file_list)
    messages: List[str] = []

    # Template-level (file-independent) name checks — reported once.
    messages.extend(validate_template_names(template_data))

    total = len(file_list) or 1
    seen_identities: Dict[Tuple[str, int], str] = {}
    for i, fpath in enumerate(file_list):
        if cancel_check is not None and cancel_check():
            messages.append("Cancelled by user.")
            break
        fname = Path(fpath).name
        if progress_callback is not None:
            progress_callback(int((i / total) * 100), fname)
        try:
            sample = extract_sample_data(fpath, template_data)
        except Exception as e:
            messages.append(f"{fname}: {e}")
            continue
        for message in sample.warnings:
            messages.append(f"{fname}: {message}")
        run_number = _coerce_run_number(
            sample.metadata.get("Run number", 0),
            field_name="Run number",
            fname=fname,
        )
        identity = (sample.sample_name, run_number)
        previous = seen_identities.get(identity)
        if previous is not None:
            messages.append(
                f"Duplicate sample/run identity: {sample.sample_name!r} run "
                f"{run_number} appears in both {previous} and {fname}."
            )
        else:
            seen_identities[identity] = fname
        for col, frac in high_nan_columns(sample).items():
            messages.append(
                f"{fname}: column '{col}' is {frac * 100:.0f}% empty/non-numeric "
                "— check the column mapping and header row."
            )

    if progress_callback is not None:
        progress_callback(100, "Validation complete")

    return messages

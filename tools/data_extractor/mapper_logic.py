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

# --- Positional (headerless) layouts ----------------------------------------
#
# Some instrument exports have no column-header row at all: the classic Nu
# Plasma cycle file goes straight from a "Sample Name is ..." line into rows of
# 19 zero readings, 19 on-peak readings, elapsed time, cycle and clock. Such a
# template sets ``"layout": "positional"`` and names its source columns
# ``"col N"`` (0-based grid index). With no header to check a pin against, the
# layout is guarded instead by ``data_start_marker`` (the data must start right
# after one matching line) and ``data_row_width`` (every data row must be
# exactly that wide), so a different acquisition method fails loudly rather
# than being read with the wrong cup assignment. Baseline subtraction
# (``baseline_cols``) is independent of the layout and works in both modes.
POSITIONAL_LAYOUT = "positional"
POSITIONAL_COLUMN_PATTERN = re.compile(r"^col (?P<index>\d+)$")

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
    exports repeat cup labels. A column with a blank header cell is labelled
    ``col N`` (its position), the same form a headerless layout uses, so it
    can still be mapped and given a name; it then matches only a file whose
    header cell at that index is blank too.
    """
    # An empty cell may load as None or NaN; neither is a name.
    raw = ["" if value is None else str(value).strip() for value in header_values]
    return [
        f"{name} [col {idx}]" if name and name.lower() != "nan" else f"col {idx}"
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
    seconds = (parsed - first_valid).dt.total_seconds().to_numpy()
    # A clock carries no date, so a run that crosses midnight steps back by
    # almost a day. Cycles are minutes apart at most; a backwards step of more
    # than half a day is therefore read as the next day, never as time travel.
    finite = np.flatnonzero(np.isfinite(seconds))
    if finite.size > 1:
        steps = np.diff(seconds[finite])
        days = np.concatenate([[0], np.cumsum(steps < -43200)])
        if days[-1]:
            seconds[finite] += days * 86400.0
            warnings.append(
                f"Time column '{column_label}' in {fname}: clock passed midnight "
                f"{int(days[-1])} time(s); later cycles were placed on the following day."
            )
    return seconds, warnings


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
        "run_number_pattern": template_data.get("run_number_pattern"),
        "intensity_baseline_corrected": template_data.get("intensity_baseline_corrected"),
        "preserve_sample_name": template_data.get("preserve_sample_name"),
        "layout": template_data.get("layout"),
        "data_start_marker": template_data.get("data_start_marker"),
        "data_row_width": template_data.get("data_row_width"),
        "baseline_cols": dict(template_data.get("baseline_cols", {}) or {}),
        "baseline_enabled": baseline_subtraction_enabled(template_data),
        "metadata_patterns": dict(template_data.get("metadata_patterns", {}) or {}),
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
    # Filled only when baselines were subtracted: per isotope, the signal as
    # read and the baseline that was taken off it (both in V, cycle-aligned).
    uncorrected_data: Dict[str, np.ndarray] = field(default_factory=dict)
    baseline_data: Dict[str, np.ndarray] = field(default_factory=dict)

def load_file_as_df(path) -> pd.DataFrame:
    """Loads a data file (.csv, .txt, .exp, .xls, .xlsx) into a Pandas DataFrame."""
    try:
        import hashlib
        import csv
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
            # Instrument exports have short metadata rows before a wider
            # measurement table (Nu Plasma also has a trailing header comma).
            # Determine the complete width with CSV quoting respected, then
            # pad short rows rather than dropping any metadata or data rows.
            # Keep pandas' existing blank-line handling so saved grid indices
            # for Neptune templates retain their meaning.
            read_options = {}
            if sep is not None:
                width = max((len(row) for row in csv.reader(StringIO(text), delimiter=sep)), default=0)
                if width:
                    read_options["names"] = list(range(width))
            frame = pd.read_csv(StringIO(text), sep=sep, header=None, engine='python', **read_options)
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
                preserve_name = field_name == "Sample Name" and template_data.get("preserve_sample_name", False)
                if field_name in ("Filename", "Sample Name") and not preserve_name:
                     clean_name = Path(val).stem
                     m_val = re.search(r'.*?(\d{2,}_.+)$', clean_name)
                     val = m_val.group(1) if m_val else clean_name
                
                extracted_meta[field_name] = val
                
                if field_name == "Sample Name":
                    sample_name = val
                
                if field_name in ("Filename", "Sample Name") and not preserve_name:
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

    # Pattern metadata: for exports that embed a value in a text line rather
    # than in a cell of its own ("Sample Name is   SRM987_20ppb_10_11"). Each
    # pattern is matched against the first-column text of every row; the first
    # match wins and its first capture group (else the whole match) is taken
    # verbatim, so a sample ID is never shortened by the Neptune name cleaner.
    for field_name, pattern in (template_data.get("metadata_patterns") or {}).items():
        if len(str(pattern)) > MAX_NAME_PATTERN_LENGTH:
            raise ValueError(f"Metadata pattern for {field_name!r} is too long.")
        compiled = re.compile(str(pattern))
        value = None
        for cell in df_raw.iloc[:, 0].tolist() if len(df_raw.columns) else []:
            if not isinstance(cell, str):
                continue
            match = compiled.search(cell)
            if match:
                value = (match.group(1) if match.lastindex else match.group(0)).strip()
                break
        if not value:
            logger.warning("Metadata pattern for %r found no match in %s.", field_name, fname)
            continue
        extracted_meta[field_name] = value
        if field_name == "Sample Name":
            sample_name = value

    # Some instruments put the acquisition number only in the source filename.
    # An explicit template rule avoids confusing a sample ID with run order.
    run_pattern = template_data.get("run_number_pattern")
    if run_pattern and "Run number" not in extracted_meta:
        if len(run_pattern) > MAX_NAME_PATTERN_LENGTH:
            raise ValueError("Run-number pattern is too long.")
        match = re.fullmatch(run_pattern, stem)
        if match is None or match.lastindex is None or not match.group(1).isdigit():
            raise ValueError(f"Run-number pattern must capture an integer from {fname}.")
        extracted_meta["Run number"] = int(match.group(1))

    # This is source-state metadata, not an instruction to subtract a baseline.
    if "intensity_baseline_corrected" in template_data:
        extracted_meta["intensity_baseline_corrected"] = template_data["intensity_baseline_corrected"]

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

def _positional_data_block(df_raw: pd.DataFrame, fname: str, template_data: Mapping[str, Any]) -> pd.DataFrame:
    """Return the data rows of a headerless export, labelled ``col N``.

    The data start is the row after the single first-column line matching
    ``data_start_marker``; ``footer_marker`` then trims the end as usual.
    Every remaining row must be exactly ``data_row_width`` cells wide (counted
    to its last non-empty cell), which is the only layout identity a headerless
    file offers.
    """
    marker = template_data.get("data_start_marker")
    if not marker:
        raise ValueError("A positional template needs a 'data_start_marker' pattern.")
    if len(str(marker)) > MAX_NAME_PATTERN_LENGTH:
        raise ValueError("Data-start marker pattern is too long.")
    compiled = re.compile(str(marker))
    first_col = df_raw.iloc[:, 0]
    hits = [i for i, cell in enumerate(first_col.tolist())
            if isinstance(cell, str) and compiled.search(cell)]
    if len(hits) != 1:
        raise ValueError(
            f"{fname}: expected exactly one line matching data-start marker "
            f"{marker!r}, found {len(hits)}; this is not the layout the template describes."
        )
    df_data = df_raw.iloc[hits[0] + 1:].copy()
    df_data.columns = [f"col {i}" for i in range(len(df_raw.columns))]

    footer_marker = template_data.get("footer_marker")
    if footer_marker:
        starts = df_data.iloc[:, 0].astype(str).str.strip().str.startswith(footer_marker, na=False)
        stop = np.where(starts)[0]
        if len(stop) == 0:
            raise ValueError(
                f"{fname}: footer marker {footer_marker!r} not found; the file may be truncated."
            )
        df_data = df_data.iloc[:stop[0]]
    if df_data.empty:
        raise ValueError(f"{fname}: no data rows between the data-start marker and the footer.")

    expected = template_data.get("data_row_width")
    if expected is not None:
        filled = df_data.notna() & ~df_data.astype(str).apply(lambda s: s.str.strip()).isin(["", "nan"])
        widths = [int(np.flatnonzero(row).max()) + 1 if row.any() else 0 for row in filled.to_numpy()]
        wrong = [(int(idx), w) for idx, w in zip(df_data.index, widths) if w != int(expected)]
        if wrong:
            shown = "; ".join(f"row {r}: {w} cells" for r, w in wrong[:_MAX_REPORTED_BAD_CELLS])
            raise ValueError(
                f"{fname}: {len(wrong)} data row(s) are not {expected} cells wide ({shown}); "
                "the cup/column layout differs from the template."
            )
    return df_data


def _apply_baseline_subtraction(
    final_columns: Dict[str, np.ndarray],
    df_data: pd.DataFrame,
    baseline_cols: Mapping[str, str],
    *,
    fname: str,
    unit_evidence: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Subtract each channel's per-cycle baseline reading.

    Returns ``(audit record, uncorrected signals, baselines)``; the two arrays
    per isotope are kept so the output can store exactly what was subtracted.

    ``baseline_cols`` maps an extracted isotope target to the source column
    holding its baseline (zero) reading on the same cycle row. It works the same
    for header-pinned names (``H5 (Z1) [col 5]``) and positional ``col N``. Every
    extracted isotope channel must have one, so the output is never a mix of
    corrected and raw channels, and a baseline missing where the signal exists
    is an error rather than a silently NaN cycle.

    Units: signal and baseline are each parsed to volts first (``101.5 mV`` and
    ``0.1015`` are the same value), then subtracted. For a non-ratio channel
    ``config.input_units.parse_token`` only ever yields V and raises on anything
    else (counts, ratios, kV), so an unsupported unit stops the file there.
    """
    missing = sorted(set(final_columns) - set(baseline_cols))
    extra = sorted(set(baseline_cols) - set(final_columns))
    if missing or extra:
        raise ValueError(
            f"{fname}: baseline columns must cover exactly the extracted isotope channels "
            f"(no baseline for: {', '.join(missing) or '-'}; baseline for unmapped/absent: "
            f"{', '.join(extra) or '-'})."
        )
    record: Dict[str, Any] = {
        "applied": True,
        "method": "per-cycle on-peak minus baseline column, both in V, applied by the extractor",
        "channels": {},
    }
    uncorrected: Dict[str, np.ndarray] = {}
    baselines: Dict[str, np.ndarray] = {}

    def units_of(evidence, target):
        return sorted({(e["source_unit"] or "", e["canonical_unit"] or "", e["status"])
                       for e in evidence.get(target, [])})

    for target, source in baseline_cols.items():
        resolved = _resolve_data_column_name(source, list(df_data.columns), fname)
        if resolved is None:
            raise ValueError(f"{fname}: baseline column {source!r} for {target} is not present.")
        # Baseline readings are not output datasets, so their unit evidence stays in
        # this audit record: the HDF5 reader requires every channel-evidence
        # entry to match a written dataset.
        zero_evidence: Dict[str, Any] = {}
        zero = parse_numeric_column(
            df_data[resolved], fname=fname, column_label=resolved,
            target_name=target, ratio=False, evidence=zero_evidence,
        )
        zero_units = units_of(zero_evidence, target)
        signal_units = units_of(unit_evidence, target)
        signal = final_columns[target]
        bad = np.isfinite(signal) & ~np.isfinite(zero)
        if bad.any():
            raise ValueError(
                f"{fname}: baseline column {source!r} for {target} is empty or non-numeric on "
                f"{int(bad.sum())} cycle(s) that carry a signal."
            )
        uncorrected[target] = signal.copy()
        baselines[target] = zero.copy()
        final_columns[target] = signal - zero
        finite = zero[np.isfinite(zero)]
        record["channels"][target] = {
            "baseline_source": source,
            "baseline_mean_V": float(finite.mean()) if finite.size else None,
            "baseline_min_V": float(finite.min()) if finite.size else None,
            "baseline_max_V": float(finite.max()) if finite.size else None,
            "signal_input_units": [list(u) for u in signal_units],
            "baseline_input_units": [list(u) for u in zero_units],
        }
    return record, uncorrected, baselines


# Nu Plasma II headers name each Faraday reading "<cup> (<block>)" and its
# zero "<cup> (Z<block>)", e.g. "H5 (1)" and "H5 (Z1)".
_SIGNAL_LABEL = re.compile(r"^(?P<cup>.+?)\s*\((?P<block>\d+)\)$")


def suggest_baseline_cols(
    columns: List[str],
    isotope_cols: Mapping[str, str],
    existing: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Propose ``{isotope target: baseline source}`` pairs for review.

    Header labels pair by cup name: ``H5 (1) [col 26]`` -> ``H5 (Z1) [col 5]``
    when exactly one such column exists. Positional ``col N`` sources pair by
    the column offset of a pair the user already made (classic Nu: zero col i,
    signal col 19+i), because a headerless file carries no names to match.
    Returns ``(pairs, notes)``; existing pairs are kept and never overwritten.
    Nothing is applied here: the caller shows the proposal for confirmation.
    """
    existing = dict(existing or {})
    pairs: Dict[str, str] = dict(existing)
    notes: List[str] = []
    by_base: Dict[str, List[str]] = {}
    for label in columns:
        by_base.setdefault(base_column_name(label), []).append(label)

    offsets = set()
    for target, base_src in existing.items():
        src = next((s for s, t in isotope_cols.items() if t == target), None)
        a = POSITIONAL_COLUMN_PATTERN.fullmatch(str(src or ""))
        b = POSITIONAL_COLUMN_PATTERN.fullmatch(str(base_src))
        if a and b:
            offsets.add(int(a.group("index")) - int(b.group("index")))

    for source, target in isotope_cols.items():
        if target in pairs:
            continue
        positional = POSITIONAL_COLUMN_PATTERN.fullmatch(str(source))
        if positional:
            if len(offsets) != 1:
                notes.append(
                    f"{target}: map one baseline by hand first, so the column offset "
                    "can be learned" if not offsets else
                    f"{target}: existing pairs use different offsets {sorted(offsets)}; not guessing"
                )
                continue
            index = int(positional.group("index")) - next(iter(offsets))
            candidate = f"col {index}"
            if index < 0 or candidate not in columns:
                notes.append(f"{target}: offset points to {candidate}, which does not exist")
                continue
            pairs[target] = candidate
            continue
        match = _SIGNAL_LABEL.fullmatch(base_column_name(source))
        if not match:
            notes.append(f"{target}: {source!r} is not a '<cup> (<block>)' label")
            continue
        wanted = f"{match.group('cup')} (Z{match.group('block')})"
        found = by_base.get(wanted, [])
        if len(found) != 1:
            notes.append(f"{target}: {len(found)} column(s) named {wanted!r}")
            continue
        pairs[target] = found[0]
    return pairs, notes


def derive_positional_layout(df_raw: pd.DataFrame, first_data_row: int) -> Dict[str, Any]:
    """Describe a headerless block whose first data row the user selected.

    Returns ``data_start_marker`` (a pattern built from the line just above,
    which must be text and unique in the file), ``footer_marker`` (the first
    text line after the numeric block) and ``data_row_width`` (the width every
    data row shares). Raises ValueError when the file does not support a
    positional reading, rather than producing guard values that guard nothing.
    """
    def is_number(value):
        text = str(value).strip()
        if text.lower() in ("", "nan"):
            return False
        try:
            float(text)
            return True
        except ValueError:
            return False

    if not 0 < first_data_row < len(df_raw):
        raise ValueError("Select a data row that has a line above it.")
    above = df_raw.iloc[first_data_row - 1, 0]
    # Mixed columns load as strings, so a number in the line above is still a str.
    if not isinstance(above, str) or not above.strip() or is_number(above):
        raise ValueError("The line above the data must be text (e.g. 'Sample Name is ...').")
    if not is_number(df_raw.iloc[first_data_row, 0]):
        raise ValueError("The selected row does not start with a number; select the first data row.")
    # Keep the label, drop the per-file value: text before a run of 2+ spaces,
    # a tab, a colon or '=' (e.g. 'Sample Name is   SRM987' -> 'Sample Name is').
    label = re.split(r"\s{2,}|\t|:|=", above.strip(), maxsplit=1)[0].strip()
    if not label:
        raise ValueError("Could not take a label from the line above the data.")
    # Spaces need no escaping; leaving them readable keeps saved templates legible.
    marker = "^" + re.escape(label).replace("\\ ", " ") + r"\b"
    compiled = re.compile(marker)
    first_col = df_raw.iloc[:, 0].tolist()
    hits = [i for i, v in enumerate(first_col) if isinstance(v, str) and compiled.search(v)]
    if hits != [first_data_row - 1]:
        raise ValueError(f"The line {label!r} is not unique in this file; cannot anchor the data on it.")

    footer_row = next((i for i in range(first_data_row, len(df_raw)) if not is_number(first_col[i])), None)
    if footer_row is None or footer_row == first_data_row:
        raise ValueError("No numeric data block followed by a text footer line was found.")
    footer = str(first_col[footer_row]).strip()
    block = df_raw.iloc[first_data_row:footer_row]
    filled = block.notna() & ~block.astype(str).apply(lambda s: s.str.strip()).isin(["", "nan"])
    widths = {int(np.flatnonzero(row).max()) + 1 for row in filled.to_numpy() if row.any()}
    if len(widths) != 1:
        raise ValueError(f"Data rows have different widths {sorted(widths)}; not a fixed layout.")
    return {"layout": POSITIONAL_LAYOUT, "data_start_marker": marker,
            "footer_marker": footer, "data_row_width": widths.pop()}


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


    positional = template_data.get("layout") == POSITIONAL_LAYOUT
    if positional:
        sources = list(iso_cols) + list(ratio_cols) + [
            c for c in (time_col_name, cycle_col_name) if c
        ] + list((template_data.get("baseline_cols") or {}).values())
        bad = [s for s in sources if not POSITIONAL_COLUMN_PATTERN.fullmatch(str(s).strip())]
        if bad:
            raise ValueError(
                f"Positional template columns must be named 'col N'; got {', '.join(map(repr, bad))}."
            )
        if template_data.get("footer_row_idx") is not None:
            raise ValueError(
                "A headerless (positional) template ends each file's data at its footer "
                "marker; a fixed footer_row_idx would be ignored, so it is refused. "
                "Remove footer_row_idx from the template."
            )
        df_data = _positional_data_block(df_raw, fname, template_data)
        footer_row_idx = None
        footer_marker = None  # already applied by _positional_data_block
    else:
        if header_idx >= len(df_raw):
            raise ValueError(f"Header row {header_idx} is out of bounds for file {fname}.")

        # Raw values, as the window reads them: astype(str) would turn an
        # empty cell into the name 'None'.
        header_values = df_raw.iloc[header_idx].tolist()
        column_labels = disambiguate_header_values(header_values)

        df_data = df_raw.iloc[header_idx+1:].copy()
        df_data.columns = column_labels
        footer_row_idx = template_data.get("footer_row_idx")

    # Footer slicing
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

    baseline_cols = dict(template_data.get("baseline_cols") or {})
    baseline_enabled = baseline_subtraction_enabled(template_data)
    uncorrected_data: Dict[str, np.ndarray] = {}
    baseline_data: Dict[str, np.ndarray] = {}
    if baseline_enabled:
        if not baseline_cols:
            raise ValueError(
                "Baseline subtraction is switched on but no baseline columns are assigned."
            )
        if template_data.get("intensity_baseline_corrected") is False:
            raise ValueError(
                "Template both maps baseline columns and declares the intensities "
                "not baseline corrected; remove one of the two."
            )
        if ratio_cols:
            raise ValueError("Baseline subtraction applies to isotope channels only; remove ratio_cols.")
        iso_only = {t: v for t, v in final_columns.items()}
        record, uncorrected_data, baseline_data = _apply_baseline_subtraction(
            iso_only, df_data, baseline_cols, fname=fname, unit_evidence=unit_evidence,
        )
        extracted_meta["baseline_correction"] = record
        final_columns.update(iso_only)
        # Source-state metadata now describes what the extractor wrote.
        extracted_meta["intensity_baseline_corrected"] = True
    else:
        # Off (or nothing assigned): values are exported as read. Assignments
        # stay in the template, and the output says plainly that the extractor
        # subtracted nothing.
        extracted_meta["baseline_correction"] = {"applied": False}
        if baseline_cols:
            extracted_meta["baseline_correction"]["assignments_kept"] = baseline_cols

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
        uncorrected_data=uncorrected_data,
        baseline_data=baseline_data,
    )


def baseline_subtraction_enabled(template_data: Mapping[str, Any]) -> bool:
    """Whether a template asks for baseline subtraction.

    ``baseline_enabled`` switches subtraction on or off while the assignments in
    ``baseline_cols`` are kept. Templates written before the switch existed
    have no key; for them, assigning baselines meant subtracting them.
    """
    if "baseline_enabled" in template_data:
        value = template_data["baseline_enabled"]
        if type(value) is not bool:
            raise ValueError("baseline_enabled must be true or false.")
        return value
    return bool(template_data.get("baseline_cols"))


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

        # What was subtracted, cycle by cycle: kept outside 'intensities' so the
        # reader never mistakes a raw or baseline array for a measured channel.
        if sample.baseline_data:
            audit = grp.create_group("baseline_correction")
            audit.attrs["method"] = "per-cycle signal minus baseline, both in V"
            for isotope, baseline in sample.baseline_data.items():
                channel = audit.create_group(isotope)
                for label, values in (("uncorrected", sample.uncorrected_data[isotope]),
                                      ("baseline", baseline)):
                    channel.create_dataset(label, data=values).attrs["units"] = "V"
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
            f.attrs["created_by"] = "Data Extractor (PyQt)"
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
                    "writer": "data_extractor",
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

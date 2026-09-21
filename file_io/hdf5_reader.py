"""HDF5 file reader for TraceISO."""

from __future__ import annotations

import ast
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import h5py
import numpy as np

from domain.filters.outlier import sample_cycle_key
from domain.models import CycleData, Sample
from domain.elements.base import ElementConfig
from domain.elements.registry import detect_element
from domain.uncertainty.mc_result import MCResultRecord

_ALLOWED_DATA_PREFERENCES = {"auto", "raw", "corrected"}
_DEFAULT_MAX_FILE_SIZE_MB = 500.0
_DEFAULT_MAX_CYCLES_PER_DATASET = 100_000
_DEFAULT_MAX_SAMPLES = 10_000
_HDF5_EXT_ERROR = getattr(h5py, "HDF5ExtError", OSError)
_log = logging.getLogger(__name__)

# Dataset names that are metadata helpers, not isotope channels.
# Checked case-insensitively so 'cycle', 'Cycle', 'CYCLE' all match.
_NON_ISOTOPE_KEYS: frozenset = frozenset({
    "cycle", "time", "index", "timestamp", "header", "position",
    "mass", "date", "datetime", "sequence",
})


def _is_isotope_dataset(name: str) -> bool:
    """Return True only if *name* looks like an isotope channel (not metadata)."""
    if name.lower() in _NON_ISOTOPE_KEYS:
        return False
    # Accept numeric-only names like "25" (will be normalised later) and
    # correctly-formed isotope labels like "7Li", "87Sr", "206Pb".
    # Reject anything that is purely alphabetic (e.g. "Cycle", "Time").
    stripped = name.strip()
    return bool(re.match(r"^\d+[A-Za-z]*$", stripped) or re.match(r"^[A-Za-z]+\d+$", stripped))

# Alias mapping for common alternative sample type names in HDF5 metadata
_TYPE_ALIASES: Dict[str, str] = {
    "BLANK": "BLK", "STANDARD": "STD", "SAMPLE": "SMP",
}
_VALID_TYPES = {"SMP", "STD", "BLK"}


@dataclass
class HDF5LoadResult:
    """Result of loading an HDF5 file."""

    samples: List[Sample]
    detected_isotopes: Set[str] = field(default_factory=set)
    detected_element: Optional[ElementConfig] = None
    element_symbol: str = ""
    file_structure: str = "direct"
    warnings: List[str] = field(default_factory=list)
    isotope_system_hint: str = ""
    # Extraction completion state declared by the writing tool: "completed",
    # "completed_with_errors", "cancelled" or "failed". Empty when the file
    # declares none — an older extractor output or another writer — which is
    # unknown, not a claim of completeness.
    completion_status: str = ""
    upstream_provenance_json: str = ""


def load_mc_results_from_archive(
    source: Union[str, Path, bytes],
) -> Dict[str, Dict[str, MCResultRecord]]:
    """Read the durable Monte Carlo records back out of a processed archive.

    Returns ``{archive sample ID: {ratio name: record}}``. Archive IDs are the
    collision-free ``s000001`` keys written in session order; sample names are
    not keys because distinct observations may legitimately share a name.
    The record itself retains the sample name and run number.

    This is deliberately a
    narrow, read-only accessor rather than a pipeline import: a processed
    archive must never be fed back through the correction chain as raw input
    (see :func:`load_hdf5`), but its Monte Carlo results must be re-readable so
    a saved run can be inspected and its identity verified.

    **Scope.** TraceISO does not reopen a processed archive as a working
    session, by design — :func:`load_hdf5` refuses one to prevent
    double-correction — so there is no application path that restores these
    records into live sample state, and none is intended. This function exists
    for archival inspection and external verification: reading a saved run's
    values, provenance and semantics back out of the file that carries them.
    Callers should treat the returned records as read-only evidence, not as
    state to re-attach to a session.

    Payloads are migrated through :meth:`MCResultRecord.from_dict`, so an
    archive written by an older schema is readable and is never relabelled as
    a fixed-draw SSB/delta model replay.
    """
    from io import BytesIO

    handle: object
    if isinstance(source, (bytes, bytearray)):
        handle = BytesIO(bytes(source))
    else:
        handle = str(Path(source))

    out: Dict[str, Dict[str, MCResultRecord]] = {}
    with h5py.File(handle, "r") as f:
        samples_grp = f.get("samples")
        if samples_grp is None:
            return out
        for sample_id in samples_grp:
            sample_grp = samples_grp[sample_id]
            mc_grp = sample_grp.get("mc_results")
            if mc_grp is None:
                continue
            sample_name = _decode_attr(sample_grp.attrs.get("name", sample_id))
            try:
                sample_run = int(sample_grp.attrs.get("run_number", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                sample_run = 0
            per_ratio: Dict[str, MCResultRecord] = {}
            for ratio_id in mc_grp:
                ratio_grp = mc_grp[ratio_id]
                ratio_name = _decode_attr(ratio_grp.attrs.get("name", ratio_id))
                raw = ratio_grp["record.json"][()]
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                record = MCResultRecord.from_dict(json.loads(raw))
                if record.ratio_name and record.ratio_name != ratio_name:
                    raise ValueError(
                        f"MC record {sample_id!r} contradicts its archive ratio: "
                        f"{record.ratio_name!r} != {ratio_name!r}."
                    )
                if record.sample_name and record.sample_name != sample_name:
                    raise ValueError(
                        f"MC record {sample_id!r} contradicts its archive sample: "
                        f"{record.sample_name!r} != {sample_name!r}."
                    )
                if (
                    record.sample_run_number
                    and sample_run
                    and record.sample_run_number != sample_run
                ):
                    raise ValueError(
                        f"MC record {sample_id!r} contradicts its archive run: "
                        f"{record.sample_run_number} != {sample_run}."
                    )
                per_ratio[ratio_name] = record
            if per_ratio:
                out[str(sample_id)] = per_ratio
    return out


def _decode_attr(value: object) -> str:
    """Return an HDF5 attribute as text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_hdf5(
    path: Union[str, Path],
    data_preference: str = "auto",
    *,
    max_file_size_mb: float = _DEFAULT_MAX_FILE_SIZE_MB,
    max_cycles_per_dataset: int = _DEFAULT_MAX_CYCLES_PER_DATASET,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
) -> HDF5LoadResult:
    """Load samples from an HDF5 file."""
    if data_preference not in _ALLOWED_DATA_PREFERENCES:
        allowed = ", ".join(sorted(_ALLOWED_DATA_PREFERENCES))
        raise ValueError(
            f"Invalid data_preference '{data_preference}'. "
            f"Expected one of: {allowed}."
        )

    path = Path(path)
    try:
        file_size_bytes = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"Unable to access HDF5 file '{path}': {exc}") from exc

    max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
    if file_size_bytes > max_file_size_bytes:
        raise ValueError(
            f"HDF5 file '{path}' is {file_size_bytes / (1024 * 1024):.1f} MB, "
            f"which exceeds the configured limit of {max_file_size_mb:.1f} MB."
        )

    warnings: List[str] = []
    all_isotopes: Set[str] = set()
    file_structure = "direct"
    element_hint = None
    completion_status = ""

    samples: List[Sample] = []
    group_errors: List[str] = []

    try:
        with h5py.File(str(path), "r") as f:
            file_kind = f.attrs.get("file_kind", "")
            if isinstance(file_kind, bytes):
                file_kind = file_kind.decode("utf-8", errors="replace")
            if str(file_kind) == "traceiso_processed_archive":
                raise ValueError(
                    "Processed TraceISO archive; re-import is not supported."
                )

            # File-level attribute for element
            upstream = f.attrs.get('upstream_provenance_json', '')
            upstream_sources = None
            if isinstance(upstream, bytes): upstream = upstream.decode('utf-8')
            if upstream:
                from config.upstream_provenance import validate
                upstream = validate(str(upstream))
                upstream_sources = {item['source_id']: item for item in json.loads(upstream)['sources']}
            elif 'provenance_json' in f.attrs:
                legacy = f.attrs['provenance_json']
                if isinstance(legacy, bytes): legacy = legacy.decode('utf-8')
                upstream = json.dumps({'schema': 'legacy_unverified', 'record': json.loads(legacy)})
            element_hint = f.attrs.get("isotope_system", None)
            if isinstance(element_hint, bytes):
                element_hint = element_hint.decode("utf-8")

            sample_group_names = [
                key
                for key in f.keys()
                if isinstance(f[key], h5py.Group) and "intensities" in f[key]
            ]
            if len(sample_group_names) > max_samples:
                raise ValueError(
                    f"HDF5 file '{path}' contains {len(sample_group_names)} sample groups, "
                    f"which exceeds the configured limit of {max_samples}."
                )

            group_structures = {
                group_name: _detect_group_structure(f[group_name]["intensities"])
                for group_name in sample_group_names
            }
            identity_ordinals = {
                group_name: ordinal
                for ordinal, group_name in enumerate(sample_group_names, start=1)
            }
            structures = set(group_structures.values())
            if len(structures) > 1:
                direct_groups = sorted(
                    name for name, structure in group_structures.items()
                    if structure == "direct"
                )
                nested_groups = sorted(
                    name for name, structure in group_structures.items()
                    if structure == "raw_corrected"
                )
                raise ValueError(
                    "Mixed HDF5 sample structures are unsupported: "
                    f"direct: {', '.join(direct_groups) or '(none)'}; "
                    f"raw_corrected: {', '.join(nested_groups) or '(none)'}."
                )
            if structures:
                file_structure = next(iter(structures))

            # Validate schema structure
            completion_status = read_completion_status(f)
            schema_warnings = _validate_schema(f)
            warnings.extend(schema_warnings)

            # Iterate over sample groups
            for group_name in f.keys():
                # Skip non-group objects (e.g. root datasets)
                obj = f[group_name]
                if not isinstance(obj, h5py.Group):
                    continue
                
                group = obj

                # Skip groups that don't look like samples (no intensities)
                if "intensities" not in group:
                    continue

                try:
                    metadata, meta_warnings = _load_metadata(group)
                    warnings.extend(meta_warnings)
                    if upstream_sources is not None:
                        source = upstream_sources.get(metadata.get('extractor_source_id'))
                        if source is None or metadata.get('source_sha256') != source['sha256']:
                            raise ValueError('Sample source identity does not match extractor provenance')
                        channels = metadata.get('channel_units', {})
                        from config.input_units import UNIT_CONTRACT, LEGACY_CONVENTION
                        if (channels.get('schema') != UNIT_CONTRACT
                                or channels.get('bare_number_convention') != LEGACY_CONVENTION
                                or channels.get('channels') != source['units']):
                            raise ValueError('Sample channel evidence does not match extractor provenance')
                        from config.input_units import parse_token
                        evidenced_datasets = set()
                        for channel, rows in channels['channels'].items():
                            is_ratio = '/' in channel or '_' in channel
                            dataset_name = ('ratios/' + channel.replace('/', '_')
                                            if is_ratio else 'intensities/' + channel)
                            ds = group.get(dataset_name)
                            evidenced_datasets.add(dataset_name)
                            if not isinstance(ds, h5py.Dataset) or len(ds) != len(rows):
                                raise ValueError('Channel conversion evidence has no matching dataset')
                            expected_units = 'dimensionless' if is_ratio else 'V'
                            if ds.attrs.get('unit_contract') != UNIT_CONTRACT or ds.attrs.get('units') != expected_units:
                                raise ValueError('Versioned channel lacks its canonical unit declaration')
                            values = _read_dataset(ds, max_cycles_per_dataset)
                            for index, row in enumerate(rows):
                                token = parse_token('1 ' + row['source_unit'], ratio=is_ratio)
                                if (row['canonical_unit'] != token.canonical_unit
                                        or row['scale'] != token.scale
                                        or row['status'] not in {token.status, 'missing'}):
                                    raise ValueError('Invalid channel conversion evidence')
                                if (row['status'] == 'missing') != bool(np.isnan(values[index])):
                                    raise ValueError('Channel missing-value evidence contradicts its data')
                        measured_datasets = {
                            f'{layer}/{name}'
                            for layer in ('intensities', 'ratios') if layer in group
                            for name, dataset in group[layer].items()
                            if isinstance(dataset, h5py.Dataset)
                            and not (layer == 'intensities' and name in {'Time', 'Cycle'})
                        }
                        if evidenced_datasets != measured_datasets:
                            raise ValueError('Channel conversion evidence does not cover every measured dataset')
                    metadata.setdefault("_source_file_name", path.name)
                    if upstream:
                        metadata['upstream_provenance_json'] = upstream

                    sample_type = _coerce_sample_type(
                        metadata,
                        group_name=group_name,
                        metadata_failed=bool(meta_warnings),
                        warnings=warnings,
                    )
                    raw_run_number = metadata.get("Run number")
                    run_number, run_number_status = _coerce_run_number_with_status(
                        raw_run_number,
                        missing="Run number" not in metadata,
                    )
                    if run_number_status != "valid":
                        metadata["_identity_ordinal"] = identity_ordinals[group_name]
                        metadata[f"_run_number_{run_number_status}"] = True
                        warning = (
                            f"Sample '{group_name}': "
                            f"{run_number_status} run number; scientific run number "
                            f"remains {run_number} and identity ordinal "
                            f"{identity_ordinals[group_name]} is used only for UI state."
                        )
                        warnings.append(warning)
                        _log.warning(warning)

                    intensities: Dict[str, CycleData] = {}
                    group_structure = group_structures[group_name]
                    raw_ints = _load_intensity_group(
                        group["intensities"],
                        data_preference,
                        group_structure,
                        max_cycles_per_dataset=max_cycles_per_dataset,
                    )
                    if len({arr.shape for arr in raw_ints.values()}) > 1:
                        raise ValueError("Intensity channels have incompatible cycle lengths.")
                    for iso, arr in raw_ints.items():
                        all_isotopes.add(iso)
                        intensities[iso] = _cycle_data_from_array(
                            arr,
                            sample_name=group_name,
                            channel_name=iso,
                            channel_kind="isotope",
                            warnings=warnings,
                        )

                    ratios: Dict[str, CycleData] = {}
                    if "ratios" in group:
                        raw_ratios = _load_ratio_group(
                            group["ratios"],
                            data_preference,
                            group_structure,
                            max_cycles_per_dataset=max_cycles_per_dataset,
                        )
                        for rname, arr in raw_ratios.items():
                            ratios[rname] = _cycle_data_from_array(
                                arr,
                                sample_name=group_name,
                                channel_name=rname,
                                channel_kind="ratio",
                                warnings=warnings,
                            )

                    sample = Sample(
                        name=group_name,
                        sample_type=sample_type,
                        run_number=run_number,
                        metadata=metadata,
                        intensities=intensities,
                        ratios=ratios,
                    )
                    samples.append(sample)
                except ValueError as exc:
                    message = f"Group '{group_name}' skipped: {exc}"
                    group_errors.append(message)
                    warnings.append(message)
    except (OSError, _HDF5_EXT_ERROR) as exc:
        raise ValueError(f"Failed to open HDF5 file '{path}': {exc}") from exc

    if group_errors:
        raise ValueError("Import refused; no partial session loaded. " + "; ".join(group_errors))

    # Sort samples by run_number
    samples.sort(key=lambda s: s.run_number)
    _validate_unique_sample_state_keys(samples)

    # Normalize isotope names (fix "25" → "25Mg" etc.)
    detected_elem_symbol = _detect_element_from_isotopes(all_isotopes)
    if detected_elem_symbol:
        mapping = _build_normalization_map(all_isotopes, detected_elem_symbol)
        if mapping:
            _apply_normalization(samples, mapping)
            all_isotopes = {mapping.get(iso, iso) for iso in all_isotopes}

    # Auto-detect ElementConfig
    detected_config = detect_element(list(all_isotopes))
    if detected_config is None and element_hint:
        try:
            from domain.elements.registry import get_element
            detected_config = get_element(element_hint)
        except ValueError:
            warnings.append(
                f"isotope_system '{element_hint}' has no registered TraceISO "
                "ElementConfig; this element cannot be processed by TraceISO's "
                "correction/uncertainty pipeline (data was extracted/exported "
                "but is not TraceISO-ready)."
            )

    # Normalize ratio names (e.g. "87Sr_86Sr", "Sr87/Sr86" -> "87Sr/86Sr")
    ratio_element = (
        detected_config.symbol
        if detected_config is not None
        else (detected_elem_symbol or str(element_hint or ""))
    )
    if ratio_element:
        ratio_warnings = _apply_ratio_name_normalization(samples, ratio_element)
        warnings.extend(ratio_warnings)

    return HDF5LoadResult(
        samples=samples,
        detected_isotopes=all_isotopes,
        detected_element=detected_config,
        element_symbol=detected_config.symbol if detected_config else "",
        file_structure=file_structure,
        warnings=warnings,
        isotope_system_hint=str(element_hint or ""),
        completion_status=completion_status,
        upstream_provenance_json=upstream,
    )


def _load_metadata(group: h5py.Group) -> Tuple[Dict, List[str]]:
    """Load metadata from a sample group."""
    if "metadata" not in group:
        return {}, []
    raw = group["metadata"][()]
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    else:
        text = str(raw)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result, []
        return {}, [
            f"Metadata parse failure in group '{group.name}': "
            "decoded metadata is not a JSON object. "
            "Sample type will default to SMP."
        ]
    except json.JSONDecodeError:
        try:
            result = ast.literal_eval(text)
            if not isinstance(result, dict):
                return {}, [
                    f"Metadata parse failure in group '{group.name}': "
                    "decoded metadata is not a mapping. "
                    "Sample type will default to SMP."
                ]
            _log.warning(
                "Group '%s': metadata could not be parsed as JSON; "
                "fell back to ast.literal_eval. "
                "Python-specific types (tuples, bare True/False/None) may be present.",
                group.name,
            )
            return result, []
        except Exception:
            return {}, [
                f"Metadata parse failure in group '{group.name}': "
                f"could not decode as JSON or Python literal. "
                f"Sample type will default to SMP."
            ]


def _coerce_sample_type(
    metadata: Dict,
    *,
    group_name: str,
    metadata_failed: bool,
    warnings: List[str],
) -> str:
    raw_type = str(metadata.get("Type") or "").strip().upper()
    sample_type = _TYPE_ALIASES.get(raw_type, raw_type)

    if not raw_type:
        if metadata_failed:
            metadata["_type_fallback"] = "metadata_parse_failure"
            warnings.append(
                f"Group '{group_name}': metadata unreadable — "
                f"classified as SMP by default. Verify sample role."
            )
        else:
            metadata["_type_fallback"] = "missing_type_field"
            warnings.append(
                f"Group '{group_name}': no 'Type' field in metadata — "
                f"classified as SMP by default."
            )
        return "SMP"

    if sample_type not in _VALID_TYPES:
        metadata["_type_fallback"] = f"unknown_type:{raw_type}"
        warnings.append(
            f"Group '{group_name}': unknown sample type '{raw_type}' — "
            f"classified as SMP by default. "
            f"Expected one of: {', '.join(sorted(_VALID_TYPES))}."
        )
        return "SMP"

    return sample_type


def _coerce_run_number(value: object) -> int:
    run_number, _status = _coerce_run_number_with_status(value)
    return run_number


def _coerce_run_number_with_status(
    value: object,
    *,
    missing: bool = False,
) -> Tuple[int, Literal["valid", "missing", "invalid"]]:
    """Return the scientific run number and metadata quality status."""
    if missing:
        return 0, "missing"
    if value is None:
        return 0, "invalid"
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return 0, "missing"
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0, "invalid"
    if isinstance(value, bool) or not np.isfinite(parsed) or not parsed.is_integer() or parsed < 0:
        return 0, "invalid"
    return int(parsed), "valid"


def _validate_unique_sample_state_keys(samples: List[Sample]) -> None:
    """Reject sample identities that would collide in session state."""
    by_key: Dict[str, List[str]] = {}
    for sample in samples:
        key = sample_cycle_key(sample)
        by_key.setdefault(key, []).append(sample.name)

    collisions = {
        key: names for key, names in by_key.items()
        if len(names) > 1
    }
    if collisions:
        details = "; ".join(
            f"{key}: {', '.join(names)}"
            for key, names in sorted(collisions.items())
        )
        raise ValueError(f"Duplicate sample state identities detected: {details}.")


def _detect_group_structure(
    int_grp: object,
) -> Literal["direct", "raw_corrected"]:
    """Classify one sample's intensities container."""
    if not isinstance(int_grp, h5py.Group):
        raise ValueError("Malformed intensities entry: expected an HDF5 group.")

    has_nested_keys = "raw" in int_grp or "corrected" in int_grp
    if not has_nested_keys:
        return "direct"

    extra_dataset_keys = [
        key
        for key in int_grp.keys()
        if key not in {"raw", "corrected"} and isinstance(int_grp[key], h5py.Dataset)
    ]
    if extra_dataset_keys:
        raise ValueError(
            f"Malformed intensities group '{int_grp.name}': direct datasets cannot be "
            "mixed with 'raw'/'corrected' subgroups "
            f"(found: {', '.join(sorted(extra_dataset_keys))})."
        )
    return "raw_corrected"


def _cycle_data_from_array(
    values: np.ndarray,
    *,
    sample_name: str,
    channel_name: str,
    channel_kind: str,
    warnings: List[str],
) -> CycleData:
    """Create finite-masked CycleData and report rejected file cycles."""
    array = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(array)
    n_bad = int(np.sum(~finite_mask))
    if n_bad:
        message = (
            f"Sample '{sample_name}' {channel_kind} '{channel_name}': "
            f"{n_bad} non-finite cycle(s) masked at ingest."
        )
        warnings.append(message)
        _log.warning(message)
    return CycleData(values=array, mask=finite_mask)


def _load_intensity_group(
    int_grp: h5py.Group,
    preference: str,
    structure: str,
    *,
    max_cycles_per_dataset: int,
) -> Dict[str, np.ndarray]:
    """Load intensity datasets from an HDF5 group, skipping metadata helpers."""
    result: Dict[str, np.ndarray] = {}

    if structure == "raw_corrected":
        subgroup = _select_raw_corrected_subgroup(
            int_grp,
            preference=preference,
            group_label="intensities",
            require_subgroups=True,
        )
        for iso in subgroup.keys():
            if not _is_isotope_dataset(iso):
                _log.debug("Skipping non-isotope key '%s' in raw_corrected intensities.", iso)
                continue
            result[iso] = _read_dataset(subgroup[iso], max_cycles_per_dataset)
    else:
        # Direct structure
        for iso in int_grp.keys():
            if not _is_isotope_dataset(iso):
                _log.debug("Skipping non-isotope key '%s' in direct intensities.", iso)
                continue
            ds = int_grp[iso]
            if not isinstance(ds, h5py.Dataset):
                raise ValueError(
                    f"Malformed intensities group '{int_grp.name}': expected dataset "
                    f"for isotope '{iso}', found HDF5 group."
                )
            result[iso] = _read_dataset(ds, max_cycles_per_dataset)

    return result


def _load_ratio_group(
    ratio_grp: h5py.Group,
    preference: str,
    structure: str,
    *,
    max_cycles_per_dataset: int,
) -> Dict[str, np.ndarray]:
    """Load ratio datasets from an HDF5 group."""
    result: Dict[str, np.ndarray] = {}

    if structure == "raw_corrected":
        has_raw_key = "raw" in ratio_grp
        has_corr_key = "corrected" in ratio_grp
        if not has_raw_key and not has_corr_key:
            # Ratios are optional; an empty ratios group is acceptable.
            if len(ratio_grp.keys()) == 0:
                return result
            raise ValueError(
                f"Malformed ratios group '{ratio_grp.name}': expected 'raw'/'corrected' "
                "subgroups for raw_corrected file structure."
            )

        subgroup = _select_raw_corrected_subgroup(
            ratio_grp,
            preference=preference,
            group_label="ratios",
            require_subgroups=False,
        )
        if subgroup is not None:
            for rname in subgroup.keys():
                result[rname] = _read_dataset(subgroup[rname], max_cycles_per_dataset)
    else:
        for rname in ratio_grp.keys():
            ds = ratio_grp[rname]
            if not isinstance(ds, h5py.Dataset):
                raise ValueError(
                    f"Malformed ratios group '{ratio_grp.name}': expected dataset "
                    f"for ratio '{rname}', found HDF5 group."
                )
            result[rname] = _read_dataset(ds, max_cycles_per_dataset)

    return result


def _read_dataset(ds: h5py.Dataset, max_cycles_per_dataset: int) -> np.ndarray:
    """Read a dataset after validating its shape, dtype, and cycle count."""
    units = ds.attrs.get("units", ds.attrs.get("unit", ""))
    if isinstance(units, bytes):
        units = units.decode("utf-8")
    units = str(units).strip().lower()
    contract = ds.attrs.get('unit_contract', '')
    if isinstance(contract, bytes):
        contract = contract.decode('utf-8')
    if contract:
        from config.input_units import UNIT_CONTRACT
        if contract != UNIT_CONTRACT or not units:
            raise ValueError(f"Dataset {ds.name!r}: unknown unit contract or missing canonical units")
    if units:
        intensity = "/intensities/" in ds.name
        allowed = {"v", "volt", "volts"} if intensity else {"1", "ratio", "dimensionless"}
        if units not in allowed:
            raise ValueError(f"Dataset {ds.name!r}: unsupported declared units {units!r}; convert explicitly before import.")
    if ds.shape is None or len(ds.shape) != 1:
        raise ValueError(
            f"Dataset '{ds.name}' has shape {ds.shape}; "
            "expected a 1-D numeric array."
        )
    if np.issubdtype(ds.dtype, np.complexfloating):
        raise ValueError(
            f"Dataset '{ds.name}' has complex dtype '{ds.dtype}'; "
            "TraceISO ingests real-valued intensity/ratio data only "
            "(a complex dataset would silently lose its imaginary component "
            "when cast to float64)."
        )
    if not np.issubdtype(ds.dtype, np.number):
        raise ValueError(
            f"Dataset '{ds.name}' has dtype '{ds.dtype}'; "
            "expected a numeric dtype."
        )
    cycle_count = int(ds.shape[0])
    if cycle_count > max_cycles_per_dataset:
        raise ValueError(
            f"Dataset '{ds.name}' contains {cycle_count} cycles, "
            f"which exceeds the configured limit of {max_cycles_per_dataset}."
        )
    return ds[:]


def _validate_raw_corrected_subgroup(
    parent_grp: h5py.Group,
    subgroup_name: str,
    *,
    group_label: str,
) -> Optional[h5py.Group]:
    """Return a validated raw/corrected subgroup or raise on malformed structure."""
    if subgroup_name not in parent_grp:
        return None

    obj = parent_grp[subgroup_name]
    if not isinstance(obj, h5py.Group):
        raise ValueError(
            f"Malformed {group_label} group '{parent_grp.name}': expected subgroup "
            f"'{subgroup_name}' to be an HDF5 group."
        )

    for child_name in obj.keys():
        child = obj[child_name]
        if not isinstance(child, h5py.Dataset):
            raise ValueError(
                f"Malformed {group_label} subgroup '{obj.name}': expected dataset "
                f"children only, but found non-dataset '{child_name}'."
            )

    return obj


def _select_raw_corrected_subgroup(
    parent_grp: h5py.Group,
    *,
    preference: str,
    group_label: str,
    require_subgroups: bool,
) -> Optional[h5py.Group]:
    """Select the raw/corrected subgroup for a validated raw_corrected container."""
    raw_grp = _validate_raw_corrected_subgroup(
        parent_grp,
        "raw",
        group_label=group_label,
    )
    corr_grp = _validate_raw_corrected_subgroup(
        parent_grp,
        "corrected",
        group_label=group_label,
    )

    if raw_grp is None and corr_grp is None:
        if require_subgroups:
            raise ValueError(
                f"Malformed {group_label} group '{parent_grp.name}': expected 'raw'/'corrected' "
                "subgroups for raw_corrected file structure."
            )
        return None

    has_raw = raw_grp is not None and len(raw_grp.keys()) > 0
    has_corr = corr_grp is not None and len(corr_grp.keys()) > 0

    subgroup = None
    if preference == "raw" and has_raw:
        subgroup = raw_grp
    elif preference == "corrected" and has_corr:
        subgroup = corr_grp
    elif preference == "auto":
        subgroup = corr_grp if has_corr else (raw_grp if has_raw else None)
    else:
        subgroup = raw_grp if has_raw else (corr_grp if has_corr else None)

    if subgroup is None and (raw_grp is not None or corr_grp is not None):
        raise ValueError(
            f"Malformed {group_label} group '{parent_grp.name}': raw/corrected structure "
            "exists but contains no datasets."
        )

    return subgroup


def _detect_element_from_isotopes(isotopes: Set[str]) -> Optional[str]:
    """Detect element symbol from properly formatted isotope names."""
    counts: Counter = Counter()
    for iso in isotopes:
        m = re.match(r"^\d+([A-Z][a-z]*)$", str(iso))
        if m:
            counts[m.group(1)] += 1
    if counts:
        return counts.most_common(1)[0][0]
    return None


def _build_normalization_map(
    isotopes: Set[str], element: str,
) -> Dict[str, str]:
    """Build mapping of malformed isotope names → normalized names."""
    mapping: Dict[str, str] = {}
    for iso in isotopes:
        normalized = _normalize_isotope_name(str(iso), element)
        if normalized != iso:
            mapping[iso] = normalized
    return mapping


def _normalize_isotope_name(name: str, element: str) -> str:
    """Normalize a single isotope name to '##Element' format."""
    name = name.strip()

    # Already correct: "25Mg"
    if re.match(r"^\d+[A-Z][a-z]*$", name):
        return name

    # Just number: "25" → "25Mg"
    if name.isdigit():
        return f"{name}{element}"

    # Reverse format: "Mg25" → "25Mg"
    m = re.match(r"^([A-Z][a-z]*)(\d+)$", name)
    if m:
        return f"{m.group(2)}{m.group(1)}"

    nums = re.findall(r"\d+", name)
    if nums:
        _log.warning(
            "Isotope name %r fallback normalized to %r%s",
            name, nums[0], element
        )
        return f"{nums[0]}{element}"

    return name


def _normalize_ratio_token(token: str, element: str) -> str:
    """Normalize one isotope token within a ratio name."""
    token = token.strip()
    element = element.strip()

    # Already "87Sr" style
    if re.match(r"^\d+[A-Z][a-z]*$", token):
        return token

    # "Sr87" -> "87Sr"
    m = re.match(r"^([A-Z][a-z]*)(\d+)$", token)
    if m:
        return f"{m.group(2)}{m.group(1)}"

    # "87" -> "87Sr" when element is known
    if token.isdigit() and element:
        return f"{token}{element}"

    # Fallback: keep as-is
    return token


def _normalize_ratio_name(name: str, element: str) -> str:
    """Normalize a ratio name to canonical 'num/den' format."""
    raw = str(name).strip()

    # Split once using common separators used in files/UI aliases
    parts = re.split(r"[\\/ _]", raw)
    parts = [p for p in parts if p]
    if len(parts) != 2:
        return raw

    numerator = _normalize_ratio_token(parts[0], element)
    denominator = _normalize_ratio_token(parts[1], element)
    return f"{numerator}/{denominator}"


def _apply_ratio_name_normalization(
    samples: List[Sample],
    element: str,
) -> List[str]:
    """Normalize sample ratio keys in-place and report collisions."""
    warnings: List[str] = []

    for sample in samples:
        if not sample.ratios:
            continue

        normalized_ratios: Dict[str, CycleData] = {}
        collisions: List[str] = []

        for old_key, cd in sample.ratios.items():
            new_key = _normalize_ratio_name(old_key, element)
            if new_key in normalized_ratios:
                collisions.append(old_key)
                continue
            normalized_ratios[new_key] = cd

        sample.ratios = normalized_ratios

        if collisions:
            warnings.append(
                f"Sample '{sample.name}': duplicate ratio aliases collapsed to canonical names "
                f"(dropped: {', '.join(collisions)})."
            )

    return warnings


def _apply_normalization(
    samples: List[Sample], mapping: Dict[str, str],
) -> None:
    """Apply isotope name normalization to all samples in-place."""
    # Validate the complete proposed mapping before mutating any observation.
    for sample in samples:
        for layer in (sample.intensities, sample.corrected_intensities):
            seen = {}
            for old in layer:
                canonical = mapping.get(old, old)
                if canonical in seen:
                    raise ValueError(f"Sample '{sample.name}': ambiguous isotope aliases {seen[canonical]!r} and {old!r} map to {canonical!r}.")
                seen[canonical] = old
    for sample in samples:
        # Rename intensities keys
        new_ints: Dict[str, CycleData] = {}
        for old, cd in sample.intensities.items():
            new_name = mapping.get(old, old)
            new_ints[new_name] = cd
        sample.intensities = new_ints

        if sample.corrected_intensities:
            new_corr: Dict[str, CycleData] = {}
            for old, cd in sample.corrected_intensities.items():
                new_name = mapping.get(old, old)
                new_corr[new_name] = cd
            sample.corrected_intensities = new_corr


_COMPLETED_EXTRACTION_STATUSES = frozenset({"completed", "completed_with_errors"})


def read_completion_status(f: h5py.File) -> str:
    """Return the extraction completion state a file declares, or ``""``.

    An empty result means the file declares none — an output from before the
    extractor stamped it, or from another writer. That is *unknown*, and is
    deliberately not reported as complete.
    """
    raw = f.attrs.get("completion_status", "")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return str(raw).strip()


def _completion_warnings(f: h5py.File, status: str) -> List[str]:
    """Describe an extraction that did not run to the end of its file list."""
    if not status or status in _COMPLETED_EXTRACTION_STATUSES:
        return []

    def _count(name: str) -> Optional[int]:
        value = f.attrs.get(name)
        return None if value is None else int(value)

    intended = _count("files_intended")
    converted = _count("files_converted")
    unprocessed = _count("files_unprocessed")

    scope = ""
    if intended is not None and converted is not None:
        scope = f" after converting {converted} of {intended} source file(s)"
    tail = ""
    if unprocessed:
        tail = f"; {unprocessed} file(s) were never converted"

    if status == "cancelled":
        reason = "was cancelled by the user"
    elif status == "failed":
        reason = "stopped on an unrecoverable error"
    else:
        reason = f"reports completion status '{status}'"
    return [
        f"Incomplete extraction: the batch that produced this file {reason}"
        f"{scope}{tail}. This session is partial — it is not the full run."
    ]


def _validate_schema(f: h5py.File) -> List[str]:
    """Validate HDF5 file schema and return a list of warnings."""
    warnings = []

    warnings.extend(_completion_warnings(f, read_completion_status(f)))

    # We expect 'isotope_system' or at least some identifier.
    # The new extractor adds 'schema_version', 'provenance_json'.
    if "isotope_system" not in f.attrs:
        warnings.append("Missing root attribute 'isotope_system'. Element detection may fail.")


    has_valid_sample = False
    for key in f.keys():
        obj = f[key]
        if isinstance(obj, h5py.Group) and "intensities" in obj:
            has_valid_sample = True
            break
    
    if not has_valid_sample:
        warnings.append("No valid sample groups found (must contain 'intensities' group).")

    return warnings

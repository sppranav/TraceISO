"""HDF5 export for TraceISO — the ``traceiso.processed.v1`` archive.

This writer produces a self-describing, versioned *processed-data archive*: raw
cycle arrays, every correction layer, masks, uncertainty budgets, and full
provenance, readable by any HDF5 client (h5py, MATLAB, R, Julia, HDFView).

Design rules that keep the archive safe and lossless:

- **Stable IDs as paths.** Ratio names contain ``/`` (e.g. ``87Sr/86Sr``),
  which HDF5 reads as a path separator. Samples/channels/ratios are written as
  ``s000001``/``c0001``/``r0001`` groups carrying the real name in a ``name``
  attribute, so no name can inject a nested group or collide.
- **Mask is authoritative.** Every ``CycleData`` stores both ``values``
  (float64, NaN preserved natively) and ``mask`` (uint8). Validity lives in the
  mask, never in dropped values.
- **Hybrid native + JSON, split by shape.** Every *per-cycle numeric series*
  (intensity/ratio layers, SSB-corrected cycles, per-cycle delta, the pre-anchor
  Sr ratio) is a native typed dataset so NaN positions and cycle alignment
  survive losslessly. Only *scalars, labels, nested fit/bracket details, and the
  uncertainty budget* — irregular, schema-evolving data with no NaN-bearing
  cycle array — are stored as NaN-safe UTF-8 JSON datasets. No pickle. This
  split is deliberate: routing a per-cycle array through JSON would coerce its
  NaN cycles to ``null`` and silently change its float semantics.
- **Root identity.** ``file_kind``/``schema_version`` are written so a future
  loader can distinguish a processed archive from raw instrument input and never
  feed corrected arrays back through the pipeline as raw (double-correction).
- **Monte Carlo results.** A completed cross-check is written under
  ``samples/<id>/mc_results/r<NNNN>/record.json`` with its ratio name,
  execution ID, semantics version, result space and freshness as group
  attributes, and the record schema named once at the root
  (``mc_result_schema_name``/``mc_result_schema_version``). The payload is
  scalar/label metadata, so it belongs on the JSON side of the hybrid split;
  it carries no draw array. :func:`file_io.hdf5_reader.load_mc_results_from_archive`
  reads it back. The ``freshness`` attribute states whether the stored result
  still matched the configuration and budget in this archive when it was
  written; see :func:`domain.uncertainty.mc_result.mc_record_freshness`.

This module is pure: it imports no Streamlit and takes a plain ``provenance``
dict rather than reading session state.

The schema constants and writer implementation in this module are the
authoritative definition of the processed archive layout.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, Optional

import h5py
import numpy as np

from config.constants import APP_VERSION
from domain.models import CycleData, ProcessingResult, Sample, UncertaintyBudget
from domain.uncertainty.mc_result import (
    MC_RESULT_SCHEMA_NAME,
    MC_RESULT_SCHEMA_VERSION,
    has_mc_results,
    log_mc_results_exported,
    mc_record_freshness_for_budget,
)
from domain.uncertainty.scope import uncertainty_scope_payload
from domain.provenance import normalize_provenance

# Provenance carries the SHA-1 upload hash retained by the application. A
# SHA-256 source hash would need to be captured at upload time in ui/sidebar.py
# because the original bytes are discarded after loading.

FILE_KIND = "traceiso_processed_archive"
SCHEMA_NAME = "traceiso.processed"
#: ``1.1`` adds the Hg interference-corrected intensity and ratio layers and the
#: per-sample ``correction_records.json``. A ``1.0`` archive carries neither; it
#: is read as holding no Hg records, never widened into a ``1.1`` archive.
#: ``1.2`` adds, when present, ``ratios/pb_standard_corrected`` (Pb-standard
#: calibrated final ratios), ``delta_permil/pb_calibrated`` (calibrated delta
#: cycles) and the ``pb_calibration`` / ``pb_calibrated_delta`` record families.
#: A ``1.1`` archive may carry Hg records only.
#: ``1.3`` adds the separate ``ratios/sr_standard_corrected`` layer.
SCHEMA_VERSION = "1.3"
SCHEMA_READABLE_VERSIONS = ("1.0", "1.1", "1.2", "1.3")

#: Format token used in the ``engine_b_mc.exported`` lifecycle record. It is a
#: fixed word, never a filename or a path.
_MC_EXPORT_FORMAT = "hdf5"

_STR_DTYPE = h5py.string_dtype(encoding="utf-8")

# Private metadata keys promoted out of the JSON internal-state dump because
# they carry a NaN-bearing cycle array (the pre-anchor Sr ratio). See
# domain/pipeline.py session-anchoring step.
_PRE_ANCHOR_RATIO_KEY = "_pre_anchor_ratio"
_PRE_ANCHOR_RATIO_NAME_KEY = "_pre_anchor_ratio_name"

# Intensity layers written under samples/<id>/intensities/<layer>/
_INTENSITY_LAYERS = (
    ("raw", "intensities"),
    ("blank_corrected", "blank_corrected_intensities"),
    ("corrected", "corrected_intensities"),
    ("interference_corrected", "interference_corrected_intensities"),
)

# Ratio layers written under samples/<id>/ratios/<layer>/
_RATIO_LAYERS = (
    ("raw", "ratios"),
    ("blank_corrected", "blank_corrected_ratios"),
    ("corrected", "corrected_ratios"),
    ("iif_corrected", "iif_corrected_ratios"),
    ("drift_corrected", "drift_corrected_ratios"),
    ("interference_corrected", "interference_corrected_ratios"),
)


def export_to_hdf5(
    result: ProcessingResult,
    *,
    provenance: Optional[Dict[str, Any]] = None,
    include_raw: bool = True,
    include_uncertainty: bool = True,
    compression: Optional[str] = "gzip",
    compression_level: int = 4,
    uncertainty_config: Optional[object] = None,
) -> bytes:
    """Serialize a ``ProcessingResult`` to a ``traceiso.processed.v1`` archive.

    Returns the raw HDF5 file bytes (built in memory), suitable for a Streamlit
    download. ``include_raw=False`` omits the raw intensity/ratio layers;
    ``include_uncertainty=False`` omits per-ratio budgets. Compression is applied
    only to non-empty numeric arrays (HDF5 rejects chunked filters on empty data).
    """
    bio = BytesIO()
    with h5py.File(bio, "w") as f:
        f.attrs["file_kind"] = FILE_KIND
        f.attrs["schema_name"] = SCHEMA_NAME
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["app_version"] = str(APP_VERSION)
        f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        f.attrs["isotope_system"] = result.element_symbol or ""

        prov_grp = f.create_group("provenance")
        _write_json(prov_grp, "export.json", normalize_provenance(provenance))
        if include_uncertainty and any(sample.uncertainty for sample in result.samples):
            _write_json(
                prov_grp,
                "uncertainty_scope.json",
                uncertainty_scope_payload(),
            )

        session_grp = f.create_group("session")
        _write_string_array(session_grp, "warnings", result.warnings)
        _write_json(session_grp, "quality_metrics.json", result.quality_metrics)

        samples_grp = f.create_group("samples")
        sample_ids = []
        for idx, sample in enumerate(result.samples, start=1):
            sample_id = f"s{idx:06d}"
            sample_ids.append(sample_id)
            _write_sample(
                samples_grp.create_group(sample_id),
                sample,
                include_raw=include_raw,
                include_uncertainty=include_uncertainty,
                compression=compression,
                compression_level=compression_level,
                uncertainty_config=uncertainty_config,
            )
        _write_string_array(session_grp, "sample_order", sample_ids)

        if include_uncertainty and has_mc_results(result.samples):
            # Stated once at the root so a reader can identify the per-ratio
            # Monte Carlo payloads without opening a sample group.
            f.attrs["mc_result_schema_name"] = MC_RESULT_SCHEMA_NAME
            f.attrs["mc_result_schema_version"] = MC_RESULT_SCHEMA_VERSION

    if include_uncertainty and has_mc_results(result.samples):
        log_mc_results_exported(result.samples, export_format=_MC_EXPORT_FORMAT)

    return bio.getvalue()


def _write_sample(
    grp: h5py.Group,
    sample: Sample,
    *,
    include_raw: bool,
    include_uncertainty: bool,
    compression: Optional[str],
    compression_level: int,
    uncertainty_config: Optional[object] = None,
) -> None:
    """Write one sample group: attrs, metadata, all layers, derived, budgets."""
    grp.attrs["name"] = "" if sample.name is None else sample.name
    grp.attrs["sample_type"] = "" if sample.sample_type is None else sample.sample_type
    grp.attrs["run_number"] = int(sample.run_number or 0)
    # Session-scoped observation identity. Distinct from name and run number,
    # both of which may repeat across observations in one session.
    grp.attrs["observation_id"] = str(sample.observation_id or "")
    # Sample has no `excluded` field; exclusion is a UI concept carried in
    # metadata when present, defaulting to False at the domain level.
    grp.attrs["excluded"] = bool((sample.metadata or {}).get("excluded", False))

    _write_metadata(
        grp,
        sample.metadata,
        compression=compression,
        compression_level=compression_level,
    )
    _write_json(grp, "used_blanks.json", sample.used_blanks)
    _write_json(grp, "used_blank_ids.json", sample.used_blank_ids)
    if getattr(sample, "correction_records", None):
        # Frozen records are finite scalar/label evidence, so they sit on the
        # JSON side of the archive; strict readers reject unknown versions.
        from domain.pb_correction_records import correction_records_payload

        _write_json(grp, "correction_records.json", correction_records_payload(sample))
    _write_string_array(grp, "warnings", sample.warnings)

    intensities_grp = grp.create_group("intensities")
    for layer_name, attr in _INTENSITY_LAYERS:
        if layer_name == "raw" and not include_raw:
            continue
        _write_channel_layer(
            intensities_grp.create_group(layer_name),
            getattr(sample, attr),
            prefix="c",
            compression=compression,
            compression_level=compression_level,
        )

    ratios_grp = grp.create_group("ratios")
    for layer_name, attr in _RATIO_LAYERS:
        if layer_name == "raw" and not include_raw:
            continue
        _write_channel_layer(
            ratios_grp.create_group(layer_name),
            getattr(sample, attr),
            prefix="r",
            compression=compression,
            compression_level=compression_level,
        )
    # Written only when present, so an archive of any other route keeps the
    # group layout it had before calibration existed.
    if getattr(sample, "sr_standard_corrected_ratios", None):
        _write_channel_layer(
            ratios_grp.create_group("sr_standard_corrected"),
            sample.sr_standard_corrected_ratios,
            prefix="r",
            compression=compression,
            compression_level=compression_level,
        )
    if getattr(sample, "pb_standard_corrected_ratios", None):
        _write_channel_layer(
            ratios_grp.create_group("pb_standard_corrected"),
            sample.pb_standard_corrected_ratios,
            prefix="r",
            compression=compression,
            compression_level=compression_level,
        )
    if getattr(sample, "pb_calibrated_delta_cycles", None):
        _write_channel_layer(
            grp.create_group("delta_permil").create_group("pb_calibrated"),
            sample.pb_calibrated_delta_cycles,
            prefix="r",
            compression=compression,
            compression_level=compression_level,
        )

    _write_derived(
        grp,
        sample,
        compression=compression,
        compression_level=compression_level,
    )

    if include_uncertainty:
        unc_grp = grp.create_group("uncertainty")
        for idx, (ratio_name, budget) in enumerate(sample.uncertainty.items(), start=1):
            ratio_grp = unc_grp.create_group(f"r{idx:04d}")
            ratio_grp.attrs["name"] = ratio_name
            _write_json(ratio_grp, "budget.json", _budget_to_dict(budget))

        # Durable Monte Carlo cross-check records, one JSON dataset per ratio.
        # The record is scalar/label metadata with no NaN-bearing cycle array,
        # so it belongs on the JSON side of this archive's hybrid split. Its
        # own encoding already carries non-finite values losslessly, so it is
        # written verbatim rather than through ``_json_safe``.
        mc_results = getattr(sample, "mc_results", None) or {}
        if mc_results:
            mc_grp = grp.create_group("mc_results")
            mc_grp.attrs["schema_name"] = MC_RESULT_SCHEMA_NAME
            mc_grp.attrs["schema_version"] = MC_RESULT_SCHEMA_VERSION
            for idx, (ratio_name, record) in enumerate(mc_results.items(), start=1):
                ratio_grp = mc_grp.create_group(f"r{idx:04d}")
                ratio_grp.attrs["name"] = ratio_name
                ratio_grp.attrs["execution_id"] = record.execution_id
                ratio_grp.attrs["semantics_version"] = record.semantics_version
                ratio_grp.attrs["result_space"] = record.effective_result_space
                # Freshness relates the stored record to the budget written in
                # the same archive, so it is a group attribute rather than a
                # field inside the canonical payload.
                ratio_grp.attrs["freshness"] = mc_record_freshness_for_budget(
                    record,
                    (getattr(sample, "uncertainty", None) or {}).get(ratio_name),
                    uncertainty_config,
                    sample=sample,
                )
                data = json.dumps(record.to_dict(), allow_nan=False)
                ratio_grp.create_dataset("record.json", data=data, dtype=_STR_DTYPE)


def _write_channel_layer(
    grp: h5py.Group,
    mapping: Dict[str, CycleData],
    *,
    prefix: str,
    compression: Optional[str],
    compression_level: int,
) -> None:
    """Write a dict of named ``CycleData`` as stable-ID child groups."""
    for idx, (name, cd) in enumerate(mapping.items(), start=1):
        child = grp.create_group(f"{prefix}{idx:04d}")
        child.attrs["name"] = name
        _write_array(
            child,
            "values",
            np.asarray(cd.values, dtype=np.float64),
            compression=compression,
            compression_level=compression_level,
        )
        _write_array(
            child,
            "mask",
            np.asarray(cd.mask, dtype=np.uint8),
            compression=compression,
            compression_level=compression_level,
        )


def _write_derived(
    grp: h5py.Group,
    sample: Sample,
    *,
    compression: Optional[str],
    compression_level: int,
) -> None:
    """Write SSB and delta results as native per-cycle arrays plus JSON detail.

    Each derived ratio entry carries one per-cycle numeric series (the
    SSB-corrected cycles or the per-cycle delta). That series is written as a
    native ``values``/``mask`` pair so its NaN cycles survive; every remaining
    scalar/label field (bracket names, k-factor, delta summary statistics) goes
    to a sibling JSON dataset.
    """
    derived_grp = grp.create_group("derived")

    ssb_grp = derived_grp.create_group("ssb")
    for idx, (ratio_name, entry) in enumerate(sample.ssb_results.items(), start=1):
        child = ssb_grp.create_group(f"r{idx:04d}")
        child.attrs["name"] = ratio_name
        _write_cycle_series(
            child,
            entry,
            values_key="ssb_corrected_cycles",
            mask_key="ssb_mask",
            details_name="details.json",
            compression=compression,
            compression_level=compression_level,
        )

    delta_grp = derived_grp.create_group("delta")
    for idx, (ratio_name, entry) in enumerate(sample.delta_results.items(), start=1):
        child = delta_grp.create_group(f"r{idx:04d}")
        child.attrs["name"] = ratio_name
        _write_cycle_series(
            child,
            entry,
            values_key="delta_per_cycle",
            mask_key="delta_mask",
            details_name="summary.json",
            compression=compression,
            compression_level=compression_level,
        )


def _write_cycle_series(
    grp: h5py.Group,
    entry: Dict[str, Any],
    *,
    values_key: str,
    mask_key: str,
    details_name: str,
    compression: Optional[str],
    compression_level: int,
) -> None:
    """Write one derived entry: native ``values``/``mask`` for the per-cycle
    series (when present) and a JSON dataset for all remaining fields.

    The per-cycle array may be absent (e.g. delta computed without per-cycle
    output); only the JSON detail is written in that case.
    """
    values = entry.get(values_key)
    if values is not None:
        _write_array(
            grp,
            "values",
            np.asarray(values, dtype=np.float64),
            compression=compression,
            compression_level=compression_level,
        )
    mask = entry.get(mask_key)
    if mask is not None:
        _write_array(
            grp,
            "mask",
            np.asarray(mask, dtype=np.uint8),
            compression=compression,
            compression_level=compression_level,
        )
    details = {k: v for k, v in entry.items() if k not in (values_key, mask_key)}
    _write_json(grp, details_name, details)


def _write_metadata(
    grp: h5py.Group,
    metadata: Optional[Dict[str, Any]],
    *,
    compression: Optional[str],
    compression_level: int,
) -> None:
    """Split sample metadata into documented public fields and internal state.

    Public keys (no leading underscore) are written to ``metadata.json``.
    Private pipeline keys (leading underscore) are never silently merged into
    that dump; they go to a clearly labelled ``internal_state/`` group. The
    pre-anchor Sr ratio — a ``CycleData`` with NaN-bearing cycles — is promoted
    to a native ``values``/``mask`` dataset rather than flattened to JSON, so its
    float semantics and cycle alignment are preserved.
    """
    metadata = metadata or {}
    public = {k: v for k, v in metadata.items() if not k.startswith("_")}
    private = {k: v for k, v in metadata.items() if k.startswith("_")}

    _write_json(grp, "metadata.json", public)

    if not private:
        return

    internal_grp = grp.create_group("internal_state")
    pre_anchor = private.pop(_PRE_ANCHOR_RATIO_KEY, None)
    if isinstance(pre_anchor, CycleData):
        pa_grp = internal_grp.create_group("pre_anchor_ratio")
        pa_grp.attrs["name"] = str(private.get(_PRE_ANCHOR_RATIO_NAME_KEY, ""))
        _write_array(
            pa_grp,
            "values",
            np.asarray(pre_anchor.values, dtype=np.float64),
            compression=compression,
            compression_level=compression_level,
        )
        _write_array(
            pa_grp,
            "mask",
            np.asarray(pre_anchor.mask, dtype=np.uint8),
            compression=compression,
            compression_level=compression_level,
        )
    _write_json(internal_grp, "internal_state.json", private)


def _write_array(
    grp: h5py.Group,
    name: str,
    arr: np.ndarray,
    *,
    compression: Optional[str],
    compression_level: int,
) -> None:
    """Create a numeric dataset, compressing only non-empty arrays.

    HDF5 forbids chunked-filter pipelines (gzip/shuffle) on zero-length data, so
    empty channels fall back to a plain contiguous dataset.
    """
    if arr.size > 0 and compression:
        grp.create_dataset(
            name,
            data=arr,
            compression=compression,
            compression_opts=compression_level,
            shuffle=True,
        )
    else:
        grp.create_dataset(name, data=arr)


def _write_json(grp: h5py.Group, name: str, obj: Any) -> None:
    """Store *obj* as a NaN-safe UTF-8 JSON scalar dataset."""
    payload = json.dumps(_json_safe(obj), allow_nan=False)
    grp.create_dataset(name, data=payload, dtype=_STR_DTYPE)


def _write_string_array(grp: h5py.Group, name: str, items) -> None:
    """Store a list of strings as a 1-D UTF-8 dataset (empty-safe)."""
    values = [str(x) for x in (items or [])]
    grp.create_dataset(name, shape=(len(values),), data=values, dtype=_STR_DTYPE)


def _budget_to_dict(budget: UncertaintyBudget) -> Dict[str, Any]:
    """Convert an ``UncertaintyBudget`` (incl. contributors) to a JSON-safe dict."""
    if is_dataclass(budget):
        return _json_safe(asdict(budget))
    return _json_safe(budget)


def _json_safe(obj: Any) -> Any:
    """Recursively coerce *obj* into JSON-serializable types.

    NumPy scalars/arrays become Python types; non-finite floats become ``None``;
    sets become sorted lists; dataclasses are expanded; anything else falls back
    to ``str`` so a serializer error can never abort the export.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return [_json_safe(x) for x in sorted(obj, key=str)]
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    return str(obj)

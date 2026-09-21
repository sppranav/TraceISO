"""Read-only access to the correction records in a processed TraceISO archive.

Like :func:`file_io.hdf5_reader.load_mc_results_from_archive`, this is archival
inspection, not a working-session reopen: a processed archive is never fed back
through the correction chain. Records are rebuilt through the strict record
parser, so an unknown record version or family is rejected rather than guessed.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Union

import h5py

from domain.pb_correction_records import HG_RECORD_FAMILY, correction_records_from_payload
from file_io.hdf5_writer import FILE_KIND, SCHEMA_READABLE_VERSIONS

_RECORDS_DATASET = "correction_records.json"
#: Record families each archive version may hold; ``None`` means every family
#: this build reads.
_FAMILIES_BY_VERSION = {"1.1": frozenset({HG_RECORD_FAMILY}), "1.2": None, "1.3": None}


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_correction_records_from_archive(
    source: Union[str, Path, bytes],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return ``{archive sample ID: {family: {ratio name: record}}}``.

    A ``1.0`` archive predates correction records; it returns no entries, and
    its Hg flag keeps its historical meaning (on ordinary SSB, not applied). A
    ``1.1`` archive may hold Hg records only; calibration families need ``1.2``.
    """
    handle: object = BytesIO(bytes(source)) if isinstance(source, (bytes, bytearray)) else str(Path(source))
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    with h5py.File(handle, "r") as f:
        if _text(f.attrs.get("file_kind", "")) != FILE_KIND:
            raise ValueError("Not a processed TraceISO archive.")
        version = _text(f.attrs.get("schema_version", ""))
        if version not in SCHEMA_READABLE_VERSIONS:
            raise ValueError(
                f"Unsupported processed-archive version {version!r}; this build reads "
                f"{', '.join(SCHEMA_READABLE_VERSIONS)}."
            )
        samples = f.get("samples")
        if samples is None:
            return out
        for sample_id in samples:
            group = samples[sample_id]
            if _RECORDS_DATASET not in group:
                continue
            if version == "1.0":
                raise ValueError(
                    f"Archive sample {sample_id!r} declares version 1.0 but carries "
                    "correction records, which that version does not define."
                )
            out[str(sample_id)] = correction_records_from_payload(
                json.loads(_text(group[_RECORDS_DATASET][()])),
                allowed_families=_FAMILIES_BY_VERSION.get(version),
            )
    return out

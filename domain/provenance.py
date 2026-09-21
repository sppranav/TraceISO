"""Canonical provenance shared by the GUI service and exporters."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd
import scipy

from config.constants import APP_VERSION
from config.effective_configuration import EffectiveConfiguration


PROVENANCE_SCHEMA_NAME = "traceiso.provenance"
PROVENANCE_SCHEMA_VERSION = "1.2"
#: ``1.0`` records remain readable and carry only the effective-configuration
#: digest. They cannot be reconstructed or independently verified, because the
#: payload that digest was taken over was never saved; such a record stays
#: unknown rather than being rebuilt from the library that happens to be
#: loaded today.
PROVENANCE_READABLE_VERSIONS = ("1.0", "1.1", "1.2")


@dataclass(frozen=True)
class AnalysisProvenance:
    created_utc: str
    software_version: str
    software_commit: str
    software_dirty: bool | None
    source_filename: str
    input_sha256: str
    effective_configuration_sha256: str
    crm_record_ids: tuple[str, ...]
    dependencies: Mapping[str, str]
    rng: Mapping[str, Any]
    #: A073: the complete versioned payload the digest above was taken over.
    #: A SHA-256 alone cannot be checked or reconstructed by a reader, so the
    #: snapshot travels with it. Empty means "not recorded" - unknown, never
    #: "the settings were the defaults".
    effective_configuration: Mapping[str, Any] = field(default_factory=dict)
    cycle_masks: Mapping[str, Any] = field(default_factory=dict)
    cycle_windows: Mapping[str, Any] = field(default_factory=dict)
    rejection_settings: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    software_identity: Mapping[str, Any] = field(default_factory=dict)
    schema_name: str = PROVENANCE_SCHEMA_NAME
    schema_version: str = PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_name != PROVENANCE_SCHEMA_NAME:
            raise ValueError(f"Unsupported provenance schema {self.schema_name!r}")
        if self.schema_version not in PROVENANCE_READABLE_VERSIONS:
            raise ValueError(
                f"Unsupported provenance version {self.schema_version!r}; "
                f"this build reads {', '.join(PROVENANCE_READABLE_VERSIONS)}."
            )
        if not self.created_utc.endswith(("Z", "+00:00")):
            raise ValueError("created_utc must state UTC with Z or +00:00")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["crm_record_ids"] = list(self.crm_record_ids)
        payload["warnings"] = list(self.warnings)
        if self.schema_version in ("1.0", "1.1"):
            payload.pop("software_identity", None)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalysisProvenance":
        raw = dict(payload)
        version = str(raw.get("schema_version", PROVENANCE_SCHEMA_VERSION))
        if version not in PROVENANCE_READABLE_VERSIONS:
            raise ValueError(
                f"Unsupported provenance version {version!r}; this build reads "
                f"{', '.join(PROVENANCE_READABLE_VERSIONS)}."
            )
        raw["crm_record_ids"] = tuple(raw.get("crm_record_ids", ()))
        raw["warnings"] = tuple(raw.get("warnings", ()))
        raw.setdefault("effective_configuration", {})
        # A 1.0 record is read as a 1.0 record. Restamping it as 1.1 would
        # claim it carries a payload it does not have.
        return cls(**raw)


def reconstruct_effective_configuration(
    provenance: AnalysisProvenance | Mapping[str, Any],
) -> EffectiveConfiguration:
    """Rebuild the effective configuration a saved analysis actually used.

    Reads only the exported payload: no managed library, no session state and
    no default is consulted, so the result is what the run used rather than
    what the current installation would produce. ``EffectiveConfiguration``
    verifies the embedded digest, and this additionally checks it against the
    provenance record's own advertised digest.

    Raises ``ValueError`` when the payload is absent - a record that did not
    save one cannot be reconstructed, and must stay unknown.
    """
    payload = normalize_provenance(provenance)
    snapshot = payload.get("effective_configuration") or {}
    if not snapshot:
        raise ValueError(
            "This provenance record carries no effective-configuration payload; "
            "its configuration cannot be reconstructed or verified."
        )
    config = EffectiveConfiguration.from_dict(snapshot)
    advertised = str(payload.get("effective_configuration_sha256", "") or "")
    if advertised and advertised != config.sha256:
        raise ValueError(
            "Effective-configuration payload does not match the digest the "
            "provenance record advertises."
        )
    return config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
    }


def normalize_provenance(value: AnalysisProvenance | Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, AnalysisProvenance):
        return value.to_dict()
    # Read compatibility for pre-1.0 caller-supplied metadata. New production
    # paths always pass AnalysisProvenance; unversioned mappings remain
    # serializable without being falsely relabelled as the canonical schema.
    if "schema_version" not in value and "schema_name" not in value:
        return dict(value)
    return AnalysisProvenance.from_dict(value).to_dict()


def canonical_provenance_json(value: AnalysisProvenance | Mapping[str, Any]) -> str:
    return json.dumps(normalize_provenance(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_provenance(
    config: EffectiveConfiguration,
    *, source_path: str | Path, software_commit: str | None = None, software_dirty: bool | None = None,
    created_utc: str | None = None, cycle_masks: Mapping[str, Any] | None = None,
    warnings: tuple[str, ...] = (), rng: Mapping[str, Any] | None = None,
) -> AnalysisProvenance:
    from config.software_identity import software_identity
    measured = software_identity()
    return AnalysisProvenance(
        created_utc=created_utc or utc_now(),
        software_version=APP_VERSION,
        software_commit=measured['commit'] if software_commit is None else software_commit,
        software_dirty=measured['dirty'] if software_commit is None and software_dirty is None else software_dirty,
        software_identity=measured,
        source_filename=Path(source_path).name,
        input_sha256=sha256_file(source_path),
        effective_configuration_sha256=config.sha256,
        effective_configuration=config.to_dict(),
        crm_record_ids=config.crm_record_ids,
        dependencies=dependency_versions(),
        rng=dict(rng or {"bit_generator": "PCG64", "seed": None}),
        cycle_masks=dict(cycle_masks or {}),
        cycle_windows=dict(config.cycle_windows),
        rejection_settings=dict(config.rejection_settings),
        warnings=warnings,
    )

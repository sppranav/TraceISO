"""Versioned, digestible effective analysis configurations.

An effective configuration is an exported record of settings used for one
analysis.  It is deliberately not a built-in publication profile.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping

from config.settings import DriftConfig, KappaAssignment, KappaFactor, ProcessingConfig, UncertaintyConfig


EFFECTIVE_CONFIG_SCHEMA_NAME = "traceiso.effective_configuration"
EFFECTIVE_CONFIG_SCHEMA_VERSION = "2.4"
#: ``1.0`` records remain readable. They are *narrower*, not equivalent: they
#: carry no reference contents, no custom contributor definitions, no profile
#: defaults and no per-sample applicability, so a 1.0 digest cannot distinguish
#: analyses that differ only in those inputs. A historical 1.0 record therefore
#: stays incomplete evidence; it is never widened into a 2.0 record by filling
#: the missing fields from today's managed library.
#:
#: ``2.1`` adds ``correction_semantics``. The ``apply_hg_interference_correction``
#: field kept its name while its meaning changed: before 2.1 the ordinary Pb SSB
#: route stored the flag without subtracting any Hg. A 2.0 record is read with
#: that historical meaning and never relabelled as 2.1.
#:
#: ``2.2`` adds ``processing.pb_standard_calibration`` (Pb-standard calibration
#: after Tl normalization, with its role and material assignments) and the
#: ``pb_tl_standard_calibration`` / ``pb_calibrated_delta`` semantics. An older
#: record never claims an enabled calibration.
#: 2.3 adds explicit Sr calibration-standard observation IDs and a separate output layer.
EFFECTIVE_CONFIG_READABLE_VERSIONS = ("1.0", "2.0", "2.1", "2.2", "2.3", "2.4")
_PRE_SEMANTICS_VERSIONS = ("1.0", "2.0")
_PRE_CALIBRATION_VERSIONS = ("1.0", "2.0", "2.1")
_CALIBRATION_SEMANTICS_KEYS = ("pb_tl_standard_calibration", "pb_calibrated_delta")

#: Correction semantics this build applies to the settings it records.
CURRENT_CORRECTION_SEMANTICS: Mapping[str, str] = {
    "uncertainty_qualification": "phase2.restricted_scientific_models.v1",
    "sr_standard_calibration": "sr_explicit_standards.v2",
    "pb_hg_correction": "pb_hg_correction.v1",
    "pb_tl_standard_calibration": "pb_tl_standard_calibration.v1",
    "pb_calibrated_delta": "pb_calibrated_delta.v1",
}
#: Reading of the Hg flag in a record older than ``2.1``.
LEGACY_PB_HG_CORRECTION_SEMANTICS = "pb_hg_correction.legacy_unversioned"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class EffectiveConfiguration:
    """Complete settings and evidence context for one processing run."""

    workflow_id: str
    element: str
    processing: Mapping[str, Any]
    uncertainty: Mapping[str, Any]
    contributor_states: Mapping[str, bool] = field(default_factory=dict)
    contributor_magnitudes: Mapping[str, Any] = field(default_factory=dict)
    contributor_pdfs: Mapping[str, str] = field(default_factory=dict)
    crm_record_ids: tuple[str, ...] = ()
    #: The numerical contents of the references actually resolved, keyed by
    #: role. A record ID says *which* certificate was used, not what it said,
    #: and a managed certificate is editable under an unchanged ID.
    reference_contents: Mapping[str, Any] = field(default_factory=dict)
    #: Per-observation contributor applicability as resolved: assigned profile,
    #: explicit overrides, opted-in custom terms and any sample-specific
    #: numerical inputs. Supplied on the sample, not on either dataclass.
    sample_overrides: Mapping[str, Any] = field(default_factory=dict)
    #: User-defined contributor definitions in force, including each one's
    #: magnitude, type, degrees of freedom and probability distribution.
    custom_contributor_definitions: Mapping[str, Any] = field(default_factory=dict)
    #: Contributor defaults of every uncertainty profile a sample may be
    #: assigned to, built-in and user-defined alike.
    profile_defaults: Mapping[str, Any] = field(default_factory=dict)
    cycle_windows: Mapping[str, Any] = field(default_factory=dict)
    rejection_settings: Mapping[str, Any] = field(default_factory=dict)
    #: Versioned meaning of correction settings whose names outlived a change
    #: in behaviour, e.g. ``{"pb_hg_correction": "pb_hg_correction.v1"}``.
    #: Absent from, and refused on, records older than ``2.1``.
    correction_semantics: Mapping[str, str] = field(default_factory=dict)
    scientific_inputs: Mapping[str, Any] = field(default_factory=dict)
    evidence_created_utc: str = ""
    schema_name: str = EFFECTIVE_CONFIG_SCHEMA_NAME
    schema_version: str = EFFECTIVE_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.workflow_id.strip():
            raise ValueError("workflow_id must not be empty")
        if not self.element.strip():
            raise ValueError("element must not be empty")
        if self.schema_name != EFFECTIVE_CONFIG_SCHEMA_NAME:
            raise ValueError(f"Unsupported effective-configuration schema {self.schema_name!r}")
        if self.schema_version not in EFFECTIVE_CONFIG_READABLE_VERSIONS:
            raise ValueError(
                f"Unsupported effective-configuration version {self.schema_version!r}; "
                f"this build reads {', '.join(EFFECTIVE_CONFIG_READABLE_VERSIONS)}."
            )
        if self.schema_version != "2.4" and self.scientific_inputs:
            raise ValueError("scientific_inputs requires effective-configuration version 2.4")
        if self.schema_version in _PRE_SEMANTICS_VERSIONS and self.correction_semantics:
            raise ValueError(
                f"Effective-configuration version {self.schema_version} does not define "
                "correction_semantics."
            )
        if self.schema_version not in ("2.3", "2.4") and (
            "sr_standard_calibration" in self.correction_semantics
            or (self.processing or {}).get("sr_calibration_standard_ids")
        ):
            raise ValueError("Explicit Sr calibration selections require effective-configuration version 2.3.")
        if self.schema_version in _PRE_CALIBRATION_VERSIONS:
            calibration = (self.processing or {}).get("pb_standard_calibration")
            if any(key in self.correction_semantics for key in _CALIBRATION_SEMANTICS_KEYS) or (
                isinstance(calibration, Mapping) and calibration.get("enabled")
            ):
                raise ValueError(
                    f"Effective-configuration version {self.schema_version} does not define "
                    "Pb-standard calibration."
                )

    def canonical_payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.schema_version != "2.4":
            payload.pop("scientific_inputs", None)
        if self.schema_version in _PRE_SEMANTICS_VERSIONS:
            # A historical record digests over the fields it was written with.
            payload.pop("correction_semantics", None)
        payload["crm_record_ids"] = list(self.crm_record_ids)
        # Tuples and lists serialize identically, so a payload that survived a
        # JSON round trip must digest to the same value as the object that
        # produced it. Normalizing here keeps ``from_dict`` verification exact.
        return json.loads(_canonical_json(payload))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.canonical_payload()).encode("utf-8")).hexdigest()

    def to_dict(self, *, include_digest: bool = True) -> Dict[str, Any]:
        payload = self.canonical_payload()
        if include_digest:
            payload["sha256"] = self.sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EffectiveConfiguration":
        raw = dict(payload)
        declared_digest = raw.pop("sha256", None)
        config = cls(
            workflow_id=str(raw.pop("workflow_id", "")),
            element=str(raw.pop("element", "")),
            processing=dict(raw.pop("processing", {})),
            uncertainty=dict(raw.pop("uncertainty", {})),
            contributor_states=dict(raw.pop("contributor_states", {})),
            contributor_magnitudes=dict(raw.pop("contributor_magnitudes", {})),
            contributor_pdfs=dict(raw.pop("contributor_pdfs", {})),
            crm_record_ids=tuple(raw.pop("crm_record_ids", ())),
            reference_contents=dict(raw.pop("reference_contents", {})),
            sample_overrides=dict(raw.pop("sample_overrides", {})),
            custom_contributor_definitions=dict(
                raw.pop("custom_contributor_definitions", {})
            ),
            profile_defaults=dict(raw.pop("profile_defaults", {})),
            cycle_windows=dict(raw.pop("cycle_windows", {})),
            rejection_settings=dict(raw.pop("rejection_settings", {})),
            correction_semantics=dict(raw.pop("correction_semantics", {})),
            scientific_inputs=dict(raw.pop("scientific_inputs", {})),
            evidence_created_utc=str(raw.pop("evidence_created_utc", "")),
            schema_name=str(raw.pop("schema_name", EFFECTIVE_CONFIG_SCHEMA_NAME)),
            schema_version=str(raw.pop("schema_version", EFFECTIVE_CONFIG_SCHEMA_VERSION)),
        )
        if raw:
            raise ValueError(f"Unknown effective-configuration fields: {sorted(raw)}")
        if declared_digest is not None and str(declared_digest) != config.sha256:
            raise ValueError("Effective-configuration SHA-256 digest does not match its content")
        return config

    def processing_config(self) -> ProcessingConfig:
        raw = dict(self.processing)
        drift = raw.get("drift")
        if isinstance(drift, Mapping):
            raw["drift"] = DriftConfig(**dict(drift))
        return ProcessingConfig(**raw)

    def uncertainty_config(self) -> UncertaintyConfig:
        raw = dict(self.uncertainty)
        factors = raw.get("kappa_factors")
        if isinstance(factors, list):
            raw["kappa_factors"] = [KappaFactor(**item) for item in factors]
        assignments = raw.get("kappa_assignments")
        if isinstance(assignments, Mapping):
            raw["kappa_assignments"] = KappaAssignment(**dict(assignments))
        return UncertaintyConfig(**raw)


def build_effective_configuration(
    *, workflow_id: str, element: str, processing_config: ProcessingConfig,
    uncertainty_config: UncertaintyConfig, **evidence: Any,
) -> EffectiveConfiguration:
    """Capture the effective dataclass values without adding scientific defaults."""
    evidence.setdefault("correction_semantics", dict(CURRENT_CORRECTION_SEMANTICS))
    return EffectiveConfiguration(
        workflow_id=workflow_id,
        element=element,
        processing=asdict(processing_config),
        uncertainty=asdict(uncertainty_config),
        **evidence,
    )


def pb_hg_correction_semantics(config: EffectiveConfiguration) -> str:
    """The meaning of ``apply_hg_interference_correction`` in this record.

    Records older than ``2.1`` read as ``pb_hg_correction.legacy_unversioned``:
    on the ordinary SSB route that flag was stored without being applied.
    """
    if config.schema_version in _PRE_SEMANTICS_VERSIONS:
        return LEGACY_PB_HG_CORRECTION_SEMANTICS
    return str(config.correction_semantics.get("pb_hg_correction", "") or "")

"""Frozen, versioned records of Pb-standard calibration after Tl normalization.

Two record families sit beside the Hg family in ``sample.correction_records``:

* ``pb_calibration`` — one :class:`PbCalibrationRecord` per non-blank observation
  and Pb ratio while calibration is requested. For a sample or an independent QC
  observation it **governs** the final layer: ``applied`` selects
  ``pb_standard_corrected_ratios``; ``unavailable`` leaves the ratio with no final
  value, never the Tl-only value in its place. Calibration standards, unassigned
  standards and ``not_used`` observations carry ``not_requested`` with the reason
  and keep their Tl-only layer as a diagnostic.
* ``pb_calibrated_delta`` — one :class:`PbCalibratedDeltaRecord` per corrected
  observation and ratio when calibrated delta is requested. It binds the delta to
  the digest of the calibration record it was computed from and states, as data,
  that the combined uncertainty and Monte Carlo of calibrated delta are not
  calculated (owner-approved deferral), so no consumer can read a zero.

Records carry scalars, identities and digests only; cycle arrays live in the
sample's layers. ``stale`` is never written here: it is derived on read from the
calibration input digest (``domain.calibration_dependencies``).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from domain.layer_status import (
    APPLIED,
    BLOCKING_STATUSES,
    NOT_REQUESTED,
    PRODUCER_STATUSES,
    REVIEW_INVALID_CYCLE_EXCLUSION,
    UNAVAILABLE,
)

PB_CALIBRATION_RECORD_FAMILY = "pb_calibration"
PB_CALIBRATED_DELTA_RECORD_FAMILY = "pb_calibrated_delta"
PB_CALIBRATION_FAMILIES = frozenset({PB_CALIBRATION_RECORD_FAMILY, PB_CALIBRATED_DELTA_RECORD_FAMILY})

PB_CALIBRATION_SCHEMA_NAME = "traceiso.pb_standard_calibration"
#: ``1.1`` (C05) adds ``residual_diagnostic``. A ``1.0`` record stays readable,
#: carries no diagnostic and keeps its original serialization and digest.
PB_CALIBRATION_SCHEMA_VERSION = "1.1"
PB_CALIBRATION_READABLE_VERSIONS = ("1.0", "1.1")
PB_CALIBRATION_SEMANTICS = "pb_tl_standard_calibration.v1"

PB_CALIBRATED_DELTA_SCHEMA_NAME = "traceiso.pb_calibrated_delta"
PB_CALIBRATED_DELTA_SCHEMA_VERSION = "1.0"
PB_CALIBRATED_DELTA_READABLE_VERSIONS = ("1.0",)
PB_CALIBRATED_DELTA_SEMANTICS = "pb_calibrated_delta.v1"

#: ``quality_metrics`` key holding the session-level calibration evidence.
PB_CALIBRATION_QUALITY_KEY = "pb_standard_calibration"
#: Engine label of every budget of a calibrated Pb result: the extended Engine C
#: budget of an applied absolute ratio (C05), and the unavailable or
#: not-calculated budget that stands in when there is no value or no model.
PB_CALIBRATION_BUDGET_ENGINE = "pb_tl_standard_calibration"

MODE_LOCAL_SSB = "local_ssb"
MODE_SESSION_MEAN_K = "session_mean_k"
REQUESTED_MODES = frozenset({MODE_LOCAL_SSB, MODE_SESSION_MEAN_K})
APPLIED_LOCAL_ALTERNATING = "local_ssb_alternating"
APPLIED_LOCAL_BLOCK = "local_ssb_block"
APPLIED_SESSION_MEAN_K = "session_mean_k"
APPLIED_MODES = frozenset({APPLIED_LOCAL_ALTERNATING, APPLIED_LOCAL_BLOCK, APPLIED_SESSION_MEAN_K})

#: Inspector/export label of the final layer for each applied mode.
FINAL_LAYER_LABELS: Mapping[str, str] = {
    APPLIED_LOCAL_ALTERNATING: "Tl-normalized + SSB-corrected",
    APPLIED_LOCAL_BLOCK: "Tl-normalized + SSB-corrected",
    APPLIED_SESSION_MEAN_K: "Tl-normalized + session-standard-corrected",
}
TL_ONLY_LAYER_LABEL = "Tl-normalized"

ROLE_SAMPLE = "sample"
ROLE_INDEPENDENT_QC = "independent_qc"
ROLE_CALIBRATION_STANDARD = "calibration_standard"
ROLE_UNASSIGNED_STANDARD = "unassigned_standard"
ROLE_NOT_USED = "not_used"
ROLES = frozenset({
    ROLE_SAMPLE, ROLE_INDEPENDENT_QC, ROLE_CALIBRATION_STANDARD, ROLE_UNASSIGNED_STANDARD, ROLE_NOT_USED,
})
#: Roles that receive the calibrated correction.
TARGET_ROLES = frozenset({ROLE_SAMPLE, ROLE_INDEPENDENT_QC})

SOURCE_LAYER = "iif_corrected_ratios"

#: Why a requested calibrated result is unavailable. Closed vocabulary.
CALIBRATION_UNAVAILABLE_REASONS: Mapping[str, str] = {
    "reference_not_selected": "No accepted Pb reference material is selected for calibration.",
    "reference_unavailable": "The selected Pb reference material has no finite positive value for this ratio.",
    "no_explicit_calibration_assignment": "No observation is explicitly assigned the calibration_standard role.",
    "no_eligible_standards": "No calibration standard is eligible for this ratio.",
    "missing_bracket_side": "No eligible calibration standard on one or both sides; no session fallback is used.",
    "tl_absent": "The observation has no Tl channels, so it has no Tl-normalized layer to calibrate.",
    "tl_failed": "Tl channels are present but no Tl-normalized layer exists for this ratio.",
    "no_valid_cycles": "No valid Tl-normalized cycle remains after calibration-layer exclusion.",
    "nonfinite_calibration_factor": "The calibration factor is not finite and positive.",
}
#: Why an observation is not corrected while calibration is requested.
CALIBRATION_NOT_REQUESTED_REASONS: Mapping[str, str] = {
    "calibration_standard_not_self_corrected": (
        "Calibration standards are not corrected by their own calibration; the Tl-normalized value is a diagnostic."
    ),
    "standard_role_unassigned": "Standard-typed observation without an explicit role; it is neither corrected nor used.",
    "not_used": "Observation explicitly marked not used.",
}
#: Why a candidate standard was not used for a bracket side, block or pool.
ELIGIBILITY_REASONS: Mapping[str, str] = {
    "excluded": "Observation is excluded.",
    "blank_not_eligible": "Blanks are never calibration standards.",
    "role_unassigned": "No calibration_standard role is assigned.",
    "independent_qc": "Independent QC never enters estimation.",
    "not_used": "Observation is marked not used.",
    "material_unassigned": "No material is assigned.",
    "other_material": "The assigned material differs from the selected reference material.",
    "tl_absent": "No Tl channels.",
    "tl_failed": "No Tl-normalized layer for this ratio.",
    "insufficient_valid_cycles": "Fewer valid calibration cycles than the configured minimum.",
    "nonfinite_mean": "The mean of the valid calibration cycles is not finite and positive.",
}
#: Why an individual cycle was excluded in the calibration layer.
INVALID_CALIBRATION_CYCLE_REASONS: Mapping[str, str] = {
    "nonfinite_tl_normalized_value": "The Tl-normalized value is not finite on its channel support.",
    "nonpositive_tl_normalized_value": "The Tl-normalized value is zero or negative.",
}
MEMBER_SIDES = frozenset({"prev", "next", "session"})
SKIP_SIDES = frozenset({"prev", "next", "session"})

NOT_CALCULATED = "not_calculated"
DELTA_UNCERTAINTY_REASON_CODE = "not_implemented_for_calibrated_pb_delta"
DELTA_UNCERTAINTY_REASON = (
    "Combined uncertainty and Monte Carlo of calibrated Pb delta values are not implemented "
    "(owner-approved deferral); they are not calculated and never reported as zero."
)
#: Reason an applied absolute calibrated result has no budget: it was reached
#: through a route that does not run the extended Engine C (C05 implements the
#: Pb-Tl route). Budgets persisted by C04 sessions also carry it.
ABSOLUTE_UNCERTAINTY_PENDING_REASON_CODE = "calibrated_pb_absolute_pending_extended_engine_c"
PRECISION_LABELS: Mapping[str, str] = {
    "none": "",
    "sd": "cycle scatter (SD)",
    "se": "precision of the mean (SE)",
}
RESULT_SPACE_ABSOLUTE = "absolute_ratio"
RESULT_SPACE_DELTA = "delta_permil"


def result_identity(ratio_name: str, result_space: str) -> str:
    """Stable identity of one reported result; absolute keeps the bare ratio name."""
    if result_space == RESULT_SPACE_ABSOLUTE:
        return str(ratio_name)
    return f"{ratio_name}|{result_space}"


def _plain(value: Any) -> Any:
    """Copy into JSON-safe built-ins, refusing non-finite floats."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if hasattr(value, "item") and not isinstance(value, (int, float)):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Pb calibration records cannot carry non-finite numbers.")
        return value
    return value


def _optional_finite(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    text = json.dumps(_plain(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cycles(entries: Any, vocabulary: Mapping[str, str]) -> Tuple[Tuple[int, str], ...]:
    cycles = tuple((int(c), str(r)) for c, r in (entries or ()))
    for cycle, reason in cycles:
        if cycle < 1 or reason not in vocabulary:
            raise ValueError(f"Invalid excluded cycle entry {(cycle, reason)!r}")
    return cycles


@dataclass(frozen=True)
class CalibrationMember:
    """One standard observation used by a calibration, with its weight in K."""

    observation_id: str
    label: str
    sample_type: str
    run_number: float
    side: str
    s_mean: float
    n_valid: int
    #: Standard error of the standard's calibration cycles; ``None`` when n < 2.
    se: Optional[float]
    k_individual: float
    #: Weight of this standard's mean in B (local) or of its K in mean(K) (session).
    weight: float
    window: Optional[Tuple[int, int]] = None
    support_n_valid: int = 0
    excluded_cycles: Tuple[Tuple[int, str], ...] = ()
    excluded_fraction: Optional[float] = None
    review_flags: Tuple[str, ...] = ()
    support_mask_sha256: str = ""

    def __post_init__(self) -> None:
        if self.side not in MEMBER_SIDES:
            raise ValueError(f"Unknown calibration member side {self.side!r}")
        for name in ("s_mean", "k_individual", "weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Calibration member {name} must be finite and positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "run_number", float(self.run_number))
        object.__setattr__(self, "n_valid", int(self.n_valid))
        object.__setattr__(self, "support_n_valid", int(self.support_n_valid))
        if self.se is not None:
            object.__setattr__(self, "se", _optional_finite(self.se, "se"))
        if self.n_valid < 2 and self.se is not None:
            raise ValueError("A one-cycle calibration standard has no measured SE")
        if self.window is not None:
            object.__setattr__(self, "window", (int(self.window[0]), int(self.window[1])))
        cycles = _cycles(self.excluded_cycles, INVALID_CALIBRATION_CYCLE_REASONS)
        object.__setattr__(self, "excluded_cycles", cycles)
        object.__setattr__(self, "review_flags", tuple(str(f) for f in self.review_flags))
        if cycles and REVIEW_INVALID_CYCLE_EXCLUSION not in self.review_flags:
            raise ValueError("A standard with an excluded invalid cycle must carry the review flag")
        if self.excluded_fraction is not None:
            object.__setattr__(self, "excluded_fraction", float(self.excluded_fraction))

    def to_dict(self) -> Dict[str, Any]:
        return _plain({
            "observation_id": self.observation_id, "label": self.label, "sample_type": self.sample_type,
            "run_number": self.run_number, "side": self.side, "s_mean": self.s_mean, "n_valid": self.n_valid,
            "se": self.se, "k_individual": self.k_individual, "weight": self.weight,
            "window": list(self.window) if self.window is not None else None,
            "support_n_valid": self.support_n_valid,
            "excluded_cycles": [[c, r] for c, r in self.excluded_cycles],
            "excluded_fraction": self.excluded_fraction, "review_flags": list(self.review_flags),
            "support_mask_sha256": self.support_mask_sha256,
        })

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationMember":
        raw = dict(payload)
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown calibration member fields: {unknown}")
        if raw.get("window") is not None:
            raw["window"] = tuple(raw["window"])
        raw["excluded_cycles"] = tuple(tuple(e) for e in raw.get("excluded_cycles", ()))
        raw["review_flags"] = tuple(raw.get("review_flags", ()))
        return cls(**raw)


@dataclass(frozen=True)
class SkippedObservation:
    """A candidate the calibration search tested and did not use, with its reason."""

    observation_id: str
    label: str
    sample_type: str
    run_number: float
    side: str
    reason_code: str

    def __post_init__(self) -> None:
        if self.side not in SKIP_SIDES:
            raise ValueError(f"Unknown skipped side {self.side!r}")
        if self.reason_code not in ELIGIBILITY_REASONS:
            raise ValueError(f"Unknown eligibility reason {self.reason_code!r}")
        object.__setattr__(self, "run_number", float(self.run_number))

    def to_dict(self) -> Dict[str, Any]:
        return _plain({name: getattr(self, name) for name in self.__dataclass_fields__})

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkippedObservation":
        raw = dict(payload)
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown skipped-observation fields: {unknown}")
        return cls(**raw)


_CALIBRATION_FIELDS = (
    "observation_id", "ratio_name", "role", "requested_mode", "applied_mode", "status", "reason_code",
    "reason", "source_layer", "reference", "members", "skipped", "bracket_estimate",
    "mean_inverse_standard", "k_applied", "n_cycles", "support_n_valid", "n_valid", "excluded_cycles",
    "excluded_fraction", "review_flags", "support_mask_sha256", "pool_id", "calibration_input_digest",
    "sample_chain_digest", "layout", "residual_diagnostic", "semantics_version", "schema_name", "schema_version",
)
#: Fields a ``1.0`` record never carried; omitted from its serialization so its digest is unchanged.
_FIELDS_ADDED_IN_1_1 = ("residual_diagnostic",)


@dataclass(frozen=True)
class PbCalibrationRecord:
    """What the Pb-standard calibration did for one observation and ratio."""

    observation_id: str
    ratio_name: str
    role: str
    requested_mode: str
    status: str
    applied_mode: str = ""
    reason_code: str = ""
    reason: str = ""
    source_layer: str = SOURCE_LAYER
    #: ``material``, ``record_id``, ``material_id``, ``ratio_name``, ``value``,
    #: ``uncertainty``, ``k``, ``uncertainty_semantics``, ``derived``. An
    #: unassigned uncertainty stays ``None``.
    reference: Mapping[str, Any] = field(default_factory=dict)
    members: Tuple[CalibrationMember, ...] = ()
    skipped: Tuple[SkippedObservation, ...] = ()
    #: Local modes: the bracket estimate B = weighted mean of the member means.
    bracket_estimate: Optional[float] = None
    #: Session mode: mean_j(1/S_j), so that K = C * mean_inverse_standard.
    mean_inverse_standard: Optional[float] = None
    k_applied: Optional[float] = None
    n_cycles: int = 0
    #: The sample's Tl-only support before calibration-layer exclusion.
    support_n_valid: int = 0
    n_valid: int = 0
    excluded_cycles: Tuple[Tuple[int, str], ...] = ()
    excluded_fraction: Optional[float] = None
    review_flags: Tuple[str, ...] = ()
    support_mask_sha256: str = ""
    pool_id: str = ""
    calibration_input_digest: str = ""
    sample_chain_digest: str = ""
    #: Layout diagnostics (bracket position fraction, or run position in the pool).
    layout: Mapping[str, Any] = field(default_factory=dict)
    #: Session-level held-out residual diagnostic of this ratio's calibration
    #: (``domain.uncertainty.pb_calibration_residual``); applied targets only.
    residual_diagnostic: Mapping[str, Any] = field(default_factory=dict)
    semantics_version: str = PB_CALIBRATION_SEMANTICS
    schema_name: str = PB_CALIBRATION_SCHEMA_NAME
    schema_version: str = PB_CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_name != PB_CALIBRATION_SCHEMA_NAME:
            raise ValueError(f"Unsupported Pb calibration record schema {self.schema_name!r}")
        if self.schema_version not in PB_CALIBRATION_READABLE_VERSIONS:
            raise ValueError(
                f"Unsupported Pb calibration record version {self.schema_version!r}; "
                f"this build reads {', '.join(PB_CALIBRATION_READABLE_VERSIONS)}."
            )
        if self.semantics_version != PB_CALIBRATION_SEMANTICS:
            raise ValueError(f"Unknown Pb calibration semantics {self.semantics_version!r}")
        if self.role not in ROLES:
            raise ValueError(f"Unknown calibration role {self.role!r}")
        if self.requested_mode not in REQUESTED_MODES:
            raise ValueError(f"Unknown requested calibration mode {self.requested_mode!r}")
        if self.status not in PRODUCER_STATUSES:
            raise ValueError(f"Calibration record status {self.status!r} is not a producer status")
        if self.source_layer != SOURCE_LAYER:
            raise ValueError("Pb-standard calibration consumes the Tl-only layer only")
        targeted = self.role in TARGET_ROLES
        if self.status == NOT_REQUESTED:
            if targeted or self.reason_code not in CALIBRATION_NOT_REQUESTED_REASONS:
                raise ValueError(f"Invalid not_requested calibration record ({self.role}, {self.reason_code})")
        elif not targeted:
            raise ValueError("Only samples and independent QC receive a calibrated result")
        if self.status == UNAVAILABLE and self.reason_code not in CALIBRATION_UNAVAILABLE_REASONS:
            raise ValueError(f"Unknown calibration reason code {self.reason_code!r}")
        members = tuple(
            m if isinstance(m, CalibrationMember) else CalibrationMember.from_dict(m) for m in self.members
        )
        skipped = tuple(
            s if isinstance(s, SkippedObservation) else SkippedObservation.from_dict(s) for s in self.skipped
        )
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "skipped", skipped)
        for name in ("bracket_estimate", "mean_inverse_standard", "k_applied"):
            object.__setattr__(self, name, _optional_finite(getattr(self, name), name))
        if self.status == APPLIED:
            if self.applied_mode not in APPLIED_MODES:
                raise ValueError(f"Applied calibration needs an applied mode, got {self.applied_mode!r}")
            expected_request = MODE_SESSION_MEAN_K if self.applied_mode == APPLIED_SESSION_MEAN_K else MODE_LOCAL_SSB
            if expected_request != self.requested_mode:
                raise ValueError("Applied calibration mode does not match the requested mode")
            if not members or self.k_applied is None or self.k_applied <= 0:
                raise ValueError("An applied calibration records its members and a positive K")
            if abs(sum(m.weight for m in members) - 1.0) > 1e-12:
                raise ValueError("Calibration member weights must sum to one")
            if self.applied_mode == APPLIED_SESSION_MEAN_K and self.mean_inverse_standard is None:
                raise ValueError("A session calibration records mean(1/S)")
            if self.applied_mode != APPLIED_SESSION_MEAN_K and self.bracket_estimate is None:
                raise ValueError("A local calibration records its bracket estimate B")
        elif self.applied_mode:
            raise ValueError("Only an applied calibration carries an applied mode")
        cycles = _cycles(self.excluded_cycles, INVALID_CALIBRATION_CYCLE_REASONS)
        object.__setattr__(self, "excluded_cycles", cycles)
        object.__setattr__(self, "review_flags", tuple(str(f) for f in self.review_flags))
        if cycles and REVIEW_INVALID_CYCLE_EXCLUSION not in self.review_flags:
            raise ValueError("An observation with an excluded invalid cycle must carry the review flag")
        if self.excluded_fraction is not None:
            fraction = float(self.excluded_fraction)
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError("excluded_fraction must lie in [0, 1]")
            object.__setattr__(self, "excluded_fraction", fraction)
        object.__setattr__(self, "reference", _plain(self.reference or {}))
        object.__setattr__(self, "layout", _plain(self.layout or {}))
        object.__setattr__(self, "residual_diagnostic", _plain(self.residual_diagnostic or {}))
        if self.schema_version == "1.0" and self.residual_diagnostic:
            raise ValueError("A 1.0 Pb calibration record carries no residual diagnostic")

    @property
    def governs_final(self) -> bool:
        return self.role in TARGET_ROLES

    @property
    def n_excluded(self) -> int:
        return len(self.excluded_cycles)

    @property
    def final_layer_label(self) -> str:
        return FINAL_LAYER_LABELS.get(self.applied_mode, "Tl + Pb-standard-corrected")

    def to_dict(self) -> Dict[str, Any]:
        payload = {name: getattr(self, name) for name in _CALIBRATION_FIELDS}
        payload["members"] = [m.to_dict() for m in self.members]
        payload["skipped"] = [s.to_dict() for s in self.skipped]
        payload["excluded_cycles"] = [[c, r] for c, r in self.excluded_cycles]
        payload["review_flags"] = list(self.review_flags)
        if self.schema_version == "1.0":
            for name in _FIELDS_ADDED_IN_1_1:
                payload.pop(name, None)
        return _plain(payload)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PbCalibrationRecord":
        raw = dict(payload)
        unknown = sorted(set(raw) - set(_CALIBRATION_FIELDS))
        if unknown:
            raise ValueError(f"Unknown Pb calibration record fields: {unknown}")
        version = str(raw.get("schema_version", ""))
        if version not in PB_CALIBRATION_READABLE_VERSIONS:
            raise ValueError(
                f"Unsupported Pb calibration record version {version!r}; "
                f"this build reads {', '.join(PB_CALIBRATION_READABLE_VERSIONS)}."
            )
        raw["members"] = tuple(CalibrationMember.from_dict(m) for m in raw.get("members", ()))
        raw["skipped"] = tuple(SkippedObservation.from_dict(s) for s in raw.get("skipped", ()))
        raw["excluded_cycles"] = tuple(tuple(e) for e in raw.get("excluded_cycles", ()))
        raw["review_flags"] = tuple(raw.get("review_flags", ()))
        return cls(**raw)


_DELTA_FIELDS = (
    "observation_id", "ratio_name", "status", "reason_code", "reason", "applied_mode", "estimator",
    "scale_factor", "calibration_record_sha256", "reference_material", "reference_record_id",
    "delta_mean", "n_valid", "precision_statistic", "precision_value", "precision_label",
    "uncertainty_status", "uncertainty_reason_code", "mc_status", "mc_reason_code", "result_space",
    "result_identity", "semantics_version", "schema_name", "schema_version",
)

DELTA_ESTIMATORS: Mapping[str, str] = {
    APPLIED_LOCAL_ALTERNATING: "1000*(X/B-1)",
    APPLIED_LOCAL_BLOCK: "1000*(X/B-1)",
    APPLIED_SESSION_MEAN_K: "1000*(X*mean(1/S)-1)",
}


@dataclass(frozen=True)
class PbCalibratedDeltaRecord:
    """A calibrated delta result and the explicit absence of its uncertainty."""

    observation_id: str
    ratio_name: str
    status: str
    reason_code: str = ""
    reason: str = ""
    applied_mode: str = ""
    estimator: str = ""
    #: 1/B (local) or mean_j(1/S_j) (session): delta_i = 1000*(X_i*scale - 1).
    scale_factor: Optional[float] = None
    calibration_record_sha256: str = ""
    reference_material: str = ""
    reference_record_id: str = ""
    #: Mean over the stored calibrated support, in permil.
    delta_mean: Optional[float] = None
    n_valid: int = 0
    precision_statistic: str = "none"
    #: Cycle scatter or precision of the mean in permil; never a combined uncertainty.
    precision_value: Optional[float] = None
    precision_label: str = ""
    uncertainty_status: str = NOT_CALCULATED
    uncertainty_reason_code: str = DELTA_UNCERTAINTY_REASON_CODE
    mc_status: str = NOT_CALCULATED
    mc_reason_code: str = DELTA_UNCERTAINTY_REASON_CODE
    result_space: str = RESULT_SPACE_DELTA
    result_identity: str = ""
    semantics_version: str = PB_CALIBRATED_DELTA_SEMANTICS
    schema_name: str = PB_CALIBRATED_DELTA_SCHEMA_NAME
    schema_version: str = PB_CALIBRATED_DELTA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_name != PB_CALIBRATED_DELTA_SCHEMA_NAME:
            raise ValueError(f"Unsupported calibrated delta record schema {self.schema_name!r}")
        if self.schema_version not in PB_CALIBRATED_DELTA_READABLE_VERSIONS:
            raise ValueError(
                f"Unsupported calibrated delta record version {self.schema_version!r}; "
                f"this build reads {', '.join(PB_CALIBRATED_DELTA_READABLE_VERSIONS)}."
            )
        if self.semantics_version != PB_CALIBRATED_DELTA_SEMANTICS:
            raise ValueError(f"Unknown calibrated delta semantics {self.semantics_version!r}")
        if self.result_space != RESULT_SPACE_DELTA:
            raise ValueError("A calibrated delta record lives in the delta_permil result space")
        expected_identity = result_identity(self.ratio_name, RESULT_SPACE_DELTA)
        if not self.result_identity:
            object.__setattr__(self, "result_identity", expected_identity)
        elif self.result_identity != expected_identity:
            raise ValueError("Calibrated delta result identity does not match its ratio")
        # The deferral is data, not a missing field: only these values are legal.
        if (self.uncertainty_status, self.uncertainty_reason_code) != (NOT_CALCULATED, DELTA_UNCERTAINTY_REASON_CODE):
            raise ValueError("Calibrated delta combined uncertainty is not calculated in this build")
        if (self.mc_status, self.mc_reason_code) != (NOT_CALCULATED, DELTA_UNCERTAINTY_REASON_CODE):
            raise ValueError("Calibrated delta Monte Carlo is not calculated in this build")
        if self.status not in (APPLIED, UNAVAILABLE):
            raise ValueError(f"Calibrated delta status {self.status!r} is not applied or unavailable")
        if self.precision_statistic not in PRECISION_LABELS:
            raise ValueError(f"Unknown precision statistic {self.precision_statistic!r}")
        object.__setattr__(self, "scale_factor", _optional_finite(self.scale_factor, "scale_factor"))
        object.__setattr__(self, "delta_mean", _optional_finite(self.delta_mean, "delta_mean"))
        object.__setattr__(self, "precision_value", _optional_finite(self.precision_value, "precision_value"))
        object.__setattr__(self, "n_valid", int(self.n_valid))
        if self.status == APPLIED:
            if self.applied_mode not in APPLIED_MODES or self.delta_mean is None or self.scale_factor is None:
                raise ValueError("An applied calibrated delta records its mode, scale and mean")
            if self.estimator != DELTA_ESTIMATORS[self.applied_mode]:
                raise ValueError("Calibrated delta estimator does not match its mode")
            if not self.calibration_record_sha256:
                raise ValueError("An applied calibrated delta is bound to its calibration record")
        else:
            if self.reason_code not in CALIBRATION_UNAVAILABLE_REASONS:
                raise ValueError(f"Unknown calibrated delta reason code {self.reason_code!r}")
            if self.delta_mean is not None or self.precision_value is not None:
                raise ValueError("An unavailable calibrated delta carries no value")
        if self.precision_statistic == "none" and self.precision_value is not None:
            raise ValueError("No precision value without a precision statistic")
        expected_label = PRECISION_LABELS[self.precision_statistic]
        if self.precision_label != expected_label:
            object.__setattr__(self, "precision_label", expected_label)

    def to_dict(self) -> Dict[str, Any]:
        return _plain({name: getattr(self, name) for name in _DELTA_FIELDS})

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PbCalibratedDeltaRecord":
        raw = dict(payload)
        unknown = sorted(set(raw) - set(_DELTA_FIELDS))
        if unknown:
            raise ValueError(f"Unknown calibrated delta record fields: {unknown}")
        version = str(raw.get("schema_version", ""))
        if version not in PB_CALIBRATED_DELTA_READABLE_VERSIONS:
            raise ValueError(
                f"Unsupported calibrated delta record version {version!r}; "
                f"this build reads {', '.join(PB_CALIBRATED_DELTA_READABLE_VERSIONS)}."
            )
        return cls(**raw)


def calibration_records(sample: Any) -> Dict[str, PbCalibrationRecord]:
    records = getattr(sample, "correction_records", None) or {}
    return records.get(PB_CALIBRATION_RECORD_FAMILY, {}) or {}


def governing_calibration_record(sample: Any, ratio_name: str) -> Optional[PbCalibrationRecord]:
    """The calibration record that decides this ratio's final layer, if any."""
    record = calibration_records(sample).get(ratio_name)
    if record is None or not record.governs_final:
        return None
    return record


def calibration_final_status(sample: Any, ratio_name: str) -> str:
    record = governing_calibration_record(sample, ratio_name)
    return record.status if record is not None else NOT_REQUESTED


def calibration_blocks_final_value(sample: Any, ratio_name: str) -> bool:
    """Whether a requested calibration leaves this ratio without a final value."""
    return calibration_final_status(sample, ratio_name) in BLOCKING_STATUSES


def calibrated_delta_record(sample: Any, ratio_name: str) -> Optional[PbCalibratedDeltaRecord]:
    records = getattr(sample, "correction_records", None) or {}
    return (records.get(PB_CALIBRATED_DELTA_RECORD_FAMILY, {}) or {}).get(ratio_name)


def sample_has_calibration(sample: Any) -> bool:
    return bool(calibration_records(sample))


RECORD_CLASSES = {
    PB_CALIBRATION_RECORD_FAMILY: PbCalibrationRecord,
    PB_CALIBRATED_DELTA_RECORD_FAMILY: PbCalibratedDeltaRecord,
}
RECORD_SCHEMAS = {
    PB_CALIBRATION_RECORD_FAMILY: (PB_CALIBRATION_SCHEMA_NAME, PB_CALIBRATION_SCHEMA_VERSION),
    PB_CALIBRATED_DELTA_RECORD_FAMILY: (PB_CALIBRATED_DELTA_SCHEMA_NAME, PB_CALIBRATED_DELTA_SCHEMA_VERSION),
}

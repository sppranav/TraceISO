"""Frozen, versioned records of the Pb 204Hg interference correction.

One :class:`HgCorrectionRecord` exists per observation and 204Pb-bearing ratio
whenever the correction was requested. It records what was asked for, what was
done, from which inputs, on which cycle support, and why any cycle or ratio was
left out — the evidence a reviewer needs and the status the selectors enforce.

Two routes produce records with different authority:

* ``route="ssb"`` (ordinary Pb SSB) **governs** the final layer: when it is
  ``unavailable`` the ratio has no final value, and nothing falls back to the
  uncorrected blank-corrected ratio.
* ``route="pb_tl"`` records the local Hg subtraction the legacy Pb-Tl chain
  already performs before Tl normalization. It is diagnostic only
  (``governs_final=False``): the Tl-normalized result keeps its historical
  availability, and the saved intermediates never enter a final, display or
  export fallback chain.

Records never carry cycle arrays; the arrays live in the sample's
``interference_corrected_*`` layers.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from domain.layer_status import (
    APPLIED,
    BLOCKING_STATUSES,
    NOT_REQUESTED,
    PRODUCER_STATUSES,
    REVIEW_INVALID_CYCLE_EXCLUSION,
)

HG_RECORD_FAMILY = "hg"

HG_RECORD_SCHEMA_NAME = "traceiso.pb_hg_correction"
HG_RECORD_SCHEMA_VERSION = "1.0"
HG_RECORD_READABLE_VERSIONS = ("1.0",)

#: Semantics of a result processed by this correction.
PB_HG_CORRECTION_SEMANTICS = "pb_hg_correction.v1"
#: A saved ordinary-SSB result whose settings requested Hg correction before
#: that route applied it. The flag was stored, but no Hg was subtracted.
PB_HG_LEGACY_FLAG_NOT_APPLIED = "pb_hg_correction.legacy_flag_not_applied"
#: A saved configuration or archive that predates versioned Hg semantics, so
#: which route applied the flag cannot be read from the record itself.
PB_HG_LEGACY_UNVERSIONED = "pb_hg_correction.legacy_unversioned"
PB_HG_SEMANTICS = frozenset({
    PB_HG_CORRECTION_SEMANTICS, PB_HG_LEGACY_FLAG_NOT_APPLIED, PB_HG_LEGACY_UNVERSIONED,
})

#: Sample metadata key carrying the per-run request summary.
PB_HG_CORRECTION_STATE_KEY = "_pb_hg_correction_state"

ROUTE_SSB = "ssb"
ROUTE_PB_TL = "pb_tl"
ROUTES = frozenset({ROUTE_SSB, ROUTE_PB_TL})

SOURCE_TL = "tl"
SOURCE_NATURAL_RATIO = "natural_ratio"
SOURCES = frozenset({SOURCE_TL, SOURCE_NATURAL_RATIO})

#: Why the correction is unavailable for a ratio. Closed vocabulary.
HG_UNAVAILABLE_REASONS: Mapping[str, str] = {
    "hg_reference_unresolved": "The 204Hg/202Hg reference ratio or the Hg masses could not be resolved.",
    "hg_monitor_absent": "The 202Hg monitor channel is absent.",
    "pb204_absent": "The 204Pb channel is absent.",
    "ratio_channel_absent": "The other isotope of this ratio is absent.",
    "tl_channel_incomplete": "Only one of the two Tl channels is present, so neither the Tl factor nor the natural-ratio assumption applies.",
    "tl_reference_unresolved": "Tl channels are present but the Tl reference ratio or masses are not a usable value.",
    "misaligned_channels": "Required channels do not have the same number of cycles; no array was truncated.",
    "no_valid_cycles": "No valid cycle remains after the canonical support and invalid-cycle exclusion.",
}

#: Why an individual cycle was excluded from 204Pb-bearing ratios.
INVALID_CYCLE_REASONS: Mapping[str, str] = {
    "nonfinite_tl_factor": "The per-cycle Tl mass-bias factor is not finite.",
    "nonfinite_corrected_204Pb": "The Hg-corrected 204Pb is not finite.",
    "nonpositive_corrected_204Pb": "The Hg-corrected 204Pb is zero or negative.",
}

_FIELDS = (
    "observation_id", "ratio_name", "route", "governs_final", "requested", "status",
    "reason_code", "reason", "source", "hg_reference", "tl_reference", "masses",
    "intensity_basis", "n_cycles", "support_n_valid", "n_valid", "excluded_cycles",
    "excluded_fraction", "review_flags", "support_mask_sha256", "diagnostics",
    "semantics_version", "schema_name", "schema_version",
)


def mask_sha256(mask: np.ndarray) -> str:
    """Digest a Boolean cycle mask, independent of its array dtype."""
    data = np.ascontiguousarray(np.asarray(mask, dtype=bool), dtype=np.uint8)
    return hashlib.sha256(data.tobytes()).hexdigest()


def _plain(value: Any) -> Any:
    """Copy a nested mapping/sequence into JSON-safe built-ins, refusing non-finite floats."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, float)):
        out = float(value)
        if not math.isfinite(out):
            raise ValueError("Hg correction records cannot carry non-finite numbers.")
        return out
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


@dataclass(frozen=True)
class HgCorrectionRecord:
    """What the Hg correction did for one observation and ratio."""

    observation_id: str
    ratio_name: str
    route: str
    governs_final: bool
    requested: bool
    status: str
    reason_code: str = ""
    reason: str = ""
    source: Optional[str] = None
    #: ``record_id``, ``ratio_name``, ``value``, ``uncertainty``, ``k``,
    #: ``uncertainty_semantics``. An unassigned uncertainty stays ``None``.
    hg_reference: Mapping[str, Any] = field(default_factory=dict)
    #: ``ratio_name`` and ``value`` of the Tl reference, when ``source="tl"``.
    tl_reference: Mapping[str, Any] = field(default_factory=dict)
    masses: Mapping[str, float] = field(default_factory=dict)
    intensity_basis: str = ""
    n_cycles: int = 0
    #: Cycles on the canonical support before invalid-cycle exclusion.
    support_n_valid: int = 0
    n_valid: int = 0
    #: ``(cycle, reason_code)`` pairs, 1-based cycles, in cycle order.
    excluded_cycles: Tuple[Tuple[int, str], ...] = ()
    #: Excluded cycles over ``support_n_valid``; ``None`` when there is no support.
    excluded_fraction: Optional[float] = None
    review_flags: Tuple[str, ...] = ()
    support_mask_sha256: str = ""
    #: Route diagnostics that never reject a cycle (legacy Pb-Tl counters).
    diagnostics: Mapping[str, int] = field(default_factory=dict)
    semantics_version: str = PB_HG_CORRECTION_SEMANTICS
    schema_name: str = HG_RECORD_SCHEMA_NAME
    schema_version: str = HG_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_name != HG_RECORD_SCHEMA_NAME:
            raise ValueError(f"Unsupported Hg correction record schema {self.schema_name!r}")
        if self.schema_version not in HG_RECORD_READABLE_VERSIONS:
            raise ValueError(
                f"Unsupported Hg correction record version {self.schema_version!r}; "
                f"this build reads {', '.join(HG_RECORD_READABLE_VERSIONS)}."
            )
        if self.semantics_version not in PB_HG_SEMANTICS:
            raise ValueError(f"Unknown Hg correction semantics {self.semantics_version!r}")
        if self.route not in ROUTES:
            raise ValueError(f"Unknown Hg correction route {self.route!r}")
        if self.status not in PRODUCER_STATUSES:
            raise ValueError(f"Hg correction record status {self.status!r} is not a producer status")
        if self.governs_final != (self.route == ROUTE_SSB):
            raise ValueError("Only the ordinary SSB Hg correction governs the final layer")
        if self.source is not None and self.source not in SOURCES:
            raise ValueError(f"Unknown Hg correction source {self.source!r}")
        if self.status == APPLIED and self.source is None:
            raise ValueError("An applied Hg correction must record its source")
        if self.status != APPLIED and self.status != NOT_REQUESTED:
            if self.reason_code not in HG_UNAVAILABLE_REASONS:
                raise ValueError(f"Unknown Hg correction reason code {self.reason_code!r}")
        cycles = tuple((int(c), str(r)) for c, r in self.excluded_cycles)
        for cycle, reason in cycles:
            if cycle < 1 or reason not in INVALID_CYCLE_REASONS:
                raise ValueError(f"Invalid excluded cycle entry {(cycle, reason)!r}")
        object.__setattr__(self, "excluded_cycles", cycles)
        object.__setattr__(self, "review_flags", tuple(str(f) for f in self.review_flags))
        if cycles and REVIEW_INVALID_CYCLE_EXCLUSION not in self.review_flags:
            raise ValueError("An observation with an excluded invalid cycle must carry the review flag")
        if self.excluded_fraction is not None:
            fraction = float(self.excluded_fraction)
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError("excluded_fraction must lie in [0, 1]")
            object.__setattr__(self, "excluded_fraction", fraction)
        for name in ("hg_reference", "tl_reference", "masses", "diagnostics"):
            object.__setattr__(self, name, _plain(getattr(self, name) or {}))

    @property
    def n_excluded(self) -> int:
        return len(self.excluded_cycles)

    def to_dict(self) -> Dict[str, Any]:
        payload = {name: getattr(self, name) for name in _FIELDS}
        payload["excluded_cycles"] = [[c, r] for c, r in self.excluded_cycles]
        payload["review_flags"] = list(self.review_flags)
        return _plain(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HgCorrectionRecord":
        raw = dict(payload)
        unknown = sorted(set(raw) - set(_FIELDS))
        if unknown:
            raise ValueError(f"Unknown Hg correction record fields: {unknown}")
        version = str(raw.get("schema_version", ""))
        if version not in HG_RECORD_READABLE_VERSIONS:
            raise ValueError(
                f"Unsupported Hg correction record version {version!r}; "
                f"this build reads {', '.join(HG_RECORD_READABLE_VERSIONS)}."
            )
        raw["excluded_cycles"] = tuple(tuple(item) for item in raw.get("excluded_cycles", ()))
        raw["review_flags"] = tuple(raw.get("review_flags", ()))
        return cls(**raw)


def hg_records(sample: Any) -> Dict[str, HgCorrectionRecord]:
    """The sample's Hg records keyed by ratio name (empty when not requested)."""
    records = getattr(sample, "correction_records", None) or {}
    return records.get(HG_RECORD_FAMILY, {}) or {}


def governing_hg_record(sample: Any, ratio_name: str) -> Optional[HgCorrectionRecord]:
    """The Hg record that decides this ratio's final layer, if any."""
    record = hg_records(sample).get(ratio_name)
    if record is None or not record.governs_final:
        return None
    return record


def hg_final_status(sample: Any, ratio_name: str) -> str:
    """Status the final-layer selectors enforce for this ratio."""
    record = governing_hg_record(sample, ratio_name)
    return record.status if record is not None else NOT_REQUESTED


def hg_blocks_final_value(sample: Any, ratio_name: str) -> bool:
    """Whether a requested Hg correction leaves this ratio without a final value."""
    return hg_final_status(sample, ratio_name) in BLOCKING_STATUSES


def correction_records_payload(sample: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Canonical JSON-safe payload of every correction record on a sample."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for family, by_ratio in (getattr(sample, "correction_records", None) or {}).items():
        out[str(family)] = {str(r): rec.to_dict() for r, rec in sorted(by_ratio.items())}
    return out


def correction_records_from_payload(
    payload: Mapping[str, Any],
    *,
    allowed_families: Optional[frozenset] = None,
) -> Dict[str, Dict[str, Any]]:
    """Strictly rebuild correction records; an unknown family is rejected.

    ``allowed_families`` narrows what a container version may hold, so a
    container that predates a family cannot be read as carrying it.
    """
    from domain.pb_calibration_records import RECORD_CLASSES

    classes = {HG_RECORD_FAMILY: HgCorrectionRecord, **RECORD_CLASSES}
    out: Dict[str, Dict[str, Any]] = {}
    for family, by_ratio in dict(payload).items():
        record_cls = classes.get(family)
        if record_cls is None or (allowed_families is not None and family not in allowed_families):
            raise ValueError(f"Unknown correction record family {family!r}")
        out[family] = {
            str(ratio): record_cls.from_dict(record)
            for ratio, record in dict(by_ratio).items()
        }
    return out


def pb_hg_semantics_for_legacy_state(
    *,
    hg_flag: bool,
    applied_correction_state: Optional[Mapping[str, Any]],
    has_hg_records: bool,
) -> str:
    """Interpret a saved result that carries no versioned Hg evidence.

    The ordinary SSB route read the Hg flag only from this release onward, so
    a saved SSB result with the flag on received no Hg correction. A Pb-Tl
    result did. Without a recorded route the meaning cannot be decided.
    """
    if has_hg_records:
        return PB_HG_CORRECTION_SEMANTICS
    if not hg_flag:
        return NOT_REQUESTED
    state = applied_correction_state if isinstance(applied_correction_state, Mapping) else {}
    if state.get("pb_tl") is True:
        return PB_HG_LEGACY_UNVERSIONED
    if state.get("ssb") is True:
        return PB_HG_LEGACY_FLAG_NOT_APPLIED
    return PB_HG_LEGACY_UNVERSIONED

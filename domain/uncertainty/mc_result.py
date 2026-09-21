"""Durable, versioned Monte Carlo cross-check result records.

Phase 2 of the Engine B fixed-draw plan owns *ownership and persistence* of a
completed Monte Carlo result. It owns no scientific calculation: every number
in a record is copied verbatim, at full machine precision, from the frozen
Phase 1 :class:`~domain.uncertainty.monte_carlo.MCCrossCheckResult` contract.

Why this module exists
----------------------

Before Phase 2 the authoritative completed MC result lived only in a Streamlit
session cache keyed by a configuration hash. That made it invisible to the
domain layer, impossible to export, and silently lost on session end. A
:class:`MCResultRecord` is instead:

- **attached to sample/ratio scientific state** — ``Sample.mc_results`` is a
  per-ratio mapping alongside ``Sample.uncertainty``;
- **immutable** — a frozen dataclass with tuple (not list) collections, so
  rendering or exporting a result cannot mutate it;
- **versioned** — it carries its own ``schema_name``/``schema_version`` plus
  the scientific ``semantics_version`` and ``result_space``, so a reader can
  tell a fixed-draw SSB/delta model replay from a legacy additive basis-space result
  without guessing;
- **self-describing** — execution ID, identities, modes, contributor model,
  RNG, draw counts, interval convention, configuration/input digests,
  software/environment identity, warnings and scope travel with the values.

It is an ordinary result record. It is deliberately *not* a publication
profile or manifest, and it never carries the draw array: see
:data:`DRAW_ARRAY_POLICY`.

Numeric encoding
----------------

``to_dict`` emits JSON-safe values at full precision. Finite floats are plain
JSON numbers, which round-trip exactly through ``repr``. Non-finite floats —
``inf`` is the ordinary degrees-of-freedom value for a Type B contributor, and
``nan`` is how a suppressed moment diagnostic reports itself — are encoded as
the strings ``"NaN"``, ``"Infinity"`` and ``"-Infinity"``. That keeps a
payload valid under ``json.dumps(..., allow_nan=False)`` while staying
lossless; :func:`decode_float` reverses it.
"""

from __future__ import annotations

import math
import platform
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple

from config.constants import APP_VERSION
from domain.uncertainty.sr_chain_identity import (
    ENGINE_A_ENGINE,
    ENGINE_A_FRESHNESS_IDENTITY_VERSION,
    SUPPORTED_ENGINE_A_SEMANTICS,
    active_engine_a_semantics,
)
from domain.uncertainty.monte_carlo import (
    ENGINE_B_CURRENT_FIXED_DRAW_SEMANTICS,
    ENGINE_B_FIXED_DRAW_PB_HG_SEMANTICS,
    ENGINE_B_FIXED_DRAW_SEMANTICS,
    ENGINE_B_FIXED_DRAW_SEMANTICS_V1,
    ENGINE_B_FIXED_DRAW_SEMANTICS_V2,
    ENGINE_B_LEGACY_SEMANTICS,
    log_engine_b_mc_exported,
    MC_COVERAGE_PROBABILITY,
    MC_INTERVAL_CONVENTION,
    MC_PERCENTILE_METHOD,
    RESULT_SPACE_ABSOLUTE_RATIO,
    RESULT_SPACE_BASIS_RATIO,
    RESULT_SPACE_DELTA_PERMIL,
)

#: Identity of the persisted record schema. ``schema_name`` distinguishes this
#: payload from any other JSON blob in an export; ``schema_version`` changes
#: whenever the field set changes in a way a reader must know about.
MC_RESULT_SCHEMA_NAME = "traceiso.engine_b_mc_result"
MC_RESULT_SCHEMA_VERSION = "1.5"

#: Version stamped on a payload that predates this schema. A legacy payload is
#: readable but is never relabelled as a fixed-draw SSB/delta model replay.
MC_RESULT_LEGACY_SCHEMA_VERSION = "0"

#: Every schema version this build can read. ``1.0`` predates
#: :attr:`MCResultRecord.budget_digest`; ``1.0`` and ``1.1`` both predate the
#: complete configuration identity introduced in ``1.2``. Their values remain
#: readable, but freshness is *unknown* rather than falsely reported current.
MC_RESULT_READABLE_SCHEMA_VERSIONS = ("0", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5")

#: Version of the freshness identity algorithm. ``v2`` hashes the complete
#: uncertainty configuration and requires both configuration and budget
#: identities before a record may be called current. ``v3`` adds the sampling
#: distribution of every budget row, so a user-defined contributor whose PDF
#: changed at an unchanged magnitude is disclosed instead of read as current.
#: A record stamped with an older identity migrates as *unknown*, never fresh.
MC_FRESHNESS_IDENTITY_VERSION = "engine_b.freshness.v4"

#: Semantics identifier used when an older payload does not declare one. It is
#: intentionally distinct from both ``ENGINE_B_FIXED_DRAW_SEMANTICS`` and
#: ``ENGINE_B_LEGACY_SEMANTICS``: an undeclared payload is unknown, not known
#: to be either.
MC_SEMANTICS_UNDECLARED = "engine_b.undeclared.legacy.v0"

#: Result space used when an older payload does not declare one.
RESULT_SPACE_UNKNOWN = "unknown"

#: Resolved engine identifiers that are *not* Engine B. Engine A (internal
#: normalization) and Engine C (Pb-Tl external normalization) have their own
#: chain replay and their own semantics; a record from either must not be
#: judged against, or labelled by, the Engine B fixed-draw contract.
NON_ENGINE_B_ENGINES = frozenset(
    {"internal_normalization", "pb_tl_external_normalization"}
)

#: Freshness of a stored result relative to the configuration and the budget
#: that are active *now*. A stored Monte Carlo result is a snapshot: it stays
#: valid only while the settings and the measured data behind it are the ones
#: it was computed from. These four states are the vocabulary every surface —
#: the panel and all four exporters — uses to say so.
MC_FRESHNESS_CURRENT = "current"
MC_FRESHNESS_STALE_CONFIGURATION = "stale_configuration"
MC_FRESHNESS_STALE_INPUTS = "stale_inputs"
MC_FRESHNESS_UNKNOWN = "unknown"

#: Human-readable statement for each freshness state.
MC_FRESHNESS_LABELS: Dict[str, str] = {
    MC_FRESHNESS_CURRENT: (
        "Current: matches the active uncertainty configuration and the active "
        "analytical budget."
    ),
    MC_FRESHNESS_STALE_CONFIGURATION: (
        "STALE — produced under a different uncertainty configuration or Sr "
        "correction method than the one now active. Re-run Monte Carlo before "
        "using or reporting it."
    ),
    MC_FRESHNESS_STALE_INPUTS: (
        "STALE — the measured inputs changed after this result was produced "
        "(for example a cycle mask, cycle range or outlier-filter edit). The "
        "analytical budget beside it has been recomputed and this result has "
        "not. Re-run Monte Carlo before using or reporting it."
    ),
    MC_FRESHNESS_UNKNOWN: (
        "Unknown — this result predates freshness tracking, or no active "
        "budget was available to compare against. Treat it as unverified."
    ),
}

#: Why a record never carries the Monte Carlo draw array. Stated here so an
#: exporter, a reviewer and a reader of an archive all see the same rule.
DRAW_ARRAY_POLICY = (
    "Full Monte Carlo draw arrays are never persisted or exported by default. "
    "A record stores the summary, the coverage interval and the provenance "
    "needed to reproduce the run from its seed."
)

#: Reported unit label for each result space. Presentation only.
RESULT_SPACE_UNITS: Dict[str, str] = {
    RESULT_SPACE_DELTA_PERMIL: "‰",
    RESULT_SPACE_ABSOLUTE_RATIO: "",
    RESULT_SPACE_BASIS_RATIO: "",
    RESULT_SPACE_UNKNOWN: "",
}

#: Shared, format-independent field labels. Every exporter and the UI use these
#: so a combined standard uncertainty is called ``u_c`` everywhere and an
#: expanded uncertainty is never shown without its coverage metadata.
MC_FIELD_LABELS: Dict[str, str] = {
    "mc_mean": "MC mean",
    "mc_std": "MC standard deviation (ddof=1)",
    "mc_lower": "MC coverage interval, lower",
    "mc_upper": "MC coverage interval, upper",
    "gum_u_c": "u_c (GUM combined standard uncertainty)",
    "gum_u_expanded": "U (GUM expanded uncertainty)",
    "coverage_probability": "Coverage probability",
    "gum_coverage_factor_k": "Coverage factor k",
}


def encode_float(value: Optional[float]) -> Any:
    """Return a JSON-safe, lossless encoding of one float.

    ``None`` stays ``None`` — it means *not applicable* (for example a mean
    whose Student-t moment is undefined), which is a different statement from
    ``NaN``. Non-finite floats become their canonical string token.
    """
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric):
        return "NaN"
    if math.isinf(numeric):
        return "Infinity" if numeric > 0 else "-Infinity"
    return numeric


def decode_float(value: Any) -> Optional[float]:
    """Reverse :func:`encode_float` exactly."""
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip()
        if token in {"NaN", "Infinity", "-Infinity"}:
            return float(token)
        raise ValueError(f"Unrecognised float token in MC result payload: {value!r}")
    return float(value)


@dataclass(frozen=True)
class MCContributorRecord:
    """Declared placement and distribution of one sampled contributor."""

    name: str
    placement: str
    distribution: str
    degrees_of_freedom: float
    type_ab: str
    source_field: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "placement": self.placement,
            "distribution": self.distribution,
            "degrees_of_freedom": encode_float(self.degrees_of_freedom),
            "type_ab": self.type_ab,
            "source_field": self.source_field,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MCContributorRecord":
        dof = decode_float(payload.get("degrees_of_freedom"))
        return cls(
            name=str(payload.get("name", "")),
            placement=str(payload.get("placement", "")),
            distribution=str(payload.get("distribution", "")),
            degrees_of_freedom=float("nan") if dof is None else dof,
            type_ab=str(payload.get("type_ab", "")),
            source_field=str(payload.get("source_field", "")),
        )


@dataclass(frozen=True)
class MCResultRecord:
    """One completed Monte Carlo cross-check, owned by sample/ratio state.

    Every numeric field holds the canonical full-precision machine value.
    Presentation rounding is applied separately, at display time, by
    :mod:`ui.formatting`; nothing in this record is pre-rounded.
    """

    # --- schema identity ---------------------------------------------------
    schema_name: str = MC_RESULT_SCHEMA_NAME
    schema_version: str = MC_RESULT_SCHEMA_VERSION
    status: str = "completed"
    reason_code: str = ""

    # --- scientific identity ----------------------------------------------
    execution_id: str = ""
    sample_name: str = ""
    sample_run_number: int = 0
    ratio_name: str = ""
    prev_std_label: str = ""
    next_std_label: str = ""

    # --- semantics and result space ---------------------------------------
    semantics_version: str = ""
    result_space: str = ""
    engine: str = ""
    output_mode: str = ""
    bracket_mode: str = ""
    validation_mode: str = ""
    apply_ssb_kernel: Optional[bool] = None
    delta_from_ssb: Optional[bool] = None
    blank_uncertainty_input: str = ""
    blank_placement: str = ""

    # --- single-transform evidence ----------------------------------------
    iteration_return_space: str = ""
    effective_result_space: str = ""
    post_loop_transform_applied: bool = False
    delta_reference_applied: Optional[float] = None
    absolute_scale_factor_applied: Optional[float] = None
    nominal_reported_value: Optional[float] = None
    nominal_bracket_mean: Optional[float] = None
    nominal_delta_reference: Optional[float] = None

    # --- execution semantics ----------------------------------------------
    requested_draws: int = 0
    completed_draws: int = 0
    n_dropped: int = 0
    seed: Optional[int] = None
    bit_generator: str = ""

    # --- canonical Monte Carlo summary ------------------------------------
    mc_mean: Optional[float] = None
    mc_std: Optional[float] = None
    mc_std_diagnostic: Optional[float] = None
    mc_lower: Optional[float] = None
    mc_upper: Optional[float] = None
    moment_status: str = ""
    min_type_a_dof: Optional[float] = None

    # --- coverage interval convention -------------------------------------
    coverage_probability: float = MC_COVERAGE_PROBABILITY
    interval_convention: str = MC_INTERVAL_CONVENTION
    percentile_method: str = MC_PERCENTILE_METHOD

    # --- descriptive comparison with the analytical budget ----------------
    gum_center: Optional[float] = None
    gum_u_c: Optional[float] = None
    gum_u_expanded: Optional[float] = None
    gum_coverage_factor_k: Optional[float] = None
    gum_lower: Optional[float] = None
    gum_upper: Optional[float] = None
    absolute_center_difference: Optional[float] = None
    center_difference_u_c: Optional[float] = None
    std_difference_pct: Optional[float] = None
    convergence_mean_pct: Optional[float] = None
    convergence_std_pct: Optional[float] = None
    convergence_half_width_pct: Optional[float] = None
    convergence_q025_pct: Optional[float] = None
    convergence_q975_pct: Optional[float] = None

    # --- contributor model -------------------------------------------------
    contributors: Tuple[MCContributorRecord, ...] = ()

    # --- configuration and input identity ---------------------------------
    config_digest: str = ""
    input_digest: str = ""
    replay_snapshot_json: str = ""
    freshness_identity_version: str = MC_FRESHNESS_IDENTITY_VERSION
    #: Digest of the analytical budget this result was computed against.
    #: ``config_digest`` alone cannot detect a data-only change — a cycle
    #: mask, cycle range or outlier-filter edit moves the measured inputs
    #: while every setting stays identical — so this is the second half of the
    #: freshness test. Historical ``1.0``/``1.1`` identities migrate as
    #: *unknown*, never *fresh*. See :func:`mc_record_freshness`.
    budget_digest: str = ""

    # --- software and environment identity --------------------------------
    app_version: str = ""
    python_version: str = ""
    numpy_version: str = ""
    os_platform: str = ""
    software_commit: str = ""

    # --- warnings and scope ------------------------------------------------
    warnings: Tuple[str, ...] = ()
    scope_note: str = ""

    # --- draw-array policy -------------------------------------------------
    #: Always ``False``. The field exists so an archive states the policy
    #: rather than leaving a reader to infer it from an absent key.
    draw_array_persisted: bool = False
    draw_array_policy: str = DRAW_ARRAY_POLICY

    created_utc: str = ""

    # -- derived, non-stored views -----------------------------------------

    @property
    def is_engine_b(self) -> bool:
        """True unless the record came from Engine A or Engine C.

        An empty ``engine`` is treated as Engine B, because only the Engine B
        path ever wrote a record without one.
        """
        return self.engine not in NON_ENGINE_B_ENGINES

    @property
    def is_chain_replay(self) -> bool:
        """True only for a declared Engine B fixed-draw SSB/delta model replay."""
        return self.semantics_version in {
            ENGINE_B_FIXED_DRAW_SEMANTICS_V1,
            ENGINE_B_FIXED_DRAW_SEMANTICS_V2,
            ENGINE_B_FIXED_DRAW_SEMANTICS,
            ENGINE_B_FIXED_DRAW_PB_HG_SEMANTICS,
        }

    @property
    def is_legacy_semantics(self) -> bool:
        """True for an Engine B result that is not the fixed-draw replay.

        Engine A and Engine C results are not Engine B legacy results; they
        are a different engine, and this stays ``False`` for them.
        """
        return self.is_engine_b and not self.is_chain_replay

    @property
    def semantics_label(self) -> str:
        """Short human-readable statement of what this result is."""
        if self.semantics_version == ENGINE_B_FIXED_DRAW_PB_HG_SEMANTICS:
            return (
                "Fixed-draw SSB/delta model-replay cross-check with linearized "
                "Pb Hg-correction terms"
            )
        if self.is_chain_replay:
            return "Fixed-draw SSB/delta model-replay cross-check"
        if not self.is_engine_b:
            return f"Monte Carlo chain replay ({self.engine})"
        if self.semantics_version == ENGINE_B_LEGACY_SEMANTICS:
            return "Legacy additive basis-space Monte Carlo"
        if self.semantics_version == MC_SEMANTICS_UNDECLARED:
            return "Legacy Monte Carlo, semantics not declared"
        return f"Monte Carlo ({self.semantics_version or 'unknown semantics'})"

    @property
    def unit_label(self) -> str:
        """Unit of the stored summary values. Presentation only."""
        return RESULT_SPACE_UNITS.get(
            self.effective_result_space or self.result_space, ""
        )

    @property
    def transformed_once(self) -> bool:
        """True when the stored values were transformed exactly once.

        Either the iteration already returned the reported space and the
        frozen post-loop transform was skipped, or it returned basis space and
        exactly one post-loop transform ran. Anything else means the result
        space, the transform flag and the recorded operands disagree.

        The basis-space arm checks the operands too, not just the flag: a
        transform that ran must have recorded exactly one of the delta
        reference or the absolute scale factor, and a space change must be
        accompanied by a transform. That is what makes this a real
        double-transform sentinel rather than a statement about the reported
        arm alone.
        """
        returned_reported = self.iteration_return_space in {
            RESULT_SPACE_ABSOLUTE_RATIO,
            RESULT_SPACE_DELTA_PERMIL,
        }
        if returned_reported:
            # A reported-space iteration must have skipped the frozen
            # post-loop transform, and must not carry either operand.
            return not (
                self.post_loop_transform_applied
                or self.delta_reference_applied is not None
                or self.absolute_scale_factor_applied is not None
            )

        operands = sum(
            1
            for operand in (
                self.delta_reference_applied,
                self.absolute_scale_factor_applied,
            )
            if operand is not None
        )
        space_changed = bool(
            self.effective_result_space
            and self.effective_result_space != self.iteration_return_space
        )
        if self.post_loop_transform_applied:
            # Exactly one transform ran: exactly one operand, and it must have
            # actually moved the result out of basis space.
            return operands == 1 and space_changed
        # No transform ran: no operand, and the space cannot have moved.
        return operands == 0 and not space_changed

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical JSON-safe payload at full precision."""
        payload: Dict[str, Any] = {}
        for spec in fields(self):
            value = getattr(self, spec.name)
            if spec.name == "contributors":
                payload[spec.name] = [item.to_dict() for item in value]
            elif spec.name == "warnings":
                payload[spec.name] = list(value)
            elif isinstance(value, float):
                payload[spec.name] = encode_float(value)
            else:
                payload[spec.name] = value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MCResultRecord":
        """Rebuild a record from a payload, migrating older schemas first."""
        migrated = migrate_mc_result_payload(payload)
        kwargs: Dict[str, Any] = {}
        known = {spec.name for spec in fields(cls)}
        for name in known:
            if name not in migrated:
                continue
            value = migrated[name]
            if name == "contributors":
                kwargs[name] = tuple(
                    MCContributorRecord.from_dict(item) for item in (value or ())
                )
            elif name == "warnings":
                kwargs[name] = tuple(str(item) for item in (value or ()))
            elif name in _OPTIONAL_FLOAT_FIELDS or name in _REQUIRED_FLOAT_FIELDS:
                decoded = decode_float(value)
                if name in _REQUIRED_FLOAT_FIELDS and decoded is None:
                    continue
                kwargs[name] = decoded
            elif name in _INT_FIELDS:
                kwargs[name] = None if value is None else int(value)
            elif name in _BOOL_FIELDS:
                kwargs[name] = None if value is None else bool(value)
            else:
                kwargs[name] = value
        return cls(**kwargs)


_OPTIONAL_FLOAT_FIELDS = frozenset(
    {
        "delta_reference_applied",
        "absolute_scale_factor_applied",
        "nominal_reported_value",
        "nominal_bracket_mean",
        "nominal_delta_reference",
        "mc_mean",
        "mc_std",
        "mc_std_diagnostic",
        "mc_lower",
        "mc_upper",
        "min_type_a_dof",
        "gum_center",
        "gum_u_c",
        "gum_u_expanded",
        "gum_coverage_factor_k",
        "gum_lower",
        "gum_upper",
        "absolute_center_difference",
        "center_difference_u_c",
        "std_difference_pct",
        "convergence_mean_pct",
        "convergence_std_pct",
        "convergence_half_width_pct",
        "convergence_q025_pct",
        "convergence_q975_pct",
    }
)

_REQUIRED_FLOAT_FIELDS = frozenset({"coverage_probability"})

_INT_FIELDS = frozenset(
    {
        "sample_run_number",
        "requested_draws",
        "completed_draws",
        "n_dropped",
        "seed",
    }
)

_BOOL_FIELDS = frozenset(
    {
        "apply_ssb_kernel",
        "delta_from_ssb",
        "post_loop_transform_applied",
        "draw_array_persisted",
    }
)


def migrate_mc_result_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Upgrade an older MC result payload to the current schema.

    The only rule that matters scientifically: a payload that does not declare
    its semantics is *not* promoted to a fixed-draw SSB/delta model replay. It is
    stamped :data:`MC_SEMANTICS_UNDECLARED` and an unknown result space, so a
    reader, an exporter and the UI all describe it as legacy.
    """
    migrated = dict(payload)
    version = str(migrated.get("schema_version", "") or "")

    if not version or version == MC_RESULT_LEGACY_SCHEMA_VERSION:
        # Pre-schema payload: it may carry a subset of the historical
        # ``MCValidationResult`` field names and nothing else.
        migrated.setdefault("schema_name", MC_RESULT_SCHEMA_NAME)
        if not str(migrated.get("semantics_version", "") or ""):
            migrated["semantics_version"] = MC_SEMANTICS_UNDECLARED
        if not str(migrated.get("result_space", "") or ""):
            migrated["result_space"] = RESULT_SPACE_UNKNOWN
        if not str(migrated.get("effective_result_space", "") or ""):
            migrated["effective_result_space"] = migrated["result_space"]
        # ``n_iter`` was the historical single draw-count field; it recorded
        # the completed count.
        if "n_iter" in migrated:
            migrated.setdefault("completed_draws", migrated["n_iter"])
            migrated.setdefault("requested_draws", migrated["n_iter"])
        for legacy_key, current_key in (
            ("mc_lower_95", "mc_lower"),
            ("mc_upper_95", "mc_upper"),
        ):
            if legacy_key in migrated:
                migrated.setdefault(current_key, migrated[legacy_key])
        migrated["schema_version"] = MC_RESULT_SCHEMA_VERSION
        migrated["freshness_identity_version"] = ""
        migrated["migrated_from_schema_version"] = version or (
            MC_RESULT_LEGACY_SCHEMA_VERSION
        )
        return migrated

    if version in {"1.0", "1.1", "1.2", "1.3"}:
        # ``1.0`` predates ``budget_digest`` and ``1.1`` used an incomplete
        # configuration allow-list. Preserve every stored value, but never
        # compare either historical identity under the stricter 1.2 rules.
        if version == "1.0":
            migrated.setdefault("budget_digest", "")
        migrated["freshness_identity_version"] = ""
        migrated["schema_version"] = MC_RESULT_SCHEMA_VERSION
        migrated.setdefault("status", "completed")
        migrated.setdefault("reason_code", "")
        migrated["migrated_from_schema_version"] = version
        return migrated

    if version != MC_RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported MC result schema version {version!r}; this build "
            f"reads {', '.join(MC_RESULT_READABLE_SCHEMA_VERSIONS)}."
        )
    return migrated


def mc_record_freshness(
    record: "MCResultRecord",
    *,
    current_config_digest: str = "",
    current_budget_digest: str = "",
    current_input_digest: str = "",
    current_semantics_version: str = "",
    current_engine_b_semantics_version: str = "",
) -> str:
    """Return how a stored result stands against what is active *now*.

    A completed Monte Carlo result is a snapshot of one configuration applied
    to one set of measured inputs. Two independent things can invalidate it,
    and both must be tested:

    - the **uncertainty configuration** changed — caught by ``config_digest``;
    - the **measured inputs** changed while every setting stayed identical —
      a cycle mask, a cycle range or an outlier-filter edit — caught only by
      ``budget_digest``, because none of those touches a setting.

    Missing information yields :data:`MC_FRESHNESS_UNKNOWN`, never
    :data:`MC_FRESHNESS_CURRENT`: a result whose freshness cannot be
    established must not be presented as verified. A configuration mismatch
    is reported ahead of an input mismatch because it is the broader
    invalidation.

    A ``current`` verdict requires both complete identities. Some MC-only
    settings, notably contributor probability distributions, alter sampling
    without altering the analytical GUM budget. A caller with only the budget
    can still diagnose a budget mismatch as stale inputs, but cannot honestly
    certify a matching record as current.

    An Engine A record must also carry the semantics of the Sr correction method
    that is active now, supplied as ``current_semantics_version``. A retired,
    unknown or undeclared method is never current; with otherwise matching
    identities a record from the other supported method is stale, and a missing
    active method leaves freshness unknown (A025).

    An Engine B fixed-draw record must carry the semantics the ratio needs now,
    supplied as ``current_engine_b_semantics_version`` when the caller knows the
    sample: a ratio that now carries an applied Pb Hg correction is not answered
    by a replay that did not model it, and the reverse.
    """
    if (
        getattr(record, "is_engine_b", False)
        and str(getattr(record, "semantics_version", "") or "")
        not in ENGINE_B_CURRENT_FIXED_DRAW_SEMANTICS
    ):
        return MC_FRESHNESS_UNKNOWN

    semantics = str(getattr(record, "semantics_version", "") or "")
    is_engine_a = str(getattr(record, "engine", "") or "") == ENGINE_A_ENGINE
    if is_engine_a and semantics not in SUPPORTED_ENGINE_A_SEMANTICS:
        # Retired, unknown or undeclared Sr method: readable, never current.
        return MC_FRESHNESS_UNKNOWN

    stored_config = str(getattr(record, "config_digest", "") or "")
    stored_budget = str(getattr(record, "budget_digest", "") or "")
    current_config = str(current_config_digest or "")
    current_budget = str(current_budget_digest or "")

    identity_version = str(
        getattr(record, "freshness_identity_version", "") or ""
    )
    expected_identity = (
        ENGINE_A_FRESHNESS_IDENTITY_VERSION if is_engine_a else MC_FRESHNESS_IDENTITY_VERSION
    )
    if identity_version != expected_identity:
        return MC_FRESHNESS_UNKNOWN
    if (
        getattr(record, "is_engine_b", False)
        and current_engine_b_semantics_version
        and semantics != current_engine_b_semantics_version
    ):
        return MC_FRESHNESS_STALE_CONFIGURATION
    if stored_config and current_config and stored_config != current_config:
        return MC_FRESHNESS_STALE_CONFIGURATION
    if stored_budget and current_budget:
        if stored_budget != current_budget:
            return MC_FRESHNESS_STALE_INPUTS
    if stored_config and current_config and stored_budget and current_budget:
        if is_engine_a:
            active_semantics = str(current_semantics_version or "")
            if not active_semantics:
                return MC_FRESHNESS_UNKNOWN
            if active_semantics != semantics:
                return MC_FRESHNESS_STALE_CONFIGURATION
        if str(getattr(record, "engine", "")) in {ENGINE_A_ENGINE, "pb_tl_external_normalization"} and "pb_standard_calibration" not in semantics:
            if record.schema_version != MC_RESULT_SCHEMA_VERSION or not record.replay_snapshot_json or not current_input_digest:
                return MC_FRESHNESS_UNKNOWN
            if record.input_digest != current_input_digest:
                return MC_FRESHNESS_STALE_INPUTS
        return MC_FRESHNESS_CURRENT
    return MC_FRESHNESS_UNKNOWN


def mc_freshness_label(freshness: str) -> str:
    """Return the shared human-readable statement for a freshness state."""
    return MC_FRESHNESS_LABELS.get(freshness, MC_FRESHNESS_LABELS[MC_FRESHNESS_UNKNOWN])


def mc_record_freshness_for_budget(
    record: "MCResultRecord",
    budget: Any,
    uncertainty_config: Any = None,
    *,
    sample: Any = None,
    element_config: Any = None,
    current_input_digest: str = "",
) -> str:
    """Freshness of ``record`` against a live budget and configuration.

    The exporter-side convenience wrapper: every export format has the
    sample's current :class:`~domain.models.UncertaintyBudget` to hand, so
    each one can state freshness without re-running anything. Passing
    ``uncertainty_config=None`` can disclose a mismatching budget as stale,
    but a matching budget remains freshness-unknown because the configuration
    half of the identity was not supplied.
    """
    from domain.pb_calibration_records import governing_calibration_record
    from domain.uncertainty.monte_carlo import (
        engine_b_budget_digest,
        engine_b_config_digest,
        engine_b_semantics_for_ratio,
    )

    calibration = (
        governing_calibration_record(sample, str(getattr(record, "ratio_name", "") or ""))
        if sample is not None
        else None
    )
    if calibration is not None:
        # A Pb-standard-calibrated ratio is answered only by the extended Engine C
        # absolute-ratio replay of the calibration record in force now (C05).
        from domain.layer_status import APPLIED
        from domain.uncertainty.engine_pb_calibrated import calibration_identity_for_record
        from domain.uncertainty.monte_carlo import PB_CALIBRATED_MC_SEMANTICS

        if (
            str(getattr(record, "semantics_version", "") or "") != PB_CALIBRATED_MC_SEMANTICS
            or str(getattr(budget, "output_mode", "") or "").strip().lower() == "delta"
        ):
            return MC_FRESHNESS_STALE_CONFIGURATION
        stored_identity = str(getattr(record, "input_digest", "") or "")
        current_identity = str(getattr(budget, "replay_input_digest", "") or "")
        if calibration.status != APPLIED:
            return MC_FRESHNESS_STALE_INPUTS
        stored_calibration = stored_identity.split(".", 1)[0]
        if stored_calibration != calibration_identity_for_record(calibration):
            return MC_FRESHNESS_STALE_INPUTS
        if not stored_identity or not current_identity:
            return MC_FRESHNESS_UNKNOWN
        if stored_identity != current_identity:
            return MC_FRESHNESS_STALE_INPUTS

    current_config = ""
    if uncertainty_config is not None:
        try:
            current_config = engine_b_config_digest(uncertainty_config)
        except Exception:  # pragma: no cover - defensive, never fatal to export
            current_config = ""
    current_engine_b_semantics = ""
    if sample is not None and getattr(record, "is_engine_b", False):
        current_engine_b_semantics = engine_b_semantics_for_ratio(
            sample, str(getattr(record, "ratio_name", "") or ""),
        )
    return mc_record_freshness(
        record,
        current_input_digest=current_input_digest or (
            getattr(budget, "_current_mc_input_digests", {}).get(record.input_digest, "")
        ),
        current_config_digest=current_config,
        current_budget_digest=engine_b_budget_digest(budget),
        # Engine A needs the active Sr method; the sample's recorded method (and
        # the element configuration when a caller has it) supplies that evidence.
        current_semantics_version=active_engine_a_semantics(sample, element_config),
        current_engine_b_semantics_version=current_engine_b_semantics,
    )


def build_mc_result_record(
    result: Any,
    *,
    sample: Any = None,
    ratio_name: str = "",
    requested_draws: Optional[int] = None,
    app_version: str = APP_VERSION,
    software_commit: str = "",
    created_utc: Optional[str] = None,
) -> MCResultRecord:
    """Build a durable record from a completed :class:`MCCrossCheckResult`.

    Values are copied, never recomputed: this function performs no scientific
    arithmetic. ``sample`` and ``ratio_name`` are accepted so a caller that
    already knows the identity can supply it; when the result carries its own
    identity fields those win, because they were captured by the engine at run
    time and cannot have drifted since.
    """
    def take(name: str, default: Any = None) -> Any:
        return getattr(result, name, default)

    completed = int(take("completed_draws", 0) or take("n_iter", 0) or 0)
    requested = int(
        take("requested_draws", 0)
        or (requested_draws if requested_draws is not None else 0)
        or completed
    )

    record_sample_name = str(take("sample_name", "") or "")
    if not record_sample_name and sample is not None:
        record_sample_name = str(getattr(sample, "name", "") or "")
    record_run_number = int(take("sample_run_number", 0) or 0)
    if not record_run_number and sample is not None:
        record_run_number = int(getattr(sample, "run_number", 0) or 0)
    record_ratio = str(take("ratio_name", "") or "") or str(ratio_name or "")

    contributors: Tuple[MCContributorRecord, ...] = tuple(
        MCContributorRecord(
            name=spec.name,
            placement=spec.placement,
            distribution=spec.distribution,
            degrees_of_freedom=float(spec.degrees_of_freedom),
            type_ab=spec.type_ab,
            source_field=spec.source_field,
        )
        for spec in (take("contributor_specs", ()) or ())
    )
    if not contributors:
        # A legacy or non-Engine-B result exposes placements only.
        contributors = tuple(
            MCContributorRecord(
                name=str(name),
                placement=str(placement),
                distribution="",
                degrees_of_freedom=float("nan"),
                type_ab="",
                source_field="",
            )
            for name, placement in (take("contributor_placements", ()) or ())
        )

    result_space = str(take("result_space", "") or "") or RESULT_SPACE_UNKNOWN
    iteration_space = str(take("iteration_return_space", "") or "") or result_space
    effective_space = str(take("effective_result_space", "") or "") or result_space

    return MCResultRecord(
        status=str(take("status", "completed") or "completed"),
        reason_code=str(take("reason_code", "") or ""),
        execution_id=str(take("execution_id", "") or ""),
        sample_name=record_sample_name,
        sample_run_number=record_run_number,
        ratio_name=record_ratio,
        prev_std_label=str(take("prev_std_label", "") or ""),
        next_std_label=str(take("next_std_label", "") or ""),
        semantics_version=str(take("semantics_version", "") or ""),
        result_space=result_space,
        engine=str(take("engine", "") or ""),
        output_mode=str(take("output_mode", "") or ""),
        bracket_mode=str(take("bracket_mode", "") or ""),
        validation_mode=str(take("validation_mode", "") or ""),
        apply_ssb_kernel=take("apply_ssb_kernel"),
        delta_from_ssb=take("delta_from_ssb"),
        blank_uncertainty_input=str(take("blank_uncertainty_input", "") or ""),
        blank_placement=str(take("blank_placement", "") or ""),
        iteration_return_space=iteration_space,
        effective_result_space=effective_space,
        post_loop_transform_applied=bool(take("post_loop_transform_applied", False)),
        delta_reference_applied=_optional_float(take("delta_reference_applied")),
        absolute_scale_factor_applied=_optional_float(
            take("absolute_scale_factor_applied")
        ),
        nominal_reported_value=_optional_float(take("nominal_reported_value")),
        nominal_bracket_mean=_optional_float(take("nominal_bracket_mean")),
        nominal_delta_reference=_optional_float(take("nominal_delta_reference")),
        requested_draws=requested,
        completed_draws=completed,
        n_dropped=int(take("n_dropped", 0) or 0),
        seed=None if take("seed") is None else int(take("seed")),
        bit_generator=str(take("bit_generator", "") or ""),
        mc_mean=_optional_float(take("mc_mean")),
        mc_std=_optional_float(take("mc_std")),
        mc_std_diagnostic=_optional_float(take("mc_std_diagnostic")),
        mc_lower=_optional_float(take("mc_lower_95")),
        mc_upper=_optional_float(take("mc_upper_95")),
        moment_status=str(take("moment_status", "") or ""),
        min_type_a_dof=_optional_float(take("min_type_a_dof")),
        coverage_probability=float(
            take("coverage_probability", MC_COVERAGE_PROBABILITY)
        ),
        interval_convention=str(
            take("interval_convention", MC_INTERVAL_CONVENTION) or ""
        ),
        percentile_method=str(take("percentile_method", MC_PERCENTILE_METHOD) or ""),
        gum_center=_optional_float(take("gum_center")),
        gum_u_c=_optional_float(take("gum_u_c")),
        gum_u_expanded=_optional_float(take("gum_u_expanded")),
        gum_coverage_factor_k=_optional_float(take("gum_coverage_factor_k")),
        gum_lower=_optional_float(take("gum_lower")),
        gum_upper=_optional_float(take("gum_upper")),
        absolute_center_difference=_optional_float(
            take("absolute_center_difference")
        ),
        center_difference_u_c=_optional_float(take("center_difference_u_c")),
        std_difference_pct=_optional_float(take("std_difference_pct")),
        convergence_mean_pct=_optional_float(take("convergence_mean_pct")),
        convergence_std_pct=_optional_float(take("convergence_std_pct")),
        convergence_half_width_pct=_optional_float(
            take("convergence_half_width_pct")
        ),
        convergence_q025_pct=_optional_float(take("convergence_q025_pct")),
        convergence_q975_pct=_optional_float(take("convergence_q975_pct")),
        contributors=contributors,
        config_digest=str(take("config_digest", "") or ""),
        input_digest=str(take("input_digest", "") or ""),
        replay_snapshot_json=str(take("replay_snapshot_json", "") or ""),
        freshness_identity_version=(
            ENGINE_A_FRESHNESS_IDENTITY_VERSION
            if str(take("engine", "") or "") == ENGINE_A_ENGINE
            else MC_FRESHNESS_IDENTITY_VERSION
        ),
        budget_digest=str(take("budget_digest", "") or ""),
        app_version=str(app_version or ""),
        python_version=platform.python_version(),
        numpy_version=str(take("numpy_version", "") or ""),
        # Operating-system family only. No hostname, user name or path: a
        # persisted result must not carry personal or machine identity.
        os_platform=platform.system(),
        software_commit=str(software_commit or ""),
        warnings=tuple(str(item) for item in (take("warnings", ()) or ())),
        scope_note=str(take("scope_note", "") or ""),
        created_utc=created_utc or datetime.now(timezone.utc).isoformat(),
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def attach_mc_result(sample: Any, ratio_name: str, record: MCResultRecord) -> None:
    """Attach a completed record to a sample's per-ratio MC result mapping.

    This is the single write path. Rendering and exporting only read.

    The record's own ``sample_name``/``ratio_name`` must agree with the
    sample and ratio it is being filed under. Every exporter labels its rows
    from the *outer* sample and ratio key while the record carries its own
    identity internally, so a mismatch would produce an export row that
    contradicts itself — a Monte Carlo result attributed to the wrong sample
    or ratio. Making that a hard error here keeps the association structurally
    unbreakable instead of relying on every future caller to be careful.
    An empty identity on the record is permitted: legacy and non-Engine-B
    results do not always carry one.
    """
    if not ratio_name:
        raise ValueError("An MC result must be attached to a named ratio.")

    record_ratio = str(getattr(record, "ratio_name", "") or "")
    if record_ratio and record_ratio != ratio_name:
        raise ValueError(
            "Refusing to attach a Monte Carlo result to the wrong ratio: the "
            f"record was computed for {record_ratio!r} but is being filed "
            f"under {ratio_name!r}."
        )

    record_sample = str(getattr(record, "sample_name", "") or "")
    target_sample = str(getattr(sample, "name", "") or "")
    if record_sample and target_sample and record_sample != target_sample:
        raise ValueError(
            "Refusing to attach a Monte Carlo result to the wrong sample: the "
            f"record was computed for {record_sample!r} but is being filed "
            f"under {target_sample!r}."
        )

    record_run = int(getattr(record, "sample_run_number", 0) or 0)
    target_run = int(getattr(sample, "run_number", 0) or 0)
    if record_run and target_run and record_run != target_run:
        raise ValueError(
            "Refusing to attach a Monte Carlo result to the wrong run: the "
            f"record was computed for run {record_run} but is being filed "
            f"under run {target_run}."
        )

    if getattr(sample, "mc_results", None) is None:
        sample.mc_results = {}
    sample.mc_results[ratio_name] = record


def get_mc_result(sample: Any, ratio_name: str) -> Optional[MCResultRecord]:
    """Return the durable record for one sample/ratio, or ``None``."""
    mapping = getattr(sample, "mc_results", None) or {}
    return mapping.get(ratio_name)


def iter_mc_results(samples: Any):
    """Yield ``(sample, ratio_name, record)`` for every durable MC result."""
    for sample in samples or ():
        mapping = getattr(sample, "mc_results", None) or {}
        for ratio_name, record in mapping.items():
            yield sample, ratio_name, record


def has_mc_results(samples: Any) -> bool:
    """True when at least one sample carries a durable MC result."""
    for _ in iter_mc_results(samples):
        return True
    return False


def log_mc_results_exported(samples: Any, *, export_format: str) -> None:
    """Emit one bounded ``engine_b_mc.exported`` record per exported run.

    Called once by each exporter, after it has written its Monte Carlo
    payload. Deduplicating on the execution ID keeps the record count tied to
    the number of *runs* exported, not the number of rows written, so a large
    session cannot flood the log.

    Only Engine B runs carry an execution ID, and only they are reported under
    this Engine B event code. The payload is the execution ID, the
    result-schema version and a fixed format token: never an output path, a
    filename, a summary value or a draw array.
    """
    seen: set = set()
    for _sample, _ratio_name, record in iter_mc_results(samples):
        if not record.execution_id or record.execution_id in seen:
            continue
        seen.add(record.execution_id)
        log_engine_b_mc_exported(
            record.execution_id,
            result_schema_version=record.schema_version,
            export_format=export_format,
        )

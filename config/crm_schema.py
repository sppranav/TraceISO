"""Versioned schema for ``config/crm_library.json``.

The CRM library is a public serialized format, so it carries an explicit
version and a migration path (release plan Gate 2, items 1 and 14).

Schema history
--------------
``1.0``
    Element blocks with ``reference_materials`` only.
``2.0``
    Adds element-level ``reference_data`` (masses, natural ratios) and
    ``internal_normalization``.
``3.0``
    Adds record identity and value semantics so two records for the same
    material can be told apart and audited (review item A-3):

    * ``record_id`` - stable identity for a reference-material record and for
      an element-level natural ratio. Provenance stores these, not display
      names.
    * ``material_id`` - the underlying material. Two records may share it;
      that is what makes a same-material divergence check possible.
    * ``display_label`` - a label that names the *value concept*, not just the
      material, so "NIST SRM 987" cannot silently mean two different numbers.
    * ``value_kind`` - see :data:`VALUE_KINDS`.
    * ``source`` - a structured block (``citation``/``doi``/``url``) instead of
      a bare string.
    * ``normalization`` - the normalization convention a value was reported
      under, or ``None``.
    * ``uncertainty_semantics`` - see :data:`UNCERTAINTY_SEMANTICS`. This is
      what stops a stated source limit from being silently read as ``U, k=2``.
    * ``coverage_status`` - optional legacy metadata for imported records.
    * an uncertainty of ``null`` with ``k`` of ``null`` and
      ``uncertainty_semantics = "unassigned"``, meaning *unknown*. Numeric zero
      is never used for unknown, because ``u = 0`` asserts the quantity is
      exact.

Migration is one-way and in memory: a ``1.0`` or ``2.0`` file still loads,
with ``value_kind`` and ``uncertainty_semantics`` filled in conservatively.
A version this build does not know is rejected rather than guessed at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

#: The schema version this build writes.
CURRENT_SCHEMA_VERSION = "3.0"

#: Versions this build can read. Anything else - including a future version -
#: is rejected with an explicit message rather than parsed on a guess.
SUPPORTED_SCHEMA_VERSIONS: Tuple[str, ...] = ("1.0", "2.0", "3.0")

#: What kind of quantity a stored value is. ``unspecified`` is deliberate: it
#: is what a record gets when its source string does not establish a kind, and
#: it must never be upgraded to a specific claim by a code default.
VALUE_KINDS = frozenset({
    "certificate",
    "consensus",
    "representative_abundance",
    "assigned_correction",
    "normalization_convention",
    "laboratory_assigned",
    "unspecified",
})

#: How a stored uncertainty must be read.
#:
#: ``expanded_uncertainty``
#:     ``U`` at the stated coverage factor ``k``; ``u = U / k``.
#: ``standard_uncertainty``
#:     already ``k = 1``.
#: ``source_stated_limit``
#:     the source quotes a limit whose coverage semantics it does not state.
#:     ``k`` is ``None`` and the limit is not converted to a standard
#:     uncertainty automatically.
#: ``assigned_exact``
#:     stored as ``u = 0`` because the value is treated as exact where it is
#:     used (a normalization convention, for example).
#: ``unassigned``
#:     unknown. ``uncertainty`` and ``k`` are ``None`` and the quantity is
#:     omitted from propagation - never treated as zero.
#: ``derived``
#:     produced by propagation from another record rather than stored.
UNCERTAINTY_SEMANTICS = frozenset({
    "expanded_uncertainty",
    "standard_uncertainty",
    "source_stated_limit",
    "assigned_exact",
    "unassigned",
    "derived",
})

#: ``coverage_status`` values.
COVERAGE_CONFIRMED = "confirmed"
COVERAGE_CARRIED_FORWARD_UNCONFIRMED = "carried_forward_unconfirmed"


class UnsupportedSchemaVersionError(ValueError):
    """Raised for a CRM library version this build cannot read."""


class UnassignedUncertaintyError(ValueError):
    """Raised when a caller demands an uncertainty that is unassigned.

    Distinct from a plain missing-record ``ValueError`` so callers can report
    *unassigned* rather than *absent*, and so neither is silently read as zero.
    """


# ---------------------------------------------------------------------------
# Version handling
# ---------------------------------------------------------------------------

def _version_tuple(version: str) -> Optional[Tuple[int, ...]]:
    try:
        return tuple(int(part) for part in str(version).split("."))
    except (TypeError, ValueError):
        return None


def check_schema_version(version: object) -> str:
    """Return *version* as a supported schema string, or raise.

    A version newer than :data:`CURRENT_SCHEMA_VERSION` is named as such, so
    the failure reads as "this file is newer than this build" rather than as a
    corrupt file.
    """
    text = str(version).strip()
    if text in SUPPORTED_SCHEMA_VERSIONS:
        return text

    parsed = _version_tuple(text)
    current = _version_tuple(CURRENT_SCHEMA_VERSION)
    if parsed is not None and current is not None and parsed > current:
        raise UnsupportedSchemaVersionError(
            f"Unsupported CRM library version {text!r}: it is newer than this "
            f"build supports (highest known: {CURRENT_SCHEMA_VERSION}). "
            "Upgrade TraceISO rather than editing the file."
        )
    raise UnsupportedSchemaVersionError(
        f"Unsupported CRM library version {text!r}; expected one of "
        f"{list(SUPPORTED_SCHEMA_VERSIONS)}."
    )


# ---------------------------------------------------------------------------
# Row-level semantics
# ---------------------------------------------------------------------------

def default_semantics_for_k(k: object) -> str:
    """Return the semantics a pre-3.0 row implies from its coverage factor."""
    try:
        k_value = float(k)
    except (TypeError, ValueError):
        return "standard_uncertainty"
    if not math.isfinite(k_value) or k_value <= 0:
        return "standard_uncertainty"
    if abs(k_value - 1.0) < 1e-12:
        return "standard_uncertainty"
    return "expanded_uncertainty"


def normalize_semantics(value: object, *, k: object = None) -> str:
    """Return a known semantics token, falling back to the ``k`` implication."""
    text = str(value).strip() if value is not None else ""
    if text in UNCERTAINTY_SEMANTICS:
        return text
    return default_semantics_for_k(k)


def normalize_value_kind(value: object) -> str:
    """Return a known value kind, or ``unspecified``."""
    text = str(value).strip() if value is not None else ""
    return text if text in VALUE_KINDS else "unspecified"


def normalize_source(value: object) -> Dict[str, Optional[str]]:
    """Return a structured source block from either schema shape."""
    if isinstance(value, Mapping):
        citation = value.get("citation")
        return {
            "citation": "" if citation is None else str(citation),
            "doi": None if value.get("doi") is None else str(value.get("doi")),
            "url": None if value.get("url") is None else str(value.get("url")),
        }
    return {
        "citation": "" if value is None else str(value),
        "doi": None,
        "url": None,
    }


def is_unassigned(uncertainty: object, semantics: object = None) -> bool:
    """Return True when a row carries no uncertainty at all.

    ``None`` is unassigned. Numeric zero is *not*: zero is a claim that the
    quantity is exact, and the two must never collapse into each other.
    """
    if str(semantics).strip() == "unassigned":
        return True
    return uncertainty is None


def standard_uncertainty(
    uncertainty: object,
    k: object,
    semantics: object = None,
) -> Optional[float]:
    """Normalize a stored uncertainty to ``k = 1``.

    Returns ``None`` for an unassigned row and for any row whose numbers are
    not usable, so a caller can never divide an unknown into a number.
    """
    semantics_text = str(semantics).strip()
    if is_unassigned(uncertainty, semantics_text):
        return None
    # A stated limit is retained for provenance and display only. Without a
    # source-defined coverage model there is no defensible divisor that turns
    # it into a standard uncertainty.
    if semantics_text == "source_stated_limit":
        return None
    try:
        u_value = float(uncertainty)
        k_value = float(k)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(u_value) or u_value < 0:
        return None
    if not math.isfinite(k_value) or k_value <= 0:
        return None
    return u_value / k_value


def reciprocal_with_uncertainty(
    value: float,
    standard_u: Optional[float],
) -> Tuple[float, Optional[float]]:
    """Return ``(1/x, u(1/x))`` for a stored canonical ratio.

    ``u(1/x) = u(x) / x**2`` is the first-order propagation of ``y = 1/x``
    (GUM:2008 section 5.1.2, with a single input quantity). This is the only
    supported way to obtain the reciprocal orientation of a stored ratio: the
    reciprocal is never maintained as a second editable number.

    An unassigned input uncertainty propagates as unassigned, not as zero.
    """
    x = float(value)
    if not math.isfinite(x) or x == 0.0:
        raise ValueError(
            "A reciprocal ratio requires a finite non-zero canonical value."
        )
    inverse = 1.0 / x
    if standard_u is None:
        return inverse, None
    u = float(standard_u)
    if not math.isfinite(u) or u < 0:
        return inverse, None
    return inverse, u / (x * x)


def rb_natural_ratio_from_abundances(
    x_rb87: float = 0.2783,
    u_x_rb87: float = 0.0002,
) -> Tuple[float, float]:
    """Return ``87Rb/85Rb`` and its standard uncertainty from CIAAW abundances.

    The value ``0.385617`` is derived from the CIAAW representative isotopic
    abundances ``x(85Rb) = 0.7217(2)`` and
    ``x(87Rb) = 0.2783(2)``. Rubidium has two stable isotopes, so the two
    abundances are tied by ``x(85Rb) = 1 - x(87Rb)`` and are therefore fully
    anti-correlated. Propagating the single free quantity gives

        R      = x87 / (1 - x87)
        dR/dx  = 1 / (1 - x87)**2
        u(R)   = u(x87) / (1 - x87)**2

    Treating the two abundances as independent would understate ``u(R)``. This
    function exists so the stored pair is reproducible arithmetic rather than
    an assumed ``k = 1`` label.
    """
    x87 = float(x_rb87)
    if not 0.0 < x87 < 1.0:
        raise ValueError("x(87Rb) must lie strictly between 0 and 1.")
    x85 = 1.0 - x87
    ratio = x87 / x85
    u_ratio = float(u_x_rb87) / (x85 * x85)
    return ratio, u_ratio


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate_library(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Return *data* as a schema-3.0 payload.

    A 1.0/2.0 payload is upgraded in memory: every record gains an identity, a
    label, ``value_kind = "unspecified"`` and semantics implied by its stored
    ``k``. Nothing about the source is invented; an old file simply declares
    less than a new one.
    """
    version = check_schema_version(data.get("version", "1.0"))
    if version == CURRENT_SCHEMA_VERSION:
        return dict(data)

    migrated: Dict[str, Any] = {"version": CURRENT_SCHEMA_VERSION, "elements": {}}
    elements = data.get("elements") or {}
    if not isinstance(elements, Mapping):
        return migrated

    for symbol, block in elements.items():
        if not isinstance(block, Mapping):
            continue
        new_block: Dict[str, Any] = {"reference_materials": {}}

        for rm_name, rm in (block.get("reference_materials") or {}).items():
            if not isinstance(rm, Mapping):
                continue
            ratios: Dict[str, Any] = {}
            for ratio_name, row in (rm.get("ratios") or {}).items():
                if not isinstance(row, Mapping):
                    continue
                new_row = dict(row)
                # Only rows that actually carry an uncertainty get a semantics
                # declaration. A pre-3.0 row that simply omits the field is
                # *incomplete*, not a declaration that the uncertainty is
                # unassigned - the loader must still reject it.
                if "uncertainty" in row:
                    new_row.setdefault(
                        "uncertainty_semantics", default_semantics_for_k(row.get("k")),
                    )
                ratios[ratio_name] = new_row
            new_block["reference_materials"][rm_name] = {
                "record_id": f"crm:{str(symbol).lower()}:{_slug(rm_name)}:unspecified",
                "material_id": rm_name,
                "display_label": str(rm_name),
                "value_kind": "unspecified",
                "source": normalize_source(rm.get("source")),
                "normalization": None,
                "description": rm.get("description", ""),
                "ratios": ratios,
                "masses": rm.get("masses", {}),
            }

        if "default_reference_material" in block:
            new_block["default_reference_material"] = block["default_reference_material"]

        ref_data = block.get("reference_data")
        if isinstance(ref_data, Mapping):
            naturals: Dict[str, Any] = {}
            for ratio_name, row in (ref_data.get("natural_ratios") or {}).items():
                if not isinstance(row, Mapping):
                    continue
                new_row = {
                    "record_id": f"nat:{str(symbol).lower()}:{_slug(ratio_name)}",
                    "display_label": str(ratio_name),
                    "value_kind": "unspecified",
                    "source": normalize_source(row.get("source")),
                }
                # Carry the numeric fields across only where the old file
                # actually had them, so a missing field stays missing and is
                # rejected downstream rather than becoming "unassigned".
                for field_name in ("value", "uncertainty", "k"):
                    if field_name in row:
                        new_row[field_name] = row[field_name]
                if "uncertainty" in row:
                    new_row["uncertainty_semantics"] = default_semantics_for_k(
                        row.get("k"),
                    )
                naturals[ratio_name] = new_row
            new_block["reference_data"] = {
                "masses": ref_data.get("masses", {}),
                "natural_ratios": naturals,
            }

        internal = block.get("internal_normalization")
        if isinstance(internal, Mapping):
            new_internal = dict(internal)
            new_internal.setdefault("convention", "unspecified")
            new_block["internal_normalization"] = new_internal

        migrated["elements"][symbol] = new_block

    return migrated


def _slug(text: object) -> str:
    out: List[str] = []
    for ch in str(text).strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")


# ---------------------------------------------------------------------------
# Same-material divergence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DivergenceWarning:
    """One same-material advisory raised by :func:`same_material_warnings`."""

    code: str
    element: str
    material_id: str
    ratio_name: str
    record_ids: Tuple[str, ...]
    message: str


#: Same material, two different value concepts. Not an error: it is why the
#: records have distinguishable labels. Selecting between them is a scientific
#: choice the operator has to make knowingly (review item A-3).
DIFFERENT_VALUE_KIND = "DIFFERENT_VALUE_KIND"

#: The two values differ by more than their combined expanded uncertainty.
VALUE_DIVERGENCE = "VALUE_DIVERGENCE"

#: The two values differ but at least one carries no comparable uncertainty,
#: so compatibility cannot be assessed at all.
UNCERTAINTY_NOT_COMPARABLE = "UNCERTAINTY_NOT_COMPARABLE"

#: Coverage factor used when comparing two records. This is a display-level
#: compatibility screen, not a metrological conformity statement.
DIVERGENCE_COVERAGE_FACTOR = 2.0


def same_material_warnings(
    elements: Mapping[str, Any],
    *,
    coverage_factor: float = DIVERGENCE_COVERAGE_FACTOR,
) -> List[DivergenceWarning]:
    """Return advisories for records that share a ``material_id``.

    Two records of the same material that report the same ratio are compared
    on every ratio they share. The screen is deliberately simple and is not a
    conformity assessment: it exists so a library holding, for example, both
    an SRM 987 certificate value and an SRM 987 consensus value cannot present
    them as interchangeable.
    """
    warnings: List[DivergenceWarning] = []
    if not isinstance(elements, Mapping):
        return warnings

    for symbol, block in sorted(elements.items()):
        if not isinstance(block, Mapping):
            continue
        by_material: Dict[str, List[Tuple[str, Mapping[str, Any]]]] = {}
        for rm_name, rm in (block.get("reference_materials") or {}).items():
            if not isinstance(rm, Mapping):
                continue
            material_id = str(rm.get("material_id") or rm_name)
            by_material.setdefault(material_id, []).append((rm_name, rm))

        for material_id, records in sorted(by_material.items()):
            if len(records) < 2:
                continue
            for i in range(len(records)):
                for j in range(i + 1, len(records)):
                    warnings.extend(
                        _compare_records(
                            str(symbol),
                            material_id,
                            records[i],
                            records[j],
                            coverage_factor,
                        )
                    )
    return warnings


def _record_id(name: str, record: Mapping[str, Any]) -> str:
    return str(record.get("record_id") or name)


def _compare_records(
    symbol: str,
    material_id: str,
    left: Tuple[str, Mapping[str, Any]],
    right: Tuple[str, Mapping[str, Any]],
    coverage_factor: float,
) -> List[DivergenceWarning]:
    left_name, left_rm = left
    right_name, right_rm = right
    ids = (_record_id(left_name, left_rm), _record_id(right_name, right_rm))

    out: List[DivergenceWarning] = []
    left_kind = normalize_value_kind(left_rm.get("value_kind"))
    right_kind = normalize_value_kind(right_rm.get("value_kind"))
    if left_kind != right_kind:
        out.append(DivergenceWarning(
            code=DIFFERENT_VALUE_KIND,
            element=symbol,
            material_id=material_id,
            ratio_name="",
            record_ids=ids,
            message=(
                f"{material_id} has two records with different value kinds "
                f"({left_kind} vs {right_kind}). They are not interchangeable; "
                "select the one your measurement model requires."
            ),
        ))

    left_ratios = left_rm.get("ratios") or {}
    right_ratios = right_rm.get("ratios") or {}
    for ratio_name in sorted(set(left_ratios) & set(right_ratios)):
        left_row = left_ratios[ratio_name]
        right_row = right_ratios[ratio_name]
        if not isinstance(left_row, Mapping) or not isinstance(right_row, Mapping):
            continue
        try:
            left_value = float(left_row.get("value"))
            right_value = float(right_row.get("value"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(left_value) and math.isfinite(right_value)):
            continue
        difference = abs(left_value - right_value)
        if difference == 0.0:
            continue

        left_u = standard_uncertainty(
            left_row.get("uncertainty"),
            left_row.get("k"),
            left_row.get("uncertainty_semantics"),
        )
        right_u = standard_uncertainty(
            right_row.get("uncertainty"),
            right_row.get("k"),
            right_row.get("uncertainty_semantics"),
        )
        if left_u is None or right_u is None:
            out.append(DivergenceWarning(
                code=UNCERTAINTY_NOT_COMPARABLE,
                element=symbol,
                material_id=material_id,
                ratio_name=ratio_name,
                record_ids=ids,
                message=(
                    f"{material_id} {ratio_name}: the two records differ by "
                    f"{difference:.6g} but at least one carries no assigned "
                    "uncertainty, so their compatibility cannot be assessed."
                ),
            ))
            continue

        combined = math.hypot(left_u, right_u)
        if combined <= 0.0 or difference > coverage_factor * combined:
            out.append(DivergenceWarning(
                code=VALUE_DIVERGENCE,
                element=symbol,
                material_id=material_id,
                ratio_name=ratio_name,
                record_ids=ids,
                message=(
                    f"{material_id} {ratio_name}: the two records differ by "
                    f"{difference:.6g}, more than their combined expanded "
                    f"uncertainty at k={coverage_factor:g} "
                    f"({coverage_factor * combined:.6g})."
                ),
            ))
    return out

"""Certified Reference Material (CRM) library access.

The on-disk format is versioned; see :mod:`config.crm_schema` for the schema
history, the migration path, and the meaning of ``value_kind``,
``uncertainty_semantics`` and an *unassigned* uncertainty.
"""

import json
import logging
import math
import re
import threading
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from config.crm_schema import (
    COVERAGE_CARRIED_FORWARD_UNCONFIRMED,
    CURRENT_SCHEMA_VERSION,
    DivergenceWarning,
    UnassignedUncertaintyError,
    UnsupportedSchemaVersionError,
    check_schema_version,
    is_unassigned,
    migrate_library,
    normalize_semantics,
    normalize_source,
    normalize_value_kind,
    reciprocal_with_uncertainty,
    same_material_warnings,
    standard_uncertainty as _standard_uncertainty_from_semantics,
)


@dataclass
class NaturalRatio:
    """Element-level natural ratio with uncertainty metadata.

    ``uncertainty`` and ``k`` are ``None`` when the library declares the
    uncertainty *unassigned*. Callers must branch on that rather than
    substituting zero: ``u = 0`` asserts the quantity is exact, which is a
    stronger claim than "unknown".
    """

    value: float
    uncertainty: Optional[float] = 0.0
    k: Optional[float] = 1.0
    source: str = ""
    record_id: str = ""
    display_label: str = ""
    value_kind: str = "unspecified"
    uncertainty_semantics: str = "standard_uncertainty"
    coverage_status: str = ""
    source_doi: Optional[str] = None
    source_url: Optional[str] = None
    note: str = ""

    @property
    def is_uncertainty_unassigned(self) -> bool:
        """True when this row carries no uncertainty at all."""
        return is_unassigned(self.uncertainty, self.uncertainty_semantics)


@dataclass
class ReferenceData:
    """Element-level reference data used by correction engines."""

    masses: Dict[str, float] = field(default_factory=dict)
    natural_ratios: Dict[str, NaturalRatio] = field(default_factory=dict)


@dataclass
class CRM:
    """One certified ratio of one reference-material record.

    Identity is ``record_id``, not ``name``: two records may describe the same
    material under different value concepts (review item A-3), and provenance
    must be able to say which one produced a number.
    """

    name: str
    element: str
    ratio_name: str  # e.g., "7Li/6Li" or "11B/10B"
    ratio: float
    uncertainty: Optional[float] = 0.0  # stated uncertainty at coverage factor k
    k: Optional[float] = 1.0  # coverage factor accompanying uncertainty
    reference: str = ""
    record_id: str = ""
    material_id: str = ""
    display_label: str = ""
    value_kind: str = "unspecified"
    uncertainty_semantics: str = "standard_uncertainty"
    coverage_status: str = ""
    normalization: Optional[Dict[str, Any]] = None
    source_doi: Optional[str] = None

    @property
    def is_uncertainty_unassigned(self) -> bool:
        """True when this row carries no uncertainty at all."""
        return is_unassigned(self.uncertainty, self.uncertainty_semantics)

    @property
    def has_unconfirmed_coverage(self) -> bool:
        """True when the stored coverage divisor is not established by the source."""
        return self.coverage_status == COVERAGE_CARRIED_FORWARD_UNCONFIRMED

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


# JSON library loader (crm_library.json from CRM Manager)

_JSON_LIBRARY_PATH = Path(__file__).resolve().parent / "crm_library.json"
_json_cache: Optional[Dict[str, List[CRM]]] = None
_json_root_cache: Optional[Dict] = None
_json_cache_path: Optional[Path] = None
_json_root_cache_path: Optional[Path] = None
_log = logging.getLogger(__name__)
# item 72: protect process-global cache mutation against concurrent Streamlit sessions
_cache_lock = threading.Lock()

# A017: monotonic generation of the loaded library. Every consumer-side cache
# that memoizes resolved certificate contents — notably the runtime CRM LRU in
# ``domain.uncertainty.runtime`` — must include this in its key, because a
# certificate edit followed by :func:`reload_library` changes the numbers while
# every other key field (element symbol, material name) stays identical.
_library_generation = 0


def _normalize_isotope_label(label: str) -> str:
    """Normalize isotope labels to the JSON form (e.g. ``Sr86`` -> ``86Sr``)."""
    raw = str(label).strip()
    if not raw:
        return raw

    m = re.match(r"^([A-Za-z]+)(\d+)$", raw)
    if m:
        return f"{m.group(2)}{m.group(1)}"

    m = re.match(r"^(\d+)([A-Za-z]+)$", raw)
    if m:
        return f"{m.group(1)}{m.group(2)}"

    return raw


def _normalize_ratio_name(ratio_name: str) -> str:
    """Normalize ratio-name separators and isotope token order."""
    raw = str(ratio_name).strip().replace("\\", "/").replace("_", "/")
    if raw.count("/") != 1:
        return raw
    numerator, denominator = raw.split("/", 1)
    return (
        f"{_normalize_isotope_label(numerator)}/"
        f"{_normalize_isotope_label(denominator)}"
    )


def _load_json_root() -> Dict:
    """Load raw crm_library.json payload."""
    from config.recorded_dependencies import current_dependencies
    recorded = current_dependencies()
    if recorded is not None:
        return recorded.reference_library
    if not _JSON_LIBRARY_PATH.exists():
        _log.warning("Managed CRM library not found at '%s'.", _JSON_LIBRARY_PATH)
        return {}
    try:
        with open(_JSON_LIBRARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("Failed to read %s: %s", _JSON_LIBRARY_PATH, exc)
        return {}
    if not isinstance(data, dict):
        raise ValueError("CRM library must be a JSON object.")
    # Rejects an unknown or future version outright; upgrades 1.0/2.0 in
    # memory so an older library still loads (config.crm_schema).
    check_schema_version(data.get("version", "1.0"))
    data = migrate_library(data)
    elements = data.get("elements", {})
    if not isinstance(elements, dict):
        raise ValueError("CRM library 'elements' block must be a JSON object.")
    for element, payload in elements.items():
        if not isinstance(payload, dict):
            continue
        default_name = payload.get("default_reference_material")
        if default_name is None:
            continue
        reference_materials = payload.get("reference_materials", {})
        if (
            not isinstance(default_name, str)
            or not default_name.strip()
            or not isinstance(reference_materials, dict)
            or default_name not in reference_materials
        ):
            raise ValueError(
                f"Default reference material {default_name!r} for element "
                f"{element!r} is not present in reference_materials."
            )
    return data


def _get_json_root_cached() -> Dict:
    """Return cached raw ``crm_library.json`` contents (item 72: thread-safe)."""
    from config.recorded_dependencies import current_dependencies
    recorded = current_dependencies()
    if recorded is not None:
        return recorded.reference_library
    global _json_root_cache, _json_root_cache_path
    current_path = _JSON_LIBRARY_PATH.resolve()
    with _cache_lock:
        if _json_root_cache is None or _json_root_cache_path != current_path:
            _json_root_cache = _load_json_root()
            _json_root_cache_path = current_path
        return _json_root_cache


def _load_json_library() -> Dict[str, List[CRM]]:
    """Load ``crm_library.json`` into a flat ``{element: [CRM, ...]}`` mapping."""
    data = _load_json_root()
    result: Dict[str, List[CRM]] = {}
    elements = data.get("elements", {}) if isinstance(data, dict) else {}
    if not isinstance(elements, dict):
        _log.warning("CRM library 'elements' block is not a JSON object; ignoring.")
        return result

    for element, edata in elements.items():
        if not isinstance(edata, dict):
            _log.warning("CRM element entry %r is not a JSON object; skipping.", element)
            continue
        reference_materials = edata.get("reference_materials", {})
        if not isinstance(reference_materials, dict):
            _log.warning(
                "CRM element %r reference_materials block is not a JSON object; skipping.",
                element,
            )
            continue
        crm_list: List[CRM] = []
        for crm_name, crm_data in reference_materials.items():
            if not isinstance(crm_data, dict):
                _log.warning("CRM entry %r for %r is not a JSON object; skipping.", crm_name, element)
                continue
            source_block = normalize_source(crm_data.get("source"))
            source_citation = source_block.get("citation", "")
            record_id = str(crm_data.get("record_id", "") or crm_name)
            material_id = str(crm_data.get("material_id", "") or crm_name)
            display_label = str(crm_data.get("display_label", "") or crm_name)
            value_kind = normalize_value_kind(crm_data.get("value_kind"))
            normalization = crm_data.get("normalization")
            if normalization is not None and not isinstance(normalization, Mapping):
                normalization = None
            elif isinstance(normalization, Mapping):
                normalization = dict(normalization)
            ratios = crm_data.get("ratios", {})
            if not isinstance(ratios, dict):
                _log.warning(
                    "CRM entry %r for %r has non-object ratios; skipping.",
                    crm_name,
                    element,
                )
                continue
            for ratio_name, rdata in ratios.items():
                if not isinstance(rdata, dict):
                    _log.warning(
                        "CRM ratio %r for %r/%r is not a JSON object; skipping.",
                        ratio_name,
                        element,
                        crm_name,
                    )
                    continue
                missing_fields = [
                    field_name
                    for field_name in ("value", "uncertainty", "k")
                    if field_name not in rdata
                ]
                if missing_fields:
                    _log.warning(
                        "CRM ratio %r for %r/%r is missing required field(s) %s; skipping.",
                        ratio_name,
                        element,
                        crm_name,
                        ", ".join(missing_fields),
                    )
                    continue
                semantics = normalize_semantics(
                    rdata.get("uncertainty_semantics"), k=rdata.get("k"),
                )
                unassigned = is_unassigned(rdata.get("uncertainty"), semantics)
                try:
                    ratio = float(rdata["value"])
                except (KeyError, TypeError, ValueError) as exc:
                    _log.warning(
                        "CRM ratio %r for %r/%r has an invalid value: %s; skipping.",
                        ratio_name,
                        element,
                        crm_name,
                        exc,
                    )
                    continue
                if not math.isfinite(ratio) or ratio <= 0:
                    _log.warning(
                        "CRM ratio %r for %r/%r has a non-finite or non-positive "
                        "value; skipping.",
                        ratio_name,
                        element,
                        crm_name,
                    )
                    continue

                uncertainty: Optional[float]
                k: Optional[float]
                if unassigned:
                    # Unknown, not zero. Kept as a usable value row whose
                    # uncertainty is simply absent.
                    uncertainty = None
                    k = None
                    semantics = "unassigned"
                elif semantics == "source_stated_limit":
                    try:
                        uncertainty = float(rdata["uncertainty"])
                    except (KeyError, TypeError, ValueError) as exc:
                        _log.warning(
                            "CRM ratio %r for %r/%r has an invalid source-stated limit: %s; skipping.",
                            ratio_name, element, crm_name, exc,
                        )
                        continue
                    if not math.isfinite(uncertainty) or uncertainty < 0:
                        _log.warning(
                            "CRM ratio %r for %r/%r has an invalid source-stated limit; skipping.",
                            ratio_name, element, crm_name,
                        )
                        continue
                    k = None
                else:
                    try:
                        uncertainty = float(rdata["uncertainty"])
                        k = float(rdata["k"])
                    except (KeyError, TypeError, ValueError) as exc:
                        _log.warning(
                            "CRM ratio %r for %r/%r has invalid numeric fields: %s; skipping.",
                            ratio_name,
                            element,
                            crm_name,
                            exc,
                        )
                        continue
                    if (
                        not math.isfinite(uncertainty)
                        or uncertainty < 0
                        or not math.isfinite(k)
                        or k <= 0
                    ):
                        _log.warning(
                            "CRM ratio %r for %r/%r has non-finite or out-of-range values; skipping.",
                            ratio_name,
                            element,
                            crm_name,
                        )
                        continue
                crm_list.append(CRM(
                    name=crm_name,
                    element=element,
                    ratio_name=ratio_name,
                    ratio=ratio,
                    uncertainty=uncertainty,
                    k=k,
                    reference=source_citation,
                    record_id=record_id,
                    material_id=material_id,
                    display_label=display_label,
                    value_kind=value_kind,
                    uncertainty_semantics=semantics,
                    coverage_status=str(rdata.get("coverage_status", "") or ""),
                    normalization=normalization,
                    source_doi=source_block.get("doi"),
                ))
        if crm_list:
            result[element] = crm_list
    return result


def _get_database() -> Dict[str, List[CRM]]:
    """Return the managed CRM database loaded from ``crm_library.json`` (item 72: thread-safe).

    item 78: The CRM derivation graph is built lazily and is O(d * r) per
    element where d is the derivation depth and r is the number of stored
    ratios.  In practice d ≤ 2 for the supported elements; the cache ensures
    the graph is traversed at most once per process lifetime (or after
    :func:`reload_library` is called).
    """
    from config.recorded_dependencies import current_dependencies
    recorded = current_dependencies()
    if recorded is not None:
        if recorded.database is None:
            recorded.database = _load_json_library()
        return recorded.database
    global _json_cache, _json_cache_path
    current_path = _JSON_LIBRARY_PATH.resolve()
    with _cache_lock:
        if _json_cache is None or _json_cache_path != current_path:
            _json_cache = _load_json_library()
            _json_cache_path = current_path
            _log.debug(
                "Loaded CRM library from JSON (%d elements)",
                len(_json_cache),
            )
        return _json_cache


def library_generation() -> int:
    """Return the current generation of the loaded managed library.

    Increments on every :func:`reload_library`. Downstream memoized lookups
    key on it so a re-read of an edited certificate cannot be served from a
    cache whose only other key fields — element and material name — the edit
    deliberately preserved.
    """
    from config.recorded_dependencies import current_dependencies, recorded_library_generation
    recorded = current_dependencies()
    return recorded_library_generation(recorded) if recorded is not None else _library_generation


def reload_library() -> None:
    """Invalidate the JSON cache so the next query re-reads the file."""
    global _json_cache, _json_root_cache, _json_cache_path, _json_root_cache_path
    global _library_generation
    with _cache_lock:
        _json_cache = None
        _json_root_cache = None
        _json_cache_path = None
        _json_root_cache_path = None
        _library_generation += 1


def resolved_reference_snapshot(
    element: str, name: str,
) -> Tuple[Tuple[Any, ...], ...]:
    """Return the numerical contents of one reference material, as resolved.

    A record ID names *which* certificate an analysis used; it does not say
    what that certificate said. This returns the values actually consumed —
    per stored row, the record identity together with the ratio value, the
    stated uncertainty, its coverage factor and its uncertainty semantics — so
    a cache key or a provenance digest changes when an editable certificate's
    numbers change under an unchanged ID.

    ``None`` uncertainty and ``None`` k are preserved: an unassigned
    uncertainty is not zero, and flattening it here would assert exactness.
    """
    rows = []
    for crm in get_crm_records(element, name):
        rows.append((
            str(crm.record_id or ""),
            str(crm.material_id or ""),
            str(crm.ratio_name),
            float(crm.ratio),
            None if crm.uncertainty is None else float(crm.uncertainty),
            None if crm.k is None else float(crm.k),
            str(crm.uncertainty_semantics or ""),
            str(crm.value_kind or ""),
            str(crm.coverage_status or ""),
        ))
    return tuple(sorted(rows, key=lambda row: (row[0], row[2])))


def is_json_source_for_element(element: str) -> bool:
    """Return True if this element is currently sourced from JSON library (item 72).

    Routes through the main :func:`_get_database` cache so the JSON file is
    loaded at most once.
    """
    db = _get_database()
    return bool(db and element.capitalize() in db)


def get_internal_normalization(element: str) -> Optional[Tuple[str, float]]:
    """Get element-level internal-normalization setting from JSON library."""
    elem_data = (
        _get_json_root_cached().get("elements", {})
        .get(element.capitalize(), {})
    )
    norm_data = elem_data.get("internal_normalization", {})
    if not isinstance(norm_data, dict):
        return None

    ratio_name = str(norm_data.get("ratio_name", "")).strip().replace("\\", "/")
    if ratio_name.count("/") != 1:
        return None

    try:
        value = float(norm_data.get("value"))
    except (TypeError, ValueError):
        return None

    if (not math.isfinite(value)) or value <= 0:
        return None

    return ratio_name, value


def get_reference_data(element: str) -> Optional[ReferenceData]:
    """Get the full element-level ``reference_data`` block from JSON."""
    elem_data = (
        _get_json_root_cached().get("elements", {})
        .get(element.capitalize(), {})
    )
    raw = elem_data.get("reference_data")
    if not isinstance(raw, dict):
        return None

    masses: Dict[str, float] = {}
    for isotope, value in raw.get("masses", {}).items():
        try:
            mass = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(mass) and mass > 0:
            masses[str(isotope)] = mass

    natural_ratios: Dict[str, NaturalRatio] = {}
    for ratio_name, payload in raw.get("natural_ratios", {}).items():
        if not isinstance(payload, dict):
            _log.warning(
                "Natural ratio %r for %r is not a JSON object; skipping.",
                ratio_name,
                element,
            )
            continue
        missing_fields = [
            field_name
            for field_name in ("value", "uncertainty", "k")
            if field_name not in payload
        ]
        if missing_fields:
            _log.warning(
                "Natural ratio %r for %r is missing required field(s) %s; skipping.",
                ratio_name,
                element,
                ", ".join(missing_fields),
            )
            continue
        semantics = normalize_semantics(
            payload.get("uncertainty_semantics"), k=payload.get("k"),
        )
        unassigned = is_unassigned(payload.get("uncertainty"), semantics)
        try:
            value = float(payload["value"])
        except (TypeError, ValueError) as exc:
            _log.warning(
                "Natural ratio %r for %r has an invalid value: %s; skipping.",
                ratio_name,
                element,
                exc,
            )
            continue
        if not math.isfinite(value) or value <= 0:
            _log.warning(
                "Natural ratio %r for %r has a non-finite or non-positive value; skipping.",
                ratio_name,
                element,
            )
            continue

        uncertainty: Optional[float]
        k: Optional[float]
        if unassigned:
            # Unknown, not zero. The value stays usable; only its uncertainty
            # is absent, and every propagation path must omit it rather than
            # treat it as exact.
            uncertainty = None
            k = None
            semantics = "unassigned"
        elif semantics == "source_stated_limit":
            try:
                uncertainty = float(payload["uncertainty"])
            except (KeyError, TypeError, ValueError) as exc:
                _log.warning(
                    "Natural ratio %r for %r has an invalid source-stated limit: %s; skipping.",
                    ratio_name, element, exc,
                )
                continue
            if not math.isfinite(uncertainty) or uncertainty < 0:
                _log.warning(
                    "Natural ratio %r for %r has an invalid source-stated limit; skipping.",
                    ratio_name, element,
                )
                continue
            k = None
        else:
            try:
                uncertainty = float(payload["uncertainty"])
                k = float(payload["k"])
            except (TypeError, ValueError) as exc:
                _log.warning(
                    "Natural ratio %r for %r has invalid numeric fields: %s; skipping.",
                    ratio_name,
                    element,
                    exc,
                )
                continue
            if (
                not math.isfinite(uncertainty)
                or uncertainty < 0
                or not math.isfinite(k)
                or k <= 0
            ):
                _log.warning(
                    "Natural ratio %r for %r has non-finite or out-of-range values; skipping.",
                    ratio_name,
                    element,
                )
                continue

        source_block = normalize_source(payload.get("source"))
        normalized_name = _normalize_ratio_name(ratio_name)
        natural_ratios[normalized_name] = NaturalRatio(
            value=value,
            uncertainty=uncertainty,
            k=k,
            source=source_block.get("citation", ""),
            record_id=str(payload.get("record_id", "") or normalized_name),
            display_label=str(payload.get("display_label", "") or normalized_name),
            value_kind=normalize_value_kind(payload.get("value_kind")),
            uncertainty_semantics=semantics,
            coverage_status=str(payload.get("coverage_status", "") or ""),
            source_doi=source_block.get("doi"),
            source_url=source_block.get("url"),
            note=str(payload.get("note", "") or ""),
        )

    if not masses and not natural_ratios:
        return None

    return ReferenceData(masses=masses, natural_ratios=natural_ratios)


def scientific_library_snapshot() -> dict:
    """Copy the loaded reference contents, including masses and source metadata."""
    from copy import deepcopy
    return deepcopy(_get_json_root_cached())


def get_element_masses(element: str) -> Optional[Dict[str, float]]:
    """Get all managed isotope masses for an element."""
    reference_data = get_reference_data(element)
    if reference_data is None or not reference_data.masses:
        return None
    return dict(reference_data.masses)


def get_isotope_mass(isotope_label: str) -> Optional[float]:
    """Get a managed isotope mass for labels like ``86Sr`` or ``Sr86``."""
    target = _normalize_isotope_label(isotope_label)
    for element_data in _get_json_root_cached().get("elements", {}).values():
        reference_data = element_data.get("reference_data", {})
        masses = reference_data.get("masses", {}) if isinstance(reference_data, dict) else {}
        if target in masses:
            try:
                mass = float(masses[target])
            except (TypeError, ValueError):
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Skipping non-parseable mass value for isotope %r in element entry.",
                    target,
                )
                continue
            if math.isfinite(mass) and mass > 0:
                return mass
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Skipping non-finite or non-positive mass %r for isotope %r.",
                mass,
                target,
            )
            continue
    return None


def get_masses_for_isotopes(isotope_labels: Sequence[str]) -> Dict[str, float]:
    """Return managed masses for each requested isotope label."""
    masses: Dict[str, float] = {}
    missing: List[str] = []
    for label in isotope_labels:
        normalized = _normalize_isotope_label(label)
        mass = get_isotope_mass(normalized)
        if mass is None:
            missing.append(normalized)
        else:
            masses[normalized] = mass
    if missing:
        raise ValueError(
            f"Managed reference masses missing for isotope(s) {', '.join(missing)} "
            "in config/crm_library.json."
        )
    return masses


def require_isotope_mass(isotope_label: str) -> float:
    """Return a managed isotope mass or raise a clear ``ValueError``."""
    mass = get_isotope_mass(isotope_label)
    if mass is None:
        raise ValueError(
            f"Managed reference mass missing for isotope '{_normalize_isotope_label(isotope_label)}' "
            "in config/crm_library.json."
        )
    return mass


def get_natural_ratio_record(
    element: str,
    ratio_name: str,
) -> Optional[NaturalRatio]:
    """Return the full :class:`NaturalRatio` record, metadata included."""
    reference_data = get_reference_data(element)
    if reference_data is None:
        return None
    return reference_data.natural_ratios.get(_normalize_ratio_name(ratio_name))


def get_natural_ratio(
    element: str,
    ratio_name: str,
) -> Optional[Tuple[float, Optional[float], Optional[float]]]:
    """Get ``(value, uncertainty, k)`` for an element-level natural ratio.

    ``uncertainty`` and ``k`` are ``None`` when the record declares the
    uncertainty unassigned. A source-stated limit retains its numeric value
    while returning ``k=None``; callers must not treat that limit as a
    standard uncertainty.
    """
    ratio = get_natural_ratio_record(element, ratio_name)
    if ratio is None:
        return None
    if ratio.is_uncertainty_unassigned:
        return float(ratio.value), None, None
    return (
        float(ratio.value),
        float(ratio.uncertainty),
        None if ratio.k is None else float(ratio.k),
    )


def require_natural_ratio(
    element: str,
    ratio_name: str,
) -> Tuple[float, Optional[float], Optional[float]]:
    """Return a managed natural ratio payload or raise a clear ``ValueError``."""
    payload = get_natural_ratio(element, ratio_name)
    if payload is None:
        raise ValueError(
            f"Managed natural ratio '{_normalize_ratio_name(ratio_name)}' for element "
            f"'{element.capitalize()}' is missing in config/crm_library.json."
        )
    return payload


def derived_reciprocal_natural_ratio(
    element: str,
    stored_ratio_name: str,
) -> Optional[Tuple[float, Optional[float]]]:
    """Return the reciprocal of a stored natural ratio and its uncertainty.

    The reciprocal orientation is never stored as a second editable number;
    it is derived here from the canonical record. An uncertainty that cannot
    be converted to a standard uncertainty stays unavailable.
    """
    record = get_natural_ratio_record(element, stored_ratio_name)
    if record is None:
        return None
    return reciprocal_with_uncertainty(
        float(record.value),
        _standard_uncertainty_from_semantics(
            record.uncertainty, record.k, record.uncertainty_semantics,
        ),
    )


def get_natural_ratio_relative_uncertainty(
    element: str,
    ratio_name: str,
) -> Optional[float]:
    """Get relative standard uncertainty for an element-level natural ratio.

    ``None`` means "not available" — either the record is absent, or its
    uncertainty is unassigned. It never means zero.
    """
    record = get_natural_ratio_record(element, ratio_name)
    if record is None:
        return None
    value = float(record.value)
    if not math.isfinite(value) or value <= 0:
        return None
    standard = _standard_uncertainty_from_semantics(
        record.uncertainty, record.k, record.uncertainty_semantics,
    )
    if standard is None:
        return None
    return standard / value


def require_natural_ratio_relative_uncertainty(
    element: str,
    ratio_name: str,
) -> float:
    """Return relative standard uncertainty or raise a clear error.

    Raises :class:`UnassignedUncertaintyError` when the record exists but
    declares no uncertainty, so a caller can say *unassigned* rather than
    *missing* — and so neither can be quietly rounded down to zero.
    """
    normalized = _normalize_ratio_name(ratio_name)
    record = get_natural_ratio_record(element, ratio_name)
    if record is not None and record.is_uncertainty_unassigned:
        raise UnassignedUncertaintyError(
            f"Managed natural ratio '{normalized}' for element "
            f"'{element.capitalize()}' has an unassigned uncertainty. It is "
            "omitted from propagation; it is not zero."
        )
    value = get_natural_ratio_relative_uncertainty(element, ratio_name)
    if value is None:
        raise ValueError(
            f"Managed natural-ratio uncertainty for '{normalized}' "
            f"on element '{element.capitalize()}' is missing or invalid in config/crm_library.json."
        )
    return value


def library_divergence_warnings() -> List[DivergenceWarning]:
    """Return same-material advisories for the shipped CRM library."""
    root = _get_json_root_cached()
    return same_material_warnings(root.get("elements", {}) or {})


# Public query API

def get_crm_names(element: str) -> List[str]:
    """Get unique CRM names for an element (for dropdown)."""
    crms = _get_database().get(element.capitalize(), [])
    # Deduplicate (some CRMs have multiple ratios)
    unique = list(dict.fromkeys(crm.name for crm in crms))
    return unique


def get_crm(element: str, name: str) -> Optional[CRM]:
    """Get a specific CRM by element and name."""
    for crm in _get_database().get(element.capitalize(), []):
        if crm.name == name:
            return crm
    return None


def _get_crm_ratios_flat(
    element: str, name: str,
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Get CRM ratios as ``{ratio: (value, standard_uncertainty)}``."""
    result: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for crm in _get_database().get(element.capitalize(), []):
        if crm.name == name:
            standard_uncertainty = _standard_uncertainty_from_certificate(crm)
            if standard_uncertainty is not None:
                result[crm.ratio_name] = (crm.ratio, standard_uncertainty)
    return result


def get_crm_records(element: str, name: str) -> List[CRM]:
    """Return the full stored records for one reference material.

    Unlike :func:`get_crm_ratios` this keeps rows whose uncertainty is
    unassigned, so a UI or an export can show the value and say the
    uncertainty is unknown.
    """
    return [
        crm for crm in _get_database().get(element.capitalize(), [])
        if crm.name == name
    ]


def _standard_uncertainty_from_certificate(crm: CRM) -> Optional[float]:
    """Return certificate uncertainty normalized to k=1, or None if unusable."""
    return _standard_uncertainty_from_semantics(
        crm.uncertainty, crm.k, crm.uncertainty_semantics,
    )


def standard_uncertainty_from_values(
    uncertainty: object,
    coverage_factor: object,
    uncertainty_semantics: object = None,
) -> Optional[float]:
    """Normalize a stated uncertainty to k=1, failing closed on invalid input.

    An unassigned uncertainty or a source-stated limit without a defensible
    conversion returns ``None``. Neither is divided into a number or changed
    to zero.
    """
    return _standard_uncertainty_from_semantics(
        uncertainty, coverage_factor, uncertainty_semantics,
    )


# Derived-ratio solver (ported from data_loader.py)

# A derivation node is ``(value, sensitivities)``. ``sensitivities`` maps each
# *original* certificate ratio name to the partial derivative of the node value
# with respect to that certificate value. ``None`` marks an uncertainty the
# solver refuses to define (a degenerate step), which propagates outward as NaN.
_DerivationNode = Tuple[float, Optional[Dict[str, float]]]

_MAX_DERIVATION_DEPTH = 15


def _combined_sensitivities(
    first: Optional[Dict[str, float]],
    first_factor: float,
    second: Optional[Dict[str, float]],
    second_factor: float,
) -> Optional[Dict[str, float]]:
    """Add two scaled sensitivity maps, keeping one entry per original input.

    Summing before the caller squares is what makes a reused certificate input
    cancel when its net sensitivity is zero.
    """
    if first is None or second is None:
        return None
    merged: Dict[str, float] = {}
    for key, value in first.items():
        merged[key] = merged.get(key, 0.0) + value * first_factor
    for key, value in second.items():
        merged[key] = merged.get(key, 0.0) + value * second_factor
    return merged


def _uncertainty_from_sensitivities(
    sensitivities: Optional[Dict[str, float]],
    crm_ratios: Dict[str, Tuple[Optional[float], Optional[float]]],
) -> float:
    """Combine per-input contributions by root sum of squares.

    Distinct certificate inputs are still treated as independent: the managed
    CRM schema does not carry certificate covariance, so adding one would need
    new sourced metadata and a separately validated schema change. What is
    combined here is the *same* input reached through several paths, summed
    once before squaring.
    """
    if sensitivities is None:
        return float("nan")
    total = 0.0
    for key in sorted(sensitivities):
        entry = crm_ratios.get(key)
        if entry is None:
            return float("nan")
        _, input_uncertainty = entry
        if input_uncertainty is None:
            return float("nan")
        term = float(sensitivities[key]) * float(input_uncertainty)
        if not math.isfinite(term):
            return float("nan")
        total += term * term
    return math.sqrt(total)


def _derive_certified_ratio_node(
    target_ratio_str: str,
    crm_ratios: Dict[str, Tuple[Optional[float], Optional[float]]],
    derivation_stack: List[str],
    memo: Dict[str, Optional[_DerivationNode]],
) -> Optional[_DerivationNode]:
    """Recursively derive a certified ratio, tracking original-input sensitivities.

    Traversal order is fixed by ``(quality score, isotope name)`` so the chosen
    path, and therefore the reported value, does not depend on set iteration
    order, which is hash-seeded and differs between application starts.
    """
    if target_ratio_str in memo:
        return memo[target_ratio_str]
    if target_ratio_str in derivation_stack:
        return None
    if len(derivation_stack) > _MAX_DERIVATION_DEPTH:
        return None
    current_derivation_stack = derivation_stack + [target_ratio_str]
    try:
        num_target, den_target = target_ratio_str.split("/")
    except ValueError:
        memo[target_ratio_str] = None
        return None

    # Direct match
    if target_ratio_str in crm_ratios:
        val_direct, unc_direct = crm_ratios[target_ratio_str]
        if val_direct is not None and unc_direct is not None:
            node: _DerivationNode = (float(val_direct), {target_ratio_str: 1.0})
            memo[target_ratio_str] = node
            return node

    # Inverse match: d(1/x)/dx = -1/x**2
    inv_target_ratio_str = f"{den_target}/{num_target}"
    if inv_target_ratio_str in crm_ratios:
        val_inv, unc_inv = crm_ratios[inv_target_ratio_str]
        if val_inv is not None and unc_inv is not None and val_inv != 0:
            node = (
                1.0 / val_inv,
                {inv_target_ratio_str: -1.0 / (val_inv ** 2)},
            )
            memo[target_ratio_str] = node
            return node

    # Derivation via intermediate isotope
    all_isotopes_in_crm: set = set()
    for r_str_loop in crm_ratios.keys():
        try:
            n_loop, d_loop = r_str_loop.split("/")
            all_isotopes_in_crm.add(n_loop)
            all_isotopes_in_crm.add(d_loop)
        except ValueError:
            continue

    # Sort intermediates to prefer direct paths
    def intermediate_quality_score(intermediate_C: str) -> float:
        if intermediate_C == num_target or intermediate_C == den_target:
            return float("inf")
        direct_ratios_available = 0
        path1_direct = (
            f"{num_target}/{intermediate_C}" in crm_ratios
            and f"{den_target}/{intermediate_C}" in crm_ratios
        )
        path2_direct = (
            f"{num_target}/{intermediate_C}" in crm_ratios
            and f"{intermediate_C}/{den_target}" in crm_ratios
        )
        if path1_direct:
            direct_ratios_available += 2
        if path2_direct:
            direct_ratios_available += 1
        return -direct_ratios_available

    sorted_intermediates = sorted(
        all_isotopes_in_crm,
        key=lambda candidate: (intermediate_quality_score(candidate), candidate),
    )

    for intermediate_isotope_C in sorted_intermediates:
        if (
            intermediate_isotope_C == num_target
            or intermediate_isotope_C == den_target
        ):
            continue

        # Path 1: A/B = (A/C) / (B/C)
        res1_p1 = _derive_certified_ratio_node(
            f"{num_target}/{intermediate_isotope_C}",
            crm_ratios, current_derivation_stack, memo,
        )
        if res1_p1:
            res2_p1 = _derive_certified_ratio_node(
                f"{den_target}/{intermediate_isotope_C}",
                crm_ratios, current_derivation_stack, memo,
            )
            if res2_p1:
                val1, sens1 = res1_p1
                val2, sens2 = res2_p1
                if val2 == 0:
                    continue
                derived_val = val1 / val2
                if math.isnan(val1) or math.isnan(val2) or val1 == 0:
                    # Preserved refusal: a zero numerator leaves the relative
                    # form undefined, and the result is reported as unusable.
                    derived_sens: Optional[Dict[str, float]] = None
                else:
                    # d(a/b) = da/b - a*db/b**2
                    derived_sens = _combined_sensitivities(
                        sens1, 1.0 / val2,
                        sens2, -val1 / (val2 ** 2),
                    )
                node = (derived_val, derived_sens)
                memo[target_ratio_str] = node
                return node

        # Path 2: A/B = (A/C) * (C/B)
        res1_p2 = _derive_certified_ratio_node(
            f"{num_target}/{intermediate_isotope_C}",
            crm_ratios, current_derivation_stack, memo,
        )
        if res1_p2:
            res2_p2 = _derive_certified_ratio_node(
                f"{intermediate_isotope_C}/{den_target}",
                crm_ratios, current_derivation_stack, memo,
            )
            if res2_p2:
                val1_p2, sens1_p2 = res1_p2
                val2_p2, sens2_p2 = res2_p2
                derived_val_p2 = val1_p2 * val2_p2
                if math.isnan(val1_p2) or math.isnan(val2_p2):
                    derived_sens_p2: Optional[Dict[str, float]] = None
                else:
                    # d(a*b) = b*da + a*db
                    derived_sens_p2 = _combined_sensitivities(
                        sens1_p2, val2_p2,
                        sens2_p2, val1_p2,
                    )
                node = (derived_val_p2, derived_sens_p2)
                memo[target_ratio_str] = node
                return node

    memo[target_ratio_str] = None
    return None


def _derive_certified_ratio(
    target_ratio_str: str,
    crm_ratios: Dict[str, Tuple[Optional[float], Optional[float]]],
    derivation_stack: List[str],
    memo: Dict[str, Optional[_DerivationNode]],
) -> Optional[Tuple[float, float]]:
    """Recursively derive a certified ratio as ``(value, standard uncertainty)``.

    Sensitivities are accumulated back to the original certificate inputs and
    only then squared, so an input appearing on both sides of a quotient (and
    therefore cancelling) carries no uncertainty into the result. Distinct
    certificate inputs remain independent (covariance = 0); the managed CRM
    schema does not carry certificate covariance.
    """
    node = _derive_certified_ratio_node(
        target_ratio_str, crm_ratios, derivation_stack, memo,
    )
    if node is None:
        return None
    value, sensitivities = node
    return value, _uncertainty_from_sensitivities(sensitivities, crm_ratios)


# Public API (enhanced with derivation)

def _validated_derived_value(
    result: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """Reject non-finite or negative-uncertainty derived certificate values."""
    if result is None:
        return None
    try:
        value, uncertainty = map(float, result)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(value) or not math.isfinite(uncertainty) or uncertainty < 0:
        return None
    return value, uncertainty

def get_crm_ratios(
    element: str,
    name: str,
    derive: bool = True,
) -> Dict[str, Tuple[float, float, float]]:
    """Get certified ratios as ``(value, uncertainty, k)`` tuples.

    Direct rows retain their stated certificate uncertainty and coverage
    factor. Derived rows contain propagated standard uncertainty with ``k=1``.
    """
    # Collect directly-certified ratios
    result: Dict[str, Tuple[float, float, float]] = {}
    crm_data: Dict[str, Tuple[float, float]] = {}  # for derivation solver
    for crm in _get_database().get(element.capitalize(), []):
        if crm.name == name:
            if crm.is_uncertainty_unassigned:
                # Fails closed rather than publishing the value with u = 0,
                # which would assert the certificate is exact.
                _log.warning(
                    "CRM record %r ratio %r declares an unassigned uncertainty; "
                    "the row is excluded from certified-value resolution "
                    "instead of being treated as exact.",
                    crm.record_id or crm.name,
                    crm.ratio_name,
                )
                continue
            result[crm.ratio_name] = (crm.ratio, crm.uncertainty, crm.k)
            standard_uncertainty = _standard_uncertainty_from_certificate(crm)
            if standard_uncertainty is not None:
                crm_data[crm.ratio_name] = (crm.ratio, standard_uncertainty)

    if not derive:
        return result

    # Derive inverse ratios and any multi-hop ratios reachable from the
    # directly certified set.  Use the existing _derive_certified_ratio
    # solver which handles inverses and A/C / B/C derivation paths.
    memo: Dict[str, object] = {}
    # Collect all isotopes that appear in direct ratio names
    isotopes: set = set()
    for ratio_str in list(crm_data.keys()):
        parts = ratio_str.split("/")
        if len(parts) == 2:
            isotopes.update(parts)

    # Try every A/B combination reachable from known isotopes
    sorted_isotopes = sorted(isotopes)
    for num in sorted_isotopes:
        for den in sorted_isotopes:
            if num == den:
                continue
            target = f"{num}/{den}"
            if target in result:
                continue
            derived = _validated_derived_value(
                _derive_certified_ratio(target, crm_data, [], memo)
            )
            if derived is not None:
                val, unc = derived
                result[target] = (val, unc, 1.0)

    return result


def derive_certified_value(
    element: str,
    ratio_str: str,
    crm_name: Optional[str] = None,
) -> Optional[Tuple[float, float]]:
    """Return ``(value, standard_uncertainty)`` via the recursive solver."""
    if crm_name is None:
        crm_name = get_default_crm_name(element)
    if crm_name is None:
        return None

    flat = _get_crm_ratios_flat(element, crm_name)
    if not flat:
        return None

    memo: Dict[str, Optional[Tuple[float, float]]] = {}
    return _validated_derived_value(
        _derive_certified_ratio(ratio_str, flat, [], memo)
    )


def get_all_certified_ratios(
    element: str,
    crm_name: Optional[str] = None,
) -> Dict[str, CRM]:
    """Return direct certified ratios for one CRM as ``{ratio_name: CRM}``."""
    if crm_name is None:
        crm_name = get_default_crm_name(element)
    if crm_name is None:
        return {}

    rows = _get_database().get(element.capitalize(), [])
    result: Dict[str, CRM] = {
        crm.ratio_name: crm for crm in rows if crm.name == crm_name
    }
    return result


def get_default_crm_name(element: str) -> Optional[str]:
    """Get an explicit, order-independent default CRM name for an element."""
    element_key = element.capitalize()
    payload = _get_json_root_cached().get("elements", {}).get(element_key, {})
    if isinstance(payload, dict):
        configured = payload.get("default_reference_material")
        if isinstance(configured, str) and configured.strip():
            return configured

    names = get_crm_names(element_key)
    if len(names) == 1:
        return names[0]
    if len(names) > 1:
        _log.warning(
            "Element %r has multiple reference materials but no explicit "
            "default_reference_material; no order-dependent fallback was selected.",
            element_key,
        )
    return None


def list_elements() -> List[str]:
    """List all elements with available CRMs."""
    return sorted(_get_database().keys())

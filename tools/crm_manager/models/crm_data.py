"""CRM data model - load / save / validate crm_library.json."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from config.crm_schema import (
    CURRENT_SCHEMA_VERSION,
    UNCERTAINTY_SEMANTICS,
    VALUE_KINDS,
    is_unassigned,
    migrate_library,
    standard_uncertainty,
    normalize_semantics,
    normalize_source,
    normalize_value_kind,
)
from config.reference_materials import NaturalRatio, ReferenceData
from domain.ratio_utils import normalize_ratio_name as _norm_ratio
from tools.crm_manager.constants import DEFAULT_CRM_LIBRARY_PATH

# Default location in the repo-level config/ directory.
_DEFAULT_PATH = DEFAULT_CRM_LIBRARY_PATH


def _ratio_lookup_key(ratio_name: object) -> str:
    """The key the application's accessors will actually look under."""
    text = str(ratio_name)
    return _norm_ratio(text) or text.strip()


def _ratio_name_is_resolvable(ratio_name: object) -> bool:
    """Can a canonical ``A/B`` request ever match this authored name?

    ``normalize_ratio_name`` maps documented aliases — ``Sr88/Sr86``, the
    legacy ``_`` and ``\\`` separators, surrounding space — onto the canonical
    form, and those must keep working. What cannot work is a name with no
    single separator or with an empty side: it normalizes to itself, no
    canonical request matches it, and the quantity is absent to the
    application while still sitting in the file (audit A089). Counting
    separators alone is not enough, because ``/86Sr`` has exactly one.
    """
    key = _ratio_lookup_key(ratio_name)
    if key.count("/") != 1:
        return False
    numerator, denominator = key.split("/")
    return bool(numerator.strip()) and bool(denominator.strip())


def _ratio_name_collisions(
    ratio_names: Iterable[str],
) -> List[Tuple[str, str, str]]:
    """Authored names that share one lookup key, as ``(later, first, key)``.

    Two spellings of the same ratio are two rows in the file and one row to
    the accessor, so whichever loses is silently absent — the same failure as
    an unmatchable name, reached from the other direction.
    """
    first_seen: Dict[str, str] = {}
    collisions: List[Tuple[str, str, str]] = []
    for name in ratio_names:
        key = _ratio_lookup_key(name)
        if key in first_seen and first_seen[key] != str(name):
            collisions.append((str(name), first_seen[key], key))
        else:
            first_seen.setdefault(key, str(name))
    return collisions


# Dataclasses

@dataclass
class CertifiedRatio:
    """A single certified isotope ratio with uncertainty.

    ``uncertainty`` and ``k`` are ``None`` when the row declares the
    uncertainty *unassigned*. That is not the same as ``0.0``, which claims
    the value is exact.
    """
    value: float
    uncertainty: Optional[float] = 0.0
    k: Optional[float] = 2.0  # coverage factor
    uncertainty_semantics: str = "expanded_uncertainty"
    coverage_status: str = ""
    note: str = ""

    @property
    def is_uncertainty_unassigned(self) -> bool:
        return is_unassigned(self.uncertainty, self.uncertainty_semantics)


@dataclass
class ReferenceMaterial:
    """A reference-material record.

    A record is one *value concept* for a material, not the material itself:
    ``material_id`` is shared by records describing the same material, while
    ``record_id``, ``display_label`` and ``value_kind`` keep them apart
    (review item A-3).
    """
    name: str
    element: str
    source: str = ""
    description: str = ""
    ratios: Dict[str, CertifiedRatio] = field(default_factory=dict)
    masses: Dict[str, float] = field(default_factory=dict)
    record_id: str = ""
    material_id: str = ""
    display_label: str = ""
    value_kind: str = "unspecified"
    source_doi: Optional[str] = None
    source_url: Optional[str] = None
    normalization: Optional[Dict[str, object]] = None

    def ratio_names(self) -> List[str]:
        return list(self.ratios.keys())


@dataclass
class InternalNormalization:
    """Element-level internal-normalization definition."""

    ratio_name: str
    value: float
    convention: str = "unspecified"



@dataclass
class ElementEntry:
    """All CRMs for a single element."""
    symbol: str
    default_reference_material: Optional[str] = None
    reference_data: Optional[ReferenceData] = None
    internal_normalization: Optional[InternalNormalization] = None
    reference_materials: Dict[str, ReferenceMaterial] = field(default_factory=dict)

    def crm_names(self) -> List[str]:
        return list(self.reference_materials.keys())


@dataclass
class CRMLibrary:
    """Top-level container for the entire CRM library."""
    version: str = CURRENT_SCHEMA_VERSION
    elements: Dict[str, ElementEntry] = field(default_factory=dict)

    # Convenience helpers

    def element_symbols(self) -> List[str]:
        return sorted(self.elements.keys())

    def get_element(self, symbol: str) -> Optional[ElementEntry]:
        return self.elements.get(symbol)

    def get_crm(self, symbol: str, crm_name: str) -> Optional[ReferenceMaterial]:
        elem = self.elements.get(symbol)
        if elem:
            return elem.reference_materials.get(crm_name)
        return None

    def add_element(self, symbol: str) -> ElementEntry:
        if symbol not in self.elements:
            self.elements[symbol] = ElementEntry(symbol=symbol)
        return self.elements[symbol]

    def add_crm(
        self, symbol: str, crm: ReferenceMaterial, *, replace: bool = False,
    ) -> None:
        """Store *crm* under ``symbol``.

        Refuses to overwrite an existing record of the same name unless the
        caller says that is what it means. Assigning to the dictionary key
        unconditionally let "Add CRM" on an existing name silently replace a
        complete certificate with an empty one (audit A081).
        """
        elem = self.add_element(symbol)
        if not replace and crm.name in elem.reference_materials:
            raise ValueError(
                f"{symbol} already has a reference material named "
                f"{crm.name!r}. Open that record to edit it, or choose "
                "another name."
            )
        elem.reference_materials[crm.name] = crm

    def remove_crm(self, symbol: str, crm_name: str) -> bool:
        elem = self.elements.get(symbol)
        if elem and crm_name in elem.reference_materials:
            del elem.reference_materials[crm_name]
            if elem.default_reference_material == crm_name:
                elem.default_reference_material = None
            # Only auto-remove the element entry if it has no reference_data
            # and no internal_normalization — otherwise element-level data
            # (masses, natural ratios) would be silently destroyed.
            if (
                not elem.reference_materials
                and elem.reference_data is None
                and elem.internal_normalization is None
            ):
                del self.elements[symbol]
            return True
        return False

    def remove_element(self, symbol: str) -> bool:
        if symbol in self.elements:
            del self.elements[symbol]
            return True
        return False

    def total_crms(self) -> int:
        return sum(len(e.reference_materials) for e in self.elements.values())

    def deep_copy(self) -> "CRMLibrary":
        """Return an independent deep copy (for undo / discard)."""
        return copy.deepcopy(self)


# Serialisation

def _source_out(citation: str, doi: object, url: object) -> dict:
    return {
        "citation": citation or "",
        "doi": doi if doi else None,
        "url": url if url else None,
    }


def _library_to_dict(lib: CRMLibrary) -> dict:
    """Convert library to a schema-3.0 JSON-serialisable dict.

    Every field the schema defines is written back, so editing one ratio in
    the manager cannot silently drop a record id, a value kind, a source DOI
    or an uncertainty-semantics declaration.
    """
    out = {"version": CURRENT_SCHEMA_VERSION, "elements": {}}
    for sym, elem in sorted(lib.elements.items()):
        crms = {}
        for crm_name, crm in elem.reference_materials.items():
            ratios = {}
            for rname, r in crm.ratios.items():
                unassigned = r.is_uncertainty_unassigned
                row = {
                    "value": r.value,
                    "uncertainty": None if unassigned else r.uncertainty,
                    "k": None if unassigned else r.k,
                    "uncertainty_semantics": (
                        "unassigned" if unassigned else r.uncertainty_semantics
                    ),
                }
                if r.coverage_status:
                    row["coverage_status"] = r.coverage_status
                if r.note:
                    row["note"] = r.note
                ratios[rname] = row
            crms[crm_name] = {
                "record_id": crm.record_id or crm_name,
                "material_id": crm.material_id or crm_name,
                "display_label": crm.display_label or crm_name,
                "value_kind": normalize_value_kind(crm.value_kind),
                "source": _source_out(crm.source, crm.source_doi, crm.source_url),
                "normalization": crm.normalization,
                "description": crm.description,
                "ratios": ratios,
                "masses": dict(crm.masses),
            }
        elem_out = {"reference_materials": crms}
        if elem.default_reference_material:
            elem_out["default_reference_material"] = elem.default_reference_material
        if elem.reference_data is not None:
            naturals = {}
            for rname, r in elem.reference_data.natural_ratios.items():
                unassigned = r.is_uncertainty_unassigned
                row = {
                    "record_id": r.record_id or rname,
                    "display_label": r.display_label or rname,
                    "value_kind": normalize_value_kind(r.value_kind),
                    "value": r.value,
                    "uncertainty": None if unassigned else r.uncertainty,
                    "k": None if unassigned else r.k,
                    "uncertainty_semantics": (
                        "unassigned" if unassigned else r.uncertainty_semantics
                    ),
                    "source": _source_out(r.source, r.source_doi, r.source_url),
                }
                if r.coverage_status:
                    row["coverage_status"] = r.coverage_status
                if r.note:
                    row["note"] = r.note
                naturals[rname] = row
            elem_out["reference_data"] = {
                "masses": dict(elem.reference_data.masses),
                "natural_ratios": naturals,
            }
        if elem.internal_normalization is not None:
            elem_out["internal_normalization"] = {
                "ratio_name": elem.internal_normalization.ratio_name,
                "value": elem.internal_normalization.value,
                "convention": elem.internal_normalization.convention,
            }
        out["elements"][sym] = elem_out
    return out


def _dict_to_library(data: dict) -> CRMLibrary:
    """Parse a dict (from JSON) into a CRMLibrary.

    A 1.0 or 2.0 payload is migrated to 3.0 first, so the manager always works
    on one shape and always saves the current schema.
    """
    data = migrate_library(data)
    lib = CRMLibrary(version=data.get("version", CURRENT_SCHEMA_VERSION))
    for sym, elem_data in data.get("elements", {}).items():
        ref_data = None
        ref_data_block = elem_data.get("reference_data")
        if isinstance(ref_data_block, dict):
            masses: Dict[str, float] = {}
            for isotope, value in ref_data_block.get("masses", {}).items():
                try:
                    mass = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(mass) and mass > 0:
                    masses[str(isotope)] = mass

            natural_ratios: Dict[str, NaturalRatio] = {}
            for ratio_name, ratio_data in ref_data_block.get("natural_ratios", {}).items():
                if not isinstance(ratio_data, dict):
                    continue
                semantics = normalize_semantics(
                    ratio_data.get("uncertainty_semantics"), k=ratio_data.get("k"),
                )
                unassigned = is_unassigned(ratio_data.get("uncertainty"), semantics)
                try:
                    value = float(ratio_data.get("value"))
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(value) or value <= 0:
                    continue
                uncertainty: Optional[float]
                k: Optional[float]
                if unassigned:
                    uncertainty = None
                    k = None
                    semantics = "unassigned"
                elif semantics == "source_stated_limit":
                    try:
                        uncertainty = float(ratio_data.get("uncertainty"))
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(uncertainty) or uncertainty < 0:
                        continue
                    k = None
                else:
                    try:
                        uncertainty = float(ratio_data.get("uncertainty", 0.0))
                        k = float(ratio_data.get("k", 1.0))
                    except (TypeError, ValueError):
                        continue
                    if (
                        not math.isfinite(uncertainty)
                        or not math.isfinite(k)
                        or uncertainty < 0
                        or k <= 0
                    ):
                        continue
                key = _norm_ratio(str(ratio_name)) or str(ratio_name).strip()
                source_block = normalize_source(ratio_data.get("source"))
                natural_ratios[key] = NaturalRatio(
                    value=value,
                    uncertainty=uncertainty,
                    k=k,
                    source=source_block.get("citation", ""),
                    record_id=str(ratio_data.get("record_id", "") or key),
                    display_label=str(ratio_data.get("display_label", "") or key),
                    value_kind=normalize_value_kind(ratio_data.get("value_kind")),
                    uncertainty_semantics=semantics,
                    coverage_status=str(ratio_data.get("coverage_status", "") or ""),
                    source_doi=source_block.get("doi"),
                    source_url=source_block.get("url"),
                    note=str(ratio_data.get("note", "") or ""),
                )

            if masses or natural_ratios:
                ref_data = ReferenceData(
                    masses=masses,
                    natural_ratios=natural_ratios,
                )

        internal_norm_data = elem_data.get("internal_normalization")
        internal_norm = None
        if isinstance(internal_norm_data, dict):
            ratio_name = str(internal_norm_data.get("ratio_name", "")).strip()
            value = internal_norm_data.get("value")
            if ratio_name and value is not None:
                # The same acceptance rule as
                # ``config.reference_materials.get_internal_normalization``:
                # finite, positive, and in A/B form. Reading a value the
                # application treats as absent is how the manager came to
                # believe it had saved one (audit A089).
                try:
                    norm_value = float(value)
                except (TypeError, ValueError):
                    norm_value = float("nan")
                normalized_name = (
                    _norm_ratio(ratio_name.replace("\\", "/")) or ratio_name
                )
                if (
                    math.isfinite(norm_value)
                    and norm_value > 0
                    and normalized_name.count("/") == 1
                ):
                    internal_norm = InternalNormalization(
                        ratio_name=normalized_name,
                        value=norm_value,
                        convention=str(
                            internal_norm_data.get("convention", "") or "unspecified"
                        ),
                    )

        elem = ElementEntry(
            symbol=sym,
            default_reference_material=elem_data.get("default_reference_material"),
            reference_data=ref_data,
            internal_normalization=internal_norm,
        )
        for crm_name, crm_data in elem_data.get("reference_materials", {}).items():
            ratios = {}
            for rname, rdata in crm_data.get("ratios", {}).items():
                semantics = normalize_semantics(
                    rdata.get("uncertainty_semantics"), k=rdata.get("k"),
                )
                unassigned = is_unassigned(rdata.get("uncertainty"), semantics)
                try:
                    value = float(rdata["value"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(value) or value <= 0:
                    continue
                uncertainty: Optional[float]
                k: Optional[float]
                if unassigned:
                    uncertainty = None
                    k = None
                    semantics = "unassigned"
                elif semantics == "source_stated_limit":
                    try:
                        uncertainty = float(rdata.get("uncertainty"))
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(uncertainty) or uncertainty < 0:
                        continue
                    k = None
                else:
                    try:
                        uncertainty = float(rdata.get("uncertainty", 0.0))
                        k = float(rdata.get("k", 2.0))
                    except (TypeError, ValueError):
                        continue
                    if (
                        not math.isfinite(uncertainty)
                        or not math.isfinite(k)
                        or uncertainty < 0
                        or k <= 0
                    ):
                        continue
                ratios[rname] = CertifiedRatio(
                    value=value,
                    uncertainty=uncertainty,
                    k=k,
                    uncertainty_semantics=semantics,
                    coverage_status=str(rdata.get("coverage_status", "") or ""),
                    note=str(rdata.get("note", "") or ""),
                )
            source_block = normalize_source(crm_data.get("source"))
            normalization = crm_data.get("normalization")
            crm = ReferenceMaterial(
                name=crm_name,
                element=sym,
                source=source_block.get("citation", ""),
                description=crm_data.get("description", ""),
                ratios=ratios,
                masses=crm_data.get("masses", {}),
                record_id=str(crm_data.get("record_id", "") or crm_name),
                material_id=str(crm_data.get("material_id", "") or crm_name),
                display_label=str(crm_data.get("display_label", "") or crm_name),
                value_kind=normalize_value_kind(crm_data.get("value_kind")),
                source_doi=source_block.get("doi"),
                source_url=source_block.get("url"),
                normalization=(
                    dict(normalization) if isinstance(normalization, dict) else None
                ),
            )
            elem.reference_materials[crm_name] = crm
        lib.elements[sym] = elem
    return lib


# File I/O

def _unknown_vocabulary_in_payload(data: dict) -> List[str]:
    """Name every explicit vocabulary token a current-schema payload gets wrong.

    Only for a payload already at the current schema. A 1.0/2.0 file declares
    less rather than declaring something wrong, and its established migration
    supplies the missing tokens; running this on one would reject files the
    manager is meant to upgrade.

    The tokens have to be read here, off the raw payload, because
    ``_dict_to_library`` normalizes an unknown ``value_kind`` to
    ``unspecified`` and reinterprets unknown semantics from ``k``. After that
    the model looks valid and the next save writes the substitution as though
    the author had chosen it.
    """
    if str(data.get("version", "1.0")) != CURRENT_SCHEMA_VERSION:
        return []

    unknown: List[str] = []

    def check(where: str, value_kind: object, semantics: object) -> None:
        if value_kind is not None and str(value_kind) not in VALUE_KINDS:
            unknown.append(f"{where}: unknown value_kind {str(value_kind)!r}")
        if semantics is not None and str(semantics) not in UNCERTAINTY_SEMANTICS:
            unknown.append(
                f"{where}: unknown uncertainty_semantics {str(semantics)!r}"
            )

    for symbol, elem_data in (data.get("elements") or {}).items():
        if not isinstance(elem_data, dict):
            continue
        ref_block = elem_data.get("reference_data") or {}
        for ratio_name, ratio_data in (ref_block.get("natural_ratios") or {}).items():
            if isinstance(ratio_data, dict):
                check(
                    f"{symbol}: natural ratio {ratio_name}",
                    ratio_data.get("value_kind"),
                    ratio_data.get("uncertainty_semantics"),
                )
        for crm_name, crm_data in (elem_data.get("reference_materials") or {}).items():
            if not isinstance(crm_data, dict):
                continue
            check(f"{symbol}/{crm_name}", crm_data.get("value_kind"), None)
            for ratio_name, ratio_data in (crm_data.get("ratios") or {}).items():
                if isinstance(ratio_data, dict):
                    check(
                        f"{symbol}/{crm_name}: {ratio_name}",
                        None,
                        ratio_data.get("uncertainty_semantics"),
                    )
    return unknown


def load_library(path: Optional[Path] = None) -> CRMLibrary:
    """Load a CRM library from JSON.  Returns empty library if file missing.

    Refuses a current-schema file whose declared vocabulary is outside the
    schema, rather than normalizing it and letting the substitution be saved
    back as the author's choice. Checking assignments in memory is not import
    protection: a file authored elsewhere reaches the parser directly.
    """
    path = path or _DEFAULT_PATH
    if not path.exists():
        return CRMLibrary()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    unknown = _unknown_vocabulary_in_payload(data)
    if unknown:
        listed = "; ".join(sorted(unknown)[:8])
        if len(unknown) > 8:
            listed += f"; and {len(unknown) - 8} more"
        raise CRMPayloadError(
            f"{path.name} declares vocabulary outside the schema, which would "
            f"be silently replaced on load: {listed}. Correct the file before "
            "opening it."
        )
    return _dict_to_library(data)


class CRMLibraryConflictError(RuntimeError):
    """The library on disk changed since this editor read it."""


class CRMPayloadError(ValueError):
    """The payload about to be written would not survive being read back."""


def _dropped_by_the_reader(data: dict) -> List[str]:
    """Name everything in *data* the library reader would silently discard.

    ``_dict_to_library`` applies the same acceptance rules as the application's
    reference accessor: a non-finite or non-positive value, an uncertainty that
    cannot be normalized, a normalization outside A/B form. Anything it drops
    is a value the application would treat as absent no matter what the file
    says, so writing it is worse than refusing to.
    """
    reloaded = _dict_to_library(copy.deepcopy(data))
    missing: List[str] = []
    for symbol, elem_data in data.get("elements", {}).items():
        elem = reloaded.get_element(symbol)
        if elem is None:
            missing.append(f"element {symbol}")
            continue
        if elem_data.get("internal_normalization") is not None:
            if elem.internal_normalization is None:
                missing.append(f"{symbol}: internal normalization")
        ref_block = elem_data.get("reference_data") or {}
        for ratio_name in (ref_block.get("natural_ratios") or {}):
            stored = (
                elem.reference_data.natural_ratios
                if elem.reference_data is not None
                else {}
            )
            key = _norm_ratio(str(ratio_name)) or str(ratio_name).strip()
            if key not in stored:
                missing.append(f"{symbol}: natural ratio {ratio_name}")
        for isotope in (ref_block.get("masses") or {}):
            stored_masses = (
                elem.reference_data.masses if elem.reference_data is not None else {}
            )
            if str(isotope) not in stored_masses:
                missing.append(f"{symbol}: mass {isotope}")
        for crm_name, crm_data in (elem_data.get("reference_materials") or {}).items():
            crm = elem.reference_materials.get(crm_name)
            if crm is None:
                missing.append(f"{symbol}/{crm_name}")
                continue
            for ratio_name in (crm_data.get("ratios") or {}):
                if ratio_name not in crm.ratios:
                    missing.append(f"{symbol}/{crm_name}: {ratio_name}")
    return missing


def _authoring_contract_error(lib: CRMLibrary) -> Optional[str]:
    """Describe the authored library's validation errors, or ``None``.

    Run on the **model**, before ``_library_to_dict``. That ordering is the
    point: the serializer calls ``normalize_value_kind`` on its way out, so an
    unknown ``value_kind`` is already ``unspecified`` by the time any check of
    the serialized text could look at it. A payload-only gate is structurally
    unable to see that class of error, however carefully it is written.
    """
    errors = [issue for issue in validate_library(lib) if issue.level == "error"]
    if not errors:
        return None
    listed = "\n".join(
        f"  [{issue.element}] {issue.crm}: {issue.message}" for issue in errors[:8]
    )
    if len(errors) > 8:
        listed += f"\n  ... and {len(errors) - 8} more error(s)"
    return (
        f"The library has {len(errors)} validation error(s) and was not "
        f"written:\n\n{listed}\n\nCorrect them and save again. Warnings, "
        "records with no ratios, and quantities declared unassigned do not "
        "block a save."
    )


def _serialize_library(lib: CRMLibrary) -> str:
    """Serialize *lib*, refusing anything the application could not honour.

    Three stages, in this order, because each catches what the next cannot.
    First the authored model is validated, since serialization normalizes some
    invalid tokens into valid ones. Then ``allow_nan=False``: ``NaN`` and
    ``Infinity`` are not JSON, and the accessor reads a normalization written
    that way as absent. Then the text is read back through the library parser,
    which covers everything else it filters (audit A089).
    """
    contract_error = _authoring_contract_error(lib)
    if contract_error is not None:
        raise CRMPayloadError(contract_error)

    data = _library_to_dict(lib)
    try:
        text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        raise CRMPayloadError(
            "The library contains a value that is not a number (NaN or "
            f"Infinity) and cannot be written: {exc}. Correct the value, or "
            "clear it to record the quantity as unassigned."
        ) from exc

    dropped = _dropped_by_the_reader(json.loads(text))
    if dropped:
        listed = "; ".join(sorted(dropped)[:8])
        if len(dropped) > 8:
            listed += f"; and {len(dropped) - 8} more"
        raise CRMPayloadError(
            "These entries would be discarded when the library is read back, "
            f"so they must not be saved as if they were in effect: {listed}."
        )
    return text + "\n"


def crm_library_revision(path: Optional[Path] = None) -> str:
    """Return the content revision of the library file (``""`` if absent)."""
    from config.managed_file_writes import file_content_revision

    return file_content_revision(path or _DEFAULT_PATH)


def save_library(
    lib: CRMLibrary,
    path: Optional[Path] = None,
    *,
    expected_revision: Optional[str] = None,
    warnings: Optional[List] = None,
) -> str:
    """Save a CRM library as one serialized cross-process transaction.

    Two managers can have the same library open — the desktop launchers allow
    it — so the expected revision is re-read while an exclusive lock is held
    and the replacement happens under that same lock. Without it, the second
    save wrote a whole-library snapshot taken before the first one landed and
    silently reverted it. Returns the revision written.

    Pass *warnings* to collect ``ManagedWriteWarning`` records for backup
    maintenance that did not succeed. The save still commits; the caller can
    then say so while reporting that the recovery copy did not move.
    """
    path = path or _DEFAULT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _serialize_library(lib)

    from config.managed_file_writes import commit_managed_text

    return commit_managed_text(
        path,
        text,
        expected_revision=expected_revision,
        conflict_error=CRMLibraryConflictError,
        conflict_message=(
            "The CRM library was modified by another session since it was "
            "loaded here. Reload it and reapply this change."
        ),
        # One-deep recovery copy, rotated only once the new file has
        # committed. See commit_managed_text.
        backup_path=path.with_name(path.name + ".bak"),
        warnings=warnings,
    )


# Validation

@dataclass
class ValidationIssue:
    """A single validation finding."""
    level: str  # "warning" or "error"
    element: str
    crm: str
    message: str


def _reciprocal_uncertainty_issues(
    symbol: str,
    crm_name: str,
    ratio_name: str,
    inverse_name: str,
    ratio: CertifiedRatio,
    inverse: CertifiedRatio,
) -> List[ValidationIssue]:
    """Warn when a stored reciprocal pair's uncertainties disagree.

    For ``y = 1/x`` the standard uncertainties must satisfy
    ``u(y) = u(x) / x**2``. A stored pair that fails this is two independently
    maintained numbers pretending to be one quantity (review item A-5).
    """
    u_forward = standard_uncertainty(
        ratio.uncertainty, ratio.k, ratio.uncertainty_semantics,
    )
    u_inverse = standard_uncertainty(
        inverse.uncertainty, inverse.k, inverse.uncertainty_semantics,
    )
    if u_forward is None or u_inverse is None:
        return []
    try:
        expected = u_forward / (float(ratio.value) ** 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return []
    if expected <= 0 and u_inverse <= 0:
        return []
    tolerance = 0.01 * max(expected, u_inverse)
    if abs(expected - u_inverse) <= tolerance:
        return []
    return [ValidationIssue(
        "warning", symbol, crm_name,
        f"{inverse_name}: stored standard uncertainty {u_inverse:.6g} is not "
        f"the reciprocal transform of {ratio_name} "
        f"(u/x^2 = {expected:.6g}). Store one canonical orientation and derive "
        "the reciprocal with its uncertainty.",
    )]


def _uncertainty_issues(
    symbol: str,
    crm_name: str,
    ratio_name: str,
    uncertainty: Optional[float],
    k: Optional[float],
    semantics: str,
) -> List[ValidationIssue]:
    """Validate one uncertainty/coverage pair, unassigned rows included.

    An unassigned row must carry no numbers at all. A row that declares
    itself unassigned while storing ``0`` is rejected outright: that is
    exactly the zero-means-unknown confusion the schema exists to prevent
    rather than zero.
    """
    issues: List[ValidationIssue] = []
    if semantics == "source_stated_limit":
        if uncertainty is None:
            issues.append(ValidationIssue(
                "error", symbol, crm_name,
                f"{ratio_name}: a source-stated limit must retain its quoted value.",
            ))
            return issues
        try:
            u_value = float(uncertainty)
        except (TypeError, ValueError):
            u_value = float("nan")
        if not math.isfinite(u_value) or u_value < 0:
            issues.append(ValidationIssue(
                "error", symbol, crm_name,
                f"{ratio_name}: source-stated limit must be finite and >= 0.",
            ))
        if k is not None:
            issues.append(ValidationIssue(
                "error", symbol, crm_name,
                f"{ratio_name}: source-stated limit has no established coverage factor; leave k blank.",
            ))
        return issues

    unassigned = is_unassigned(uncertainty, semantics)
    if unassigned:
        if uncertainty is not None:
            issues.append(ValidationIssue(
                "error", symbol, crm_name,
                f"{ratio_name}: an unassigned uncertainty must be blank, not "
                f"{uncertainty!r}. Zero would claim the value is exact.",
            ))
        if k is not None:
            issues.append(ValidationIssue(
                "error", symbol, crm_name,
                f"{ratio_name}: an unassigned uncertainty has no coverage "
                "factor; leave k blank.",
            ))
        return issues

    if uncertainty is None or k is None:
        issues.append(ValidationIssue(
            "error", symbol, crm_name,
            f"{ratio_name}: uncertainty and k must both be given unless the "
            "row is declared unassigned.",
        ))
        return issues

    try:
        u_value = float(uncertainty)
        k_value = float(k)
    except (TypeError, ValueError):
        issues.append(ValidationIssue(
            "error", symbol, crm_name, f"{ratio_name}: non-numeric uncertainty or k.",
        ))
        return issues

    if (not math.isfinite(u_value)) or u_value < 0:
        issues.append(ValidationIssue(
            "error", symbol, crm_name,
            f"{ratio_name}: uncertainty must be a finite value >= 0.",
        ))
    if (not math.isfinite(k_value)) or k_value <= 0:
        issues.append(ValidationIssue(
            "error", symbol, crm_name,
            f"{ratio_name}: k must be a finite value > 0.",
        ))
    return issues


def _vocabulary_issues(
    symbol: str,
    crm_name: str,
    ratio_name: str,
    value_kind: Optional[str],
    semantics: Optional[str],
) -> List[ValidationIssue]:
    """Reject value kinds and uncertainty semantics outside the schema."""
    issues: List[ValidationIssue] = []
    if value_kind is not None and value_kind not in VALUE_KINDS:
        issues.append(ValidationIssue(
            "error", symbol, crm_name,
            f"{ratio_name}: unknown value_kind {value_kind!r}; expected one of "
            f"{sorted(VALUE_KINDS)}.",
        ))
    if semantics is not None and semantics not in UNCERTAINTY_SEMANTICS:
        issues.append(ValidationIssue(
            "error", symbol, crm_name,
            f"{ratio_name}: unknown uncertainty_semantics {semantics!r}; expected "
            f"one of {sorted(UNCERTAINTY_SEMANTICS)}.",
        ))
    return issues


def validate_library(lib: CRMLibrary) -> List[ValidationIssue]:
    """Run consistency checks on the library."""
    issues: List[ValidationIssue] = []
    for sym, elem in lib.elements.items():
        if (
            elem.default_reference_material is not None
            and elem.default_reference_material not in elem.reference_materials
        ):
            issues.append(ValidationIssue(
                "error",
                sym,
                "(element)",
                "Default reference material must name an existing CRM.",
            ))
        if elem.reference_data is not None:
            for isotope, mass in elem.reference_data.masses.items():
                try:
                    mass_value = float(mass)
                except (TypeError, ValueError):
                    mass_value = float("nan")
                if (not math.isfinite(mass_value)) or mass_value <= 0:
                    issues.append(ValidationIssue(
                        "error",
                        sym,
                        "(element)",
                        f"{isotope}: mass must be a finite value > 0.",
                    ))

            for later, first, key in _ratio_name_collisions(
                elem.reference_data.natural_ratios
            ):
                issues.append(ValidationIssue(
                    "error",
                    sym,
                    "(element)",
                    f"{later}: natural ratio resolves to the same lookup key "
                    f"'{key}' as {first}, so only one of them would be "
                    "readable. Keep one spelling.",
                ))

            for ratio_name, ratio in elem.reference_data.natural_ratios.items():
                if not _ratio_name_is_resolvable(ratio_name):
                    issues.append(ValidationIssue(
                        "error",
                        sym,
                        "(element)",
                        f"{ratio_name}: natural ratio must be in A/B format.",
                    ))
                try:
                    value = float(ratio.value)
                except (TypeError, ValueError):
                    value = float("nan")
                if (not math.isfinite(value)) or value <= 0:
                    issues.append(ValidationIssue(
                        "error",
                        sym,
                        "(element)",
                        f"{ratio_name}: value must be a finite value > 0.",
                    ))
                issues.extend(
                    _uncertainty_issues(
                        sym,
                        "(element)",
                        ratio_name,
                        ratio.uncertainty,
                        ratio.k,
                        ratio.uncertainty_semantics,
                    )
                )
                issues.extend(
                    _vocabulary_issues(
                        sym,
                        "(element)",
                        ratio_name,
                        ratio.value_kind,
                        ratio.uncertainty_semantics,
                    )
                )

        if elem.internal_normalization is not None:
            if not _ratio_name_is_resolvable(elem.internal_normalization.ratio_name):
                issues.append(ValidationIssue(
                    "error",
                    sym,
                    "(element)",
                    "Internal normalization ratio must be in A/B format.",
                ))
            try:
                norm_value = float(elem.internal_normalization.value)
            except (TypeError, ValueError):
                norm_value = float("nan")
            if (not math.isfinite(norm_value)) or norm_value <= 0:
                issues.append(ValidationIssue(
                    "error",
                    sym,
                    "(element)",
                    "Internal normalization value must be a finite value > 0.",
                ))

        for crm_name, crm in elem.reference_materials.items():
            if not crm.ratios:
                issues.append(ValidationIssue(
                    "warning", sym, crm_name, "No certified ratios defined",
                ))
            issues.extend(
                _vocabulary_issues(sym, crm_name, "(record)", crm.value_kind, None)
            )
            for later, first, key in _ratio_name_collisions(crm.ratios):
                issues.append(ValidationIssue(
                    "error", sym, crm_name,
                    f"{later}: resolves to the same lookup key '{key}' as "
                    f"{first}, so only one of them would be readable. Keep one "
                    "spelling.",
                ))
            for rname, r in crm.ratios.items():
                if not _ratio_name_is_resolvable(rname):
                    issues.append(ValidationIssue(
                        "error", sym, crm_name,
                        f"{rname}: certified ratio must be in A/B format.",
                    ))
                if not math.isfinite(r.value) or r.value <= 0:
                    issues.append(ValidationIssue(
                        "error", sym, crm_name,
                        f"{rname}: value must be finite and > 0",
                    ))
                issues.extend(
                    _uncertainty_issues(
                        sym,
                        crm_name,
                        rname,
                        r.uncertainty,
                        r.k,
                        r.uncertainty_semantics,
                    )
                )
                issues.extend(
                    _vocabulary_issues(
                        sym, crm_name, rname, None, r.uncertainty_semantics,
                    )
                )
                # Check inverse consistency: if A/B and B/A both exist
                parts = rname.split("/")
                if len(parts) == 2:
                    inv = f"{parts[1]}/{parts[0]}"
                    if inv in crm.ratios and rname < inv:
                        # Storing both orientations lets the two numbers drift
                        # apart. Review item A-5 found exactly that, and found
                        # it in the *uncertainties* as well as the values, so
                        # both are checked here.
                        other = crm.ratios[inv]
                        product = r.value * other.value
                        if abs(product - 1.0) > 0.01:
                            issues.append(ValidationIssue(
                                "warning", sym, crm_name,
                                f"{rname} × {inv} = {product:.6f} (expected ≈ 1.0). "
                                "Store one canonical orientation and derive the "
                                "reciprocal.",
                            ))
                        issues.extend(
                            _reciprocal_uncertainty_issues(
                                sym, crm_name, rname, inv, r, other,
                            )
                        )
    return issues


@dataclass(frozen=True)
class DerivationInputs:
    """Certificate rows prepared for the derivation solver.

    ``usable`` maps each ratio to ``(value, standard uncertainty)`` — the
    solver propagates standard uncertainties, so every input is normalized to
    ``k = 1`` first. ``unusable`` names the rows that could not be normalized
    and says why, so an incomplete row is reported rather than silently
    dropped or silently treated as exact.
    """

    usable: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    unusable: Tuple[Tuple[str, str], ...] = ()


def collect_derivation_inputs(
    ratios: Dict[str, CertifiedRatio],
) -> DerivationInputs:
    """Normalize editor rows into solver inputs, listing what cannot be used.

    An unassigned uncertainty and a source-stated limit have no defensible
    standard uncertainty, so neither is converted: the row is reported instead.
    Zero is a real claim of exactness and stays a usable input.
    """
    usable: Dict[str, Tuple[float, float]] = {}
    unusable: List[Tuple[str, str]] = []
    for rname, rdata in ratios.items():
        if not rname:
            continue
        key = _norm_ratio(rname) or rname.strip()
        try:
            value = float(rdata.value)
        except (TypeError, ValueError):
            unusable.append((key, "the certified value is not a number"))
            continue
        if not math.isfinite(value) or value == 0:
            unusable.append((key, "the certified value is zero or not finite"))
            continue
        semantics = getattr(rdata, "uncertainty_semantics", "")
        u_standard = standard_uncertainty(rdata.uncertainty, rdata.k, semantics)
        if u_standard is None:
            if is_unassigned(rdata.uncertainty, semantics):
                reason = "its uncertainty is unassigned"
            elif str(semantics).strip() == "source_stated_limit":
                reason = (
                    "a source-stated limit has no coverage factor to convert"
                )
            else:
                reason = "its uncertainty and coverage factor are not usable"
            unusable.append((key, reason))
            continue
        usable[key] = (value, u_standard)
    return DerivationInputs(usable=usable, unusable=tuple(unusable))


@dataclass(frozen=True)
class PreviewRatio:
    """One row of the derived-ratio preview.

    ``uncertainty`` and ``k`` are reported as the row states them, and
    ``uncertainty_semantics`` says which quantity that is — a certificate row
    quoted at k=2 is an *expanded* uncertainty and must not be displayed under
    a standard-uncertainty heading. Derived rows always carry a standard
    uncertainty at k=1.
    """

    value: float
    uncertainty: Optional[float]
    k: Optional[float]
    uncertainty_semantics: str
    origin: str  # "stored" or "derived"


def preview_certified_ratios(
    ratios: Dict[str, CertifiedRatio],
) -> Dict[str, PreviewRatio]:
    """Return the stored and derivable ratios implied by *ratios*.

    Computed from the record handed in — the editor's current, possibly
    unsaved, rows — rather than from whatever library the process loaded at
    start-up. Mirrors ``config.reference_materials.get_crm_ratios``: rows with
    no usable standard uncertainty are kept for display but excluded from the
    derivation, and every derived row is a standard uncertainty at k = 1.
    """
    preview: Dict[str, PreviewRatio] = {}
    for rname, rdata in ratios.items():
        if not rname:
            continue
        key = _norm_ratio(rname) or rname.strip()
        try:
            value = float(rdata.value)
        except (TypeError, ValueError):
            continue
        unassigned = rdata.is_uncertainty_unassigned
        preview[key] = PreviewRatio(
            value=value,
            uncertainty=None if unassigned else rdata.uncertainty,
            k=None if unassigned else rdata.k,
            uncertainty_semantics=(
                "unassigned" if unassigned else rdata.uncertainty_semantics
            ),
            origin="stored",
        )

    solver_inputs = collect_derivation_inputs(ratios).usable
    try:
        from config.reference_materials import (
            _derive_certified_ratio,
            _validated_derived_value,
        )
    except Exception:  # noqa: BLE001 - preview must not block editing
        return preview

    isotopes = set()
    for ratio_name in solver_inputs:
        parts = ratio_name.split("/")
        if len(parts) == 2:
            isotopes.update(parts)

    memo: Dict[str, object] = {}
    for numerator in sorted(isotopes):
        for denominator in sorted(isotopes):
            if numerator == denominator:
                continue
            target = f"{numerator}/{denominator}"
            if target in preview:
                continue
            derived = _validated_derived_value(
                _derive_certified_ratio(target, solver_inputs, [], memo)
            )
            if derived is None:
                continue
            derived_value, derived_uncertainty = derived
            preview[target] = PreviewRatio(
                value=derived_value,
                uncertainty=derived_uncertainty,
                k=1.0,
                uncertainty_semantics="standard_uncertainty",
                origin="derived",
            )
    return preview


def derive_ratio_from_certified(
    target_ratio: str,
    ratios: Dict[str, CertifiedRatio],
) -> Optional[Tuple[float, float]]:
    """Derive a ratio and its **standard** uncertainty from certified rows.

    Each input is converted to ``k = 1`` using its own supplied coverage
    factor before propagation — the solver takes standard uncertainties, and
    feeding it expanded ones understated a mixed-k derivation. The returned
    uncertainty is therefore a standard uncertainty (``k = 1``), the same
    convention :func:`config.reference_materials.get_crm_ratios` uses for the
    rows it derives. Returns ``None`` when the target cannot be derived.
    """
    target = _norm_ratio(target_ratio) or target_ratio.strip()
    if not target or target.count("/") != 1:
        return None

    raw = collect_derivation_inputs(ratios).usable

    if target in raw:
        return raw[target]

    try:
        # Reuse app solver to keep derivation and uncertainty behavior aligned.
        from config.reference_materials import _derive_certified_ratio
    except Exception:
        return None

    derived = _derive_certified_ratio(target, raw, [], {})
    if derived is None:
        return None

    try:
        value = float(derived[0])
        uncertainty = float(derived[1])
    except (TypeError, ValueError):
        return None

    if not (math.isfinite(value) and math.isfinite(uncertainty)):
        return None
    return value, uncertainty

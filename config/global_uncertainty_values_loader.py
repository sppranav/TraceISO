"""Loader for lab-managed global uncertainty defaults.

This module owns file I/O for ``config/global_uncertainty_values.json``.
Domain engines receive values through ``UncertaintyConfig`` and never read this
file directly.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from config.contributor_names import parse_config_bool
from config.settings import (
    CustomUncertaintyContributor,
    DEFAULT_K1_SAMPLE_DECOMPOSITION_DISTRIBUTION,
    DEFAULT_K1_SAMPLE_DECOMPOSITION_PERMIL,
    DEFAULT_K2_MATRIX_SEPARATION_DISTRIBUTION,
    DEFAULT_K2_MATRIX_SEPARATION_PERMIL,
    DEFAULT_K3_PROCEDURAL_BLANK_DISTRIBUTION,
    DEFAULT_K3_PROCEDURAL_BLANK_PERMIL,
    DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_DISTRIBUTION,
    DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_PERMIL,
    DEFAULT_K6_MATRIX_EFFECTS_DISTRIBUTION,
    DEFAULT_K6_MATRIX_EFFECTS_PERMIL,
    DEFAULT_K7_RESIDUAL_INTERFERENCES_DISTRIBUTION,
    DEFAULT_K7_RESIDUAL_INTERFERENCES_PERMIL,
    DEFAULT_KAPPA_DRIFT_DISTRIBUTION,
    DEFAULT_REPROD_DIG_SD,
)
from config.uncertainty_profiles_loader import ProfileConflictError
from config.validation import (
    load_json_file,
    normalize_distribution as _normalize_distribution,
    parse_custom_contributor_entries,
    parse_dof as _parse_dof,
    parse_nonnegative_finite_float as _parse_nonnegative_finite_float,
)


GLOBAL_UNCERTAINTY_VALUES_PATH = Path(__file__).with_name(
    "global_uncertainty_values.json"
)
LEGACY_CUSTOM_CONTRIBUTORS_PATH = Path(__file__).with_name(
    "custom_uncertainty_contributors.json"
)

_CURRENT_VERSION = "1.0"
_custom_contributor_cache: Dict[
    tuple[Path, Path],
    tuple[tuple[Optional[int], Optional[int]], Dict[str, List[CustomUncertaintyContributor]]],
] = {}
_custom_contributor_cache_lock = threading.Lock()


@dataclass(frozen=True)
class KappaSpec:
    key: str
    contributor_name: str
    value_attr: str
    distribution_attr: str
    default_value: float
    default_distribution: str


KAPPA_SPECS: Dict[str, KappaSpec] = {
    "k1_sample_decomposition": KappaSpec(
        key="k1_sample_decomposition",
        contributor_name="u_k1_sample_decomposition",
        value_attr="k1_sample_decomposition_permil",
        distribution_attr="k1_sample_decomposition_distribution",
        default_value=DEFAULT_K1_SAMPLE_DECOMPOSITION_PERMIL,
        default_distribution=DEFAULT_K1_SAMPLE_DECOMPOSITION_DISTRIBUTION,
    ),
    "k2_matrix_separation": KappaSpec(
        key="k2_matrix_separation",
        contributor_name="u_k2_matrix_separation",
        value_attr="k2_matrix_separation_permil",
        distribution_attr="k2_matrix_separation_distribution",
        default_value=DEFAULT_K2_MATRIX_SEPARATION_PERMIL,
        default_distribution=DEFAULT_K2_MATRIX_SEPARATION_DISTRIBUTION,
    ),
    "k3_procedural_blank": KappaSpec(
        key="k3_procedural_blank",
        contributor_name="u_k3_procedural_blank",
        value_attr="k3_procedural_blank_permil",
        distribution_attr="k3_procedural_blank_distribution",
        default_value=DEFAULT_K3_PROCEDURAL_BLANK_PERMIL,
        default_distribution=DEFAULT_K3_PROCEDURAL_BLANK_DISTRIBUTION,
    ),
    "k4_bracketing_standard_heterogeneity": KappaSpec(
        key="k4_bracketing_standard_heterogeneity",
        contributor_name="u_k4_bracketing_standard_heterogeneity",
        value_attr="k4_bracketing_standard_heterogeneity_permil",
        distribution_attr="k4_bracketing_standard_heterogeneity_distribution",
        default_value=DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_PERMIL,
        default_distribution=DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_DISTRIBUTION,
    ),
    "k6_matrix_effects": KappaSpec(
        key="k6_matrix_effects",
        contributor_name="u_k6_matrix_effects",
        value_attr="k6_matrix_effects_permil",
        distribution_attr="k6_matrix_effects_distribution",
        default_value=DEFAULT_K6_MATRIX_EFFECTS_PERMIL,
        default_distribution=DEFAULT_K6_MATRIX_EFFECTS_DISTRIBUTION,
    ),
    "k7_residual_interferences": KappaSpec(
        key="k7_residual_interferences",
        contributor_name="u_k7_residual_interferences",
        value_attr="k7_residual_interferences_permil",
        distribution_attr="k7_residual_interferences_distribution",
        default_value=DEFAULT_K7_RESIDUAL_INTERFERENCES_PERMIL,
        default_distribution=DEFAULT_K7_RESIDUAL_INTERFERENCES_DISTRIBUTION,
    ),
}
K5_KEY = "k5_instrumental_drift"
KNOWN_KAPPA_KEYS = set(KAPPA_SPECS) | {K5_KEY}
KNOWN_ELEMENT_BLOCK_KEYS = frozenset(
    {"ssb_kappa_defaults", "sr_engine_a_defaults", "custom_contributors"}
)


@dataclass(frozen=True)
class SrEngineADefaultSpec:
    key: str
    contributor_name: str
    value_attr: str
    default_value: float
    units: str
    source: str


SR_ENGINE_A_DEFAULT_SPECS: Dict[str, SrEngineADefaultSpec] = {
    "u_bias_qc": SrEngineADefaultSpec(
        key="u_bias_qc",
        contributor_name="u_bias_qc",
        value_attr="sr_qc_bias_abs",
        default_value=0.0,
        units="absolute_ratio",
        source="lab supplied",
    ),
    "u_reprod_dig": SrEngineADefaultSpec(
        key="u_reprod_dig",
        contributor_name="u_reprod_dig",
        value_attr="u_reprod_dig_sd",
        default_value=DEFAULT_REPROD_DIG_SD,
        units="absolute_ratio",
        source="lab supplied",
    ),
    "enable_sr_norm_ratio_uncertainty": SrEngineADefaultSpec(
        key="enable_sr_norm_ratio_uncertainty",
        contributor_name="u_norm_ratio",
        value_attr="sr_norm_ratio_u_abs",
        default_value=0.0,
        units="absolute_ratio",
        source="published normalization method",
    ),
}
KNOWN_SR_ENGINE_A_DEFAULT_KEYS = set(SR_ENGINE_A_DEFAULT_SPECS)


@dataclass(frozen=True)
class KappaDefault:
    u_rel_permil: float = 0.0
    enabled: bool = True
    distribution: str = "normal"
    source: str = ""
    calculation: str = ""


@dataclass(frozen=True)
class SrEngineADefault:
    value: float = 0.0
    enabled: bool = False
    units: str = ""
    source: str = ""


@dataclass(frozen=True)
class ElementGlobalUncertaintyValues:
    element_symbol: str
    ssb_kappa_defaults: Dict[str, KappaDefault] = field(default_factory=dict)
    sr_engine_a_defaults: Dict[str, SrEngineADefault] = field(default_factory=dict)
    custom_contributors: List[CustomUncertaintyContributor] = field(default_factory=list)


@dataclass(frozen=True)
class GlobalUncertaintyValues:
    version: str = _CURRENT_VERSION
    elements: Dict[str, ElementGlobalUncertaintyValues] = field(default_factory=dict)


def _normalize_element_symbol(value: object) -> str:
    symbol = str(value or "").strip()
    if not symbol:
        raise ValueError("Element symbol must be non-empty.")
    return symbol[:1].upper() + symbol[1:].lower()


def _format_dof(value: float) -> object:
    return "inf" if math.isinf(float(value)) else float(value)


def _parse_custom_contributors(
    element_symbol: str,
    entries: object,
) -> List[CustomUncertaintyContributor]:
    """Parse custom contributors through the shared validator (audit A045)."""
    return parse_custom_contributor_entries(element_symbol, entries)


def _parse_kappa_defaults(raw: object, *, element_symbol: str) -> Dict[str, KappaDefault]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"ssb_kappa_defaults for {element_symbol!r} must be a JSON object."
        )

    unknown = set(raw) - KNOWN_KAPPA_KEYS
    if unknown:
        raise ValueError(
            f"Unknown kappa default key(s) for {element_symbol!r}: {sorted(unknown)}."
        )

    parsed: Dict[str, KappaDefault] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Kappa default {key!r} must be a JSON object.")

        if key == K5_KEY:
            calculation = str(entry.get("calculation") or "standard_sequence")
            if calculation != "standard_sequence":
                raise ValueError(
                    "k5_instrumental_drift calculation must be 'standard_sequence' in v1."
                )
            parsed[key] = KappaDefault(
                enabled=parse_config_bool(
                    entry.get("enabled", True),
                    field_name=f"Kappa default {key!r} enabled",
                ),
                calculation=calculation,
                distribution=_normalize_distribution(
                    entry.get("distribution", DEFAULT_KAPPA_DRIFT_DISTRIBUTION),
                    field_name=f"Kappa default {key!r}",
                ),
            )
            continue

        spec = KAPPA_SPECS[key]
        if "u_rel_permil" not in entry:
            raise ValueError(
                f"Kappa default {key!r} for {element_symbol!r} requires "
                "u_rel_permil."
            )
        u_rel_permil = _parse_nonnegative_finite_float(
            entry["u_rel_permil"],
            field_name=f"Kappa default {key!r} u_rel_permil",
        )
        source = str(entry.get("source") or "").strip()
        if not source:
            raise ValueError(f"Kappa default {key!r} requires a non-empty source.")
        parsed[key] = KappaDefault(
            u_rel_permil=u_rel_permil,
            enabled=parse_config_bool(
                entry.get("enabled", True),
                field_name=f"Kappa default {key!r} enabled",
            ),
            distribution=_normalize_distribution(
                entry.get("distribution", spec.default_distribution),
                field_name=f"Kappa default {key!r}",
            ),
            source=source,
        )

    return parsed


def _parse_sr_engine_a_defaults(raw: object, *, element_symbol: str) -> Dict[str, SrEngineADefault]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"sr_engine_a_defaults for {element_symbol!r} must be a JSON object."
        )

    unknown = set(raw) - KNOWN_SR_ENGINE_A_DEFAULT_KEYS
    if unknown:
        raise ValueError(
            f"Unknown Sr Engine A default key(s) for {element_symbol!r}: {sorted(unknown)}."
        )

    parsed: Dict[str, SrEngineADefault] = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Sr Engine A default {key!r} must be a JSON object.")
        spec = SR_ENGINE_A_DEFAULT_SPECS[key]
        if "value" not in entry:
            raise ValueError(
                f"Sr Engine A default {key!r} for {element_symbol!r} requires value."
            )
        value = _parse_nonnegative_finite_float(
            entry["value"],
            field_name=f"Sr Engine A default {key!r} value",
        )
        units = str(entry.get("units") or spec.units).strip()
        if units != spec.units:
            raise ValueError(
                f"Sr Engine A default {key!r} units must be {spec.units!r}."
            )
        source = str(entry.get("source") or spec.source).strip()
        parsed[key] = SrEngineADefault(
            value=value,
            enabled=parse_config_bool(
                entry.get("enabled", False),
                field_name=f"Sr Engine A default {key!r} enabled",
            ),
            units=units,
            source=source,
        )
    return parsed


def _parse_global_values(raw: object) -> GlobalUncertaintyValues:
    if not isinstance(raw, dict):
        raise ValueError("Global uncertainty values file must be a JSON object.")

    version = str(raw.get("version") or _CURRENT_VERSION)
    if version != _CURRENT_VERSION:
        raise ValueError(
            f"Unsupported global uncertainty values version {version!r}; "
            f"expected {_CURRENT_VERSION!r}."
        )

    raw_elements = raw.get("elements", {})
    if not isinstance(raw_elements, dict):
        raise ValueError("'elements' must be a JSON object.")

    elements: Dict[str, ElementGlobalUncertaintyValues] = {}
    for raw_symbol, payload in raw_elements.items():
        element_symbol = _normalize_element_symbol(raw_symbol)
        if element_symbol in elements:
            raise ValueError(
                f"Duplicate element symbol {element_symbol!r} in global uncertainty values "
                f"(entries {raw_symbol!r} and an earlier key normalise to the same symbol)."
            )
        if not isinstance(payload, dict):
            raise ValueError(f"Element entry {raw_symbol!r} must be a JSON object.")
        unknown = set(payload) - KNOWN_ELEMENT_BLOCK_KEYS
        if unknown:
            raise ValueError(
                f"Unknown global uncertainty block key(s) for "
                f"{element_symbol!r}: {sorted(unknown)}."
            )
        elements[element_symbol] = ElementGlobalUncertaintyValues(
            element_symbol=element_symbol,
            ssb_kappa_defaults=_parse_kappa_defaults(
                payload.get("ssb_kappa_defaults", {}),
                element_symbol=element_symbol,
            ),
            sr_engine_a_defaults=_parse_sr_engine_a_defaults(
                payload.get("sr_engine_a_defaults", {}),
                element_symbol=element_symbol,
            ),
            custom_contributors=_parse_custom_contributors(
                element_symbol,
                payload.get("custom_contributors", []),
            ),
        )

    return GlobalUncertaintyValues(version=version, elements=elements)


def load_global_uncertainty_values(
    path: Path = GLOBAL_UNCERTAINTY_VALUES_PATH,
) -> GlobalUncertaintyValues:
    """Load and validate ``global_uncertainty_values.json``."""
    if not path.exists():
        return GlobalUncertaintyValues()
    raw = load_json_file(path, "global uncertainty values")
    return _parse_global_values(raw)


def _kappa_to_json(value: KappaDefault, *, include_magnitude: bool = True) -> dict:
    if not include_magnitude:
        return {
            "enabled": bool(value.enabled),
            "calculation": value.calculation or "standard_sequence",
            "distribution": value.distribution or DEFAULT_KAPPA_DRIFT_DISTRIBUTION,
        }
    return {
        "u_rel_permil": float(value.u_rel_permil),
        "enabled": bool(value.enabled),
        "distribution": value.distribution,
        "source": value.source,
    }


def _sr_engine_a_default_to_json(value: SrEngineADefault) -> dict:
    return {
        "value": float(value.value),
        "enabled": bool(value.enabled),
        "units": value.units,
        "source": value.source,
    }


def _custom_to_json(contributor: CustomUncertaintyContributor) -> dict:
    return {
        "name": contributor.name,
        "display_name": contributor.display_name,
        "u_rel_permil": float(contributor.u_rel_permil),
        "type_ab": contributor.type_ab,
        "degrees_of_freedom": _format_dof(contributor.degrees_of_freedom),
        "distribution": contributor.distribution,
        "description": contributor.description,
        "source": contributor.reference,
        "enabled": bool(contributor.enabled),
    }


def global_uncertainty_values_revision(
    path: Path = GLOBAL_UNCERTAINTY_VALUES_PATH,
) -> str:
    """Return the content revision of the global values file (``""`` if absent)."""
    from config.managed_file_writes import file_content_revision

    return file_content_revision(path)


def save_global_uncertainty_values(
    values: GlobalUncertaintyValues,
    path: Path = GLOBAL_UNCERTAINTY_VALUES_PATH,
    *,
    expected_mtime: Optional[float] = None,
    expected_revision: Optional[str] = None,
) -> str:
    """Write global uncertainty values as one serialized cross-process transaction.

    The supplied expectation — a content revision from
    :func:`global_uncertainty_values_revision`, or the legacy modification time
    — is re-read while an exclusive lock is held, so two overlapping saves
    cannot both decide they are current. ``ProfileConflictError`` (from the
    profiles loader) reports the conflict. Returns the revision written.
    """
    text = validate_global_uncertainty_values(values)
    from config.managed_file_writes import commit_managed_text
    return commit_managed_text(
        path, text, expected_mtime=expected_mtime,
        expected_revision=expected_revision, conflict_error=ProfileConflictError,
        conflict_message="Global uncertainty values file was modified by another session. Please reload and retry.",
    )


def validate_global_uncertainty_values(values: GlobalUncertaintyValues) -> str:
    """Validate and serialize the exact save payload without filesystem effects."""
    payload = {
        "version": values.version,
        "elements": {},
    }
    for element_symbol, element_values in sorted(values.elements.items()):
        kappa_payload: dict = {}
        for key in KAPPA_SPECS:
            if key in element_values.ssb_kappa_defaults:
                kappa_payload[key] = _kappa_to_json(
                    element_values.ssb_kappa_defaults[key]
                )
        if K5_KEY in element_values.ssb_kappa_defaults:
            kappa_payload[K5_KEY] = _kappa_to_json(
                element_values.ssb_kappa_defaults[K5_KEY],
                include_magnitude=False,
            )
        element_payload = {
            "custom_contributors": [
                _custom_to_json(item) for item in element_values.custom_contributors
            ],
        }
        if kappa_payload:
            element_payload["ssb_kappa_defaults"] = kappa_payload
        if element_values.sr_engine_a_defaults:
            element_payload["sr_engine_a_defaults"] = {
                key: _sr_engine_a_default_to_json(element_values.sr_engine_a_defaults[key])
                for key in SR_ENGINE_A_DEFAULT_SPECS
                if key in element_values.sr_engine_a_defaults
            }
        payload["elements"][element_symbol] = element_payload

    # Validate the exact payload we will write.
    _parse_global_values(payload)

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def get_element_values(
    element_symbol: str,
    values: Optional[GlobalUncertaintyValues] = None,
) -> Optional[ElementGlobalUncertaintyValues]:
    """Return the persisted global-uncertainty values for *element_symbol*.

    Returns ``None`` if the element has no configured entry. Loads the defaults
    file when *values* is not supplied.
    """
    if values is None:
        values = load_global_uncertainty_values()
    return values.elements.get(_normalize_element_symbol(element_symbol))


def get_ssb_kappa_defaults_for_element(
    element_symbol: str,
    values: Optional[GlobalUncertaintyValues] = None,
) -> Dict[str, KappaDefault]:
    """Return the SSB κ-factor defaults (keyed by κ name) for *element_symbol*.

    Empty dict if the element is not configured.
    """
    element_values = get_element_values(element_symbol, values)
    if element_values is None:
        return {}
    return dict(element_values.ssb_kappa_defaults)


def get_sr_engine_a_defaults_for_element(
    element_symbol: str,
    values: Optional[GlobalUncertaintyValues] = None,
) -> Dict[str, SrEngineADefault]:
    """Return the Sr Engine A contributor defaults (keyed by contributor name).

    Empty dict if *element_symbol* is not configured.
    """
    element_values = get_element_values(element_symbol, values)
    if element_values is None:
        return {}
    return dict(element_values.sr_engine_a_defaults)


def get_custom_contributors_for_element(
    element_symbol: str,
    values: Optional[GlobalUncertaintyValues] = None,
) -> List[CustomUncertaintyContributor]:
    """Return the user-defined custom uncertainty contributors for *element_symbol*.

    Empty list if the element has none configured.
    """
    element_values = get_element_values(element_symbol, values)
    if element_values is None:
        return []
    return list(element_values.custom_contributors)


def _parse_legacy_custom_contributors(raw: object) -> Dict[str, List[CustomUncertaintyContributor]]:
    if not isinstance(raw, dict):
        raise ValueError(
            "Legacy custom contributor library must be a JSON object keyed by element symbol."
        )
    out: Dict[str, List[CustomUncertaintyContributor]] = {}
    for raw_symbol, entries in raw.items():
        element_symbol = _normalize_element_symbol(raw_symbol)
        if element_symbol in out:
            # Checked before the assignment: two spellings of one element would
            # otherwise overwrite each other and drop a saved definition, even
            # though the standalone loader rejects exactly this input.
            raise ValueError(
                f"Duplicate element symbol {element_symbol!r} in legacy custom "
                f"contributor library (entry {raw_symbol!r} normalizes to an "
                "existing key)."
            )
        out[element_symbol] = _parse_custom_contributors(element_symbol, entries)
    return out


def load_legacy_custom_contributors(
    path: Path = LEGACY_CUSTOM_CONTRIBUTORS_PATH,
) -> Dict[str, List[CustomUncertaintyContributor]]:
    """Load the pre-v1 global custom-contributor file if present."""
    if not path.exists():
        return {}
    raw = load_json_file(path, "legacy custom contributor")
    return _parse_legacy_custom_contributors(raw)


def _merge_custom_contributor_lists(
    authoritative: List[CustomUncertaintyContributor],
    legacy: List[CustomUncertaintyContributor],
) -> List[CustomUncertaintyContributor]:
    """Merge custom contributors by name; authoritative entries win."""
    merged = list(authoritative)
    seen = {item.name for item in merged}
    for item in legacy:
        if item.name not in seen:
            merged.append(item)
            seen.add(item.name)
    return merged


def _merge_custom_contributor_libraries(
    authoritative: Dict[str, List[CustomUncertaintyContributor]],
    legacy: Dict[str, List[CustomUncertaintyContributor]],
) -> Dict[str, List[CustomUncertaintyContributor]]:
    merged = {
        element_symbol: list(contributors)
        for element_symbol, contributors in authoritative.items()
    }
    for element_symbol, contributors in legacy.items():
        merged[element_symbol] = _merge_custom_contributor_lists(
            merged.get(element_symbol, []),
            contributors,
        )
    return {key: value for key, value in merged.items() if value}


def load_custom_contributors_from_global_values(
    path: Path = GLOBAL_UNCERTAINTY_VALUES_PATH,
    *,
    legacy_path: Path = LEGACY_CUSTOM_CONTRIBUTORS_PATH,
) -> Dict[str, List[CustomUncertaintyContributor]]:
    """Return custom contributor definitions using global values first."""
    mtimes = tuple(
        candidate.stat().st_mtime_ns if candidate.exists() else None
        for candidate in (path, legacy_path)
    )
    cache_key = (path, legacy_path)
    with _custom_contributor_cache_lock:
        cached = _custom_contributor_cache.get(cache_key)
        if cached is not None and cached[0] == mtimes:
            return cached[1]

    if path.exists():
        values = load_global_uncertainty_values(path)
        global_custom = {
            element_symbol: list(element_values.custom_contributors)
            for element_symbol, element_values in values.elements.items()
            if element_values.custom_contributors
        }
        legacy = load_legacy_custom_contributors(legacy_path)
        result = _merge_custom_contributor_libraries(global_custom, legacy)
    else:
        result = load_legacy_custom_contributors(legacy_path)

    with _custom_contributor_cache_lock:
        _custom_contributor_cache[cache_key] = (mtimes, result)
    return result


def migrate_legacy_custom_contributors_to_global_values(
    *,
    legacy_path: Path = LEGACY_CUSTOM_CONTRIBUTORS_PATH,
    base_values: Optional[GlobalUncertaintyValues] = None,
) -> GlobalUncertaintyValues:
    """Return global values with legacy custom contributors merged in memory."""
    values = base_values or GlobalUncertaintyValues()
    legacy = load_legacy_custom_contributors(legacy_path)
    elements = dict(values.elements)
    for element_symbol, contributors in legacy.items():
        existing = elements.get(
            element_symbol,
            ElementGlobalUncertaintyValues(element_symbol=element_symbol),
        )
        elements[element_symbol] = ElementGlobalUncertaintyValues(
            element_symbol=element_symbol,
            ssb_kappa_defaults=dict(existing.ssb_kappa_defaults),
            sr_engine_a_defaults=dict(existing.sr_engine_a_defaults),
            custom_contributors=_merge_custom_contributor_lists(
                list(existing.custom_contributors),
                contributors,
            ),
        )
    return GlobalUncertaintyValues(version=values.version, elements=elements)


def contributor_enabled_defaults_for_element(
    element_symbol: str,
    values: Optional[GlobalUncertaintyValues] = None,
) -> Dict[str, bool]:
    """Map element kappa defaults to canonical contributor toggle defaults."""
    kappa_defaults = get_ssb_kappa_defaults_for_element(element_symbol, values)
    out: Dict[str, bool] = {}
    for key, spec in KAPPA_SPECS.items():
        if key in kappa_defaults:
            out[spec.contributor_name] = bool(kappa_defaults[key].enabled)
    if K5_KEY in kappa_defaults:
        out["u_k5_instrumental_drift"] = bool(kappa_defaults[K5_KEY].enabled)
    sr_defaults = get_sr_engine_a_defaults_for_element(element_symbol, values)
    for key, spec in SR_ENGINE_A_DEFAULT_SPECS.items():
        if key in sr_defaults:
            out[spec.contributor_name] = bool(sr_defaults[key].enabled)
    if _normalize_element_symbol(element_symbol) == "Sr":
        # Complete the shipped Sr checkbox preset. The value-backed QC and
        # digestion choices above remain editable in the JSON profile.
        out.setdefault("u_prec", True)
        out.setdefault("enable_srm_repeatability", True)
        out.setdefault("u_blank", False)
        out.setdefault("u_interf", False)
    return out


def iter_kappa_specs() -> Iterable[KappaSpec]:
    """Return all registered κ-factor specifications (``KappaSpec``)."""
    return tuple(KAPPA_SPECS.values())


def iter_sr_engine_a_default_specs() -> Iterable[SrEngineADefaultSpec]:
    """Return all registered Sr Engine A default-contributor specifications."""
    return tuple(SR_ENGINE_A_DEFAULT_SPECS.values())

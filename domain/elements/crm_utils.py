"""Helpers for building element configs from CRM library data."""

from __future__ import annotations

from typing import Dict, Iterable

from config.reference_materials import (
    get_crm_ratios,
    get_default_crm_name,
    get_crm,
)
from domain.elements.base import CertifiedValue


def resolve_reference_material(symbol: str, preferred_name: str) -> str:
    """Resolve the default CRM name for an element from managed library data."""
    if get_crm(symbol, preferred_name) is not None:
        return preferred_name

    default_name = get_default_crm_name(symbol)
    if default_name is not None:
        return default_name

    raise ValueError(
        f"No reference material configured for element '{symbol}' in config/crm_library.json."
    )


def build_certified_values(
    symbol: str,
    reference_material: str,
    required_ratios: Iterable[str],
) -> Dict[str, CertifiedValue]:
    """Return required certified values from managed CRM data.

    Derived and inverse ratios are allowed when reachable from the
    direct certified ratios stored for *reference_material*.
    """
    crm_ratios = get_crm_ratios(symbol, reference_material, derive=True)
    certified_values: Dict[str, CertifiedValue] = {}
    missing = []
    for ratio_name in required_ratios:
        payload = crm_ratios.get(ratio_name)
        if payload is None:
            missing.append(ratio_name)
            continue
        value, uncertainty, k = payload
        certified_values[ratio_name] = CertifiedValue(
            value=value,
            uncertainty=uncertainty,
            k=k,
            source=reference_material,
        )

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(
            f"Reference material '{reference_material}' for element '{symbol}' "
            f"is missing required certified ratios: {missing_text}."
        )

    return certified_values


def resolve_optional_certified_value(
    symbol: str,
    reference_material: str,
    ratio_name: str,
) -> CertifiedValue | None:
    """Return a certified value for one ratio from a named reference material.

    Unlike :func:`build_certified_values`, this returns ``None`` instead of
    raising when *ratio_name* is not certified in *reference_material* —
    intended for optional literature reference values (e.g. GeoReM) that need
    not cover every ratio an element reports.
    """
    crm_ratios = get_crm_ratios(symbol, reference_material, derive=True)
    payload = crm_ratios.get(ratio_name)
    if payload is None:
        return None
    value, uncertainty, k = payload
    return CertifiedValue(
        value=value,
        uncertainty=uncertainty,
        k=k,
        source=reference_material,
    )

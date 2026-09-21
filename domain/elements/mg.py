"""Magnesium element configuration."""

from domain.elements.base import ElementConfig
from domain.elements.crm_utils import (
    build_certified_values,
    resolve_reference_material,
)


def build_mg_config() -> ElementConfig:
    """Build Mg config from managed CRM library data."""
    default_ratios = {
        "25Mg/24Mg": ("25Mg", "24Mg"),
        "26Mg/24Mg": ("26Mg", "24Mg"),
        "26Mg/25Mg": ("26Mg", "25Mg"),
    }
    reference_material = resolve_reference_material("Mg", "ERM-AE143")

    return ElementConfig(
        symbol="Mg",
        isotopes=["24Mg", "25Mg", "26Mg"],
        default_ratios=default_ratios,
        certified_values=build_certified_values("Mg", reference_material, default_ratios.keys()),
        reference_material=reference_material,
        correction_steps=["blank", "filter", "ssb", "delta", "uncertainty"],
        supports_ssb=True,
        supports_delta=True,
        ratio_name_aliases={
            "25Mg/24Mg": ["Mg25/Mg24", "Mg25\\Mg24"],
            "26Mg/24Mg": ["Mg26/Mg24", "Mg26\\Mg24"],
            "26Mg/25Mg": ["Mg26/Mg25", "Mg26\\Mg25"],
        },
    )


# Backward-compatible lazy constant (item 146).
def __getattr__(name: str):
    if name == "MG_CONFIG":
        val = build_mg_config()
        globals()["MG_CONFIG"] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

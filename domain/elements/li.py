"""Lithium element configuration."""

from domain.elements.base import ElementConfig
from domain.elements.crm_utils import (
    build_certified_values,
    resolve_reference_material,
)


def build_li_config() -> ElementConfig:
    """Build Li config from managed CRM library data."""
    default_ratios = {
        "7Li/6Li": ("7Li", "6Li"),
        "6Li/7Li": ("6Li", "7Li"),
    }
    reference_material = resolve_reference_material("Li", "NIST RM 8545 (LSVEC)")

    return ElementConfig(
        symbol="Li",
        isotopes=["6Li", "7Li"],
        default_ratios=default_ratios,
        certified_values=build_certified_values("Li", reference_material, default_ratios.keys()),
        reference_material=reference_material,
        correction_steps=["blank", "filter", "ssb", "delta", "uncertainty"],
        supports_ssb=True,
        supports_delta=True,
        ratio_name_aliases={
            "7Li/6Li": ["Li7/Li6", "Li7\\Li6"],
            "6Li/7Li": ["Li6/Li7", "Li6\\Li7"],
        },
    )


# Backward-compatible lazy constant (item 146).
def __getattr__(name: str):
    if name == "LI_CONFIG":
        val = build_li_config()
        globals()["LI_CONFIG"] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

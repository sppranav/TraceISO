"""Boron element configuration."""

from domain.elements.base import ElementConfig
from domain.elements.crm_utils import (
    build_certified_values,
    resolve_reference_material,
)


def build_b_config() -> ElementConfig:
    """Build B config from managed CRM library data."""
    default_ratios = {
        "11B/10B": ("11B", "10B"),
        "10B/11B": ("10B", "11B"),
    }
    reference_material = resolve_reference_material("B", "NIST SRM 951")

    return ElementConfig(
        symbol="B",
        isotopes=["10B", "11B"],
        default_ratios=default_ratios,
        certified_values=build_certified_values("B", reference_material, default_ratios.keys()),
        reference_material=reference_material,
        correction_steps=["blank", "filter", "ssb", "delta", "uncertainty"],
        supports_ssb=True,
        supports_delta=True,
        ratio_name_aliases={
            "11B/10B": ["B11/B10", "B11\\B10"],
            "10B/11B": ["B10/B11", "B10\\B11"],
        },
    )


# Backward-compatible lazy constant (item 146).
def __getattr__(name: str):
    if name == "B_CONFIG":
        val = build_b_config()
        globals()["B_CONFIG"] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

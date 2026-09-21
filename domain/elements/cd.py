"""Cadmium element configuration."""

from domain.elements.base import ElementConfig
from domain.elements.crm_utils import (
    build_certified_values,
    resolve_reference_material,
)


def build_cd_config() -> ElementConfig:
    """Build Cd config from managed CRM library data."""
    default_ratios = {
        "114Cd/111Cd": ("114Cd", "111Cd"),
        "110Cd/111Cd": ("110Cd", "111Cd"),
        "112Cd/111Cd": ("112Cd", "111Cd"),
    }
    reference_material = resolve_reference_material("Cd", "BAM-I012")

    return ElementConfig(
        symbol="Cd",
        isotopes=[
            "106Cd",
            "110Cd",
            "111Cd",
            "112Cd",
            "113Cd",
            "114Cd",
            "116Cd",
        ],
        default_ratios=default_ratios,
        certified_values=build_certified_values("Cd", reference_material, default_ratios.keys()),
        reference_material=reference_material,
        correction_steps=["blank", "filter", "ssb", "delta", "uncertainty"],
        supports_ssb=True,
        supports_delta=True,
        ratio_name_aliases={
            "114Cd/111Cd": ["Cd114/Cd111", "Cd114\\Cd111"],
            "110Cd/111Cd": ["Cd110/Cd111", "Cd110\\Cd111"],
            "112Cd/111Cd": ["Cd112/Cd111", "Cd112\\Cd111"],
        },
    )


# Backward-compatible lazy constant (item 146).
def __getattr__(name: str):
    if name == "CD_CONFIG":
        val = build_cd_config()
        globals()["CD_CONFIG"] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

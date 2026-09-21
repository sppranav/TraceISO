"""Lead element configuration."""

from config.reference_materials import get_internal_normalization
from domain.elements.base import DataLayer, ElementConfig
from domain.elements.crm_utils import (
    build_certified_values,
    resolve_reference_material,
)


def build_pb_config() -> ElementConfig:
    """Build Pb config from managed CRM library data."""
    default_ratios = {
        "206Pb/204Pb": ("206Pb", "204Pb"),
        "207Pb/206Pb": ("207Pb", "206Pb"),
        "208Pb/206Pb": ("208Pb", "206Pb"),
    }
    reference_material = resolve_reference_material("Pb", "NIST SRM 981")
    internal_norm = get_internal_normalization("Tl")
    if internal_norm is not None:
        norm_ratio_name, norm_value = internal_norm
    else:
        norm_ratio_name, norm_value = None, None

    return ElementConfig(
        symbol="Pb",
        isotopes=["202Hg", "203Tl", "204Pb", "205Tl", "206Pb", "207Pb", "208Pb"],
        data_layers=(
            DataLayer.RAW,
            DataLayer.BLANK_CORRECTED,
            DataLayer.IIF_CORRECTED,
            DataLayer.SSB_CORRECTED,
            DataLayer.DRIFT_CORRECTED,
        ),
        default_ratios=default_ratios,
        certified_values=build_certified_values("Pb", reference_material, default_ratios.keys()),
        reference_material=reference_material,
        correction_steps=["blank", "filter", "ssb", "delta", "uncertainty"],
        supports_ssb=True,
        supports_delta=True,
        mass_bias_law="russell",
        normalization_ratio=norm_ratio_name,
        normalization_value=norm_value,
        ratio_name_aliases={
            "206Pb/204Pb": ["Pb206/Pb204", "Pb206\\Pb204"],
            "207Pb/206Pb": ["Pb207/Pb206", "Pb207\\Pb206"],
            "208Pb/206Pb": ["Pb208/Pb206", "Pb208\\Pb206"],
        },
    )


# Backward-compatible lazy constant (item 146).
def __getattr__(name: str):
    if name == "PB_CONFIG":
        val = build_pb_config()
        globals()["PB_CONFIG"] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

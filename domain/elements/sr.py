"""Strontium element configuration."""

from config.reference_materials import get_internal_normalization
from domain.elements.base import DataLayer, ElementConfig, MonitorSpec
from domain.elements.crm_utils import (
    build_certified_values,
    resolve_reference_material,
)


SR_NORMALIZATION_RATIO = "86Sr/88Sr"
SR_CORRECTION_ROLES = {
    "normalization_numerator": "86Sr",
    "target_numerator": "87Sr",
    "normalization_denominator": "88Sr",
    "rb_monitor": "85Rb",
    "rb_interfering_mass": "87Rb",
    "kr_monitor": "83Kr",
    "kr84_target": "84Sr",
    "kr84_interfering_mass": "84Kr",
    "kr86_interfering_mass": "86Kr",
}

# explicit in _sr_correction_loop because they depend on K1 from Step 1.
SR_MONITORS: tuple = (
    MonitorSpec(
        corrected_isotope="87Sr",
        interfering_isotope="87Rb",
        monitor_isotope="85Rb",
        natural_ratio_key="87Rb/85Rb",
        family="f",
    ),
    MonitorSpec(
        corrected_isotope="84Sr",
        interfering_isotope="84Kr",
        monitor_isotope="83Kr",
        natural_ratio_key="84Kr/83Kr",
        family="f",
    ),
    MonitorSpec(
        corrected_isotope="86Sr",
        interfering_isotope="86Kr",
        monitor_isotope="83Kr",
        natural_ratio_key="86Kr/83Kr",
        family="f",
    ),
)


def build_sr_config() -> ElementConfig:
    """Build Sr config from managed CRM library data."""
    default_ratios = {
        "87Sr/86Sr": ("87Sr", "86Sr"),
        "88Sr/86Sr": ("88Sr", "86Sr"),
        "84Sr/86Sr": ("84Sr", "86Sr"),
    }
    reference_material = resolve_reference_material("Sr", "NIST SRM 987")
    certified_values = build_certified_values(
        "Sr",
        reference_material,
        list(default_ratios.keys()) + [SR_NORMALIZATION_RATIO],
    )
    internal_norm = get_internal_normalization("Sr")
    if internal_norm is not None:
        norm_ratio_name, norm_value = internal_norm
    else:
        norm_ratio_name, norm_value = SR_NORMALIZATION_RATIO, None

    return ElementConfig(
        symbol="Sr",
        isotopes=["82Kr", "83Kr", "84Sr", "85Rb", "86Sr", "87Sr", "88Sr", "90Zr", "91Zr"],
        default_ratios=default_ratios,
        certified_values=certified_values,
        reference_material=reference_material,
        correction_steps=[
            "blank", "filter", "interference", "mass_bias", "uncertainty",
        ],
        data_layers=(
            DataLayer.RAW,
            DataLayer.BLANK_CORRECTED,
            DataLayer.INTERFERENCE_CORRECTED,
            DataLayer.IIF_CORRECTED,
            DataLayer.DRIFT_CORRECTED,
        ),
        supports_ssb=False,
        supports_delta=False,
        iterations=2,
        mass_bias_law="russell",
        normalization_ratio=norm_ratio_name,
        normalization_value=norm_value,
        correction_roles=dict(SR_CORRECTION_ROLES),
        monitors=SR_MONITORS,
        ratio_name_aliases={
            "87Sr/86Sr": ["Sr87/Sr86", "Sr87\\Sr86"],
            "88Sr/86Sr": ["Sr88/Sr86", "Sr88\\Sr86"],
            "84Sr/86Sr": ["Sr84/Sr86", "Sr84\\Sr86"],
        },
    )


# Backward-compatible lazy constant — accessed on first use, not at import
# time, so a malformed CRM entry for this element does not prevent the app
# from loading other elements (item 146).
def __getattr__(name: str):
    if name == "SR_CONFIG":
        val = build_sr_config()
        globals()["SR_CONFIG"] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

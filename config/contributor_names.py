"""Canonical uncertainty contributor names and legacy aliases."""

from __future__ import annotations

from typing import Mapping, Optional


LABEL_U_PREC = "Sample measurement precision"
LABEL_U_STD = "Bracketing-standard precision"
LABEL_U_K4 = "Bracketing-standard heterogeneity"


CONTRIBUTOR_NAME_ALIASES = {
    "u_CRM": "u_crm",
    "u_k1": "u_k1_sample_decomposition",
    "u_k2": "u_k2_matrix_separation",
    "u_k3": "u_k3_procedural_blank",
    "u_k4": "u_k4_bracketing_standard_heterogeneity",
    "u_k5": "u_k5_instrumental_drift",
    "u_k6": "u_k6_matrix_effects",
    "u_k7": "u_k7_residual_interferences",
    "_sr_norm_ratio_u": "u_norm_ratio",
    "enable_sr_norm_ratio_uncertainty": "u_norm_ratio",
}


def canonical_contributor_name(name: object) -> str:
    """Return the canonical contributor ID for *name*."""
    raw = str(name)
    return CONTRIBUTOR_NAME_ALIASES.get(raw, raw)


def canonical_contributor_display_label(name: object) -> Optional[str]:
    """Return the canonical display label for app-wide renamed contributors."""
    canonical = canonical_contributor_name(name)
    if canonical == "u_prec":
        return LABEL_U_PREC
    if canonical == "u_std":
        return LABEL_U_STD
    if canonical == "u_k4_bracketing_standard_heterogeneity":
        return f"{LABEL_U_K4} (k4)"
    return None


def parse_config_bool(value: object, *, field_name: str = "value") -> bool:
    """Parse a configuration boolean without Python truthy string coercion."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{field_name} must be a boolean, got {value!r}.")


def canonical_contributor_mapping(values: Mapping[str, object]) -> dict[str, bool]:
    """Return a bool mapping keyed by canonical contributor IDs.

    If both a canonical name and a legacy alias are present, the canonical
    value wins.
    """
    out: dict[str, bool] = {}
    alias_values: list[tuple[str, bool]] = []
    for raw_name, value in values.items():
        raw = str(raw_name)
        canonical = canonical_contributor_name(raw)
        item = (
            canonical,
            parse_config_bool(value, field_name=f"Contributor {raw!r} enabled"),
        )
        if raw == canonical:
            out[canonical] = item[1]
        else:
            alias_values.append(item)

    for canonical, value in alias_values:
        out.setdefault(canonical, value)

    return dict(sorted(out.items()))

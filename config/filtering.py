"""Shared filter-method naming and compatibility helpers."""

from __future__ import annotations

from typing import List, Optional


FILTER_METHOD_NONE = "None"
FILTER_METHOD_STANDARD_DEVIATION = "Standard deviation"
FILTER_METHOD_MAD = "MAD"
FILTER_METHOD_IQR = "IQR"

REMOVED_FILTER_METHODS = frozenset(
    {
        "Z-score",
        "Modified Z-score",
        "Modified Z-score (legacy)",
        "Modified Z-score (MAD)",
        "Grubbs",
        "Chauvenet",
    }
)

VISIBLE_FILTER_METHOD_OPTIONS: List[str] = [
    FILTER_METHOD_NONE,
    FILTER_METHOD_STANDARD_DEVIATION,
    FILTER_METHOD_MAD,
    FILTER_METHOD_IQR,
]

_FILTER_METHOD_ALIASES = {
    "none": FILTER_METHOD_NONE,
    "standard deviation": FILTER_METHOD_STANDARD_DEVIATION,
    "std": FILTER_METHOD_STANDARD_DEVIATION,
    "std dev": FILTER_METHOD_STANDARD_DEVIATION,
    "sigma": FILTER_METHOD_STANDARD_DEVIATION,
    "mad": FILTER_METHOD_MAD,
    "robust mad": FILTER_METHOD_MAD,
    "iqr": FILTER_METHOD_IQR,
}

_FILTER_METHOD_DISPLAY_NAMES = {
    FILTER_METHOD_NONE: "None",
    FILTER_METHOD_STANDARD_DEVIATION: "Standard deviation",
    FILTER_METHOD_MAD: "Robust MAD",
    FILTER_METHOD_IQR: "IQR",
}


def normalize_filter_method_name(method: Optional[str]) -> str:
    """Return the canonical internal name for a filter method."""
    if method is None:
        return FILTER_METHOD_NONE

    raw = getattr(method, "value", method)
    raw = str(raw).strip()
    if not raw:
        return FILTER_METHOD_NONE

    return _FILTER_METHOD_ALIASES.get(raw.lower(), raw)




def get_filter_method_display_name(method: Optional[str]) -> str:
    """Return the UI label for a stored filter method."""
    normalized = normalize_filter_method_name(method)
    return _FILTER_METHOD_DISPLAY_NAMES.get(normalized, normalized)


def get_filter_method_options(current_method: Optional[str]) -> List[str]:
    """Return the visible UI options for outlier filtering.

    The current_method parameter is accepted for compatibility but currently unused,
    as all visible options are always offered.
    """
    return list(VISIBLE_FILTER_METHOD_OPTIONS)


def is_removed_filter_method(method: Optional[str]) -> bool:
    """Return True when *method* names an outlier filter removed from TraceISO."""
    raw = getattr(method, "value", method)
    raw = "" if raw is None else str(raw).strip()
    return raw.lower() in {item.lower() for item in REMOVED_FILTER_METHODS}


# Alias for backward compatibility / migration checks
is_legacy_filter_method = is_removed_filter_method

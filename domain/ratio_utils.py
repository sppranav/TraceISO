"""Shared helpers for isotope and ratio name normalization."""

from __future__ import annotations

import re
from typing import Optional


def normalize_ratio_token(token: str) -> str:
    """Normalize isotope token variants such as ``Sr87`` to ``87Sr``."""
    raw = str(token).strip()
    match = re.match(r"^([A-Za-z]+)(\d+)$", raw)
    if match:
        return f"{match.group(2)}{match.group(1)}"
    match = re.match(r"^(\d+)([A-Za-z]+)$", raw)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return raw


def normalize_ratio_name(ratio_name: Optional[str]) -> Optional[str]:
    """Normalize separators and isotope-token order for a ratio name.

    Backslash and underscore are treated as legacy ratio separators. Custom
    ratio names that need literal underscores must be canonicalized before they
    enter domain ratio dictionaries.
    """
    if ratio_name is None:
        return None
    raw = str(ratio_name).strip().replace("\\", "/").replace("_", "/")
    if raw.count("/") != 1:
        return raw or None
    numerator, denominator = raw.split("/", 1)
    return f"{normalize_ratio_token(numerator)}/{normalize_ratio_token(denominator)}"


def element_symbol_from_isotope(isotope_label: str) -> Optional[str]:
    """Extract the element symbol from an isotope label such as ``91Zr``."""
    token = normalize_ratio_token(isotope_label)
    match = re.match(r"^\d+([A-Za-z]+)$", token)
    if not match:
        return None
    return match.group(1).capitalize()

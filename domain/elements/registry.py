"""Element configuration registry for TraceISO."""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional

from domain.elements.base import ElementConfig
from domain.elements.li import build_li_config
from domain.elements.b import build_b_config
from domain.elements.sr import build_sr_config
from domain.elements.mg import build_mg_config
from domain.elements.cd import build_cd_config
from domain.elements.pb import build_pb_config


_BUILDERS: Dict[str, Callable[[], ElementConfig]] = {
    "Li": build_li_config,
    "B": build_b_config,
    "Sr": build_sr_config,
    "Mg": build_mg_config,
    "Cd": build_cd_config,
    "Pb": build_pb_config,
}

# Lightweight isotope signatures used for fast element detection.
# This avoids building full CRM-backed configs during file load.
_DETECTION_ISOTOPES: Dict[str, List[str]] = {
    "Li": ["6Li", "7Li"],
    "B": ["10B", "11B"],
    "Sr": ["82Kr", "83Kr", "84Sr", "85Rb", "86Sr", "87Sr", "88Sr"],
    "Mg": ["24Mg", "25Mg", "26Mg"],
    "Cd": ["106Cd", "110Cd", "111Cd", "112Cd", "113Cd", "114Cd", "116Cd"],
    "Pb": ["204Pb", "206Pb", "207Pb", "208Pb"],
}


def get_element(symbol: str) -> ElementConfig:
    """Look up an element config by symbol (case-insensitive)."""
    key = symbol.strip().capitalize()
    if key not in _BUILDERS:
        raise ValueError(
            f"Unknown element: {symbol!r}. Available: {list(_BUILDERS.keys())}",
        )
    return _BUILDERS[key]()


def list_elements() -> List[str]:
    """Return all registered element symbols."""
    return list(_BUILDERS.keys())


_MIN_DETECTION_MATCHES = 2


def detect_element(isotope_names: List[str]) -> Optional[ElementConfig]:
    """Auto-detect element from a list of isotope column names.

    Requires at least ``_MIN_DETECTION_MATCHES`` signature matches to avoid
    mis-classifying a file that happens to contain one monitor isotope (item 151).
    """
    matches = {
        symbol: [iso for iso in isotopes if iso in isotope_names]
        for symbol, isotopes in _DETECTION_ISOTOPES.items()
    }
    best_count = max((len(found) for found in matches.values()), default=0)
    if best_count < _MIN_DETECTION_MATCHES:
        return None
    candidates = sorted(symbol for symbol, found in matches.items() if len(found) == best_count)
    if len(candidates) > 1:
        details = ", ".join(f"{symbol}: {matches[symbol]}" for symbol in candidates)
        warnings.warn(
            f"Ambiguous element detection tie ({details}); selecting {candidates[0]} "
            "deterministically.",
            RuntimeWarning,
            stacklevel=2,
        )
    return get_element(candidates[0])

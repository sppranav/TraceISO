"""Loader for lab-managed custom uncertainty contributor definitions.

Reads ``config/custom_uncertainty_contributors.json`` (element-keyed list).
File I/O lives here, never in ``domain/`` or engines.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.settings import CustomUncertaintyContributor
from config.validation import (
    load_json_file,
    parse_custom_contributor_entries,
)

_logger = logging.getLogger(__name__)

# Default JSON path: same directory as this file.
CUSTOM_CONTRIBUTORS_PATH = Path(__file__).with_name("custom_uncertainty_contributors.json")

# item 74: per-path mtime cache so repeated UI renders do not re-parse JSON
_contributor_cache: Dict[Path, Tuple[float, Dict[str, List[CustomUncertaintyContributor]]]] = {}
_contributor_cache_lock = threading.Lock()


def load_custom_contributors(
    path: Path = CUSTOM_CONTRIBUTORS_PATH,
) -> Dict[str, List[CustomUncertaintyContributor]]:
    """Load and validate the custom contributor JSON library.

    Returns a dict keyed by element symbol (e.g. ``"Li"``, ``"B"``).
    Returns an empty dict when the file does not exist.

    Raises ``ValueError`` for malformed entries.
    """
    from config.recorded_dependencies import current_dependencies
    recorded = current_dependencies()
    if recorded is not None and path == CUSTOM_CONTRIBUTORS_PATH:
        from copy import deepcopy
        return deepcopy(recorded.custom_contributors)
    if path == CUSTOM_CONTRIBUTORS_PATH:
        from config.global_uncertainty_values_loader import (
            GLOBAL_UNCERTAINTY_VALUES_PATH,
            load_custom_contributors_from_global_values,
        )

        if GLOBAL_UNCERTAINTY_VALUES_PATH.exists():
            return load_custom_contributors_from_global_values()
    else:
        from config.global_uncertainty_values_loader import GLOBAL_UNCERTAINTY_VALUES_PATH

        if GLOBAL_UNCERTAINTY_VALUES_PATH.exists():
            _logger.warning(
                "load_custom_contributors called with a non-default path %r; "
                "the global uncertainty values file at %r will be shadowed.",
                str(path),
                str(GLOBAL_UNCERTAINTY_VALUES_PATH),
            )

    if not path.exists():
        return {}

    # item 74: return cached result if the file has not changed since last load
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    with _contributor_cache_lock:
        cached_mtime, cached_result = _contributor_cache.get(path, (None, None))
        if cached_mtime == mtime and cached_result is not None:
            return cached_result

    raw = load_json_file(path, "custom contributor")
    if not isinstance(raw, dict):
        raise ValueError(
            "Custom contributor library must be a JSON object keyed by element symbol."
        )

    from config.global_uncertainty_values_loader import _normalize_element_symbol

    result: Dict[str, List[CustomUncertaintyContributor]] = {}
    for raw_element_symbol, entries in raw.items():
        element_symbol = _normalize_element_symbol(raw_element_symbol)
        if element_symbol in result:
            raise ValueError(
                f"Duplicate element symbol {element_symbol!r} in custom contributor "
                f"library (entry {raw_element_symbol!r} normalizes to an existing key)."
            )
        result[element_symbol] = parse_custom_contributor_entries(
            element_symbol, entries
        )

    # item 74: store in mtime cache before returning
    with _contributor_cache_lock:
        _contributor_cache[path] = (mtime, result)

    return result


def contributors_for_element(
    element_symbol: str,
    library: Optional[Dict[str, List[CustomUncertaintyContributor]]] = None,
) -> List[CustomUncertaintyContributor]:
    """Return definitions for a single element from the library.

    If *library* is ``None``, loads from the default JSON path on every call.
    """
    if library is None:
        library = load_custom_contributors()
    return list(library.get(element_symbol, []))

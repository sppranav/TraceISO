"""Cell-validation helpers for the Global Uncertainty Manager.

``tools/global_uncertainty_manager`` edits the
``config/global_uncertainty_values.json`` schema and must enforce
the exact same acceptance rules as its loader
(``config/global_uncertainty_values_loader.py``) so that clicking Save can
never write a value the loader will then reject on the next load.
"""

from __future__ import annotations

import math

from config.validation import parse_dof, parse_nonnegative_finite_float


def parse_permil_cell(raw: str, field_name: str) -> float:
    """Parse a u_rel_permil-style cell using the loader's non-negative finite contract.

    An empty cell defaults to 0.0, matching prior UI behaviour.
    """
    text = (raw or "").strip() or "0.0"
    return parse_nonnegative_finite_float(text, field_name=field_name)


def is_valid_permil_cell(raw: str) -> bool:
    """Return whether *raw* would be accepted by :func:`parse_permil_cell`.

    Delegates to the parser itself (rather than duplicating its
    preprocessing) so the two can never diverge again by construction — a
    prior hand-rolled ``if not raw`` check treated whitespace-only cells
    (e.g. ``"   "``) as invalid even though the parser strips and accepts
    them as ``0.0``.
    """
    try:
        parse_permil_cell(raw, field_name="value")
        return True
    except ValueError:
        return False


def parse_dof_cell(dof_raw: str, *, type_ab: str, field_name: str) -> float:
    """Parse a degrees_of_freedom cell, enforcing finite DoF for Type A rows.

    Mirrors ``config/global_uncertainty_values_loader.py``'s
    ``_parse_custom_contributors``: DoF must be finite >= 1 or 'inf', and Type
    A contributors may never use 'inf' (GUM Type A DoF is always finite).
    ``type_ab`` is normalized the same way the loader normalizes it
    (``strip().upper()``) so this helper enforces the same rule for any
    caller, not only ones that pre-normalize the UI's own value.
    """
    text = (dof_raw or "inf").strip() or "inf"
    dof = parse_dof(text)
    normalized_type_ab = str(type_ab or "B").strip().upper()
    if normalized_type_ab == "A" and not math.isfinite(dof):
        raise ValueError(
            f"{field_name}: Type A contributors require finite "
            f"degrees_of_freedom, got {dof_raw!r}."
        )
    return dof


def is_valid_dof_cell(dof_raw: str, *, type_ab: str) -> bool:
    """Return whether *dof_raw* would be accepted by :func:`parse_dof_cell`."""
    try:
        parse_dof_cell(dof_raw, type_ab=type_ab, field_name="degrees_of_freedom")
        return True
    except ValueError:
        return False

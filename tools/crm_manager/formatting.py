"""Display formatting shared by the CRM manager widgets.

Every editable cell in the manager is round-tripped through text: the widget
renders a number, the user may or may not touch it, and the save collector
parses whatever text is there. Any formatter used on that path must therefore
be *lossless* — a certificate uncertainty of ``4.2e-7`` rendered as
``0.000000`` and read back is a silent 100% reduction of a declared
uncertainty (audit A062).

So fixed-point rendering here states a **minimum** number of decimals, not a
maximum: it grows until the text reconstructs the original float exactly.
``format_uncertainty(2e-5)`` is still ``"0.000020"``; ``4.2e-7`` becomes
``"0.00000042"`` rather than a row of zeros.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Optional

#: What the editor shows, and accepts, for an uncertainty that is unassigned.
#: A blank cell - never ``0`` - because ``u = 0`` asserts the quantity is exact
#: while unassigned means it is unknown.
UNASSIGNED_DISPLAY = ""

#: Decimals an uncertainty shows unless the value needs more to survive the
#: round trip. Six keeps the small certificate uncertainties readable and out
#: of scientific notation.
UNCERTAINTY_MIN_DECIMALS = 6

#: Beyond this, positional notation stops being legible and ``repr`` is used.
_MAX_POSITIONAL_LENGTH = 32

#: Significant digits kept by the read-only display formatters.
DISPLAY_SIGNIFICANT_DIGITS = 6
DISPLAY_VALUE_SIGNIFICANT_DIGITS = 10


def _round_significant(value: Optional[float], digits: int) -> Optional[float]:
    """Round to *digits* significant digits, leaving ``None`` and non-finite alone."""
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return number
    return float(f"{number:.{digits}g}")


def format_exact(value: Optional[float], *, min_decimals: int = 0) -> str:
    """Render *value* as plain decimal text that parses back to it exactly.

    ``repr`` already gives the shortest string that round-trips a float; this
    re-expresses that string in positional notation, so no cell ever shows
    ``4.2e-07`` and no cell ever loses a digit. Non-finite input is returned
    as ``repr`` rather than being rendered as a number.
    """
    if value is None:
        return UNASSIGNED_DISPLAY
    number = float(value)
    if not math.isfinite(number):
        return repr(number)
    integral, _, fractional = format(Decimal(repr(number)), "f").partition(".")
    if len(fractional) < min_decimals:
        fractional = fractional.ljust(min_decimals, "0")
    text = f"{integral}.{fractional}" if fractional else integral
    if len(text) > _MAX_POSITIONAL_LENGTH:
        # A magnitude no certificate carries. Scientific notation is still
        # exact and, unlike 300 zeros, still readable in a table cell.
        return repr(number)
    return text


def format_uncertainty(value: Optional[float]) -> str:
    """Return an uncertainty in fixed-point notation, six decimals or more.

    Used for **editable** cells, so it is exact: ``None`` renders as a blank
    cell, and a value too small for six decimals gets the decimals it needs
    instead of being rounded away.
    """
    return format_exact(value, min_decimals=UNCERTAINTY_MIN_DECIMALS)


def format_display_uncertainty(value: Optional[float]) -> str:
    """Return an uncertainty for a read-only label or preview cell.

    Rounded to :data:`DISPLAY_SIGNIFICANT_DIGITS` significant digits, so a
    derived quantity does not spill seventeen digits of float noise into a
    table, and then padded to six decimals. Rounding to *significant* digits
    rather than to a fixed decimal place is what keeps a small quantity such
    as ``4.2e-7`` visible instead of collapsing it to ``0.000000``.
    """
    return format_exact(
        _round_significant(value, DISPLAY_SIGNIFICANT_DIGITS),
        min_decimals=UNCERTAINTY_MIN_DECIMALS,
    )


def format_display_value(value: Optional[float]) -> str:
    """Return a value for a read-only preview cell, positional and rounded."""
    return format_exact(_round_significant(value, DISPLAY_VALUE_SIGNIFICANT_DIGITS))


def format_coverage_factor(value: Optional[float]) -> str:
    """Return a coverage factor, blank when there is no uncertainty to cover."""
    return format_exact(value)


def format_value(value: Optional[float]) -> str:
    """Return a certified value, a natural ratio or a mass, losslessly."""
    return format_exact(value)


def parse_optional_number(text: str) -> Optional[float]:
    """Parse an editor cell, mapping a blank cell to ``None`` (unassigned).

    Raises ``ValueError`` for text that is neither blank nor a number, so a
    typo can never be silently read as unassigned.
    """
    stripped = str(text).strip()
    if not stripped:
        return None
    return float(stripped)

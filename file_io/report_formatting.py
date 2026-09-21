"""Shared human-report formatting for CSV and Excel.

JSON and HDF5 remain canonical full-precision interchange formats.  Ceiling
rounding is a conservative TraceISO reporting policy, not a mandatory GUM
rounding rule.
"""

from __future__ import annotations

from file_io.sanitize import round_uncertainty_up


def report_uncertainty(value: float, significant_digits: int = 2) -> float:
    """Apply the TraceISO ceiling-rounding policy to a report uncertainty."""
    return round_uncertainty_up(value, significant_digits)

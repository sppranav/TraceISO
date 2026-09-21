"""Significant-figure formatting utilities for GUM-compliant display."""

from __future__ import annotations

import math
from typing import Optional

from file_io.report_formatting import report_uncertainty
from file_io.sanitize import round_uncertainty_up


def _decimal_places_for(u: float, n_sig: int = 2) -> int:
    """Return the number of decimal places needed to display *u* to *n_sig* sig figs."""
    if u <= 0 or not math.isfinite(u):
        return 6  # fallback
    exponent = math.floor(math.log10(abs(u)))
    dp = -(exponent - n_sig + 1)
    return max(dp, 0)


def format_value_with_uncertainty(
    value: float,
    uncertainty: float,
    *,
    unit: str = "",
    n_sig: int = 2,
) -> str:
    """Format a value ± uncertainty with matched decimal places."""
    if not math.isfinite(value):
        return "NaN"
    if uncertainty <= 0 or not math.isfinite(uncertainty):
        # No meaningful uncertainty — show 6 decimal places
        result = f"{value:.6f}"
        return f"{result} {unit}".strip()

    u_rounded = round_uncertainty_up(uncertainty, n_sig)
    dp = _decimal_places_for(u_rounded, n_sig)
    v_str = f"{value:.{dp}f}"
    u_str = f"{u_rounded:.{dp}f}"
    result = f"{v_str} \u00b1 {u_str}"
    if unit:
        result = f"{result} {unit}"
    return result


def format_uncertainty(
    uncertainty: float,
    *,
    unit: str = "",
    n_sig: int = 2,
) -> str:
    """Format an uncertainty value alone, rounded UP per GUM 7.2.6."""
    if uncertainty <= 0 or not math.isfinite(uncertainty):
        return f"0 {unit}".strip() if uncertainty == 0 else "NaN"

    u_rounded = round_uncertainty_up(uncertainty, n_sig)
    dp = _decimal_places_for(u_rounded, n_sig)
    result = f"{u_rounded:.{dp}f}"
    if unit:
        result = f"{result} {unit}"
    return result


def format_value_matched(
    value: float,
    uncertainty: float,
    *,
    n_sig: int = 2,
) -> str:
    """Format a value with decimal places matched to the uncertainty."""
    if not math.isfinite(value):
        return "NaN"
    if uncertainty <= 0 or not math.isfinite(uncertainty):
        return f"{value:.6f}"
    u_rounded = round_uncertainty_up(uncertainty, n_sig)
    dp = _decimal_places_for(u_rounded, n_sig)
    return f"{value:.{dp}f}"


def format_with_k(
    value: float,
    uncertainty: float,
    k: float,
    *,
    unit: str = "",
    n_sig: int = 2,
) -> str:
    """Format value ± U (k = ...) with unit — full GUM-compliant display."""
    base = format_value_with_uncertainty(value, uncertainty, unit=unit, n_sig=n_sig)
    return f"{base} (k = {k:.2f})"


def format_reference_value(
    value: float,
    uncertainty: Optional[float],
    coverage_factor: Optional[float],
    uncertainty_semantics: str,
) -> str:
    """Format a reference value without erasing its uncertainty semantics."""
    semantics = str(uncertainty_semantics or "").strip().lower()
    if uncertainty is None or semantics == "unassigned":
        return f"{value:.6g} · uncertainty unassigned"

    k_text = (
        f"{float(coverage_factor):g}"
        if coverage_factor is not None and math.isfinite(float(coverage_factor))
        else "unassigned"
    )
    if semantics == "assigned_exact":
        return f"{value:.6g} · assigned exact (u = 0, k = {k_text})"

    rounded_uncertainty = report_uncertainty(float(uncertainty))
    formatted = format_value_with_uncertainty(value, rounded_uncertainty)
    if semantics == "source_stated_limit":
        return f"{formatted} · source-stated limit (k unassigned)"

    kind = {
        "expanded_uncertainty": "expanded uncertainty",
        "standard_uncertainty": "standard uncertainty",
        "derived": "derived uncertainty",
    }.get(semantics, semantics.replace("_", " ") or "uncertainty semantics unassigned")
    return f"{formatted} · {kind} (k = {k_text})"


def get_unit_label(output_mode: str) -> str:
    """Return the appropriate unit label for the given output mode."""
    if output_mode == "delta":
        return "\u2030"
    return ""


def get_k_footnote(
    *,
    k: Optional[float] = None,
    coverage_method: Optional[str] = None,
    sample_specific: bool = False,
) -> str:
    """Return a user-facing footnote describing the active coverage basis."""
    if coverage_method == "welch_satterthwaite":
        if sample_specific:
            return (
                "Expanded uncertainties use Welch-Satterthwaite coverage; "
                "the applied sample-specific k values are shown with each result."
            )
        if k is not None and math.isfinite(k):
            return f"Expanded uncertainties use Welch-Satterthwaite coverage (current k = {k:.2f})."
        return (
            "Expanded uncertainties use Welch-Satterthwaite coverage; "
            "the applied k value is shown with each result."
        )
    if k is not None and math.isfinite(k):
        return f"Expanded uncertainties reported with k = {k:.2f}."
    return "Expanded uncertainties use the active coverage factor k; the applied value is shown with each result."


# Monte Carlo result presentation
#
# These helpers are the single presentation layer for a durable
# ``MCResultRecord``. They are deliberately separate from the record itself:
# the record stores canonical full-precision machine values, and nothing that
# writes an export or a file ever rounds them. Only the functions below decide
# how many digits a human sees.


def format_mc_interval(record) -> str:
    """Return the Monte Carlo coverage interval with its coverage metadata.

    A coverage interval is never shown without the probability it covers and
    the method that produced it, so a reader cannot mistake it for a ``k = 2``
    expanded uncertainty.
    """
    lower = getattr(record, "mc_lower", None)
    upper = getattr(record, "mc_upper", None)
    if lower is None or upper is None:
        return "not available"
    probability = float(getattr(record, "coverage_probability", 0.0) or 0.0)
    unit = getattr(record, "unit_label", "")
    half_width = (upper - lower) / 2.0
    lower_str = format_value_matched(lower, half_width)
    upper_str = format_value_matched(upper, half_width)
    suffix = f" {unit}" if unit else ""
    return (
        f"[{lower_str}, {upper_str}]{suffix} "
        f"({probability:.0%} {getattr(record, 'interval_convention', '')}, "
        f"method={getattr(record, 'percentile_method', '')})"
    )


def format_mc_summary_rows(record) -> list:
    """Return ``(label, display value)`` rows for one durable MC record.

    Shared by every place the application displays a stored Monte Carlo
    result, so the wording of ``u_c`` and of the expanded uncertainty cannot
    drift between panels.
    """
    unit = getattr(record, "unit_label", "")
    mean = getattr(record, "mc_mean", None)
    std = getattr(record, "mc_std", None)
    diagnostic = getattr(record, "mc_std_diagnostic", None)
    u_c = getattr(record, "gum_u_c", None)
    expanded = getattr(record, "gum_u_expanded", None)
    k = getattr(record, "gum_coverage_factor_k", None)

    rows = []
    if mean is not None:
        scale = std if std is not None else (u_c if u_c is not None else 0.0)
        rows.append(("MC mean", f"{format_value_matched(mean, scale or 0.0)}{f' {unit}' if unit else ''}"))
    else:
        rows.append(("MC mean", "undefined moment"))

    if std is not None:
        rows.append(("MC standard deviation (ddof = 1)", format_uncertainty(std, unit=unit)))
    elif diagnostic is not None:
        rows.append(
            (
                "MC standard deviation (diagnostic only)",
                format_uncertainty(diagnostic, unit=unit),
            )
        )
    else:
        rows.append(("MC standard deviation (ddof = 1)", "undefined moment"))

    rows.append(("MC coverage interval", format_mc_interval(record)))

    if u_c is not None:
        rows.append(("u_c (GUM combined standard uncertainty)", format_uncertainty(u_c, unit=unit)))
    if expanded is not None and k is not None:
        rows.append(
            (
                "U (GUM expanded uncertainty)",
                f"{format_uncertainty(expanded, unit=unit)} (k = {k:.2f})",
            )
        )
    return rows

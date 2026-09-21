"""Shared sanitization helpers for file I/O."""

from __future__ import annotations

import math
import re
from decimal import Decimal, ROUND_CEILING
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from domain.models import UncertaintyBudget


# CSV has no native text-cell type: quote formula-like strings, including minus.
# Numeric values retain their numeric type and sign.
_DANGEROUS_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")


def literal_excel_text(value: Any) -> Any:
    """Preserve XLSX text exactly; workbook finalization enforces native text type.

    Unlike CSV, XLSX does not require an apostrophe added to the stored value.
    Only use this in writers that call finalize_workbook_evidence before save.
    """
    return value

_SUPERSCRIPT_MAP = {
    "0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3",
    "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077",
    "8": "\u2078", "9": "\u2079", "+": "\u207a", "-": "\u207b",
}


def sanitize_spreadsheet_text(value: Any) -> Any:
    """Prefix spreadsheet-formula-like text so it stays literal."""
    if not isinstance(value, str) or not value:
        return value

    if value.startswith(_DANGEROUS_PREFIXES) or value.lstrip().startswith(_DANGEROUS_PREFIXES):
        return "'" + value
    return value


def round_uncertainty_up(u: float, n_sig: int = 2) -> float:
    """Apply TraceISO's conservative ceiling policy to *n_sig* figures.

    This is a project reporting choice, not a claim that the GUM mandates
    upward rounding.

    The policy is stated in *decimal* significant figures, so the arithmetic
    is done in :mod:`decimal` on the value's shortest round-tripping decimal
    form rather than on the binary double. Scaling a double by a power of ten
    and calling :func:`math.ceil` reads 0.14 as 14.000000000000002 and lifts
    an already two-figure value to 0.15 -- a 7% inflation contributed by the
    export alone, and one that moves again on a second application. Working
    from ``repr`` makes the helper exact at decimal boundaries and idempotent:
    a value already expressed at *n_sig* figures is returned unchanged.
    """
    if u <= 0 or not math.isfinite(u):
        return u
    # Ensure n_sig is at least 1
    n_sig = max(1, int(n_sig))
    # ``repr`` is the shortest decimal that round-trips to this double, i.e.
    # the decimal the value stands for; Decimal(float) would reintroduce the
    # binary residue this helper exists to avoid.
    value = Decimal(repr(float(u)))
    quantum = Decimal(1).scaleb(value.adjusted() - n_sig + 1)
    return float(value.quantize(quantum, rounding=ROUND_CEILING))


def format_contributor_name(name: str) -> str:
    """Format an uncertainty contributor key for user-facing exports."""
    from config.contributor_names import canonical_contributor_display_label

    if not name:
        return ""
    canonical_label = canonical_contributor_display_label(name)
    if canonical_label is not None:
        return canonical_label
    if name == "u_std_repeatability":
        return "Standard repeatability"
    if name == "u_kappa_drift":
        return "Instrumental drift"
    if name == "u_k5_instrumental_drift":
        return "Instrumental drift (k5)"
    if name == "u_k1_sample_decomposition":
        return "Sample decomposition (k1)"
    if name in {"u_k2", "u_k2_matrix_separation"}:
        return "Matrix separation (k2)"
    if name in {"u_k3", "u_k3_procedural_blank"}:
        return "Procedural blank (k3)"
    if name == "u_k6_matrix_effects":
        return "Matrix effects (k6)"
    if name in {"u_k7", "u_k7_residual_interferences"}:
        return "Residual interferences (k7)"
    return name.replace("_", " ").replace("u ", "").strip().title()


def cycle_data_equivalent(left: Any, right: Any) -> bool:
    """Backward-compatible alias for numerical cycle-data equivalence."""
    from domain.layers import cycle_data_nearly_equal

    return cycle_data_nearly_equal(left, right)


def safe_float(value: Any) -> Optional[float]:
    """Convert to float, replacing NaN/Inf with None for JSON compatibility."""
    if value is None:
        return None
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def to_superscript(text: str) -> str:
    """Convert digit string to Unicode superscript characters."""
    return "".join(_SUPERSCRIPT_MAP.get(ch, ch) for ch in text)


def format_ratio_token(token: str) -> str:
    """Format a single isotope token: '87Sr' -> '\\u2078\\u2077Sr'."""
    match = re.match(r"^(\d+)([A-Za-z].*)$", token.strip())
    if not match:
        return token.strip()
    mass, element = match.groups()
    return f"{to_superscript(mass)}{element}"


def format_isotope_label(name: str) -> str:
    """Format isotope or ratio label with Unicode superscript mass numbers.

    Examples:
        '87Sr/86Sr' -> '\\u2078\\u2077Sr/\\u2078\\u2076Sr'
        '11B'       -> '\\u00b9\\u00b9B'
    """
    if "/" not in name:
        return format_ratio_token(name)
    num, den = name.split("/", 1)
    return f"{format_ratio_token(num)}/{format_ratio_token(den)}"


#: Contributor states that are not a gap in the budget: a deliberate user or
#: profile choice, or a term that does not enter the reported quantity. These
#: must never be reported as "Incomplete".
_BY_DESIGN_STATES = frozenset({"BY_SAMPLE_DESIGN", "BY_GLOBAL_DESIGN", "NOT_APPLICABLE"})


def build_budget_note(budget: "UncertaintyBudget") -> str:
    """Return a short note flagging genuinely missing budget contributors.

    Only contributors whose data could not be evaluated (``MISSING_DATA``)
    count as "Incomplete". Contributors switched off in the global uncertainty
    configuration (``BY_GLOBAL_DESIGN``) or by a per-sample profile
    (``BY_SAMPLE_DESIGN``) are intentional omissions, and a term that does not
    apply to the output (``NOT_APPLICABLE``, e.g. the CRM certificate in delta
    output) is not missing either; all are excluded from the note.
    """
    limitations = list(getattr(budget, "coverage_limitations", None) or ())
    coverage_note = "; ".join(
        f"{item.get('code', 'coverage_limitation')}: {item.get('explanation', '')}"
        for item in limitations
    )
    if not hasattr(budget, "contributors") or not budget.contributors:
        return coverage_note
    missing = [
        c
        for c in budget.contributors
        if not c.is_active and getattr(c, "state", "") not in _BY_DESIGN_STATES
    ]
    if not missing:
        return coverage_note
    kappa_missing = [c for c in missing if "kappa" in c.name.lower()]
    if kappa_missing and len(kappa_missing) == len(missing):
        note = "No \u03ba-factors assigned"
        return f"{note}; {coverage_note}" if coverage_note else note
    if kappa_missing:
        names = ", ".join(
            c.display_name or c.name for c in missing if c not in kappa_missing
        )
        note = (
            f"No \u03ba-factors assigned; also missing: {names}"
            if names
            else "No \u03ba-factors assigned"
        )
        return f"{note}; {coverage_note}" if coverage_note else note
    names = ", ".join(c.display_name or c.name for c in missing)
    note = f"Incomplete: {names}"
    return f"{note}; {coverage_note}" if coverage_note else note


def summarize_budget_state(budget: "UncertaintyBudget") -> str:
    """Return the contributor-state summary of a numeric budget.

    Precedence: a per-sample design exclusion, then a global one, then any
    contributor that could not be evaluated, otherwise ``ACTIVE``. A term that
    does not apply to the output is neither an exclusion nor a gap. Shared by
    the CSV ``State`` column and the Excel ``Budget State`` column.
    """
    states = {
        getattr(c, "state", "")
        for c in (getattr(budget, "contributors", []) or [])
        if not c.is_active
    }
    # A missing or unapproved model is more consequential than an unrelated
    # contributor that the operator deliberately disabled.  Keep that coverage
    # gap visible in the compact state column; the note carries its full reason.
    if "NO_APPROVED_MODEL" in states:
        return "INCOMPLETE COVERAGE"
    if "MISSING_DATA" in states or any(state not in _BY_DESIGN_STATES for state in states):
        return "MISSING DATA"
    if "BY_SAMPLE_DESIGN" in states:
        return "BY SAMPLE DESIGN"
    if "BY_GLOBAL_DESIGN" in states:
        return "BY GLOBAL DESIGN"
    return "ACTIVE"

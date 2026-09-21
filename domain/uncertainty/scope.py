"""Canonical uncertainty-budget scope semantics shared by UI and exports."""

from __future__ import annotations

import json
from typing import Any, Dict


INVALID_BUDGET_SCOPES = frozenset({"insufficient_data", "unavailable"})

UNCERTAINTY_SCOPE_SCHEMA_NAME = "traceiso.uncertainty_scope"
UNCERTAINTY_SCOPE_SCHEMA_VERSION = "1.1"


#: Canonical statement of the scalar per-ratio budget boundary (review item S-4).
#:
#: Each ratio budget is evaluated independently and combined by RSS within that
#: ratio. Correlations *between* ratios - shared bracketing standards, shared CRM
#: values, shared mass-bias parameters, common session effects - are not
#: propagated and no covariance is exported. Treating two exported ratio budgets
#: as independent in a downstream multi-ratio calculation therefore misstates the
#: uncertainty of the derived quantity.
#:
#: This text is shown wherever a per-ratio budget is displayed or exported and is
#: embedded in every serialized uncertainty export through the versioned scope
#: object below.
CROSS_RATIO_INDEPENDENCE_NOTE = (
    "Scalar per-ratio uncertainty budgets are not covariance-aware. Ratios are "
    "evaluated independently and no cross-ratio covariance is propagated or "
    "exported, so these budgets must not be treated as independent when several "
    "ratios are combined downstream."
    " Coverage is limited to implemented components; reference authority and "
    "laboratory model qualification remain incomplete. Sr two-refinement "
    "stability is not an accuracy bound."
)


def uncertainty_scope_payload() -> Dict[str, object]:
    """Return the versioned cross-ratio boundary shared by every exporter."""
    return {
        "schema_name": UNCERTAINTY_SCOPE_SCHEMA_NAME,
        "schema_version": UNCERTAINTY_SCOPE_SCHEMA_VERSION,
        "budget_basis": "scalar_per_ratio",
        "cross_ratio_covariance_exported": False,
        "downstream_independence_assumption_permitted": False,
        "reference_authority_status": "incomplete",
        "laboratory_coverage_status": "not_qualified",
        "note": CROSS_RATIO_INDEPENDENCE_NOTE,
    }


def canonical_uncertainty_scope_json() -> str:
    """Serialize the scope object identically in JSON, CSV, Excel, and HDF5."""
    return json.dumps(
        uncertainty_scope_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def budget_scope(budget: Any) -> str:
    """Return a normalized uncertainty-budget scope string."""
    return str(getattr(budget, "budget_scope", "") or "").strip().lower()


def is_invalid_budget_scope(budget: Any) -> bool:
    """Return whether a budget cannot support numeric uncertainty output."""
    return budget_scope(budget) in INVALID_BUDGET_SCOPES


def budget_scope_label(budget: Any) -> str:
    """Return the stable export/UI state label for an invalid budget."""
    scope = budget_scope(budget)
    if scope == "insufficient_data":
        return "INSUFFICIENT DATA"
    if scope == "unavailable":
        if (
            str(getattr(budget, "engine", "") or "") == "pb_tl_standard_calibration"
            and str(getattr(budget, "output_mode", "") or "").strip().lower() == "delta"
            and str(getattr(budget, "scope_note", "") or "").startswith("Not calculated")
        ):
            # Calibrated Pb delta: the deferred uncertainty is not calculated.
            return "NOT CALCULATED"
        return "UNAVAILABLE"
    return ""


def budget_scope_note(budget: Any) -> str:
    """Return an explicit reason for an invalid budget scope."""
    note = str(getattr(budget, "scope_note", "") or "").strip()
    if note:
        return note
    scope = budget_scope(budget)
    if scope == "insufficient_data":
        return "Insufficient data to calculate an uncertainty budget."
    if scope == "unavailable":
        return "Uncertainty budget is unavailable for this result."
    return ""

"""Held-out residual diagnostic of the Pb-standard calibration functional (combined Pb plan C05, P8).

The calibration corrects a sample with an equal-average functional of standard
means: the nearest eligible standard on each side (alternating), the neighbouring
blocks (block average) or the whole pool (session mean of K). Holding one unit of
the pool out and applying *the same functional* to it gives the correction error
that functional leaves at a standard position:

    alternating  r_j = S_j / ((S_{j-1} + S_{j+1}) / 2) - 1          interior members only
    block        r_k = mean_k / ((mean_{k-1} + mean_{k+1}) / 2) - 1  interior blocks only
    session      r_j = S_j * mean_{k != j}(1 / S_k) - 1              every member

No interpolation or extrapolation is used, so endpoints have no row. With the
linear-form coefficients ``A`` of those rows (+1 at the held-out unit, -1/2 per
bracketing side, observation-level ``1/n`` and ``-1/(2 n)`` weights for blocks,
``-1/(N-1)`` for the session) and ``M = A'A``:

    tau2   = (sum r^2 - sum_o u_o^2 M_oo) / tr(M)
    nu_res = 2 tau2^2 / (2 tr((M S)^2) / tr(M)^2 + sum_o 2 u_o^4 M_oo^2 / (nu_o tr(M)^2)),
             S = diag(tau2 + u_o^2)

``u_o`` is the relative standard error of standard ``o``'s mean and ``nu_o =
n_o - 1``. For the session map with equal ``u_o`` this is the ISO 5725-2
variance-component estimator and C01's centering-map Satterthwaite form.

The estimate includes any layout or drift mismatch the functional leaves
(remediation design section 5.3), hence its label. It is a diagnostic: the held-out
rows are statistically dependent and no effective-DoF model for them has been
validated, so under owner decision D3 it does not enter a combined uncertainty or
Monte Carlo. A non-positive estimate or fewer than two rows is ``not_resolved``,
never zero (owner decision Q-03).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

RESIDUAL_METHOD = "held_out_equal_average_functional.v1"
RESIDUAL_LABEL = "residual_scatter_including_layout_mismatch"
RESIDUAL_ESTIMATED = "estimated"
RESIDUAL_NOT_RESOLVED = "not_resolved"

RESIDUAL_REASONS: Mapping[str, str] = {
    "fewer_than_two_held_out_rows": (
        "Fewer than two held-out rows: interior standards (alternating), interior blocks (block average) "
        "or pool members (session) are needed to estimate residual scatter."
    ),
    "pool_member_without_se": "A standard entering a held-out row has no measured standard error (one valid cycle).",
    "nonpositive_residual_estimate": (
        "The residual scatter is fully explained by the precision of the standard means (estimate <= 0)."
    ),
    "record_predates_residual_diagnostic": (
        "The calibration record predates the residual diagnostic (schema 1.0); reprocess the session."
    ),
}

MODE_ALTERNATING = "local_ssb_alternating"
MODE_BLOCK = "local_ssb_block"
MODE_SESSION = "session_mean_k"


@dataclass(frozen=True)
class PoolStandard:
    """One calibration-eligible standard in active run order, on its frozen support."""

    observation_id: str
    s_mean: float
    se: Optional[float]
    n_valid: int


def held_out_rows(
    means: Sequence[float], mode: str, blocks: Optional[Sequence[Sequence[int]]] = None,
) -> List[Tuple[float, Dict[int, float], Tuple[int, ...]]]:
    """``(residual, {pool index: linear coefficient}, held-out pool indices)`` per row."""
    rows: List[Tuple[float, Dict[int, float], Tuple[int, ...]]] = []
    if mode == MODE_ALTERNATING:
        for j in range(1, len(means) - 1):
            bracket = (means[j - 1] + means[j + 1]) / 2.0
            rows.append((means[j] / bracket - 1.0, {j: 1.0, j - 1: -0.5, j + 1: -0.5}, (j,)))
    elif mode == MODE_BLOCK:
        units = [list(block) for block in (blocks or ())]
        centres = [float(np.mean([means[i] for i in unit])) for unit in units]
        for k in range(1, len(units) - 1):
            bracket = (centres[k - 1] + centres[k + 1]) / 2.0
            coefficients: Dict[int, float] = {i: 1.0 / len(units[k]) for i in units[k]}
            for side in (k - 1, k + 1):
                for i in units[side]:
                    coefficients[i] = coefficients.get(i, 0.0) - 1.0 / (2.0 * len(units[side]))
            rows.append((centres[k] / bracket - 1.0, coefficients, tuple(units[k])))
    elif mode == MODE_SESSION:
        n = len(means)
        for j in range(n if n >= 2 else 0):
            others = [means[k] for k in range(n) if k != j]
            residual = means[j] * float(np.mean([1.0 / v for v in others])) - 1.0
            rows.append((residual, {k: (1.0 if k == j else -1.0 / (n - 1)) for k in range(n)}, (j,)))
    else:
        raise ValueError(f"Unknown calibration mode {mode!r}")
    return rows


def calibration_residual_diagnostic(
    pool: Sequence[PoolStandard],
    mode: str,
    blocks: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, Any]:
    """The residual diagnostic of one ratio's calibration, as a JSON-safe mapping."""
    ids = [p.observation_id for p in pool]
    means = [float(p.s_mean) for p in pool]
    rows = held_out_rows(means, mode, blocks)
    out: Dict[str, Any] = {
        "method": RESIDUAL_METHOD, "label": RESIDUAL_LABEL, "applied_mode": mode,
        "status": RESIDUAL_NOT_RESOLVED, "reason_code": "", "reason": "",
        "n_pool": len(pool), "n_rows": len(rows), "pool_observation_ids": ids,
        "rows": [{"held_out": [ids[i] for i in held], "residual": float(r)} for r, _c, held in rows],
        "trace_M": None, "sum_r2": None, "precision_term": None, "tau2": None, "nu_res": None,
    }

    def not_resolved(code: str) -> Dict[str, Any]:
        out.update(reason_code=code, reason=RESIDUAL_REASONS[code])
        return out

    if len(rows) < 2:
        return not_resolved("fewer_than_two_held_out_rows")
    a = np.zeros((len(rows), len(pool)))
    r = np.array([row[0] for row in rows], dtype=float)
    for i, (_res, coefficients, _held) in enumerate(rows):
        for j, value in coefficients.items():
            a[i, j] = value
    m = a.T @ a
    diag = np.diag(m)
    trace = float(np.trace(m))
    if any(diag[j] > 0.0 and pool[j].se is None for j in range(len(pool))):
        return not_resolved("pool_member_without_se")
    u2 = np.array([0.0 if p.se is None else (float(p.se) / float(p.s_mean)) ** 2 for p in pool])
    sum_r2 = float(r @ r)
    precision = float(np.sum(u2 * diag))
    tau2 = (sum_r2 - precision) / trace
    out.update(trace_M=trace, sum_r2=sum_r2, precision_term=precision, tau2=tau2)
    if not math.isfinite(tau2) or tau2 <= 0.0:
        return not_resolved("nonpositive_residual_estimate")
    ms = m @ np.diag(tau2 + u2)
    variance_term = 2.0 * float(np.trace(ms @ ms)) / trace ** 2
    precision_dof_term = sum(
        2.0 * u2[j] ** 2 * diag[j] ** 2 / ((pool[j].n_valid - 1) * trace ** 2)
        for j in range(len(pool))
        if diag[j] > 0.0 and pool[j].n_valid >= 2
    )
    out.update(status=RESIDUAL_ESTIMATED, nu_res=2.0 * tau2 ** 2 / (variance_term + precision_dof_term))
    return out


def relative_sample_effect_variance(diagnostic: Mapping[str, Any], relative_sensitivities: Sequence[float]) -> Optional[float]:
    """``tau2 (1 + sum a_j^2)``: the sample's own run effect plus each standard's through K (C01 P8)."""
    tau2 = diagnostic.get("tau2") if diagnostic.get("status") == RESIDUAL_ESTIMATED else None
    if tau2 is None:
        return None
    return float(tau2) * (1.0 + float(sum(a * a for a in relative_sensitivities)))

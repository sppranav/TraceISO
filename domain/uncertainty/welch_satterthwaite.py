"""Welch-Satterthwaite effective degrees of freedom and coverage factor."""

from __future__ import annotations

import math
import sys
from typing import List, Optional, Sequence, Tuple
import warnings

from domain.models import UncertaintyContributor


def _logsumexp(log_terms: Sequence[float]) -> float:
    """Return ``log(sum(exp(term)))`` without losing extreme finite terms."""
    largest = max(log_terms)
    return largest + math.log(math.fsum(math.exp(term - largest) for term in log_terms))


def _stable_finite_ws_dof(
    positive_uncertainties: Sequence[float],
    finite_dof_components: Sequence[Tuple[float, float]],
) -> Optional[float]:
    """Evaluate the finite-DoF W-S quotient in log space.

    Returns ``None`` when the result is not representable in IEEE-754. That is
    deliberately distinct from ``inf``: a positive contributor with finite DoF is
    evidence *against* the Gaussian limit, so an arithmetic failure must never be
    reported as infinite degrees of freedom.

    Scaling by the largest uncertainty before exponentiating keeps the fourth
    powers away from the underflow and overflow ends of the range; W-S is
    invariant to that common factor.
    """
    log_u = [math.log(u) for u in positive_uncertainties]
    largest = max(log_u)
    log_numerator = 2.0 * _logsumexp([2.0 * (value - largest) for value in log_u])
    log_denominator = _logsumexp([
        4.0 * (math.log(u) - largest) - math.log(dof)
        for u, dof in finite_dof_components
    ])
    log_result = log_numerator - log_denominator
    if not math.isfinite(log_result):
        return None
    # Representable-range guards, so exp() cannot silently saturate to inf or 0.
    if log_result > math.log(sys.float_info.max):
        return None
    if log_result < math.log(sys.float_info.min):
        return None
    resolved = math.exp(log_result)
    return resolved if math.isfinite(resolved) and resolved > 0.0 else None


def effective_dof(
    components: Sequence[Tuple[float, float]],
) -> float:
    """Welch-Satterthwaite effective degrees of freedom.

    Takes a sequence of (u_i, nu_i) tuples where u_i is the standard
    uncertainty of the i-th contributor as it appears in the combined
    variance u_c^2 = sum(u_i^2), and nu_i is the corresponding degrees
    of freedom. All u_i must be in the same scale as u_c.

    ``inf`` is returned only when it is *meaningful*: no components, no non-zero
    uncertainty, or every contributor is Type B. When a positive finite-DoF
    contributor exists but the quotient is not representable, the result is
    ``nan`` — never ``inf``, which :func:`coverage_factor` would otherwise turn
    into an anti-conservative k = 2 justified by nothing but numerical underflow.
    """
    if not components:
        return float('inf')

    positive_uncertainties: List[float] = []
    finite_dof_components: List[Tuple[float, float]] = []
    for u_i, nu_i in components:
        u_i = float(u_i)
        if not math.isfinite(u_i):
            # A non-finite contributor makes the combined variance meaningless.
            return float('nan')
        if u_i > 0.0:
            positive_uncertainties.append(u_i)
        if nu_i == float('inf'):
            # Type B / effectively infinite DoF → term vanishes.
            continue
        if nu_i <= 0 or not math.isfinite(nu_i):
            if u_i > 0.0:
                warnings.warn(
                    f"Welch-Satterthwaite: skipping contributor with invalid DoF={nu_i!r}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            continue
        if u_i > 0.0:
            finite_dof_components.append((u_i, float(nu_i)))

    if not positive_uncertainties:
        # Every contributor is zero: no variance, so no DoF information.
        return float('inf')
    if not finite_dof_components:
        # Every contributing term is Type B — the genuine Gaussian limit.
        return float('inf')

    resolved = _stable_finite_ws_dof(positive_uncertainties, finite_dof_components)
    return float('nan') if resolved is None else resolved


def effective_dof_from_contributors(
    contributors: List[UncertaintyContributor],
) -> float:
    """Convenience wrapper: extract (u_i, nu_i) from UncertaintyContributor list.

    Uses ``value_abs`` — appropriate for Engine A which combines in absolute units.
    For Engine B/C which combine in relative permil space, use
    :func:`effective_dof_from_contributors_permil` instead (scale-invariant in
    practice, but explicit consistency is preferred).
    """
    components = [
        (c.value_abs, c.degrees_of_freedom)
        for c in contributors
        if c.is_active
    ]
    return effective_dof(components)


def effective_dof_from_contributors_permil(
    contributors: List[UncertaintyContributor],
) -> float:
    """Convenience wrapper using ``value_rel_permil`` for W-S weights.

    Engine B (SSB-delta) and Engine C (Pb-Tl) combine in relative permil space.
    Calling this helper makes the W-S scale explicit and self-consistent with
    those engines' RSS step.  The result is numerically identical to
    :func:`effective_dof_from_contributors` (W-S is scale-invariant when all
    contributors share the same proportionality to ratio_mean), but the explicit
    scale guards against future accidental mixing.
    """
    components = [
        (c.value_rel_permil, c.degrees_of_freedom)
        for c in contributors
        if c.is_active
    ]
    return effective_dof(components)


def coverage_factor(
    nu_eff: float,
    confidence: float = 0.95,
) -> float:
    """Coverage factor k from Student-t for supported effective DoF.

    ``nu_eff = nan`` means :func:`effective_dof` could not represent the quotient
    even though a finite-DoF contributor was present. That is not the Gaussian
    limit, so it must not silently yield k = 2; it raises instead.
    """
    if math.isnan(nu_eff):
        raise ValueError(
            "Welch-Satterthwaite effective DoF is not representable for this budget; "
            "k cannot be derived from it. Use a fixed coverage factor, or check the "
            "contributor magnitudes."
        )
    if math.isinf(nu_eff) and nu_eff > 0.0:
        return 2.0
    if not math.isfinite(nu_eff) or nu_eff < 1.0:
        raise ValueError(
            "Welch-Satterthwaite effective DoF must be finite >= 1 or positive infinity."
        )

    from scipy.stats import t as t_dist

    alpha_half = (1.0 - confidence) / 2.0
    return float(t_dist.ppf(1.0 - alpha_half, float(nu_eff)))

"""GUM-compliant uncertainty propagation primitives for TraceISO.

Pure, shared helpers used by all three uncertainty engines: each is a single
Type A (statistical) or Type B (systematic) evaluation, or an independent RSS
combination. Cross-ratio covariance is not modelled here (retired 2026-06-19).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from config.constants import FRACTION_TO_PPM


# Type A — statistical

def u_precision(ratio_values: np.ndarray) -> Tuple[float, float, float]:
    """Within-run precision from the finite cycles, as ``(SE, SD, mean)``.

    SD uses ``ddof=1`` and ``SE = SD / sqrt(n)``. Returns NaNs when fewer than
    two finite values are present (mean is still returned for exactly one).
    """
    values = ratio_values[np.isfinite(ratio_values)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if n < 2:
        return float("nan"), float("nan"), mean
    sd = float(np.std(values, ddof=1))
    se = sd / np.sqrt(n)
    return se, sd, mean


def select_precision_value(mode: str, se: float, sd: float) -> Tuple[float, str]:
    """Pick the active u_prec value for *mode* (``"se"`` default, or ``"sd"``);
    returns ``(value, resolved_mode)``."""
    normalized = str(mode or "se").strip().lower()
    if normalized == "sd":
        return float(sd), "sd"
    return float(se), "se"


def u_blank_uncorrelated(
    u_num_blank: float,
    u_den_blank: float,
    num_corrected: float,
    den_corrected: float,
) -> float:
    """Blank uncertainty on a ratio, treating the numerator and denominator
    blanks as independent.

    Propagates the blank SDs through the sensitivity coefficients
    ``c_num = -1/den`` and ``c_den = num/den**2``. Returns 0 if the denominator
    is zero or non-finite.
    """
    if not np.isfinite(den_corrected) or den_corrected == 0:
        return 0.0
    c_num = -1.0 / den_corrected
    c_den = num_corrected / (den_corrected ** 2)
    return np.sqrt((c_num * u_num_blank) ** 2 + (c_den * u_den_blank) ** 2)


def u_blank_correlated(
    u_num_blank: float,
    u_den_blank: float,
    num_corrected: float,
    den_corrected: float,
    correlation: float,
) -> float:
    """Blank uncertainty on a ratio when the numerator and denominator blanks
    are correlated.

    Adds the covariance term to the independent variance. The two sensitivity
    coefficients have opposite signs, so a positive correlation *reduces* the
    result. Invalid required inputs raise ValueError.
    """
    if not np.isfinite(correlation) or not -1 <= correlation <= 1:
        raise ValueError("Blank correlation must be finite and lie in [-1, 1].")
    if not all(np.isfinite(v) for v in (u_num_blank, u_den_blank, num_corrected, den_corrected)) or min(u_num_blank, u_den_blank) < 0 or den_corrected == 0:
        raise ValueError("Required blank propagation input is invalid.")
    if not np.isfinite(den_corrected) or den_corrected == 0:
        return 0.0
    c_num = -1.0 / den_corrected
    c_den = num_corrected / (den_corrected ** 2)
    covariance = u_num_blank * u_den_blank * correlation
    variance = (
        (c_num * u_num_blank) ** 2
        + (c_den * u_den_blank) ** 2
        + 2.0 * c_num * c_den * covariance
    )
    return np.sqrt(max(variance, 0.0))


def u_ssb_standards(
    prev_std_se: float,
    next_std_se: float,
) -> float:
    """SSB bracketing-standard uncertainty: RSS of the two flanking standard
    SEs, halved."""
    return np.sqrt(prev_std_se ** 2 + next_std_se ** 2) / 2.0


def u_session_reproducibility(
    std_means: np.ndarray,
) -> Tuple[float, float, int]:
    """Session reproducibility as the SD (``ddof=1``) of the per-standard means —
    the full scatter, not its standard error.

    Returns ``(sd, mean, dof)`` with ``dof = n - 1``; SD is NaN when fewer than
    two finite standards are available.
    """
    values = std_means[np.isfinite(std_means)]
    n = len(values)
    if n < 2:
        return (
            float("nan"),
            float(np.mean(values)) if n == 1 else float("nan"),
            max(n - 1, 0),
        )
    mean_val = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    return sd, mean_val, n - 1


# Type B — systematic

def u_certified_value(
    u_certified: float,
    k_factor: float = 2.0,
) -> float:
    """Convert a CRM certificate's expanded uncertainty to a standard
    uncertainty (``U / k``).

    The certificate's ``k`` is independent of the report coverage factor. Raises
    ``ValueError`` on a non-positive ``k`` or a negative/non-finite ``U``.
    """
    if not np.isfinite(k_factor) or k_factor <= 0:
        raise ValueError(
            f"CRM coverage factor k must be finite and > 0, got {k_factor!r}."
        )
    if not np.isfinite(u_certified) or u_certified < 0:
        raise ValueError(
            "CRM expanded uncertainty must be a non-negative finite value, "
            f"got {u_certified!r}."
        )
    return u_certified / k_factor


def u_rectangular(half_width: float) -> float:
    """Rectangular (uniform) distribution: ``u = a / sqrt(3)`` for half-width ``a``."""
    return half_width / np.sqrt(3.0)


# Independent combination

def combine_type_a(*components: float) -> float:
    """RSS combination of independent Type A components."""
    return np.sqrt(sum(c ** 2 for c in components))


def combine_type_b(*components: float) -> float:
    """RSS combination of independent Type B components."""
    return np.sqrt(sum(c ** 2 for c in components))


def combine_and_expand(
    u_type_a: float,
    u_type_b: float,
    coverage_factor: float = 2.0,
) -> Tuple[float, float]:
    """Combine Type A and Type B by independent RSS and expand by ``k``;
    returns ``(u_c, U)``."""
    u_c = np.sqrt(u_type_a ** 2 + u_type_b ** 2)
    return u_c, coverage_factor * u_c


def to_ppm(u_absolute: float, ratio_value: float) -> float:
    """Relative uncertainty in ppm (``|u / ratio| * 1e6``); 0 if the ratio is
    zero or non-finite."""
    if ratio_value == 0 or not np.isfinite(ratio_value):
        return 0.0
    return abs(u_absolute / ratio_value) * FRACTION_TO_PPM

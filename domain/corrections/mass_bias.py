"""Mass-bias (fractionation) correction using Russell's exponential law."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MassBiasResult:
    """Result of mass-bias correction for one sample."""

    f_values: np.ndarray
    f_mean: float
    target_gamma: float
    normalization_k: np.ndarray
    target_k: np.ndarray
    n_valid_cycles: int = -1  # -1 = legacy (not checked)

    @property
    def f_factor(self) -> np.ndarray:
        """Backward-compatible alias for ``f_values``."""
        return self.f_values

    @property
    def gamma_87_86(self) -> float:
        """Backward-compatible alias for the current Sr target gamma."""
        return self.target_gamma

    @property
    def k_86_88(self) -> np.ndarray:
        """Backward-compatible alias for the current Sr normalization K."""
        return self.normalization_k

    @property
    def k_87_86(self) -> np.ndarray:
        """Backward-compatible alias for the current Sr target K."""
        return self.target_k


def _require_mass(value: Optional[float], label: str) -> float:
    """Return *value* as float or raise a clear ``ValueError``."""
    if value is None:
        raise ValueError(
            f"Managed reference mass '{label}' is required for mass-bias correction."
        )
    return float(value)


def calculate_f_factor(
    normalization_ratio_measured: np.ndarray,
    normalization_ratio_reference: float,
    normalization_numerator_mass: Optional[float] = None,
    normalization_denominator_mass: Optional[float] = None,
) -> np.ndarray:
    """Per-cycle fractionation factor *f* from Russell's law:
    ``f = ln(R_ref / R_meas) / ln(M_num / M_den)`` over the normalization isotope
    pair. Non-positive or non-finite measured ratios yield NaN."""
    normalization_numerator_mass = _require_mass(
        normalization_numerator_mass,
        "normalization numerator",
    )
    normalization_denominator_mass = _require_mass(
        normalization_denominator_mass,
        "normalization denominator",
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.log(normalization_ratio_reference / normalization_ratio_measured) / np.log(
            normalization_numerator_mass / normalization_denominator_mass
        )
    return f


def calculate_gamma(
    numerator_mass: float,
    denominator_mass: float,
    normalization_numerator_mass: Optional[float] = None,
    normalization_denominator_mass: Optional[float] = None,
) -> float:
    """Compute mass-dependent exponent gamma.

    gamma = ln(M_target_num / M_target_den) / ln(M_norm_num / M_norm_den)
    """
    normalization_numerator_mass = _require_mass(
        normalization_numerator_mass,
        "normalization numerator",
    )
    normalization_denominator_mass = _require_mass(
        normalization_denominator_mass,
        "normalization denominator",
    )
    return np.log(numerator_mass / denominator_mass) / np.log(
        normalization_numerator_mass / normalization_denominator_mass
    )


def calculate_k_factors(
    normalization_ratio_measured: np.ndarray,
    normalization_ratio_reference: float,
    normalization_numerator_mass: Optional[float] = None,
    target_numerator_mass: Optional[float] = None,
    normalization_denominator_mass: Optional[float] = None,
    target_denominator_mass: Optional[float] = None,
) -> MassBiasResult:
    """K-factors for internal normalisation (IIF correction).

    Per cycle, ``normalization_k = R_ref / R_meas`` and
    ``target_k = normalization_k ** gamma``. Non-finite cycles become NaN; an
    all-invalid normalization returns a NaN result flagged with
    ``n_valid_cycles=0`` so downstream code can mark the sample invalid."""
    normalization_numerator_mass = _require_mass(
        normalization_numerator_mass,
        "normalization numerator",
    )
    target_numerator_mass = _require_mass(
        target_numerator_mass,
        "target numerator",
    )
    normalization_denominator_mass = _require_mass(
        normalization_denominator_mass,
        "normalization denominator",
    )
    if target_denominator_mass is None:
        target_denominator_mass = normalization_numerator_mass
    else:
        target_denominator_mass = _require_mass(
            target_denominator_mass,
            "target denominator",
        )
    f_values = calculate_f_factor(
        normalization_ratio_measured,
        normalization_ratio_reference,
        normalization_numerator_mass,
        normalization_denominator_mass,
    )

    # Check for all-invalid normalization support before computing mean
    n_finite = int(np.sum(np.isfinite(f_values)))
    if n_finite == 0:
        import logging
        logging.getLogger(__name__).warning(
            "calculate_k_factors: zero finite f-values from normalization ratio — "
            "all K-factors will be NaN. Sample should be marked invalid."
        )
        # Return explicit NaN result with a flag for downstream detection
        nan_array = np.full_like(f_values, np.nan)
        return MassBiasResult(
            f_values=f_values,
            f_mean=float("nan"),
            target_gamma=calculate_gamma(
                target_numerator_mass,
                target_denominator_mass,
                normalization_numerator_mass,
                normalization_denominator_mass,
            ),
            normalization_k=nan_array,
            target_k=nan_array,
            n_valid_cycles=0,
        )

    target_gamma = calculate_gamma(
        target_numerator_mass,
        target_denominator_mass,
        normalization_numerator_mass,
        normalization_denominator_mass,
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        normalization_k = normalization_ratio_reference / normalization_ratio_measured
        # Replace non-finite K-factors with nan so individual invalid cycles don't
        # propagate inf into downstream arithmetic (e.g. apply_iif_correction).
        normalization_k = np.where(np.isfinite(normalization_k), normalization_k, np.nan)
        target_k = np.power(normalization_k, target_gamma)

    return MassBiasResult(
        f_values=f_values,
        f_mean=float(np.nanmean(f_values)),
        target_gamma=target_gamma,
        normalization_k=normalization_k,
        target_k=target_k,
        n_valid_cycles=n_finite,
    )


def apply_iif_correction(
    ratio_values: np.ndarray,
    k_factors: np.ndarray,
) -> np.ndarray:
    """Apply the IIF K-factor correction per cycle: ``ratio * k`` element-wise
    (never mean × mean)."""
    return ratio_values * k_factors


def mass_bias_correct_ratio(
    ratio_measured: np.ndarray,
    f: np.ndarray,
    m_numerator: float,
    m_denominator: float,
) -> np.ndarray:
    """Apply Russell's law to correct a ratio using precomputed *f*.

    R_true = R_measured × (M_num / M_den)^f
    """
    return ratio_measured * np.power(m_numerator / m_denominator, f)

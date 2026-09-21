"""Isobaric interference corrections for Sr isotope analysis.

Two forms are provided: an *f-family* (natural-abundance scaling with a measured
Russell exponent, used by a declared single-pass Sr configuration) and a
*K-family* (mass-bias-corrected scaling, used by the Sr initialization with unity K
and by both K refinements in ``domain/corrections/sr_chain.py``). Both subtract a monitored interferent from the target isotope using a
proxy monitor isotope and its natural abundance.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import logging
import numpy as np

from domain.corrections.mass_bias import (
    calculate_gamma,
)
from domain.elements.base import MonitorSpec


_logger = logging.getLogger(__name__)
_NEGATIVE_FRACTION_WARN = 0.10


def _require_scalar(value: Optional[float], label: str) -> float:
    """Return *value* as float or raise a clear ``ValueError``."""
    if value is None:
        raise ValueError(
            f"Managed reference value '{label}' is required for interference correction."
        )
    return float(value)


def _isobaric_interference_f(
    *,
    target: np.ndarray,
    monitor: np.ndarray,
    f: np.ndarray,
    natural_ratio: float,
    m_monitor: float,
    m_interferent: float,
) -> tuple:
    """Natural-abundance (f-family) interference correction:
    ``interference = monitor * natural_ratio * (m_monitor / m_interferent) ** f``,
    then ``corrected = target - interference``."""
    if not (target.shape == monitor.shape == f.shape):
        raise ValueError(
            f"Interference primitive shape mismatch: target={target.shape}, "
            f"monitor={monitor.shape}, f={f.shape}. All three must match."
        )
    interference = monitor * natural_ratio * np.power(m_monitor / m_interferent, f)
    corrected = target - interference
    return corrected, interference


def _isobaric_interference_k(
    *,
    target: np.ndarray,
    monitor: np.ndarray,
    k_norm: np.ndarray,
    natural_ratio: float,
    m_interferent: float,
    m_monitor: float,
    m_norm_num: float,
    m_norm_den: float,
) -> tuple:
    """Apply K-factor isobaric interference correction.

    Gamma uses the interferent/monitor mass pair, both with their own
    precise atomic masses.
    """
    if not (target.shape == monitor.shape == k_norm.shape):
        raise ValueError(
            f"Interference primitive shape mismatch: target={target.shape}, "
            f"monitor={monitor.shape}, k_norm={k_norm.shape}. All three must match."
        )
    gamma = calculate_gamma(m_interferent, m_monitor, m_norm_num, m_norm_den)
    interference = monitor * natural_ratio * np.power(k_norm, -gamma)
    corrected = target - interference
    return corrected, interference




def rb_interference_correction(
    sr87: np.ndarray,
    rb85: np.ndarray,
    f: np.ndarray,
    rb87_rb85_natural: Optional[float] = None,
    m85: Optional[float] = None,
    m87: Optional[float] = None,
) -> tuple:
    """Correct 87Sr for 87Rb isobaric interference."""
    rb87_rb85_natural = _require_scalar(rb87_rb85_natural, "Rb 87Rb/85Rb")
    m85 = _require_scalar(m85, "85Rb")
    m87 = _require_scalar(m87, "87Rb")
    return _isobaric_interference_f(
        target=sr87,
        monitor=rb85,
        f=f,
        natural_ratio=rb87_rb85_natural,
        m_monitor=m85,
        m_interferent=m87,
    )


def kr84_interference_correction(
    sr84: np.ndarray,
    kr83: np.ndarray,
    f: np.ndarray,
    kr84_kr83_natural: Optional[float] = None,
    m83: Optional[float] = None,
    m84: Optional[float] = None,
) -> tuple:
    """Correct 84Sr for 84Kr isobaric interference."""
    kr84_kr83_natural = _require_scalar(kr84_kr83_natural, "Kr 84Kr/83Kr")
    m83 = _require_scalar(m83, "83Kr")
    m84 = _require_scalar(m84, "84Kr")
    return _isobaric_interference_f(
        target=sr84,
        monitor=kr83,
        f=f,
        natural_ratio=kr84_kr83_natural,
        m_monitor=m83,
        m_interferent=m84,
    )


def kr86_interference_correction(
    sr86: np.ndarray,
    kr83: np.ndarray,
    f: np.ndarray,
    kr86_kr83_natural: Optional[float] = None,
    m83: Optional[float] = None,
    m86_kr: Optional[float] = None,
) -> tuple:
    """Correct 86Sr for 86Kr isobaric interference."""
    kr86_kr83_natural = _require_scalar(kr86_kr83_natural, "Kr 86Kr/83Kr")
    m83 = _require_scalar(m83, "83Kr")
    m86_kr = _require_scalar(m86_kr, "86Kr")
    return _isobaric_interference_f(
        target=sr86,
        monitor=kr83,
        f=f,
        natural_ratio=kr86_kr83_natural,
        m_monitor=m83,
        m_interferent=m86_kr,
    )


def hg204_interference_correction(
    pb204: np.ndarray,
    hg202: np.ndarray,
    f_tl: np.ndarray,
    hg204_hg202_natural: Optional[float] = None,
    m202: Optional[float] = None,
    m204_hg: Optional[float] = None,
) -> tuple:
    """Correct apparent 204Pb for 204Hg interference monitored on 202Hg."""
    hg204_hg202_natural = _require_scalar(hg204_hg202_natural, "Hg 204Hg/202Hg")
    m202 = _require_scalar(m202, "202Hg")
    m204_hg = _require_scalar(m204_hg, "204Hg")
    return _isobaric_interference_f(
        target=pb204,
        monitor=hg202,
        f=f_tl,
        natural_ratio=hg204_hg202_natural,
        m_monitor=m202,
        m_interferent=m204_hg,
    )




def rb_interference_mass_bias_corrected(
    sr87: np.ndarray,
    rb85: np.ndarray,
    k_norm: np.ndarray,
    rb87_rb85_natural: Optional[float] = None,
    m85: Optional[float] = None,
    m87: Optional[float] = None,
    m_norm_num: Optional[float] = None,
    m_norm_den: Optional[float] = None,
) -> tuple:
    """Rb interference correction with mass-bias K factors.

    gamma_Rb = ln(M87_Rb / M85_Rb) / ln(M_norm_num / M_norm_den)
    mass_bias = K_norm ^ (−gamma_Rb)
    87Rb_interf = 85Rb × (87Rb/85Rb)_nat × mass_bias
    """
    rb87_rb85_natural = _require_scalar(rb87_rb85_natural, "Rb 87Rb/85Rb")
    m85 = _require_scalar(m85, "85Rb")
    m87 = _require_scalar(m87, "87Rb")
    m_norm_num = _require_scalar(m_norm_num, "normalization numerator")
    m_norm_den = _require_scalar(m_norm_den, "normalization denominator")
    return _isobaric_interference_k(
        target=sr87,
        monitor=rb85,
        k_norm=k_norm,
        natural_ratio=rb87_rb85_natural,
        m_interferent=m87,
        m_monitor=m85,
        m_norm_num=m_norm_num,
        m_norm_den=m_norm_den,
    )


def kr84_interference_mass_bias_corrected(
    sr84: np.ndarray,
    kr83: np.ndarray,
    k_norm: np.ndarray,
    kr84_kr83_natural: Optional[float] = None,
    m83: Optional[float] = None,
    m84: Optional[float] = None,
    m_norm_num: Optional[float] = None,
    m_norm_den: Optional[float] = None,
) -> tuple:
    """Kr-84 interference correction with mass-bias K factors."""
    kr84_kr83_natural = _require_scalar(kr84_kr83_natural, "Kr 84Kr/83Kr")
    m83 = _require_scalar(m83, "83Kr")
    m84 = _require_scalar(m84, "84Kr")
    m_norm_num = _require_scalar(m_norm_num, "normalization numerator")
    m_norm_den = _require_scalar(m_norm_den, "normalization denominator")
    return _isobaric_interference_k(
        target=sr84,
        monitor=kr83,
        k_norm=k_norm,
        natural_ratio=kr84_kr83_natural,
        m_interferent=m84,
        m_monitor=m83,
        m_norm_num=m_norm_num,
        m_norm_den=m_norm_den,
    )


def kr86_interference_mass_bias_corrected(
    sr86: np.ndarray,
    kr83: np.ndarray,
    k_norm: np.ndarray,
    kr86_kr83_natural: Optional[float] = None,
    m83: Optional[float] = None,
    m86_kr: Optional[float] = None,
    m_norm_num: Optional[float] = None,
    m_norm_den: Optional[float] = None,
) -> tuple:
    """Kr-86 interference correction with mass-bias K factors."""
    kr86_kr83_natural = _require_scalar(kr86_kr83_natural, "Kr 86Kr/83Kr")
    m83 = _require_scalar(m83, "83Kr")
    m86_kr = _require_scalar(m86_kr, "86Kr")
    m_norm_num = _require_scalar(m_norm_num, "normalization numerator")
    m_norm_den = _require_scalar(m_norm_den, "normalization denominator")
    return _isobaric_interference_k(
        target=sr86,
        monitor=kr83,
        k_norm=k_norm,
        natural_ratio=kr86_kr83_natural,
        m_interferent=m86_kr,
        m_monitor=m83,
        m_norm_num=m_norm_num,
        m_norm_den=m_norm_den,
    )


def apply_interference_corrections(
    intensities: Dict[str, np.ndarray],
    f: np.ndarray,
    monitors: Tuple[MonitorSpec, ...],
    reference_data: Dict[str, float],
    enabled_interferents: Optional[Set[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Generic Step-1 (f-family) interference correction loop driven by MonitorSpec.

    Iterates over *monitors* and applies ``_isobaric_interference_f`` for each
    entry where both ``monitor_isotope`` and ``corrected_isotope`` are present
    in *intensities*. Missing measurement channels are silently skipped.
    Missing managed reference data raises ValueError (fail closed).
    """
    result: Dict[str, np.ndarray] = {}
    for spec in monitors:
        if not isinstance(spec, MonitorSpec) or spec.family != "f":
            continue
        if (
            enabled_interferents is not None
            and spec.interfering_isotope not in enabled_interferents
        ):
            continue
        # Skip if measurement channels are absent (analytically, no correction applies).
        if (
            spec.monitor_isotope not in intensities
            or spec.corrected_isotope not in intensities
        ):
            continue
        # Fail closed if managed reference data is missing (metrologically required).
        try:
            natural_ratio = reference_data[spec.natural_ratio_key]
            m_monitor = reference_data[spec.monitor_isotope]
            m_interferent = reference_data[spec.interfering_isotope]
        except KeyError as exc:
            raise ValueError(
                f"Managed reference data missing for monitor "
                f"{spec.corrected_isotope}<-{spec.interfering_isotope}/"
                f"{spec.monitor_isotope}: key {exc.args[0]!r} not in reference_data. "
                f"Check config/crm_library.json."
            ) from None
        if natural_ratio is None or m_monitor is None or m_interferent is None:
            raise ValueError(
                f"Managed reference data is None for monitor "
                f"{spec.corrected_isotope}<-{spec.interfering_isotope}/"
                f"{spec.monitor_isotope}. Check config/crm_library.json."
            )
        corrected, _ = _isobaric_interference_f(
            target=intensities[spec.corrected_isotope],
            monitor=intensities[spec.monitor_isotope],
            f=f,
            natural_ratio=natural_ratio,
            m_monitor=m_monitor,
            m_interferent=m_interferent,
        )
        n_neg = int(np.sum(corrected < 0))
        n_total = corrected.size
        if n_total > 0 and (n_neg / n_total) >= _NEGATIVE_FRACTION_WARN:
            _logger.warning(
                "apply_interference_corrections: %d/%d corrected %s cycles "
                "are negative (%.1f%%). Interferent (%s via %s) likely "
                "dominates the analyte signal.",
                n_neg,
                n_total,
                spec.corrected_isotope,
                100.0 * n_neg / n_total,
                spec.interfering_isotope,
                spec.monitor_isotope,
            )
        result[spec.corrected_isotope] = corrected
    return result

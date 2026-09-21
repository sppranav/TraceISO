"""Kragten numerical perturbation for Sr isobaric interference uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np

from domain.corrections.sr_chain import (
    SR_CHAIN_METHOD,
    SrInterferenceTerm,
    run_sr_chain,
)

@dataclass
class InterferenceUncertainty:
    """Result of Kragten perturbation for Sr interference corrections.

    All absolute values are on the 87Sr/86Sr ratio scale.
    All relative values are in permil.
    """

    u_rb_abs: float
    u_rb_rel_permil: float

    u_kr86_abs: float
    u_kr86_rel_permil: float

    u_kr84_abs: float
    u_kr84_rel_permil: float

    u_interf_abs: float
    u_interf_rel_permil: float

    # Type B -- infinite degrees of freedom
    degrees_of_freedom: float = float("inf")


@dataclass(frozen=True)
class SrInterferenceReferenceInputs:
    """Managed reference data and the Sr method required for the chain replay.

    ``chain_method`` is the Sr correction method the replay evaluates; it must be
    the method production used for the sample (see
    :mod:`domain.uncertainty.sr_chain_identity`).
    """

    rb87_rb85: float
    kr84_kr83: float
    kr86_kr83: float
    u_rel_rb: Optional[float]
    u_rel_kr84: float
    u_rel_kr86: float
    normalization_numerator: str
    normalization_denominator: str
    m_norm_num: float
    m_norm_den: float
    m86_sr: float
    m87_sr: float
    m88_sr: float
    m83_kr: float
    m84_kr: float
    m86_kr: float
    m85_rb: float
    m87_rb: float
    chain_method: str = SR_CHAIN_METHOD


def _run_sr_correction_chain(
    intensities: Dict[str, np.ndarray],
    normalization_value: float,
    mask: Optional[np.ndarray] = None,
    *,
    reference_inputs: SrInterferenceReferenceInputs,
    rb87_rb85: Optional[float] = None,
    kr84_kr83: Optional[float] = None,
    kr86_kr83: Optional[float] = None,
    apply_iif: bool = True,
    apply_interference: bool = True,
    enabled_interferents: Optional[Set[str]] = None,
) -> float:
    """Replay the production Sr chain for one sample and return its masked mean.

    The replay runs :func:`domain.corrections.sr_chain.run_sr_chain` with
    ``reference_inputs.chain_method``, so the Kragten sensitivities and the
    Engine A Monte Carlo replay evaluate the method production used. Only the
    87Rb and 86Kr terms are replayed: the 84Kr correction changes 84Sr alone,
    which neither the supported 86Sr/88Sr normalization nor 87Sr/86Sr reads, so
    ``kr84_kr83`` has no effect here. A chain that cannot complete returns the
    NaN sentinel; an unknown method raises.
    """
    norm_num_label = reference_inputs.normalization_numerator
    norm_den_label = reference_inputs.normalization_denominator
    if norm_num_label not in intensities or norm_den_label not in intensities:
        return float("nan")  # item 51: NaN sentinel — callers guard with np.isfinite

    rb87_rb85 = (
        reference_inputs.rb87_rb85 if rb87_rb85 is None else float(rb87_rb85)
    )
    kr86_kr83 = (
        reference_inputs.kr86_kr83 if kr86_kr83 is None else float(kr86_kr83)
    )
    rb_enabled = (
        apply_interference
        and (enabled_interferents is None or "87Rb" in enabled_interferents)
    )
    kr86_enabled = (
        apply_interference
        and (enabled_interferents is None or "86Kr" in enabled_interferents)
    )

    terms: List[SrInterferenceTerm] = []
    if rb_enabled and "85Rb" in intensities:
        terms.append(
            SrInterferenceTerm(
                target="87Sr",
                monitor="85Rb",
                natural_ratio=rb87_rb85,
                m_interferent=reference_inputs.m87_rb,
                m_monitor=reference_inputs.m85_rb,
            )
        )
    if kr86_enabled and "83Kr" in intensities:
        terms.append(
            SrInterferenceTerm(
                target="86Sr",
                monitor="83Kr",
                natural_ratio=kr86_kr83,
                m_interferent=reference_inputs.m86_kr,
                m_monitor=reference_inputs.m83_kr,
            )
        )

    chain = run_sr_chain(
        intensities,
        terms=terms,
        normalization_numerator=norm_num_label,
        normalization_denominator=norm_den_label,
        normalization_reference=normalization_value,
        m_norm_num=reference_inputs.m_norm_num,
        m_norm_den=reference_inputs.m_norm_den,
        m_target_num=reference_inputs.m87_sr,
        m_target_den=reference_inputs.m86_sr,
        method=reference_inputs.chain_method,
    )
    # With no interference term and no normalization the K-factors are never
    # read, so an unusable normalization ratio does not invalidate the replay.
    if not chain.complete and (apply_iif or terms):
        return float("nan")

    ratio_87_86 = chain.final.target_ratio("87Sr", "86Sr", apply_iif=apply_iif)

    # Apply mask and return mean
    if mask is not None:
        ratio_87_86 = ratio_87_86[mask]
    if not np.all(np.isfinite(ratio_87_86)):
        return float("nan")
    valid = ratio_87_86
    if len(valid) == 0:
        return float("nan")  # item 51: NaN sentinel — callers guard with np.isfinite
    return float(np.mean(valid))


def compute_interference_uncertainty(
    intensities: Dict[str, np.ndarray],
    normalization_value: float,
    mask: Optional[np.ndarray] = None,
    *,
    reference_inputs: SrInterferenceReferenceInputs,
    apply_iif: bool = True,
    enabled_interferents: Optional[Set[str]] = None,
) -> InterferenceUncertainty:
    """Kragten numerical perturbation for Sr interference uncertainty."""
    rb_nom = reference_inputs.rb87_rb85
    kr86_nom = reference_inputs.kr86_kr83

    # Absolute perturbation sizes
    delta_rb = (
        rb_nom * reference_inputs.u_rel_rb
        if reference_inputs.u_rel_rb is not None
        else 0.0
    )
    delta_kr86 = kr86_nom * reference_inputs.u_rel_kr86

    # Common kwargs for the correction chain
    common = dict(
        intensities=intensities,
        normalization_value=normalization_value,
        mask=mask,
        reference_inputs=reference_inputs,
        apply_iif=apply_iif,
        enabled_interferents=enabled_interferents,
    )

    # Nominal (unperturbed) result
    R_nom = _run_sr_correction_chain(**common)

    if not np.isfinite(R_nom) or R_nom == 0.0:
        raise ValueError("Sr interference uncertainty nominal chain is unavailable.")

    u_rb = 0.0
    if enabled_interferents is None or "87Rb" in enabled_interferents:
        R_rb_up = _run_sr_correction_chain(
            **common, rb87_rb85=rb_nom + delta_rb,
        )
        R_rb_down = _run_sr_correction_chain(
            **common, rb87_rb85=rb_nom - delta_rb,
        )
        # item 51: treat NaN chain-replay as zero sensitivity (missing isotopes
        # means the perturbation has no defined effect on the ratio).
        if not np.isfinite(R_rb_up) or not np.isfinite(R_rb_down):
            raise ValueError("Sr Rb reference perturbation failed on selected support.")
        if np.isfinite(R_rb_up) and np.isfinite(R_rb_down):
            u_rb = abs(R_rb_up - R_rb_down) / 2.0

    u_kr86 = 0.0
    if "83Kr" in intensities and (
        enabled_interferents is None or "86Kr" in enabled_interferents
    ):
        R_kr86_up = _run_sr_correction_chain(
            **common, kr86_kr83=kr86_nom + delta_kr86,
        )
        R_kr86_down = _run_sr_correction_chain(
            **common, kr86_kr83=kr86_nom - delta_kr86,
        )
        # item 51: treat NaN chain-replay as zero sensitivity.
        if not np.isfinite(R_kr86_up) or not np.isfinite(R_kr86_down):
            raise ValueError("Sr Kr reference perturbation failed on selected support.")
        if np.isfinite(R_kr86_up) and np.isfinite(R_kr86_down):
            u_kr86 = abs(R_kr86_up - R_kr86_down) / 2.0

    # 84Kr correction modifies 84Sr only, which does not feed back into
    # the 87Sr/86Sr ratio chain.  The contribution is identically zero

    # use with 84Sr/86Sr ratios.
    u_kr84 = 0.0

    u_interf = float(np.sqrt(u_rb**2 + u_kr86**2 + u_kr84**2))

    return InterferenceUncertainty(
        u_rb_abs=u_rb,
        u_rb_rel_permil=(u_rb / R_nom) * 1000.0,
        u_kr86_abs=u_kr86,
        u_kr86_rel_permil=(u_kr86 / R_nom) * 1000.0,
        u_kr84_abs=u_kr84,
        u_kr84_rel_permil=(u_kr84 / R_nom) * 1000.0,
        u_interf_abs=u_interf,
        u_interf_rel_permil=(u_interf / R_nom) * 1000.0,
    )

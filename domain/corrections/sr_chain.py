"""Sr Rb/Kr interference and internal-normalization chain (A025).

The correction is circular: the normalization K-factor comes from the
interference-corrected 86Sr/88Sr, while the mass-bias-aware Kr and Rb
corrections need K. The standard TraceISO method follows the laboratory
workbook sequence:

1. **Initialization** subtracts ``monitor * natural ratio`` from each enabled
   target, with no mass-bias term, and derives ``K0`` from the corrected
   normalization ratio.
2. **Refinement 1** restarts from the original blank-corrected intensities,
   subtracts ``monitor * natural ratio * K0**(-gamma_interferent)`` and derives
   ``K1``. ``gamma_interferent = ln(m_interferent/m_monitor) /
   ln(m_norm_num/m_norm_den)``.
3. **Refinement 2** restarts from the originals again with ``K1`` and derives
   ``K2``.

Only K carries between stages; interference is never subtracted cumulatively.
The reported target ratio is formed from the refinement-2 intensities and, when
internal normalization is applied, multiplied once per cycle by
``K2**gamma_target``.

An element configuration declaring ``iterations <= 1`` selects the single-pass
model instead: the Russell exponent ``f`` is measured on the uncorrected
normalization ratio, each enabled interference is subtracted once with
``(m_monitor/m_interferent)**f``, and the final K comes from the corrected
normalization ratio. No initialization or refinement is added to it.

Two refinements is a fixed method, not a convergence criterion. The movement of
the final ratio between the two refinements is reported as a numerical stability
diagnostic only: a small movement neither demonstrates accuracy nor identifies
the physical root of the circular correction, which is why A025 stays open.

This module is pure array arithmetic built on the existing interference and
mass-bias primitives. ``ProcessingPipeline`` and the Kragten/Monte Carlo replay
both call it with the same method, so production and the uncertainty replays
evaluate one model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from domain.corrections.interference import (
    _isobaric_interference_f,
    _isobaric_interference_k,
)
from domain.corrections.mass_bias import (
    MassBiasResult,
    apply_iif_correction,
    calculate_f_factor,
    calculate_k_factors,
)

#: Identity of the standard Sr method. Change it whenever the arithmetic changes.
SR_CHAIN_METHOD = "sr_natural_init_two_refinements_v1"
SR_CHAIN_INITIALIZATION_PASSES = 1
SR_CHAIN_REFINEMENT_PASSES = 2

#: Identity of the one-pass model kept for element configurations that declare
#: ``iterations <= 1``: a single natural-abundance correction scaled by the
#: Russell exponent measured on the uncorrected normalization ratio.
SR_SINGLE_PASS_METHOD = "sr_single_pass_measured_f_v1"

#: Every method this build can evaluate. Anything else is refused, never
#: silently treated as the standard method.
SUPPORTED_SR_CHAIN_METHODS: Tuple[str, ...] = (SR_CHAIN_METHOD, SR_SINGLE_PASS_METHOD)

#: Sample metadata written by the Sr chain. Cleared before every processing run.
SR_CHAIN_METADATA_KEYS: Tuple[str, ...] = (
    "_sr_chain_method",
    "_sr_chain_initialization_passes",
    "_sr_chain_refinement_passes",
    "_sr_refinement_change_abs",
    "_sr_refinement_change_rel",
    "_sr_refinement_n_compared",
    "_sr_refinement_max_abs_change",
    "_sr_refinement_max_rel_change",
    "_sr_refinement_threshold_rel",
    "_sr_refinement_exceeds_threshold",
)

#: Diagnostics of the retired convergence-driven solver. A sample carrying any of
#: them from an earlier session must not keep a convergence claim the current
#: method never makes, so they are cleared too.
RETIRED_SR_SOLVER_METADATA_KEYS: Tuple[str, ...] = (
    "_sr_solver_iterations",
    "_sr_solver_delta",
    "_sr_solver_converged",
    "_sr_solver_max_abs_f",
    "_sr_solver_reason",
    "_sr_solver_contraction",
    "_sr_solver_diverged",
    "_sr_solver_outside_envelope",
)


class UnknownSrChainMethodError(ValueError):
    """An Sr correction method this build does not implement was requested."""


def require_sr_chain_method(method: object) -> str:
    """Return ``method`` if it is a supported Sr method identifier, else raise."""
    text = method if isinstance(method, str) else ""
    if text not in SUPPORTED_SR_CHAIN_METHODS:
        raise UnknownSrChainMethodError(
            f"Unknown Sr correction method {method!r}. This build implements "
            f"{', '.join(SUPPORTED_SR_CHAIN_METHODS)}; reprocess the session "
            f"instead of evaluating it with a substitute method."
        )
    return text


def sr_chain_method_for_iterations(iterations: object) -> str:
    """The element-configuration rule: ``iterations <= 1`` selects the single pass."""
    return SR_SINGLE_PASS_METHOD if int(iterations) <= 1 else SR_CHAIN_METHOD


@dataclass(frozen=True)
class SrInterferenceTerm:
    """One monitored isobaric interference on a named target channel."""

    target: str
    monitor: str
    natural_ratio: float
    m_interferent: float
    m_monitor: float


@dataclass(frozen=True)
class SrChainStage:
    """Corrected intensities and K-factors of one correction evaluation."""

    label: str
    intensities: Mapping[str, np.ndarray]
    normalization_ratio: np.ndarray
    mass_bias: MassBiasResult

    @property
    def valid(self) -> bool:
        """False when no cycle gave a finite normalization K-factor."""
        return self.mass_bias.n_valid_cycles != 0

    def target_ratio(
        self, numerator: str, denominator: str, *, apply_iif: bool,
    ) -> np.ndarray:
        """Per-cycle target ratio of this stage, normalized with its own K."""
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = self.intensities[numerator] / self.intensities[denominator]
        if apply_iif:
            ratio = apply_iif_correction(ratio, self.mass_bias.target_k)
        return ratio


@dataclass(frozen=True)
class SrChainResult:
    """All evaluated stages; evaluation stops at the first invalid stage."""

    stages: Tuple[SrChainStage, ...]
    planned_stages: int
    method: str = SR_CHAIN_METHOD

    @property
    def failed_stage(self) -> Optional[SrChainStage]:
        return next((stage for stage in self.stages if not stage.valid), None)

    @property
    def complete(self) -> bool:
        return self.failed_stage is None and len(self.stages) == self.planned_stages

    @property
    def final(self) -> SrChainStage:
        return self.stages[-1]


def stage_labels(method: str = SR_CHAIN_METHOD) -> Tuple[str, ...]:
    if require_sr_chain_method(method) == SR_SINGLE_PASS_METHOD:
        return ("single_pass",)
    return ("initialization",) + tuple(
        f"refinement_{index}" for index in range(1, SR_CHAIN_REFINEMENT_PASSES + 1)
    )


def run_sr_chain(
    intensities: Mapping[str, np.ndarray],
    *,
    terms: Sequence[SrInterferenceTerm],
    normalization_numerator: str,
    normalization_denominator: str,
    normalization_reference: float,
    m_norm_num: float,
    m_norm_den: float,
    m_target_num: float,
    m_target_den: float,
    method: str = SR_CHAIN_METHOD,
) -> SrChainResult:
    """Run the selected Sr method on original blank-corrected intensities.

    ``intensities`` are never modified; every stage starts from them again.
    Within a stage the terms are applied in order, so two terms on one channel
    both subtract. The standard initialization uses a unity K, for which
    ``K**(-gamma)`` is exactly 1.
    """
    method = require_sr_chain_method(method)
    originals = {name: np.asarray(values, dtype=float) for name, values in intensities.items()}
    labels = stage_labels(method)

    def evaluate(label: str, corrected) -> SrChainStage:
        with np.errstate(divide="ignore", invalid="ignore"):
            normalization_ratio = (
                corrected[normalization_numerator] / corrected[normalization_denominator]
            )
        mass_bias = calculate_k_factors(
            normalization_ratio_measured=normalization_ratio,
            normalization_ratio_reference=normalization_reference,
            normalization_numerator_mass=m_norm_num,
            target_numerator_mass=m_target_num,
            normalization_denominator_mass=m_norm_den,
            target_denominator_mass=m_target_den,
        )
        return SrChainStage(
            label=label,
            intensities=corrected,
            normalization_ratio=normalization_ratio,
            mass_bias=mass_bias,
        )

    if method == SR_SINGLE_PASS_METHOD:
        with np.errstate(divide="ignore", invalid="ignore"):
            measured = originals[normalization_numerator] / originals[normalization_denominator]
        f = calculate_f_factor(
            normalization_ratio_measured=measured,
            normalization_ratio_reference=normalization_reference,
            normalization_numerator_mass=m_norm_num,
            normalization_denominator_mass=m_norm_den,
        )
        corrected = {name: values.copy() for name, values in originals.items()}
        for term in terms:
            corrected[term.target], _ = _isobaric_interference_f(
                target=corrected[term.target],
                monitor=originals[term.monitor],
                f=f,
                natural_ratio=float(term.natural_ratio),
                m_monitor=float(term.m_monitor),
                m_interferent=float(term.m_interferent),
            )
        return SrChainResult(
            stages=(evaluate(labels[0], corrected),), planned_stages=1, method=method,
        )

    stages = []
    k_norm: Optional[np.ndarray] = None
    for label in labels:
        corrected = {name: values.copy() for name, values in originals.items()}
        for term in terms:
            target = corrected[term.target]
            stage_k = np.ones(target.shape, dtype=float) if k_norm is None else k_norm
            corrected[term.target], _ = _isobaric_interference_k(
                target=target,
                monitor=originals[term.monitor],
                k_norm=stage_k,
                natural_ratio=float(term.natural_ratio),
                m_interferent=float(term.m_interferent),
                m_monitor=float(term.m_monitor),
                m_norm_num=float(m_norm_num),
                m_norm_den=float(m_norm_den),
            )
        stage = evaluate(label, corrected)
        stages.append(stage)
        if not stage.valid:
            break
        k_norm = stage.mass_bias.normalization_k
    return SrChainResult(stages=tuple(stages), planned_stages=len(labels), method=method)


@dataclass(frozen=True)
class SrRefinementStability:
    """Movement of the final target ratio between refinements 1 and 2.

    Per-cycle arrays hold NaN where either refinement has no finite, non-zero
    ratio, so an unavailable cycle is never reported as zero movement. The
    summaries cover only compared cycles inside the supplied mask and are
    ``None`` when there are none.
    """

    change_abs: np.ndarray
    change_rel: np.ndarray
    n_compared: int
    max_abs_change: Optional[float]
    max_rel_change: Optional[float]
    threshold_rel: float

    @property
    def exceeds_threshold(self) -> Optional[bool]:
        if self.max_rel_change is None:
            return None
        return self.max_rel_change > self.threshold_rel


def refinement_stability(
    result: SrChainResult,
    *,
    numerator: str,
    denominator: str,
    apply_iif: bool,
    threshold_rel: float,
    mask: Optional[np.ndarray] = None,
) -> Optional[SrRefinementStability]:
    """Compare refinement 1 and refinement 2, each normalized with its own K.

    Returns ``None`` for an incomplete chain and for the single-pass method,
    which has no refinement to compare. This is a numerical diagnostic, kept
    apart from the measurement uncertainty budget.
    """
    if result.method != SR_CHAIN_METHOD or not result.complete or len(result.stages) < 3:
        return None
    previous = result.stages[-2].target_ratio(numerator, denominator, apply_iif=apply_iif)
    final = result.final.target_ratio(numerator, denominator, apply_iif=apply_iif)
    comparable = np.isfinite(previous) & np.isfinite(final) & (final != 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        change_abs = np.where(comparable, np.abs(final - previous), np.nan)
        change_rel = np.where(comparable, change_abs / np.abs(final), np.nan)

    active = comparable.copy()
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        n = min(mask_array.size, active.size)
        active[n:] = False
        active[:n] &= mask_array[:n]
    n_compared = int(np.count_nonzero(active))
    return SrRefinementStability(
        change_abs=change_abs,
        change_rel=change_rel,
        n_compared=n_compared,
        max_abs_change=float(np.max(change_abs[active])) if n_compared else None,
        max_rel_change=float(np.max(change_rel[active])) if n_compared else None,
        threshold_rel=float(threshold_rel),
    )

"""Fixed-size Monte Carlo cross-check engine for TraceISO uncertainty budgets.

The Monte Carlo result is a descriptive distributional cross-check of an
analytical GUM budget. It is a descriptive, fixed-draw comparison and
carries no pass/fail verdict for new Engine B semantics.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import (
    MISSING,
    dataclass,
    field,
    fields as dataclass_fields,
    is_dataclass,
    replace,
)
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from config.reference_materials import (
    get_natural_ratio_relative_uncertainty,
    require_isotope_mass,
    require_natural_ratio,
    require_natural_ratio_relative_uncertainty,
)
from config.settings import (
    CustomUncertaintyContributor,
    DEFAULT_MC_ITERATIONS_STANDARD,
    ProcessingConfig,
    UncertaintyConfig,
    is_russell_law_normalization_engine,
)
from domain.filters.outlier import resolve_cycle_range, sample_cycle_key
from domain.elements.base import CertifiedValue, ElementConfig
from domain.models import Sample, UncertaintyBudget
from domain.layer_status import APPLIED as _LAYER_APPLIED
from domain.pb_calibration_records import (
    DELTA_UNCERTAINTY_REASON,
    DELTA_UNCERTAINTY_REASON_CODE,
    PB_CALIBRATION_BUDGET_ENGINE,
    governing_calibration_record,
)
from domain.output_scale import (
    INPUT_LAYER_NORMALIZED,
    INPUT_LAYER_PRE_ANCHOR,
    resolve_output_scale,
)
from domain.corrections.delta import delta_from_values
from domain.corrections.ssb import ssb_correct_single
from domain.ratio_selection import (
    get_best_ratio_data,
    select_best_delta_ratio_layer,
    select_best_pre_ssb_ratio_layer,
)
from domain.uncertainty.blank import (
    blank_input_sigmas,
    compute_blank_stats_3var,
    compute_blank_uncertainty,
    get_paired_blank_voltages,
    resolve_blank_channel_weights,
    resolve_blank_correction_mode,
    resolve_blank_samples,
    resolve_blank_samples_for_uncertainty,
    _compute_correlation,
)
from domain.uncertainty.engine_internal_sr import (
    _get_iif_or_best_ratio,
    _get_pre_anchor_ratio,
    _resolve_active_sr_normalization_pair,
    _resolve_active_sr_normalization_value,
    _resolve_norm_ratio_u_abs,
    compute_digestion_reproducibility_term,
    compute_qc_bias_term,
    AUTOMATIC_REFERENCE_BIAS_ENABLED,
    compute_reference_bias_term,
)
from domain.uncertainty.engine_ssb import (
    _compute_bracketing_standard_uncertainty,
    _get_corrected_intensity_mean,
    _resolve_certified_reference_value,
    _uses_classic_delta_reference,
    resolve_classic_delta_bracket,
)
from domain.uncertainty.sr_chain_identity import (
    engine_a_semantics_for_method,
    replay_sr_chain_method,
)
from domain.uncertainty.kragten import (
    SrInterferenceReferenceInputs,
    _run_sr_correction_chain,
)
from domain.uncertainty.contributors import build_custom_contributor_rows, is_contributor_active
from domain.uncertainty.propagation import (
    select_precision_value,
    u_certified_value,
    u_precision,
)
from domain.uncertainty.reprod import compute_reprod
from domain.uncertainty.pb_hg_ssb_propagation import (
    BLANK_INVALID_COVARIANCE,
    PB_HG_SSB_PROPAGATION_METHOD,
    U_BLANK,
    U_HG_MASS_BIAS_MODEL,
    U_HG_TL_REFERENCE,
    U_INTERF,
    compute_pb_hg_ssb_propagation,
    is_pb_hg_ssb_ratio,
)
from domain.uncertainty.sr_sample_values import (
    resolve_sr_digestion_inputs,
    resolve_sr_qc_bias_inputs,
)

_LOG = logging.getLogger(__name__)

# Contributors drawn with a hardcoded (non-configurable, non-normal) MC
# distribution — the single source of truth shared with the Uncertainty tab's
# MC panel so its "Distribution" column cannot drift from what the sampler
# actually draws.
#
# u_bias_ref was the only entry. Its rectangular draw came from the automatic
# reference-bias model is disabled, so the map is
# empty for V1: advertising a distribution for a term that is never drawn
# would misdescribe the cross-check. It stays as the extension point.
FIXED_NON_NORMAL_CONTRIBUTOR_DISTRIBUTIONS: Dict[str, str] = {}


def _select_u_prec_from_values(
    ratio_values: np.ndarray,
    uncertainty_config: UncertaintyConfig,
) -> float:
    """Return the configured u_prec value from per-cycle values."""
    se, sd, _mean = u_precision(np.asarray(ratio_values, dtype=np.float64))
    selected, _mode = select_precision_value(
        getattr(uncertainty_config, "u_prec_mode", "se"),
        se,
        sd,
    )
    return float(selected)


# ---------------------------------------------------------------------------
# Cross-check contracts
#
# Ported from the reviewed public release per the development/public Monte
# Carlo reconciliation matrix (dispositions E1-E8, R1-R7, F1-F5, C1-C5,
# P1-P5, D1-D6). The matrix forbids wholesale replacement of either file and
# forbids restoring formal pass/fail validation semantics.
# ---------------------------------------------------------------------------

#: Closed set of stable, testable failure reason codes (disposition E1-E6).
MC_CROSS_CHECK_REASON_CODES: frozenset = frozenset(
    {
        "missing_chain_input",
        "invalid_sampled_parameter",
        "nonfinite_model_output",
        "invalid_chain_output",
        "sr_calibration_unavailable",
        "invalid_covariance",
        # A Pb-standard-calibrated ratio: its delta MC is deferred (owner-
        # approved). Its absolute MC is the extended Engine C (C05), which
        # refuses with the codes above.
        "not_implemented_for_calibrated_pb_delta",
    }
)


class MCCrossCheckError(ValueError):
    """A scientific input/model condition that prevents a complete cross-check.

    Subclasses ``ValueError`` so existing ``except ValueError`` callers keep
    working unchanged while gaining a stable ``reason_code``.
    """

    def __init__(
        self,
        *,
        engine: str,
        reason_code: str,
        reason: str,
        iteration: Optional[int] = None,
    ) -> None:
        self.engine = str(engine)
        self.iteration = iteration
        self.reason_code = str(reason_code)
        self.reason = str(reason)
        location = f" at iteration {iteration}" if iteration is not None else ""
        super().__init__(f"{self.engine}{location}: {self.reason}")


# --- Engine B fixed-draw semantics -----------------------------------------

#: Semantics identifier for the fixed-draw SSB/delta chain replay.
ENGINE_B_FIXED_DRAW_SEMANTICS_V1 = "engine_b.fixed_draw.v1"
ENGINE_B_FIXED_DRAW_SEMANTICS_V2 = "engine_b.fixed_draw.v2"
ENGINE_B_FIXED_DRAW_SEMANTICS = "engine_b.fixed_draw.v3.joint_blank"

#: The fixed-draw replay of a Pb ratio whose ordinary-SSB Hg correction was
#: applied. The blank, Hg-reference and Tl-reference terms of the Hg-corrected
#: chain are placed on the reported output at their first-order sensitivities:
#: a linearized approximation, not a draw-level replay of the Hg chain.
ENGINE_B_FIXED_DRAW_PB_HG_SEMANTICS = "engine_b.fixed_draw.v2.pb_hg_linearized.v2.runtime_support"

#: Fixed-draw semantics this build produces; only these can be current.
ENGINE_B_CURRENT_FIXED_DRAW_SEMANTICS: frozenset = frozenset(
    {ENGINE_B_FIXED_DRAW_SEMANTICS, ENGINE_B_FIXED_DRAW_PB_HG_SEMANTICS}
)


def engine_b_semantics_for_ratio(sample: Any, ratio_name: str) -> str:
    """Fixed-draw semantics an Engine B record for this ratio must carry now."""
    if sample is not None and is_pb_hg_ssb_ratio(sample, ratio_name):
        return ENGINE_B_FIXED_DRAW_PB_HG_SEMANTICS
    return ENGINE_B_FIXED_DRAW_SEMANTICS

#: Semantics identifier for the retained additive basis-space Engine B path.
ENGINE_B_LEGACY_SEMANTICS = "engine_b.additive_basis.legacy.v1.joint_blank"

#: Iteration result spaces. ``basis_ratio`` is the legacy space and is the only
#: space for which the frozen post-loop transform is applied.
RESULT_SPACE_BASIS_RATIO = "basis_ratio"
RESULT_SPACE_ABSOLUTE_RATIO = "absolute_ratio"
RESULT_SPACE_DELTA_PERMIL = "delta_permil"

#: Result spaces in which an iteration already returns the final reported
#: quantity. The post-loop transform must be a strict no-op for these.
REPORTED_RESULT_SPACES: frozenset = frozenset(
    {RESULT_SPACE_ABSOLUTE_RATIO, RESULT_SPACE_DELTA_PERMIL}
)

#: Declared measurement-model placement of one sampled contributor.
PLACEMENT_SSB_INPUT = "ssb_input"
PLACEMENT_DELTA_INPUT = "delta_input"
PLACEMENT_OUTPUT_LEVEL = "output_level"
PLACEMENT_BLANK_FALLBACK = "blank_fallback"
#: A shared input drawn once and mapped onto the reported output through its
#: first-order sensitivity (Pb Hg-corrected blank observations).
PLACEMENT_LINEARIZED_OUTPUT = "linearized_output"

#: Student-t moment hierarchy. A scaled t has a mean only for nu > 1 and a
#: variance only for nu > 2, while its quantiles exist for every nu > 0, so the
#: central percentile interval stays the primary reported summary at every DoF.
MOMENT_STATUS_DEFINED = "MEAN_AND_VARIANCE_DEFINED"
MOMENT_STATUS_VARIANCE_UNDEFINED = "VARIANCE_UNDEFINED"
MOMENT_STATUS_MEAN_AND_VARIANCE_UNDEFINED = "MEAN_AND_VARIANCE_UNDEFINED"

#: Central coverage-interval convention. Pinned so a NumPy default change
#: cannot silently move a reported endpoint.
MC_COVERAGE_PROBABILITY = 0.95
MC_INTERVAL_CONVENTION = "central_percentile"
MC_PERCENTILE_METHOD = "linear"
MC_LOWER_PERCENTILE = 2.5
MC_UPPER_PERCENTILE = 97.5

#: Explicit label for the finite-DoF draw implemented by
#: :func:`_draw_standard_uncertainty`. Keeping this distinct from ``normal``
#: makes the immutable parameter record describe the PDF actually sampled.
MC_DISTRIBUTION_SCALED_STUDENT_T = "scaled_student_t"

#: Monte Carlo semantics of a Pb-standard-calibrated absolute Pb ratio (combined
#: Pb plan C05): every draw replays the target and every recorded calibration
#: standard through the Pb-Tl chain and recomputes K from the same functional.
PB_CALIBRATED_MC_SEMANTICS = "engine_c.pb_standard_calibration.v1"
PLACEMENT_TARGET_MEAN = "target_mean_input"
PLACEMENT_STANDARD_MEAN = "standard_mean_input"
PLACEMENT_CALIBRATION_REFERENCE = "calibration_reference_input"
PLACEMENT_SHARED_CHAIN = "shared_chain_input"
#: Draws evaluated per vectorized replay; results do not depend on it.
PB_CALIBRATED_MC_BATCH = 2048

#: Relative floor below which a sampled denominator is refused as near-zero.
#: Scale-aware: compared against the nominal magnitude of the same quantity.
_NEAR_ZERO_RELATIVE_FLOOR = 1.0e-9

#: Minimum observations behind every directly sampled Engine B bracket side.
#:
#: A bracket side estimated from n observations is drawn as a scaled Student-t
#: at nu = n - 1. Two observations therefore give nu = 1, a Cauchy, whose tails
#: routinely carry the sampled bracketing mean B = (R_p + R_n) / 2 out of the
#: positive model domain. Because B is the SSB and delta denominator, that is a
#: genuine domain violation and the run must fail closed; measured on a
#: production two-standard block fixture the per-draw probability is ~3e-5 to
#: ~5e-5, so a 100 000-draw run completes only about 1% of the time and a
#: 500 000-draw run essentially never does. Offering a configuration that
#: cannot complete is worse than declaring it unavailable, so a side with fewer
#: than three observations is refused deterministically during parameter
#: resolution, before any draw.
#:
#: This rule deliberately supersedes the earlier preflight statement
#: that block-average replay "remains available" at nu = 1: PF-B1 settled the
#: PDF and DoF, which are unchanged here, not the availability of a run that
#: almost never finishes.
#:
#: The rule applies identically to ``block_average`` (n = standards per block
#: side), ``alternating`` and ``classic_delta`` (n = valid cycles), because the
#: constraint is the role of R_p/R_n as positive bracket inputs rather than the
#: bracket mode. It does **not** extend to output-level contributors, where
#: nu = 1 cannot invalidate the bracket denominator and remains supported. The
#: distribution itself is never truncated, floored, replaced by a normal, or
#: filtered by discarding draws.
ENGINE_B_MIN_BRACKET_SIDE_OBSERVATIONS = 3

#: Unit of the per-side observation count, by bracket mode, for error wording.
_BRACKET_SIDE_OBSERVATION_NOUN = {
    "block_average": "standards in its block",
    "alternating": "valid cycles",
    "classic_delta": "valid cycles",
}


def _require_bracket_side_observations(
    *,
    prev_n: int,
    next_n: int,
    bracket_mode: str,
    engine: str,
) -> None:
    """Refuse a directly sampled bracket side with too few observations.

    Applies to every Engine B bracket mode. See
    :data:`ENGINE_B_MIN_BRACKET_SIDE_OBSERVATIONS` for why the rule is about the
    bracket denominator rather than about the bracket mode.

    The two causes stay separable in the message: fewer than two observations
    define no standard error at all, while exactly two define one at nu = 1.
    """
    minimum = ENGINE_B_MIN_BRACKET_SIDE_OBSERVATIONS
    if prev_n >= minimum and next_n >= minimum:
        return
    smallest = min(prev_n, next_n)
    noun = _BRACKET_SIDE_OBSERVATION_NOUN.get(bracket_mode, "observations")
    if smallest < 2:
        detail = (
            f"has {smallest} {noun}, so its standard error has no defined "
            "degrees of freedom"
        )
    else:
        detail = (
            f"has {smallest} {noun}, giving nu = {smallest - 1}, at which the "
            "Student-t has no finite mean or variance and its tails routinely "
            "drive the sampled bracketing mean out of the positive model "
            "domain, so a fixed-draw run would almost always fail part-way "
            "through"
        )
    raise MCCrossCheckError(
        engine=engine,
        reason_code="missing_chain_input",
        reason=(
            f"The {bracket_mode} bracket side {detail}. The fixed-draw Engine B "
            f"replay requires at least {minimum} observations behind every "
            "directly sampled bracket side."
        ),
    )


def _require_finite(
    value: float,
    *,
    engine: str,
    reason_code: str,
    what: str,
) -> float:
    """Return *value* as a float, refusing NaN/inf with a stable reason code."""
    out = float(value)
    if not np.isfinite(out):
        raise MCCrossCheckError(
            engine=engine,
            reason_code=reason_code,
            reason=f"{what} is not finite.",
        )
    return out


def _require_safe_denominator(
    value: float,
    *,
    nominal_scale: float,
    engine: str,
    what: str,
) -> float:
    """Refuse a non-finite or scale-aware near-zero division denominator.

    ``nominal_scale`` is the magnitude of the same quantity at its nominal
    value, so the floor tracks the units of the input instead of assuming
    ratios of order one.
    """
    out = float(value)
    if not np.isfinite(out):
        raise MCCrossCheckError(
            engine=engine,
            reason_code="invalid_chain_output",
            reason=f"{what} is not finite.",
        )
    scale = abs(float(nominal_scale))
    floor = _NEAR_ZERO_RELATIVE_FLOOR * scale if np.isfinite(scale) and scale > 0.0 else 0.0
    if abs(out) <= floor or out == 0.0:
        raise MCCrossCheckError(
            engine=engine,
            reason_code="invalid_chain_output",
            reason=f"{what} is zero or within the near-zero floor of its nominal scale.",
        )
    return out


@dataclass
class MCCrossCheckResult:
    """Neutral numerical diagnostics from a completed Monte Carlo cross-check.

    Field order is preserved from the historical ``MCValidationResult`` so
    existing keyword construction in callers and tests keeps working
    (reconciliation dispositions R1-R7). The verdict fields are retained but,
    for new Engine B semantics, ``passed`` and ``failure_reason`` are never
    produced and ``agreement_pct`` / ``center_deviation_pct`` are read-only
    legacy mirrors of the descriptive diagnostics rather than thresholds.
    """

    mc_lower_95: float
    mc_upper_95: float
    mc_mean: Optional[float]
    mc_std: Optional[float]
    gum_lower: float
    gum_upper: float
    gum_u_expanded: float
    n_iter: int
    agreement_pct: Optional[float] = None
    passed: Optional[bool] = None
    validation_mode: str = "custom"
    convergence_mean_pct: float = 0.0
    convergence_std_pct: float = 0.0
    convergence_half_width_pct: float = 0.0
    convergence_q025_pct: float = 0.0
    convergence_q975_pct: float = 0.0
    convergence_checkpoints: Tuple[Tuple[int, float, float, float, float], ...] = ()
    center_deviation_pct: float = 0.0
    scope_note: str = ""
    failure_reason: str = ""
    mc_samples: Optional[np.ndarray] = None
    #: Structurally always zero: a non-finite draw fails the run (disposition
    #: D1/R6) instead of being dropped or nominally substituted.
    n_dropped: int = 0
    status: str = "completed"
    reason_code: str = ""

    # --- descriptive diagnostics (disposition R3) --------------------------
    gum_u_c: float = 0.0
    absolute_center_difference: Optional[float] = None
    center_difference_u_c: Optional[float] = None
    std_difference_pct: Optional[float] = None

    # --- coverage-interval convention (plan 6.3 / A-7) ---------------------
    coverage_probability: float = MC_COVERAGE_PROBABILITY
    interval_convention: str = MC_INTERVAL_CONVENTION
    percentile_method: str = MC_PERCENTILE_METHOD

    # --- Engine B semantics and provenance ---------------------------------
    semantics_version: str = ""
    result_space: str = ""
    execution_id: str = ""
    bracket_mode: str = ""
    output_mode: str = ""
    #: True when the frozen post-loop basis-space transform was applied. It is
    #: applied only for ``basis_ratio`` results and must never be applied to a
    #: result that an iteration already returned in reported space.
    post_loop_transform_applied: bool = False
    moment_status: str = MOMENT_STATUS_DEFINED
    min_type_a_dof: Optional[float] = None
    #: MC standard deviation retained purely as a diagnostic when the moment
    #: status says it is not a valid dispersion estimate.
    mc_std_diagnostic: Optional[float] = None
    contributor_placements: Tuple[Tuple[str, str], ...] = ()
    warnings: Tuple[str, ...] = ()
    #: Canonical digest of the settings the run actually used.
    config_digest: str = ""
    #: Canonical digest of the resolved scientific inputs. Raw values and
    #: labels are not logged; this digest prevents execution-ID collisions
    #: when the configuration is unchanged but the bracket/data differ.
    input_digest: str = ""
    replay_snapshot_json: str = ""
    seed: Optional[int] = None
    bit_generator: str = ""
    numpy_version: str = ""

    # --- Phase 2 persistence provenance (plan 6.2, item P1-O6) -------------
    # Additive, non-calculational identity/provenance fields. Nothing below is
    # read by the sampler, the model kernel or any summary statistic; they are
    # populated once at the single return site so a durable, self-describing
    # result can be persisted and exported without re-running the model.
    #: Scientific identity of the run, so a persisted result cannot become
    #: detached from the sample/ratio it belongs to.
    sample_name: str = ""
    sample_run_number: int = 0
    ratio_name: str = ""
    #: Resolved engine identifier (``ssb_delta``, ``internal_normalization``, ...).
    engine: str = ""
    #: Draws asked for, kept separate from ``n_iter``/``completed_draws``.
    requested_draws: int = 0
    #: Draws actually completed. Mirrors ``n_iter`` and is stored explicitly so
    #: persistence never has to infer which count ``n_iter`` meant.
    completed_draws: int = 0
    #: Frozen bracket identity used for this run.
    prev_std_label: str = ""
    next_std_label: str = ""
    #: Declared model path: whether the SSB kernel was evaluated and whether
    #: delta was taken against the same certified reference (so ``C`` cancels).
    apply_ssb_kernel: Optional[bool] = None
    delta_from_ssb: Optional[bool] = None
    #: Configured sample blank model: ``SD``/``SE`` and its declared placement.
    blank_uncertainty_input: str = ""
    blank_placement: str = ""
    #: Nominal (undrawn) model evaluation, in reported space, plus the two
    #: scale quantities the near-zero denominator guards use.
    nominal_reported_value: Optional[float] = None
    nominal_bracket_mean: Optional[float] = None
    nominal_delta_reference: Optional[float] = None
    #: GUM comparison centre and coverage factor behind ``gum_lower``/``gum_upper``.
    gum_center: Optional[float] = None
    gum_coverage_factor_k: Optional[float] = None
    #: Space each iteration returned *before* any post-loop transform.
    iteration_return_space: str = ""
    #: Space the stored values are actually in *after* the post-loop stage.
    #: Together with ``iteration_return_space`` and
    #: ``post_loop_transform_applied`` this proves the output was transformed
    #: exactly once: either the iteration returned reported space and no
    #: transform ran, or it returned basis space and exactly one ran.
    effective_result_space: str = ""
    #: The frozen legacy transform operands, recorded only when that transform
    #: actually ran. Both stay ``None`` for new Engine B reported-space runs.
    delta_reference_applied: Optional[float] = None
    absolute_scale_factor_applied: Optional[float] = None
    #: Full declared specification of every sampled contributor.
    contributor_specs: Tuple["EngineBContributorSpec", ...] = ()
    #: Digest of the analytical budget this run was made against. Paired with
    #: ``config_digest`` it lets a stored result be tested for freshness: the
    #: configuration digest catches a settings change, this one catches a
    #: data change (mask, cycle range, outlier filter) that leaves every
    #: setting untouched. Empty means *unknown*, never *fresh*.
    budget_digest: str = ""


# One-release import compatibility. These are aliases, not separate verdict
# APIs (disposition R2).
MCValidationResult = MCCrossCheckResult


def _mc_summary(values: np.ndarray) -> Tuple[float, float, float, float]:
    """Return mean, sd, q2.5, q97.5 for a Monte Carlo sample.

    The sample standard deviation is pinned to ``ddof=1`` and the central
    percentile endpoints pass ``method`` explicitly, so neither a library
    default change nor an edit elsewhere can silently move a reported value.
    """
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)),
        float(
            np.percentile(values, MC_LOWER_PERCENTILE, method=MC_PERCENTILE_METHOD)
        ),
        float(
            np.percentile(values, MC_UPPER_PERCENTILE, method=MC_PERCENTILE_METHOD)
        ),
    )


def _finite_positive(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out > 0.0 else None


def _resolve_delta_reference_for_mc(
    sample: Sample,
    ratio_name: str,
    *,
    gum_budget: Optional[UncertaintyBudget],
    certified_value: Optional[CertifiedValue],
) -> Optional[float]:
    budget_ref = _finite_positive(getattr(gum_budget, "delta_reference_value", None))
    if budget_ref is not None:
        return budget_ref

    delta_data = (getattr(sample, "delta_results", {}) or {}).get(ratio_name, {})
    if isinstance(delta_data, dict):
        runtime_ref = _finite_positive(delta_data.get("std_mean"))
        if runtime_ref is not None:
            return runtime_ref

    return None


def _normalize_mc_distribution(value: object) -> str:
    distribution = str(value or "normal").strip().lower()
    if distribution == "gaussian":
        distribution = "normal"
    if distribution in {"student_t", "scaled_t"}:
        distribution = MC_DISTRIBUTION_SCALED_STUDENT_T
    if distribution not in {
        "normal",
        "rectangular",
        MC_DISTRIBUTION_SCALED_STUDENT_T,
    }:
        raise ValueError(
            f"Unsupported Monte Carlo uncertainty distribution {value!r}."
        )
    return distribution


def _draw_standard_uncertainty(
    rng: np.random.Generator,
    sigma: float,
    distribution: str = "normal",
    *,
    degrees_of_freedom: float = float("inf"),
) -> float:
    """Draw a zero-centred perturbation with standard uncertainty *sigma*."""
    if sigma <= 0.0:
        return 0.0
    distribution = _normalize_mc_distribution(distribution)
    if distribution == "rectangular":
        half_width = sigma * np.sqrt(3.0)
        return float(rng.uniform(-half_width, half_width))
    dof = float(degrees_of_freedom)
    if distribution == MC_DISTRIBUTION_SCALED_STUDENT_T:
        if np.isinf(dof) and dof > 0.0:
            return float(rng.normal(0.0, sigma))
        if not np.isfinite(dof) or dof <= 0.0:
            raise ValueError(
                "A scaled Student-t draw requires positive degrees of freedom."
            )
        return float(rng.standard_t(dof) * sigma)
    if distribution == "normal" and np.isfinite(dof) and dof > 0.0:
        return float(rng.standard_t(dof) * sigma)
    return float(rng.normal(0.0, sigma))


def _relative_change_pct(estimate: float, final: float) -> float:
    """Return a stable relative-difference percentage."""
    estimate = float(estimate)
    final = float(final)
    if np.isclose(final, 0.0):
        return 0.0 if np.isclose(estimate, 0.0) else float("nan")
    return abs(estimate - final) / abs(final) * 100.0


def _build_convergence_diagnostics(
    results: np.ndarray,
) -> Tuple[
    Tuple[Tuple[int, float, float, float, float], ...],
    float,
    float,
    float,
    float,
    float,
]:
    """Build cumulative-checkpoint diagnostics for a finished MC run."""
    n_iter = int(len(results))
    if n_iter <= 0:
        return (), 0.0, 0.0, 0.0, 0.0, 0.0

    checkpoint_counts = sorted({
        max(50, int(round(n_iter * frac)))
        for frac in (0.10, 0.25, 0.50, 0.75)
        if max(50, int(round(n_iter * frac))) < n_iter
    })
    checkpoint_counts.append(n_iter)

    checkpoints = []
    for count in checkpoint_counts:
        subset = results[:count]
        mean_val, std_val, q025, q975 = _mc_summary(subset)
        checkpoints.append((count, mean_val, std_val, q025, q975))

    final_count, final_mean, final_std, final_q025, final_q975 = checkpoints[-1]
    final_half_width = (final_q975 - final_q025) / 2.0

    mean_pct = 0.0
    std_pct = 0.0
    half_width_pct = 0.0
    q025_pct = 0.0
    q975_pct = 0.0
    for _, mean_val, std_val, q025, q975 in checkpoints[:-1]:
        mean_pct = max(mean_pct, _relative_change_pct(mean_val, final_mean))
        std_pct = max(std_pct, _relative_change_pct(std_val, final_std))
        q025_pct = max(q025_pct, _relative_change_pct(q025, final_q025))
        q975_pct = max(q975_pct, _relative_change_pct(q975, final_q975))
        half_width = (q975 - q025) / 2.0
        half_width_pct = max(
            half_width_pct,
            _relative_change_pct(half_width, final_half_width),
        )

    return (
        tuple(checkpoints),
        mean_pct,
        std_pct,
        half_width_pct,
        q025_pct,
        q975_pct,
    )


def build_mc_debug_report(
    sample: Sample,
    ratio_name: str,
    *,
    all_samples: List[Sample],
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    gum_budget: UncertaintyBudget,
    certified_value: Optional[CertifiedValue] = None,
    processing_config: Optional[ProcessingConfig] = None,
    ratio_values: Optional[np.ndarray] = None,
    ratio_mean: Optional[float] = None,
    ratio_mask: Optional[np.ndarray] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    mc_result: Optional[MCValidationResult] = None,
    seed: Optional[int] = None,
) -> str:
    """Return a copyable text dump of the exact GUM and MC inputs in use."""
    resolved = uncertainty_config.resolve_engine(
        element_config.symbol if element_config else "",
        processing_config=processing_config,
    )

    if ratio_values is None:
        cd = get_best_ratio_data(sample, ratio_name)
        if cd is None:
            raise MCCrossCheckError(
                engine=resolved,
                reason_code="missing_chain_input",
                reason=f"No ratio data are available for {ratio_name} on {sample.name}.",
            )
        ratio_values = cd.valid_values

    if len(ratio_values) < 2:
        raise ValueError(f"Need >= 2 valid cycles, got {len(ratio_values)}")

    finite = ratio_values[np.isfinite(ratio_values)]
    if ratio_mean is None:
        ratio_mean = float(np.mean(finite)) if len(finite) > 0 else 0.0
    if ratio_mean == 0.0 or not np.isfinite(ratio_mean):
        raise MCCrossCheckError(
            engine=resolved,
            reason_code="invalid_sampled_parameter",
            reason="The ratio mean must be finite and non-zero.",
        )

    ratio_sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    ratio_se = ratio_sd / np.sqrt(len(finite)) if len(finite) > 1 else 0.0

    def _fmt(value: object) -> str:
        if isinstance(value, bool):
            return "True" if value else "False"
        if value is None:
            return "None"
        if isinstance(value, float):
            if np.isinf(value):
                return "inf"
            return f"{value:.10g}"
        if isinstance(value, np.ndarray):
            return np.array2string(
                value,
                precision=8,
                separator=", ",
                suppress_small=False,
                max_line_width=240,
            )
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(_fmt(v) for v in value) + "]"
        return str(value)

    lines = [
        "# MC debug report",
        f"sample.name = {sample.name}",
        f"sample.type = {sample.sample_type}",
        f"ratio_name = {ratio_name}",
        f"engine = {resolved}",
        f"output_mode = {uncertainty_config.output_mode}",
        f"mc_seed = {_fmt(seed)}",
        f"ratio_n = {len(finite)}",
        f"ratio_mask_n_total = {_fmt(int(len(ratio_mask))) if ratio_mask is not None else 'None'}",
        f"ratio_mask_n_included = {_fmt(int(np.sum(np.asarray(ratio_mask, dtype=bool)))) if ratio_mask is not None else 'None'}",
        f"ratio_mean_runtime = {_fmt(ratio_mean)}",
        f"ratio_sd_runtime = {_fmt(ratio_sd)}",
        f"ratio_se_runtime = {_fmt(ratio_se)}",
        f"budget.ratio_value = {_fmt(gum_budget.ratio_value)}",
        f"budget.u_combined_abs = {_fmt(gum_budget.u_combined_abs)}",
        f"budget.expanded_abs = {_fmt(gum_budget.expanded_abs)}",
        f"budget.effective_dof = {_fmt(gum_budget.effective_dof)}",
        f"budget.coverage_factor_k = {_fmt(gum_budget.coverage_factor_k)}",
        f"budget.dominant_contributor = {gum_budget.dominant_contributor or '-'}",
    ]

    for contributor in gum_budget.contributors:
        lines.append(
            "budget.contributor."
            f"{contributor.name} = active:{_fmt(contributor.is_active)} "
            f"abs:{_fmt(float(contributor.value_abs))} "
            f"permil:{_fmt(float(contributor.value_rel_permil))} "
            f"dof:{_fmt(float(contributor.degrees_of_freedom))} "
            f"pct:{_fmt(float(contributor.percentage_contribution))}"
        )

    if resolved == "internal_normalization":
        contributor_lookup = {c.name: c for c in gum_budget.contributors}
        reprod_result = getattr(gum_budget, "reprod_result", None)

        def _abs(name: str) -> float:
            contributor = contributor_lookup.get(name)
            if contributor is None:
                return 0.0
            return float(contributor.value_abs)

        def _rel_pct(abs_value: float) -> float:
            return (float(abs_value) / ratio_mean) * 100.0 if ratio_mean else 0.0

        term1_sd_rel_pct = _rel_pct(ratio_sd)
        term1_se_rel_pct = _rel_pct(ratio_se)
        term1_traceiso_rel_pct = _rel_pct(_abs("u_prec"))

        term2_sd_rel_pct = _rel_pct(_abs("u_std_repeatability"))
        term2_se_rel_pct = _rel_pct(_abs("u_std_repeatability_se"))
        term2_traceiso_rel_pct = term2_sd_rel_pct if term2_sd_rel_pct > 0 else term2_se_rel_pct

        term2_n = 0
        term2_mean = 0.0
        term2_method = "unavailable"
        if reprod_result is not None:
            try:
                included = np.asarray(reprod_result.std_included, dtype=bool)
                means = np.asarray(reprod_result.std_means, dtype=np.float64)
                if len(included) == len(means):
                    selected = means[included]
                    selected = selected[np.isfinite(selected)]
                    term2_n = int(len(selected))
                    if term2_n > 0:
                        term2_mean = float(np.mean(selected))
                term2_method = str(getattr(reprod_result, "method", term2_method))
            except (TypeError, ValueError, IndexError) as exc:
                _LOG.debug("Ignoring malformed reproducibility diagnostics: %s", exc)

        term3_ref_value = 0.0
        try:
            if certified_value is not None:
                term3_ref_value = float(certified_value.value)
            else:
                payload = element_config.certified_values.get(ratio_name)
                if payload is not None:
                    term3_ref_value = float(payload.value)
        except (TypeError, ValueError, OverflowError) as exc:
            _LOG.debug("Ignoring malformed certified reference value: %s", exc)
            term3_ref_value = 0.0
        # u_bias_ref stored |Δ_ref|/sqrt(3) under the automatic model that
        # is disabled, so this term is now always 0. The
        # reconstruction is kept behind the same flag as the model itself.
        term3_bias_rel_pct = (
            _rel_pct(_abs("u_bias_ref") * np.sqrt(3.0))
            if AUTOMATIC_REFERENCE_BIAS_ENABLED
            else 0.0
        )

        term4_family = "NIST987"
        term4_n = 0
        term4_mean = 0.0
        term4_reprod_sd_rel_pct = _rel_pct(_abs("u_reprod_dig"))

        term5_bias_qc_abs = _abs("u_bias_qc")
        if term5_bias_qc_abs > 0.0:
            # u_bias_qc is already the user-supplied standard uncertainty
            # transferred to the sample basis.
            term5_qc_bias_rel_pct = _rel_pct(term5_bias_qc_abs)
            term5_note = "provided"
        else:
            term5_qc_bias_rel_pct = None
            term5_note = (
                "requires a resolved QC material with certified 87Sr/86Sr in the same session"
            )

        uc_rel_pct_sd_basis_4term = float(
            np.sqrt(
                max(
                    term1_sd_rel_pct**2
                    + term2_sd_rel_pct**2
                    + term3_bias_rel_pct**2
                    + term4_reprod_sd_rel_pct**2,
                    0.0,
                )
            )
        )
        u_rel_pct_k2_sd_basis_4term = 2.0 * uc_rel_pct_sd_basis_4term
        uc_rel_pct_sem_basis_4term = float(
            np.sqrt(
                max(
                    term1_se_rel_pct**2
                    + term2_se_rel_pct**2
                    + term3_bias_rel_pct**2
                    + term4_reprod_sd_rel_pct**2,
                    0.0,
                )
            )
        )
        u_rel_pct_k2_sem_basis_4term = 2.0 * uc_rel_pct_sem_basis_4term

        available_terms = 0
        for value in (
            term1_sd_rel_pct,
            term2_sd_rel_pct,
            term3_bias_rel_pct,
            term4_reprod_sd_rel_pct,
            term5_qc_bias_rel_pct if term5_qc_bias_rel_pct is not None else 0.0,
        ):
            if float(value) > 0.0:
                available_terms += 1

        lines.extend(
            [
                "# BAM Eq. 5 comparison (Sr, diagnostic only)",
                f"bam_eq5.term1.sample_repeatability_sd_rel_pct = {_fmt(term1_sd_rel_pct)}",
                f"bam_eq5.term1.sample_repeatability_se_rel_pct = {_fmt(term1_se_rel_pct)}",
                f"bam_eq5.term1.traceiso_u_prec_rel_pct = {_fmt(term1_traceiso_rel_pct)}",
                f"bam_eq5.term2.nist_n = {_fmt(term2_n)}",
                f"bam_eq5.term2.nist_mean = {_fmt(term2_mean)}",
                f"bam_eq5.term2.nist_repeatability_sd_rel_pct = {_fmt(term2_sd_rel_pct)}",
                f"bam_eq5.term2.nist_repeatability_se_rel_pct = {_fmt(term2_se_rel_pct)}",
                f"bam_eq5.term2.traceiso_u_std_repeatability_rel_pct = {_fmt(term2_traceiso_rel_pct)}",
                f"bam_eq5.term2.traceiso_u_std_repeatability_method = {term2_method}",
                f"bam_eq5.term3.nist_reference_value = {_fmt(term3_ref_value)}",
                f"bam_eq5.term3.nist_bias_rel_pct = {_fmt(term3_bias_rel_pct)}",
                f"bam_eq5.term4.family = {term4_family}",
                f"bam_eq5.term4.family_n = {_fmt(term4_n)}",
                f"bam_eq5.term4.family_mean = {_fmt(term4_mean)}",
                f"bam_eq5.term4.reproducibility_sd_rel_pct = {_fmt(term4_reprod_sd_rel_pct)}",
                (
                    f"bam_eq5.term5.qc_bias_rel_pct = {_fmt(term5_qc_bias_rel_pct)}"
                    if term5_qc_bias_rel_pct is not None
                    else "bam_eq5.term5.qc_bias_rel_pct = unavailable"
                ),
                f"bam_eq5.term5.note = {term5_note}",
                f"bam_eq5.available_terms = {available_terms}/5",
                f"bam_eq5.uc_rel_pct_sd_basis_4term = {_fmt(uc_rel_pct_sd_basis_4term)}",
                f"bam_eq5.U_rel_pct_k2_sd_basis_4term = {_fmt(u_rel_pct_k2_sd_basis_4term)}",
                f"bam_eq5.uc_rel_pct_sem_basis_4term = {_fmt(uc_rel_pct_sem_basis_4term)}",
                f"bam_eq5.U_rel_pct_k2_sem_basis_4term = {_fmt(u_rel_pct_k2_sem_basis_4term)}",
            ]
        )

    if resolved == "internal_normalization":
        if processing_config is None:
            raise ValueError(
                "Monte Carlo debug for Engine A requires ProcessingConfig."
            )
        params = _extract_internal_perturbation_params(
            sample,
            ratio_name,
            ratio_values,
            ratio_mean,
            all_samples,
            element_config,
            uncertainty_config,
            processing_config=processing_config,
            certified_value=certified_value,
            gum_budget=gum_budget,
            ratio_mask=ratio_mask,
            cycle_ranges=cycle_ranges,
        )
        lines.extend(
            [
                f"processing.blank_mode = {processing_config.blank_mode}",
                f"processing.apply_interference_correction = {_fmt(processing_config.apply_interference_correction)}",
                f"processing.apply_mass_bias_correction = {_fmt(processing_config.apply_mass_bias_correction)}",
                f"processing.normalization_ratio_override = {_fmt(processing_config.normalization_ratio_override)}",
                f"processing.normalization_value_override = {_fmt(processing_config.normalization_value_override)}",
                f"mc_param.normalization_value = {_fmt(params.normalization_value)}",
                f"mc_param.normalization_value_sd = {_fmt(params.normalization_value_sd)}",
                f"mc_param.sr_anchor_factor = {_fmt(params.sr_anchor_factor)}",
                f"mc_param.post_chain_scale = {_fmt(params.post_chain_scale)}",
                f"mc_param.post_chain_scale_permil_offset = "
                f"{_fmt((params.post_chain_scale - 1.0) * 1000.0)}",
                f"mc_param.apply_interference = {_fmt(params.apply_interference)}",
                f"mc_param.apply_iif = {_fmt(params.apply_iif)}",
                f"mc_param.norm_ratio_chain_enabled = {_fmt(params.mode_b_chain_enabled)}",
                f"mc_param.interference_chain_enabled = {_fmt(params.interference_chain_enabled)}",
                f"mc_param.u_prec = {_fmt(params.u_prec)}",
                f"mc_param.u_norm_ratio_fallback = {_fmt(params.u_norm_ratio_mode_b_fallback)}",
                f"mc_param.u_std_repeatability = {_fmt(params.u_std_repeatability)}",
                f"mc_param.u_kappa_drift = {_fmt(params.u_kappa_drift)}",
                f"mc_param.u_bias_ref = {_fmt(params.u_bias_ref)}",
                f"mc_param.u_bias_qc = {_fmt(params.u_bias_qc)}",
                f"mc_param.u_reprod_dig = {_fmt(params.u_reprod_dig)}",
                f"mc_param.u_crm = {_fmt(params.u_crm)}",
                f"mc_param.u_ref_value = {_fmt(params.u_ref_value)}",
                f"mc_param.u_interf_fallback = {_fmt(params.u_interf_fallback)}",
                f"mc_param.u_blank_fallback = {_fmt(params.u_blank_fallback)}",
                f"mc_param.rb87_rb85_nominal = {_fmt(params.reference_inputs.rb87_rb85)}",
                f"mc_param.rb87_rb85_sd = {_fmt(params.reference_inputs.rb87_rb85 * params.reference_inputs.u_rel_rb if params.reference_inputs.u_rel_rb is not None else None)}",
                f"mc_param.kr86_kr83_nominal = {_fmt(params.reference_inputs.kr86_kr83)}",
                f"mc_param.kr86_kr83_sd = {_fmt(params.reference_inputs.kr86_kr83 * params.reference_inputs.u_rel_kr86)}",
                f"mc_param.blank_blocks = {_fmt(len(params.blank_blocks))}",
            ]
        )
        for isotope, values in sorted(params.base_corrected_intensities.items()):
            finite_vals = values[np.isfinite(values)]
            mean_val = float(np.mean(finite_vals)) if len(finite_vals) > 0 else np.nan
            lines.append(
                f"mc_param.base_intensity.{isotope} = mean:{_fmt(mean_val)} n:{_fmt(len(values))}"
            )
        for index, block in enumerate(params.blank_blocks, start=1):
            lines.extend(
                [
                    f"mc_param.blank_block_{index}.isotopes = {_fmt(block.isotopes)}",
                    f"mc_param.blank_block_{index}.mean_vector = {_fmt(block.mean_vector)}",
                    f"mc_param.blank_block_{index}.covariance_matrix = {_fmt(block.covariance_matrix)}",
                ]
            )
    else:
        params = _extract_perturbation_params(
            sample,
            ratio_name,
            ratio_values,
            ratio_mean,
            all_samples,
            element_config,
            uncertainty_config,
            certified_value,
            gum_budget=gum_budget,
            cycle_ranges=cycle_ranges,
            processing_config=processing_config,
        )
        lines.extend(
            [
                f"mc_param.u_prec = {_fmt(params.u_prec)}",
                f"mc_param.u_std = {_fmt(params.u_std)}",
                f"mc_param.u_std_repeatability = {_fmt(params.u_std_repeatability)}",
                f"mc_param.u_k1_sample_decomposition = {_fmt(params.u_k1_sample_decomposition)}",
                f"mc_param.u_k2_matrix_separation = {_fmt(params.u_k2_matrix_separation)}",
                f"mc_param.u_k3_procedural_blank = {_fmt(params.u_k3_procedural_blank)}",
                f"mc_param.u_k4_bracketing_standard_heterogeneity = {_fmt(params.u_k4_bracketing_standard_heterogeneity)}",
                f"mc_param.u_k5_instrumental_drift = {_fmt(params.u_k5_instrumental_drift)}",
                f"mc_param.u_k6_matrix_effects = {_fmt(params.u_k6_matrix_effects)}",
                f"mc_param.u_k7_residual_interferences = {_fmt(params.u_k7_residual_interferences)}",
                f"mc_param.u_crm = {_fmt(params.u_crm)}",
                f"mc_param.u_blank_fallback = {_fmt(params.u_blank_fallback)}",
                f"mc_param.blank_num_mean = {_fmt(params.blank_num_mean)}",
                f"mc_param.blank_den_mean = {_fmt(params.blank_den_mean)}",
                f"mc_param.blank_num_sd = {_fmt(params.blank_num_sd)}",
                f"mc_param.blank_den_sd = {_fmt(params.blank_den_sd)}",
                f"mc_param.blank_correlation = {_fmt(params.blank_correlation)}",
                f"mc_param.num_corrected_mean = {_fmt(params.num_corrected_mean)}",
                f"mc_param.den_corrected_mean = {_fmt(params.den_corrected_mean)}",
                f"mc_param.blank_blocks = {_fmt(len(params.blank_blocks))}",
            ]
        )
        for index, block in enumerate(params.blank_blocks, start=1):
            lines.extend(
                [
                    f"mc_param.blank_block_{index}.mean_vector = {_fmt(block.mean_vector)}",
                    f"mc_param.blank_block_{index}.covariance_matrix = {_fmt(block.covariance_matrix)}",
                ]
            )

    if mc_result is not None:
        mc_half_width = (mc_result.mc_upper_95 - mc_result.mc_lower_95) / 2.0
        lines.extend(
            [
                f"mc_result.n_iter = {mc_result.n_iter}",
                f"mc_result.mc_lower_95 = {_fmt(mc_result.mc_lower_95)}",
                f"mc_result.mc_upper_95 = {_fmt(mc_result.mc_upper_95)}",
                f"mc_result.mc_half_width = {_fmt(mc_half_width)}",
                f"mc_result.mc_mean = {_fmt(mc_result.mc_mean)}",
                f"mc_result.mc_std = {_fmt(mc_result.mc_std)}",
                f"mc_result.validation_mode = {_fmt(mc_result.validation_mode)}",
                f"mc_result.gum_lower = {_fmt(mc_result.gum_lower)}",
                f"mc_result.gum_upper = {_fmt(mc_result.gum_upper)}",
                f"mc_result.gum_u_expanded = {_fmt(mc_result.gum_u_expanded)}",
                f"mc_result.agreement_pct = {_fmt(mc_result.agreement_pct)}",
                f"mc_result.convergence_mean_pct = {_fmt(mc_result.convergence_mean_pct)}",
                f"mc_result.convergence_std_pct = {_fmt(mc_result.convergence_std_pct)}",
                f"mc_result.convergence_half_width_pct = {_fmt(mc_result.convergence_half_width_pct)}",
                f"mc_result.convergence_q025_pct = {_fmt(mc_result.convergence_q025_pct)}",
                f"mc_result.convergence_q975_pct = {_fmt(mc_result.convergence_q975_pct)}",
                f"mc_result.convergence_checkpoints = {_fmt(mc_result.convergence_checkpoints)}",
                f"mc_result.center_deviation_pct = {_fmt(mc_result.center_deviation_pct)}",
                f"mc_result.passed = {_fmt(mc_result.passed)}",
            ]
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bounded runtime lifecycle logging
#
# Diagnostics only: the scientific result and its provenance fields stay
# authoritative. Nothing is printed, nothing is logged per draw, and no input
# array, draw array, filesystem path or unrestricted source metadata is ever
# emitted.
# ---------------------------------------------------------------------------

ENGINE_B_MC_EVENT_STARTED = "engine_b_mc.started"
ENGINE_B_MC_EVENT_COMPLETED = "engine_b_mc.completed"
ENGINE_B_MC_EVENT_FAILED = "engine_b_mc.failed"
ENGINE_B_MC_EVENT_INVALID_INPUT = "engine_b_mc.invalid_input"
ENGINE_B_MC_EVENT_LEGACY_USED = "engine_b_mc.legacy_used"
ENGINE_B_MC_EVENT_EXPORTED = "engine_b_mc.exported"

ENGINE_B_MC_EVENT_CODES: frozenset = frozenset(
    {
        ENGINE_B_MC_EVENT_STARTED,
        ENGINE_B_MC_EVENT_COMPLETED,
        ENGINE_B_MC_EVENT_FAILED,
        ENGINE_B_MC_EVENT_INVALID_INPUT,
        ENGINE_B_MC_EVENT_LEGACY_USED,
        ENGINE_B_MC_EVENT_EXPORTED,
    }
)

#: Stable diagnostic code shared by the user-facing warning and the log record
#: when a drawn Type A input has a degrees of freedom at which the Student-t
#: variance is undefined.
REASON_LOW_DOF_UNDEFINED_VARIANCE = "low_dof_undefined_variance"

#: Hard ceiling on the characters of any single logged field value.
_LOG_VALUE_MAX_CHARS = 120

#: Replacement written in place of any filesystem path material.
_LOG_OMITTED_PATH = "<omitted:path>"

#: Filesystem path material within one whitespace-delimited token.
#:
#: Sample, ratio and custom-contributor identifiers all originate in imported
#: source metadata, and a failure ``reason`` interpolates them mid-sentence, so
#: refusing only a value that *begins* as a path leaves the embedded case open.
#: A slash between isotope tokens is deliberately **not** path material:
#: identifiers such as ``7Li/6Li`` are required log fields and must survive
#: intact. Path prefixes are recognized at the start of a token or after a
#: small punctuation delimiter so interpolated ``path=/...`` forms are also
#: covered. A backslash is path material anywhere because it is not legitimate
#: in an Engine B identifier.
_LOG_PATH_TOKEN = re.compile(
    r"""(?x)
    (?:
        [A-Za-z]:[\\/]         # Windows drive path anywhere in the token
      | \\                       # Windows/UNC path material anywhere
      | file:/+                  # file URI
      | (?:^|[=(:,;'"\[])(?:    # path prefix at token start or listed delimiter
            /                    # POSIX absolute, including one component
          | \.{1,2}/             # ./ or ../ relative path
          | ~/                   # home-relative path
        )
    )
    """
)


def _sanitize_log_value(value: object) -> str:
    """Render one lifecycle-log field as a short, safe scalar string.

    Arrays and containers are refused outright rather than truncated, so a
    draw array or a raw input vector cannot reach the log through a new call
    site.
    """
    if isinstance(value, np.ndarray) or isinstance(value, (list, tuple, set, dict)):
        return "<omitted:collection>"
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.10g}"
    text = str(value)
    # Identifiers can originate in imported sample metadata. Refuse absolute
    # filesystem paths and flatten control characters before they reach the
    # line-oriented logger; truncation alone would still leak a user path or
    # permit a newline to forge a second lifecycle-looking record.
    is_windows_absolute = (
        len(text) >= 3
        and text[1] == ":"
        and text[0].isalpha()
        and text[2] in {"\\", "/"}
    )
    if is_windows_absolute or text.startswith(("/", "\\\\")):
        return _LOG_OMITTED_PATH
    text = "".join(character if character.isprintable() else " " for character in text)
    text = " ".join(text.split())
    # A value that only *contains* path material still leaks it. This matters
    # most for a failure ``reason``, which interpolates an imported sample or
    # ratio identifier into the middle of an English sentence, where the
    # whole-value test above can never reach it. Redact per token so the
    # surrounding diagnostic text, and slash-bearing ratio names, survive.
    text = " ".join(
        _LOG_OMITTED_PATH if _LOG_PATH_TOKEN.search(token) else token
        for token in text.split(" ")
    )
    if len(text) > _LOG_VALUE_MAX_CHARS:
        text = text[:_LOG_VALUE_MAX_CHARS] + "..."
    return text


def _emit_engine_b_event(level: int, event_code: str, **fields: object) -> None:
    """Emit one bounded Engine B lifecycle record through the module logger."""
    if event_code not in ENGINE_B_MC_EVENT_CODES:
        raise ValueError(f"Unknown Engine B lifecycle event code {event_code!r}.")
    payload = " ".join(
        f"{key}={_sanitize_log_value(value)}" for key, value in sorted(fields.items())
    )
    _LOG.log(level, "%s %s", event_code, payload)


def log_engine_b_mc_exported(
    execution_id: str,
    *,
    result_schema_version: str,
    export_format: str,
    record_count: int = 1,
    level: int = logging.INFO,
) -> None:
    """Emit one bounded ``engine_b_mc.exported`` lifecycle record.

    Called from the export/integration boundary, once per format per export —
    never once per record and never per draw. The payload is deliberately
    limited to the execution ID, the result-schema version, the format name
    and how many records were written: no file path, no directory, no user or
    machine name, and no summary or draw array. ``export_format`` is a short
    format token such as ``json``/``excel``/``csv``/``hdf5``; a caller must
    not pass a filename through it.
    """
    _emit_engine_b_event(
        level,
        ENGINE_B_MC_EVENT_EXPORTED,
        execution_id=execution_id,
        result_schema_version=result_schema_version,
        export_format=export_format,
        record_count=int(record_count),
    )


def _canonical_config_digest_value(value: object) -> object:
    """Return a stable, order-independent representation for config hashing."""
    if isinstance(value, (float, np.floating)):
        return ("float", float(value).hex())
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return ("int", int(value))
    if isinstance(value, (str, bool)) or value is None:
        return value
    if is_dataclass(value):
        return tuple(
            (
                spec.name,
                _canonical_config_digest_value(
                    _dataclass_field_value(value, spec)
                ),
            )
            for spec in dataclass_fields(value)
        )
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (
                    str(key),
                    _canonical_config_digest_value(item),
                )
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_config_digest_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(
            sorted(
                (_canonical_config_digest_value(item) for item in value),
                key=repr,
            )
        )
    return (type(value).__qualname__, str(value))


def _dataclass_field_value(instance: object, spec: object) -> object:
    """Read a dataclass field, restoring its declared default when absent.

    Older pickled session configurations can lack fields introduced by newer
    releases. Their dataclass type still declares the default, so hashing that
    default matches the behavior the runtime would use and avoids turning a
    provenance check into a load-time failure.
    """
    name = getattr(spec, "name")
    if hasattr(instance, name):
        return getattr(instance, name)
    default = getattr(spec, "default", MISSING)
    if default is not MISSING:
        return default
    default_factory = getattr(spec, "default_factory", MISSING)
    if default_factory is not MISSING:
        return default_factory()
    return None


def engine_b_config_digest(uncertainty_config: UncertaintyConfig) -> str:
    """Return the complete canonical identity of the active MC configuration.

    Hash every declared field, including draw count and contributor PDFs.  A
    short hand-maintained allow-list previously omitted distribution fields,
    allowing a normal-to-rectangular change to alter the MC result while a
    durable record and every export still reported it as current.
    """
    canonical = tuple(
        (
            spec.name,
            _canonical_config_digest_value(
                _dataclass_field_value(uncertainty_config, spec)
            ),
        )
        for spec in dataclass_fields(uncertainty_config)
    )
    return hashlib.sha256(repr(canonical).encode("utf-8")).hexdigest()[:16]


def engine_b_budget_digest(gum_budget: object) -> str:
    """Return a canonical digest of the analytical budget a run was made against.

    This is the *input-side* freshness identity, and it is deliberately
    distinct from :func:`engine_b_config_digest`. The configuration digest
    changes only when a setting changes; this digest changes whenever the
    measured data behind the budget changes — a cycle mask, a cycle range or
    an outlier-filter edit all move the resolved ratio, the cycle count and
    the contributor magnitudes without touching a single setting.

    Persisting it lets a stored Monte Carlo result be compared against the
    budget that is current *now*, so a result that no longer corresponds to
    the data beside it can be disclosed instead of silently presented and
    exported as if it were fresh. Floats are hashed through ``float.hex()``
    so the digest is exact rather than format-dependent.

    Returns an empty string when no budget is available, which callers must
    treat as *freshness unknown* — never as *fresh*.
    """
    if gum_budget is None:
        return ""

    hasher = hashlib.sha256()

    def add(value: object) -> None:
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"\x00")

    def add_float(value: object) -> None:
        if value is None:
            add("none")
            return
        try:
            add(float(value).hex())
        except (TypeError, ValueError):
            add("nan")

    for name in (
        "ratio_value",
        "u_combined_abs",
        "u_combined_rel_permil",
        "expanded_abs",
        "effective_dof",
        "coverage_factor_k",
        "basis_ratio_value",
        "delta_reference_value",
        "delta_scale_factor",
        "certified_reference_value",
        "absolute_scale_factor",
    ):
        add_float(getattr(gum_budget, name, None))

    add(int(getattr(gum_budget, "n_cycles", 0) or 0))
    add(str(getattr(gum_budget, "engine", "") or ""))
    add(str(getattr(gum_budget, "output_mode", "") or ""))
    add(str(getattr(gum_budget, "budget_scope", "") or ""))

    # Contributor magnitudes move with the data even when the contributor set
    # does not, so both the identity and the value of each term are hashed.
    for contributor in getattr(gum_budget, "contributors", ()) or ():
        add(str(getattr(contributor, "name", "")))
        add_float(getattr(contributor, "value_abs", None))
        add_float(getattr(contributor, "degrees_of_freedom", None))
        add(str(getattr(contributor, "type_ab", "")))
        add(bool(getattr(contributor, "is_active", True)))
        add(str(getattr(contributor, "state", "")))
        # A029: a custom term's PDF is supplied outside UncertaintyConfig, so
        # the configuration digest cannot see it. Without it here, changing a
        # user-defined contributor from normal to rectangular moved the
        # sampled interval while every stored record still read "current".
        add(str(getattr(contributor, "distribution", "") or ""))

    return hasher.hexdigest()[:16]


def _engine_b_execution_id(
    *,
    sample_name: str,
    sample_run_number: int = 0,
    ratio_name: str,
    engine: str,
    semantics_version: str,
    output_mode: str,
    bracket_mode: str,
    n_iter: int,
    seed: Optional[int],
    config_digest: str = "",
    input_digest: str = "",
) -> str:
    """Derive one stable execution ID for a locked Engine B run.

    Derived rather than random so the same locked inputs, settings and seed
    correlate the persisted result with its runtime records on a re-run. The
    sample run number is part of observation identity: repeated observations
    may legitimately share a sample name and every numerical input. The
    bracket mode and configuration digest are included so two runs that differ
    in what they actually computed cannot share an identifier either.
    """
    digest = hashlib.sha256(
        "|".join(
            (
                str(sample_name),
                str(int(sample_run_number)),
                str(ratio_name),
                str(engine),
                str(semantics_version),
                str(output_mode),
                str(bracket_mode),
                str(config_digest),
                str(input_digest),
                str(int(n_iter)),
                "none" if seed is None else str(int(seed)),
            )
        ).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def _resolved_replay_evidence(*, custom_base, draw_params, semantics_version,
        element_config, processing_config, uncertainty_config, custom_contributor_library,
        all_samples, sample, cycle_ranges, ratio_mask, ratio_values, ratio_mean,
        certified_value, profile_defaults, gum_budget, rng_initial_state):
    """One payload builder for execution and read-only current-input resolution."""
    from config.scientific_identity import scientific_configuration
    custom_sigmas = getattr(custom_base, "custom_contributor_sigmas", None) or getattr(custom_base, "custom_sigmas", {})
    custom_distributions = getattr(custom_base, "custom_contributor_distributions", None) or getattr(custom_base, "custom_distributions", {})
    custom_dofs = getattr(custom_base, "custom_contributor_estimation_dofs", None) or getattr(custom_base, "custom_estimation_dofs", {})
    replay_evidence = {
        "schema": "traceiso.mc_replay_inputs.v2",
        "method": semantics_version,
        "resolved_parameters": draw_params,
        "scientific_configuration": scientific_configuration(element_config),
        "processing": processing_config,
        "uncertainty": uncertainty_config,
        "custom_definitions": custom_contributor_library or {},
        "custom_resolved_inputs": {
            name: {"scale": sigma, "units": "basis_ratio", "pdf": custom_distributions.get(name, "normal"),
                   "estimation_dof": custom_dofs.get(name, float("inf")),
                   "pdf_shape_dof": None, "placement": "additive_output", "dependence_group": name}
            for name, sigma in custom_sigmas.items()
        },
        # Canonical Sample snapshots exclude stored uncertainty/MC results,
        # retaining classification, every layer, masks and blank membership.
        "observations": list(all_samples or [sample]),
        "cycle_ranges": cycle_ranges or {},
        "ratio_mask": ratio_mask,
        "ratio_values": ratio_values,
        "ratio_mean": ratio_mean,
        "certified_value": certified_value,
        "profile_defaults": profile_defaults,
        "analytical_contributor_specifications": gum_budget.contributors,
        "rng_initial_state": rng_initial_state,
        "units": {"intensities": "V", "ratios": "1", "relative_uncertainty": "permil"},
        "custom_pdf_policy": "declared_pdf; DoF estimates uncertainty, not PDF shape",
    }
    return replay_evidence


def monte_carlo_cross_check(
    sample: Sample,
    ratio_name: str,
    *,
    all_samples: List[Sample],
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    processing_config: Optional[ProcessingConfig] = None,
    gum_budget: UncertaintyBudget,
    certified_value: Optional[CertifiedValue] = None,
    ratio_values: Optional[np.ndarray] = None,
    ratio_mean: Optional[float] = None,
    ratio_mask: Optional[np.ndarray] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    n_iter: int = DEFAULT_MC_ITERATIONS_STANDARD,
    validation_mode: str = "custom",
    tolerance_pct: float = 5.0,
    progress_callback: Optional[Callable[[float], None]] = None,
    seed: Optional[int] = None,
    custom_contributor_library: Optional[Dict[str, List[CustomUncertaintyContributor]]] = None,
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
) -> MCCrossCheckResult:
    """Run a fixed-size Monte Carlo distributional cross-check.

    For new Engine B semantics every draw re-evaluates the production
    SSB/delta measurement model and the iteration returns the final
    reported quantity, so the frozen post-loop basis-space transform is
    skipped. ``tolerance_pct`` is accepted for signature compatibility and
    has no effect on a new Engine B summary; the result carries descriptive
    diagnostics and no pass/fail verdict.
    """
    resolved = uncertainty_config.resolve_engine(
        element_config.symbol if element_config else "",
        processing_config=processing_config,
    )

    from domain.sr_standard_calibration import sr_calibration_message
    sr_refusal = sr_calibration_message(sample, ratio_name)
    if sr_refusal:
        raise MCCrossCheckError(
            engine=resolved, reason_code="sr_calibration_unavailable", reason=sr_refusal,
        )

    calibrated_params: Optional["_PbCalibratedMCParams"] = None
    calibration = governing_calibration_record(sample, ratio_name)
    if calibration is not None:
        # A Pb-standard-calibrated result is answered only by the extended
        # Engine C replay of its absolute ratio. Unavailable results and delta
        # are refused before any draw; the Tl-only Engine C replay never answers.
        requested_mode = str(
            getattr(gum_budget, "output_mode", None) or uncertainty_config.output_mode or ""
        ).strip().lower()
        if calibration.status != _LAYER_APPLIED:
            raise MCCrossCheckError(
                engine=resolved, reason_code="missing_chain_input",
                reason=(
                    f"Pb-standard calibration of {ratio_name} is {calibration.status} for "
                    f"{sample.name} ({calibration.reason_code}); there is no final value to cross-check."
                ),
            )
        if requested_mode == "delta" or str(uncertainty_config.output_mode or "").strip().lower() == "delta":
            raise MCCrossCheckError(
                engine=resolved, reason_code=DELTA_UNCERTAINTY_REASON_CODE,
                reason=DELTA_UNCERTAINTY_REASON,
            )
        calibrated_params = _resolve_pb_calibrated_mc_params(
            sample,
            ratio_name,
            resolved=resolved,
            all_samples=all_samples,
            element_config=element_config,
            uncertainty_config=uncertainty_config,
            processing_config=processing_config,
            gum_budget=gum_budget,
            ratio_mask=ratio_mask,
            cycle_ranges=cycle_ranges,
            custom_contributor_library=custom_contributor_library,
        )

    if resolved == "internal_normalization" and ratio_name != "87Sr/86Sr":
        output_mode = str(
            getattr(gum_budget, "output_mode", None) or uncertainty_config.output_mode
        )
        return MCCrossCheckResult(
            mc_lower_95=None, mc_upper_95=None, mc_mean=None, mc_std=None,
            gum_lower=float("nan"), gum_upper=float("nan"),
            gum_u_expanded=float(getattr(gum_budget, "expanded_abs", 0.0) or 0.0),
            n_iter=0, status="not_supported",
            reason_code="engine_a_ratio_not_supported",
            scope_note="Engine A Monte Carlo supports 87Sr/86Sr only.",
            engine=resolved, output_mode=output_mode,
            sample_name=str(sample.name or ""),
            sample_run_number=int(sample.run_number or 0), ratio_name=ratio_name,
            requested_draws=int(n_iter), completed_draws=0, warnings=(
                "No draws were run: this Engine A ratio is not supported by Monte Carlo.",
            ),
        )

    if gum_budget.budget_scope in {"unavailable", "insufficient_data"}:
        raise MCCrossCheckError(engine=resolved, reason_code="missing_chain_input",
            reason=gum_budget.scope_note or "Required analytical input is unavailable; no draws were run.")
    if any(c.is_active and (not np.isfinite(c.value_abs) or not np.isfinite(c.value_rel_permil)) for c in gum_budget.contributors):
        raise MCCrossCheckError(engine=resolved, reason_code="invalid_sampled_parameter", reason="Nonfinite active uncertainty input.")

    if ratio_values is None:
        cd = get_best_ratio_data(sample, ratio_name)
        if cd is None:
            raise MCCrossCheckError(
                engine=resolved,
                reason_code="missing_chain_input",
                reason=f"No ratio data are available for {ratio_name} on {sample.name}.",
            )
        ratio_values = cd.valid_values

    if len(ratio_values) < 2:
        raise ValueError(f"Need >= 2 valid cycles, got {len(ratio_values)}")

    if ratio_mean is None:
        finite = ratio_values[np.isfinite(ratio_values)]
        ratio_mean = float(np.mean(finite)) if len(finite) > 0 else 0.0

    if ratio_mean == 0.0 or not np.isfinite(ratio_mean):
        raise MCCrossCheckError(
            engine=resolved,
            reason_code="invalid_sampled_parameter",
            reason="The ratio mean must be finite and non-zero.",
        )

    if is_russell_law_normalization_engine(resolved) and processing_config is None:
        raise MCCrossCheckError(
            engine=resolved,
            reason_code="missing_chain_input",
            reason=(
                "Internal-normalization Monte Carlo requires ProcessingConfig "
                "to replay the correction chain."
            ),
        )

    rng = np.random.default_rng(seed)

    _scope_note = ""

    # Engine C (Pb-Tl) uses distinct chain-replay machinery from Sr.
    is_pb_tl = resolved == "pb_tl_external_normalization"

    if calibrated_params is not None:
        params = calibrated_params
    elif is_pb_tl:
        params = _extract_pb_tl_perturbation_params(
            sample,
            ratio_name,
            ratio_mean,
            all_samples,
            element_config,
            uncertainty_config,
            processing_config,
            gum_budget=gum_budget,
            ratio_mask=ratio_mask,
            cycle_ranges=cycle_ranges,
            custom_contributor_library=custom_contributor_library,
            profile_defaults=profile_defaults,
        )
        if params.hg_perturbation_expected and not params.hg_interf_chain_enabled:
            _scope_note = (
                "204Hg/202Hg assigned correction-ratio uncertainty not "
                "perturbed — no assigned uncertainty is available"
            )
    elif resolved == "internal_normalization":
        params = _extract_internal_perturbation_params(
            sample,
            ratio_name,
            ratio_values,
            ratio_mean,
            all_samples,
            element_config,
            uncertainty_config,
            processing_config=processing_config,
            certified_value=certified_value,
            gum_budget=gum_budget,
            ratio_mask=ratio_mask,
            cycle_ranges=cycle_ranges,
            custom_contributor_library=custom_contributor_library,
            profile_defaults=profile_defaults,
        )
    else:
        params = _extract_perturbation_params(
            sample, ratio_name, ratio_values, ratio_mean,
            all_samples, element_config, uncertainty_config, certified_value,
            gum_budget=gum_budget,
            cycle_ranges=cycle_ranges,
            custom_contributor_library=custom_contributor_library,
            profile_defaults=profile_defaults,
            processing_config=processing_config,
        )

    # ------------------------------------------------------------------
    # Engine B: choose between the fixed-draw chain replay and the retained
    # legacy additive path, then lock the iteration result space.
    # ------------------------------------------------------------------
    is_engine_b = not is_pb_tl and resolved != "internal_normalization"
    output_mode = str(
        getattr(gum_budget, "output_mode", None) or uncertainty_config.output_mode
    )
    engine_b_params: Optional[EngineBFixedDrawParams] = None
    semantics_version = ""
    result_space = RESULT_SPACE_BASIS_RATIO
    bracket_mode = ""
    execution_id = ""
    run_warnings: List[str] = ["Input-PDF moment diagnostics do not prove existence of nonlinear output moments."]
    legacy_min_type_a_dof: Optional[float] = None
    legacy_moment_status = MOMENT_STATUS_DEFINED
    # ``n_iter`` is rebound to the completed count after the loop, so the
    # requested count is captured here for persistence (plan 6.2).
    requested_draws = int(n_iter)

    # A026/D4: A/C precision is the estimated mean of a finite normal sample.
    # The analytical SE is therefore the scale of Student-t(n-1), not the SD
    # of a Gaussian substitute.  Precision is added after each deterministic
    # chain replay, so its t moment boundary also governs this additive output.
    if calibrated_params is not None:
        semantics_version = PB_CALIBRATED_MC_SEMANTICS
        result_space = RESULT_SPACE_ABSOLUTE_RATIO
        bracket_mode = calibrated_params.applied_mode
        legacy_min_type_a_dof = calibrated_params.min_type_a_dof
        legacy_moment_status = _moment_status_from_dof(legacy_min_type_a_dof)
        run_warnings.extend(calibrated_params.warnings)
    elif is_pb_tl:
        semantics_version = "engine_c.chain_replay.v3.joint_tl_blank"
        if params.u_prec > 0.0 and np.isfinite(params.u_prec_dof):
            legacy_min_type_a_dof = float(params.u_prec_dof)
            legacy_moment_status = _moment_status_from_dof(legacy_min_type_a_dof)
    elif resolved == "internal_normalization":
        # A025: the semantics name the Sr method the chain replay evaluates, so a
        # record from one method is never read as current for another.
        semantics_version = engine_a_semantics_for_method(params.reference_inputs.chain_method)
        if params.u_prec > 0.0 and np.isfinite(params.u_prec_dof):
            legacy_min_type_a_dof = float(params.u_prec_dof)
            legacy_moment_status = _moment_status_from_dof(legacy_min_type_a_dof)

    # Engine A records carry the complete configuration identity as well; their
    # freshness additionally requires the active Sr method (sr_chain_identity).
    is_engine_a = (not is_pb_tl) and resolved == "internal_normalization"
    config_digest = (
        engine_b_config_digest(uncertainty_config)
    )
    input_digest = calibrated_params.input_digest if calibrated_params is not None else ""

    def _make_execution_id(
        mode: str,
        semantics: str,
        resolved_input_digest: str = "",
    ) -> str:
        return _engine_b_execution_id(
            sample_name=sample.name,
            sample_run_number=int(getattr(sample, "run_number", 0) or 0),
            ratio_name=ratio_name,
            engine=resolved,
            semantics_version=semantics,
            output_mode=output_mode,
            bracket_mode=mode,
            n_iter=n_iter,
            seed=seed,
            config_digest=config_digest,
            input_digest=resolved_input_digest,
        )

    if calibrated_params is not None:
        execution_id = _make_execution_id(bracket_mode, semantics_version, input_digest)

    if is_engine_b:
        execution_id = _make_execution_id("unresolved", ENGINE_B_FIXED_DRAW_SEMANTICS)
        try:
            engine_b_params = _resolve_engine_b_fixed_draw_params(
                params,
                sample=sample,
                ratio_name=ratio_name,
                all_samples=all_samples,
                element_config=element_config,
                uncertainty_config=uncertainty_config,
                processing_config=processing_config,
                certified_value=certified_value,
                gum_budget=gum_budget,
                ratio_mask=ratio_mask,
                cycle_ranges=cycle_ranges,
            )
        except MCCrossCheckError as exc:
            _emit_engine_b_event(
                logging.ERROR,
                ENGINE_B_MC_EVENT_FAILED,
                execution_id=execution_id,
                completed_draws=0,
                iteration="none",
                reason_code=exc.reason_code,
                reason=exc.reason,
            )
            raise

        if engine_b_params is None:
            # Structured chain inputs do not exist for this case, so the fixed
            # replay is unavailable and the explicitly versioned legacy path
            # runs instead. A malformed structured bracket does not reach here:
            # it raises above.
            semantics_version = ENGINE_B_LEGACY_SEMANTICS
            result_space = RESULT_SPACE_BASIS_RATIO
            execution_id = _make_execution_id("legacy", ENGINE_B_LEGACY_SEMANTICS)
            run_warnings.append(
                "Engine B fixed-draw chain replay unavailable for this case; "
                "the legacy additive basis-space path was used."
            )
            legacy_type_a_inputs = (
                ("u_prec", params.u_prec, params.u_prec_dof),
                ("u_std", params.u_std, params.u_std_dof),
                (
                    "u_std_repeatability",
                    params.u_std_repeatability,
                    params.u_std_repeatability_dof,
                ),
            )
            drawn_dofs = [
                float(dof)
                for _name, sigma, dof in legacy_type_a_inputs
                if sigma > 0.0 and np.isfinite(float(dof))
            ]
            legacy_min_type_a_dof = min(drawn_dofs) if drawn_dofs else None
            legacy_moment_status = _moment_status_from_dof(
                legacy_min_type_a_dof
            )
            for input_name, sigma, dof in legacy_type_a_inputs:
                if sigma > 0.0 and np.isfinite(float(dof)) and float(dof) <= 2.0:
                    run_warnings.append(
                        f"{input_name} is drawn at {float(dof):g} degrees of "
                        "freedom, where the Student-t variance is undefined."
                    )
            _emit_engine_b_event(
                logging.WARNING,
                ENGINE_B_MC_EVENT_LEGACY_USED,
                execution_id=execution_id,
                legacy_semantics_version=ENGINE_B_LEGACY_SEMANTICS,
            )
        else:
            semantics_version = engine_b_params.semantics_version
            result_space = engine_b_params.result_space
            bracket_mode = engine_b_params.bracket_mode
            input_digest = engine_b_input_digest(engine_b_params)
            execution_id = _make_execution_id(
                bracket_mode,
                semantics_version,
                input_digest,
            )
            for input_name, placement, dof in engine_b_params.low_dof_disclosures:
                run_warnings.append(
                    f"{input_name} is drawn at {dof:g} degrees of freedom, where "
                    "the Student-t variance is undefined."
                )
                _emit_engine_b_event(
                    logging.WARNING,
                    ENGINE_B_MC_EVENT_INVALID_INPUT,
                    execution_id=execution_id,
                    input=input_name,
                    placement=placement,
                    reason_code=REASON_LOW_DOF_UNDEFINED_VARIANCE,
                    degrees_of_freedom=dof,
                )
            if engine_b_params.hg_propagation_method:
                run_warnings.append(
                    ("Joint ordinary SSB blank terms are drawn at first-order sensitivities; no nonlinear blank-chain replay. "
                     if engine_b_params.hg_propagation_method.startswith("ordinary_ssb")
                     else "Hg-corrected Pb terms are drawn at their first-order sensitivities; not a draw-level replay of the Hg correction chain. ")
                    + f"({engine_b_params.hg_propagation_method})"
                )
                # A term the GUM budget could not evaluate is not drawn either;
                # say which, so the cross-check does not read as covering it.
                for contributor in getattr(gum_budget, "contributors", ()) or ():
                    if (
                        contributor.name in (U_BLANK, U_INTERF, U_HG_TL_REFERENCE, U_HG_MASS_BIAS_MODEL)
                        and not contributor.is_active
                        and contributor.state not in {"BY_SAMPLE_DESIGN", "BY_GLOBAL_DESIGN", "NOT_APPLICABLE"}
                    ):
                        run_warnings.append(
                            f"{contributor.name} is not drawn: {contributor.inactive_reason}"
                        )

        _emit_engine_b_event(
            logging.INFO,
            ENGINE_B_MC_EVENT_STARTED,
            execution_id=execution_id,
            sample=sample.name,
            ratio=ratio_name,
            output_mode=output_mode,
            bracket_mode=bracket_mode or "legacy",
            result_space=result_space,
            requested_draws=n_iter,
            seed=seed,
            semantics_version=semantics_version,
        )

    draw_params: object = engine_b_params if engine_b_params is not None else params
    from config.scientific_identity import canonical_json, digest, scientific_configuration
    replay_evidence = _resolved_replay_evidence(
        custom_base=calibrated_params if calibrated_params is not None else params,
        draw_params=draw_params, semantics_version=semantics_version,
        element_config=element_config, processing_config=processing_config,
        uncertainty_config=uncertainty_config, custom_contributor_library=custom_contributor_library,
        all_samples=all_samples, sample=sample, cycle_ranges=cycle_ranges,
        ratio_mask=ratio_mask, ratio_values=ratio_values, ratio_mean=ratio_mean,
        certified_value=certified_value, profile_defaults=profile_defaults,
        gum_budget=gum_budget, rng_initial_state=rng.bit_generator.state,
    )
    if not (is_engine_a or (is_pb_tl and calibrated_params is None)):
        # Existing fixed-B/calibrated records already carry their own input
        # evidence. Add the formerly lost custom estimation DoF explicitly.
        replay_evidence = {key: replay_evidence[key] for key in (
            "schema", "method", "custom_resolved_inputs", "custom_pdf_policy")}
    replay_snapshot_json = canonical_json(replay_evidence)
    if is_engine_a or (is_pb_tl and calibrated_params is None):
        input_digest = digest(replay_evidence)
        execution_id = _make_execution_id("chain", semantics_version, input_digest)
    started_at = time.perf_counter()

    try:
        _preflight_cross_check_params(draw_params, engine=resolved)
        results = np.empty(n_iter, dtype=np.float64)

        # Report progress in chunks for efficiency
        chunk_size = max(1, n_iter // 100)

        if calibrated_params is not None:
            results = _run_pb_calibrated_draws(
                calibrated_params, rng, n_iter, progress_callback, engine=resolved,
            )

        for i in range(n_iter if calibrated_params is None else 0):
            try:
                if is_pb_tl:
                    value = _mc_iteration_pb_tl(params, rng)
                elif resolved == "internal_normalization":
                    value = _mc_iteration_internal(params, rng)
                elif engine_b_params is not None:
                    value = _mc_iteration_engine_b_fixed_draw(engine_b_params, rng)
                else:
                    value = _mc_iteration_ssb(params, rng)
            except MCCrossCheckError as exc:
                if exc.iteration is None:
                    exc.iteration = i
                    exc.args = (f"{exc.engine} at iteration {i}: {exc.reason}",)
                raise
            if not np.isfinite(value):
                raise MCCrossCheckError(
                    engine=resolved,
                    iteration=i,
                    reason_code="nonfinite_model_output",
                    reason="The model produced a non-finite result.",
                )
            results[i] = value

            if progress_callback is not None and (i + 1) % chunk_size == 0:
                progress_callback((i + 1) / n_iter)

        # Final progress callback
        if progress_callback is not None:
            progress_callback(1.0)

        # --------------------------------------------------------------
        # Post-loop transform. Exactly one owner: an iteration that already
        # returned reported space owns its own transform, so this block is a
        # strict no-op for those results and cannot apply a second time.
        # --------------------------------------------------------------
        gum_center = ratio_mean
        post_loop_transform_applied = False
        # Persistence-only record of which space the iteration itself returned
        # and, when the frozen legacy transform runs, the exact operands it
        # used. Neither is read back by any calculation.
        iteration_return_space = result_space
        effective_result_space = result_space
        delta_reference_applied: Optional[float] = None
        absolute_scale_factor_applied: Optional[float] = None
        if result_space in REPORTED_RESULT_SPACES:
            gum_center = float(draw_params.nominal_reported_value)  # type: ignore[attr-defined]
        elif not is_russell_law_normalization_engine(resolved) and output_mode == "delta":
            delta_ref = _resolve_delta_reference_for_mc(
                sample,
                ratio_name,
                gum_budget=gum_budget,
                certified_value=certified_value,
            )
            if delta_ref is None:
                delta_ref = ratio_mean
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                results = (results / delta_ref - 1.0) * 1000.0
            gum_center = (ratio_mean / delta_ref - 1.0) * 1000.0
            post_loop_transform_applied = True
            delta_reference_applied = float(delta_ref)
            effective_result_space = RESULT_SPACE_DELTA_PERMIL
        elif not is_russell_law_normalization_engine(resolved) and output_mode == "absolute_ratio":
            report_ratio = _finite_positive(getattr(gum_budget, "ratio_value", None))
            basis_ratio = _finite_positive(getattr(gum_budget, "basis_ratio_value", None))
            absolute_scale = _finite_positive(getattr(gum_budget, "absolute_scale_factor", None))
            if absolute_scale is None and report_ratio is not None and basis_ratio is not None:
                absolute_scale = report_ratio / basis_ratio
            if absolute_scale is not None and not np.isclose(absolute_scale, 1.0):
                with np.errstate(over="ignore", invalid="ignore"):
                    results = results * absolute_scale
                gum_center = (
                    report_ratio
                    if report_ratio is not None
                    else ratio_mean * absolute_scale
                )
                post_loop_transform_applied = True
                absolute_scale_factor_applied = float(absolute_scale)
                effective_result_space = RESULT_SPACE_ABSOLUTE_RATIO
            elif report_ratio is not None:
                gum_center = report_ratio

        if not np.all(np.isfinite(results)):
            raise MCCrossCheckError(
                engine=resolved,
                reason_code="nonfinite_model_output",
                reason="The transformed model results contain a non-finite value.",
            )
    except MCCrossCheckError as exc:
        if is_engine_b:
            _emit_engine_b_event(
                logging.ERROR,
                ENGINE_B_MC_EVENT_FAILED,
                execution_id=execution_id,
                completed_draws=0 if exc.iteration is None else int(exc.iteration),
                iteration="none" if exc.iteration is None else int(exc.iteration),
                reason_code=exc.reason_code,
                reason=exc.reason,
            )
        raise

    n_iter = int(results.size)
    if n_iter < 2:
        mc_mean_raw = float(results[0]) if n_iter == 1 else float("nan")
        mc_std_raw = float("nan")
        mc_lower = mc_upper = mc_mean_raw
    else:
        mc_mean_raw, mc_std_raw, mc_lower, mc_upper = _mc_summary(results)
    (
        convergence_checkpoints,
        convergence_mean_pct,
        convergence_std_pct,
        convergence_half_width_pct,
        convergence_q025_pct,
        convergence_q975_pct,
    ) = _build_convergence_diagnostics(results)

    # A Student-t has a mean only above one degree of freedom and a variance
    # only above two, so a summary moment that is mathematically undefined is
    # disclosed and suppressed. The central percentile interval stays the
    # primary reported summary at every degrees of freedom.
    moment_status = (
        engine_b_params.moment_status
        if engine_b_params is not None
        else legacy_moment_status
    )
    mc_mean: Optional[float] = mc_mean_raw
    mc_std: Optional[float] = mc_std_raw
    mc_std_diagnostic: Optional[float] = None
    if moment_status == MOMENT_STATUS_VARIANCE_UNDEFINED:
        mc_std_diagnostic = mc_std_raw
        mc_std = None
    elif moment_status == MOMENT_STATUS_MEAN_AND_VARIANCE_UNDEFINED:
        mc_mean = None
        mc_std = None

    gum_u_expanded = gum_budget.expanded_abs
    gum_lower = gum_center - gum_u_expanded
    gum_upper = gum_center + gum_u_expanded

    mc_half_width = (mc_upper - mc_lower) / 2.0
    gum_half_width = gum_u_expanded

    if gum_half_width > 0:
        agreement_pct = abs(mc_half_width - gum_half_width) / gum_half_width * 100.0
    else:
        agreement_pct = np.nan if not np.isfinite(mc_half_width) else (0.0 if mc_half_width == 0.0 else 100.0)

    # Numerical centre diagnostic: how far the MC distribution centre sits
    # from the GUM point estimate, as a percentage of the expanded
    # uncertainty. Descriptive only; it is not a validation criterion.
    if mc_mean is None:
        center_deviation_pct = float("nan")
    elif gum_u_expanded > 0.0:
        center_deviation_pct = abs(mc_mean - gum_center) / gum_u_expanded * 100.0
    else:
        center_deviation_pct = 0.0

    gum_u_c = float(getattr(gum_budget, "u_combined_abs", 0.0) or 0.0)
    if not np.isfinite(gum_u_c) or gum_u_c < 0.0:
        gum_u_c = 0.0
    absolute_center_difference = (
        abs(mc_mean - gum_center) if mc_mean is not None else None
    )
    center_difference_u_c = (
        absolute_center_difference / gum_u_c
        if absolute_center_difference is not None and gum_u_c > 0.0
        else None
    )
    std_difference_pct = (
        abs(mc_std - gum_u_c) / gum_u_c * 100.0
        if mc_std is not None and gum_u_c > 0.0
        else None
    )

    passed: Optional[bool] = None
    failure_reason = ""
    # Deprecated compatibility fields stay unset on every current public route.

    if is_engine_b:
        _emit_engine_b_event(
            logging.INFO,
            ENGINE_B_MC_EVENT_COMPLETED,
            execution_id=execution_id,
            completed_draws=n_iter,
            elapsed_seconds=time.perf_counter() - started_at,
            mc_mean=mc_mean,
            mc_std=mc_std,
            mc_lower=mc_lower,
            mc_upper=mc_upper,
            coverage_probability=MC_COVERAGE_PROBABILITY,
            interval_convention=MC_INTERVAL_CONVENTION,
            percentile_method=MC_PERCENTILE_METHOD,
            moment_status=moment_status,
            result_space=result_space,
            semantics_version=semantics_version,
            warning_count=len(run_warnings),
        )

    return MCCrossCheckResult(
        mc_lower_95=mc_lower,
        mc_upper_95=mc_upper,
        mc_mean=mc_mean,
        mc_std=mc_std,
        gum_lower=gum_lower,
        gum_upper=gum_upper,
        gum_u_expanded=gum_u_expanded,
        agreement_pct=agreement_pct,
        n_iter=n_iter,
        passed=passed,
        validation_mode=validation_mode,
        convergence_mean_pct=convergence_mean_pct,
        convergence_std_pct=convergence_std_pct,
        convergence_half_width_pct=convergence_half_width_pct,
        convergence_q025_pct=convergence_q025_pct,
        convergence_q975_pct=convergence_q975_pct,
        convergence_checkpoints=convergence_checkpoints,
        center_deviation_pct=center_deviation_pct,
        scope_note=_scope_note,
        failure_reason=failure_reason,
        mc_samples=results.copy(),
        n_dropped=0,
        gum_u_c=gum_u_c,
        absolute_center_difference=absolute_center_difference,
        center_difference_u_c=center_difference_u_c,
        std_difference_pct=std_difference_pct,
        semantics_version=semantics_version,
        result_space=result_space,
        execution_id=execution_id,
        bracket_mode=bracket_mode,
        output_mode=output_mode if (is_engine_b or calibrated_params is not None) else "",
        post_loop_transform_applied=post_loop_transform_applied,
        moment_status=moment_status,
        min_type_a_dof=(
            engine_b_params.min_type_a_dof
            if engine_b_params is not None
            else legacy_min_type_a_dof
        ),
        mc_std_diagnostic=mc_std_diagnostic,
        contributor_placements=(
            draw_params.contributor_placements()
            if isinstance(draw_params, (EngineBFixedDrawParams, _PbCalibratedMCParams))
            else tuple((s.name, s.placement) for s in _ordinary_contributor_specs(params, gum_budget))
        ),
        warnings=tuple(run_warnings),
        config_digest=config_digest,
        input_digest=input_digest,
        replay_snapshot_json=replay_snapshot_json,
        budget_digest=engine_b_budget_digest(gum_budget),
        seed=seed,
        bit_generator=type(rng.bit_generator).__name__,
        numpy_version=np.__version__,
        sample_name=sample.name,
        sample_run_number=int(getattr(sample, "run_number", 0) or 0),
        ratio_name=ratio_name,
        engine=resolved,
        requested_draws=requested_draws,
        completed_draws=n_iter,
        prev_std_label=(
            engine_b_params.prev_std_label if engine_b_params is not None else ""
        ),
        next_std_label=(
            engine_b_params.next_std_label if engine_b_params is not None else ""
        ),
        apply_ssb_kernel=(
            engine_b_params.apply_ssb_kernel if engine_b_params is not None else None
        ),
        delta_from_ssb=(
            engine_b_params.delta_from_ssb if engine_b_params is not None else None
        ),
        blank_uncertainty_input=(
            engine_b_params.blank_uncertainty_input
            if engine_b_params is not None
            else (calibrated_params.blank_uncertainty_input if calibrated_params is not None else "")
        ),
        blank_placement=(
            engine_b_params.blank_placement
            if engine_b_params is not None
            else (calibrated_params.blank_placement if calibrated_params is not None else "")
        ),
        nominal_reported_value=(
            float(draw_params.nominal_reported_value)  # type: ignore[attr-defined]
            if isinstance(draw_params, (EngineBFixedDrawParams, _PbCalibratedMCParams))
            else None
        ),
        nominal_bracket_mean=(
            float(engine_b_params.nominal_bracket_mean)
            if engine_b_params is not None
            else None
        ),
        nominal_delta_reference=(
            float(engine_b_params.nominal_delta_reference)
            if engine_b_params is not None
            else None
        ),
        gum_center=float(gum_center),
        gum_coverage_factor_k=float(
            getattr(gum_budget, "coverage_factor_k", 0.0) or 0.0
        ),
        iteration_return_space=iteration_return_space,
        effective_result_space=effective_result_space,
        delta_reference_applied=delta_reference_applied,
        absolute_scale_factor_applied=absolute_scale_factor_applied,
        contributor_specs=(
            draw_params.contributor_specs()
            if isinstance(draw_params, (EngineBFixedDrawParams, _PbCalibratedMCParams))
            else _ordinary_contributor_specs(params, gum_budget)
        ),
    )


# One-release call compatibility. Both names invoke the same implementation
# and the alias is the same object, so ``__kwdefaults__`` introspection and
# existing imports keep working (disposition F2/F3).
monte_carlo_validate = monte_carlo_cross_check


# Perturbation parameter extraction

def _ordinary_contributor_specs(params, budget=None):
    """Describe the draws made by the ordinary A/C and legacy B kernels."""
    specs = []
    rows = {c.name: c for c in getattr(budget, "contributors", ())}
    for name in ("u_prec", "u_std", "u_std_repeatability", "u_crm", "u_ref_value",
                 "u_bias_qc", "u_reprod_dig", "u_interf", "u_interf_fallback",
                 "u_blank_fallback", "u_norm_ratio_mode_b_fallback", "u_kappa_drift",
                 "u_k1_sample_decomposition", "u_k2_matrix_separation", "u_k3_procedural_blank",
                 "u_k4_bracketing_standard_heterogeneity", "u_k5_instrumental_drift",
                 "u_k6_matrix_effects", "u_k7_residual_interferences"):
        if float(getattr(params, name, 0.) or 0.) <= 0:
            continue
        dof = float(getattr(params, name + "_dof", float("inf")))
        distribution = "scaled_student_t" if np.isfinite(dof) else "normal"
        if name in {"u_kappa_drift", "u_k5_instrumental_drift"}:
            distribution = params.kappa_drift_distribution
        elif name.startswith("u_k"):
            distribution = (getattr(params, "kappa_distributions", None) or {}).get(name, "normal")
        specs.append(EngineBContributorSpec(name, "output_level", distribution, dof,
                     "A" if name in {"u_prec", "u_std", "u_std_repeatability"} else "B", f"resolved_parameters.{name}"))
    for name, field_name in (("u_norm_ref", "tl_norm_std"), ("u_interf", "hg_std"),
                             ("u_norm_ratio", "normalization_value_sd")):
        if float(getattr(params, field_name, 0.) or 0.) > 0:
            specs.append(EngineBContributorSpec(name, "chain_input", "normal", float("inf"), "B", f"resolved_parameters.{field_name}"))
    if getattr(params, "interference_chain_enabled", False):
        specs.append(EngineBContributorSpec("u_interf", "chain_input", "normal", float("inf"), "B", "resolved_parameters.reference_inputs"))
    if getattr(params, "blank_blocks", ()):
        specs.append(EngineBContributorSpec("u_blank", "chain_input", "multivariate_normal", float("inf"), "A", "resolved_parameters.blank_blocks"))
    for name in (getattr(params, "custom_contributor_sigmas", None) or {}):
        specs.append(EngineBContributorSpec(name, "output_level",
            (params.custom_contributor_distributions or {}).get(name, "normal"), float("inf"),
            getattr(rows.get(name), "type_ab", "B"), f"custom_resolved_inputs.{name}; estimation DoF recorded separately"))
    return tuple(specs)

@dataclass
class _SSBPerturbationParams:
    """All scalar parameters needed for one MC iteration (Engine B).

    These are extracted once from the sample/budget and reused for every
    iteration, avoiding repeated data lookups.
    """

    # Core ratio
    ratio_mean: float

    # Additive contributors on the final measurand
    u_prec: float
    u_std: float
    u_std_repeatability: float
    u_k1_sample_decomposition: float
    u_k2_matrix_separation: float
    u_k3_procedural_blank: float
    u_k4_bracketing_standard_heterogeneity: float
    u_k5_instrumental_drift: float
    u_k6_matrix_effects: float
    u_k7_residual_interferences: float
    kappa_drift_distribution: str
    u_crm: float
    u_prec_dof: float
    u_std_dof: float
    u_std_repeatability_dof: float

    # Blank (numerator + denominator SDs for perturbation)
    blank_num_mean: float      # mean blank intensity at numerator mass
    blank_den_mean: float      # mean blank intensity at denominator mass
    blank_num_sd: float        # SD of blank at numerator mass
    blank_den_sd: float        # SD of blank at denominator mass
    blank_correlation: float   # applied r for the blank covariance term
    num_corrected_mean: float  # blank-corrected intensity at numerator mass
    den_corrected_mean: float  # blank-corrected intensity at denominator mass
    u_blank_fallback: float    # fallback additive sigma when detailed blank data are unavailable
    blank_blocks: Tuple["_SSBBlankBlock", ...] = ()
    # Custom contributor perturbation sigmas (name -> absolute sigma for normal draw)
    custom_contributor_sigmas: Dict[str, float] = None  # type: ignore[assignment]
    custom_contributor_distributions: Dict[str, str] = None  # type: ignore[assignment]
    custom_contributor_estimation_dofs: Dict[str, float] = None  # type: ignore[assignment]
    kappa_distributions: Dict[str, str] = None  # type: ignore[assignment]


@dataclass
class _SSBBlankBlock:
    """One blank-file perturbation block for Engine B."""

    mean_vector: np.ndarray
    covariance_matrix: np.ndarray


@dataclass
class _InternalBlankBlock:
    """One blank-file perturbation block for Engine A."""

    isotopes: Tuple[str, ...]
    mean_vector: np.ndarray
    covariance_matrix: np.ndarray


@dataclass
class _InternalPerturbationParams:
    """All inputs needed for one Engine A Monte Carlo iteration."""

    ratio_mean: float
    ratio_mask: Optional[np.ndarray]
    base_corrected_intensities: Dict[str, np.ndarray]
    normalization_value: float
    normalization_value_sd: float
    sr_anchor_factor: float
    apply_interference: bool
    apply_iif: bool
    enabled_interferents: Set[str]
    mode_b_chain_enabled: bool
    interference_chain_enabled: bool

    u_prec: float
    u_prec_dof: float
    u_norm_ratio_mode_b_fallback: float
    u_std_repeatability: float
    u_kappa_drift: float
    kappa_drift_distribution: str
    u_bias_ref: float
    u_bias_qc: float
    u_reprod_dig: float
    u_crm: float
    u_ref_value: float
    u_interf_fallback: float
    u_blank_fallback: float

    reference_inputs: SrInterferenceReferenceInputs

    blank_blocks: Tuple[_InternalBlankBlock, ...]

    # Single scale mapping the raw chain-replay output (IIF space) into the
    # reported ratio space.  Absorbs the deterministic post-IIF scalar
    # transforms the pipeline applies — session anchoring AND drift correction —
    # so the MC distribution is centred on the GUM value.  1.0 when neither is
    # active (the plain Sr case), where the chain already reproduces the
    # reported ratio.
    post_chain_scale: float = 1.0
    # Custom contributor perturbation sigmas (name -> absolute sigma for normal draw)
    custom_contributor_sigmas: Dict[str, float] = None  # type: ignore[assignment]
    custom_contributor_distributions: Dict[str, str] = None  # type: ignore[assignment]
    custom_contributor_estimation_dofs: Dict[str, float] = None  # type: ignore[assignment]


@dataclass
class _PbTlPerturbationParams:
    """All inputs needed for one Engine C (Pb-Tl) Monte Carlo iteration.

    The deterministic Pb-Tl chain (``_run_pb_tl_correction_chain``) is replayed
    per iteration with perturbed *inputs* for the contributors that enter the
    chain naturally (blank base intensities, Tl normalization value).  The
    remaining contributors are independent random/systematic shifts on the
    output (u_prec, u_std_repeatability, u_interf, u_crm, u_kappa_drift) — this
    matches both the GUM RSS combination they contribute to and the Engine B
    convention for non-chain terms.
    """

    ratio_mean: float
    ratio_name: str
    ratio_mask: Optional[np.ndarray]
    post_chain_scale: float
    base_intensities: Dict[str, np.ndarray]
    apply_interference: bool
    apply_iif: bool

    # Tl normalisation reference value (chain input) and its 1-sigma draw width.
    tl_norm_value: float
    tl_norm_std: float

    # Additive output-level sigmas (absolute units).
    u_prec: float
    u_prec_dof: float
    u_std_repeatability: float
    u_interf: float
    u_crm: float
    u_kappa_drift: float
    kappa_drift_distribution: str

    # Per-blank-block isotope ordering + covariance (drawn from on each iter).
    blank_blocks: Tuple[object, ...]
    # Reference inputs object (PbTlReferenceInputs, kept untyped to avoid
    # importing pb_tl at module level).
    reference_inputs: object
    # Hg interference reference perturbation (chain replay, Engine C).
    hg_std: float = 0.0
    hg_interf_chain_enabled: bool = False
    # True when a Hg perturbation was expected (u_interf active, apply_hg, 204Pb
    # ratio) but could not be built — used to scope the "data unavailable" note
    # so it does not fire merely because the user disabled u_interf.
    hg_perturbation_expected: bool = False
    # Custom contributor perturbation sigmas (name -> absolute sigma for draw).
    custom_contributor_sigmas: Dict[str, float] = None  # type: ignore[assignment]
    custom_contributor_distributions: Dict[str, str] = None  # type: ignore[assignment]
    custom_contributor_estimation_dofs: Dict[str, float] = None  # type: ignore[assignment]


def _resolve_sr_reference_inputs(
    processing_config: ProcessingConfig,
    element_config: ElementConfig,
    sample: Optional[Sample] = None,
) -> SrInterferenceReferenceInputs:
    """Resolve managed Sr/Rb/Kr reference data and the Sr method for Engine A MC.

    The frozen parameters carry the method ``sample`` was processed with, so every
    draw replays that method (see ``sr_chain_identity``).
    """
    norm_num, norm_den, m_norm_num, m_norm_den = _resolve_active_sr_normalization_pair(
        processing_config,
        element_config,
    )
    return SrInterferenceReferenceInputs(
        rb87_rb85=require_natural_ratio("Rb", "87Rb/85Rb")[0],
        kr84_kr83=require_natural_ratio("Kr", "84Kr/83Kr")[0],
        kr86_kr83=require_natural_ratio("Kr", "86Kr/83Kr")[0],
        u_rel_rb=get_natural_ratio_relative_uncertainty("Rb", "87Rb/85Rb"),
        u_rel_kr84=require_natural_ratio_relative_uncertainty("Kr", "84Kr/83Kr"),
        u_rel_kr86=require_natural_ratio_relative_uncertainty("Kr", "86Kr/83Kr"),
        normalization_numerator=norm_num,
        normalization_denominator=norm_den,
        m_norm_num=m_norm_num,
        m_norm_den=m_norm_den,
        m86_sr=require_isotope_mass("86Sr"),
        m87_sr=require_isotope_mass("87Sr"),
        m88_sr=require_isotope_mass("88Sr"),
        m83_kr=require_isotope_mass("83Kr"),
        m84_kr=require_isotope_mass("84Kr"),
        m86_kr=require_isotope_mass("86Kr"),
        m85_rb=require_isotope_mass("85Rb"),
        m87_rb=require_isotope_mass("87Rb"),
        chain_method=replay_sr_chain_method(element_config, sample),
    )


def _extract_custom_contributor_params(
    sample: Sample,
    element_config: ElementConfig,
    ratio_mean: float,
    custom_contributor_library: Optional[Dict[str, List[CustomUncertaintyContributor]]],
) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, float]]:
    """Resolve declared PDFs; estimation DoF does not change custom PDF shape."""
    custom_rows = build_custom_contributor_rows(
        sample=sample,
        element_symbol=element_config.symbol,
        ratio_mean=ratio_mean,
        custom_contributor_library=custom_contributor_library or {},
    )
    sigmas: Dict[str, float] = {
        row.name: float(row.value_rel_permil) / 1000.0 * ratio_mean
        for row in custom_rows
        if row.value_rel_permil > 0.0
    }
    definition_lookup = {
        definition.name: definition
        for definition in (custom_contributor_library or {}).get(element_config.symbol, [])
    }
    distributions: Dict[str, str] = {
        row.name: _normalize_mc_distribution(
            getattr(definition_lookup.get(row.name), "distribution", "normal")
        )
        for row in custom_rows
        if row.name in sigmas
    }
    estimation_dofs = {row.name: float(row.degrees_of_freedom) for row in custom_rows if row.name in sigmas}
    return sigmas, distributions, estimation_dofs


def _extract_perturbation_params(
    sample: Sample,
    ratio_name: str,
    ratio_values: np.ndarray,
    ratio_mean: float,
    all_samples: List[Sample],
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    certified_value: Optional[CertifiedValue],
    gum_budget: Optional[UncertaintyBudget] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    custom_contributor_library: Optional[Dict[str, List[CustomUncertaintyContributor]]] = None,
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
    processing_config: Optional[ProcessingConfig] = None,
) -> _SSBPerturbationParams:
    """Extract all scalar parameters needed for MC perturbation."""

    def _active_contributor_value(name: str) -> float:
        if gum_budget is None:
            return 0.0
        contributor = gum_budget._find_contributor(name)
        if contributor is None or not contributor.is_active:
            return 0.0
        return float(contributor.value_abs)

    def _active_contributor_dof(name: str) -> float:
        if gum_budget is None:
            return float("inf")
        contributor = gum_budget._find_contributor(name)
        if contributor is None or not contributor.is_active:
            return float("inf")
        try:
            dof = float(contributor.degrees_of_freedom)
        except (TypeError, ValueError):
            return float("inf")
        return dof if dof >= 1.0 else 1.0

    _custom_names = {
        d.name
        for defs in (custom_contributor_library or {}).values()
        for d in defs
    }

    def _enabled(name: str) -> bool:
        return is_contributor_active(
            name=name,
            uncertainty_config=uncertainty_config,
            sample=sample,
            element_symbol=element_config.symbol,
            custom_contributor_names=_custom_names,
            profile_defaults=profile_defaults,
        )

    u_prec_abs = 0.0
    u_prec_dof = float("inf")
    if _enabled("u_prec"):
        if gum_budget is not None:
            u_prec_abs = _active_contributor_value("u_prec")
            u_prec_dof = _active_contributor_dof("u_prec")
        if u_prec_abs <= 0.0:
            u_prec_abs = _select_u_prec_from_values(
                ratio_values,
                uncertainty_config,
            )
            u_prec_dof = float(max(len(ratio_values) - 1, 1))

    u_std = 0.0
    u_std_dof = float("inf")
    if _enabled("u_std"):
        if gum_budget is not None:
            u_std = _active_contributor_value("u_std")
            u_std_dof = _active_contributor_dof("u_std")
        if u_std <= 0:
            u_std, _, u_std_dof = _compute_bracketing_standard_uncertainty(
                sample,
                ratio_name,
                all_samples,
                ratio_mean,
            )

    u_k1_sample_decomposition = (
        _active_contributor_value("u_k1_sample_decomposition")
        if _enabled("u_k1_sample_decomposition") else 0.0
    )

    u_k2_matrix_separation = (
        _active_contributor_value("u_k2_matrix_separation")
        if _enabled("u_k2_matrix_separation") else 0.0
    )
    u_k3_procedural_blank = (
        _active_contributor_value("u_k3_procedural_blank")
        if _enabled("u_k3_procedural_blank") else 0.0
    )
    u_k4_bracketing_standard_heterogeneity = (
        _active_contributor_value("u_k4_bracketing_standard_heterogeneity")
        if _enabled("u_k4_bracketing_standard_heterogeneity") else 0.0
    )
    u_k5_instrumental_drift = (
        _active_contributor_value("u_k5_instrumental_drift")
        if _enabled("u_k5_instrumental_drift") else 0.0
    )
    u_k6_matrix_effects = (
        _active_contributor_value("u_k6_matrix_effects")
        if _enabled("u_k6_matrix_effects") else 0.0
    )
    u_k7_residual_interferences = (
        _active_contributor_value("u_k7_residual_interferences")
        if _enabled("u_k7_residual_interferences") else 0.0
    )
    u_crm = _active_contributor_value("u_crm") if _enabled("u_crm") else 0.0

    u_std_repeatability = 0.0
    u_std_repeatability_dof = float("inf")
    if _enabled("u_std_repeatability"):
        if gum_budget is not None and gum_budget.reprod_result is not None:
            u_std_repeatability = float(gum_budget.reprod_result.u_std_repeatability_abs)
            u_std_repeatability_dof = float(
                gum_budget.reprod_result.degrees_of_freedom
            )
        else:
            reprod_result = compute_reprod(
                all_samples=all_samples,
                ratio_name=ratio_name,
                uncertainty_config=uncertainty_config,
                element_config=element_config,
            )
            u_std_repeatability = (
                (float(reprod_result.u_std_repeatability_rel_permil) / 1000.0) * ratio_mean
                if ratio_mean and float(reprod_result.u_std_repeatability_rel_permil) > 0.0
                else 0.0
            )
            u_std_repeatability_dof = float(reprod_result.degrees_of_freedom)
        if gum_budget is not None:
            _val = _active_contributor_value("u_std_repeatability")
            if _val > 0:
                u_std_repeatability = _val
            u_std_repeatability_dof = _active_contributor_dof(
                "u_std_repeatability"
            )

    blank_num_mean = 0.0
    blank_den_mean = 0.0
    blank_num_sd = 0.0
    blank_den_sd = 0.0
    blank_correlation = 0.0
    num_corrected_mean = 0.0
    den_corrected_mean = 0.0
    blank_blocks: Tuple[_SSBBlankBlock, ...] = ()
    u_blank_fallback = _active_contributor_value("u_blank") if _enabled("u_blank") else 0.0

    ratio_def = element_config.default_ratios.get(ratio_name)
    if ratio_def is not None and u_blank_fallback > 0:
        num_isotope, den_isotope = ratio_def
        blank_samples, _blank_selection = resolve_blank_samples_for_uncertainty(
            sample, all_samples,
        )
        configured_blank_mode = (
            processing_config.blank_mode if processing_config is not None else None
        )
        blank_mode = resolve_blank_correction_mode(sample, configured_blank_mode)
        if blank_samples is None:
            blank_samples = []
        elif not blank_samples:
            blank_samples = [
                s for s in all_samples
                if s.is_blank and not s.metadata.get("excluded", False)
            ]

        primary_blank = None
        if blank_samples and not (
            blank_mode == "before_and_after" and len(blank_samples) >= 2
        ):
            primary_blank = blank_samples[0]

        selected_blanks = []
        if blank_mode != "none" and blank_samples:
            if blank_mode == "before_and_after" and len(blank_samples) >= 2:
                selected_blanks = [blank_samples[0], blank_samples[-1]]
            else:
                selected_blanks = [blank_samples[0]]

        blank_blocks = _build_ssb_blank_blocks(
            selected_blanks,
            isotopes=(num_isotope, den_isotope),
            correlation_method=uncertainty_config.blank_correlation_method,
            fixed_r=uncertainty_config.blank_fixed_r,
            blank_uncertainty_input=getattr(
                uncertainty_config,
                "blank_uncertainty_input",
                "sd",
            ),
            cycle_ranges=cycle_ranges,
        )
        if blank_blocks:
            u_blank_fallback = 0.0

        if primary_blank is not None:
            from domain.uncertainty.blank import _get_blank_voltages

            v_num = _get_blank_voltages(
                primary_blank,
                num_isotope,
                cycle_ranges=cycle_ranges,
            )
            v_den = _get_blank_voltages(
                primary_blank,
                den_isotope,
                cycle_ranges=cycle_ranges,
            )
            if len(v_num) >= 2:
                blank_num_mean = float(np.mean(v_num))
                blank_num_sd = float(np.std(v_num, ddof=1))
            if len(v_den) >= 2:
                blank_den_mean = float(np.mean(v_den))
                blank_den_sd = float(np.std(v_den, ddof=1))
        elif blank_blocks:
            mean_vectors = np.array([block.mean_vector for block in blank_blocks], dtype=np.float64)
            blank_num_mean = float(np.mean(mean_vectors[:, 0]))
            blank_den_mean = float(np.mean(mean_vectors[:, 1]))
            blank_num_sd = float(np.mean([np.sqrt(block.covariance_matrix[0, 0]) for block in blank_blocks]))
            blank_den_sd = float(np.mean([np.sqrt(block.covariance_matrix[1, 1]) for block in blank_blocks]))

        # Blank-corrected intensities (sensitivity coefficients)
        num_corrected_mean = _get_corrected_intensity_mean(
            sample,
            num_isotope,
            cycle_ranges=cycle_ranges,
            ratio_name=ratio_name,
        )
        den_corrected_mean = _get_corrected_intensity_mean(
            sample,
            den_isotope,
            cycle_ranges=cycle_ranges,
            ratio_name=ratio_name,
        )
        blank_result = compute_blank_uncertainty(
            blank_samples=blank_samples,
            num_isotope=num_isotope,
            den_isotope=den_isotope,
            num_corrected_mean=num_corrected_mean,
            den_corrected_mean=den_corrected_mean,
            correlation_method=uncertainty_config.blank_correlation_method,
            fixed_r=uncertainty_config.blank_fixed_r,
            blank_correction_mode=blank_mode,
            blank_uncertainty_input=getattr(
                uncertainty_config,
                "blank_uncertainty_input",
                "sd",
            ),
            cycle_ranges=cycle_ranges,
            channel_weights=resolve_blank_channel_weights(
                sample,
                (num_isotope, den_isotope),
                blank_mode=blank_mode,
            ),
        )
        blank_correlation = float(blank_result.correlation)

    # Build custom contributor perturbation sigmas (draws happen per-iteration in _mc_iteration_ssb)
    custom_contributor_sigmas, custom_contributor_distributions, custom_contributor_estimation_dofs = _extract_custom_contributor_params(
        sample=sample,
        element_config=element_config,
        ratio_mean=ratio_mean,
        custom_contributor_library=custom_contributor_library,
    )
    kappa_distributions = {
        "u_k1_sample_decomposition": _normalize_mc_distribution(
            getattr(uncertainty_config, "k1_sample_decomposition_distribution", "rectangular")
        ),
        "u_k2_matrix_separation": _normalize_mc_distribution(
            getattr(uncertainty_config, "k2_matrix_separation_distribution", "normal")
        ),
        "u_k3_procedural_blank": _normalize_mc_distribution(
            getattr(uncertainty_config, "k3_procedural_blank_distribution", "rectangular")
        ),
        "u_k4_bracketing_standard_heterogeneity": _normalize_mc_distribution(
            getattr(
                uncertainty_config,
                "k4_bracketing_standard_heterogeneity_distribution",
                "rectangular",
            )
        ),
        "u_k6_matrix_effects": _normalize_mc_distribution(
            getattr(uncertainty_config, "k6_matrix_effects_distribution", "normal")
        ),
        "u_k7_residual_interferences": _normalize_mc_distribution(
            getattr(
                uncertainty_config,
                "k7_residual_interferences_distribution",
                "rectangular",
            )
        ),
    }

    if not is_pb_hg_ssb_ratio(sample, ratio_name):
        # Legacy B has no nonlinear blank replay. Collapse the SAME joint
        # Gaussian linear model to its basis-scale sigma, never sample-only.
        blank_blocks = ()
        u_blank_fallback = 0.0
        if _enabled("u_blank"):
            from domain.uncertainty.joint_blank import ordinary_ssb_blank
            try:
                joint = ordinary_ssb_blank(sample, ratio_name, all_samples, uncertainty_config,
                    processing_config, cycle_ranges,
                    classic_delta=uncertainty_config.output_mode == "delta" and _uses_classic_delta_reference(sample, ratio_name, uncertainty_config=uncertainty_config, processing_config=processing_config))
            except ValueError as exc:
                raise MCCrossCheckError(engine="ssb_delta", reason_code="missing_chain_input", reason=str(exc)) from exc
            u_blank_fallback = joint.u_blank_rel * abs(ratio_mean)

    return _SSBPerturbationParams(
        ratio_mean=ratio_mean,
        u_prec=u_prec_abs,
        u_std=u_std,
        u_std_repeatability=u_std_repeatability,
        u_k1_sample_decomposition=u_k1_sample_decomposition,
        u_k2_matrix_separation=u_k2_matrix_separation,
        u_k3_procedural_blank=u_k3_procedural_blank,
        u_k4_bracketing_standard_heterogeneity=u_k4_bracketing_standard_heterogeneity,
        u_k5_instrumental_drift=u_k5_instrumental_drift,
        u_k6_matrix_effects=u_k6_matrix_effects,
        u_k7_residual_interferences=u_k7_residual_interferences,
        kappa_drift_distribution=_normalize_mc_distribution(
            getattr(uncertainty_config, "kappa_drift_distribution", "normal")
        ),
        u_crm=u_crm,
        u_prec_dof=u_prec_dof,
        u_std_dof=u_std_dof,
        u_std_repeatability_dof=u_std_repeatability_dof,
        blank_num_mean=blank_num_mean,
        blank_den_mean=blank_den_mean,
        blank_num_sd=blank_num_sd,
        blank_den_sd=blank_den_sd,
        blank_correlation=blank_correlation,
        num_corrected_mean=num_corrected_mean,
        den_corrected_mean=den_corrected_mean,
        u_blank_fallback=u_blank_fallback,
        blank_blocks=blank_blocks,
        custom_contributor_sigmas=custom_contributor_sigmas,
        custom_contributor_distributions=custom_contributor_distributions,
        custom_contributor_estimation_dofs=custom_contributor_estimation_dofs,
        kappa_distributions=kappa_distributions,
    )


def _extract_internal_perturbation_params(
    sample: Sample,
    ratio_name: str,
    ratio_values: np.ndarray,
    ratio_mean: float,
    all_samples: List[Sample],
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    *,
    processing_config: ProcessingConfig,
    certified_value: Optional[CertifiedValue],
    gum_budget: Optional[UncertaintyBudget] = None,
    ratio_mask: Optional[np.ndarray] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    custom_contributor_library: Optional[Dict[str, List[CustomUncertaintyContributor]]] = None,
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
) -> _InternalPerturbationParams:
    """Extract all parameters needed for one Engine A MC iteration."""

    def _active_contributor_value(name: str) -> float:
        if gum_budget is None:
            return 0.0
        contributor = gum_budget._find_contributor(name)
        if contributor is None or not contributor.is_active:
            return 0.0
        return float(contributor.value_abs)

    def _active_contributor_dof(name: str) -> float:
        if gum_budget is None:
            return float("inf")
        contributor = gum_budget._find_contributor(name)
        if contributor is None or not contributor.is_active:
            return float("inf")
        try:
            dof = float(contributor.degrees_of_freedom)
        except (TypeError, ValueError):
            return float("inf")
        return dof if dof > 0.0 else float("inf")

    _custom_names = {
        d.name
        for defs in (custom_contributor_library or {}).values()
        for d in defs
    }

    def _enabled(name: str) -> bool:
        return is_contributor_active(
            name=name,
            uncertainty_config=uncertainty_config,
            sample=sample,
            element_symbol=element_config.symbol,
            custom_contributor_names=_custom_names,
            profile_defaults=profile_defaults,
        )

    if ratio_name != "87Sr/86Sr":
        raise ValueError(
            "Engine A Monte Carlo currently supports 87Sr/86Sr only."
        )

    ratio_mask = _resolve_internal_ratio_mask(
        sample,
        ratio_name,
        ratio_mask=ratio_mask,
        cycle_ranges=cycle_ranges,
    )

    src = (
        sample.blank_corrected_intensities
        if sample.blank_corrected_intensities
        else sample.corrected_intensities
        if sample.corrected_intensities
        else sample.intensities
    )
    norm_num_iso, norm_den_iso, _m_norm_num, _m_norm_den = _resolve_active_sr_normalization_pair(
        processing_config,
        element_config,
    )
    replay_isotopes = {
        "87Sr",
        "86Sr",
        "88Sr",
        "85Rb",
        "83Kr",
        "84Sr",
        norm_num_iso,
        norm_den_iso,
    }
    base_corrected_intensities = {
        isotope: np.asarray(cycle_data.values, dtype=np.float64).copy()
        for isotope, cycle_data in src.items()
        if isotope in replay_isotopes
    }

    normalization_value = _resolve_active_sr_normalization_value(
        processing_config,
        element_config,
    )
    normalization_value = float(normalization_value or 0.0)
    output_scale = resolve_output_scale(
        sample,
        ratio_name,
        input_layer=INPUT_LAYER_PRE_ANCHOR,
    )
    scale_components = dict(output_scale.components)
    sr_anchor_factor = scale_components.get(
        "sr_session_anchoring",
        1.0,
    )
    if (
        "sr_session_anchoring" not in scale_components
        and processing_config.sr_session_anchoring
    ):
        try:
            legacy_anchor = float(sample.metadata.get("_sr_anchor_factor", 1.0))
        except (TypeError, ValueError):
            legacy_anchor = 1.0
        if np.isfinite(legacy_anchor) and legacy_anchor > 0.0:
            sr_anchor_factor = legacy_anchor

    drift_factor = scale_components.get("drift", 1.0)
    if (
        "drift" not in scale_components
        and sample.drift_corrected_ratios
        and ratio_name in sample.drift_corrected_ratios
    ):
        iif_cd = sample.sr_standard_corrected_ratios.get(ratio_name)
        if iif_cd is None:
            iif_cd = sample.iif_corrected_ratios.get(ratio_name)
        if iif_cd is not None:
            iif_values = np.asarray(iif_cd.values, dtype=np.float64)
            iif_mask = np.asarray(iif_cd.mask, dtype=bool) & np.isfinite(iif_values)
            if ratio_mask is not None and np.shape(ratio_mask) == np.shape(iif_mask):
                iif_mask &= np.asarray(ratio_mask, dtype=bool)
            selected_iif = iif_values[iif_mask]
            if selected_iif.size:
                iif_mean = float(np.mean(selected_iif))
                candidate = ratio_mean / iif_mean if iif_mean != 0.0 else np.nan
                if np.isfinite(candidate) and candidate > 0.0:
                    drift_factor = candidate

    post_chain_scale = sr_anchor_factor * drift_factor

    u_prec = _active_contributor_value("u_prec") if _enabled("u_prec") else 0.0
    u_prec_dof = _active_contributor_dof("u_prec")
    if u_prec <= 0.0 and _enabled("u_prec"):
        u_prec = _select_u_prec_from_values(ratio_values, uncertainty_config)
        u_prec_dof = float(max(len(ratio_values) - 1, 1))

    u_norm_ratio_mode_b = (
        _active_contributor_value("u_norm_ratio")
        if _enabled("u_norm_ratio") else 0.0
    )

    # Read u_std_repeatability from whichever variant is active in the GUM budget
    # (SD = "u_std_repeatability", SE = "u_std_repeatability_se").
    _se_mode_active = _enabled("u_std_repeatability_se") and not _enabled("u_std_repeatability")
    reprod_ratio_extractor = (
        _get_pre_anchor_ratio
        if processing_config.sr_session_anchoring
        else _get_iif_or_best_ratio
    )
    if _se_mode_active:
        u_std_repeatability = (
            _active_contributor_value("u_std_repeatability_se")
            if _enabled("u_std_repeatability_se") else 0.0
        )
        if u_std_repeatability <= 0.0 and _enabled("u_std_repeatability_se"):
            reprod_result = compute_reprod(
                all_samples=all_samples,
                ratio_name=ratio_name,
                uncertainty_config=uncertainty_config,
                element_config=element_config,
                ratio_extractor=reprod_ratio_extractor,
            )
            n_std = max(int(sum(1 for inc in reprod_result.std_included if inc)), 1)
            u_std_repeatability = float(reprod_result.u_std_repeatability_abs) / np.sqrt(n_std)
    else:
        u_std_repeatability = (
            _active_contributor_value("u_std_repeatability")
            if _enabled("u_std_repeatability") else 0.0
        )
        if u_std_repeatability <= 0.0 and _enabled("u_std_repeatability"):
            reprod_result = compute_reprod(
                all_samples=all_samples,
                ratio_name=ratio_name,
                uncertainty_config=uncertainty_config,
                element_config=element_config,
                ratio_extractor=reprod_ratio_extractor,
            )
            u_std_repeatability = float(reprod_result.u_std_repeatability_abs)

    u_kappa_drift = (
        _active_contributor_value("u_kappa_drift")
        if _enabled("u_kappa_drift") else 0.0
    )
    # The automatic reference-bias model is disabled, so the
    # cross-check never draws this term. The recomputation fallback that used
    # to sit here would have re-derived |delta_ref| / sqrt(3) even when the
    # GUM budget correctly reported nothing, reintroducing the disabled model
    # through the Monte Carlo path alone.
    u_bias_ref = (
        _active_contributor_value("u_bias_ref")
        if (AUTOMATIC_REFERENCE_BIAS_ENABLED and _enabled("u_bias_ref"))
        else 0.0
    )
    u_bias_qc = _active_contributor_value("u_bias_qc") if _enabled("u_bias_qc") else 0.0
    if u_bias_qc <= 0.0 and _enabled("u_bias_qc"):
        sr_qc_bias_abs, sr_qc_cert_value = resolve_sr_qc_bias_inputs(
            sample,
            uncertainty_config,
        )
        u_bias_qc, _, _ = compute_qc_bias_term(
            observed_bias_abs=sr_qc_bias_abs,
            ratio_mean=ratio_mean,
            qc_cert_value=sr_qc_cert_value,
        )
    u_reprod_dig = (
        _active_contributor_value("u_reprod_dig")
        if _enabled("u_reprod_dig") else 0.0
    )
    if u_reprod_dig <= 0.0 and _enabled("u_reprod_dig"):
        sr_digestion_sd_abs, sr_digestion_ref_value = resolve_sr_digestion_inputs(
            sample,
            uncertainty_config,
        )
        u_reprod_dig, _, _ = compute_digestion_reproducibility_term(
            digestion_sd_abs=sr_digestion_sd_abs,
            ratio_mean=ratio_mean,
            digestion_ref_value=sr_digestion_ref_value,
        )
    u_crm = _active_contributor_value("u_crm") if _enabled("u_crm") else 0.0
    u_ref_value = (
        _active_contributor_value("u_ref_value") if _enabled("u_ref_value") else 0.0
    )

    blank_blocks = ()
    if _enabled("u_blank"):
        blank_blocks = _build_internal_blank_blocks(
            sample,
            all_samples,
            ratio_name=ratio_name,
            uncertainty_config=uncertainty_config,
            processing_config=processing_config,
            cycle_ranges=cycle_ranges,
        )
    u_blank_fallback = (
        0.0 if blank_blocks else _active_contributor_value("u_blank")
    ) if _enabled("u_blank") else 0.0

    required_chain_isotopes = {"87Sr", "86Sr", norm_num_iso, norm_den_iso}
    # Engine A normalization-ratio uncertainty is sampled through the same
    # correction-chain replay as the deterministic u_norm_ratio contributor.
    has_active_norm_ratio_contributor = bool(
        _enabled("u_norm_ratio")
        or uncertainty_config.is_control_enabled("enable_sr_norm_ratio_uncertainty")
    )
    mode_b_chain_enabled = (
        has_active_norm_ratio_contributor
        and processing_config.apply_mass_bias_correction
        and normalization_value > 0.0
        and required_chain_isotopes.issubset(base_corrected_intensities.keys())
    )
    normalization_value_sd = 0.0
    if mode_b_chain_enabled:
        normalization_value_sd = _resolve_norm_ratio_u_abs(
            uncertainty_config,
            normalization_value,
        )
        if normalization_value_sd <= 0.0:
            mode_b_chain_enabled = False
    u_norm_ratio_mode_b_fallback = 0.0 if mode_b_chain_enabled else u_norm_ratio_mode_b

    enabled_interferents: Set[str] = {
        spec.interfering_isotope
        for spec in getattr(element_config, "monitors", ()) or ()
        if (
            getattr(spec, "family", "f") == "f"
            and processing_config.is_monitor_enabled(spec.interfering_isotope)
        )
    }
    interference_required_isotopes = set(required_chain_isotopes)
    if "87Rb" in enabled_interferents:
        interference_required_isotopes.add("85Rb")
    if "86Kr" in enabled_interferents:
        interference_required_isotopes.add("83Kr")
    interference_chain_enabled = (
        _enabled("u_interf")
        and processing_config.apply_interference_correction
        and bool(enabled_interferents)
        and normalization_value > 0.0
        and interference_required_isotopes.issubset(base_corrected_intensities.keys())
    )
    u_interf_fallback = (
        0.0 if interference_chain_enabled else _active_contributor_value("u_interf")
    ) if _enabled("u_interf") else 0.0
    reference_inputs = _resolve_sr_reference_inputs(
        processing_config,
        element_config,
        sample,
    )

    # Build custom contributor perturbation sigmas (draws happen per-iteration in _mc_iteration_internal)
    custom_contributor_sigmas, custom_contributor_distributions, custom_contributor_estimation_dofs = _extract_custom_contributor_params(
        sample=sample,
        element_config=element_config,
        ratio_mean=ratio_mean,
        custom_contributor_library=custom_contributor_library,
    )

    return _InternalPerturbationParams(
        ratio_mean=ratio_mean,
        ratio_mask=ratio_mask,
        base_corrected_intensities=base_corrected_intensities,
        normalization_value=normalization_value,
        normalization_value_sd=normalization_value_sd,
        sr_anchor_factor=sr_anchor_factor,
        post_chain_scale=post_chain_scale,
        apply_interference=processing_config.apply_interference_correction,
        apply_iif=processing_config.apply_mass_bias_correction,
        enabled_interferents=enabled_interferents,
        mode_b_chain_enabled=mode_b_chain_enabled,
        interference_chain_enabled=interference_chain_enabled,
        u_prec=u_prec,
        u_prec_dof=u_prec_dof,
        u_norm_ratio_mode_b_fallback=u_norm_ratio_mode_b_fallback,
        u_std_repeatability=float(u_std_repeatability),
        u_kappa_drift=float(u_kappa_drift),
        kappa_drift_distribution=_normalize_mc_distribution(
            getattr(uncertainty_config, "kappa_drift_distribution", "normal")
        ),
        u_bias_ref=float(u_bias_ref),
        u_bias_qc=float(u_bias_qc),
        u_reprod_dig=float(u_reprod_dig),
        u_crm=float(u_crm),
        u_ref_value=float(u_ref_value),
        u_interf_fallback=float(u_interf_fallback),
        u_blank_fallback=float(u_blank_fallback),
        reference_inputs=reference_inputs,
        blank_blocks=blank_blocks,
        custom_contributor_sigmas=custom_contributor_sigmas,
        custom_contributor_distributions=custom_contributor_distributions,
        custom_contributor_estimation_dofs=custom_contributor_estimation_dofs,
    )


def _resolve_internal_ratio_mask(
    sample: Sample,
    ratio_name: str,
    *,
    ratio_mask: Optional[np.ndarray] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[np.ndarray]:
    """Resolve the runtime cycle mask for Engine A MC."""
    if ratio_mask is not None:
        return np.asarray(ratio_mask, dtype=bool).copy()

    ratio_cd = get_best_ratio_data(sample, ratio_name)
    if ratio_cd is None:
        return None

    mask = np.asarray(ratio_cd.mask, dtype=bool).copy()
    if not cycle_ranges:
        return mask

    cycle_range = resolve_cycle_range(
        cycle_ranges,
        sample_name=sample.name,
        sample_key=sample_cycle_key(sample),
    )
    if cycle_range is None:
        return mask

    start_idx = max(int(cycle_range[0]) - 1, 0)
    end_idx = min(int(cycle_range[1]), len(mask))
    range_mask = np.zeros(len(mask), dtype=bool)
    if end_idx > start_idx:
        range_mask[start_idx:end_idx] = True
    mask &= range_mask

    finite_visible_mask = mask & np.isfinite(ratio_cd.values)
    valid_indices = np.where(finite_visible_mask)[0]
    if len(valid_indices) < 3:
        return mask

    return mask


def _build_internal_blank_blocks(
    sample: Sample,
    all_samples: List[Sample],
    *,
    ratio_name: str,
    uncertainty_config: UncertaintyConfig,
    processing_config: ProcessingConfig,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Tuple[_InternalBlankBlock, ...]:
    """Build the blank-file perturbation blocks for Engine A MC."""
    blank_samples, _blank_selection = resolve_blank_samples_for_uncertainty(
        sample, all_samples,
    )
    if blank_samples is None:
        return ()
    if not blank_samples:
        blank_samples = [
            blank_sample
            for blank_sample in all_samples
            if blank_sample.is_blank and not blank_sample.metadata.get("excluded", False)
        ]
    if not blank_samples:
        return ()

    blank_mode = resolve_blank_correction_mode(sample, processing_config.blank_mode)
    if blank_mode == "none":
        return ()
    if blank_mode == "before_and_after" and len(blank_samples) >= 2:
        selected_blanks = [blank_samples[0], blank_samples[-1]]
    else:
        selected_blanks = [blank_samples[0]]

    from domain.uncertainty.engine_internal_sr import _sr_blank_isotopes
    from domain.uncertainty.engine_external_pb_tl import build_pb_tl_blank_blocks
    isotopes = _sr_blank_isotopes(sample, uncertainty_config, processing_config)
    try:
        return build_pb_tl_blank_blocks(sample, all_samples, isotopes=isotopes,
            uncertainty_config=uncertainty_config, processing_config=processing_config, cycle_ranges=cycle_ranges, paired_blank_filter=True)
    except ValueError as exc:
        raise MCCrossCheckError(engine="internal_normalization", reason_code="invalid_covariance" if "covariance" in str(exc) else "missing_chain_input", reason=str(exc)) from exc



def _build_ssb_blank_blocks(
    blank_samples: List[Sample],
    *,
    isotopes: Tuple[str, str],
    correlation_method: str,
    fixed_r: float,
    blank_uncertainty_input: str = "sd",
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Tuple[_SSBBlankBlock, ...]:
    """Build one or two covariance-aware blank perturbation blocks for Engine B."""
    blocks: List[_SSBBlankBlock] = []
    for blank in blank_samples:
        block = _build_internal_blank_block_2var(
            blank,
            isotopes,
            correlation_method=correlation_method,
            fixed_r=fixed_r,
            blank_uncertainty_input=blank_uncertainty_input,
            cycle_ranges=cycle_ranges,
        )
        if block is None:
            return ()
        mean_vector, cov_matrix = block
        blocks.append(
            _SSBBlankBlock(
                mean_vector=mean_vector,
                covariance_matrix=cov_matrix,
            )
        )
    return tuple(blocks)


def _build_internal_blank_block_2var(
    blank: Sample,
    isotopes: Tuple[str, str],
    *,
    correlation_method: str,
    fixed_r: float,
    blank_uncertainty_input: str = "sd",
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build one 2-variable blank block for Engine A MC."""
    num_iso, den_iso = isotopes
    v_num, v_den = get_paired_blank_voltages(
        blank,
        num_iso,
        den_iso,
        cycle_ranges=cycle_ranges,
    )
    n_pairs = len(v_num)
    if n_pairs < 2:
        return None

    v_num = np.asarray(v_num, dtype=np.float64)
    v_den = np.asarray(v_den, dtype=np.float64)
    u_num_sd = float(np.std(v_num, ddof=1))
    u_den_sd = float(np.std(v_den, ddof=1))
    input_sds = blank_input_sigmas(
        np.array([u_num_sd, u_den_sd], dtype=np.float64),
        n_pairs,
        blank_uncertainty_input,
    )
    correlation, _ = _compute_correlation(
        v_num,
        v_den,
        correlation_method,
        fixed_r,
    )
    covariance = float(np.clip(correlation, -1.0, 1.0) * input_sds[0] * input_sds[1])
    mean_vector = np.array([float(np.mean(v_num)), float(np.mean(v_den))], dtype=np.float64)
    cov_matrix = np.array(
        [[input_sds[0] ** 2, covariance], [covariance, input_sds[1] ** 2]],
        dtype=np.float64,
    )
    return mean_vector, cov_matrix


# Single MC iteration

def _mc_iteration_ssb(params: _SSBPerturbationParams, rng: np.random.Generator) -> float:
    """Run one MC iteration for Engine B (SSB-delta)."""
    p = params

    result = p.ratio_mean

    for sigma, dof in (
        (p.u_prec, p.u_prec_dof),
        (p.u_std, p.u_std_dof),
        (p.u_std_repeatability, p.u_std_repeatability_dof),
    ):
        result += _draw_standard_uncertainty(
            rng,
            sigma,
            "normal",
            degrees_of_freedom=dof,
        )
    result += _draw_standard_uncertainty(rng, p.u_crm, "normal")

    kappa_distributions = p.kappa_distributions or {}
    for name, sigma in (
        ("u_k1_sample_decomposition", p.u_k1_sample_decomposition),
        ("u_k2_matrix_separation", p.u_k2_matrix_separation),
        ("u_k3_procedural_blank", p.u_k3_procedural_blank),
        (
            "u_k4_bracketing_standard_heterogeneity",
            p.u_k4_bracketing_standard_heterogeneity,
        ),
        ("u_k6_matrix_effects", p.u_k6_matrix_effects),
        ("u_k7_residual_interferences", p.u_k7_residual_interferences),
    ):
        result += _draw_standard_uncertainty(
            rng,
            sigma,
            kappa_distributions.get(name, "normal"),
        )

    if p.u_k5_instrumental_drift > 0:
        result += _draw_standard_uncertainty(
            rng,
            p.u_k5_instrumental_drift,
            p.kappa_drift_distribution,
        )

    # Blank covariance-aware shift
    if p.den_corrected_mean > 0 and p.blank_blocks:
        delta_b_num = 0.0
        delta_b_den = 0.0
        n_blocks = float(len(p.blank_blocks))
        for block in p.blank_blocks:
            sampled = _sample_multivariate(
                block.mean_vector,
                block.covariance_matrix,
                rng,
            )
            delta_b_num += (sampled[0] - block.mean_vector[0]) / n_blocks
            delta_b_den += (sampled[1] - block.mean_vector[1]) / n_blocks
        c_num = -1.0 / p.den_corrected_mean
        c_den = p.num_corrected_mean / (p.den_corrected_mean ** 2)
        result += c_num * delta_b_num + c_den * delta_b_den
    elif p.u_blank_fallback > 0:
        result += rng.normal(0.0, p.u_blank_fallback)

    # Custom contributor additive perturbations
    if p.custom_contributor_sigmas:
        distributions = p.custom_contributor_distributions or {}
        for name, sigma in p.custom_contributor_sigmas.items():
            result += _draw_standard_uncertainty(
                rng,
                sigma,
                distributions.get(name, "normal"),
            )

    return result


# ---------------------------------------------------------------------------
# Engine B fixed-draw chain replay
#
# Every draw samples the declared Engine B inputs and re-evaluates the
# production SSB/delta measurement model. The iteration returns the final
# reported quantity — absolute ratio or delta in permil — so the frozen
# post-loop basis-space transform is skipped for this path and retained only
# for the legacy additive path.
# ---------------------------------------------------------------------------

#: Contributors sampled as direct model inputs. They must never also appear as
#: output-level additive terms; the exclusion is structural, not a convention.
ENGINE_B_DIRECT_INPUT_CONTRIBUTORS: frozenset = frozenset(
    {"u_prec", "u_std", "u_crm", "u_blank"}
)


@dataclass(frozen=True)
class EngineBInput:
    """One directly sampled Engine B model input."""

    name: str
    value: float
    standard_uncertainty: float
    degrees_of_freedom: float
    distribution: str
    placement: str
    type_ab: str
    source_field: str


@dataclass(frozen=True)
class EngineBContributorSpec:
    """Declared specification of one sampled Engine B contributor.

    Persistence and export metadata only: it records *how* a contributor was
    declared to be drawn (placement, distribution, degrees of freedom, GUM
    Type A/B class and the structured field it came from) so a saved result
    states its own contributor model. It is never consulted by the sampler.
    """

    name: str
    placement: str
    distribution: str
    degrees_of_freedom: float
    type_ab: str
    source_field: str


@dataclass(frozen=True)
class EngineBOutputLevelTerm:
    """One contributor sampled once on the reported output.

    ``type_ab`` and ``source_field`` are durable formulation metadata. They do
    not participate in the numerical draw, but must travel with the term so a
    persisted record does not relabel a finite-DoF Type A contributor as Type B.
    """

    name: str
    sigma: float
    distribution: str
    degrees_of_freedom: float
    placement: str = PLACEMENT_OUTPUT_LEVEL
    type_ab: str = "B"
    source_field: str = "gum_budget.contributors"


@dataclass(frozen=True, eq=False)
class EngineBLinearizedBlankTerm:
    """One blank observation drawn once and mapped onto the reported output.

    For a Pb ratio with an applied ordinary-SSB Hg correction the blank enters
    the sample and every bracket member through the Hg-corrected chain, so its
    effect per draw is ``gradient · (draw − mean)`` with the first-order
    sensitivity in reported units. The covariance is validated like any other
    blank block and is never replaced by independent channel draws.
    """

    blank_observation_id: str
    channels: Tuple[str, ...]
    mean_vector: np.ndarray
    covariance_matrix: np.ndarray
    gradient: np.ndarray


@dataclass(frozen=True)
class EngineBFixedDrawParams:
    """Immutable Engine B fixed-draw Monte Carlo parameter set.

    Carries the separate ``R_s``, ``R_p``, ``R_n`` and ``C`` estimates with
    their own standard uncertainties, degrees of freedom and distributions, the
    bracket identity and mode, the output mode, every contributor placement,
    and an explicit iteration result space plus semantics version.
    """

    semantics_version: str
    result_space: str
    output_mode: str
    bracket_mode: str
    sample_name: str
    ratio_name: str

    r_s: EngineBInput
    r_p: EngineBInput
    r_n: EngineBInput
    c: Optional[EngineBInput]

    #: True when the SSB kernel is evaluated; False for classic bracketing delta.
    apply_ssb_kernel: bool
    #: True when delta is taken against the same certified reference used by SSB,
    #: so the drawn ``C`` cancels algebraically.
    delta_from_ssb: bool

    prev_std_label: str
    next_std_label: str

    output_level_terms: Tuple[EngineBOutputLevelTerm, ...]

    #: Sample blank perturbation, applied to ``R_s`` before the SSB kernel.
    blank_blocks: Tuple["_SSBBlankBlock", ...]
    blank_c_num: float
    blank_c_den: float
    blank_uncertainty_input: str
    blank_placement: str
    #: Output-level sample blank fallback. Mutually exclusive with the blocks.
    u_blank_fallback: float
    blank_fallback_distribution: str
    blank_fallback_dof: float

    #: Nominal evaluation of the same model, in the reported space.
    nominal_reported_value: float
    #: Nominal bracket mean, used as the scale for the near-zero denominator test.
    nominal_bracket_mean: float
    #: Nominal delta denominator, used as the scale guard and as the sensitivity
    #: coefficient that maps an absolute output-level sigma into permil.
    nominal_delta_reference: float

    min_type_a_dof: Optional[float]
    moment_status: str
    #: ``(input name, declared placement, degrees of freedom)`` for every
    #: drawn contributor whose Student-t variance is undefined.
    low_dof_disclosures: Tuple[Tuple[str, str, float], ...]

    #: Linearized Hg-chain blank observations (Pb Hg ratios only). Exclusive
    #: with ``blank_blocks`` and the blank fallback.
    hg_blank_terms: Tuple[EngineBLinearizedBlankTerm, ...] = ()
    #: Identity of the Hg propagation model behind the linearized terms, or "".
    hg_propagation_method: str = ""

    def contributor_placements(self) -> Tuple[Tuple[str, str], ...]:
        """Return every sampled contributor with its declared placement."""
        rows = [
            (self.r_s.name, self.r_s.placement),
            (self.r_p.name, self.r_p.placement),
            (self.r_n.name, self.r_n.placement),
        ]
        if self.c is not None:
            rows.append((self.c.name, self.c.placement))
        if self.blank_blocks:
            rows.append(("u_blank", self.blank_placement))
        elif self.hg_blank_terms:
            rows.append(("u_blank", PLACEMENT_LINEARIZED_OUTPUT))
        elif self.u_blank_fallback > 0.0:
            rows.append(("u_blank", PLACEMENT_BLANK_FALLBACK))
        rows.extend((term.name, term.placement) for term in self.output_level_terms)
        return tuple(rows)

    def contributor_specs(self) -> Tuple[EngineBContributorSpec, ...]:
        """Return the full declared specification of every sampled contributor.

        Persistence metadata that parallels :meth:`contributor_placements`;
        it adds the configured distribution, degrees of freedom, Type A/B
        class and structured source field for each drawn term. Nothing here
        changes what is drawn.
        """
        specs = [
            EngineBContributorSpec(
                name=spec.name,
                placement=spec.placement,
                distribution=spec.distribution,
                degrees_of_freedom=float(spec.degrees_of_freedom),
                type_ab=spec.type_ab,
                source_field=spec.source_field,
            )
            for spec in (self.r_s, self.r_p, self.r_n, self.c)
            if spec is not None
        ]
        if self.blank_blocks:
            specs.append(
                EngineBContributorSpec(
                    name="u_blank",
                    placement=self.blank_placement,
                    # Detailed sample blank blocks are drawn from the stored
                    # covariance-aware multivariate normal model.
                    distribution="multivariate_normal",
                    degrees_of_freedom=float("inf"),
                    type_ab="A",
                    source_field=f"blank_blocks[{len(self.blank_blocks)}]",
                )
            )
        elif self.hg_blank_terms:
            specs.append(
                EngineBContributorSpec(
                    name="u_blank",
                    placement=PLACEMENT_LINEARIZED_OUTPUT,
                    distribution="multivariate_normal",
                    degrees_of_freedom=float("inf"),
                    type_ab="A",
                    source_field=f"hg_blank_terms[{len(self.hg_blank_terms)}]",
                )
            )
        elif self.u_blank_fallback > 0.0:
            specs.append(
                EngineBContributorSpec(
                    name="u_blank",
                    placement=PLACEMENT_BLANK_FALLBACK,
                    distribution=self.blank_fallback_distribution,
                    degrees_of_freedom=float(self.blank_fallback_dof),
                    type_ab="B",
                    source_field="u_blank_fallback",
                )
            )
        specs.extend(
            EngineBContributorSpec(
                name=term.name,
                placement=term.placement,
                distribution=term.distribution,
                degrees_of_freedom=float(term.degrees_of_freedom),
                type_ab=term.type_ab,
                source_field=term.source_field,
            )
            for term in self.output_level_terms
        )
        return tuple(specs)


def engine_b_input_digest(params: EngineBFixedDrawParams) -> str:
    """Hash the resolved fixed-draw inputs without exposing raw metadata.

    The identity includes every scalar that can change a draw distribution,
    the frozen bracket identities, and the detailed blank arrays. It is kept
    separate from the configuration digest because the same settings can be
    applied to different brackets or runtime cycle selections.
    """
    hasher = hashlib.sha256()

    def add(value: object) -> None:
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"\x00")

    for value in (
        params.semantics_version,
        params.result_space,
        params.output_mode,
        params.bracket_mode,
        params.sample_name,
        params.ratio_name,
        params.prev_std_label,
        params.next_std_label,
        params.apply_ssb_kernel,
        params.delta_from_ssb,
        params.blank_uncertainty_input,
        params.blank_placement,
        float(params.u_blank_fallback).hex(),
        params.blank_fallback_distribution,
        float(params.blank_fallback_dof).hex(),
    ):
        add(value)

    for spec in (params.r_s, params.r_p, params.r_n, params.c):
        if spec is None:
            add("none")
            continue
        for value in (
            spec.name,
            float(spec.value).hex(),
            float(spec.standard_uncertainty).hex(),
            float(spec.degrees_of_freedom).hex(),
            spec.distribution,
            spec.placement,
            spec.type_ab,
            spec.source_field,
        ):
            add(value)

    for term in params.output_level_terms:
        for value in (
            term.name,
            float(term.sigma).hex(),
            term.distribution,
            float(term.degrees_of_freedom).hex(),
            term.placement,
            term.type_ab,
            term.source_field,
        ):
            add(value)

    for block in params.blank_blocks:
        for array in (block.mean_vector, block.covariance_matrix):
            canonical = np.ascontiguousarray(array, dtype=np.float64)
            add(canonical.shape)
            hasher.update(canonical.tobytes(order="C"))
            hasher.update(b"\x00")

    # Only an Hg-corrected Pb ratio adds these, so every other Engine B input
    # identity is unchanged.
    if params.hg_propagation_method:
        add(params.hg_propagation_method)
        for term in params.hg_blank_terms:
            add(term.blank_observation_id)
            add("|".join(term.channels))
            for array in (term.mean_vector, term.covariance_matrix, term.gradient):
                canonical = np.ascontiguousarray(array, dtype=np.float64)
                add(canonical.shape)
                hasher.update(canonical.tobytes(order="C"))
                hasher.update(b"\x00")

    return hasher.hexdigest()[:16]


def _moment_status_from_dof(min_dof: Optional[float]) -> str:
    """Classify which Student-t moments the reported summary may claim.

    A scaled Student-t has a mean only for ``nu > 1`` and a variance only for
    ``nu > 2``; its quantiles exist for every ``nu > 0``. An undefined moment is
    disclosed and suppressed, never converted into a numerical failure and never
    papered over by capping, flooring or substituting a normal draw.

    This mapping — ``nu <= 1`` to ``MEAN_AND_VARIANCE_UNDEFINED``, ``nu <= 2`` to
    ``VARIANCE_UNDEFINED``, and no numerical failure for mathematical
    undefinedness — is a ratified TraceISO project decision. It stands on the
    moment conditions of the Student-t distribution and on this project's own
    convergence evidence, which shows the sample standard deviation failing to
    converge at ``nu <= 2`` while the central percentile interval converges
    normally at every degrees of freedom. It does not depend on any external or
    historical decision log.
    """
    if min_dof is None or not np.isfinite(min_dof):
        return MOMENT_STATUS_DEFINED
    if min_dof <= 1.0:
        return MOMENT_STATUS_MEAN_AND_VARIANCE_UNDEFINED
    if min_dof <= 2.0:
        return MOMENT_STATUS_VARIANCE_UNDEFINED
    return MOMENT_STATUS_DEFINED


def _structured_positive_count(payload: Mapping[str, object], key: str, *, engine: str) -> int:
    """Read a structured per-side count, failing closed rather than parsing a label.

    The ``"+"``-joined ``prev_std`` / ``next_std`` values are provenance text.
    Neither the count fallback nor the per-side mean fallback in
    ``engine_ssb`` is consulted by this path.
    """
    raw = payload.get(key)
    if raw is None or isinstance(raw, bool):
        raise MCCrossCheckError(
            engine=engine,
            reason_code="missing_chain_input",
            reason=f"The stored bracket has no structured {key}.",
        )
    try:
        numeric_count = float(raw)
    except (TypeError, ValueError, OverflowError):
        raise MCCrossCheckError(
            engine=engine,
            reason_code="missing_chain_input",
            reason=f"The stored bracket {key} is not an integer count.",
        ) from None
    if not np.isfinite(numeric_count) or not numeric_count.is_integer():
        raise MCCrossCheckError(
            engine=engine,
            reason_code="missing_chain_input",
            reason=f"The stored bracket {key} is not an integer count.",
        )
    count = int(numeric_count)
    if count <= 0:
        raise MCCrossCheckError(
            engine=engine,
            reason_code="missing_chain_input",
            reason=f"The stored bracket {key} is not a positive count.",
        )
    return count


def _structured_finite(payload: Mapping[str, object], key: str, *, engine: str) -> float:
    """Read a structured per-side float, failing closed when it is unusable."""
    raw = payload.get(key)
    if raw is None:
        raise MCCrossCheckError(
            engine=engine,
            reason_code="missing_chain_input",
            reason=f"The stored bracket has no structured {key}.",
        )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise MCCrossCheckError(
            engine=engine,
            reason_code="missing_chain_input",
            reason=f"The stored bracket {key} is not numeric.",
        ) from None
    if not np.isfinite(value):
        raise MCCrossCheckError(
            engine=engine,
            reason_code="missing_chain_input",
            reason=f"The stored bracket {key} is not finite.",
        )
    return value


def _structured_nonnegative(payload: Mapping[str, object], key: str, *, engine: str) -> float:
    """Read a finite structured standard error and reject negative spread."""
    value = _structured_finite(payload, key, engine=engine)
    if value < 0.0:
        raise MCCrossCheckError(
            engine=engine,
            reason_code="missing_chain_input",
            reason=f"The stored bracket {key} is negative.",
        )
    return value


def _resolve_engine_b_sample_estimate(
    sample: Sample,
    ratio_name: str,
    *,
    apply_ssb_kernel: bool,
    ratio_mask: Optional[np.ndarray],
    engine: str,
) -> Tuple[float, str, np.ndarray]:
    """Resolve the pre-SSB sample ratio estimate R_s and its source layer.

    ``ratio_mean`` is already SSB-corrected in every SSB output mode, so it is
    not R_s. R_s comes from the layer the deterministic SSB step itself
    consumed, restricted to the same runtime cycle selection that produced the
    reported mean, so the nominal replay reproduces production exactly.
    """
    selected = (
        select_best_pre_ssb_ratio_layer(sample, ratio_name)
        if apply_ssb_kernel
        else select_best_delta_ratio_layer(sample, ratio_name)
    )
    if selected is None or selected.data is None:
        raise MCCrossCheckError(
            engine=engine,
            reason_code="missing_chain_input",
            reason=f"No pre-SSB ratio layer is available for {ratio_name}.",
        )
    cycle_data = selected.data
    values = np.asarray(cycle_data.values, dtype=np.float64)
    mask = np.asarray(cycle_data.mask, dtype=bool) & np.isfinite(values)
    if ratio_mask is not None:
        runtime_mask = np.asarray(ratio_mask, dtype=bool)
        if runtime_mask.shape != mask.shape:
            # A runtime cycle selection that cannot be applied to this layer
            # would leave R_s computed over a different cycle set than the
            # reported mean. Fail closed rather than quietly disagree.
            raise MCCrossCheckError(
                engine=engine,
                reason_code="missing_chain_input",
                reason=(
                    "The runtime cycle selection does not align with the "
                    f"selected {selected.key} layer."
                ),
            )
        mask = mask & runtime_mask
    included = values[mask]
    if included.size == 0:
        raise MCCrossCheckError(
            engine=engine,
            reason_code="missing_chain_input",
            reason=f"The selected {selected.key} layer has no included cycles.",
        )
    return float(np.mean(included)), str(selected.key), included


def _hg_without_fixed_draw_replay(engine: str) -> MCCrossCheckError:
    """Refusal for an Hg-corrected Pb ratio that has no recorded bracket to replay.

    The legacy additive basis-space path knows nothing of the Hg-corrected
    chain, so running it would report a cross-check of a different model.
    """
    return MCCrossCheckError(
        engine=engine,
        reason_code="missing_chain_input",
        reason=(
            "The ratio carries an applied Hg interference correction but no recorded "
            "bracket for the fixed-draw replay; the legacy additive path does not model "
            "the Hg-corrected chain, so no cross-check is run."
        ),
    )


def _resolve_engine_b_fixed_draw_params(
    legacy_params: "_SSBPerturbationParams",
    *,
    sample: Sample,
    ratio_name: str,
    all_samples: List[Sample],
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    processing_config: Optional[ProcessingConfig],
    certified_value: Optional[CertifiedValue],
    gum_budget: Optional[UncertaintyBudget],
    ratio_mask: Optional[np.ndarray] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[EngineBFixedDrawParams]:
    """Build the Engine B fixed-draw parameter set, or report replay unavailable.

    Returns ``None`` when the structured chain inputs the replay needs do not
    exist at all, in which case the caller runs the explicitly versioned legacy
    path. Raises :class:`MCCrossCheckError` when structured bracket data are
    present but unusable — a corrupt bracket fails closed and is never patched
    up from a display label or a nominal substitute.
    """
    engine = "ssb_delta"
    hg_ratio = is_pb_hg_ssb_ratio(sample, ratio_name)
    output_mode = str(
        getattr(gum_budget, "output_mode", None) or uncertainty_config.output_mode
    )
    uses_classic = _uses_classic_delta_reference(
        sample,
        ratio_name,
        uncertainty_config=uncertainty_config,
        processing_config=processing_config,
    )
    classic_delta = output_mode == "delta" and uses_classic

    ssb_data = (getattr(sample, "ssb_results", {}) or {}).get(ratio_name)
    if not classic_delta:
        if not isinstance(ssb_data, dict) or not ssb_data:
            # No stored SSB bracket at all: the fixed-draw replay is
            # unavailable for this case rather than invented.
            if hg_ratio:
                raise _hg_without_fixed_draw_replay(engine)
            return None
        bracket_mode = (
            "block_average"
            if str(ssb_data.get("ssb_mode") or "") == "block_average"
            else "alternating"
        )
        prev_mean = _structured_finite(ssb_data, "prev_std_mean", engine=engine)
        next_mean = _structured_finite(ssb_data, "next_std_mean", engine=engine)
        prev_se = _structured_nonnegative(ssb_data, "prev_std_se", engine=engine)
        next_se = _structured_nonnegative(ssb_data, "next_std_se", engine=engine)
        prev_n = _structured_positive_count(ssb_data, "prev_n", engine=engine)
        next_n = _structured_positive_count(ssb_data, "next_n", engine=engine)
        _require_bracket_side_observations(
            prev_n=prev_n,
            next_n=next_n,
            bracket_mode=bracket_mode,
            engine=engine,
        )
        prev_label = str(ssb_data.get("prev_std") or "")
        next_label = str(ssb_data.get("next_std") or "")
    else:
        bracket = resolve_classic_delta_bracket(
            samples=all_samples,
            sample=sample,
            ratio_name=ratio_name,
            use_corrected=True,
            cycle_ranges=cycle_ranges,
        )
        if bracket.failure_code == "ambiguous_sample":
            raise MCCrossCheckError(
                engine=engine,
                reason_code="missing_chain_input",
                reason=(
                    "The classic-delta sample identity is ambiguous among "
                    "observations with the same name and run number."
                ),
            )
        if bracket.failure_code == "layer_mismatch":
            raise MCCrossCheckError(
                engine=engine,
                reason_code="missing_chain_input",
                reason=(
                    "The classic-delta sample and bracketing standards do not "
                    "share one production ratio layer "
                    f"({bracket.source_layer}, {bracket.previous_layer}, "
                    f"{bracket.following_layer})."
                ),
            )
        sides = bracket.sides
        if sides is None:
            if hg_ratio:
                raise _hg_without_fixed_draw_replay(engine)
            return None
        prev_side, next_side = sides
        bracket_mode = "classic_delta"
        prev_mean, next_mean = prev_side.mean, next_side.mean
        prev_se, next_se = prev_side.standard_error, next_side.standard_error
        prev_n, next_n = prev_side.n, next_side.n
        prev_label, next_label = prev_side.name, next_side.name
        _require_bracket_side_observations(
            prev_n=prev_n,
            next_n=next_n,
            bracket_mode=bracket_mode,
            engine=engine,
        )

    apply_ssb_kernel = not classic_delta
    delta_from_ssb = output_mode == "delta" and not classic_delta
    result_space = (
        RESULT_SPACE_DELTA_PERMIL if output_mode == "delta" else RESULT_SPACE_ABSOLUTE_RATIO
    )

    # --- C ---------------------------------------------------------------
    c_input: Optional[EngineBInput] = None
    if apply_ssb_kernel:
        certified_ref = _resolve_certified_reference_value(
            certified_value=certified_value,
            element_config=element_config,
            ratio_name=ratio_name,
        )
        if certified_ref is None:
            raise MCCrossCheckError(
                engine=engine,
                reason_code="missing_chain_input",
                reason=(
                    "The SSB model needs a positive finite certified reference "
                    "ratio and none is resolvable."
                ),
            )
        certificate_sigma = 0.0
        if certified_value is not None and certified_value.uncertainty > 0:
            certificate_sigma = float(
                u_certified_value(certified_value.uncertainty, certified_value.k)
            )
        # In SSB-derived delta the same C draw appears in the SSB numerator and
        # the delta denominator, so its uncertainty cancels algebraically. The
        # declared path is still evaluated so the cancellation is exercised
        # rather than assumed. In absolute-ratio output C carries u_crm, gated
        # on the user's enabled-contributor choice.
        c_sigma = (
            certificate_sigma
            if delta_from_ssb or legacy_params.u_crm > 0.0
            else 0.0
        )
        c_input = EngineBInput(
            name="C",
            value=float(certified_ref),
            standard_uncertainty=c_sigma,
            degrees_of_freedom=float("inf"),
            distribution="normal",
            placement=PLACEMENT_SSB_INPUT,
            type_ab="B",
            source_field="certified_value.value / u_certified_value(U, k)",
        )

    # --- R_s -------------------------------------------------------------
    r_s_value, r_s_layer, r_s_values = _resolve_engine_b_sample_estimate(
        sample,
        ratio_name,
        apply_ssb_kernel=apply_ssb_kernel,
        ratio_mask=ratio_mask,
        engine=engine,
    )
    r_s = EngineBInput(
        name="R_s",
        value=r_s_value,
        standard_uncertainty=(
            _select_u_prec_from_values(r_s_values, uncertainty_config)
            if legacy_params.u_prec > 0.0
            else 0.0
        ),
        degrees_of_freedom=float(legacy_params.u_prec_dof),
        distribution=(
            MC_DISTRIBUTION_SCALED_STUDENT_T
            if np.isfinite(float(legacy_params.u_prec_dof))
            else "normal"
        ),
        placement=PLACEMENT_SSB_INPUT if apply_ssb_kernel else PLACEMENT_DELTA_INPUT,
        type_ab="A",
        source_field=f"{r_s_layer} layer mean (runtime cycle selection)",
    )

    # --- R_p / R_n -------------------------------------------------------
    # The analytical budget forms one combined bracket term with a
    # Welch-Satterthwaite effective DoF. The replay instead draws each side
    # separately at its own nu = n - 1. The two models agree in variance to
    # first order but not in shape, and the difference is reported
    # descriptively rather than tuned away. No DoF flooring is applied here.
    side_sigma_enabled = legacy_params.u_std > 0.0
    bracket_placement = PLACEMENT_SSB_INPUT if apply_ssb_kernel else PLACEMENT_DELTA_INPUT
    r_p = EngineBInput(
        name="R_p",
        value=float(prev_mean),
        standard_uncertainty=float(prev_se) if side_sigma_enabled else 0.0,
        degrees_of_freedom=float(prev_n - 1),
        distribution=MC_DISTRIBUTION_SCALED_STUDENT_T,
        placement=bracket_placement,
        type_ab="A",
        source_field="ssb_results.prev_std_mean / prev_std_se / prev_n"
        if apply_ssb_kernel
        else "classic delta bracket side (preceding)",
    )
    r_n = EngineBInput(
        name="R_n",
        value=float(next_mean),
        standard_uncertainty=float(next_se) if side_sigma_enabled else 0.0,
        degrees_of_freedom=float(next_n - 1),
        distribution=MC_DISTRIBUTION_SCALED_STUDENT_T,
        placement=bracket_placement,
        type_ab="A",
        source_field="ssb_results.next_std_mean / next_std_se / next_n"
        if apply_ssb_kernel
        else "classic delta bracket side (following)",
    )

    # --- nominal evaluation ---------------------------------------------
    nominal_bracket = (r_p.value + r_n.value) / 2.0
    if not np.isfinite(nominal_bracket) or nominal_bracket <= 0.0:
        raise MCCrossCheckError(
            engine=engine,
            reason_code="invalid_chain_output",
            reason="The nominal bracketing mean is not positive and finite.",
        )
    if apply_ssb_kernel:
        nominal_delta_reference = float(c_input.value)  # type: ignore[union-attr]
        nominal_ssb = ssb_correct_single(
            r_s.value, r_p.value, r_n.value, c_input.value  # type: ignore[union-attr]
        )
        nominal_reported = (
            delta_from_values(nominal_ssb, c_input.value)  # type: ignore[union-attr]
            if delta_from_ssb
            else float(nominal_ssb)
        )
    else:
        nominal_delta_reference = nominal_bracket
        nominal_reported = delta_from_values(r_s.value, nominal_bracket)
    if not np.isfinite(nominal_reported):
        raise MCCrossCheckError(
            engine=engine,
            reason_code="nonfinite_model_output",
            reason="The nominal model evaluation is not finite.",
        )

    # --- output-level terms ---------------------------------------------
    # Direct model inputs are excluded structurally, so u_prec, the bracket
    # standard uncertainty, u_crm and the blank cannot be counted twice.
    kappa_distributions = legacy_params.kappa_distributions or {}
    candidate_terms: List[Tuple[str, float, str, float]] = [
        (
            "u_std_repeatability",
            float(legacy_params.u_std_repeatability),
            (
                MC_DISTRIBUTION_SCALED_STUDENT_T
                if np.isfinite(float(legacy_params.u_std_repeatability_dof))
                else "normal"
            ),
            float(legacy_params.u_std_repeatability_dof),
        ),
        (
            "u_k1_sample_decomposition",
            float(legacy_params.u_k1_sample_decomposition),
            kappa_distributions.get("u_k1_sample_decomposition", "normal"),
            float("inf"),
        ),
        (
            "u_k2_matrix_separation",
            float(legacy_params.u_k2_matrix_separation),
            kappa_distributions.get("u_k2_matrix_separation", "normal"),
            float("inf"),
        ),
        (
            "u_k3_procedural_blank",
            float(legacy_params.u_k3_procedural_blank),
            kappa_distributions.get("u_k3_procedural_blank", "normal"),
            float("inf"),
        ),
        (
            "u_k4_bracketing_standard_heterogeneity",
            float(legacy_params.u_k4_bracketing_standard_heterogeneity),
            kappa_distributions.get(
                "u_k4_bracketing_standard_heterogeneity", "normal"
            ),
            float("inf"),
        ),
        (
            "u_k5_instrumental_drift",
            float(legacy_params.u_k5_instrumental_drift),
            legacy_params.kappa_drift_distribution,
            float("inf"),
        ),
        (
            "u_k6_matrix_effects",
            float(legacy_params.u_k6_matrix_effects),
            kappa_distributions.get("u_k6_matrix_effects", "normal"),
            float("inf"),
        ),
        (
            "u_k7_residual_interferences",
            float(legacy_params.u_k7_residual_interferences),
            kappa_distributions.get("u_k7_residual_interferences", "normal"),
            float("inf"),
        ),
    ]
    custom_distributions = legacy_params.custom_contributor_distributions or {}
    for name, sigma in (legacy_params.custom_contributor_sigmas or {}).items():
        candidate_terms.append(
            (
                str(name),
                float(sigma),
                custom_distributions.get(name, "normal"),
                float("inf"),
            )
        )

    # Absolute-unit sigmas map into permil through the sensitivity coefficient
    # of the declared delta model, d(delta)/dR = 1000 / reference. This is the
    # model derivative, not a tuning factor.
    permil_sensitivity = (
        1000.0 / nominal_delta_reference
        if result_space == RESULT_SPACE_DELTA_PERMIL
        else 1.0
    )
    output_level_terms: List[EngineBOutputLevelTerm] = []
    budget_contributors = {
        str(getattr(contributor, "name", "")): contributor
        for contributor in (getattr(gum_budget, "contributors", ()) or ())
    }
    for name, sigma, distribution, dof in candidate_terms:
        if name in ENGINE_B_DIRECT_INPUT_CONTRIBUTORS:
            raise MCCrossCheckError(
                engine=engine,
                reason_code="invalid_sampled_parameter",
                reason=(
                    f"{name} is a direct model input and must not also be an "
                    "output-level term."
                ),
            )
        if not np.isfinite(sigma):
            raise MCCrossCheckError(
                engine=engine,
                reason_code="missing_chain_input",
                reason=f"The enabled contributor {name} has no usable numerical value.",
            )
        if sigma < 0.0:
            # A negative standard uncertainty has no statistical meaning.
            # Skipping it would silently drop an enabled contributor, and
            # passing it on would become a zero perturbation inside
            # _draw_standard_uncertainty. Fail closed instead.
            raise MCCrossCheckError(
                engine=engine,
                reason_code="missing_chain_input",
                reason=(
                    f"The enabled contributor {name} has a negative standard "
                    "uncertainty."
                ),
            )
        if sigma == 0.0:
            continue
        budget_contributor = budget_contributors.get(name)
        default_type_ab = "A" if name == "u_std_repeatability" else "B"
        type_ab = str(
            getattr(budget_contributor, "type_ab", default_type_ab) or default_type_ab
        ).upper()
        if type_ab not in {"A", "B"}:
            type_ab = default_type_ab
        output_level_terms.append(
            EngineBOutputLevelTerm(
                name=name,
                sigma=sigma * permil_sensitivity,
                distribution=_normalize_mc_distribution(distribution),
                degrees_of_freedom=float(dof),
                type_ab=type_ab,
                source_field=f"gum_budget.contributors[{name}]",
            )
        )

    # --- moment status ---------------------------------------------------
    drawn_dofs: List[float] = []
    for candidate in (r_s, r_p, r_n):
        if candidate.standard_uncertainty > 0.0 and np.isfinite(
            candidate.degrees_of_freedom
        ):
            drawn_dofs.append(float(candidate.degrees_of_freedom))
    for term in output_level_terms:
        if np.isfinite(term.degrees_of_freedom):
            drawn_dofs.append(float(term.degrees_of_freedom))
    min_dof = min(drawn_dofs) if drawn_dofs else None
    moment_status = _moment_status_from_dof(min_dof)

    disclosures: List[Tuple[str, str, float]] = []
    for candidate in (r_s, r_p, r_n):
        dof = (
            candidate.degrees_of_freedom
            if candidate.standard_uncertainty > 0
            else float("inf")
        )
        if np.isfinite(dof) and dof <= 2.0:
            disclosures.append((candidate.name, candidate.placement, float(dof)))
    for term in output_level_terms:
        if np.isfinite(term.degrees_of_freedom) and term.degrees_of_freedom <= 2.0:
            disclosures.append(
                (term.name, term.placement, float(term.degrees_of_freedom))
            )

    blank_fallback = float(legacy_params.u_blank_fallback)
    if not np.isfinite(blank_fallback) or blank_fallback < 0.0:
        raise MCCrossCheckError(
            engine=engine,
            reason_code="missing_chain_input",
            reason="The sample blank fallback has no usable numerical value.",
        )

    blank_blocks = tuple(legacy_params.blank_blocks or ())
    semantics_version = ENGINE_B_FIXED_DRAW_SEMANTICS
    hg_blank_terms: Tuple[EngineBLinearizedBlankTerm, ...] = ()
    hg_method = ""
    if hg_ratio:
        # Bounded Hg placement (Pb Hg plan P02): each Hg-chain term is drawn once
        # and mapped onto the reported output at the same first-order sensitivity
        # the GUM budget uses. Only terms active in that budget are drawn.
        propagation = compute_pb_hg_ssb_propagation(
            sample,
            ratio_name,
            all_samples=all_samples,
            uncertainty_config=uncertainty_config,
            processing_config=processing_config,
            cycle_ranges=cycle_ranges,
            classic_delta=classic_delta,
            runtime_sample_mask=ratio_mask,
        )
        # d(reported)/d(lnY): Y for an absolute ratio, delta + 1000 for delta.
        reported_scale = abs(
            float(nominal_reported) + 1000.0
            if result_space == RESULT_SPACE_DELTA_PERMIL
            else float(nominal_reported)
        )
        active_names = {
            name
            for name, contributor in budget_contributors.items()
            if getattr(contributor, "is_active", False)
        }
        # The legacy sample-only two-channel blank placement does not describe
        # the Hg-corrected chain. It is replaced, never combined with it.
        blank_blocks = ()
        blank_fallback = 0.0
        if "u_blank" in active_names:
            if propagation is None or propagation.blank_status == BLANK_INVALID_COVARIANCE:
                raise MCCrossCheckError(
                    engine=engine,
                    reason_code="invalid_covariance",
                    reason=(
                        propagation.blank_reason
                        if propagation is not None
                        else "The Hg-corrected blank model is unavailable."
                    ),
                )
            if propagation.u_blank_rel is None:
                raise MCCrossCheckError(
                    engine=engine,
                    reason_code="missing_chain_input",
                    reason=f"The Hg-corrected blank model is unavailable: {propagation.blank_reason}",
                )
            hg_blank_terms = tuple(
                EngineBLinearizedBlankTerm(
                    blank_observation_id=blank.blank_observation_id,
                    channels=tuple(blank.channels),
                    mean_vector=np.array(blank.mean_vector, dtype=np.float64, copy=True),
                    covariance_matrix=np.array(blank.covariance_matrix, dtype=np.float64, copy=True),
                    gradient=np.asarray(blank.gradient_rel, dtype=np.float64) * reported_scale,
                )
                for blank in propagation.blank_inputs
            )
        for name, u_rel in (
            (U_INTERF, None if propagation is None else propagation.u_interf_rel),
            (U_HG_TL_REFERENCE, None if propagation is None else propagation.u_tl_reference_rel),
        ):
            if name not in active_names:
                continue
            if u_rel is None or not np.isfinite(u_rel) or u_rel < 0.0:
                raise MCCrossCheckError(
                    engine=engine,
                    reason_code="missing_chain_input",
                    reason=f"The enabled contributor {name} has no usable sensitivity or reference uncertainty.",
                )
            if u_rel == 0.0:
                continue
            output_level_terms.append(
                EngineBOutputLevelTerm(
                    name=name,
                    sigma=float(u_rel) * reported_scale,
                    distribution="normal",
                    degrees_of_freedom=float("inf"),
                    type_ab="B",
                    source_field=f"{PB_HG_SSB_PROPAGATION_METHOD}.{name} (first-order sensitivity)",
                )
            )
        hg_method = PB_HG_SSB_PROPAGATION_METHOD
        semantics_version = ENGINE_B_FIXED_DRAW_PB_HG_SEMANTICS

    if not hg_ratio:
        from domain.uncertainty.joint_blank import ordinary_ssb_blank
        blank_blocks, blank_fallback = (), 0.0
        blank_row = budget_contributors.get("u_blank")
        if blank_row is not None and blank_row.is_active:
            try:
                joint = ordinary_ssb_blank(sample, ratio_name, all_samples, uncertainty_config,
                    processing_config, cycle_ranges, classic_delta, ratio_mask)
            except ValueError as exc:
                raise MCCrossCheckError(engine=engine, reason_code="missing_chain_input", reason=str(exc)) from exc
            scale = abs(float(nominal_reported) + 1000 if result_space == RESULT_SPACE_DELTA_PERMIL else float(nominal_reported))
            hg_blank_terms = tuple(EngineBLinearizedBlankTerm(b.blank_observation_id, b.channels,
                b.mean_vector.copy(), b.covariance_matrix.copy(), b.gradient_rel * scale) for b in joint.blank_inputs)
        hg_method = "ordinary_ssb_joint_blank_linear.v1" if hg_blank_terms else ""
        semantics_version = ENGINE_B_FIXED_DRAW_SEMANTICS

    if blank_blocks and blank_fallback > 0.0:
        raise MCCrossCheckError(
            engine=engine,
            reason_code="invalid_sampled_parameter",
            reason=(
                "Detailed sample blank blocks and the output-level blank "
                "fallback cannot both be active."
            ),
        )

    blank_c_num = 0.0
    blank_c_den = 0.0
    if blank_blocks:
        denominator_scale = max(
            abs(float(legacy_params.num_corrected_mean)),
            abs(float(legacy_params.den_corrected_mean)),
        )
        safe_blank_denominator = _require_safe_denominator(
            legacy_params.den_corrected_mean,
            nominal_scale=denominator_scale,
            engine=engine,
            what="The blank-sensitivity denominator intensity",
        )
        if safe_blank_denominator <= 0.0:
            raise MCCrossCheckError(
                engine=engine,
                reason_code="invalid_sampled_parameter",
                reason="The blank-sensitivity denominator intensity is not positive.",
            )
        blank_c_num = -1.0 / safe_blank_denominator
        blank_c_den = (
            legacy_params.num_corrected_mean / (safe_blank_denominator ** 2)
        )

    return EngineBFixedDrawParams(
        semantics_version=semantics_version,
        result_space=result_space,
        output_mode=output_mode,
        bracket_mode=bracket_mode,
        sample_name=str(sample.name),
        ratio_name=str(ratio_name),
        r_s=r_s,
        r_p=r_p,
        r_n=r_n,
        c=c_input,
        apply_ssb_kernel=apply_ssb_kernel,
        delta_from_ssb=delta_from_ssb,
        prev_std_label=prev_label,
        next_std_label=next_label,
        output_level_terms=tuple(output_level_terms),
        blank_blocks=blank_blocks,
        blank_c_num=blank_c_num,
        blank_c_den=blank_c_den,
        blank_uncertainty_input=str(
            getattr(uncertainty_config, "blank_uncertainty_input", "sd")
        ),
        blank_placement=PLACEMENT_SSB_INPUT,
        u_blank_fallback=blank_fallback * permil_sensitivity,
        blank_fallback_distribution="normal",
        blank_fallback_dof=float("inf"),
        nominal_reported_value=float(nominal_reported),
        nominal_bracket_mean=float(nominal_bracket),
        nominal_delta_reference=float(nominal_delta_reference),
        min_type_a_dof=min_dof,
        moment_status=moment_status,
        low_dof_disclosures=tuple(disclosures),
        hg_blank_terms=hg_blank_terms,
        hg_propagation_method=hg_method,
    )


def _draw_engine_b_input(rng: np.random.Generator, spec: EngineBInput) -> float:
    """Draw one direct model input about its estimate."""
    return spec.value + _draw_standard_uncertainty(
        rng,
        spec.standard_uncertainty,
        spec.distribution,
        degrees_of_freedom=spec.degrees_of_freedom,
    )


def _mc_iteration_engine_b_fixed_draw(
    params: EngineBFixedDrawParams,
    rng: np.random.Generator,
) -> float:
    """Run one Engine B fixed draw and return the final reported quantity.

    Samples R_s, R_p, R_n and C, applies the supported sample blank
    perturbation at the R_s boundary, recalculates the bracketing mean, the SSB
    factor and the requested absolute-ratio or delta output through the
    production numerical functions, then adds each approved output-level
    contributor exactly once.
    """
    p = params
    engine = "ssb_delta"

    r_s = _draw_engine_b_input(rng, p.r_s)

    # Sample blank perturbation, placed on R_s before the SSB calculation.
    # Detailed blocks and the output-level fallback are mutually exclusive.
    if p.blank_blocks:
        delta_num = 0.0
        delta_den = 0.0
        n_blocks = float(len(p.blank_blocks))
        for block in p.blank_blocks:
            sampled = _sample_multivariate(
                block.mean_vector,
                block.covariance_matrix,
                rng,
            )
            delta_num += (sampled[0] - block.mean_vector[0]) / n_blocks
            delta_den += (sampled[1] - block.mean_vector[1]) / n_blocks
        r_s += p.blank_c_num * delta_num + p.blank_c_den * delta_den

    r_s = _require_finite(
        r_s,
        engine=engine,
        reason_code="invalid_sampled_parameter",
        what="The sampled sample ratio R_s",
    )

    r_p = _draw_engine_b_input(rng, p.r_p)
    r_n = _draw_engine_b_input(rng, p.r_n)
    bracket_mean = _require_safe_denominator(
        (r_p + r_n) / 2.0,
        nominal_scale=p.nominal_bracket_mean,
        engine=engine,
        what="The sampled bracketing mean B",
    )
    if bracket_mean <= 0.0:
        raise MCCrossCheckError(
            engine=engine,
            reason_code="invalid_sampled_parameter",
            reason="The sampled bracketing mean B is not positive.",
        )

    if p.apply_ssb_kernel:
        c_draw = _draw_engine_b_input(rng, p.c)  # type: ignore[arg-type]
        if not np.isfinite(c_draw) or c_draw <= 0.0:
            raise MCCrossCheckError(
                engine=engine,
                reason_code="invalid_sampled_parameter",
                reason="The sampled certified reference ratio C is not positive and finite.",
            )
        r_ssb = ssb_correct_single(r_s, r_p, r_n, c_draw)
        if p.delta_from_ssb:
            # The same C draw appears in the SSB numerator and this
            # denominator, so certified-reference uncertainty cancels here and
            # remains only in absolute-ratio output.
            _require_safe_denominator(
                c_draw,
                nominal_scale=p.nominal_delta_reference,
                engine=engine,
                what="The sampled delta reference C",
            )
            value = delta_from_values(r_ssb, c_draw)
        else:
            value = float(r_ssb)
    else:
        value = delta_from_values(r_s, bracket_mean)

    value = _require_finite(
        value,
        engine=engine,
        reason_code="nonfinite_model_output",
        what="The recalculated reported value",
    )

    # Hg-corrected Pb blank observations: one validated multivariate draw each,
    # mapped onto the reported value at its first-order sensitivity.
    for blank_term in p.hg_blank_terms:
        sampled = _sample_multivariate(
            blank_term.mean_vector, blank_term.covariance_matrix, rng,
        )
        value += float(np.dot(blank_term.gradient, sampled - blank_term.mean_vector))

    for term in p.output_level_terms:
        value += _draw_standard_uncertainty(
            rng,
            term.sigma,
            term.distribution,
            degrees_of_freedom=term.degrees_of_freedom,
        )

    if p.u_blank_fallback > 0.0:
        value += _draw_standard_uncertainty(
            rng,
            p.u_blank_fallback,
            p.blank_fallback_distribution,
            degrees_of_freedom=p.blank_fallback_dof,
        )

    return float(value)


def _mc_iteration_internal(
    params: _InternalPerturbationParams,
    rng: np.random.Generator,
) -> float:
    """Run one MC iteration for Engine A (Sr internal normalisation)."""
    p = params

    corrected = {
        isotope: values.copy()
        for isotope, values in p.base_corrected_intensities.items()
    }

    if p.blank_blocks:
        blank_deltas = {
            isotope: 0.0
            for isotope in p.blank_blocks[0].isotopes
        }
        n_blocks = float(len(p.blank_blocks))
        for block in p.blank_blocks:
            sampled = _sample_multivariate(
                block.mean_vector,
                block.covariance_matrix,
                rng,
            )
            for index, isotope in enumerate(block.isotopes):
                weight = block.channel_weights[index] if getattr(block, "channel_weights", ()) else 1 / n_blocks
                blank_deltas[isotope] = blank_deltas.get(isotope, 0.0) + (sampled[index] - block.mean_vector[index]) * weight

        for isotope, delta in blank_deltas.items():
            if isotope in corrected:
                corrected[isotope] = corrected[isotope] - delta

    normalization_value = p.normalization_value
    if p.mode_b_chain_enabled and p.normalization_value_sd > 0.0:
        normalization_value = rng.normal(
            p.normalization_value,
            p.normalization_value_sd,
        )
        if not np.isfinite(normalization_value) or normalization_value <= 0.0:
            raise MCCrossCheckError(
                engine="internal_normalization",
                reason_code="invalid_sampled_parameter",
                reason="A sampled normalization reference value was not positive and finite.",
            )

    rb87_rb85 = p.reference_inputs.rb87_rb85
    kr86_kr83 = p.reference_inputs.kr86_kr83
    if p.interference_chain_enabled:
        if (
            "87Rb" in p.enabled_interferents
            and "85Rb" in corrected
            and p.reference_inputs.u_rel_rb is not None
        ):
            rb87_rb85 = rng.normal(
                p.reference_inputs.rb87_rb85,
                p.reference_inputs.rb87_rb85 * p.reference_inputs.u_rel_rb,
            )
            if not np.isfinite(rb87_rb85) or rb87_rb85 <= 0.0:
                raise MCCrossCheckError(
                    engine="internal_normalization",
                    reason_code="invalid_sampled_parameter",
                    reason="A sampled Rb reference ratio was not positive and finite.",
                )
        if "86Kr" in p.enabled_interferents and "83Kr" in corrected:
            kr86_kr83 = rng.normal(
                p.reference_inputs.kr86_kr83,
                p.reference_inputs.kr86_kr83 * p.reference_inputs.u_rel_kr86,
            )
            if not np.isfinite(kr86_kr83) or kr86_kr83 <= 0.0:
                raise MCCrossCheckError(
                    engine="internal_normalization",
                    reason_code="invalid_sampled_parameter",
                    reason="A sampled Kr reference ratio was not positive and finite.",
                )

    required_chain_isotopes = {
        "87Sr",
        "86Sr",
        p.reference_inputs.normalization_numerator,
        p.reference_inputs.normalization_denominator,
    }
    if "87Rb" in p.enabled_interferents:
        required_chain_isotopes.add("85Rb")
    if "86Kr" in p.enabled_interferents:
        required_chain_isotopes.add("83Kr")
    can_run_chain = (
        normalization_value > 0.0
        and required_chain_isotopes.issubset(corrected.keys())
    )
    if not can_run_chain:
        missing = sorted(required_chain_isotopes.difference(corrected.keys()))
        detail = f" Missing isotopes: {', '.join(missing)}." if missing else ""
        raise MCCrossCheckError(
            engine="internal_normalization",
            reason_code="missing_chain_input",
            reason=f"The Sr correction chain is missing required inputs.{detail}",
        )
    replayed_result = _run_sr_correction_chain(
        intensities=corrected,
        normalization_value=normalization_value,
        mask=p.ratio_mask,
        reference_inputs=p.reference_inputs,
        rb87_rb85=rb87_rb85,
        kr86_kr83=kr86_kr83,
        apply_iif=p.apply_iif,
        apply_interference=p.apply_interference,
        enabled_interferents=p.enabled_interferents,
    )
    # post_chain_scale maps IIF-space -> reported space (anchoring + drift).
    chain_result = replayed_result * p.post_chain_scale
    if not np.isfinite(chain_result) or chain_result == 0.0:
        raise MCCrossCheckError(
            engine="internal_normalization",
            reason_code="invalid_chain_output",
            reason="The Sr correction chain returned zero or a non-finite result.",
        )

    result = chain_result

    if p.u_prec > 0.0:
        result += _draw_standard_uncertainty(
            rng,
            p.u_prec,
            MC_DISTRIBUTION_SCALED_STUDENT_T,
            degrees_of_freedom=p.u_prec_dof,
        )

    if p.u_norm_ratio_mode_b_fallback > 0.0:
        result += rng.normal(0.0, p.u_norm_ratio_mode_b_fallback)

    for sigma in (
        p.u_std_repeatability,
        p.u_bias_qc,
        p.u_reprod_dig,
        p.u_crm,
        p.u_ref_value,
        p.u_interf_fallback,
        p.u_blank_fallback,
    ):
        if sigma > 0.0:
            result += rng.normal(0.0, sigma)

    if p.u_kappa_drift > 0.0:
        result += _draw_standard_uncertainty(
            rng,
            p.u_kappa_drift,
            p.kappa_drift_distribution,
        )
    if AUTOMATIC_REFERENCE_BIAS_ENABLED and p.u_bias_ref > 0.0:  # pragma: no cover
        # Reconstructs the rectangular half-width the disabled automatic model
        # implied. Unreachable while that model is off, and guarded so it
        # cannot be reached by a stale stored magnitude either.
        half_width = p.u_bias_ref * np.sqrt(3.0)
        result += rng.uniform(-half_width, half_width)
    # Custom contributor additive perturbations
    if p.custom_contributor_sigmas:
        distributions = p.custom_contributor_distributions or {}
        for name, sigma in p.custom_contributor_sigmas.items():
            result += _draw_standard_uncertainty(
                rng,
                sigma,
                distributions.get(name, "normal"),
            )

    return result


def _extract_pb_tl_perturbation_params(
    sample: Sample,
    ratio_name: str,
    ratio_mean: float,
    all_samples: List[Sample],
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    processing_config: ProcessingConfig,
    *,
    gum_budget: UncertaintyBudget,
    ratio_mask: Optional[np.ndarray] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    custom_contributor_library: Optional[Dict[str, List[CustomUncertaintyContributor]]] = None,
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
) -> _PbTlPerturbationParams:
    """Extract perturbation parameters for one Pb-Tl MC iteration."""
    from domain.uncertainty.engine_external_pb_tl import (
        _get_pb_tl_base_intensities,
        _resolve_blank_perturbation_isotopes,
        _resolve_pb_tl_ratio_mask,
        _resolve_pb_tl_reference_inputs,
        build_pb_tl_blank_blocks,
        resolve_pb_tl_normalization_standard_uncertainty,
    )

    _custom_names = {
        d.name
        for defs in (custom_contributor_library or {}).values()
        for d in defs
    }

    def _enabled(name: str) -> bool:
        return is_contributor_active(
            name=name,
            uncertainty_config=uncertainty_config,
            sample=sample,
            element_symbol=element_config.symbol,
            custom_contributor_names=_custom_names,
            profile_defaults=profile_defaults,
        )

    reference_inputs = _resolve_pb_tl_reference_inputs(
        ratio_name,
        processing_config=processing_config,
        element_config=element_config,
    )
    if ratio_mask is None:
        ratio_mask = _resolve_pb_tl_ratio_mask(
            sample, ratio_name, cycle_ranges=cycle_ranges
        )

    base_intensities = _get_pb_tl_base_intensities(sample)
    apply_hg = bool(getattr(processing_config, "apply_hg_interference_correction", False))

    # Blank chain perturbation is gated on the u_blank contributor: a disabled
    # u_blank must drop the blank term from MC exactly as it drops from the GUM
    # budget (matches Engine A).
    blank_blocks: Tuple[object, ...] = ()
    if _enabled("u_blank"):
        isotopes = _resolve_blank_perturbation_isotopes(
            base_intensities, ratio_name, apply_hg=apply_hg
        )
        blank_blocks = tuple(
            build_pb_tl_blank_blocks(
                sample,
                all_samples,
                isotopes=isotopes,
                uncertainty_config=uncertainty_config,
                processing_config=processing_config,
                cycle_ranges=cycle_ranges,
            )
        )

    # 1-sigma draw width for the Tl normalization ratio (gated on
    # u_norm_ref).
    tl_norm_std = 0.0
    if _enabled("u_norm_ref"):
        try:
            tl_norm_std = resolve_pb_tl_normalization_standard_uncertainty(
                uncertainty_config,
                reference_inputs.tl_norm_ratio_name,
            )
        except ValueError:
            tl_norm_std = 0.0

    # 1-sigma draw width for the 204Hg/202Hg assigned correction ratio
    # (chain-replay perturbation), gated on u_interf. The deterministic Hg
    # correction still runs via apply_interference; only the random
    # perturbation is gated. When the assigned ratio's uncertainty is
    # unassigned the width stays 0 and no draw is made - the quantity is
    # omitted, not asserted exact.
    hg_std = 0.0
    hg_interf_chain_enabled = False
    hg_perturbation_expected = (
        _enabled("u_interf")
        and apply_hg
        and "204Pb" in ratio_name
        and reference_inputs.hg204_hg202_natural is not None
    )
    if hg_perturbation_expected:
        try:
            hg_std = float(
                reference_inputs.hg204_hg202_natural
                * require_natural_ratio_relative_uncertainty("Hg", "204Hg/202Hg")
            )
            hg_interf_chain_enabled = hg_std > 0.0
        except ValueError:
            pass

    def _sigma(name: str) -> float:
        if gum_budget is None:
            return 0.0
        c = gum_budget._find_contributor(name)
        if c is None or not c.is_active:
            return 0.0
        return float(c.value_abs)

    def _dof(name: str) -> float:
        if gum_budget is None:
            return float("inf")
        c = gum_budget._find_contributor(name)
        if c is None or not c.is_active:
            return float("inf")
        try:
            value = float(c.degrees_of_freedom)
        except (TypeError, ValueError):
            return float("inf")
        return value if value > 0.0 else float("inf")

    u_prec = _sigma("u_prec")
    u_prec_dof = _dof("u_prec")
    if u_prec <= 0.0 and _enabled("u_prec"):
        ratio_cd = get_best_ratio_data(sample, ratio_name)
        if ratio_cd is not None:
            values = np.asarray(ratio_cd.values, dtype=np.float64)
            mask = np.asarray(ratio_cd.mask, dtype=bool)
            if ratio_mask is not None and len(ratio_mask) == len(values):
                mask = mask & np.asarray(ratio_mask, dtype=bool)
            u_prec = _select_u_prec_from_values(
                values[mask],
                uncertainty_config,
            )
            u_prec_dof = float(max(int(np.sum(mask)) - 1, 1))

    # Custom contributor perturbation sigmas (draws happen per-iteration in
    # _mc_iteration_pb_tl), mirroring Engines A and B.
    custom_contributor_sigmas, custom_contributor_distributions, custom_contributor_estimation_dofs = _extract_custom_contributor_params(
        sample=sample,
        element_config=element_config,
        ratio_mean=ratio_mean,
        custom_contributor_library=custom_contributor_library,
    )

    for name in ("u_bias_qc", "u_reprod_dig"):
        sigma = _sigma(name)
        if sigma > 0.0:
            custom_contributor_sigmas[name] = sigma
            custom_contributor_distributions[name] = "normal"

    return _PbTlPerturbationParams(
        ratio_mean=float(ratio_mean),
        ratio_name=ratio_name,
        ratio_mask=ratio_mask,
        post_chain_scale=resolve_output_scale(
            sample,
            ratio_name,
            input_layer=INPUT_LAYER_NORMALIZED,
        ).factor,
        base_intensities=base_intensities,
        apply_interference=apply_hg,
        apply_iif=bool(getattr(processing_config, "apply_mass_bias_correction", False)),
        tl_norm_value=float(reference_inputs.tl_norm_value),
        tl_norm_std=float(tl_norm_std),
        u_prec=float(u_prec),
        u_prec_dof=float(u_prec_dof),
        u_std_repeatability=_sigma("u_std_repeatability"),
        u_interf=_sigma("u_interf"),
        u_crm=_sigma("u_crm"),
        u_kappa_drift=_sigma("u_kappa_drift"),
        kappa_drift_distribution=_normalize_mc_distribution(
            getattr(uncertainty_config, "kappa_drift_distribution", "normal")
        ),
        blank_blocks=tuple(blank_blocks),
        reference_inputs=reference_inputs,
        hg_std=hg_std,
        hg_interf_chain_enabled=hg_interf_chain_enabled,
        hg_perturbation_expected=hg_perturbation_expected,
        custom_contributor_sigmas=custom_contributor_sigmas,
        custom_contributor_distributions=custom_contributor_distributions,
        custom_contributor_estimation_dofs=custom_contributor_estimation_dofs,
    )


def _mc_iteration_pb_tl(
    params: _PbTlPerturbationParams,
    rng: np.random.Generator,
) -> float:
    """One Pb-Tl Monte Carlo iteration via deterministic chain replay."""
    from domain.uncertainty.engine_external_pb_tl import _run_pb_tl_correction_chain

    intensities = {
        iso: arr.copy() for iso, arr in params.base_intensities.items()
    }

    # Replay perturbation #1 — blank: shift each block's isotopes by a draw
    # from the block covariance (corrected = raw - blank ⇒ +δ blank ≡ -δ corr).
    n_blank_blocks = float(len(params.blank_blocks)) if params.blank_blocks else 1.0
    for block in params.blank_blocks:
        isos = list(block.isotopes)
        if not isos:
            continue
        cov = np.asarray(block.covariance_matrix, dtype=np.float64)
        draw = _sample_multivariate(np.zeros(len(isos)), cov, rng)
        for index, (iso, delta) in enumerate(zip(isos, draw)):
            if iso in intensities:
                weight = block.channel_weights[index] if block.channel_weights else 1 / n_blank_blocks
                intensities[iso] = intensities[iso] - float(delta) * weight

    # Replay perturbation #2 — Tl normalization ratio.
    tl_norm = params.tl_norm_value
    if params.tl_norm_std > 0:
        tl_norm = float(rng.normal(params.tl_norm_value, params.tl_norm_std))
        if not np.isfinite(tl_norm) or tl_norm <= 0.0:
            raise MCCrossCheckError(
                engine="pb_tl_external_normalization",
                reason_code="invalid_sampled_parameter",
                reason="A sampled Tl normalization reference value was not positive and finite.",
            )

    # Replay perturbation #3 — 204Hg/202Hg assigned correction ratio for Hg
    # interference.
    # When the chain handles this, u_interf is NOT added as an independent term
    # below to avoid double-counting.
    reference_inputs = params.reference_inputs
    if params.hg_interf_chain_enabled and params.hg_std > 0.0:
        hg_nominal = params.reference_inputs.hg204_hg202_natural  # type: ignore[union-attr]
        perturbed_hg = float(rng.normal(float(hg_nominal), params.hg_std))
        if not np.isfinite(perturbed_hg) or perturbed_hg <= 0.0:
            raise MCCrossCheckError(
                engine="pb_tl_external_normalization",
                reason_code="invalid_sampled_parameter",
                reason="A sampled Hg reference ratio was not positive and finite.",
            )
        reference_inputs = replace(params.reference_inputs, hg204_hg202_natural=perturbed_hg)  # type: ignore[call-overload]

    chain_out = _run_pb_tl_correction_chain(
        intensities,
        tl_norm,
        params.ratio_name,
        params.ratio_mask,
        reference_inputs=reference_inputs,
        apply_interference=params.apply_interference,
        apply_iif=params.apply_iif,
    )
    if not np.isfinite(chain_out) or chain_out == 0.0:
        raise MCCrossCheckError(
            engine="pb_tl_external_normalization",
            reason_code="invalid_chain_output",
            reason="The Pb-Tl correction chain returned zero or a non-finite result.",
        )

    # Additive output-level draws (independent contributors).
    result = float(chain_out) * params.post_chain_scale
    if params.u_prec > 0:
        result += _draw_standard_uncertainty(
            rng,
            params.u_prec,
            MC_DISTRIBUTION_SCALED_STUDENT_T,
            degrees_of_freedom=params.u_prec_dof,
        )
    if params.u_std_repeatability > 0:
        result += float(rng.normal(0.0, params.u_std_repeatability))
    # u_interf: only add as an independent draw when the chain did not replay it.
    if not params.hg_interf_chain_enabled and params.u_interf > 0:
        result += float(rng.normal(0.0, params.u_interf))
    if params.u_crm > 0:
        result += float(rng.normal(0.0, params.u_crm))
    if params.u_kappa_drift > 0:
        result += _draw_standard_uncertainty(
            rng,
            params.u_kappa_drift,
            params.kappa_drift_distribution,
        )

    # Custom contributor additive perturbations (mirrors Engines A and B).
    if params.custom_contributor_sigmas:
        distributions = params.custom_contributor_distributions or {}
        for name, sigma in params.custom_contributor_sigmas.items():
            result += _draw_standard_uncertainty(
                rng,
                sigma,
                distributions.get(name, "normal"),
            )

    return result


@dataclass(frozen=True, eq=False)
class _PbCalibratedMCParams:
    """Frozen inputs of the extended Engine C draw for one calibrated absolute ratio (C05).

    ``model`` is the :class:`~domain.uncertainty.engine_pb_calibrated.CalibratedModel`
    whose nominal replay was verified against the processed calibration. Only
    contributors active in the GUM budget are drawn, each at its one placement.
    """

    model: Any
    applied_mode: str
    nominal_reported_value: float
    draw_target: bool
    target_sigma: float
    target_dof: float
    draw_members: bool
    member_sigmas: Tuple[float, ...]
    member_dofs: Tuple[float, ...]
    draw_c: bool
    c_sigma: float
    draw_tl: bool
    tl_sigma: float
    draw_hg: bool
    hg_sigma: float
    #: Blank inputs drawn once per iteration; validated by the covariance preflight.
    blank_blocks: Tuple[Any, ...]
    custom_sigmas: Dict[str, float]
    custom_distributions: Dict[str, str]
    custom_estimation_dofs: Dict[str, float]
    input_digest: str
    min_type_a_dof: Optional[float]
    warnings: Tuple[str, ...]
    specs: Tuple["EngineBContributorSpec", ...]
    blank_uncertainty_input: str
    blank_placement: str

    def contributor_specs(self) -> Tuple["EngineBContributorSpec", ...]:
        return self.specs

    def contributor_placements(self) -> Tuple[Tuple[str, str], ...]:
        return tuple((spec.name, spec.placement) for spec in self.specs)


def _resolve_pb_calibrated_mc_params(
    sample: Sample,
    ratio_name: str,
    *,
    resolved: str,
    all_samples: List[Sample],
    element_config: ElementConfig,
    uncertainty_config: UncertaintyConfig,
    processing_config: Optional[ProcessingConfig],
    gum_budget: UncertaintyBudget,
    ratio_mask: Optional[np.ndarray],
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]],
    custom_contributor_library: Optional[Dict[str, List[CustomUncertaintyContributor]]],
) -> _PbCalibratedMCParams:
    """Freeze the calibrated model and decide each contributor's single draw placement."""
    from domain.uncertainty.engine_pb_calibrated import (
        U_BLANK,
        U_INTERF,
        U_NORM_REF,
        U_PREC,
        U_REFERENCE,
        U_STD_PRECISION,
        CalibratedModelRefusal,
        calibration_identity_for_record,
        model_input_digest,
        resolve_calibrated_model,
    )
    from domain.uncertainty.scope import budget_scope_note, is_invalid_budget_scope

    if resolved != "pb_tl_external_normalization" or processing_config is None:
        raise MCCrossCheckError(
            engine=resolved, reason_code="missing_chain_input",
            reason="A Pb-standard-calibrated result is modelled only on the Pb-Tl route with its processing configuration.",
        )
    if (
        gum_budget is None
        or str(getattr(gum_budget, "engine", "") or "") != PB_CALIBRATION_BUDGET_ENGINE
        or is_invalid_budget_scope(gum_budget)
    ):
        note = budget_scope_note(gum_budget) if gum_budget is not None else "no budget was supplied"
        raise MCCrossCheckError(
            engine=resolved, reason_code="missing_chain_input",
            reason=f"The calibrated absolute-ratio budget is not available ({note}); there is nothing to cross-check.",
        )

    def active(name: str) -> bool:
        contributor = gum_budget._find_contributor(name)
        return contributor is not None and contributor.is_active

    try:
        model = resolve_calibrated_model(
            sample, ratio_name, all_samples=all_samples, element_config=element_config,
            uncertainty_config=uncertainty_config, processing_config=processing_config,
            cycle_ranges=cycle_ranges, ratio_mask=ratio_mask, include_blanks=active(U_BLANK),
        )
    except CalibratedModelRefusal as refusal:
        raise MCCrossCheckError(
            engine=resolved,
            reason_code="invalid_covariance" if refusal.code == "invalid_blank_covariance" else "missing_chain_input",
            reason=str(refusal),
        ) from refusal
    budget_value = float(getattr(gum_budget, "ratio_value", 0.0) or 0.0)
    if (
        model.target.n_valid != int(getattr(gum_budget, "n_cycles", 0) or 0)
        or not np.isfinite(model.y)
        or abs(model.y - budget_value) > 1e-9 * max(abs(budget_value), np.finfo(float).tiny)
    ):
        raise MCCrossCheckError(
            engine=resolved, reason_code="missing_chain_input",
            reason="The supplied budget describes a different calibrated cycle support than the requested one.",
        )

    specs: List[EngineBContributorSpec] = []
    dofs: List[float] = []
    warnings: List[str] = []
    draw_target = active(U_PREC) and model.target_se is not None and model.target_se > 0.0
    target_dof = float(model.target.n_valid - 1)
    if draw_target:
        specs.append(EngineBContributorSpec(
            U_PREC, PLACEMENT_TARGET_MEAN, MC_DISTRIBUTION_SCALED_STUDENT_T, target_dof, "A", "target.x_mean",
        ))
        dofs.append(target_dof)
    member_sigmas = tuple(float(m.se) for m in model.members)
    member_dofs = tuple(float(m.n_valid - 1) for m in model.members)
    draw_members = active(U_STD_PRECISION)
    if draw_members:
        for member, dof in zip(model.members, member_dofs):
            specs.append(EngineBContributorSpec(
                U_STD_PRECISION, PLACEMENT_STANDARD_MEAN, MC_DISTRIBUTION_SCALED_STUDENT_T, dof, "A",
                f"members[{member.observation_id}].s_mean",
            ))
            dofs.append(dof)
    draw_c = active(U_REFERENCE) and model.u_c is not None and model.u_c > 0.0
    if draw_c:
        specs.append(EngineBContributorSpec(
            U_REFERENCE, PLACEMENT_CALIBRATION_REFERENCE, "normal", float("inf"), "B", "record.reference.value",
        ))
    draw_tl = active(U_NORM_REF) and model.apply_hg and model.u_r_tl is not None and model.u_r_tl > 0.0
    if draw_tl:
        specs.append(EngineBContributorSpec(
            U_NORM_REF, PLACEMENT_SHARED_CHAIN, "normal", float("inf"), "B", "reference_inputs.tl_norm_value",
        ))
    draw_hg = active(U_INTERF) and model.apply_hg and model.u_r_hg is not None and model.u_r_hg > 0.0
    if draw_hg:
        specs.append(EngineBContributorSpec(
            U_INTERF, PLACEMENT_SHARED_CHAIN, "normal", float("inf"), "B", "reference_inputs.hg204_hg202_natural",
        ))
    blanks = tuple(model.blanks) if active(U_BLANK) else ()
    if blanks:
        specs.append(EngineBContributorSpec(
            U_BLANK, PLACEMENT_SHARED_CHAIN, "multivariate_normal", float("inf"), "A", f"blanks[{len(blanks)}]",
        ))
    custom_sigmas, custom_distributions, custom_estimation_dofs = _extract_custom_contributor_params(
        sample=sample, element_config=element_config, ratio_mean=model.y,
        custom_contributor_library=custom_contributor_library,
    )
    for name in ("u_bias_qc", "u_reprod_dig"):
        material_row = gum_budget._find_contributor(name)
        if active(name) and material_row is not None and material_row.value_abs > 0.0:
            custom_sigmas[name] = float(material_row.value_abs)
            custom_distributions[name] = "normal"
    custom_sigmas = {name: sigma for name, sigma in custom_sigmas.items() if active(name)}
    custom_distributions = {name: custom_distributions[name] for name in custom_sigmas}
    for name in sorted(custom_sigmas):
        row = gum_budget._find_contributor(name)
        specs.append(EngineBContributorSpec(
            name, PLACEMENT_OUTPUT_LEVEL, custom_distributions[name], float("inf"),
            str(getattr(row, "type_ab", "B") or "B"), "gum_budget.contributors",
        ))

    for name, dof in [(U_PREC, target_dof)] * draw_target + [
        (U_STD_PRECISION, d) for d in member_dofs if draw_members
    ]:
        if dof <= 2.0:
            warnings.append(f"{name} is drawn at {dof:g} degrees of freedom, where the Student-t variance is undefined.")
    for contributor in getattr(gum_budget, "contributors", ()) or ():
        if not contributor.is_active and contributor.state in {"MISSING_DATA", "NO_APPROVED_MODEL"}:
            warnings.append(f"{contributor.name} is not drawn ({contributor.state}): {contributor.inactive_reason}")

    drawn = {
        "target": (draw_target, float(model.target_se or 0.0).hex(), target_dof),
        "members": (draw_members, tuple(float(s).hex() for s in member_sigmas), member_dofs),
        "c": (draw_c, float(model.u_c or 0.0).hex()),
        "tl": (draw_tl, float(model.u_r_tl or 0.0).hex()),
        "hg": (draw_hg, float(model.u_r_hg or 0.0).hex()),
        "blanks": bool(blanks),
        "custom": tuple(sorted((n, float(s).hex(), custom_distributions[n]) for n, s in custom_sigmas.items())),
    }
    return _PbCalibratedMCParams(
        model=model, applied_mode=model.applied_mode, nominal_reported_value=float(model.y),
        draw_target=draw_target, target_sigma=float(model.target_se or 0.0), target_dof=target_dof,
        draw_members=draw_members, member_sigmas=member_sigmas, member_dofs=member_dofs,
        draw_c=draw_c, c_sigma=float(model.u_c or 0.0), draw_tl=draw_tl, tl_sigma=float(model.u_r_tl or 0.0),
        draw_hg=draw_hg, hg_sigma=float(model.u_r_hg or 0.0), blank_blocks=blanks,
        custom_sigmas=custom_sigmas, custom_distributions=custom_distributions, custom_estimation_dofs=custom_estimation_dofs,
        # The record-bound calibration identity leads, so freshness can be checked from the record alone.
        # Draw configuration has its own complete config digest.  Keep the
        # replay identity limited to the resolved observations, masks, chains,
        # covariance and route so GUM and MC can compare the same object.
        input_digest=f"{calibration_identity_for_record(model.record)}.{model_input_digest(model, {})}",
        min_type_a_dof=min(dofs) if dofs else None,
        warnings=tuple(warnings), specs=tuple(specs),
        blank_uncertainty_input=str(getattr(uncertainty_config, "blank_uncertainty_input", "") or "") if blanks else "",
        blank_placement=PLACEMENT_SHARED_CHAIN if blanks else "",
    )


def _draw_array(rng: np.random.Generator, sigma: float, distribution: str, dof: float, size: int) -> np.ndarray:
    """Vectorized :func:`_draw_standard_uncertainty` for ``size`` draws."""
    distribution = _normalize_mc_distribution(distribution)
    if sigma <= 0.0:
        return np.zeros(size)
    if distribution == "rectangular":
        half_width = sigma * np.sqrt(3.0)
        return rng.uniform(-half_width, half_width, size)
    if np.isfinite(dof):
        if dof <= 0.0:
            raise ValueError("A scaled Student-t draw requires positive degrees of freedom.")
        return rng.standard_t(dof, size) * sigma
    return rng.normal(0.0, sigma, size)


def _run_pb_calibrated_draws(
    params: _PbCalibratedMCParams,
    rng: np.random.Generator,
    n_iter: int,
    progress_callback: Optional[Callable[[float], None]],
    *,
    engine: str,
) -> np.ndarray:
    """Draw every active input once per iteration and replay the whole calibration.

    Iterations are evaluated in fixed-size vectorized batches in draw order. The first
    iteration with a non-positive sampled reference, a non-positive or non-finite
    replayed mean, or a non-finite result aborts the run with its index; nothing is
    dropped, clipped or retried.
    """
    from domain.uncertainty.engine_pb_calibrated import evaluate

    model = params.model
    results = np.empty(n_iter, dtype=np.float64)
    done = 0
    while done < n_iter:
        size = min(PB_CALIBRATED_MC_BATCH, n_iter - done)
        blank_deltas: Dict[str, np.ndarray] = {}
        for blank in params.blank_blocks:
            dim = len(blank.channels)
            if dim == 1:
                draws = rng.normal(0.0, float(np.sqrt(blank.covariance_matrix[0, 0])), (size, 1))
            else:
                try:
                    draws = rng.multivariate_normal(
                        np.zeros(dim), blank.covariance_matrix, size=size, check_valid="raise",
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    raise MCCrossCheckError(
                        engine=engine, reason_code="invalid_covariance",
                        reason="The covariance-aware sampler refused a blank covariance matrix.",
                    ) from exc
            blank_deltas[blank.blank_observation_id] = draws
        r_tl = rng.normal(model.r_tl, params.tl_sigma, size) if params.draw_tl else np.full(size, model.r_tl)
        r_hg = (
            rng.normal(model.r_hg, params.hg_sigma, size) if params.draw_hg
            else (np.full(size, model.r_hg) if model.r_hg is not None else None)
        )
        c = rng.normal(model.c_value, params.c_sigma, size) if params.draw_c else np.full(size, model.c_value)
        target_add = (
            _draw_array(rng, params.target_sigma, MC_DISTRIBUTION_SCALED_STUDENT_T, params.target_dof, size)
            if params.draw_target else None
        )
        member_add = (
            np.column_stack([
                _draw_array(rng, sigma, MC_DISTRIBUTION_SCALED_STUDENT_T, dof, size)
                for sigma, dof in zip(params.member_sigmas, params.member_dofs)
            ])
            if params.draw_members else None
        )
        custom = np.zeros(size)
        for name in sorted(params.custom_sigmas):
            custom += _draw_array(rng, params.custom_sigmas[name], params.custom_distributions[name], float("inf"), size)

        with np.errstate(invalid="ignore"):
            invalid_parameter = ~np.isfinite(r_tl) | (r_tl <= 0.0) | ~np.isfinite(c) | (c <= 0.0)
            if params.draw_hg:
                invalid_parameter |= ~np.isfinite(r_hg) | (r_hg <= 0.0)
            out = evaluate(
                model, batch=size, r_tl=r_tl, r_hg=r_hg, c=c, blank_deltas=blank_deltas,
                target_add=target_add, member_add=member_add,
            )
            s, x = out["s"], out["x"]
            invalid_chain = (
                ~np.all(np.isfinite(s), axis=1) | np.any(s <= 0.0, axis=1) | ~np.isfinite(x) | (x <= 0.0)
            )
            if not model.is_session:
                invalid_chain |= ~np.isfinite(out["bracket"]) | (out["bracket"] <= 0.0)
            y = out["y"] + custom
            nonfinite = ~np.isfinite(y)
        failed = invalid_parameter | invalid_chain | nonfinite
        if failed.any():
            i = int(np.flatnonzero(failed)[0])
            if invalid_parameter[i]:
                code, reason = "invalid_sampled_parameter", "A sampled Tl, Hg or Pb reference value was not positive and finite."
            elif invalid_chain[i]:
                code, reason = "invalid_chain_output", "A replayed calibration chain mean was not positive and finite."
            else:
                code, reason = "nonfinite_model_output", "The model produced a non-finite result."
            raise MCCrossCheckError(engine=engine, reason_code=code, reason=reason, iteration=done + i)
        results[done:done + size] = y
        done += size
        if progress_callback is not None:
            progress_callback(done / n_iter)
    return results


def _sample_multivariate(
    mean: np.ndarray,
    covariance_matrix: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a validated covariance-aware vector without deleting correlation.

    Dropping a refused correlation structure would change the reported
    uncertainty without telling anyone, so a non-PSD block fails closed.
    """
    mean = np.asarray(mean, dtype=np.float64)
    covariance_matrix = np.asarray(covariance_matrix, dtype=np.float64)
    _validate_covariance_matrix(mean, covariance_matrix, engine="monte_carlo")
    if mean.size == 0:
        return mean.copy()
    if mean.size == 1:
        # Safe without clamping only because the validator already refused a
        # negative diagonal; C1 and C4 must be ported together.
        sigma = float(np.sqrt(covariance_matrix[0, 0]))
        return np.array([rng.normal(float(mean[0]), sigma)], dtype=np.float64)
    try:
        return np.asarray(
            rng.multivariate_normal(mean, covariance_matrix, check_valid="raise"),
            dtype=np.float64,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        # item 52: non-PSD covariance — correlation structure cannot be sampled.
        raise MCCrossCheckError(
            engine="monte_carlo",
            reason_code="invalid_covariance",
            reason="The covariance-aware sampler refused the covariance matrix.",
        ) from exc


def _validate_covariance_matrix(mean, covariance_matrix, *, engine):
    from domain.uncertainty.numerical_domain import validate_covariance_matrix
    try:
        validate_covariance_matrix(mean, covariance_matrix)
    except ValueError as exc:
        raise MCCrossCheckError(engine=engine, reason_code="invalid_covariance", reason=str(exc)) from exc


def _preflight_cross_check_params(params: object, *, engine: str) -> None:
    """Validate chain availability and every covariance block before drawing.

    Runs once, before the first draw, so a malformed input fails immediately
    instead of part-way through 100 000 iterations.
    """
    for block in getattr(params, "blank_blocks", ()):
        mean = np.asarray(
            getattr(block, "mean_vector", np.zeros(len(getattr(block, "isotopes", ())))),
            dtype=np.float64,
        )
        _validate_covariance_matrix(
            mean,
            np.asarray(block.covariance_matrix, dtype=np.float64),
            engine=engine,
        )

    for blank_term in getattr(params, "hg_blank_terms", ()):
        mean = np.asarray(blank_term.mean_vector, dtype=np.float64)
        _validate_covariance_matrix(
            mean,
            np.asarray(blank_term.covariance_matrix, dtype=np.float64),
            engine=engine,
        )
        gradient = np.asarray(blank_term.gradient, dtype=np.float64)
        if gradient.shape != mean.shape or not np.all(np.isfinite(gradient)):
            raise MCCrossCheckError(
                engine=engine,
                reason_code="invalid_sampled_parameter",
                reason="A linearized blank term lacks a finite sensitivity for every sampled channel.",
            )

    if isinstance(params, _InternalPerturbationParams):
        required = {
            "87Sr",
            "86Sr",
            params.reference_inputs.normalization_numerator,
            params.reference_inputs.normalization_denominator,
        }
        missing = sorted(required.difference(params.base_corrected_intensities))
        if missing:
            raise MCCrossCheckError(
                engine=engine,
                reason_code="missing_chain_input",
                reason=f"The correction chain is missing isotopes: {', '.join(missing)}.",
            )
        if not np.isfinite(params.normalization_value) or params.normalization_value <= 0.0:
            raise MCCrossCheckError(
                engine=engine,
                reason_code="missing_chain_input",
                reason="The correction chain has no positive finite normalization value.",
            )

    if isinstance(params, _PbTlPerturbationParams):
        numerator, denominator = params.ratio_name.split("/", 1)
        required = {numerator, denominator}
        if params.apply_iif:
            required.update({"203Tl", "205Tl"})
        if params.apply_interference and "204Pb" in params.ratio_name:
            required.add("202Hg")
        missing = sorted(required.difference(params.base_intensities))
        if missing:
            raise MCCrossCheckError(
                engine=engine,
                reason_code="missing_chain_input",
                reason=f"The correction chain is missing isotopes: {', '.join(missing)}.",
            )
        if not np.isfinite(params.tl_norm_value) or params.tl_norm_value <= 0.0:
            raise MCCrossCheckError(
                engine=engine,
                reason_code="missing_chain_input",
                reason="The correction chain has no positive finite Tl normalization value.",
            )

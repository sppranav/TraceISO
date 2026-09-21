"""Standard repeatability (u_std_repeatability) computation strategies."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Literal, Optional, Tuple, overload

import numpy as np

from config.settings import UncertaintyConfig
from domain.elements.base import ElementConfig
from domain.models import CycleData, ReprodResult, Sample
from domain.ratio_selection import get_best_ratio_data


_LOG = logging.getLogger(__name__)


def standard_identity_key(sample: Sample) -> str:
    """Return a stable, JSON-safe identity key for a standard observation."""
    return f"{sample.name}__run_{sample.run_number}"


def _resolve_standard_setting(
    sample: Sample,
    mapping: Dict[str, int],
    default: int,
) -> int:
    identity = standard_identity_key(sample)
    if identity in mapping:
        return mapping[identity]
    return mapping.get(sample.name, default)


def _standard_is_excluded(sample: Sample, excluded: set[str]) -> bool:
    """Is this observation excluded by any reference the session may hold?

    Three reference forms are honoured, in order of precision: the observation
    ID, which the include/exclude controls now emit; the legacy
    ``name__run_N`` identity key; and a bare display name, whose long-standing
    meaning is name-wide exclusion. The bare name stays name-wide because
    saved sessions rely on it — what changed is that the controls no longer
    *write* one.
    """
    if sample.observation_id and sample.observation_id in excluded:
        return True
    return standard_identity_key(sample) in excluded or sample.name in excluded


# Public dispatcher

def compute_reprod(
    all_samples: List[Sample],
    ratio_name: str,
    *,
    uncertainty_config: UncertaintyConfig,
    element_config: ElementConfig,
    drift_model: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ratio_extractor: Optional[Callable[[Sample, str], Optional[CycleData]]] = None,
    position_extractor: Optional[Callable[[Sample], float]] = None,
) -> ReprodResult:
    """Master dispatcher for u_std_repeatability computation."""
    if uncertainty_config.ssb_mode == "block_average":
        return _compute_reprod_block(
            all_samples, ratio_name,
            uncertainty_config=uncertainty_config,
            drift_model=drift_model,
            ratio_extractor=ratio_extractor,
            position_extractor=position_extractor,
        )

    std_names, std_means, std_positions, std_samples = _extract_standard_data(
        all_samples,
        ratio_name,
        ratio_extractor=ratio_extractor,
        position_extractor=position_extractor,
        include_samples=True,
    )

    if len(std_names) == 0:
        return _empty_result(uncertainty_config.reprod_method)

    excluded = set(uncertainty_config.excluded_standards)
    std_included = [not _standard_is_excluded(sample, excluded) for sample in std_samples]
    std_identities = [[sample.observation_id] for sample in std_samples]

    seg_map = uncertainty_config.segment_assignments
    if seg_map:
        std_segments = [_resolve_standard_setting(sample, seg_map, 1) for sample in std_samples]
    else:
        std_segments = [1] * len(std_names)

    method = uncertainty_config.resolve_reprod_method(
        enable_ssb=uncertainty_config.enable_ssb,
        enable_delta=uncertainty_config.enable_delta,
    )
    fallback_reason: Optional[str] = None

    n_included = int(np.sum(std_included))

    if method == "drift_residuals" and drift_model is None:
        method = "loo_cross_validation"
        fallback_reason = "drift_residuals requested but no drift model available; falling back to LOO"

    if method == "loo_cross_validation" and n_included < 3:
        method = "sd_of_means"
        if fallback_reason:
            fallback_reason += "; LOO requires >= 3 included standards, falling back to sd_of_means"
        else:
            fallback_reason = "LOO requires >= 3 included standards; falling back to sd_of_means"

    if n_included < 2:
        _LOG.warning(
            "u_std_repeatability: only %d included standard(s) for '%s' — "
            "returning 0.0. Repeatability contributor will be inactive.",
            n_included,
            ratio_name,
        )
        result = _empty_result(method)
        result.std_names = std_names
        result.std_means = np.array(std_means)
        result.std_positions = np.array(std_positions)
        result.std_included = std_included
        result.std_segments = std_segments
        result.std_identities = std_identities
        result.fallback_reason = "< 2 included standards; u_std_repeatability = 0 (incomplete)"
        return result

    # Route to strategy
    if method == "sd_of_means":
        result = _sd_of_means(std_names, std_means, std_positions, std_included, std_segments)
    elif method == "loo_cross_validation":
        result = _loo_cross_validation(std_names, std_means, std_positions, std_included, std_segments)
    elif method == "drift_residuals":
        result = _drift_residuals(std_names, std_means, std_positions, std_included, std_segments, drift_model)
    elif method == "robust_mad":
        result = _robust_mad(std_names, std_means, std_positions, std_included, std_segments)
    else:
        result = _sd_of_means(std_names, std_means, std_positions, std_included, std_segments)
        fallback_reason = f"Unknown method '{uncertainty_config.reprod_method}'; falling back to sd_of_means"

    if fallback_reason:
        result.fallback_reason = fallback_reason

    result.std_identities = std_identities

    # Suppress when drift correction is active (method == drift_residuals)
    # to prevent double-counting — drift residuals already capture this (§3).
    if uncertainty_config.include_kappa_drift and method != "drift_residuals":
        kappa, deltas = _kappa_drift(std_means, std_included, std_segments)
        result.kappa_drift_permil = kappa
        result.drift_deltas = deltas

    return result


# Strategy implementations

def _sd_of_means(
    std_names: List[str],
    std_means: List[float],
    std_positions: List[float],
    std_included: List[bool],
    std_segments: List[int],
) -> ReprodResult:
    """SD of included standard means, computed per segment."""
    means_arr = np.array(std_means)
    incl_arr = np.array(std_included)

    # Session-wide SD (fallback for thin segments)
    session_included = means_arr[incl_arr]
    session_sd = float(np.std(session_included, ddof=1)) if len(session_included) >= 2 else 0.0
    session_dof = max(len(session_included) - 1, 0)

    # Per-segment computation
    unique_segs = sorted(set(std_segments))
    seg_sds: Dict[int, float] = {}
    seg_dofs: Dict[int, int] = {}
    seg_n: Dict[int, int] = {}

    for seg in unique_segs:
        seg_mask = np.array([s == seg and inc for s, inc in zip(std_segments, std_included)])
        seg_vals = means_arr[seg_mask]
        n = len(seg_vals)
        seg_n[seg] = n
        if n >= 2:
            seg_sds[seg] = float(np.std(seg_vals, ddof=1))
            seg_dofs[seg] = n - 1
        else:
            # Fallback: use session-wide SD
            seg_sds[seg] = session_sd
            seg_dofs[seg] = 0

    # Overall u_std_repeatability: if single segment, use that; otherwise pool
    if len(unique_segs) == 1:
        u_abs = seg_sds[unique_segs[0]]
        dof = seg_dofs[unique_segs[0]]
    else:
        u_abs, dof = _pooled_sd(seg_sds, seg_dofs, seg_n)
        if dof == 0 and session_dof > 0:
            u_abs, dof = session_sd, session_dof

    grand_mean = _segment_weighted_mean(means_arr, incl_arr, std_segments)
    u_rel = (u_abs / grand_mean) * 1000.0 if np.isfinite(grand_mean) and grand_mean != 0 else 0.0

    return ReprodResult(
        method="sd_of_means",
        u_std_repeatability_abs=u_abs,
        u_std_repeatability_rel_permil=u_rel,
        degrees_of_freedom=dof,
        std_names=std_names,
        std_means=np.array(std_means),
        std_positions=np.array(std_positions),
        std_included=std_included,
        std_segments=std_segments,
        segment_sds=seg_sds,
        segment_dofs=seg_dofs,
        segment_n_stds=seg_n,
    )


#: Availability vocabulary for :class:`ReprodResult`.
#:
#: ``available``      a validated estimator produced the magnitude and DoF.
#: ``not_applicable`` the session carries too few standard observations for any
#:                    between-run estimator; the contributor stays inactive with
#:                    an explicit reason and never contributes a zero to the RSS.
#: ``unavailable``    an enabled, required estimator cannot be evaluated and no
#:                    validated fallback exists, so the whole budget is refused.
REPROD_STATUS_AVAILABLE = "available"
REPROD_STATUS_NOT_APPLICABLE = "not_applicable"
REPROD_STATUS_UNAVAILABLE = "unavailable"

#: Identity of the LOO estimator: exact residual linear map, unbiased variance
#: statistic r'r/tr(A'A), Satterthwaite moment-approximated sampling DoF.
LOO_ESTIMATOR_IDENTITY = "loo_residual_map_quadratic_form_v1"

#: Drift-fit provenances the residual estimator can assign degrees of freedom to.
SAME_DATA_OLS_PROVENANCE = "same_data_ols_v1"
EXTERNAL_CURVE_PROVENANCE = "external_known_curve_v1"

#: SSE/(n - rank) for a curve fitted to exactly these standards.
OLS_RESIDUAL_ESTIMATOR_IDENTITY = "same_data_ols_residual_sd_v1"
#: SSE/n for a curve the producer declares independent of these standards.
EXTERNAL_CURVE_ESTIMATOR_IDENTITY = "external_known_curve_residual_sd_v1"


def _unsupported_estimator_fallback(
    std_names: List[str],
    std_means: List[float],
    std_positions: List[float],
    std_included: List[bool],
    std_segments: List[int],
    *,
    requested_method: str,
    estimator_identity: str,
    reason: str,
) -> ReprodResult:
    """Fall back to ``sd_of_means`` when the requested estimator is unsupported.

    A043/A042: a layout with no validated residual linear map must not be given
    a guessed effective DoF.  ``sd_of_means`` is the already-validated estimator
    for a set of standard means, so it supplies both the magnitude and its own
    correct ``n-1`` DoF, while ``requested_method``/``fallback_reason`` keep the
    reported method truthful (A098).
    """
    result = _sd_of_means(std_names, std_means, std_positions, std_included, std_segments)
    result.requested_method = requested_method
    result.fallback_reason = reason
    result.unavailable_reason = reason
    result.estimator_identity = "sd_of_means_v1"
    result.error_model = "independent_homoscedastic_standard_means"
    result.requested_estimator_identity = estimator_identity
    return result


def _loo_cross_validation(
    std_names: List[str],
    std_means: List[float],
    std_positions: List[float],
    std_included: List[bool],
    std_segments: List[int],
) -> ReprodResult:
    """Leave-one-out cross-validation of SSB interpolation, applied per segment.

    Each segment is treated independently so that LOO neighbors never cross a
    segment boundary.  Segments with fewer than 3 included standards cannot
    produce LOO residuals and are skipped; their standards receive no predicted
    value and contribute nothing to u_abs.
    """
    means_arr = np.array(std_means)
    pos_arr = np.array(std_positions)
    incl_arr = np.array(std_included)
    seg_arr = (
        np.array(std_segments, dtype=int)
        if std_segments
        else np.ones(len(std_means), dtype=int)
    )

    predicted = np.full(len(std_means), np.nan)
    all_residuals: List[float] = []
    residual_rows: List[np.ndarray] = []
    skipped_segments: List[int] = []

    def _pair_weights(x0: float, x1: float, x_target: float) -> Tuple[float, float]:
        """Weights of a two-point linear predictor, matching ``_interpolate``."""
        if x1 == x0:
            return 0.5, 0.5
        t = (x_target - x0) / (x1 - x0)
        return 1.0 - t, t

    for seg_id in np.unique(seg_arr):
        seg_incl_mask = incl_arr & (seg_arr == seg_id)
        seg_incl_idx = np.where(seg_incl_mask)[0]
        sort_order = np.argsort(pos_arr[seg_incl_idx])
        seg_incl_idx = seg_incl_idx[sort_order]

        n_seg = len(seg_incl_idx)
        if n_seg < 3:
            # LOO needs two neighbours to predict from, so such a segment
            # supplies no residual row at all rather than a guessed one.
            skipped_segments.append(int(seg_id))
            continue

        if not np.all(np.isfinite(pos_arr[seg_incl_idx])):
            return _unsupported_estimator_fallback(
                std_names, std_means, std_positions, std_included, std_segments,
                requested_method="loo_cross_validation",
                estimator_identity=LOO_ESTIMATOR_IDENTITY,
                reason=(
                    "LOO repeatability needs finite standard positions to build "
                    "its residual map; sd_of_means was used instead."
                ),
            )

        for rank, idx in enumerate(seg_incl_idx):
            pos_i = float(pos_arr[idx])
            # Every predictor is a two-point linear form, so the residual
            # y_i - yhat_i is an exact linear functional of the standard means.
            if 0 < rank < n_seg - 1:
                left_idx, right_idx = seg_incl_idx[rank - 1], seg_incl_idx[rank + 1]
            elif rank == 0:
                left_idx, right_idx = seg_incl_idx[1], seg_incl_idx[2]
            else:
                left_idx, right_idx = seg_incl_idx[rank - 2], seg_incl_idx[rank - 1]

            w_left, w_right = _pair_weights(
                float(pos_arr[left_idx]), float(pos_arr[right_idx]), pos_i
            )
            row = np.zeros(len(std_means), dtype=float)
            row[idx] += 1.0
            row[left_idx] -= w_left
            row[right_idx] -= w_right

            predicted[idx] = float(
                w_left * means_arr[left_idx] + w_right * means_arr[right_idx]
            )
            all_residuals.append(float(means_arr[idx] - predicted[idx]))
            residual_rows.append(row)

    if not residual_rows:
        return _unsupported_estimator_fallback(
            std_names, std_means, std_positions, std_included, std_segments,
            requested_method="loo_cross_validation",
            estimator_identity=LOO_ESTIMATOR_IDENTITY,
            reason=(
                "No segment holds the three included standards LOO needs to form "
                "a residual; sd_of_means was used instead."
            ),
        )

    residuals_arr = np.array(all_residuals)
    residual_map = np.vstack(residual_rows)

    # Exact moments of the LOO residual sum of squares.
    #
    # Each predictor above is a two-point linear interpolation/extrapolation in
    # position, so the residual map A reproduces any affine trend exactly and
    # therefore annihilates it: A @ (a + b*x) = 0.  With standard means
    # y = f(x) + eps and independent homoscedastic eps of variance sigma^2, the
    # residual vector is r = A eps whenever f is affine within each segment, so
    #
    #     E[r'r] = sigma^2 * tr(A'A)   ->   sigma_hat^2 = r'r / tr(A'A)
    #
    # is unbiased for the standard-mean scatter.  Curvature in f leaks into r'r
    # and inflates the estimate, so the term is conservative, never optimistic,
    # when the within-segment trend is not affine.
    gram = residual_map @ residual_map.T
    trace_gram = float(np.trace(gram))
    trace_gram_squared = float(np.sum(gram * gram))
    if not np.isfinite(trace_gram) or trace_gram <= 0.0 or not np.isfinite(
        trace_gram_squared
    ) or trace_gram_squared <= 0.0:
        return _unsupported_estimator_fallback(
            std_names, std_means, std_positions, std_included, std_segments,
            requested_method="loo_cross_validation",
            estimator_identity=LOO_ESTIMATOR_IDENTITY,
            reason=(
                "The LOO residual map is degenerate, so its variance statistic is "
                "undefined; sd_of_means was used instead."
            ),
        )

    u_abs = float(np.sqrt(float(residuals_arr @ residuals_arr) / trace_gram))

    # Sampling DoF of the quadratic form r'r = eps' M eps with M = A'A.  Matching
    # its first two moments to a scaled chi-square (Satterthwaite) gives
    # nu = tr(M)^2 / tr(M^2).  This is the standard moment APPROXIMATION for a
    # quadratic form in normal variates, exact only when M's non-zero
    # eigenvalues are equal; it is not a rank count.  For the three-point
    # layout M is rank one and it returns exactly nu = 1.
    dof = trace_gram * trace_gram / trace_gram_squared

    grand_mean = _segment_weighted_mean(means_arr, incl_arr, std_segments)
    u_rel = (u_abs / grand_mean) * 1000.0 if np.isfinite(grand_mean) and grand_mean != 0 else 0.0

    return ReprodResult(
        method="loo_cross_validation",
        u_std_repeatability_abs=u_abs,
        u_std_repeatability_rel_permil=u_rel,
        degrees_of_freedom=dof,
        std_names=std_names,
        std_means=np.array(std_means),
        std_positions=np.array(std_positions),
        std_included=std_included,
        std_segments=std_segments,
        residuals=residuals_arr,
        predicted_values=predicted,
        status=REPROD_STATUS_AVAILABLE,
        estimator_identity=LOO_ESTIMATOR_IDENTITY,
        error_model="independent_homoscedastic_standard_means",
        residual_linear_map=residual_map,
        residual_sse=float(residuals_arr @ residuals_arr),
        residual_scale=u_abs,
        skipped_segments=tuple(skipped_segments),
    )


def _drift_residuals(
    std_names: List[str],
    std_means: List[float],
    std_positions: List[float],
    std_included: List[bool],
    std_segments: List[int],
    drift_model: Callable[[np.ndarray], np.ndarray],
) -> ReprodResult:
    """SD of (actual - drift_model predicted) at standard positions."""
    means_arr = np.array(std_means)
    pos_arr = np.array(std_positions)
    incl_arr = np.array(std_included)

    incl_positions = pos_arr[incl_arr]
    incl_means = means_arr[incl_arr]

    fit_info = getattr(drift_model, "_traceiso_fit_info", None)
    if not isinstance(fit_info, dict):
        return _unsupported_estimator_fallback(
            std_names, std_means, std_positions, std_included, std_segments,
            requested_method="drift_residuals",
            estimator_identity=OLS_RESIDUAL_ESTIMATOR_IDENTITY,
            reason=(
                "Drift-residual repeatability requires frozen same-data OLS fit "
                "provenance; an external or legacy curve cannot be assigned "
                "parameter-loss degrees of freedom, so sd_of_means was used instead."
            ),
        )

    fit_x = np.asarray(fit_info.get("x_data", []), dtype=float)
    fit_y = np.asarray(fit_info.get("y_data", []), dtype=float)
    provenance = str(fit_info.get("fit_provenance") or "")
    weighted = bool(fit_info.get("weighted", False))
    correlated = bool(fit_info.get("correlated", False))

    if weighted or correlated:
        return _unsupported_estimator_fallback(
            std_names, std_means, std_positions, std_included, std_segments,
            requested_method="drift_residuals",
            estimator_identity=OLS_RESIDUAL_ESTIMATOR_IDENTITY,
            reason=(
                "Drift-residual repeatability has no validated DoF model for a "
                "weighted or correlated fit; sd_of_means was used instead."
            ),
        )

    predicted_vals = drift_model(incl_positions)
    residuals = incl_means - predicted_vals
    sse = float(residuals @ residuals)

    # Store full-length predicted array for plotting
    all_predicted = np.full(len(std_means), np.nan)
    all_predicted[incl_arr] = predicted_vals

    if provenance == SAME_DATA_OLS_PROVENANCE:
        # Parameters were estimated from exactly these observations, so the
        # residual vector lies in an (n - rank) dimensional subspace and
        # SSE/(n - rank) is the unbiased variance estimate.  The active support
        # must match the frozen fit for that parameter count to apply.
        rank = int(fit_info.get("design_rank", 0) or 0)
        dof = int(fit_info.get("residual_dof", 0) or 0)
        support_matches = (
            fit_info.get("error_model") == "independent_homoscedastic_normal"
            and fit_x.shape == incl_positions.shape
            and fit_y.shape == incl_means.shape
            and np.allclose(fit_x, incl_positions, rtol=0.0, atol=1e-12)
            and np.allclose(fit_y, incl_means, rtol=1e-12, atol=1e-15)
            and len(residuals) == int(fit_info.get("fitted_observations", -1))
        )
        if not support_matches:
            return _unsupported_estimator_fallback(
                std_names, std_means, std_positions, std_included, std_segments,
                requested_method="drift_residuals",
                estimator_identity=OLS_RESIDUAL_ESTIMATOR_IDENTITY,
                reason=(
                    "Drift-residual repeatability needs the active standard support "
                    "to match the frozen same-data OLS fit; it does not, so the "
                    "parameter-loss DoF is unknown and sd_of_means was used instead."
                ),
            )
        if dof <= 0 or rank <= 0:
            return _unsupported_estimator_fallback(
                std_names, std_means, std_positions, std_included, std_segments,
                requested_method="drift_residuals",
                estimator_identity=OLS_RESIDUAL_ESTIMATOR_IDENTITY,
                reason=(
                    "The drift fit has no valid positive residual degrees of "
                    "freedom, so its residual scale is undefined and sd_of_means "
                    "was used instead."
                ),
            )
        estimator_identity = OLS_RESIDUAL_ESTIMATOR_IDENTITY
    elif provenance == EXTERNAL_CURVE_PROVENANCE:
        # The producer declares this curve was not fitted to these observations,
        # so no parameter was estimated from them and every residual carries a
        # full degree of freedom.  Only an explicit declaration reaches here; an
        # unlabelled curve is refused below rather than assumed independent.
        rank = 0
        dof = int(len(residuals))
        if dof <= 0:
            return _unsupported_estimator_fallback(
                std_names, std_means, std_positions, std_included, std_segments,
                requested_method="drift_residuals",
                estimator_identity=EXTERNAL_CURVE_ESTIMATOR_IDENTITY,
                reason=(
                    "An externally declared drift curve needs at least one "
                    "included standard residual; sd_of_means was used instead."
                ),
            )
        estimator_identity = EXTERNAL_CURVE_ESTIMATOR_IDENTITY
    else:
        return _unsupported_estimator_fallback(
            std_names, std_means, std_positions, std_included, std_segments,
            requested_method="drift_residuals",
            estimator_identity=OLS_RESIDUAL_ESTIMATOR_IDENTITY,
            reason=(
                "Drift-residual repeatability cannot tell how many parameters a "
                "curve with fit provenance {0!r} estimated from these standards, "
                "so its DoF is unknown and sd_of_means was used instead."
            ).format(provenance or "unset"),
        )

    u_abs = float(np.sqrt(sse / dof))

    grand_mean = _segment_weighted_mean(means_arr, incl_arr, std_segments)
    u_rel = (u_abs / grand_mean) * 1000.0 if np.isfinite(grand_mean) and grand_mean != 0 else 0.0

    return ReprodResult(
        method="drift_residuals",
        u_std_repeatability_abs=u_abs,
        u_std_repeatability_rel_permil=u_rel,
        degrees_of_freedom=dof,
        std_names=std_names,
        std_means=np.array(std_means),
        std_positions=np.array(std_positions),
        std_included=std_included,
        std_segments=std_segments,
        residuals=residuals,
        predicted_values=all_predicted,
        status=REPROD_STATUS_AVAILABLE,
        estimator_identity=estimator_identity,
        error_model="independent_homoscedastic_normal",
        fitted_observations=len(residuals),
        design_rank=rank,
        residual_dof=dof,
        residual_sse=sse,
        residual_scale=u_abs,
    )


def _robust_mad(
    std_names: List[str],
    std_means: List[float],
    std_positions: List[float],
    std_included: List[bool],
    std_segments: List[int],
) -> ReprodResult:
    """MAD-based robust estimator for u_std_repeatability, per segment."""
    means_arr = np.array(std_means)
    incl_arr = np.array(std_included)

    session_included = means_arr[incl_arr]
    session_mad = _mad_sd(session_included) if len(session_included) >= 2 else 0.0
    unique_segs = sorted(set(std_segments))
    seg_sds: Dict[int, float] = {}
    seg_dofs: Dict[int, int] = {}
    seg_n: Dict[int, int] = {}

    for seg in unique_segs:
        seg_mask = np.array([s == seg and inc for s, inc in zip(std_segments, std_included)])
        seg_vals = means_arr[seg_mask]
        n = len(seg_vals)
        seg_n[seg] = n
        if n >= 2:
            seg_sds[seg] = _mad_sd(seg_vals)
            seg_dofs[seg] = 0
        else:
            seg_sds[seg] = session_mad
            seg_dofs[seg] = 0

    # Diagnostic scatter only; do not pool using invented finite-sample DoF.
    u_abs = session_mad
    dof = float("nan")

    grand_mean = _segment_weighted_mean(means_arr, incl_arr, std_segments)
    u_rel = (u_abs / grand_mean) * 1000.0 if np.isfinite(grand_mean) and grand_mean != 0 else 0.0

    result = ReprodResult(
        method="robust_mad",
        u_std_repeatability_abs=u_abs,
        u_std_repeatability_rel_permil=u_rel,
        degrees_of_freedom=dof,
        std_names=std_names,
        std_means=np.array(std_means),
        std_positions=np.array(std_positions),
        std_included=std_included,
        std_segments=std_segments,
        segment_sds=seg_sds,
        segment_dofs=seg_dofs,
        segment_n_stds=seg_n,
    )

    result.status = "unavailable"
    result.unavailable_reason = (
        "MAD scatter is diagnostic only: 0.36*N is asymptotic relative efficiency, "
        "not a qualified finite-sample degrees-of-freedom model. Supply laboratory "
        "qualification or explicitly select another repeatability estimator."
    )
    result.estimator_identity = "mad_diagnostic_unqualified.v1"
    result.degrees_of_freedom = float("nan")
    return result


def _kappa_drift(
    std_means: List[float],
    std_included: List[bool],
    std_segments: Optional[List[int]] = None,
) -> Tuple[float, np.ndarray]:
    """Return the adopted drift standard uncertainty and eligible pair deltas.

    The owner-selected Type B convention is half the mean absolute relative difference
    between consecutive included standards. Segment boundaries are hard
    calculation boundaries and are never bridged.
    """
    means_arr = np.asarray(std_means, dtype=float)
    pairs = _eligible_kappa_drift_pairs(std_included, std_segments)
    if not pairs:
        return 0.0, np.array([])

    deltas: List[float] = []
    for left_idx, right_idx in pairs:
        denominator = float(means_arr[left_idx])
        if np.isclose(denominator, 0.0, rtol=0.0, atol=1e-15):
            _LOG.warning(
                "Skipping kappa_drift pair %d->%d: left standard mean is near zero.",
                left_idx,
                right_idx,
            )
            continue
        deltas.append(
            (float(means_arr[right_idx]) - denominator) / denominator * 1000.0
        )
    if not deltas:
        return 0.0, np.array([])

    delta_array = np.asarray(deltas, dtype=float)
    kappa = float(np.mean(np.abs(delta_array)) / 2.0)
    return kappa, delta_array


def _eligible_kappa_drift_pairs(
    std_included: List[bool],
    std_segments: Optional[List[int]] = None,
) -> List[Tuple[int, int]]:
    """Return adjacent included index pairs that do not cross segments."""
    included = [bool(value) for value in std_included]
    if std_segments is None:
        segments = [1] * len(included)
    else:
        if len(std_segments) != len(included):
            raise ValueError("std_segments must have the same length as std_included")
        segments = [int(value) for value in std_segments]

    included_indices = [idx for idx, value in enumerate(included) if value]
    return [
        (left_idx, right_idx)
        for left_idx, right_idx in zip(included_indices, included_indices[1:])
        if segments[left_idx] == segments[right_idx]
    ]


# Internal helpers

@overload
def _extract_standard_data(
    all_samples: List[Sample],
    ratio_name: str,
    *,
    ratio_extractor: Optional[Callable[[Sample, str], Optional[CycleData]]] = None,
    position_extractor: Optional[Callable[[Sample], float]] = None,
    include_samples: Literal[False] = False,
) -> Tuple[List[str], List[float], List[float]]: ...


@overload
def _extract_standard_data(
    all_samples: List[Sample],
    ratio_name: str,
    *,
    ratio_extractor: Optional[Callable[[Sample, str], Optional[CycleData]]] = None,
    position_extractor: Optional[Callable[[Sample], float]] = None,
    include_samples: Literal[True],
) -> Tuple[List[str], List[float], List[float], List[Sample]]: ...


def _extract_standard_data(
    all_samples: List[Sample],
    ratio_name: str,
    *,
    ratio_extractor: Optional[Callable[[Sample, str], Optional[CycleData]]] = None,
    position_extractor: Optional[Callable[[Sample], float]] = None,
    include_samples: bool = False,
) -> (
    Tuple[List[str], List[float], List[float]]
    | Tuple[List[str], List[float], List[float], List[Sample]]
):
    """Extract names, mean ratios, and positions for all standards."""
    names: List[str] = []
    means: List[float] = []
    positions: List[float] = []
    samples: List[Sample] = []

    _extract = ratio_extractor if ratio_extractor is not None else get_best_ratio_data

    for s in all_samples:
        if not s.is_standard or s.metadata.get("excluded", False):
            continue
        cd = _extract(s, ratio_name)
        if cd is None:
            continue
        if cd.n_valid < 2:
            # Need ≥ 2 valid cycles to define a within-standard SD (ddof=1).
            # Standards skipped here do not contribute to u_std_repeatability.
            _LOG.debug(
                "Standard %r excluded from repeatability calculation: only %d valid cycle(s) "
                "for ratio %r (need >= 2).",
                s.name,
                cd.n_valid,
                ratio_name,
            )
            continue
        names.append(s.name)
        means.append(cd.mean)
        samples.append(s)
        if position_extractor is not None:
            positions.append(float(position_extractor(s)))
        else:
            positions.append(float(s.run_number))

    if include_samples:
        return names, means, positions, samples
    return names, means, positions


def _interpolate(
    x0: float, y0: float,
    x1: float, y1: float,
    x: float,
) -> float:
    """Linear interpolation between (x0, y0) and (x1, y1) at x."""
    if x1 == x0:
        return (y0 + y1) / 2.0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _extrapolate(
    x0: float, y0: float,
    x1: float, y1: float,
    x_target: float,
) -> float:
    """Linear extrapolation from line through (x0, y0) and (x1, y1)."""
    if x1 == x0:
        return (y0 + y1) / 2.0
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x_target - x0)


def _mad_sd(values: np.ndarray) -> float:
    """MAD * 1.4826 — consistent estimator of SD for normal data."""
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(mad * 1.4826)


def _pooled_sd(
    seg_sds: Dict[int, float],
    seg_dofs: Dict[int, int],
    seg_n: Dict[int, int],
) -> Tuple[float, int]:
    """Pooled standard deviation across segments (weighted by DoF).

    Uses the standard pooled-variance formula:
        s_pooled^2 = sum(dof_k * s_k^2) / sum(dof_k)
        dof_total = sum(dof_k)
    """
    total_ss = 0.0
    total_dof = 0
    for seg in seg_sds:
        dof = seg_dofs.get(seg, 0)
        if dof > 0:
            total_ss += dof * seg_sds[seg] ** 2
            total_dof += dof

    if total_dof == 0:
        return 0.0, 0

    pooled_var = total_ss / total_dof
    return float(np.sqrt(pooled_var)), total_dof


def _segment_weighted_mean(
    means_arr: np.ndarray,
    incl_arr: np.ndarray,
    std_segments: List[int],
) -> float:
    """n_k-weighted mean across segments for use as the u_rel denominator.

    When segments have different ratio levels (different instrumental regimes),
    the unweighted grand mean is a blend of those levels and produces a
    denominator that does not represent any actual measurement regime.
    Weighting by the number of included standards per segment gives a
    denominator consistent with how u_abs is pooled across segments.

    Reduces to the simple mean when there is only one segment.
    """
    seg_arr = np.array(std_segments, dtype=int) if std_segments else np.ones(len(means_arr), dtype=int)
    total_sum = 0.0
    total_n = 0
    for seg in np.unique(seg_arr):
        seg_incl = incl_arr & (seg_arr == seg)
        vals = means_arr[seg_incl]
        n = len(vals)
        if n > 0:
            total_sum += n * float(np.mean(vals))
            total_n += n
    return total_sum / total_n if total_n > 0 else 0.0


def _empty_result(method: str) -> ReprodResult:
    """Return an empty ReprodResult when no standards are available."""
    return ReprodResult(
        method=method,
        u_std_repeatability_abs=0.0,
        u_std_repeatability_rel_permil=0.0,
        degrees_of_freedom=0,
        std_names=[],
        std_means=np.array([]),
        std_positions=np.array([]),
        std_included=[],
        std_segments=[],
        fallback_reason="No standards with valid ratio data",
        # Not a refusal: a session without replicate standards has no between-run
        # repeatability to measure.  The contributor stays inactive with this
        # reason, which is the long-standing documented behaviour, rather than
        # making the whole budget unavailable.
        status=REPROD_STATUS_NOT_APPLICABLE,
        unavailable_reason=(
            "Between-run repeatability requires at least two independent "
            "standard observations with valid ratio data."
        ),
    )


# Block-averaged SSB: u_std_repeatability from block means

def _extract_block_data(
    all_samples: List[Sample],
    ratio_name: str,
    *,
    ratio_extractor: Optional[Callable[[Sample, str], Optional[CycleData]]] = None,
    position_extractor: Optional[Callable[[Sample], float]] = None,
) -> Tuple[List[str], List[float], List[float], List[float], List[int], List[List[Sample]]]:
    """Extract block-level data plus the observations behind each block.

    The block's display name is its members' labels joined with ``+``. That
    string cannot be split back into observations — two members may share a
    name, and a name containing ``+`` would split wrongly — so the members
    themselves are returned alongside it.
    """
    _extract = ratio_extractor if ratio_extractor is not None else get_best_ratio_data

    block_names: List[str] = []
    block_means: List[float] = []
    block_positions: List[float] = []
    block_sems: List[float] = []
    block_n_stds: List[int] = []
    block_members: List[List[Sample]] = []

    # Accumulate current block
    cur_names: List[str] = []
    cur_means: List[float] = []
    cur_positions: List[float] = []
    cur_members: List[Sample] = []

    def _flush() -> None:
        if not cur_means:
            return
        n = len(cur_means)
        block_names.append("+".join(cur_names))
        block_means.append(float(np.mean(cur_means)))
        block_positions.append(float(np.mean(cur_positions)))
        if n >= 2:
            block_sems.append(float(np.std(cur_means, ddof=1) / np.sqrt(n)))
        else:
            block_sems.append(0.0)
        block_n_stds.append(n)
        block_members.append(list(cur_members))

    for s in all_samples:
        if s.is_standard and not s.metadata.get("excluded", False):
            cd = _extract(s, ratio_name)
            if cd is not None and cd.n_valid >= 2:
                cur_names.append(s.name)
                cur_means.append(cd.mean)
                cur_members.append(s)
                if position_extractor is not None:
                    cur_positions.append(float(position_extractor(s)))
                else:
                    cur_positions.append(float(s.run_number))
                continue
            # Standard present but insufficient valid cycles — it splits the
            # current block; log so the break is discoverable in debug output.
            n_valid = cd.n_valid if cd is not None else 0
            _LOG.debug(
                "reprod block split: standard '%s' (run %d) has %d valid cycle(s) "
                "for %s — fewer than 2 required; flushing block [%s].",
                s.name, s.run_number, n_valid, ratio_name, ", ".join(cur_names),
            )
        # Non-standard or insufficient data → flush
        if cur_names:
            _flush()
            cur_names = []
            cur_means = []
            cur_positions = []
            cur_members = []

    # Flush trailing
    if cur_names:
        _flush()

    return (
        block_names,
        block_means,
        block_positions,
        block_sems,
        block_n_stds,
        block_members,
    )


def _compute_reprod_block(
    all_samples: List[Sample],
    ratio_name: str,
    *,
    uncertainty_config: UncertaintyConfig,
    drift_model: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ratio_extractor: Optional[Callable[[Sample, str], Optional[CycleData]]] = None,
    position_extractor: Optional[Callable[[Sample], float]] = None,
) -> ReprodResult:
    """Compute u_std_repeatability using block means instead of individual standard means."""
    (block_names, block_means, block_positions,
     block_sems, block_n_stds, block_members) = _extract_block_data(
        all_samples,
        ratio_name,
        ratio_extractor=ratio_extractor,
        position_extractor=position_extractor,
    )

    n_blocks = len(block_names)
    method = uncertainty_config.resolve_reprod_method(
        enable_ssb=uncertainty_config.enable_ssb,
        enable_delta=uncertainty_config.enable_delta,
    )
    fallback_reason: Optional[str] = None

    if n_blocks == 0:
        return _empty_result(method)

    # Exclusion mask — matched against the block's actual member observations.
    # Splitting the joined display name and comparing the pieces could not see
    # the block key the UI emits ("A+B"), nor an observation-identity exclusion.
    excluded = set(uncertainty_config.excluded_standards)
    block_included = [
        not any(_standard_is_excluded(member, excluded) for member in members)
        for members in block_members
    ]
    block_identities = [
        [member.observation_id for member in members] for members in block_members
    ]
    n_included = sum(block_included)

    # All blocks in segment 1 (segmentation is per-standard; blocks aggregate)
    block_segments = [1] * n_blocks

    # Fallback chain
    if method == "drift_residuals" and drift_model is None:
        method = "loo_cross_validation"
        fallback_reason = "drift_residuals requested but no drift model; falling back to LOO"

    if method == "loo_cross_validation" and n_included < 3:
        method = "sd_of_means"
        fr = "LOO requires >= 3 included blocks; falling back to sd_of_means"
        fallback_reason = f"{fallback_reason}; {fr}" if fallback_reason else fr

    if n_included < 2:
        result = _empty_result(method)
        result.std_names = block_names
        result.std_means = np.array(block_means)
        result.std_positions = np.array(block_positions)
        result.std_included = block_included
        result.std_segments = block_segments
        result.std_identities = block_identities
        result.n_blocks = n_blocks
        result.block_means = np.array(block_means)
        result.block_positions = np.array(block_positions)
        result.fallback_reason = "< 2 included blocks; u_std_repeatability = 0 (incomplete)"
        return result

    # Route to strategy (reuse existing implementations — they work on any
    # list of names/means/positions, whether individual stds or blocks)
    if method == "sd_of_means":
        result = _sd_of_means(block_names, block_means, block_positions, block_included, block_segments)
    elif method == "loo_cross_validation":
        result = _loo_cross_validation(block_names, block_means, block_positions, block_included, block_segments)
    elif method == "drift_residuals":
        result = _drift_residuals(block_names, block_means, block_positions, block_included, block_segments, drift_model)
    elif method == "robust_mad":
        result = _robust_mad(block_names, block_means, block_positions, block_included, block_segments)
    else:
        result = _sd_of_means(block_names, block_means, block_positions, block_included, block_segments)
        fallback_reason = f"Unknown method '{uncertainty_config.reprod_method}'; falling back to sd_of_means"

    if fallback_reason:
        result.fallback_reason = fallback_reason

    incl_sems = []
    incl_dof_wb = 0
    for sem, n, inc in zip(block_sems, block_n_stds, block_included):
        if inc and n >= 2:
            incl_sems.append(sem)
            incl_dof_wb += n - 1

    if incl_sems:
        # Pooled SEM: sqrt(mean(SEM_i^2))
        pooled_sem = float(np.sqrt(np.mean(np.array(incl_sems) ** 2)))
        grand_mean = float(np.mean(np.array(block_means)[np.array(block_included)]))
        result.within_block_sem = pooled_sem
        result.within_block_sem_rel_permil = (
            (pooled_sem / grand_mean) * 1000.0 if np.isfinite(grand_mean) and grand_mean != 0 else 0.0
        )
        result.within_block_dof = incl_dof_wb

    result.block_means = np.array(block_means)
    result.block_positions = np.array(block_positions)
    result.n_blocks = n_blocks
    result.std_identities = block_identities

    if uncertainty_config.include_kappa_drift and method != "drift_residuals":
        kappa, deltas = _kappa_drift(block_means, block_included, block_segments)
        result.kappa_drift_permil = kappa
        result.drift_deltas = deltas

    return result

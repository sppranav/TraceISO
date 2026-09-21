"""Statistical outlier detection filters for TraceISO.

Cycle-level filters applied before averaging. Three methods are supported:
standard deviation (keep ``|x - mean| <= k*SD``), MAD (``|x - median| <= k*MAD``),
and IQR (Tukey fence ``Q1 - k*IQR .. Q3 + k*IQR``); ``"None"`` disables filtering.
The ``threshold`` argument is the multiplier ``k`` (default 2.0). Fewer than 3
cycles always pass through unfiltered, since the summary statistics are unstable
there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Dict, Optional, Tuple

import numpy as np

from config.constants import DEFAULT_OUTLIER_THRESHOLD_SD
from config.filtering import (
    FILTER_METHOD_IQR,
    FILTER_METHOD_MAD,
    FILTER_METHOD_NONE,
    FILTER_METHOD_STANDARD_DEVIATION,
    REMOVED_FILTER_METHODS,
    normalize_filter_method_name,
)


def make_sample_cycle_key(
    sample_name: str,
    run_number,
    *,
    identity_ordinal: Optional[int] = None,
) -> str:
    """Return the stable per-run key used for per-sample cycle ranges."""
    if run_number is None:
        key = str(sample_name)
    else:
        key = f"{sample_name}__run_{int(run_number)}"
    if identity_ordinal is not None:
        key = f"{key}__identity_{int(identity_ordinal)}"
    return key


def sample_cycle_key(sample) -> str:
    """Return the stable cycle-range key for a sample-like object."""
    metadata = getattr(sample, "metadata", None)
    identity_ordinal = (
        metadata.get("_identity_ordinal")
        if isinstance(metadata, dict)
        else None
    )
    return make_sample_cycle_key(
        str(getattr(sample, "name", "") or ""),
        getattr(sample, "run_number", None),
        identity_ordinal=identity_ordinal,
    )


class FilterMethod(str, Enum):
    """Supported statistical outlier detection methods."""

    NONE = FILTER_METHOD_NONE
    STANDARD_DEVIATION = FILTER_METHOD_STANDARD_DEVIATION
    MAD = FILTER_METHOD_MAD
    IQR = FILTER_METHOD_IQR


@dataclass
class FilterResult:
    """Result of applying an outlier filter."""

    mask: np.ndarray
    method: str
    threshold: float
    n_rejected: int

    @property
    def n_total(self) -> int:
        return len(self.mask)

    @property
    def n_kept(self) -> int:
        return int(np.sum(self.mask))


def apply_filter(
    values: np.ndarray,
    method: str = "None",
    threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
) -> FilterResult:
    """Apply a statistical outlier filter and return a boolean keep-mask.

    Passes all cycles through unchanged when *method* is ``"None"`` or fewer than
    3 values are present (statistics unstable). ``threshold`` is the method
    multiplier ``k``."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    method = _normalize_supported_method(method)
    active_threshold = _resolve_active_threshold(method, threshold)

    # Conservative global guard for all methods: with only 1–2 cycles the
    # summary statistics are undefined or degenerate, so TraceISO passes the
    # data through unchanged rather than applying an unstable filter.
    if n < 3 or method == FILTER_METHOD_NONE:
        mask = np.ones(n, dtype=bool)
        return FilterResult(mask=mask, method=method, threshold=active_threshold, n_rejected=0)

    mask = _dispatch(values, method, active_threshold)
    n_rejected = int(n - np.sum(mask))
    return FilterResult(mask=mask, method=method, threshold=active_threshold, n_rejected=n_rejected)


def _normalize_supported_method(method: str) -> str:
    """Return the canonical supported filter name or raise for removed methods."""
    raw = getattr(method, "value", method)
    raw = "" if raw is None else str(raw).strip()
    if not raw:
        return FILTER_METHOD_NONE
    if raw in REMOVED_FILTER_METHODS:
        raise ValueError(f"Outlier filter method has been removed from TraceISO: {raw}")
    normalized = normalize_filter_method_name(raw)
    supported = {
        FILTER_METHOD_NONE,
        FILTER_METHOD_STANDARD_DEVIATION,
        FILTER_METHOD_MAD,
        FILTER_METHOD_IQR,
    }
    if normalized not in supported:
        raise ValueError(f"Unsupported outlier filter method: {raw}")
    return normalized


def _resolve_active_threshold(
    method: str,
    threshold: float,
) -> float:
    """Resolve and validate the effective threshold for *method*."""
    if method == FILTER_METHOD_NONE:
        return threshold
    if threshold <= 0.0:
        raise ValueError(f"{method} threshold must be greater than 0.")
    return float(threshold)


def _dispatch(values: np.ndarray, method: str, threshold: float) -> np.ndarray:
    """Dispatch to the correct filter implementation."""
    if method == FILTER_METHOD_STANDARD_DEVIATION:
        return _standard_deviation(values, threshold)
    if method == FILTER_METHOD_MAD:
        return _mad(values, threshold)
    if method == FILTER_METHOD_IQR:
        return _iqr(values, threshold)
    raise ValueError(f"Unsupported outlier filter method: {method}")


def _inclusive_leq(values: np.ndarray, limit: float) -> np.ndarray:
    """Compare to an upper bound with a tiny FP guard band."""
    values = np.asarray(values, dtype=np.float64)
    scale = np.maximum(np.maximum(np.abs(values), abs(limit)), 1.0)
    tol = np.finfo(np.float64).eps * 8.0 * scale
    return values <= (limit + tol)


def _inclusive_geq(values: np.ndarray, limit: float) -> np.ndarray:
    """Compare to a lower bound with a tiny FP guard band."""
    values = np.asarray(values, dtype=np.float64)
    scale = np.maximum(np.maximum(np.abs(values), abs(limit)), 1.0)
    tol = np.finfo(np.float64).eps * 8.0 * scale
    return values >= (limit - tol)


def _standard_deviation(values: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if int(np.sum(finite)) < 3:
        return np.ones(len(values), dtype=bool)
    finite_values = values[finite]
    finite_mask = standard_deviation_keep_mask(finite_values, threshold)
    result = np.zeros(len(values), dtype=bool)
    result[finite] = finite_mask
    return result


def _running_error_gamma(n: int) -> float:
    """Higham's ``gamma_m = m*u / (1 - m*u)`` relative running-error factor
    (``u = eps/2``), evaluated at ``m = n + 8`` to cover centring, squared
    accumulation, the variance divide, the square root and the ``k`` multiply
    of the centred float evaluation (Higham, *Accuracy and Stability of
    Numerical Algorithms*, 2nd ed., 2.4 and 4.2).

    It is used *only* to decide when the centred float screen sits too close
    to the exact ``k*SD`` tie to be trusted, so the exact rational predicate
    must settle that point. It is a deliberate over-estimate for that
    dispatch, never the numerical contract, so a loose value only costs a
    slower exact pass. Returns ``inf`` once ``m*u >= 1`` (astronomically large
    ``n``), which forces the exact path unconditionally.
    """
    u = float(np.finfo(np.float64).eps) / 2.0
    m = max(int(n), 1) + 8
    denom = 1.0 - m * u
    if denom <= 0.0:
        return math.inf
    return (m * u) / denom


def _input_rounding_quantum(values: np.ndarray) -> float:
    """Return ``q``: half the largest ULP across the inputs and their span.

    Declared input model: each stored value is the correctly rounded image of
    some ideal real, so ``|e_i| <= 0.5 * ULP(x_i) <= q``. ``np.spacing`` on the
    magnitude yields the larger side at a power-of-two transition, and stays
    well-defined into the subnormal range. This models *one* rounding of the
    inputs; it makes no claim about multi-step upstream computation or repeated
    re-translation of the data.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    ulp = float(np.max(np.spacing(np.abs(finite))))
    span = float(np.max(finite) - np.min(finite))
    if math.isfinite(span):
        ulp = max(ulp, float(np.spacing(abs(span))))
    return ulp / 2.0


def _exact_sd_keep_mask(
    values: np.ndarray, k: float, inherited: float
) -> np.ndarray:
    """Exact rational ``|x - mean| <= k*SD`` decision on the stored floats.

    Every ``float`` is an exact rational, so ``mean``, the centred sum of
    squares ``ss = (n-1)*var`` and ``k`` carry no rounding error here. The
    inclusive test is evaluated on squared non-negative quantities
    (``(n-1)*dev_i^2 <= k^2 * ss``) to avoid an irrational square root. A point
    that fails the exact test is still retained when it lies within the
    input-rounding envelope ``inherited`` of the tie — conservative retention
    consistent with the stored floats being a rounding of ideal values that
    were exactly on the boundary.
    """
    fr = [Fraction(float(v)) for v in values]
    n = len(fr)
    mean = sum(fr) / n
    ss = sum((v - mean) * (v - mean) for v in fr)
    kk = Fraction(k) * Fraction(k)
    scale = n - 1
    rhs = kk * ss
    env = Fraction(inherited) if inherited > 0.0 else Fraction(0)
    keep = np.empty(n, dtype=bool)
    for i, v in enumerate(fr):
        dev = v - mean
        if dev < 0:
            dev = -dev
        if scale * dev * dev <= rhs:
            keep[i] = True
            continue
        slack = dev - env
        keep[i] = slack <= 0 or scale * slack * slack <= rhs
    return keep


def standard_deviation_keep_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    """Shared inclusive ``|x - mean| <= k*SD`` rule for the SD filter, the SD
    fallback of the MAD filter, and drift's ``"sd"`` standard screen.

    The decision is affine-stable within a declared floating-point domain.
    A centred float screen (median-anchored, so a large common offset is
    removed before the deviations and SD are formed) settles the ordinary,
    well-separated case. Any point whose centred margin ``|x - mean| - k*SD``
    falls inside the combined arithmetic (``_running_error_gamma``) plus
    input-rounding (``(2 + k*sqrt(n/(n-1)))*q``) envelope of the exact tie is
    handed to :func:`_exact_sd_keep_mask`, which decides it exactly and then
    retains it if it lies within the input-rounding envelope. Non-finite
    centred intermediates (extreme dynamic range) also route to the exact
    path, which cannot overflow. Conservative retention may keep a point
    marginally outside the exact threshold; it never rejects one that is
    inside. Rationale and bound derivation:
    ``project_support/audit_reports/W2_FILTER_POLICY_DECISION.md``.
    """
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    if n < 2 or not np.all(np.isfinite(values)):
        return np.ones(n, dtype=bool)

    k = float(threshold)

    # The centred float screen may legitimately overflow on extreme dynamic
    # range; that outcome is caught below and handed to the exact path, which
    # cannot overflow, so the transient numpy warning carries no information.
    with np.errstate(over="ignore", invalid="ignore"):
        centre = float(np.median(values))
        y = values - centre
        std = float(np.std(y, ddof=1))
        if not math.isfinite(std) or std == 0.0:
            return np.ones(n, dtype=bool)

        dev = np.abs(y - np.mean(y))
    limit = k * std

    q = _input_rounding_quantum(values)
    inherited = (2.0 + k * math.sqrt(n / (n - 1.0))) * q
    scale = max(float(np.max(dev)), abs(limit), 1.0)
    arith = _running_error_gamma(n) * scale

    if math.isfinite(limit) and bool(np.all(np.isfinite(dev))) and math.isfinite(arith):
        margin = dev - limit
        clear_keep = margin <= inherited - arith
        clear_reject = margin > inherited + arith
        if bool(np.all(clear_keep | clear_reject)):
            return np.asarray(clear_keep, dtype=bool)

    return _exact_sd_keep_mask(values, k, inherited)


def _mad(values: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if int(np.sum(finite)) < 3:
        return np.ones(len(values), dtype=bool)
    finite_values = values[finite]
    median = np.median(finite_values)
    mad = np.median(np.abs(finite_values - median))
    if not np.isfinite(mad) or mad == 0:
        finite_mask = standard_deviation_keep_mask(finite_values, threshold)
    else:
        finite_mask = _inclusive_leq(np.abs(finite_values - median), threshold * mad)
    result = np.zeros(len(values), dtype=bool)
    result[finite] = finite_mask
    return result


def _iqr(values: np.ndarray, threshold: float) -> np.ndarray:
    # Tukey-style fence: lower = Q1 - k·IQR, upper = Q3 + k·IQR. TraceISO uses
    # the caller-supplied multiplier directly; the default is 2.0×IQR.
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if int(np.sum(finite)) < 3:
        return np.ones(len(values), dtype=bool)
    finite_values = values[finite]
    q1 = np.percentile(finite_values, 25)
    q3 = np.percentile(finite_values, 75)
    iqr = q3 - q1
    if not np.isfinite(q1) or not np.isfinite(q3) or not np.isfinite(iqr):
        return np.ones(len(values), dtype=bool)
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    result = np.zeros(len(values), dtype=bool)
    result[finite] = (
        _inclusive_geq(finite_values, lower) & _inclusive_leq(finite_values, upper)
    )
    return result


def calculate_thresholds(
    values: np.ndarray,
    method: str = "None",
    threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
) -> Tuple[float, float, float]:
    """Compute (center, lower, upper) for visualising filter bounds."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    method = _normalize_supported_method(method)

    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float("nan"), float("nan")

    if method == FILTER_METHOD_NONE or threshold is None:
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        return mean, mean - std, mean + std

    if method == FILTER_METHOD_STANDARD_DEVIATION:
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        return mean, mean - threshold * std, mean + threshold * std

    if method == FILTER_METHOD_MAD:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if not np.isfinite(mad) or mad == 0:
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            return mean, mean - threshold * std, mean + threshold * std
        return median, median - threshold * mad, median + threshold * mad

    if method == FILTER_METHOD_IQR:
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        median = float(np.median(values))
        return median, q1 - threshold * iqr, q3 + threshold * iqr

    raise ValueError(f"Unsupported outlier filter method: {method}")


@dataclass
class DescriptiveStats:
    """Basic descriptive statistics for an array of values."""

    mean: float
    sd: float
    se: float
    rsd_percent: float
    min_val: float
    max_val: float
    n: int


def describe_stats(
    values: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> DescriptiveStats:
    """Compute descriptive statistics on *values* filtered by *mask*."""
    values = np.asarray(values, dtype=np.float64)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if len(mask) == len(values):
            filtered = values[mask]
        else:
            # Mask length mismatch — treat values as pre-filtered
            filtered = values
    else:
        filtered = values

    filtered = filtered[np.isfinite(filtered)]
    n = len(filtered)
    if n == 0:
        nan = float("nan")
        return DescriptiveStats(nan, nan, nan, nan, nan, nan, 0)

    mean = float(np.mean(filtered))
    sd = float(np.std(filtered, ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    rsd = abs(sd / mean * 100.0) if np.isfinite(mean) and mean != 0 else 0.0
    min_val = float(np.min(filtered))
    max_val = float(np.max(filtered))

    return DescriptiveStats(
        mean=mean, sd=sd, se=se, rsd_percent=rsd,
        min_val=min_val, max_val=max_val, n=n,
    )


# Cycle-range-aware filtering

def get_filtered_values(
    values: np.ndarray,
    mask: np.ndarray,
    sample_name: str,
    cycle_ranges: Optional[Dict] = None,
    sample_key: Optional[str] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
) -> np.ndarray:
    """Get valid values with optional cycle range and re-filtering."""
    if mask is None:
        mask = np.ones(len(values), dtype=bool)

    if not cycle_ranges:
        return values[mask]

    cycle_range = resolve_cycle_range(
        cycle_ranges,
        sample_name=sample_name,
        sample_key=sample_key,
    )
    if cycle_range is None:
        return values[mask]
    if int(cycle_range[0]) <= 1 and int(cycle_range[1]) >= len(values):
        return values[mask]

    # Extract subset within cycle range (1-indexed → 0-indexed)
    start_idx = cycle_range[0] - 1
    end_idx = cycle_range[1]
    subset = values[start_idx:end_idx]
    subset_mask = mask[start_idx:end_idx] if mask is not None else np.ones(len(subset), dtype=bool)

    # Keep only currently valid (mask=True) and finite values
    finite_mask = np.isfinite(subset)
    finite_vals = subset[np.asarray(subset_mask, dtype=bool) & finite_mask]

    if len(finite_vals) < 3 or filter_method == "None":
        return finite_vals

    result = apply_filter(finite_vals, filter_method, filter_threshold)
    return finite_vals[result.mask]


def get_runtime_mask(
    values: np.ndarray,
    mask: np.ndarray,
    sample_name: str,
    cycle_ranges: Optional[Dict] = None,
    sample_key: Optional[str] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
) -> np.ndarray:
    """Return the runtime-valid boolean mask for the active cycle window."""
    runtime_mask = np.asarray(mask, dtype=bool).copy() if mask is not None else np.ones(len(values), dtype=bool)

    if not cycle_ranges:
        return runtime_mask

    cycle_range = resolve_cycle_range(
        cycle_ranges,
        sample_name=sample_name,
        sample_key=sample_key,
    )
    if cycle_range is None:
        return runtime_mask
    if int(cycle_range[0]) <= 1 and int(cycle_range[1]) >= len(values):
        return runtime_mask

    start_idx = cycle_range[0] - 1
    end_idx = cycle_range[1]

    # Invalidate cycles outside the active window — callers that need the full
    # mask for display purposes should use the incoming mask directly.
    runtime_mask[:start_idx] = False
    runtime_mask[end_idx:] = False

    subset = np.asarray(values, dtype=np.float64)[start_idx:end_idx]
    subset_mask = runtime_mask[start_idx:end_idx].copy()

    finite_visible_mask = subset_mask & np.isfinite(subset)
    valid_indices = np.where(finite_visible_mask)[0]
    if len(valid_indices) < 3 or filter_method == "None":
        return runtime_mask

    filtered = apply_filter(subset[valid_indices], filter_method, filter_threshold)
    subset_mask[valid_indices] = filtered.mask
    runtime_mask[start_idx:end_idx] = subset_mask
    return runtime_mask


def resolve_cycle_range(
    cycle_ranges: Optional[Dict],
    *,
    sample_name: str,
    sample_key: Optional[str] = None,
):
    """Resolve the active cycle range for a sample."""
    if not cycle_ranges:
        return None

    lookup_keys = []
    if sample_key:
        lookup_keys.append(sample_key)
    if sample_name:
        lookup_keys.append(sample_name)

    for key in lookup_keys:
        cycle_range = cycle_ranges.get(key)
        if cycle_range is not None:
            return cycle_range

    cycle_range = cycle_ranges.get("__global__")
    if cycle_range is not None:
        return cycle_range

    # A modern per-run key that is absent must not adopt another run's
    # window merely because the display name is shared.
    if sample_key and "__run_" in sample_key:
        return None

    if sample_name:
        prefix = f"{sample_name}__"
        matches = [value for key, value in cycle_ranges.items() if key.startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            import logging
            logging.getLogger(__name__).debug(
                "Ambiguous legacy cycle-range prefix for sample %r; ignoring %d matches.",
                sample_name,
                len(matches),
            )

    return None

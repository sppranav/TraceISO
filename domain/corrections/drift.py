"""Drift correction for TraceISO."""

from __future__ import annotations

from datetime import date, datetime
from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from numpy.exceptions import RankWarning
except ImportError:  # pragma: no cover - compatibility with NumPy < 1.25
    from numpy import RankWarning

from config.constants import DEFAULT_OUTLIER_THRESHOLD_SD
from domain.filters.outlier import (
    get_filtered_values,
    sample_cycle_key,
    standard_deviation_keep_mask,
)
from domain.models import CycleData, Sample
from domain.output_scale import record_drift_output_factor
from domain.ratio_selection import (
    get_best_pre_drift_ratio_data,
    select_best_pre_drift_ratio_layer,
)


@dataclass
class DriftResult:
    """Result of drift correction."""

    samples: List[Sample]
    fit_info: Dict
    warnings: List[str]


def fitted_mean_band(fit_info: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return x and ±2 estimated-SE half-widths for the stored OLS mean fit.

    The covariance belongs to the centered/scaled polynomial basis recorded by
    the producer.  An empty array plus a reason is returned for legacy,
    externally supplied, weighted/correlated, or incomplete fits.
    """
    if fit_info.get("fit_provenance") != "same_data_ols_v1":
        return np.array([]), np.array([]), "same-data OLS fit provenance is unavailable"
    if fit_info.get("error_model") != "independent_homoscedastic_normal":
        return np.array([]), np.array([]), "the fitted error model is unsupported"
    if bool(fit_info.get("weighted", False)) or bool(fit_info.get("correlated", False)):
        return np.array([]), np.array([]), "weighted/correlated fit bands are unsupported"
    try:
        coeffs = np.asarray(fit_info["coeffs"], dtype=float)
        covariance = np.asarray(fit_info["coefficient_covariance"], dtype=float)
        x_fit = np.asarray(fit_info["x_fit"], dtype=float)
        residual_dof = int(fit_info["residual_dof"])
        center = float(fit_info["x_center"])
        scale = float(fit_info["x_scale"])
    except (KeyError, TypeError, ValueError):
        return np.array([]), np.array([]), "coefficient covariance metadata is incomplete"
    if (
        residual_dof <= 0
        or scale == 0.0
        or covariance.shape != (len(coeffs), len(coeffs))
        or not np.all(np.isfinite(covariance))
        or not np.all(np.isfinite(x_fit))
    ):
        return np.array([]), np.array([]), "coefficient covariance or residual DoF is invalid"
    z_fit = (x_fit - center) / scale
    design = np.vander(z_fit, N=len(coeffs), increasing=False)
    variances = np.einsum("ij,jk,ik->i", design, covariance, design)
    tolerance = np.finfo(float).eps * max(float(np.max(np.abs(variances))), 1.0) * 32.0
    if np.any(variances < -tolerance):
        return np.array([]), np.array([]), "coefficient covariance yields negative variance"
    half_width = 2.0 * np.sqrt(np.maximum(variances, 0.0))
    return x_fit, half_width, ""


def _safe_rsd_percent(
    values: np.ndarray,
    *,
    label: str,
    warnings: List[str],
) -> float:
    """Return RSD% for a series without propagating zero-mean NaNs."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0

    mean_val = float(np.mean(arr))
    if not np.isfinite(mean_val) or abs(mean_val) < 1e-15:
        warnings.append(
            f"{label} RSD% unavailable because the mean is zero or non-finite."
        )
        return 0.0

    return float(np.std(arr, ddof=1) / mean_val * 100.0)


def apply_drift_correction(
    samples: List[Sample],
    ratio_name: str,
    method: str = "polynomial",
    degree: int = 2,
    x_axis: str = "run_number",
    apply_outlier_filter: bool = True,
    outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
    outlier_method: str = "mad",
    norm_mode: str = "average",
    norm_standard: int = 0,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> DriftResult:
    """Apply drift correction to a list of samples."""
    warnings: List[str] = []

    active_samples = [s for s in samples if not s.metadata.get("excluded", False)]
    try:
        x_value_by_sample = build_drift_x_value_map(samples, x_axis, active_samples=active_samples)
    except ValueError as exc:
        warnings.append(f"Drift x-axis build failed ({exc}). Skipping drift correction.")
        return DriftResult(samples=samples, fit_info={}, warnings=warnings)

    # Extract standards
    standards = [s for s in active_samples if s.is_standard]

    if len(standards) < 3:
        warnings.append(
            f"Need ≥3 standards for drift correction (found {len(standards)}). Skipping."
        )
        return DriftResult(samples=samples, fit_info={}, warnings=warnings)

    # Extract X values for standards using the same coordinate basis that
    # will be used later for correcting all samples.
    x_all = np.asarray([x_value_by_sample[id(s)] for s in standards], dtype=np.float64)
    y_all = []
    valid_standards = []
    standard_source_layers: List[str] = []

    # Extract Y values (ratio means)
    for std in standards:
        selected_layer = select_best_pre_drift_ratio_layer(std, ratio_name)
        ratio_data = selected_layer.data if selected_layer is not None else None
        valid_values = _get_active_ratio_values(
            ratio_data,
            std.name,
            cycle_ranges=cycle_ranges,
            sample_key=sample_cycle_key(std),
        )
        if len(valid_values) > 0:
            y_all.append(float(np.nanmean(valid_values)))
            valid_standards.append(std)
            standard_source_layers.append(selected_layer.key)
        else:
            y_all.append(np.nan)

    if len([y for y in y_all if not np.isnan(y)]) < 3:
        warnings.append(
            f"Need ≥3 valid standards for '{ratio_name}'. Skipping drift correction."
        )
        return DriftResult(samples=samples, fit_info={}, warnings=warnings)

    unique_source_layers = set(standard_source_layers)
    if len(unique_source_layers) != 1:
        warnings.append(
            f"Drift correction for '{ratio_name}' found standards on mixed ratio layers "
            f"({', '.join(sorted(unique_source_layers))}). Skipping drift correction."
        )
        return DriftResult(samples=samples, fit_info={}, warnings=warnings)
    drift_source_layer = standard_source_layers[0]

    # Remove NaNs
    x_all_clean = [x for x, y in zip(x_all, y_all) if not np.isnan(y)]
    y_all_clean = [y for y in y_all if not np.isnan(y)]
    x_all = np.array(x_all_clean, dtype=np.float64)
    y_all = np.array(y_all_clean, dtype=np.float64)

    rsd_before = _safe_rsd_percent(
        y_all,
        label="Pre-correction standard",
        warnings=warnings,
    )

    # Outlier filter on standards (robust MAD by default, or classic SD)
    if apply_outlier_filter and len(y_all) >= 3:
        outlier_mask = detect_outliers_standard_deviation(
            y_all, threshold=outlier_threshold, method=outlier_method,
        )
        x_data = x_all[outlier_mask]
        y_data = y_all[outlier_mask]

        n_outliers = int(np.sum(~outlier_mask))
        if n_outliers > 0:
            warnings.append(f"Removed {n_outliers} outlier standard(s) before fitting.")
    else:
        x_data = x_all
        y_data = y_all
        outlier_mask = np.ones(len(x_all), dtype=bool)

    if len(x_data) < 2:
        warnings.append("Not enough standards after outlier filtering. Skipping drift correction.")
        return DriftResult(samples=samples, fit_info={}, warnings=warnings)

    if method not in ("linear", "polynomial"):
        warnings.append(f"Unknown drift method '{method}'. Skipping.")
        return DriftResult(samples=samples, fit_info={}, warnings=warnings)

    # Enforce scientifically admissible model: degree must be < n_points
    effective_degree = degree if method == "polynomial" else 1
    n_fit_points = len(x_data)
    if effective_degree >= n_fit_points:
        warnings.append(
            f"Polynomial degree {effective_degree} requires at least "
            f"{effective_degree + 1} standards, but only {n_fit_points} are available "
            f"after outlier filtering. Skipping drift correction."
        )
        return DriftResult(samples=samples, fit_info={}, warnings=warnings)

    # Fit model
    import warnings as _warnings
    try:
        fit_degree = 1 if (method == "linear" or (method == "polynomial" and degree == 1)) else degree
        x_center = float(np.mean(x_data))
        x_scale = float(np.max(np.abs(x_data - x_center)))
        if not np.isfinite(x_center) or not np.isfinite(x_scale):
            raise ValueError("drift coordinates are non-finite")
        if x_scale == 0.0:
            x_scale = 1.0
        z_data = (x_data - x_center) / x_scale
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            coeffs = np.polyfit(z_data, y_data, deg=fit_degree)

        # Check for rank-deficient fits (numpy emits RankWarning)
        rank_warnings = [w for w in caught if issubclass(w.category, RankWarning)]
        if rank_warnings:
            warnings.append(
                f"Drift fit is rank-deficient (degree={fit_degree}, "
                f"n_standards={n_fit_points}). Skipping drift correction."
            )
            return DriftResult(samples=samples, fit_info={}, warnings=warnings)

        design = np.vander(z_data, N=fit_degree + 1, increasing=False)
        design_rank = int(np.linalg.matrix_rank(design))
        if design_rank != fit_degree + 1:
            warnings.append(
                f"Drift fit design is rank-deficient (rank={design_rank}, "
                f"parameters={fit_degree + 1}). Skipping drift correction."
            )
            return DriftResult(samples=samples, fit_info={}, warnings=warnings)
        fitted_on_support = design @ coeffs
        residual_vector = y_data - fitted_on_support
        residual_sse = float(residual_vector @ residual_vector)
        residual_dof = int(n_fit_points - design_rank)
        if residual_dof <= 0:
            warnings.append(
                "Drift fit has no positive residual degrees of freedom. "
                "Skipping drift correction."
            )
            return DriftResult(samples=samples, fit_info={}, warnings=warnings)
        residual_variance = residual_sse / residual_dof
        coefficient_covariance = residual_variance * np.linalg.inv(design.T @ design)
        residual_sd = float(np.sqrt(max(residual_variance, 0.0)))

        def fitted_func(x):
            x_arr = np.asarray(x, dtype=np.float64)
            return np.polyval(coeffs, (x_arr - x_center) / x_scale)
    except Exception as e:
        warnings.append(f"Drift fitting failed: {e}")
        return DriftResult(samples=samples, fit_info={}, warnings=warnings)

    # Mean of standards (target for normalization)
    if norm_mode == "single":
        # Use ratio mean of one specific standard; restrict to post-outlier standards
        # so an outlier cannot be selected as the normalization reference.
        surviving_standards = [s for s, keep in zip(valid_standards, outlier_mask) if keep]
        if not surviving_standards:
            surviving_standards = valid_standards
        try:
            requested_idx = int(norm_standard)
        except (TypeError, ValueError):
            requested_idx = 0
        idx = min(max(requested_idx, 0), len(surviving_standards) - 1)
        if requested_idx != idx or requested_idx != norm_standard:
            warnings.append(
                f"Reference standard index {norm_standard} is out of range after outlier "
                f"filtering ({len(surviving_standards)} standard(s) remain); "
                f"substituted '{surviving_standards[idx].name}' (index {idx})."
            )
        ref_std = surviving_standards[idx]
        ref_ratio = get_best_pre_drift_ratio_data(ref_std, ratio_name)
        ref_values = _get_active_ratio_values(
            ref_ratio,
            ref_std.name,
            cycle_ranges=cycle_ranges,
            sample_key=sample_cycle_key(ref_std),
        )
        if len(ref_values) > 0:
            std_mean = float(np.nanmean(ref_values))
        else:
            std_mean = float(np.mean(y_data))
            warnings.append(
                "Selected reference standard has no valid data; "
                "falling back to average of all standards."
            )
    else:
        std_mean = float(np.mean(y_data))

    samples_to_correct = [s for s in active_samples if s.is_standard or s.is_sample]
    for s in active_samples:
        if not s.is_standard and not s.is_sample and not s.is_blank:
            # Type eligibility is sample provenance; fit and correction failures
            # remain run-level notices in DriftResult.warnings.
            s.warnings.append(
                f"Drift correction skipped for '{s.name}' ({ratio_name}): "
                f"sample type '{s.sample_type}' is not corrected by drift."
            )

    for sample in samples_to_correct:
        x_sample = x_value_by_sample[id(sample)]
        selected_layer = select_best_pre_drift_ratio_layer(sample, ratio_name)
        ratio_data = selected_layer.data if selected_layer is not None else None

        if ratio_data is None or ratio_data.n_valid == 0:
            continue
        if selected_layer.key != drift_source_layer:
            warnings.append(
                f"Sample '{sample.name}': drift source layer '{selected_layer.key}' does "
                f"not match standard layer '{drift_source_layer}'. Skipping."
            )
            continue

        try:
            fitted_value = float(fitted_func(x_sample))

            # Guard: reject non-finite, near-zero, or sign-flipped trend
            if not np.isfinite(fitted_value):
                warnings.append(
                    f"Sample '{sample.name}': fitted drift value is non-finite. Skipping."
                )
                continue
            if abs(fitted_value) < 1e-15:
                warnings.append(
                    f"Sample '{sample.name}': fitted drift value is near-zero "
                    f"({fitted_value:.2e}). Skipping to avoid division instability."
                )
                continue
            if std_mean == 0 or np.sign(std_mean) != np.sign(fitted_value):
                warnings.append(
                    f"Sample '{sample.name}': fitted drift value "
                    f"({fitted_value:.6f}) is inconsistent with standard mean "
                    f"({std_mean:.6f}). Skipping."
                )
                continue

            # Correction: corrected = raw / fitted_trend * mean(standards)
            drift_factor = std_mean / fitted_value
            corrected_values = ratio_data.values * drift_factor

            sample.drift_corrected_ratios[ratio_name] = CycleData(
                values=corrected_values,
                mask=ratio_data.mask.copy(),
            )
            # The correction is one committed scalar per sample and ratio.
            # Uncertainty contributors evaluated on the pre-drift layer must
            # carry this same factor, so it is recorded rather than re-derived.
            record_drift_output_factor(sample, ratio_name, drift_factor)
        except Exception as e:
            warnings.append(f"Failed to correct sample '{sample.name}': {e}")

    corrected_stds = []
    for std in valid_standards:
        if ratio_name in std.drift_corrected_ratios:
            corr_data = std.drift_corrected_ratios[ratio_name]
            corr_values = _get_active_ratio_values(
                corr_data,
                std.name,
                cycle_ranges=cycle_ranges,
                sample_key=sample_cycle_key(std),
            )
            if len(corr_values) > 0:
                corrected_stds.append(float(np.nanmean(corr_values)))

    if len(corrected_stds) < 2:
        warnings.append(
            "Post-correction standard RSD% unavailable because fewer than "
            "two standards were corrected."
        )
    rsd_after = _safe_rsd_percent(
        np.asarray(corrected_stds, dtype=np.float64),
        label="Post-correction standard",
        warnings=warnings,
    )

    x_fit = np.linspace(float(np.min(x_all)), float(np.max(x_all)), 100)
    y_fit = fitted_func(x_fit)

    # Freeze the per-sample x-coordinates so that runtime budget recomputation
    # is not affected by subsequent exclusion changes (item 142). The map is
    # keyed by observation identity: run numbers may repeat, and on an index or
    # time axis two observations sharing a run number sit at genuinely
    # different coordinates, so a run-keyed map silently collapses them onto
    # whichever one was written last.
    sample_x_by_observation: Dict[str, float] = {
        str(s.observation_id): x_value_by_sample[id(s)]
        for s in active_samples
        if id(s) in x_value_by_sample
    }
    # Retained for readers of an existing fit_info. Same collapse as before; it
    # is no longer what the engines consume.
    sample_x_by_run: Dict[int, float] = {
        int(s.run_number): x_value_by_sample[id(s)]
        for s in active_samples
        if id(s) in x_value_by_sample
    }

    fit_info = {
        "x_data": x_data.tolist(),
        "y_data": y_data.tolist(),
        "x_all": x_all.tolist(),
        "y_all": y_all.tolist(),
        "outlier_mask": outlier_mask.tolist() if isinstance(outlier_mask, np.ndarray) else [True] * len(x_all),
        "x_fit": x_fit.tolist(),
        "y_fit": y_fit.tolist(),
        "coeffs": coeffs.tolist() if coeffs is not None else None,
        "coefficient_covariance": coefficient_covariance.tolist(),
        "fit_provenance": "same_data_ols_v1",
        "error_model": "independent_homoscedastic_normal",
        "weighted": False,
        "correlated": False,
        "fitted_observations": n_fit_points,
        "design_rank": design_rank,
        "residual_dof": residual_dof,
        "residual_sse": residual_sse,
        "residual_sd": residual_sd,
        "fit_format": "centered_scaled_v1",
        "x_center": x_center,
        "x_scale": x_scale,
        "method": method,
        "rsd_before": round(rsd_before, 4),
        "rsd_after": round(rsd_after, 4),
        "std_mean": round(std_mean, 6),
        "ratio_name": ratio_name,
        "source_layer": drift_source_layer,
        "x_axis": x_axis,
        "sample_x_by_run": sample_x_by_run,
        "sample_x_by_observation": sample_x_by_observation,
    }

    return DriftResult(samples=samples, fit_info=fit_info, warnings=warnings)


def build_drift_x_value_map(
    samples: List[Sample],
    x_axis: str,
    *,
    active_samples: Optional[List[Sample]] = None,
) -> Dict[int, float]:
    """Map each sample identity to its drift x-coordinate.

    Raises ``ValueError`` when the resulting x-coordinates are non-monotonic
    in run order, which would corrupt polynomial fitting (item 153).
    """
    target_samples = active_samples if active_samples is not None else samples
    if x_axis == "run_number":
        result = {id(s): float(s.run_number) for s in target_samples}
    elif x_axis == "index":
        result = {id(s): float(i) for i, s in enumerate(target_samples)}
    elif x_axis == "time_minutes":
        result = _build_time_minutes_map(samples, target_samples)
    else:
        result = {id(s): float(s.run_number) for s in target_samples}

    # Assert run-order monotonicity for the time_minutes axis only (item 153):
    # timestamp-derived coordinates must increase in run order; a violation
    # signals a midnight rollover that survived correction or a parsing error.
    # run_number and index modes are not checked here — run numbers may repeat
    # legitimately for sub-runs, and index coordinates are always 0,1,2,…
    if x_axis == "time_minutes":
        xs = [result[id(s)] for s in target_samples if id(s) in result]
        for i in range(1, len(xs)):
            if xs[i] < xs[i - 1]:
                raise ValueError(
                    f"Drift time_minutes axis is non-monotonic at position {i} "
                    f"({xs[i - 1]:.4g} min → {xs[i]:.4g} min). "
                    "Check timestamps for unresolved midnight rollovers."
                )
    return result


def supports_time_axis(samples: List[Sample]) -> bool:
    """Return True when every sample has a parseable timestamp."""
    if not samples:
        return False
    return all(_extract_sample_timestamp(sample) is not None for sample in samples)


def get_time_axis_debug_info(samples: List[Sample]) -> Dict[str, object]:
    """Return debug information explaining time-axis availability."""
    if not samples:
        return {"available": False, "reason": "No samples loaded.", "missing_samples": []}

    missing_samples = [
        sample.name
        for sample in samples
        if _extract_sample_timestamp(sample) is None
    ]
    if missing_samples:
        return {
            "available": False,
            "reason": (
                f"{len(missing_samples)} sample(s) do not have a parseable timestamp. "
                "Time axis stays hidden until all samples can be timed."
            ),
            "missing_samples": missing_samples,
        }

    first_timestamp = min(_extract_sample_timestamp(sample) for sample in samples)
    return {
        "available": True,
        "reason": "Time axis available.",
        "missing_samples": [],
        "first_timestamp": first_timestamp.isoformat(sep=' '),
    }


def _build_time_minutes_map(
    all_samples: List[Sample],
    target_samples: List[Sample],
) -> Dict[int, float]:
    """Map samples to elapsed minutes from the first timestamp in the file.

    Detects midnight rollovers by adding +24 h whenever a time-only timestamp
    is earlier than the previous one in run order (item 140).
    """
    from datetime import timedelta

    timestamps = [(sample, _extract_sample_timestamp(sample)) for sample in all_samples]
    valid = [(sample, ts) for sample, ts in timestamps if ts is not None]
    if len(valid) != len(all_samples):
        raise ValueError("Elapsed-time drift axis requires timestamps for all samples.")

    # Correct midnight rollovers in run order: if timestamp decreases, advance
    # it by one day. This handles time-only stamps like "23:58" / "00:02".
    corrected: List[datetime] = []
    offset = timedelta(0)
    prev_ts: Optional[datetime] = None
    for _sample, ts in valid:
        adjusted = ts + offset
        if prev_ts is not None and adjusted < prev_ts:
            offset += timedelta(hours=24)
            adjusted = ts + offset
        corrected.append(adjusted)
        prev_ts = adjusted

    t0 = corrected[0]
    target_ids = {id(s) for s in target_samples}
    return {
        id(sample): (adj - t0).total_seconds() / 60.0
        for (sample, _), adj in zip(valid, corrected)
        if id(sample) in target_ids
    }


def _extract_sample_timestamp(sample: Sample) -> Optional[datetime]:
    """Best-effort timestamp extraction from sample metadata."""
    metadata = sample.metadata or {}
    file_date = _extract_date_from_metadata_or_filename(metadata)

    for key in (
        "timestamp",
        "time",
        "datetime",
        "date_time",
        "acquisition_time",
        "acquisition_datetime",
        "start_time",
        "analysis_time",
        "measurement_time",
    ):
        value = _lookup_metadata_value(metadata, key)
        parsed = _parse_datetime_value(value, fallback_date=file_date)
        if parsed is not None:
            return parsed

    date_value = _lookup_metadata_value(metadata, "date")
    time_value = _lookup_metadata_value(metadata, "time")
    if date_value is not None and time_value is not None:
        parsed = _parse_datetime_value(f"{date_value} {time_value}", fallback_date=file_date)
        if parsed is not None:
            return parsed

    for key, value in metadata.items():
        normalized = str(key).strip().lower().replace(" ", "_")
        if any(token in normalized for token in ("time", "date", "timestamp")):
            parsed = _parse_datetime_value(value, fallback_date=file_date)
            if parsed is not None:
                return parsed

    return None


def _lookup_metadata_value(metadata: Dict[str, object], wanted_key: str) -> Optional[object]:
    """Case-insensitive metadata lookup with normalized separators."""
    target = wanted_key.strip().lower().replace(" ", "_")
    for key, value in metadata.items():
        normalized = str(key).strip().lower().replace(" ", "_")
        if normalized == target:
            return value
    return None


def _extract_date_from_metadata_or_filename(metadata: Dict[str, object]) -> Optional[date]:
    """Infer an analysis date from metadata or the source filename."""
    for key in ("date", "analysis_date", "acquisition_date", "measurement_date"):
        value = _lookup_metadata_value(metadata, key)
        if value is None:
            continue
        text = str(value).strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%m/%d/%Y", "%Y%m%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue

    source_name = _lookup_metadata_value(metadata, "_source_file_name")
    if source_name is None:
        return None
    match = re.search(r"(20\d{2})(\d{2})(\d{2})", str(source_name))
    if match is None:
        return None
    try:
        return datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        ).date()
    except ValueError:
        return None


def _parse_datetime_value(
    value: object,
    *,
    fallback_date: Optional[date] = None,
) -> Optional[datetime]:
    """Parse common timestamp representations from metadata."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        if numeric > 1e12:
            numeric /= 1000.0
        try:
            return datetime.fromtimestamp(numeric)
        except (OverflowError, OSError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None

    time_match = re.search(r"(\d{2}:\d{2}:\d{2})(?:[:.](\d{1,6}))?", text)
    if time_match and fallback_date is not None:
        time_text = time_match.group(1)
        frac = time_match.group(2)
        fmt = "%H:%M:%S"
        if frac:
            time_text = f"{time_text}.{frac.ljust(6, '0')}"
            fmt = "%H:%M:%S.%f"
        try:
            parsed_time = datetime.strptime(time_text, fmt).time()
            return datetime.combine(fallback_date, parsed_time)
        except ValueError:
            pass

    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _get_active_ratio_values(
    cycle_data: Optional[CycleData],
    sample_name: str,
    *,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    sample_key: Optional[str] = None,
) -> np.ndarray:
    """Return valid ratio values within the active Inspector window."""
    if cycle_data is None:
        return np.array([], dtype=np.float64)

    return np.asarray(
        get_filtered_values(
            cycle_data.values,
            cycle_data.mask,
            sample_name,
            cycle_ranges=cycle_ranges,
            sample_key=sample_key,
            filter_method="None",
            filter_threshold=2.0,
        ),
        dtype=np.float64,
    )


def detect_outliers_standard_deviation(
    data: np.ndarray,
    threshold: float = 2.0,
    method: str = "mad",
) -> np.ndarray:
    """Flag standards that are outliers for the drift fit (keep-mask of bools).

    ``method="mad"`` (default) uses robust median/MAD z-scores so a gross
    outlier cannot inflate the scale enough to mask itself (item 148); MAD is
    rescaled by 1/0.6745 to match SD for normal data, falling back to the
    global-SD rule only when MAD is zero. ``method="sd"`` uses the classic
    global mean/SD rule — kept for SOPs that specify 2-SD rejection, but note it
    is susceptible to masking when the standard count is low.
    """
    data = np.asarray(data, dtype=np.float64)

    def _sd_rule(arr: np.ndarray) -> np.ndarray:
        finite = np.isfinite(arr)
        if int(np.sum(finite)) < 3:
            return np.ones(len(arr), dtype=bool)
        result = np.zeros(len(arr), dtype=bool)
        result[finite] = standard_deviation_keep_mask(arr[finite], threshold)
        return result

    if str(method).strip().lower() == "sd":
        return _sd_rule(data)

    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if np.isfinite(mad) and mad > 0:
        # Consistent estimator of sigma for normal data: MAD / 0.6745
        robust_scale = mad / 0.6745
        return np.abs(data - median) <= threshold * robust_scale
    # MAD == 0 (all values identical): fall back to the global-SD rule
    return _sd_rule(data)


# Backward-compatible alias for older imports.
_detect_outliers_standard_deviation = detect_outliers_standard_deviation

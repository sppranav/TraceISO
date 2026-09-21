"""Blank-only QC diagnostics on the producer's applied correction layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from config.constants import DEFAULT_OUTLIER_THRESHOLD_SD
from domain.filters.outlier import get_runtime_mask, sample_cycle_key
from domain.models import Sample
from domain.ratio_selection import get_processing_ratio_data


@dataclass(frozen=True)
class BlankQCDiagnostic:
    sample_name: str
    run_number: int
    observation_id: str
    ratio_name: str
    quantity: str
    isotope: str
    value_percent: Optional[float]
    status: str
    reason: str = ""


def blank_qc_diagnostics(
    samples: Iterable[Sample],
    ratio_name: str,
    *,
    cycle_ranges: Optional[Dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
) -> Tuple[BlankQCDiagnostic, ...]:
    """Return per-observation isotope loads and signed immediate ratio change.

    ``S_i`` is the supported raw signal mean and ``B_i`` is the mean of
    ``raw - blank_corrected`` on the same accepted cycles.  The ratio estimator
    is ratio-of-supported-isotope-means immediately before and after blank
    subtraction; downstream interference, normalization and drift layers are
    intentionally excluded.
    """
    parts = ratio_name.split("/")
    if len(parts) != 2:
        return ()
    numerator, denominator = parts
    output = []
    for sample in samples:
        if sample.is_blank or sample.metadata.get("excluded", False):
            continue
        raw_num = sample.intensities.get(numerator)
        raw_den = sample.intensities.get(denominator)
        after_num = sample.blank_corrected_intensities.get(numerator)
        after_den = sample.blank_corrected_intensities.get(denominator)
        processing_ratio = get_processing_ratio_data(sample, ratio_name)
        common = (raw_num, raw_den, after_num, after_den, processing_ratio)
        if any(item is None for item in common):
            output.append(BlankQCDiagnostic(
                sample.name, sample.run_number, sample.observation_id, ratio_name,
                "ratio_change", "", None, "unavailable",
                "raw, blank-corrected, or processing-ratio channels are missing",
            ))
            continue
        lengths = {len(item.values) for item in common}  # type: ignore[union-attr]
        if len(lengths) != 1:
            output.append(BlankQCDiagnostic(
                sample.name, sample.run_number, sample.observation_id, ratio_name,
                "ratio_change", "", None, "unavailable",
                "raw, blank-corrected, and ratio series cannot be aligned",
            ))
            continue
        mask = get_runtime_mask(
            processing_ratio.values,
            processing_ratio.mask,
            sample.name,
            cycle_ranges=cycle_ranges,
            sample_key=sample_cycle_key(sample),
            filter_method=filter_method,
            filter_threshold=filter_threshold,
        )
        mask &= raw_num.mask & raw_den.mask & after_num.mask & after_den.mask
        arrays = [raw_num.values, raw_den.values, after_num.values, after_den.values]
        mask &= np.logical_and.reduce([np.isfinite(values) for values in arrays])
        if not np.any(mask):
            output.append(BlankQCDiagnostic(
                sample.name, sample.run_number, sample.observation_id, ratio_name,
                "ratio_change", "", None, "unavailable", "no common accepted cycles",
            ))
            continue

        raw_means = {
            numerator: float(np.mean(raw_num.values[mask])),
            denominator: float(np.mean(raw_den.values[mask])),
        }
        after_means = {
            numerator: float(np.mean(after_num.values[mask])),
            denominator: float(np.mean(after_den.values[mask])),
        }
        for isotope in (numerator, denominator):
            signal = raw_means[isotope]
            blank = signal - after_means[isotope]
            if not np.isfinite(signal) or signal <= 0.0 or not np.isfinite(blank):
                output.append(BlankQCDiagnostic(
                    sample.name, sample.run_number, sample.observation_id, ratio_name,
                    "blank_load", isotope, None, "unavailable",
                    "measured signal denominator is nonpositive or non-finite",
                ))
            else:
                output.append(BlankQCDiagnostic(
                    sample.name, sample.run_number, sample.observation_id, ratio_name,
                    "blank_load", isotope, 100.0 * blank / signal, "not_assessed",
                ))

        raw_denominator = raw_means[denominator]
        after_denominator = after_means[denominator]
        if raw_denominator <= 0.0 or after_denominator <= 0.0:
            output.append(BlankQCDiagnostic(
                sample.name, sample.run_number, sample.observation_id, ratio_name,
                "ratio_change", "", None, "unavailable",
                "the before/after ratio denominator signal is nonpositive",
            ))
            continue
        ratio_before = raw_means[numerator] / raw_denominator
        ratio_after = after_means[numerator] / after_denominator
        if not np.isfinite(ratio_after) or ratio_after <= 0.0:
            output.append(BlankQCDiagnostic(
                sample.name, sample.run_number, sample.observation_id, ratio_name,
                "ratio_change", "", None, "unavailable",
                "the blank-corrected ratio denominator is nonpositive or non-finite",
            ))
        else:
            output.append(BlankQCDiagnostic(
                sample.name, sample.run_number, sample.observation_id, ratio_name,
                "ratio_change", "",
                100.0 * (ratio_before - ratio_after) / ratio_after,
                "not_assessed",
            ))
    return tuple(output)

"""Blank correction algorithms for TraceISO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from domain.models import CycleData, Sample


KR_ISOTOPES = frozenset(["82Kr", "83Kr", "84Kr", "86Kr"])

# Sample metadata key holding, per isotope, the weight each bracketing blank
# role ("before"/"after") actually received in the subtracted blank mean.
BLANK_CHANNEL_WEIGHTS_KEY = "blank_channel_weights"


@dataclass
class BlankCorrectionResult:
    """Outcome of blank-correcting a list of samples."""

    samples: List[Sample]
    warnings: List[str] = field(default_factory=list)


def apply_blank_correction(
    samples: List[Sample],
    mode: str = "before",
    element_symbol: str = "",
    subtract_kr_blank: bool = False,
) -> BlankCorrectionResult:
    """Blank-correct a sequence of samples in place.

    ``subtract_kr_blank`` is a legacy ignored argument. Blank correction now
    applies uniformly to all isotope channels with matching blank data.
    """
    warnings: List[str] = []

    if mode == "none":
        for s in samples:
            if s.is_blank:
                continue
            copied = {
                iso: cd.copy() for iso, cd in s.intensities.items()
            }
            s.corrected_intensities = copied
            s.blank_corrected_intensities = {
                iso: cd.copy() for iso, cd in copied.items()
            }
        return BlankCorrectionResult(samples=samples, warnings=warnings)

    for idx, sample in enumerate(samples):
        if sample.is_blank:
            continue

        prev_blank, prev_name = _find_blank_before(samples, idx)
        next_blank, next_name = (None, None)
        if mode == "before_and_after":
            next_blank, next_name = _find_blank_after(samples, idx)

        if prev_blank is None and next_blank is None:
            # No blank found — copy raw values unchanged
            copied = {
                iso: cd.copy() for iso, cd in sample.intensities.items()
            }
            sample.corrected_intensities = copied
            sample.blank_corrected_intensities = {
                iso: cd.copy() for iso, cd in copied.items()
            }
            warnings.append(f"No blank found for sample '{sample.name}'; skipped correction.")
            continue

        # Record both the label and the identity of the blank that was
        # actually subtracted. The label alone is ambiguous whenever two blanks
        # share a name, and a consumer that re-resolves it by name can pick a
        # different observation from the one this correction used.
        if prev_name:
            sample.used_blanks["before"] = prev_name
            sample.used_blank_ids["before"] = prev_blank.observation_id
        if next_name:
            sample.used_blanks["after"] = next_name
            sample.used_blank_ids["after"] = next_blank.observation_id

        corrected: Dict[str, CycleData] = {}
        uncorrected_channels: List[str] = []
        channel_weights: Dict[str, Dict[str, float]] = {}
        for iso, cd in sample.intensities.items():
            blank_avg, weights = _bracketing_blank_mean(
                iso, prev_blank, next_blank, mode,
            )
            # Record what was actually subtracted per channel. A channel that
            # only one bracketing blank supports carries that blank at weight
            # 1, not 1/2, and a consumer that assumes equal halves would
            # understate its contribution by a factor of two.
            channel_weights[iso] = weights
            if blank_avg is not None:
                new_vals = cd.values - blank_avg
                corrected[iso] = CycleData(values=new_vals, mask=cd.mask.copy())
            else:
                corrected[iso] = cd.copy()
                uncorrected_channels.append(iso)

        sample.metadata[BLANK_CHANNEL_WEIGHTS_KEY] = channel_weights

        sample.corrected_intensities = corrected
        sample.blank_corrected_intensities = {
            iso: cd.copy() for iso, cd in corrected.items()
        }

        if uncorrected_channels:
            uncorrected_channels.sort()
            selected_blank_names = ", ".join(
                name for name in (prev_name, next_name) if name
            )
            warning = (
                f"Blank correction incomplete for sample '{sample.name}': "
                f"isotope(s) {', '.join(uncorrected_channels)} have no usable "
                f"blank mean in selected blank(s) [{selected_blank_names}]; "
                "raw (uncorrected) values were retained for those channels."
            )
            warnings.append(warning)
            sample.metadata["blank_correction_complete"] = False
            sample.metadata["blank_uncorrected_channels"] = uncorrected_channels
        else:
            sample.metadata["blank_correction_complete"] = True
            sample.metadata["blank_uncorrected_channels"] = []

    return BlankCorrectionResult(samples=samples, warnings=warnings)


def calculate_ratios(
    samples: List[Sample],
    ratio_definitions: Dict[str, Tuple[str, str]],
    use_corrected: bool = True,
) -> List[Sample]:
    """Calculate isotope ratios for every sample."""
    for sample in samples:
        # Raw ratios — merge into existing to preserve custom/file-loaded ratios
        computed = _compute_ratios(sample.intensities, ratio_definitions)
        sample.ratios.update(computed)

        # Corrected ratios
        if use_corrected and sample.corrected_intensities:
            computed_corr = _compute_ratios(
                sample.corrected_intensities, ratio_definitions
            )
            sample.corrected_ratios.update(computed_corr)

    return samples


def _find_blank_before(
    samples: List[Sample], idx: int
) -> Tuple[Optional[Sample], Optional[str]]:
    """Search backward for the nearest blank before *idx*."""
    for j in range(idx - 1, -1, -1):
        if samples[j].is_blank:
            return samples[j], samples[j].name
    return None, None


def _find_blank_after(
    samples: List[Sample], idx: int
) -> Tuple[Optional[Sample], Optional[str]]:
    """Search forward for the nearest blank after *idx*."""
    for j in range(idx + 1, len(samples)):
        if samples[j].is_blank:
            return samples[j], samples[j].name
    return None, None


def _bracketing_blank_mean(
    isotope: str,
    prev_blank: Optional[Sample],
    next_blank: Optional[Sample],
    mode: str,
) -> Tuple[Optional[float], Dict[str, float]]:
    """Return the blank value subtracted for *isotope* and its per-role weights.

    Only the bracketing blanks that actually supply a usable mean for this
    isotope enter the average, so the weight of each contributing blank is
    ``1 / n_supporting`` for that channel alone — a partially populated
    bracket does not give both sides one half. The weights are returned so the
    uncertainty consumer propagates the same coefficients that were applied.
    """
    vals: List[float] = []
    roles: List[str] = []

    if prev_blank is not None and isotope in prev_blank.intensities:
        prev_values = prev_blank.intensities[isotope].valid_values
        if len(prev_values) > 0:
            vals.append(float(np.mean(prev_values)))
            roles.append("before")

    if (
        mode == "before_and_after"
        and next_blank is not None
        and isotope in next_blank.intensities
    ):
        next_values = next_blank.intensities[isotope].valid_values
        if len(next_values) > 0:
            vals.append(float(np.mean(next_values)))
            roles.append("after")

    if vals:
        weight = 1.0 / len(vals)
        return float(np.mean(vals)), {role: weight for role in roles}
    return None, {}


def _compute_ratios(
    intensities: Dict[str, CycleData],
    ratio_definitions: Dict[str, Tuple[str, str]],
) -> Dict[str, CycleData]:
    """Compute ratios from an intensity dictionary."""
    result: Dict[str, CycleData] = {}

    for ratio_name, (numer_iso, denom_iso) in ratio_definitions.items():
        if numer_iso not in intensities or denom_iso not in intensities:
            continue

        numer = intensities[numer_iso]
        denom = intensities[denom_iso]

        if len(numer.values) != len(denom.values):
            raise ValueError(
                f"Cannot compute {ratio_name}: isotope channels {numer_iso} and "
                f"{denom_iso} have different cycle counts "
                f"({len(numer.values)} != {len(denom.values)})."
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_vals = numer.values / denom.values

        # Replace inf / nan from division by zero
        bad = ~np.isfinite(ratio_vals)
        ratio_vals[bad] = np.nan

        mask = numer.mask & denom.mask & ~bad
        result[ratio_name] = CycleData(values=ratio_vals, mask=mask)

    return result

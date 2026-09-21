"""204Hg interference correction of 204Pb on the ordinary Pb SSB route.

The apparent 204Pb signal carries 204Hg, monitored on 202Hg:

    Hg204_i  = 202Hg_i · (204Hg/202Hg)_ref · (m202 / m204)^f_i
    204Pb*_i = 204Pb_i − Hg204_i

The Hg mass-bias factor is chosen **per measurement**, not per session. When
both 203Tl and 205Tl are present, ``f_i`` is the existing Russell factor from
the measured Tl ratio. When both are absent, ``f_i = 0`` and the natural ratio
is subtracted unscaled. A measurement with only one Tl channel, or with an
unusable Tl reference, is not a natural-ratio case: its correction is
unavailable. Using Tl here does not externally normalize Pb, does not write
``iif_corrected_ratios`` and does not select Engine C.

Subtraction runs after the canonical outlier filter and before drift and SSB,
on the blank-corrected intensities. The canonical filter mask is not recomputed.

Invalid individual cycles (owner decision, 2026-09-11): on the ratio support, a
cycle whose Tl factor is non-finite, or whose corrected 204Pb is non-finite or
not positive, is excluded from the 204Pb-bearing ratios only. It keeps its
computed value under a false mask, and the record lists its reason with the
count, the fraction of support and a review flag. A denominator is never
clipped and no value is replaced by zero. Exclusion does not make the remaining
result reliable; the selection effect is an unassessed coverage item.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from config.reference_materials import get_natural_ratio_record, require_isotope_mass
from domain.corrections.interference import hg204_interference_correction
from domain.corrections.mass_bias import calculate_f_factor
from domain.layer_status import APPLIED, REVIEW_INVALID_CYCLE_EXCLUSION, UNAVAILABLE
from domain.models import CycleData, Sample
from domain.pb_correction_records import (
    HG_RECORD_FAMILY,
    HG_UNAVAILABLE_REASONS,
    ROUTE_PB_TL,
    ROUTE_SSB,
    SOURCE_NATURAL_RATIO,
    SOURCE_TL,
    HgCorrectionRecord,
    mask_sha256,
)
from domain.ratio_selection import get_processing_ratio_data
from domain.ratio_utils import normalize_ratio_name, normalize_ratio_token

PB204 = "204Pb"
HG202 = "202Hg"
HG_REFERENCE_RATIO = "204Hg/202Hg"


def resolve_hg_reference() -> Dict[str, Any]:
    """Managed Hg reference ratio, its masses and uncertainty evidence.

    An unassigned uncertainty stays ``None``: it is omitted from propagation,
    never read as an exact zero. Raises ``ValueError`` when the ratio or a mass
    cannot be resolved to a finite positive value.
    """
    record = get_natural_ratio_record("Hg", HG_REFERENCE_RATIO)
    if record is None:
        raise ValueError("Managed natural ratio '204Hg/202Hg' for element 'Hg' is missing.")
    value = float(record.value)
    m202 = float(require_isotope_mass(HG202))
    m204 = float(require_isotope_mass("204Hg"))
    if not all(math.isfinite(v) and v > 0 for v in (value, m202, m204)):
        raise ValueError("The Hg reference ratio and masses must be finite and positive.")
    unassigned = bool(getattr(record, "is_uncertainty_unassigned", False))
    uncertainty = getattr(record, "uncertainty", None)
    k = getattr(record, "k", None)
    return {
        "record_id": str(getattr(record, "record_id", "") or ""),
        "ratio_name": HG_REFERENCE_RATIO,
        "value": value,
        "uncertainty": None if unassigned or uncertainty is None else float(uncertainty),
        "k": None if unassigned or k is None else float(k),
        "uncertainty_semantics": str(getattr(record, "uncertainty_semantics", "") or ""),
        "masses": {HG202: m202, "204Hg": m204},
    }


def _ratio_isotopes(
    ratio_name: str, ratio_definitions: Mapping[str, Tuple[str, str]],
) -> Optional[Tuple[str, str]]:
    pair = ratio_definitions.get(ratio_name)
    if pair is None:
        parts = normalize_ratio_name(ratio_name).split("/")
        if len(parts) != 2:
            return None
        pair = (parts[0], parts[1])
    return normalize_ratio_token(pair[0]), normalize_ratio_token(pair[1])


def _finite_support(channel: CycleData) -> np.ndarray:
    return np.asarray(channel.mask, dtype=bool) & np.isfinite(channel.values)


def classify_invalid_cycles(
    corrected_204: np.ndarray, f: np.ndarray, support: np.ndarray, *, source: str,
) -> np.ndarray:
    """Per-cycle invalid-cycle reason code ('' for a valid or unsupported cycle)."""
    reasons = np.full(len(corrected_204), "", dtype=object)
    support = np.asarray(support, dtype=bool)
    finite_c = np.isfinite(corrected_204)
    if source == SOURCE_TL:
        reasons[support & ~np.isfinite(f)] = "nonfinite_tl_factor"
    open_ = support & (reasons == "")
    reasons[open_ & ~finite_c] = "nonfinite_corrected_204Pb"
    open_ = support & (reasons == "")
    with np.errstate(invalid="ignore"):
        nonpositive = finite_c & (corrected_204 <= 0.0)
    reasons[open_ & nonpositive] = "nonpositive_corrected_204Pb"
    return reasons


def _common_unavailability(
    src: Mapping[str, CycleData],
    hg_reference: Optional[Mapping[str, Any]],
    tl_pair: Optional[Tuple[str, str, float, float]],
    tl_reference_value: Optional[float],
) -> Tuple[str, Optional[str]]:
    """Return ``(reason_code, source)``; an empty reason means the correction can run."""
    if hg_reference is None:
        return "hg_reference_unresolved", None
    if HG202 not in src:
        return "hg_monitor_absent", None
    if PB204 not in src:
        return "pb204_absent", None
    if tl_pair is None:
        if any(str(iso).endswith("Tl") for iso in src):
            return "tl_reference_unresolved", None
        source = SOURCE_NATURAL_RATIO
    else:
        num, den, m_num, m_den = tl_pair
        present = (num in src, den in src)
        if not any(present):
            source = SOURCE_NATURAL_RATIO
        elif not all(present):
            return "tl_channel_incomplete", None
        else:
            values = (tl_reference_value, m_num, m_den)
            if any(v is None or not math.isfinite(float(v)) or float(v) <= 0 for v in values):
                return "tl_reference_unresolved", None
            source = SOURCE_TL
    channels = [PB204, HG202] + ([tl_pair[0], tl_pair[1]] if source == SOURCE_TL else [])
    if len({len(src[iso].values) for iso in channels}) != 1:
        return "misaligned_channels", None
    return "", source


def apply_ssb_hg_correction(
    samples: Sequence[Sample],
    *,
    ratio_definitions: Mapping[str, Tuple[str, str]],
    tl_pair: Optional[Tuple[str, str, float, float]],
    tl_ratio_name: Optional[str],
    tl_reference_value: Optional[float],
    warnings: List[str],
) -> None:
    """Write Hg-corrected 204Pb and 204Pb-bearing ratios plus governing records.

    ``tl_pair`` is ``(numerator, denominator, m_numerator, m_denominator)`` of the
    configured Tl normalization ratio, or ``None`` when it cannot be resolved.
    Raw and blank-corrected layers are never written.
    """
    try:
        hg_reference: Optional[Dict[str, Any]] = resolve_hg_reference()
        hg_error = ""
    except ValueError as exc:
        hg_reference, hg_error = None, str(exc)

    for sample in samples:
        if sample.is_blank:
            continue
        processing_layer = sample.blank_corrected_ratios or sample.corrected_ratios or sample.ratios
        targets = []
        for ratio_name in processing_layer:
            pair = _ratio_isotopes(ratio_name, ratio_definitions)
            if pair is not None and PB204 in pair and pair[0] != pair[1]:
                targets.append((ratio_name, pair))
        if not targets:
            continue

        if sample.corrected_intensities:
            src, basis = sample.corrected_intensities, "corrected_intensities"
        else:
            src, basis = sample.intensities, "intensities"
        n_cycles = sample.n_cycles
        reason_code, source = _common_unavailability(src, hg_reference, tl_pair, tl_reference_value)

        masses: Dict[str, float] = dict((hg_reference or {}).get("masses", {}))
        tl_reference: Dict[str, Any] = {}
        if source == SOURCE_TL and tl_pair is not None:
            masses.update({tl_pair[0]: float(tl_pair[2]), tl_pair[1]: float(tl_pair[3])})
            tl_reference = {"ratio_name": str(tl_ratio_name or ""), "value": float(tl_reference_value)}
        reference_payload = {k: v for k, v in (hg_reference or {}).items() if k != "masses"}
        base = dict(
            observation_id=sample.observation_id, route=ROUTE_SSB, governs_final=True,
            requested=True, source=source, hg_reference=reference_payload,
            tl_reference=tl_reference, masses=masses, intensity_basis=basis, n_cycles=n_cycles,
        )
        records = sample.correction_records.setdefault(HG_RECORD_FAMILY, {})

        if reason_code:
            text = HG_UNAVAILABLE_REASONS[reason_code]
            if reason_code == "hg_reference_unresolved" and hg_error:
                text = f"{text} ({hg_error})"
            for ratio_name, _pair in targets:
                records[ratio_name] = HgCorrectionRecord(
                    ratio_name=ratio_name, status=UNAVAILABLE, reason_code=reason_code,
                    reason=text, **{**base, "source": None},
                )
            warnings.append(
                f"Sample '{sample.name}': Hg interference correction unavailable "
                f"({reason_code}); {', '.join(r for r, _ in targets)} ha"
                f"{'s' if len(targets) == 1 else 've'} no corrected result. {text}"
            )
            continue

        pb204 = src[PB204]
        hg202 = src[HG202]
        n = len(pb204.values)
        if source == SOURCE_TL:
            num, den, m_num, m_den = tl_pair  # type: ignore[misc]
            with np.errstate(divide="ignore", invalid="ignore"):
                measured = src[num].values / src[den].values
            f = np.asarray(calculate_f_factor(measured, float(tl_reference_value), m_num, m_den), dtype=float)
        else:
            f = np.zeros(n, dtype=float)
        with np.errstate(all="ignore"):
            corrected_204, _hg204 = hg204_interference_correction(
                pb204=np.array(pb204.values, dtype=float, copy=True),
                hg202=np.asarray(hg202.values, dtype=float),
                f_tl=f,
                hg204_hg202_natural=float(hg_reference["value"]),  # type: ignore[index]
                m202=masses[HG202],
                m204_hg=masses["204Hg"],
            )

        channel_support = _finite_support(pb204) & _finite_support(hg202)
        if source == SOURCE_TL:
            channel_support &= _finite_support(src[tl_pair[0]]) & _finite_support(src[tl_pair[1]])  # type: ignore[index]
        reasons = classify_invalid_cycles(corrected_204, f, channel_support, source=source)
        invalid = reasons != ""
        sample.interference_corrected_intensities[PB204] = CycleData(
            values=np.array(corrected_204, dtype=float, copy=True),
            mask=channel_support & ~invalid,
        )

        for ratio_name, (num_iso, den_iso) in targets:
            other = den_iso if num_iso == PB204 else num_iso
            if other not in src:
                records[ratio_name] = HgCorrectionRecord(
                    ratio_name=ratio_name, status=UNAVAILABLE, reason_code="ratio_channel_absent",
                    reason=HG_UNAVAILABLE_REASONS["ratio_channel_absent"], **base,
                )
                continue
            processing = get_processing_ratio_data(sample, ratio_name)
            if processing is None or len(processing.values) != n or len(src[other].values) != n:
                records[ratio_name] = HgCorrectionRecord(
                    ratio_name=ratio_name, status=UNAVAILABLE, reason_code="misaligned_channels",
                    reason=HG_UNAVAILABLE_REASONS["misaligned_channels"], **base,
                )
                warnings.append(
                    f"Sample '{sample.name}': {ratio_name} Hg correction unavailable "
                    "(misaligned_channels); no cycle rows truncated."
                )
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                if den_iso == PB204:
                    values = src[other].values / corrected_204
                else:
                    values = corrected_204 / src[other].values
            support = (
                np.asarray(processing.mask, dtype=bool)
                & _finite_support(src[other])
                & channel_support
            )
            excluded_mask = support & invalid
            cycle_data = CycleData(values=values, mask=support & ~invalid)
            sample.interference_corrected_ratios[ratio_name] = cycle_data

            excluded = tuple(
                (int(i) + 1, str(reasons[i])) for i in np.flatnonzero(excluded_mask)
            )
            n_support = int(np.sum(support))
            applied = cycle_data.n_valid > 0
            records[ratio_name] = HgCorrectionRecord(
                ratio_name=ratio_name,
                status=APPLIED if applied else UNAVAILABLE,
                reason_code="" if applied else "no_valid_cycles",
                reason="" if applied else HG_UNAVAILABLE_REASONS["no_valid_cycles"],
                support_n_valid=n_support,
                n_valid=cycle_data.n_valid,
                excluded_cycles=excluded,
                excluded_fraction=(len(excluded) / n_support) if n_support else None,
                review_flags=(REVIEW_INVALID_CYCLE_EXCLUSION,) if excluded else (),
                support_mask_sha256=mask_sha256(cycle_data.mask),
                **base,
            )
            if excluded:
                warnings.append(
                    f"Sample '{sample.name}': {len(excluded)} of {n_support} cycle(s) excluded "
                    f"from {ratio_name} as invalid after Hg correction "
                    f"({', '.join(f'{c}:{r}' for c, r in excluded)}); review this result."
                )
            if not applied:
                warnings.append(
                    f"Sample '{sample.name}': {ratio_name} has no valid cycle after Hg "
                    "correction (no_valid_cycles); no corrected result is reported."
                )


def build_pb_tl_hg_record(
    *,
    sample: Sample,
    ratio_name: str,
    reason_code: str,
    hg_reference: Mapping[str, Any],
    tl_ratio_name: Optional[str],
    tl_value: Optional[float],
    masses: Mapping[str, Optional[float]],
    intensity_basis: str,
    corrected_204: Optional[np.ndarray],
    target_k: Optional[np.ndarray],
) -> HgCorrectionRecord:
    """Diagnostic record mirroring the legacy Pb-Tl Hg outcome for one ratio.

    Nothing here rejects a cycle or changes availability. The counters report
    support cycles the legacy chain keeps although their corrected 204Pb is not
    finite and positive, and valid Tl-normalized cycles with a zero K factor.
    """
    interference = sample.interference_corrected_ratios.get(ratio_name)
    iif = sample.iif_corrected_ratios.get(ratio_name)
    base = dict(
        observation_id=sample.observation_id, ratio_name=ratio_name, route=ROUTE_PB_TL,
        governs_final=False, requested=True,
        hg_reference={k: v for k, v in hg_reference.items() if k != "masses"},
        tl_reference=(
            {"ratio_name": str(tl_ratio_name or ""), "value": float(tl_value)}
            if tl_value is not None else {}
        ),
        masses={k: float(v) for k, v in masses.items() if v is not None},
        intensity_basis=intensity_basis, n_cycles=sample.n_cycles,
    )
    if interference is None:
        code = reason_code or "ratio_channel_absent"
        return HgCorrectionRecord(
            status=UNAVAILABLE, reason_code=code, reason=HG_UNAVAILABLE_REASONS[code], **base,
        )
    mask = np.asarray(interference.mask, dtype=bool)
    n_nonpositive = 0
    if corrected_204 is not None and len(corrected_204) == len(mask):
        with np.errstate(invalid="ignore"):
            valid_204 = np.isfinite(corrected_204) & (corrected_204 > 0.0)
        n_nonpositive = int(np.sum(mask & ~valid_204))
    n_zero_k = 0
    if iif is not None and target_k is not None and len(target_k) == len(iif.mask):
        n_zero_k = int(np.sum(np.asarray(iif.mask, dtype=bool) & (np.asarray(target_k) == 0.0)))
    return HgCorrectionRecord(
        status=APPLIED,
        source=SOURCE_TL,
        support_n_valid=interference.n_valid,
        n_valid=interference.n_valid,
        support_mask_sha256=mask_sha256(mask),
        diagnostics={
            "n_nonpositive_corrected_204Pb_on_support": n_nonpositive,
            "n_zero_normalization_k_on_support": n_zero_k,
            "tl_normalized_output_present": int(iif is not None),
        },
        **base,
    )


def hg_invalid_cycle_exclusions(sample: Sample, ratio_name: str) -> Tuple[Tuple[int, str], ...]:
    """1-based ``(cycle, reason)`` exclusions a governing Hg record recorded for display."""
    from domain.pb_correction_records import governing_hg_record

    record = governing_hg_record(sample, ratio_name)
    return record.excluded_cycles if record is not None else ()

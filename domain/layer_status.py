"""Status vocabulary for correction layers that can govern a final result.

A correction layer is not "present or absent": a requested correction can be
applied, can be unavailable for a stated reason, or can describe inputs that
have since changed. Selectors consult this status before walking any fallback
chain, so an unavailable requested correction never becomes an earlier layer
reported as final.

``stale`` is never written by a producer. It is derived on read by comparing a
recorded input identity with the current session, so producers only write
``not_requested``, ``applied`` and ``unavailable``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

NOT_REQUESTED = "not_requested"
APPLIED = "applied"
UNAVAILABLE = "unavailable"
STALE = "stale"

PRODUCER_STATUSES = frozenset({NOT_REQUESTED, APPLIED, UNAVAILABLE})
ALL_STATUSES = PRODUCER_STATUSES | {STALE}

#: Statuses under which a requested correction has no final value.
BLOCKING_STATUSES = frozenset({UNAVAILABLE, STALE})

#: Review flag carried by any observation from which at least one individual
#: cycle was excluded as invalid. There is deliberately no threshold: the flag
#: says the remaining result needs review, not that it is reliable.
REVIEW_INVALID_CYCLE_EXCLUSION = "review_invalid_cycle_exclusion"


def intrinsic_restorable_mask(sample: Any, layer: str, key: str, cycle_data: Any) -> np.ndarray:
    """Return support that may be restored after removing a manual overlay.

    Derived Pb layers have validity rules beyond finiteness.  The pipeline
    evaluates masked cycles so a later controller edit can recover valid data,
    but a finite invalid correction must never become valid merely because the
    user clears an exclusion.
    """
    values = np.asarray(cycle_data.values, dtype=float)
    mask = np.isfinite(values)
    original = (sample.metadata.get("_original_masks") or {}).get(layer, {}).get(key)
    if original is not None and len(original) == len(mask):
        mask &= np.asarray(original, dtype=bool)

    if layer == "interference_corrected_intensities" and key == "204Pb":
        mask &= values > 0.0

    if layer == "interference_corrected_ratios" and "204Pb" in str(key):
        pb204 = (sample.interference_corrected_intensities or {}).get("204Pb")
        if pb204 is not None and len(pb204.values) == len(mask):
            mask &= np.isfinite(pb204.values) & (np.asarray(pb204.values) > 0.0)

    if layer == "pb_standard_corrected_ratios":
        # Calibration factors are finite and positive.  A nonpositive final
        # value therefore proves that the Tl-normalized source was invalid.
        mask &= values > 0.0
        source = (sample.iif_corrected_ratios or {}).get(key)
        if source is not None and len(source.values) == len(mask):
            mask &= np.isfinite(source.values) & (np.asarray(source.values) > 0.0)

    if layer == "pb_calibrated_delta_cycles":
        source = (sample.pb_standard_corrected_ratios or {}).get(key)
        if source is not None and len(source.values) == len(mask):
            mask &= np.asarray(source.mask, dtype=bool)

    return mask

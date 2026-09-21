"""Helpers for comparing processing data layers."""

from __future__ import annotations

from typing import Optional

import numpy as np

from domain.models import CycleData


def cycle_data_equal(left: Optional[CycleData], right: Optional[CycleData]) -> bool:
    """Return True when two cycle layers have identical values and masks."""
    if left is None or right is None:
        return False
    return np.array_equal(left.values, right.values, equal_nan=True) and np.array_equal(left.mask, right.mask)


def cycle_data_nearly_equal(
    left: Optional[CycleData],
    right: Optional[CycleData],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Return True when values are numerically close and masks are identical."""
    if left is None or right is None:
        return False
    return (
        len(left.values) == len(right.values)
        and np.array_equal(left.mask, right.mask)
        and np.allclose(
            left.values,
            right.values,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    )


def distinct_from_raw(layer: Optional[CycleData], raw: Optional[CycleData]) -> bool:
    """Return True when a processed layer is present and not just a raw copy."""
    return layer is not None and not cycle_data_equal(layer, raw)

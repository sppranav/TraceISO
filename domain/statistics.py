"""Canonical cycle-level summary statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CycleStatistics:
    """Summary statistics computed from finite cycle values."""

    mean: float
    sd: float
    se: float
    rsd_percent: float
    rse_percent: float
    n: int


def calculate_cycle_statistics(values: Iterable[float]) -> CycleStatistics:
    """Return canonical statistics using only finite observations."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    n = int(len(finite))

    mean = float(np.mean(finite)) if n > 0 else np.nan
    if n > 1:
        sd = float(np.std(finite, ddof=1))
        se = float(sd / np.sqrt(n))
    else:
        sd = np.nan
        se = np.nan

    if np.isfinite(mean) and mean != 0.0:
        rsd_percent = abs(sd / mean) * 100.0 if np.isfinite(sd) else np.nan
        rse_percent = abs(se / mean) * 100.0 if np.isfinite(se) else np.nan
    else:
        rsd_percent = np.nan
        rse_percent = np.nan

    return CycleStatistics(
        mean=mean,
        sd=sd,
        se=se,
        rsd_percent=float(rsd_percent),
        rse_percent=float(rse_percent),
        n=n,
    )

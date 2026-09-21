"""Segment assignment helpers for standard repeatability."""

from __future__ import annotations

from typing import Dict, List

from domain.models import Sample
from domain.uncertainty.reprod import standard_identity_key


def assign_segments_from_breaks(
    all_samples: List[Sample],
    break_positions: List[int],
) -> Dict[str, int]:
    """Assign each sample to a segment based on break positions."""
    if not break_positions:
        assignments: Dict[str, int] = {}
        for s in all_samples:
            assignments[s.name] = 1
            assignments[standard_identity_key(s)] = 1
        return assignments

    sorted_breaks = sorted(break_positions)

    assignments: Dict[str, int] = {}
    for s in all_samples:
        pos = s.run_number
        segment = 1
        for bp in sorted_breaks:
            if pos > bp:
                segment += 1
            else:
                break
        assignments[s.name] = segment
        assignments[standard_identity_key(s)] = segment

    return assignments

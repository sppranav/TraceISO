"""Identity-respecting lookup shared by the runtime maps.

Both runtime maps — uncertainty budgets and delta results — are keyed on
``(run_number, name, ratio_name, observation_id)``. The first three fields keep
a key readable and let a map written before the identity existed still be read;
the fourth is what actually distinguishes two observations that share a label
and a run number.

Reading such a map is the place where that distinction is easiest to lose. A
missing entry and a *different* entry are not the same answer, and neither is
recoverable from numeric similarity: two observations may legitimately carry the
same value, and two that carry the same label may legitimately differ. So the
rule here is narrow. An exact identity match wins. Failing that, only a genuine
legacy key — one that records no identity at all — may be adopted, and only when
the map offers no conflicting identity for the same label. Any four-part key
carrying someone else's identity ends the search: it is positive evidence that
this map knows about identities and that this observation is not in it.

A legacy map cannot recover an identity that was never serialized. Where a
legacy key and a modern one share a prefix, this refuses rather than guesses;
that is a limit of the stored key, not something a reader can disambiguate.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, TypeVar


_Value = TypeVar("_Value")

#: ``(run_number, name, ratio_name, observation_id)``.
ObservationKey = Tuple[int, str, str, str]


def lookup_by_observation(
    entries: Mapping[Tuple[Any, ...], _Value],
    *,
    run_number: int,
    name: str,
    ratio_name: str,
    observation_id: str,
) -> Optional[_Value]:
    """Find one observation's entry, tolerating a genuine legacy key.

    Returns ``None`` when the observation is absent, including when the only
    candidate sharing its label carries a different explicit identity.
    """
    exact: ObservationKey = (run_number, name, ratio_name, observation_id)
    if exact in entries:
        return entries[exact]

    prefix = (run_number, name, ratio_name)
    legacy: Optional[_Value] = None
    legacy_count = 0
    for candidate, value in entries.items():
        if tuple(candidate[:3]) != prefix:
            continue
        candidate_id = str(candidate[3]) if len(candidate) > 3 else ""
        if candidate_id:
            # An explicit, different identity. This map distinguishes
            # observations, and this one is not the requested observation.
            return None
        legacy = value
        legacy_count += 1

    return legacy if legacy_count == 1 else None

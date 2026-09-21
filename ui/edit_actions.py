"""Explicit state transitions for user edit actions.

A100: a Streamlit render function must not reach into a committed ``Sample`` or
``ProcessingResult`` and change it. Every scientific edit the workspace offers
is an explicit action - Apply, Undo, Clear, Run - and each one lives here as a
named controller transition that performs the edit on the objects that own the
data and fires the matching invalidation event. Render branches call these;
passive rendering calls none of them.

Nothing here re-runs the pipeline. The Inspector's mask and window edits feed
the runtime layer, exactly as before; only Execute Data Reduction reprocesses.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping, Optional

import streamlit as st

from config.contributor_names import (
    canonical_contributor_mapping,
    canonical_contributor_name,
)
from domain.models import Sample
from domain.uncertainty.sr_sample_values import SR_SAMPLE_VALUES_KEY
from ui.invalidation import on_masks_changed
from ui.utils import get_sample_state_key


def _sample_key(sample: Sample) -> tuple:
    """Return a stable ``(run_number, name)`` identity tuple for *sample*."""
    return (int(sample.run_number), str(sample.name))


def _apply_exclusions_to_sample(sample: Sample, excluded_cycles: list) -> None:
    """Apply manual cycle exclusions to all CycleData masks on a sample.

    Parameters
    ----------
    sample : Sample
        The sample whose masks will be updated.
    excluded_cycles : list[int]
        1-based cycle numbers to exclude.
    """
    # Convert 1-based cycle numbers to 0-based indices
    indices = [c - 1 for c in excluded_cycles if 1 <= c <= sample.n_cycles]

    if not indices:
        return

    cycle_dicts = [
        sample.ratios,
        sample.blank_corrected_ratios,
        sample.corrected_ratios,
        sample.iif_corrected_ratios,
        sample.drift_corrected_ratios,
        sample.interference_corrected_ratios,
        sample.sr_standard_corrected_ratios,
        sample.pb_standard_corrected_ratios,
        sample.pb_calibrated_delta_cycles,
        sample.intensities,
        sample.blank_corrected_intensities,
        sample.corrected_intensities,
        sample.interference_corrected_intensities,
    ]

    for cd_dict in cycle_dicts:
        if cd_dict:
            for cd in cd_dict.values():
                for idx in indices:
                    if idx < len(cd.mask):
                        cd.mask[idx] = False


def _resolve_manual_exclusion_target(sample: Sample, state: Any) -> Sample:
    """Return the raw session sample that should own manual exclusions."""
    raw_samples = getattr(state, "samples", None) or []
    if not raw_samples:
        return sample

    observation_id = str(getattr(sample, "observation_id", "") or "")
    if observation_id:
        identity_matches = [
            candidate for candidate in raw_samples
            if candidate.observation_id == observation_id
        ]
        if len(identity_matches) == 1:
            return identity_matches[0]
        if identity_matches:
            return sample

    sample_key = get_sample_state_key(sample)
    legacy = [
        candidate for candidate in raw_samples
        if get_sample_state_key(candidate) == sample_key
        or (candidate.name == sample.name and candidate.run_number == sample.run_number)
    ]
    if len(legacy) == 1:
        return legacy[0]

    return sample


def _find_matching_result_sample(state: Any, target: Sample) -> Optional[Sample]:
    """Resolve the result observation by ID, with unique legacy fallback."""
    if not getattr(state, "has_result", False) or state.result is None:
        return None
    observation_id = str(getattr(target, "observation_id", "") or "")
    if observation_id:
        matches = [s for s in state.result.samples if s.observation_id == observation_id]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return None
    legacy = [
        s for s in state.result.samples
        if s.name == target.name and s.run_number == target.run_number
    ]
    return legacy[0] if len(legacy) == 1 else None


def commit_manual_exclusions(
    state: Any, sample: Sample, excluded_cycles: list[int],
) -> None:
    """Apply the user's manual cycle selection to the raw and result samples.

    A100: this is an explicit user action, not part of rendering. It runs as a
    controller transition so the edit and its invalidation always happen
    together, and so a renderer never reaches into a committed ``Sample``.

    Intentionally does NOT invalidate ``state.result``: the Inspector is
    runtime-only, so it keeps showing the processed correction layers
    (blank-corrected, IIF, drift) and the runtime budget recomputes from the
    updated masks on the next render. Only Execute Data Reduction re-runs the
    pipeline.
    """
    target = _resolve_manual_exclusion_target(sample, state)

    _restore_original_masks(target)
    target.metadata.pop("manual_exclusions", None)

    if excluded_cycles:
        _save_original_masks(target)
        _apply_exclusions_to_sample(target, excluded_cycles)
        target.metadata["manual_exclusions"] = sorted(excluded_cycles)

    # Mirror mask changes to the pipeline result sample so the Inspector
    # shows processed correction layers (blank-corrected, IIF etc.) without
    # requiring a full pipeline re-run after each exclusion.
    result_target = _find_matching_result_sample(state, target)
    if result_target is not None:
        _restore_original_masks(result_target)
        result_target.metadata.pop("manual_exclusions", None)
        if excluded_cycles:
            _save_original_masks(result_target)
            _apply_exclusions_to_sample(result_target, excluded_cycles)
            result_target.metadata["manual_exclusions"] = sorted(excluded_cycles)

    on_masks_changed(state)


def _save_original_masks(sample: Sample) -> None:
    """Snapshot all CycleData masks before manual exclusions are applied.

    Only saves once — subsequent calls are no-ops if snapshot already exists.
    """
    if "_original_masks" in sample.metadata:
        return  # Already saved

    snapshot: dict = {}
    for label, cd_dict in [
        ("ratios", sample.ratios),
        ("blank_corrected_ratios", sample.blank_corrected_ratios),
        ("corrected_ratios", sample.corrected_ratios),
        ("iif_corrected_ratios", sample.iif_corrected_ratios),
        ("drift_corrected_ratios", sample.drift_corrected_ratios),
        ("interference_corrected_ratios", sample.interference_corrected_ratios),
        ("sr_standard_corrected_ratios", sample.sr_standard_corrected_ratios),
        ("pb_standard_corrected_ratios", sample.pb_standard_corrected_ratios),
        ("pb_calibrated_delta_cycles", sample.pb_calibrated_delta_cycles),
        ("intensities", sample.intensities),
        ("blank_corrected_intensities", sample.blank_corrected_intensities),
        ("corrected_intensities", sample.corrected_intensities),
        ("interference_corrected_intensities", sample.interference_corrected_intensities),
    ]:
        if cd_dict:
            snapshot[label] = {k: v.mask.copy() for k, v in cd_dict.items()}

    sample.metadata["_original_masks"] = snapshot


def _restore_original_masks(sample: Sample) -> None:
    """Restore masks from the snapshot saved before manual exclusions."""
    snapshot = sample.metadata.get("_original_masks")
    if snapshot is None:
        return

    dict_map = {
        "ratios": sample.ratios,
        "blank_corrected_ratios": sample.blank_corrected_ratios,
        "corrected_ratios": sample.corrected_ratios,
        "iif_corrected_ratios": sample.iif_corrected_ratios,
        "drift_corrected_ratios": sample.drift_corrected_ratios,
        "interference_corrected_ratios": sample.interference_corrected_ratios,
        "sr_standard_corrected_ratios": sample.sr_standard_corrected_ratios,
        "pb_standard_corrected_ratios": sample.pb_standard_corrected_ratios,
        "pb_calibrated_delta_cycles": sample.pb_calibrated_delta_cycles,
        "intensities": sample.intensities,
        "blank_corrected_intensities": sample.blank_corrected_intensities,
        "corrected_intensities": sample.corrected_intensities,
        "interference_corrected_intensities": sample.interference_corrected_intensities,
    }

    for label, cd_dict in dict_map.items():
        saved = snapshot.get(label, {})
        if cd_dict:
            for key, mask_copy in saved.items():
                if key in cd_dict:
                    cd_dict[key].mask = mask_copy.copy()

    sample.metadata.pop("_original_masks", None)
    sample.metadata.pop("manual_exclusions", None)


# ---------------------------------------------------------------------------
# Contributor applicability - the Apply, Save and Undo actions of the
# per-sample matrix and the profile manager. Same rule as above: the render
# branch decides and calls; the write to the committed Sample happens here.
# ---------------------------------------------------------------------------

def _write_applicability(
    sample: Sample,
    *,
    profile: str,
    overrides: dict[str, bool],
    custom_enabled: list[str],
    note: str,
    source: str = "user_assigned",
    profile_defaults: Mapping[str, bool] | None = None,
    owned_built_in: Iterable[str] | None = None,
    owned_custom: Iterable[str] | None = None,
) -> None:
    """Persist contributor applicability to ``sample.metadata``.

    ``owned_built_in`` / ``owned_custom`` name the contributors the calling
    action actually put on screen. When given, the patch is applied onto the
    complete saved mapping and settings for every other contributor are kept.
    An editor that projects a visible subset must pass them; omitting them keeps
    the wholesale-replacement behaviour a profile apply legitimately wants.
    """
    overrides_canonical = canonical_contributor_mapping(overrides)
    if profile_defaults is not None:
        defaults_canonical = canonical_contributor_mapping(profile_defaults)
        overrides_canonical = {
            name: value
            for name, value in overrides_canonical.items()
            if defaults_canonical.get(name, True) != value
        }

    sample.metadata = dict(sample.metadata or {})
    existing_uc = sample.metadata.get("uncertainty_contributors") or {}

    # A079: a save of the visible controls used to replace the whole mapping,
    # so a contributor the current element/engine hides lost its stored setting.
    if owned_built_in is not None:
        owned_names = {canonical_contributor_name(name) for name in owned_built_in}
        stored_overrides = existing_uc.get("overrides") if isinstance(existing_uc, dict) else None
        preserved_overrides = (
            {
                name: value
                for name, value in canonical_contributor_mapping(stored_overrides).items()
                if name not in owned_names
            }
            if isinstance(stored_overrides, Mapping)
            else {}
        )
        preserved_overrides.update(overrides_canonical)
        overrides_canonical = preserved_overrides

    if owned_custom is not None:
        owned_custom_names = {str(name) for name in owned_custom}
        stored_custom = existing_uc.get("custom_enabled") if isinstance(existing_uc, dict) else None
        preserved_custom = (
            [str(name) for name in stored_custom if str(name) not in owned_custom_names]
            if isinstance(stored_custom, (list, tuple, set))
            else []
        )
        custom_enabled = sorted(set(custom_enabled) | set(preserved_custom))
    preserved_values = (
        copy.deepcopy(existing_uc.get(SR_SAMPLE_VALUES_KEY))
        if isinstance(existing_uc, dict)
        and isinstance(existing_uc.get(SR_SAMPLE_VALUES_KEY), dict)
        else None
    )
    sample.metadata["uncertainty_contributors"] = {
        "profile": profile,
        "source": source if source in {"user_assigned", "fallback"} else "user_assigned",
        "overrides": dict(sorted(overrides_canonical.items())),
        "custom_enabled": sorted(custom_enabled),
        "note": note,
    }
    if preserved_values:
        sample.metadata["uncertainty_contributors"][SR_SAMPLE_VALUES_KEY] = preserved_values



def _write_sr_sample_value_overrides(
    sample: Sample,
    *,
    values: dict[str, dict[str, float]],
) -> None:
    """Persist sample-specific Sr value overrides while preserving applicability."""
    sample.metadata = dict(sample.metadata or {})
    uc = sample.metadata.get("uncertainty_contributors")
    if not isinstance(uc, dict):
        uc = {}
    else:
        uc = dict(uc)

    if values:
        uc[SR_SAMPLE_VALUES_KEY] = values
    else:
        uc.pop(SR_SAMPLE_VALUES_KEY, None)

    if uc:
        sample.metadata["uncertainty_contributors"] = uc
    else:
        sample.metadata.pop("uncertainty_contributors", None)


def delete_uncertainty_profile(
    state: Any,
    *,
    deleted_profile: str,
    fallback_profile: str,
    profiles: Mapping[str, Any],
    expected_mtime: float | None,
) -> int:
    """Persist and commit a profile deletion as one recoverable transition."""
    from config.uncertainty_profiles_loader import save_user_profiles
    from domain.uncertainty.contributors import BUILTIN_PROFILES

    if deleted_profile not in profiles or getattr(profiles[deleted_profile], "builtin", False):
        raise ValueError("Select an existing user profile to delete.")
    if fallback_profile == deleted_profile or fallback_profile not in profiles:
        raise ValueError("Select a valid fallback profile.")

    previous_users = {
        name: profile for name, profile in profiles.items()
        if not getattr(profile, "builtin", False) and name not in BUILTIN_PROFILES
    }
    candidate_users = dict(previous_users)
    candidate_users.pop(deleted_profile)

    collections = [getattr(state, "samples", None) or []]
    result = getattr(state, "result", None)
    if result is not None:
        collections.append(getattr(result, "samples", None) or [])
    targets = []
    for collection in collections:
        for sample in collection:
            uc = (getattr(sample, "metadata", None) or {}).get("uncertainty_contributors")
            if isinstance(uc, dict) and uc.get("profile") == deleted_profile:
                targets.append(sample)

    staged = []
    for sample in targets:
        metadata = copy.deepcopy(sample.metadata or {})
        uc = dict(metadata.get("uncertainty_contributors") or {})
        uc["profile"] = fallback_profile
        uc["source"] = "fallback" if fallback_profile == "custom" else "user_assigned"
        if fallback_profile == "custom":
            uc.setdefault("overrides", {})
        # Existing overrides, custom selections, notes, and sample-specific
        # magnitudes remain explicit observation metadata across reassignment.
        metadata["uncertainty_contributors"] = uc
        staged.append((sample, copy.deepcopy(sample.metadata), metadata))

    new_revision = save_user_profiles(candidate_users, expected_mtime=expected_mtime)
    try:
        for sample, _old, metadata in staged:
            sample.metadata = metadata
    except Exception:
        for sample, old, _metadata in staged:
            sample.metadata = old
        save_user_profiles(previous_users, expected_revision=new_revision)
        raise

    from ui.invalidation import on_masks_changed
    state.reload_uncertainty_profiles()
    on_masks_changed(state)
    return len(targets)



BULK_UNDO_STATE_KEY = "_uncertainty_profile_bulk_undo"

# A080: the bulk profile operation writes exactly one metadata field. The
# snapshot used to record the whole dict and Undo replaced the whole dict, so
# undoing a profile assignment also erased whatever a later, unrelated edit had
# written - cycle exclusion records, original-mask recovery snapshots,
# contributor numerical inputs - while leaving the mask arrays themselves
# changed. Undo now owns that one field and nothing else.
BULK_UNDO_OWNED_FIELD = "uncertainty_contributors"


def _owned_bulk_field(sample: Sample):
    """Return a detached copy of the only metadata field bulk assignment owns."""
    return copy.deepcopy((sample.metadata or {}).get(BULK_UNDO_OWNED_FIELD))


def _store_bulk_undo(samples: Iterable[Sample]) -> None:
    """Record the pre-assignment value of the owned field for each target."""
    st.session_state[BULK_UNDO_STATE_KEY] = [
        [_sample_key(sample), _owned_bulk_field(sample), None]
        for sample in samples
    ]


def _record_bulk_undo_applied(samples: Iterable[Sample]) -> None:
    """Stamp what the assignment actually wrote, so later edits are detectable.

    Undo compares against this stamp rather than reverting blindly: a sample
    whose profile settings were changed again after the bulk assignment is a
    conflict the user has to see, not something to silently overwrite.
    """
    snapshot = st.session_state.get(BULK_UNDO_STATE_KEY)
    if not snapshot:
        return
    applied = {_sample_key(sample): _owned_bulk_field(sample) for sample in samples}
    st.session_state[BULK_UNDO_STATE_KEY] = [
        [key, before, applied.get(tuple(key), previous)]
        for key, before, previous in snapshot
    ]


def _restore_bulk_undo(samples: Iterable[Sample]) -> tuple[int, int]:
    """Revert the owned field where it is unchanged; return (restored, skipped)."""
    snapshot = st.session_state.get(BULK_UNDO_STATE_KEY)
    if not snapshot:
        return 0, 0
    by_key = {_sample_key(sample): sample for sample in samples}
    restored = 0
    skipped = 0
    for entry in snapshot:
        key, before, applied = entry[0], entry[1], entry[2] if len(entry) > 2 else None
        sample = by_key.get(tuple(key))
        if sample is None:
            continue
        metadata = dict(sample.metadata or {})
        if applied is not None and metadata.get(BULK_UNDO_OWNED_FIELD) != applied:
            skipped += 1
            continue
        if before is None:
            metadata.pop(BULK_UNDO_OWNED_FIELD, None)
        else:
            metadata[BULK_UNDO_OWNED_FIELD] = copy.deepcopy(before)
        sample.metadata = metadata
        restored += 1
    st.session_state.pop(BULK_UNDO_STATE_KEY, None)
    return restored, skipped

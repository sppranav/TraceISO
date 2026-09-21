"""Streamlit helpers for uncertainty contributor profile management."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd
import streamlit as st

from config.contributor_names import (
    canonical_contributor_mapping,
    canonical_contributor_name,
)
from config.uncertainty_profiles_loader import (
    ProfileConflictError,
    make_profile_name,
    save_user_profiles,
)
from domain.models import Sample
from domain.uncertainty.contributors import (
    BUILTIN_PROFILES,
    ContributorProfile,
    SampleContributorApplicability,
    freeze_profile,
)


def profile_options(profiles: Mapping[str, ContributorProfile]) -> list[str]:
    """Return stable profile names in display order."""
    builtins = [name for name in ("full_chemistry", "reference_no_chemistry", "custom") if name in profiles]
    user_profiles = sorted(
        (name for name, profile in profiles.items() if not profile.builtin),
        key=lambda name: profiles[name].display_name.lower(),
    )
    return builtins + user_profiles


def profile_label(profile: ContributorProfile) -> str:
    """Return a compact user-facing label for a profile."""
    label = profile.display_name or profile.name
    suffix = "built-in" if profile.builtin else profile.name
    return f"{label} ({suffix})"


def _safe_profile_slug_value(label: str) -> str:
    try:
        return make_profile_name(label)
    except ValueError:
        return ""


def _profile_defaults_for_sample(
    applicability: SampleContributorApplicability,
    profile_defaults: Mapping[str, Mapping[str, bool]],
    contributor_name: str,
) -> bool:
    contributor_name = canonical_contributor_name(contributor_name)
    defaults = canonical_contributor_mapping(profile_defaults.get(applicability.profile, {}))
    return bool(defaults.get(contributor_name, True))


def create_profile_from_sample(
    sample: Sample,
    *,
    display_name: str,
    description: str,
    profiles: Mapping[str, ContributorProfile],
    profile_defaults: Mapping[str, Mapping[str, bool]],
    built_in_contributors: Sequence[tuple[str, str]],
    custom_contributors: Sequence[object],
    name: str = "",
) -> ContributorProfile:
    """Create a user profile from a sample's effective contributor pattern."""
    profile_name = make_profile_name(name or display_name)
    if profile_name in profiles or profile_name in BUILTIN_PROFILES:
        raise ValueError(f"Profile {profile_name!r} already exists.")
    label = display_name.strip()
    if not label:
        raise ValueError("Profile display name cannot be empty.")

    known_profiles = set(profile_defaults.keys())
    applicability = SampleContributorApplicability.from_sample(
        sample,
        known_profiles=known_profiles,
    )
    defaults = {}
    for contributor_name, _label in built_in_contributors:
        contributor_name = canonical_contributor_name(contributor_name)
        overrides = canonical_contributor_mapping(applicability.overrides)
        enabled = bool(overrides.get(
            contributor_name,
            _profile_defaults_for_sample(
                applicability,
                profile_defaults,
                contributor_name,
            ),
        ))
        if enabled is False:
            defaults[contributor_name] = False

    custom_names = {getattr(definition, "name", "") for definition in custom_contributors}
    custom_enabled = tuple(
        sorted(
            name for name in applicability.custom_enabled
            if name in custom_names
        )
    )

    return freeze_profile(
        ContributorProfile(
            name=profile_name,
            display_name=label,
            description=description.strip(),
            defaults=defaults,
            custom_enabled_defaults=custom_enabled,
            builtin=False,
        )
    )


def profile_defaults_from_row(
    row: Mapping[str, object],
    *,
    built_in_contributors: Sequence[tuple[str, str]],
    custom_contributors: Sequence[object],
) -> tuple[dict[str, bool], tuple[str, ...]]:
    """Return profile defaults from an edited matrix row."""
    defaults = {
        canonical_contributor_name(name): False
        for name, _label in built_in_contributors
        if bool(row.get(name, True)) is False
    }
    custom_enabled = tuple(
        sorted(
            getattr(definition, "name", "")
            for definition in custom_contributors
            if bool(row.get(getattr(definition, "name", ""), False))
        )
    )
    return defaults, custom_enabled


def remap_samples_from_deleted_profile(
    samples: Sequence[Sample],
    *,
    deleted_profile: str,
    fallback_profile: str,
    profiles: Mapping[str, ContributorProfile],
) -> int:
    """Remap samples assigned to a deleted profile to a fallback profile."""
    fallback = profiles.get(fallback_profile)
    custom_enabled_defaults = (
        list(fallback.custom_enabled_defaults)
        if fallback is not None
        else []
    )
    count = 0
    for sample in samples:
        metadata = dict(sample.metadata or {})
        uc = metadata.get("uncertainty_contributors") or {}
        if not isinstance(uc, dict) or uc.get("profile") != deleted_profile:
            continue
        new_uc = dict(uc)
        new_uc["profile"] = fallback_profile
        new_uc["source"] = "fallback" if fallback_profile == "custom" else "user_assigned"
        if fallback_profile == "custom":
            new_uc.setdefault("overrides", {})
        else:
            new_uc["overrides"] = {}
            new_uc["custom_enabled"] = custom_enabled_defaults
        metadata["uncertainty_contributors"] = new_uc
        sample.metadata = metadata
        count += 1
    return count


def _user_profiles_from_registry(
    profiles: Mapping[str, ContributorProfile],
) -> dict[str, ContributorProfile]:
    return {
        name: profile
        for name, profile in profiles.items()
        if not profile.builtin and name not in BUILTIN_PROFILES
    }


def _save_user_profiles_checked(
    profiles: dict[str, ContributorProfile],
    *,
    expected_mtime: float | None,
) -> None:
    """Persist profiles with the mtime captured when the registry was loaded."""
    save_user_profiles(profiles, expected_mtime=expected_mtime)


def _checkbox_grid(
    *,
    key_prefix: str,
    defaults: Mapping[str, bool],
    built_in_contributors: Sequence[tuple[str, str]],
    custom_enabled: Sequence[str],
    custom_contributors: Sequence[object],
) -> tuple[dict[str, bool], tuple[str, ...]]:
    st.caption("Contributor defaults")
    cols = st.columns(3)
    canonical_defaults = canonical_contributor_mapping(defaults)
    # A079: the grid shows only the contributors the current element/engine
    # exposes. Start from the stored defaults for everything it does not show,
    # so an unrelated save does not silently re-enable a hidden contributor.
    visible_built_ins = {
        canonical_contributor_name(name) for name, _label in built_in_contributors
    }
    out_defaults: dict[str, bool] = {
        name: value
        for name, value in canonical_defaults.items()
        if name not in visible_built_ins
    }
    for idx, (name, label) in enumerate(built_in_contributors):
        canonical_name = canonical_contributor_name(name)
        with cols[idx % 3]:
            enabled = st.checkbox(
                label,
                value=bool(canonical_defaults.get(canonical_name, True)),
                key=f"{key_prefix}_builtin_{name}",
            )
        if enabled is False:
            out_defaults[canonical_name] = False

    visible_custom = {
        str(getattr(definition, "name", "")) for definition in custom_contributors
    }
    out_custom = [
        str(name) for name in custom_enabled if str(name) not in visible_custom
    ]
    if custom_contributors:
        st.caption("Custom contributor defaults")
        custom_cols = st.columns(3)
        for idx, definition in enumerate(custom_contributors):
            name = getattr(definition, "name", "")
            label = getattr(definition, "display_name", name)
            with custom_cols[idx % 3]:
                enabled = st.checkbox(
                    label,
                    value=name in set(custom_enabled),
                    key=f"{key_prefix}_custom_{name}",
                )
            if enabled:
                out_custom.append(name)
    return out_defaults, tuple(sorted(out_custom))


def render_profile_manager(
    *,
    profiles: Mapping[str, ContributorProfile],
    profile_defaults: Mapping[str, Mapping[str, bool]],
    samples: Sequence[Sample],
    built_in_contributors: Sequence[tuple[str, str]],
    custom_contributors: Sequence[object],
    expanded: bool = False,
    include_standards: bool = False,
    expected_mtime: float | None = None,
) -> bool:
    """Render profile manager controls. Return True when profiles changed."""
    changed = False
    with st.expander("Manage profiles", expanded=expanded):
        options = profile_options(profiles)
        library_rows = [
            {
                "Profile": name,
                "Display Name": profiles[name].display_name,
                "Built In": profiles[name].builtin,
                "Description": profiles[name].description,
            }
            for name in options
        ]
        st.dataframe(pd.DataFrame(library_rows), hide_index=True, width="stretch")

        st.caption("Create from sample")
        eligible = [
            sample for sample in samples
            if (sample.is_sample or (include_standards and sample.is_standard))
            and not (sample.metadata or {}).get("excluded", False)
        ]
        if eligible:
            sample_labels = {
                f"{sample.run_number}: {sample.name}": sample
                for sample in eligible
            }
            source_label = st.selectbox(
                "Source sample",
                options=list(sample_labels.keys()),
                key="uncertainty_profile_create_source",
            )
            create_display = st.text_input(
                "New profile display name",
                key="uncertainty_profile_create_display",
            )
            create_slug = st.text_input(
                "Profile ID",
                value=_safe_profile_slug_value(create_display) if create_display.strip() else "",
                key="uncertainty_profile_create_slug",
                help=(
                    "Stored lowercase identifier, also called a slug. "
                    "It is written to sample metadata and cannot be renamed after creation."
                ),
            )
            create_description = st.text_area(
                "Description",
                key="uncertainty_profile_create_description",
            )
            if st.button("Create profile from sample", key="uncertainty_profile_create_button"):
                try:
                    user_profiles = _user_profiles_from_registry(profiles)
                    profile = create_profile_from_sample(
                        sample_labels[source_label],
                        display_name=create_display,
                        description=create_description,
                        profiles=profiles,
                        profile_defaults=profile_defaults,
                        built_in_contributors=built_in_contributors,
                        custom_contributors=custom_contributors,
                        name=create_slug,
                    )
                    user_profiles[profile.name] = profile
                    _save_user_profiles_checked(
                        user_profiles,
                        expected_mtime=expected_mtime,
                    )
                    st.success(f"Created profile {profile.name}.")
                    changed = True
                except (ProfileConflictError, ValueError) as exc:
                    st.error(str(exc))

        st.divider()
        st.caption("Duplicate profile")
        duplicate_source = st.selectbox(
            "Profile to duplicate",
            options=options,
            format_func=lambda name: profile_label(profiles[name]),
            key="uncertainty_profile_duplicate_source",
        )
        duplicate_display = st.text_input(
            "Duplicate display name",
            value=f"{profiles[duplicate_source].display_name} copy",
            key="uncertainty_profile_duplicate_display",
        )
        duplicate_slug = st.text_input(
            "Duplicate profile ID",
            value=_safe_profile_slug_value(duplicate_display) if duplicate_display.strip() else "",
            key="uncertainty_profile_duplicate_slug",
            help=(
                "Stored lowercase identifier, also called a slug. "
                "It is written to sample metadata and cannot be renamed after creation."
            ),
        )
        if st.button("Duplicate", key="uncertainty_profile_duplicate_button"):
            try:
                user_profiles = _user_profiles_from_registry(profiles)
                display_label = duplicate_display.strip()
                if not display_label:
                    raise ValueError("Display name cannot be empty.")
                profile_name = make_profile_name(duplicate_slug or display_label)
                if profile_name in profiles or profile_name in BUILTIN_PROFILES:
                    raise ValueError(f"Profile {profile_name!r} already exists.")
                source = profiles[duplicate_source]
                user_profiles[profile_name] = freeze_profile(
                    ContributorProfile(
                        name=profile_name,
                        display_name=display_label,
                        description=source.description,
                        defaults=dict(source.defaults),
                        custom_enabled_defaults=tuple(source.custom_enabled_defaults),
                        builtin=False,
                    )
                )
                _save_user_profiles_checked(
                    user_profiles,
                    expected_mtime=expected_mtime,
                )
                st.success(f"Duplicated profile as {profile_name}.")
                changed = True
            except (ProfileConflictError, ValueError) as exc:
                st.error(str(exc))

        user_options = [
            name for name in options
            if not profiles[name].builtin
        ]
        st.divider()
        st.caption("Edit user profile")
        if not user_options:
            st.caption("No user profiles have been created yet.")
        else:
            selected = st.selectbox(
                "User profile",
                options=user_options,
                format_func=lambda name: profile_label(profiles[name]),
                key="uncertainty_profile_edit_selected",
            )
            profile = profiles[selected]
            st.text_input(
                "Profile ID",
                value=profile.name,
                disabled=True,
                key="uncertainty_profile_edit_slug",
            )
            display_name = st.text_input(
                "Display name",
                value=profile.display_name,
                key="uncertainty_profile_edit_display",
            )
            description = st.text_area(
                "Description",
                value=profile.description,
                key="uncertainty_profile_edit_description",
            )
            defaults, custom_enabled = _checkbox_grid(
                key_prefix=f"uncertainty_profile_edit_{selected}",
                defaults=profile.defaults,
                built_in_contributors=built_in_contributors,
                custom_enabled=profile.custom_enabled_defaults,
                custom_contributors=custom_contributors,
            )
            edit_col, delete_col = st.columns(2)
            with edit_col:
                if st.button("Save profile", key="uncertainty_profile_save_button"):
                    try:
                        if not display_name.strip():
                            raise ValueError("Display name cannot be empty.")
                        user_profiles = _user_profiles_from_registry(profiles)
                        user_profiles[selected] = freeze_profile(
                            ContributorProfile(
                                name=selected,
                                display_name=display_name.strip(),
                                description=description.strip(),
                                defaults=defaults,
                                custom_enabled_defaults=custom_enabled,
                                builtin=False,
                                created_at=profile.created_at,
                            )
                        )
                        _save_user_profiles_checked(
                            user_profiles,
                            expected_mtime=expected_mtime,
                        )
                        st.success("Profile saved.")
                        changed = True
                    except (ProfileConflictError, ValueError) as exc:
                        st.error(str(exc))
            with delete_col:
                fallback_options = [
                    name for name in options
                    if name != selected
                ]
                fallback = st.selectbox(
                    "Delete fallback profile",
                    options=fallback_options,
                    index=fallback_options.index("custom") if "custom" in fallback_options else 0,
                    format_func=lambda name: profile_label(profiles[name]),
                    key="uncertainty_profile_delete_fallback",
                )
                confirm = st.text_input(
                    "Type profile ID to delete",
                    key="uncertainty_profile_delete_confirm",
                )
                if st.button("Delete profile", key="uncertainty_profile_delete_button"):
                    try:
                        if confirm != selected:
                            raise ValueError("Delete confirmation does not match the profile slug.")
                        from ui.edit_actions import delete_uncertainty_profile
                        from ui.state import get_state
                        remapped = delete_uncertainty_profile(
                            get_state(),
                            deleted_profile=selected,
                            fallback_profile=fallback,
                            profiles=profiles,
                            expected_mtime=expected_mtime,
                        )
                        st.success(f"Deleted {selected} and reassigned {remapped} observation(s).")
                        changed = True
                    except (ProfileConflictError, ValueError) as exc:
                        st.error(str(exc))

    return changed

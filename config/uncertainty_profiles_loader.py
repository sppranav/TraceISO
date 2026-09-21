"""Loader for user-managed uncertainty contributor profiles.

File I/O lives here so the uncertainty domain resolver remains pure.
Built-in profiles are defined in :mod:`domain.uncertainty.contributors`; this
module only persists user-created profiles.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional

from config.contributor_names import canonical_contributor_mapping
from domain.uncertainty.contributors import (
    BUILTIN_PROFILES,
    ContributorProfile,
    freeze_profile,
    merge_profile_defaults,
)


UNCERTAINTY_PROFILES_PATH = Path(__file__).with_name("uncertainty_profiles.json")

_CURRENT_VERSION = 1
_MIGRATIONS: Dict[int, Callable[[dict], dict]] = {}
_VALID_PROFILE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9]+")
_SLUG_DEDUP_RE = re.compile(r"_+")


class ProfileConflictError(RuntimeError):
    """Raised when a profile save would overwrite another session's edit."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_profile_name(label: str) -> str:
    """Return a stable slug ID for a user profile display label."""
    slug = _SLUG_CLEAN_RE.sub("_", str(label).lower()).strip("_")
    slug = _SLUG_DEDUP_RE.sub("_", slug)
    if not slug:
        raise ValueError("Profile name must contain at least one alphanumeric character.")
    if slug and slug[0].isdigit():
        slug = f"p_{slug}"
    # Truncate to 64 characters before validation
    slug = slug[:64].rstrip("_")
    if not slug:
        raise ValueError("Profile name must contain at least one alphanumeric character.")
    if not _VALID_PROFILE_NAME_RE.match(slug):
        raise ValueError(
            "Profile name must start with a lowercase letter, contain only "
            "lowercase letters, digits, and underscores, and be between 2 and 64 characters long."
        )
    return slug


def _apply_migrations(data: dict) -> dict:
    """Apply sequential migrations from stored version to the current schema."""
    version = data.get("version", 1)
    if not isinstance(version, int):
        raise ValueError("Uncertainty profile JSON version must be an integer.")
    if version > _CURRENT_VERSION:
        raise ValueError(
            "Uncertainty profile file was created by a newer TraceISO version. "
            "Please update TraceISO and retry."
        )
    while version < _CURRENT_VERSION:
        next_version = version + 1
        if next_version not in _MIGRATIONS:
            raise ValueError(
                f"No migration defined from uncertainty profile version {version} "
                f"to {next_version}. This version of TraceISO cannot upgrade this file."
            )
        migrate = _MIGRATIONS[next_version]
        data = migrate(data)
        version += 1
    data["version"] = _CURRENT_VERSION
    return data


def _validate_profile_name(name: str) -> None:
    if not _VALID_PROFILE_NAME_RE.match(name):
        raise ValueError(
            f"Profile name {name!r} must match ^[a-z][a-z0-9_]{{1,63}}$ and be between 2 and 64 characters long."
        )
    if name in BUILTIN_PROFILES:
        raise ValueError(f"User profile name {name!r} collides with a built-in profile.")


def _normalize_defaults(defaults: Mapping[str, object]) -> Dict[str, bool]:
    raw: Dict[str, bool] = {}
    for key, value in defaults.items():
        name = str(key)
        if not name.startswith("u_"):
            raise ValueError(f"Profile default key {name!r} must start with 'u_'.")
        if not isinstance(value, bool):
            raise ValueError(f"Profile default {name!r} must be a boolean.")
        if value is True:
            raise ValueError(
                f"Profile default {name!r} is True; omit enabled contributors "
                "from profile defaults."
            )
        raw[name] = value

    normalized: Dict[str, bool] = {}
    for name, value in canonical_contributor_mapping(raw).items():
        if value is False:
            normalized[name] = False
    return dict(sorted(normalized.items()))


def _normalize_custom_enabled(values: object) -> tuple:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ValueError("custom_enabled_defaults must be a JSON list.")
    out = []
    for value in values:
        name = str(value)
        if not name.startswith("u_custom_"):
            raise ValueError(
                f"Custom contributor default {name!r} must start with 'u_custom_'."
            )
        out.append(name)
    return tuple(sorted(set(out)))


def _profile_from_json(entry: object) -> ContributorProfile:
    if not isinstance(entry, dict):
        raise ValueError("Each uncertainty profile entry must be a JSON object.")

    name = str(entry.get("name") or "").strip()
    _validate_profile_name(name)

    display_name = str(entry.get("display_name") or "").strip()
    if not display_name:
        raise ValueError(f"Profile {name!r} must have a non-empty display_name.")

    defaults_raw = entry.get("defaults") or {}
    if not isinstance(defaults_raw, dict):
        raise ValueError(f"Profile {name!r} defaults must be a JSON object.")

    profile = ContributorProfile(
        name=name,
        display_name=display_name,
        description=str(entry.get("description") or ""),
        defaults=_normalize_defaults(defaults_raw),
        custom_enabled_defaults=_normalize_custom_enabled(
            entry.get("custom_enabled_defaults") or []
        ),
        builtin=False,
        created_at=str(entry.get("created_at") or ""),
        updated_at=str(entry.get("updated_at") or ""),
    )
    return freeze_profile(profile)


def _profile_to_json(profile: ContributorProfile) -> dict:
    now = _utc_now()
    created_at = getattr(profile, "created_at", "") or now
    updated_at = getattr(profile, "updated_at", "") or now
    return {
        "name": profile.name,
        "display_name": profile.display_name,
        "description": profile.description,
        "defaults": _normalize_defaults(profile.defaults),
        "custom_enabled_defaults": list(profile.custom_enabled_defaults),
        "created_at": created_at,
        "updated_at": updated_at,
    }


def load_user_profiles(
    path: Path = UNCERTAINTY_PROFILES_PATH,
) -> Dict[str, ContributorProfile]:
    """Load user profiles. Missing files return an empty dict."""
    if not path.exists():
        return {}

    from config.validation import load_json_file
    raw = load_json_file(path, "uncertainty profile")
    if not isinstance(raw, dict):
        raise ValueError("Uncertainty profile file must be a JSON object.")
    raw = _apply_migrations(raw)

    entries = raw.get("profiles", [])
    if not isinstance(entries, list):
        raise ValueError("Uncertainty profile file 'profiles' must be a list.")

    profiles: Dict[str, ContributorProfile] = {}
    for entry in entries:
        profile = _profile_from_json(entry)
        if profile.name in profiles:
            raise ValueError(f"Duplicate uncertainty profile name {profile.name!r}.")
        profiles[profile.name] = profile
    return profiles


def user_profiles_mtime(path: Path = UNCERTAINTY_PROFILES_PATH) -> float:
    """Return the current user-profile file mtime, or -1 when it is absent."""
    return path.stat().st_mtime if path.exists() else -1.0


def user_profiles_revision(path: Path = UNCERTAINTY_PROFILES_PATH) -> str:
    """Return the content revision of the user-profile file (``""`` if absent)."""
    from config.managed_file_writes import file_content_revision

    return file_content_revision(path)


def save_user_profiles(
    profiles: Dict[str, ContributorProfile],
    path: Path = UNCERTAINTY_PROFILES_PATH,
    *,
    expected_mtime: Optional[float] = None,
    expected_revision: Optional[str] = None,
) -> str:
    """Write user profiles as one serialized cross-process transaction.

    The supplied expectation — a content revision from
    :func:`user_profiles_revision`, or the legacy modification time — is
    re-read while an exclusive lock is held, so two overlapping saves cannot
    both decide they are current. Returns the revision of what was written.
    """
    entries = []
    seen = set()
    for name, profile in sorted(profiles.items(), key=lambda item: item[0]):
        if getattr(profile, "builtin", False) or name in BUILTIN_PROFILES:
            continue
        _validate_profile_name(str(name))
        if not str(getattr(profile, "display_name", "") or "").strip():
            raise ValueError(f"Profile {name!r} must have a non-empty display_name.")
        if name in seen:
            raise ValueError(f"Duplicate uncertainty profile name {name!r}.")
        seen.add(name)
        entries.append(_profile_to_json(profile))

    payload = {
        "version": _CURRENT_VERSION,
        "profiles": entries,
    }
    from config.managed_file_writes import commit_managed_text

    return commit_managed_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True),
        expected_mtime=expected_mtime,
        expected_revision=expected_revision,
        conflict_error=ProfileConflictError,
        conflict_message=(
            "Profile file was modified by another session. Please reload and retry."
        ),
    )


def load_all_profiles(
    path: Path = UNCERTAINTY_PROFILES_PATH,
) -> Dict[str, ContributorProfile]:
    """Return built-ins plus user profiles. Built-ins win on collision."""
    profiles = dict(BUILTIN_PROFILES)
    for name, profile in load_user_profiles(path).items():
        if name not in profiles:
            profiles[name] = profile
    return profiles


def profile_defaults_for_resolver(
    path: Path = UNCERTAINTY_PROFILES_PATH,
) -> Dict[str, Dict[str, bool]]:
    """Return resolver-ready profile defaults for built-in and user profiles."""
    return merge_profile_defaults(load_user_profiles(path))

"""Shared sample-aware contributor resolver for TraceISO uncertainty engines.

Single source of truth for contributor applicability. All paths — runtime
budgets, pipeline-stored budgets, Monte Carlo, exports — must use this module
to resolve contributor state.  No Streamlit imports, no file I/O.

References
----------
GUM:2008 §1, JCGM 100:2008 — contributor classification A/B.
"""

from __future__ import annotations

import re
import types
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Set

from config.contributor_names import (
    canonical_contributor_mapping,
    canonical_contributor_name,
)
from config.settings import CustomUncertaintyContributor, UncertaintyConfig
from domain.models import Sample, UncertaintyContributor


# ---------------------------------------------------------------------------
# Contributor state enum
# ---------------------------------------------------------------------------

class ContributorState(str, Enum):
    """Classification of why a contributor is included or excluded."""

    ACTIVE = "ACTIVE"
    BY_SAMPLE_DESIGN = "BY_SAMPLE_DESIGN"    # disabled by per-sample metadata / profile
    BY_GLOBAL_DESIGN = "BY_GLOBAL_DESIGN"    # disabled by global UncertaintyConfig
    MISSING_DATA = "MISSING_DATA"            # engine: data absent at computation time
    # The data are present and the contributor is enabled, but this release
    # has no approved model for turning them into an uncertainty. Distinct
    # from MISSING_DATA, which would blame the data, and from the two DESIGN
    # states, which would blame the operator's configuration.
    NO_APPROVED_MODEL = "NO_APPROVED_MODEL"
    # Enabled, but the term does not enter the reported quantity at all — for
    # example the CRM certificate in delta output, where the reference value
    # cancels. Not a gap in the budget, and not an operator choice either.
    NOT_APPLICABLE = "NOT_APPLICABLE"


# ---------------------------------------------------------------------------
# Profile defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContributorProfile:
    """Reusable defaults for sample-level contributor applicability."""

    name: str
    display_name: str
    description: str = ""
    defaults: Mapping[str, bool] = field(default_factory=dict)
    custom_enabled_defaults: tuple = ()
    builtin: bool = False
    created_at: str = ""
    updated_at: str = ""


def freeze_profile(profile: ContributorProfile) -> ContributorProfile:
    """Return *profile* with immutable default mappings."""
    return ContributorProfile(
        name=profile.name,
        display_name=profile.display_name,
        description=profile.description,
        defaults=types.MappingProxyType(canonical_contributor_mapping(profile.defaults)),
        custom_enabled_defaults=tuple(profile.custom_enabled_defaults),
        builtin=profile.builtin,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


BUILTIN_PROFILES: Dict[str, ContributorProfile] = {
    profile.name: freeze_profile(profile)
    for profile in (
        ContributorProfile(
            name="full_chemistry",
            display_name="Full chemistry",
            description="Use the complete uncertainty contributor set.",
            defaults={},
            builtin=True,
        ),
        ContributorProfile(
            name="reference_no_chemistry",
            display_name="Reference material, no chemistry",
            description=(
                "Disable chemistry correction contributors for matrix separation, "
                "procedural blank, and matrix effects."
            ),
            defaults={
                "u_k2_matrix_separation": False,
                "u_k3_procedural_blank": False,
                "u_k6_matrix_effects": False,
                "u_reprod_dig": False,
            },
            builtin=True,
        ),
        ContributorProfile(
            name="custom",
            display_name="Custom row settings",
            description="Manual per-sample contributor choices.",
            defaults={},
            builtin=True,
        ),
    )
}

#: Per-profile forced contributor states.  Only *False* entries are listed;
#: missing names fall through to the global UncertaintyConfig.
PROFILE_DEFAULTS: Dict[str, Dict[str, bool]] = {
    name: dict(profile.defaults)
    for name, profile in BUILTIN_PROFILES.items()
}


def profile_defaults_from_profiles(
    profiles: Mapping[str, ContributorProfile],
) -> Dict[str, Dict[str, bool]]:
    """Return resolver-ready ``{profile_name: defaults}`` from profile objects."""
    return {
        str(name): canonical_contributor_mapping(profile.defaults)
        for name, profile in profiles.items()
    }


def merge_profile_defaults(
    user_profiles: Optional[Mapping[str, ContributorProfile]] = None,
) -> Dict[str, Dict[str, bool]]:
    """Return built-in profile defaults plus non-colliding user defaults."""
    profiles = {
        name: canonical_contributor_mapping(profile.defaults)
        for name, profile in BUILTIN_PROFILES.items()
    }
    for name, profile in (user_profiles or {}).items():
        if name not in BUILTIN_PROFILES:
            profiles[str(name)] = canonical_contributor_mapping(profile.defaults)
    return profiles


def normalize_profile_defaults(
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
) -> Dict[str, Dict[str, bool]]:
    """Return resolver-ready defaults with current built-ins taking precedence."""
    profiles = {
        name: canonical_contributor_mapping(profile.defaults)
        for name, profile in BUILTIN_PROFILES.items()
    }
    for name, defaults in (profile_defaults or {}).items():
        if name not in BUILTIN_PROFILES:
            profiles[str(name)] = canonical_contributor_mapping(defaults)
    return profiles


# ---------------------------------------------------------------------------
# SampleContributorApplicability dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleContributorApplicability:
    """Parsed per-sample contributor applicability from ``Sample.metadata``."""

    profile: str = "full_chemistry"
    overrides: Mapping[str, bool] = field(default_factory=dict)  # type: ignore[assignment]
    custom_enabled: frozenset = field(default_factory=frozenset)
    note: str = ""
    source: str = "user_assigned"

    @classmethod
    def from_sample(
        cls,
        sample: Sample,
        *,
        known_profiles: Optional[Set[str]] = None,
    ) -> "SampleContributorApplicability":
        """Safely parse per-sample applicability from ``sample.metadata``.

        Returns the default (full-chemistry, no overrides) when metadata is
        absent, malformed, or missing the ``uncertainty_contributors`` key.
        """
        raw = (getattr(sample, "metadata", {}) or {}).get("uncertainty_contributors") or {}
        if not isinstance(raw, dict):
            return cls()

        profile = str(raw.get("profile") or "full_chemistry")
        source = str(raw.get("source") or "user_assigned")
        if known_profiles is None:
            from config.recorded_dependencies import current_dependencies
            recorded = current_dependencies()
            implicit_profiles = recorded.profile_defaults if recorded is not None else PROFILE_DEFAULTS
            if profile not in implicit_profiles:
                profile = "full_chemistry"
                source = "fallback"
        elif profile not in known_profiles:
            profile = "custom"
            source = "fallback"

        overrides_raw = raw.get("overrides") or {}
        overrides: Dict[str, bool] = (
            canonical_contributor_mapping(overrides_raw)
            if isinstance(overrides_raw, dict)
            else {}
        )

        custom_raw = raw.get("custom_enabled") or []
        custom_enabled: frozenset = (
            frozenset(str(n) for n in custom_raw)
            if isinstance(custom_raw, list)
            else frozenset()
        )

        return cls(
            profile=profile,
            overrides=overrides,
            custom_enabled=custom_enabled,
            note=str(raw.get("note") or ""),
            source=source if source in {"user_assigned", "fallback"} else "user_assigned",
        )

    def to_metadata(self) -> dict:
        """Serialise to the ``sample.metadata["uncertainty_contributors"]`` shape."""
        return {
            "profile": self.profile,
            "source": self.source,
            "overrides": canonical_contributor_mapping(self.overrides),
            "custom_enabled": sorted(self.custom_enabled),
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Core resolver
# ---------------------------------------------------------------------------

def resolve_contributor_state(
    *,
    name: str,
    uncertainty_config: UncertaintyConfig,
    sample: Sample,
    element_symbol: str,
    custom_contributor_names: Optional[Set[str]] = None,
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
) -> ContributorState:
    """Return the configured contributor state for *sample* before data checks.

    Precedence (highest first):
    1. Custom contributor: opt-in per sample - default ``BY_SAMPLE_DESIGN``.
    2. Global ``UncertaintyConfig.is_contributor_enabled`` hard-disable.
    3. Per-sample explicit override in ``sample.metadata``.
    4. Profile-default forced state.
    5. Global ``UncertaintyConfig.is_contributor_enabled`` enabled state.

    Per-sample overrides can restore a contributor disabled by a sample profile,
    but they cannot re-enable a built-in contributor that the global budget
    configuration has explicitly excluded.

    Engines may downgrade ``ACTIVE`` to ``MISSING_DATA`` when data are absent.
    The resolver never sets ``MISSING_DATA``; that is an engine responsibility.
    """
    from config.recorded_dependencies import current_dependencies
    recorded = current_dependencies()
    implicit_profiles = recorded.profile_defaults if recorded is not None else PROFILE_DEFAULTS
    all_profile_defaults = (
        normalize_profile_defaults(profile_defaults)
        if profile_defaults is not None
        else implicit_profiles
    )
    known_profiles = set(all_profile_defaults.keys()) if profile_defaults is not None else None
    applicability = SampleContributorApplicability.from_sample(
        sample,
        known_profiles=known_profiles,
    )
    name = canonical_contributor_name(name)
    custom_names: Set[str] = {
        canonical_contributor_name(custom_name)
        for custom_name in (custom_contributor_names or set())
    }

    # 1. Custom contributors: opt-in per sample, default off.
    if name in custom_names:
        return (
            ContributorState.ACTIVE
            if name in applicability.custom_enabled
            else ContributorState.BY_SAMPLE_DESIGN
        )

    # 2. A global contributor exclusion is a hard budget gate. This prevents
    # persisted per-sample "True" overrides from resurrecting a contributor
    # the user unchecked in Uncertainty Configuration.
    if not uncertainty_config.is_contributor_enabled(name, element_symbol=element_symbol):
        return ContributorState.BY_GLOBAL_DESIGN

    # 3. Per-sample explicit override beats profile defaults.
    if name in applicability.overrides:
        return (
            ContributorState.ACTIVE
            if bool(applicability.overrides[name])
            else ContributorState.BY_SAMPLE_DESIGN
        )

    # 4. Profile default.
    profile_defaults_for_sample = canonical_contributor_mapping(
        all_profile_defaults.get(applicability.profile, {})
    )
    if name in profile_defaults_for_sample:
        return (
            ContributorState.ACTIVE
            if bool(profile_defaults_for_sample[name])
            else ContributorState.BY_SAMPLE_DESIGN
        )

    return ContributorState.ACTIVE


def is_contributor_active(
    *,
    name: str,
    uncertainty_config: UncertaintyConfig,
    sample: Sample,
    element_symbol: str,
    custom_contributor_names: Optional[Set[str]] = None,
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
) -> bool:
    """Convenience wrapper: ``True`` iff ``resolve_contributor_state`` returns ``ACTIVE``."""
    return (
        resolve_contributor_state(
            name=name,
            uncertainty_config=uncertainty_config,
            sample=sample,
            element_symbol=element_symbol,
            custom_contributor_names=custom_contributor_names,
            profile_defaults=profile_defaults,
        )
        == ContributorState.ACTIVE
    )


def inactive_reason_for_state(
    state: ContributorState,
    *,
    contributor_name: str,
    profile: str = "",
) -> str:
    """Return a human-readable explanation for a non-ACTIVE contributor state."""
    if state == ContributorState.BY_SAMPLE_DESIGN:
        suffix = f" (profile: {profile})" if profile else ""
        return f"{contributor_name} disabled by sample contributor applicability{suffix}."
    if state == ContributorState.BY_GLOBAL_DESIGN:
        return f"{contributor_name} disabled by global uncertainty configuration."
    if state == ContributorState.MISSING_DATA:
        return f"{contributor_name} could not be evaluated from available data."
    if state == ContributorState.NO_APPROVED_MODEL:
        return (
            f"{contributor_name} is unavailable in this release: no approved "
            "uncertainty model is defined for it."
        )
    if state == ContributorState.NOT_APPLICABLE:
        return f"{contributor_name} does not apply to the reported output."
    return ""


def not_applicable_reason(contributor_name: str, *, output_mode: str) -> str:
    """Return why an enabled contributor cannot enter this output, or "".

    Only the CRM certificate term qualifies. A delta is measured against the
    bracketing reference material, so its certified value — and that value's
    uncertainty — cancels; every engine's ``_compute_crm`` already returns zero
    in delta output. Reporting that zero as ``MISSING_DATA`` wrongly flagged
    every delta budget as incomplete.
    """
    if canonical_contributor_name(contributor_name) != "u_crm":
        return ""
    if str(output_mode or "").strip().lower() != "delta":
        return ""
    return (
        "u_crm does not apply to delta output: the reference material's "
        "certified value, and so its uncertainty, cancels in the delta."
    )


# ---------------------------------------------------------------------------
# Known built-in contributor name sets (for UI matrix column generation)
# ---------------------------------------------------------------------------

#: Built-in contributor names used by Engine B (SSB/delta — Li, B, Mg, Cd, Pb).
ENGINE_SSB_CONTRIBUTOR_NAMES: List[str] = [
    "u_prec",
    "u_std",
    "u_std_repeatability",
    "u_blank",
    "u_k1_sample_decomposition",
    "u_k2_matrix_separation",
    "u_k3_procedural_blank",
    "u_k4_bracketing_standard_heterogeneity",
    "u_k5_instrumental_drift",
    "u_k6_matrix_effects",
    "u_k7_residual_interferences",
    "u_crm",
]

#: Built-in contributor names used by Engine A (internal normalisation — Sr).
ENGINE_INTERNAL_CONTRIBUTOR_NAMES: List[str] = [
    "u_prec",
    "u_std_repeatability",
    "u_blank",
    "u_interf",
    "u_crm",
    "u_ref_value",
    "u_bias_ref",
    "u_bias_qc",
    "u_reprod_dig",
    "u_norm_ratio",
]

#: Built-in contributor names used by Engine C (Pb-Tl external normalisation).
#: This is display metadata for UI routing only; Pb-Tl budget math lives in
#: ``domain.uncertainty.engine_external_pb_tl``.
ENGINE_PB_TL_CONTRIBUTOR_NAMES: List[str] = [
    "u_prec",
    "u_std_repeatability",
    "u_blank",
    "u_norm_ref",
    "u_interf",
    "u_crm",
    "u_kappa_drift",
    "u_bias_qc",
    "u_reprod_dig",
]


# ---------------------------------------------------------------------------
# Custom contributor row builder
# ---------------------------------------------------------------------------

def build_custom_contributor_rows(
    *,
    sample: Sample,
    element_symbol: str,
    ratio_mean: float,
    custom_contributor_library: Mapping[str, Sequence[CustomUncertaintyContributor]],
) -> List[UncertaintyContributor]:
    """Materialise enabled custom contributor rows for *sample*.

    Filtering by *element_symbol* happens here — callers pass the full library
    dict. This is the single filter point: no caller needs to pre-filter.

    Skips definitions that:
    - are disabled globally (``enabled=False``),
    - are not listed in ``sample.metadata["uncertainty_contributors"]["custom_enabled"]``,
    - have ``u_rel_permil <= 0``.

    ``percentage_contribution`` is set to 0.0 and will be filled by the
    engine's RSS combination step.
    """
    applicability = SampleContributorApplicability.from_sample(sample)
    custom_definitions: Sequence[CustomUncertaintyContributor] = list(
        custom_contributor_library.get(element_symbol, [])
    )
    out: List[UncertaintyContributor] = []

    for definition in custom_definitions:
        if not definition.enabled:
            continue
        if definition.name not in applicability.custom_enabled:
            continue
        u_rel_permil = max(float(definition.u_rel_permil), 0.0)
        if u_rel_permil <= 0.0:
            continue

        ratio_scale = abs(float(ratio_mean)) if ratio_mean else 0.0
        out.append(
            UncertaintyContributor(
                name=definition.name,
                display_name=definition.display_name,
                value_abs=ratio_scale * u_rel_permil / 1000.0,
                value_rel_permil=u_rel_permil,
                type_ab=definition.type_ab,
                degrees_of_freedom=float(definition.degrees_of_freedom),
                percentage_contribution=0.0,
                description=definition.description or "User-defined uncertainty contributor.",
                reference=definition.reference,
                is_active=True,
                state=ContributorState.ACTIVE.value,
                distribution=str(definition.distribution or ""),
            )
        )

    return out


# ---------------------------------------------------------------------------
# Name helper
# ---------------------------------------------------------------------------

_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9]+")
_SLUG_DEDUP_RE = re.compile(r"_+")


def make_custom_contributor_name(label: str) -> str:
    """Return a canonical ``u_custom_<slug>`` name for a human-readable label.

    Raises ``ValueError`` if *label* produces an empty slug.
    """
    slug = _SLUG_CLEAN_RE.sub("_", label.lower()).strip("_")
    slug = _SLUG_DEDUP_RE.sub("_", slug)
    if not slug:
        raise ValueError(
            "Contributor label must contain at least one alphanumeric character."
        )
    return f"u_custom_{slug}"

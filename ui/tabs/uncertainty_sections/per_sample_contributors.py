"""Per-sample contributor applicability matrix for the Uncertainty tab.

Renders an expander with a Streamlit form that lets the user toggle which
uncertainty contributors apply to each sample without triggering a full
pipeline re-run.  Changes are staged inside the form and only committed to
``Sample.metadata`` when the user clicks **Apply changes**.
"""

from __future__ import annotations

import copy
import fnmatch
import json
import math
from typing import Iterable, Mapping

import pandas as pd
import streamlit as st

from config.contributor_names import (
    canonical_contributor_mapping,
    canonical_contributor_name,
)
from domain.models import Sample
from domain.uncertainty.contributors import (
    BUILTIN_PROFILES,
    PROFILE_DEFAULTS,
    ContributorProfile,
    SampleContributorApplicability,
    normalize_profile_defaults,
)
from domain.uncertainty.sr_sample_values import (
    SR_DIGESTION_KEY,
    SR_QC_BIAS_KEY,
    SR_SAMPLE_VALUES_KEY,
)
from ui.edit_actions import (
    _sample_key,
    _record_bulk_undo_applied,
    _restore_bulk_undo,
    _store_bulk_undo,
    _write_applicability,
    _write_sr_sample_value_overrides,
)
from ui.runtime_budget_cache import clear_runtime_budget_cache
from ui.state import get_state
from ui.tabs.uncertainty_sections.profile_manager import (
    profile_label,
    profile_options,
    render_profile_manager,
)


def _sample_key_token(sample: Sample) -> str:
    """Return an Arrow-safe editor token for a sample identity."""
    return json.dumps(_sample_key(sample), separators=(",", ":"))


def _sample_key_from_editor_value(value: object) -> tuple | None:
    """Decode a sample identity written into the Streamlit editor frame."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None

    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (int(value[0]), str(value[1]))
        except (TypeError, ValueError):
            return None

    return None


def _eligible_samples(
    samples: Iterable[Sample], include_standards: bool = False
) -> list[Sample]:
    """Return non-excluded measurement samples, and standards when include_standards is True."""
    return [
        sample for sample in samples
        if (sample.is_sample or (include_standards and sample.is_standard))
        and not (sample.metadata or {}).get("excluded", False)
    ]


def _finite_positive_or_none(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def _sr_values_block(sample: Sample) -> dict:
    uc = (sample.metadata or {}).get("uncertainty_contributors") or {}
    if not isinstance(uc, dict):
        return {}
    values = uc.get(SR_SAMPLE_VALUES_KEY) or {}
    return values if isinstance(values, dict) else {}


def _value_or_default(
    sample: Sample,
    contributor_name: str,
    field_name: str,
    default_value: float,
) -> float:
    block = _sr_values_block(sample).get(contributor_name) or {}
    if not isinstance(block, dict):
        return float(default_value)
    override = _finite_positive_or_none(block.get(field_name))
    return float(override if override is not None else default_value)


def _store_if_differs(
    out: dict,
    *,
    contributor_name: str,
    field_name: str,
    value: object,
    default_value: float,
) -> None:
    parsed = _finite_positive_or_none(value)
    default_parsed = _finite_positive_or_none(default_value)
    if parsed is None:
        return
    if default_parsed is not None and math.isclose(
        parsed,
        default_parsed,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        return
    out.setdefault(contributor_name, {})[field_name] = float(parsed)


def render_sr_sample_value_matrix(
    samples: list[Sample],
    *,
    qc_bias_enabled: bool,
    reprod_dig_enabled: bool,
    sr_qc_bias_abs: float,
    sr_qc_cert_value: float,
    u_reprod_dig_sd: float,
    u_reprod_dig_ref_value: float,
    expanded: bool = True,
) -> bool:
    """Render Sr/Pb per-sample numeric inputs for optional top-down terms."""
    if not (qc_bias_enabled or reprod_dig_enabled):
        return False

    eligible = _eligible_samples(samples, include_standards=False)
    if not eligible:
        st.info("No active samples are available for sample-specific contributor values.")
        return False

    with st.expander("Sample-specific contributor values", expanded=expanded):
        st.caption(
            "Enter absolute numerator and denominator values for each sample. "
            "Cells matching the session defaults are stored as inherited values."
        )

        defaults = {
            "qc_bias_numerator": float(sr_qc_bias_abs),
            "qc_certified_ratio": float(sr_qc_cert_value),
            "digestion_sd": float(u_reprod_dig_sd),
            "digestion_reference_ratio": float(u_reprod_dig_ref_value),
        }

        rows = []
        for sample in eligible:
            row = {
                "_key": _sample_key_token(sample),
                "Sample": sample.name,
                "Run": sample.run_number,
            }
            if qc_bias_enabled:
                row["QC bias numerator"] = _value_or_default(
                    sample,
                    SR_QC_BIAS_KEY,
                    "numerator_abs",
                    defaults["qc_bias_numerator"],
                )
                row["QC certified ratio"] = _value_or_default(
                    sample,
                    SR_QC_BIAS_KEY,
                    "denominator_ratio",
                    defaults["qc_certified_ratio"],
                )
            if reprod_dig_enabled:
                row["Digestion SD"] = _value_or_default(
                    sample,
                    SR_DIGESTION_KEY,
                    "sd_abs",
                    defaults["digestion_sd"],
                )
                row["Digestion reference ratio"] = _value_or_default(
                    sample,
                    SR_DIGESTION_KEY,
                    "denominator_ratio",
                    defaults["digestion_reference_ratio"],
                )
            rows.append(row)

        column_config = {
            "_key": None,
            "Sample": st.column_config.TextColumn("Sample", disabled=True),
            "Run": st.column_config.NumberColumn("Run", disabled=True, width="small"),
        }
        if qc_bias_enabled:
            column_config["QC bias numerator"] = st.column_config.NumberColumn(
                "QC bias numerator",
                min_value=0.0,
                format="%.10f",
                help="Absolute standard uncertainty numerator for the processed control sample.",
            )
            column_config["QC certified ratio"] = st.column_config.NumberColumn(
                "QC certified ratio",
                min_value=0.0,
                format="%.10f",
                help="Denominator ratio for the processed control material.",
            )
        if reprod_dig_enabled:
            column_config["Digestion SD"] = st.column_config.NumberColumn(
                "Digestion SD",
                min_value=0.0,
                format="%.10f",
                help="Absolute SD of independently processed digestion means.",
            )
            column_config["Digestion reference ratio"] = st.column_config.NumberColumn(
                "Digestion reference ratio",
                min_value=0.0,
                format="%.10f",
                help="Denominator ratio for the material used to estimate digestion SD.",
            )

        with st.form("sr_sample_contributor_values_form", clear_on_submit=False):
            edited = st.data_editor(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                num_rows="fixed",
                disabled=["_key", "Sample", "Run"],
                column_config=column_config,
                key="sr_sample_contributor_values_editor",
            )
            apply_changes = st.form_submit_button("Apply contributor values", type="primary")

        if apply_changes:
            by_key = {_sample_key(sample): sample for sample in eligible}
            missing_qc = 0
            missing_dig = 0
            for _, row in edited.iterrows():
                key = _sample_key_from_editor_value(row["_key"])
                sample = by_key.get(key)
                if sample is None:
                    continue

                values: dict[str, dict[str, float]] = copy.deepcopy(
                    _sr_values_block(sample)
                )
                if qc_bias_enabled:
                    qc_values: dict[str, dict[str, float]] = {}
                    _store_if_differs(
                        qc_values,
                        contributor_name=SR_QC_BIAS_KEY,
                        field_name="numerator_abs",
                        value=row.get("QC bias numerator"),
                        default_value=defaults["qc_bias_numerator"],
                    )
                    _store_if_differs(
                        qc_values,
                        contributor_name=SR_QC_BIAS_KEY,
                        field_name="denominator_ratio",
                        value=row.get("QC certified ratio"),
                        default_value=defaults["qc_certified_ratio"],
                    )
                    if qc_values.get(SR_QC_BIAS_KEY):
                        values[SR_QC_BIAS_KEY] = qc_values[SR_QC_BIAS_KEY]
                    else:
                        values.pop(SR_QC_BIAS_KEY, None)
                    qc_num = _finite_positive_or_none(row.get("QC bias numerator"))
                    qc_den = _finite_positive_or_none(row.get("QC certified ratio"))
                    if qc_num is not None and qc_den is None:
                        missing_qc += 1

                if reprod_dig_enabled:
                    digestion_values: dict[str, dict[str, float]] = {}
                    _store_if_differs(
                        digestion_values,
                        contributor_name=SR_DIGESTION_KEY,
                        field_name="sd_abs",
                        value=row.get("Digestion SD"),
                        default_value=defaults["digestion_sd"],
                    )
                    _store_if_differs(
                        digestion_values,
                        contributor_name=SR_DIGESTION_KEY,
                        field_name="denominator_ratio",
                        value=row.get("Digestion reference ratio"),
                        default_value=defaults["digestion_reference_ratio"],
                    )
                    if digestion_values.get(SR_DIGESTION_KEY):
                        values[SR_DIGESTION_KEY] = digestion_values[SR_DIGESTION_KEY]
                    else:
                        values.pop(SR_DIGESTION_KEY, None)
                    dig_sd = _finite_positive_or_none(row.get("Digestion SD"))
                    dig_den = _finite_positive_or_none(row.get("Digestion reference ratio"))
                    if dig_sd is not None and dig_den is None:
                        missing_dig += 1

                _write_sr_sample_value_overrides(sample, values=values)

            _sync_applicability_to_state_samples(by_key.values())
            clear_runtime_budget_cache()
            if missing_qc:
                st.warning(
                    f"{missing_qc} sample(s) have a QC numerator without a positive certified ratio."
                )
            if missing_dig:
                st.warning(
                    f"{missing_dig} sample(s) have a digestion SD without a positive reference ratio."
                )
            st.success("Sample-specific contributor values updated.")
            st.rerun()
            return True

    return False


def _sync_applicability_to_matching_samples(
    source_samples: Iterable[Sample],
    target_samples: Iterable[Sample] | None,
) -> int:
    """Copy applicability metadata from processed samples back to raw samples."""
    if target_samples is None:
        return 0

    targets = {_sample_key(sample): sample for sample in target_samples}
    count = 0
    for source in source_samples:
        target = targets.get(_sample_key(source))
        if target is None:
            continue

        source_metadata = dict(getattr(source, "metadata", {}) or {})
        target.metadata = dict(getattr(target, "metadata", {}) or {})
        if "uncertainty_contributors" in source_metadata:
            target.metadata["uncertainty_contributors"] = copy.deepcopy(
                source_metadata["uncertainty_contributors"]
            )
        else:
            target.metadata.pop("uncertainty_contributors", None)
        count += 1

    return count


def _sync_applicability_to_state_samples(source_samples: Iterable[Sample]) -> int:
    """Keep applicability edits made on ``result.samples`` for future processing."""
    try:
        state = get_state()
    except Exception:
        return 0
    return _sync_applicability_to_matching_samples(
        source_samples,
        getattr(state, "samples", None),
    )


def _effective_builtin_enabled(
    applicability: SampleContributorApplicability,
    contributor_name: str,
    profile_defaults: Mapping[str, Mapping[str, bool]] | None = None,
) -> bool:
    """Return the checkbox value implied by overrides plus profile defaults."""
    contributor_name = canonical_contributor_name(contributor_name)
    overrides = canonical_contributor_mapping(applicability.overrides)
    if contributor_name in overrides:
        return bool(overrides[contributor_name])
    defaults = (
        normalize_profile_defaults(profile_defaults)
        if profile_defaults is not None
        else PROFILE_DEFAULTS
    )
    profile_defaults_for_sample = canonical_contributor_mapping(
        defaults.get(applicability.profile, {})
    )
    return bool(profile_defaults_for_sample.get(contributor_name, True))


def _profile_default_enabled(
    profile: str,
    contributor_name: str,
    profile_defaults: Mapping[str, Mapping[str, bool]] | None = None,
) -> bool:
    """Return the default enabled state for one contributor under one profile."""
    contributor_name = canonical_contributor_name(contributor_name)
    defaults = (
        normalize_profile_defaults(profile_defaults)
        if profile_defaults is not None
        else PROFILE_DEFAULTS
    )
    profile_defaults_for_sample = canonical_contributor_mapping(defaults.get(profile, {}))
    return bool(profile_defaults_for_sample.get(contributor_name, True))


def _edited_builtin_overrides(
    row: Mapping[str, object],
    *,
    built_in_names: Iterable[str],
    profile: str,
    profile_defaults: Mapping[str, Mapping[str, bool]] | None = None,
) -> dict[str, bool]:
    """Return minimal overrides from an edited matrix row.

    When only the profile cell changed, Streamlit keeps old checkbox values in
    the staged row. Treat unchanged checkbox cells as following the new profile
    default so stale all-True overrides do not mask profile defaults.
    """
    initial_profile = str(row.get("_initial_profile") or profile)
    profile_changed = profile != initial_profile
    overrides: dict[str, bool] = {}

    for raw_name in built_in_names:
        name = canonical_contributor_name(raw_name)
        current = bool(row.get(raw_name, row.get(name, True)))
        initial_key = f"_initial_{raw_name}"
        initial = bool(row.get(initial_key, current))
        effective = current
        if profile_changed and current == initial:
            effective = _profile_default_enabled(profile, name, profile_defaults)

        profile_default = _profile_default_enabled(profile, name, profile_defaults)
        if effective != profile_default:
            overrides[name] = effective

    return dict(sorted(overrides.items()))


def _sample_matches_search(sample: Sample, term: str) -> bool:
    """Return whether *sample* matches a case-insensitive text/glob query."""
    query = str(term or "").strip().lower()
    if not query:
        return True
    fields = [
        str(sample.name).lower(),
        str(sample.run_number).lower(),
        str(sample.sample_type).lower(),
    ]
    if "*" in query:
        return any(fnmatch.fnmatchcase(field, query) for field in fields)
    return any(query in field for field in fields)


def _apply_profile_to_sample(
    sample: Sample,
    *,
    profile: str,
    reset_overrides: bool,
    profiles: Mapping[str, ContributorProfile],
    source: str = "user_assigned",
    profile_defaults: Mapping[str, Mapping[str, bool]] | None = None,
) -> None:
    """Apply a profile to one sample while preserving unrelated metadata."""
    resolved_profile_defaults = (
        normalize_profile_defaults(profile_defaults)
        if profile_defaults is not None
        else None
    )
    selected_profile = profiles.get(profile)
    if selected_profile is None:
        raise ValueError(f"Unknown uncertainty profile {profile!r}.")
    applicability = SampleContributorApplicability.from_sample(
        sample,
        known_profiles=set(profiles.keys()),
    )
    if reset_overrides:
        overrides = {}
        custom_enabled = list(selected_profile.custom_enabled_defaults)
    else:
        overrides = dict(applicability.overrides)
        custom_enabled = list(applicability.custom_enabled)
    _write_applicability(
        sample,
        profile=profile,
        overrides=overrides,
        custom_enabled=custom_enabled,
        note=applicability.note,
        source=source,
        profile_defaults=(
            resolved_profile_defaults.get(profile, {})
            if resolved_profile_defaults is not None
            else None
        ),
    )


def apply_profile_to_samples(
    samples: list[Sample],
    *,
    selected_keys: set[tuple],
    profile: str,
    reset_overrides: bool,
    profiles: Mapping[str, ContributorProfile],
    profile_defaults: Mapping[str, Mapping[str, bool]] | None = None,
    include_standards: bool = False,
) -> int:
    """Apply *profile* to eligible samples with keys in *selected_keys*."""
    count = 0
    for sample in _eligible_samples(samples, include_standards=include_standards):
        if _sample_key(sample) not in selected_keys:
            continue
        _apply_profile_to_sample(
            sample,
            profile=profile,
            reset_overrides=reset_overrides,
            profiles=profiles,
            profile_defaults=profile_defaults,
        )
        count += 1
    return count


def _override_indicator(applicability: SampleContributorApplicability) -> str:
    """Return a compact, Arrow-safe marker of row-level deviations.

    Override detail stays in ``sample.metadata``; this is display only.
    """
    n_overrides = len(canonical_contributor_mapping(applicability.overrides))
    n_custom = len(applicability.custom_enabled)
    parts: list[str] = []
    if n_overrides:
        parts.append(f"{n_overrides} override" + ("s" if n_overrides != 1 else ""))
    if n_custom:
        parts.append(f"{n_custom} custom")
    return " | ".join(parts) if parts else "-"


def _profile_distribution(
    eligible: list[Sample],
    profiles: Mapping[str, ContributorProfile],
) -> list[tuple[str, int]]:
    """Count eligible samples per resolved profile (non-zero, stable order)."""
    counts: dict[str, int] = {}
    known = set(profiles.keys())
    for sample in eligible:
        applicability = SampleContributorApplicability.from_sample(
            sample,
            known_profiles=known,
        )
        counts[applicability.profile] = counts.get(applicability.profile, 0) + 1
    return [(name, counts[name]) for name in profiles if counts.get(name)]


def _render_profile_chips(
    distribution: list[tuple[str, int]],
    profiles: Mapping[str, ContributorProfile],
) -> None:
    """Render compact per-profile sample-count metrics above the editor."""
    if not distribution:
        return
    chips = distribution[:6]
    cols = st.columns(len(chips))
    for col, (name, count) in zip(cols, chips):
        label = profile_label(profiles[name]) if name in profiles else name
        col.metric(label, count)


def render_per_sample_contributor_matrix(
    samples: list[Sample],
    *,
    built_in_contributors: list[tuple[str, str]],
    custom_contributors: list,
    expanded: bool = False,
    profile_manager_expanded: bool = False,
    include_standards: bool = False,
) -> bool:
    """Render the per-sample contributor applicability matrix inside an expander.

    Parameters
    ----------
    samples:
        Full session sample list (method filters to eligible measurement
        samples internally).
    built_in_contributors:
        List of ``(name, display_label)`` tuples for the built-in
        uncertainty contributors active for the current element/engine
        combination.
    custom_contributors:
        List of ``CustomUncertaintyContributor`` objects from the library.

    Returns
    -------
    bool
        ``True`` if applicability was changed and written during this call
        (i.e. *Apply changes* was submitted); ``False`` otherwise.
    """
    eligible = _eligible_samples(samples, include_standards=include_standards)
    if not eligible:
        st.info("No active samples are available for per-sample contributor settings.")
        return False

    state = get_state()
    try:
        profiles: dict[str, ContributorProfile] = state.uncertainty_profiles
        profile_defaults = normalize_profile_defaults(state.uncertainty_profile_defaults)
    except Exception as exc:
        st.error(f"User uncertainty profiles could not be loaded: {exc}")
        profiles = dict(BUILTIN_PROFILES)
        profile_defaults = PROFILE_DEFAULTS

    with st.expander("Sample contributor applicability", expanded=expanded):
        st.caption(
            "Select which uncertainty contributors apply to each sample. "
            "Changes are staged until **Apply changes** is clicked."
        )

        if render_profile_manager(
            profiles=profiles,
            profile_defaults=profile_defaults,
            samples=samples,
            built_in_contributors=built_in_contributors,
            custom_contributors=custom_contributors,
            expanded=profile_manager_expanded,
            include_standards=include_standards,
            expected_mtime=getattr(state, "uncertainty_profiles_mtime", None),
        ):
            _sync_applicability_to_state_samples(samples)
            state.reload_uncertainty_profiles()
            st.rerun()

        option_names = profile_options(profiles)

        _render_profile_chips(
            _profile_distribution(eligible, profiles),
            profiles,
        )
        editor_counter = st.session_state.get(
            "sample_contributor_applicability_editor_counter",
            0,
        )
        editor_key = f"sample_contributor_applicability_editor_{editor_counter}"

        st.caption("Bulk assignment")
        search_col, profile_col, reset_col, apply_col = st.columns([3, 2, 2, 2])
        with search_col:
            search_term = st.text_input(
                "Sample search",
                key="uncertainty_profile_bulk_search",
                placeholder="name, run number, type, or glob",
            )
        match_keys = {
            _sample_key(sample)
            for sample in eligible
            if _sample_matches_search(sample, search_term)
        }
        with profile_col:
            bulk_profile = st.selectbox(
                "Profile",
                options=option_names,
                format_func=lambda name: profile_label(profiles[name]),
                key="uncertainty_profile_bulk_profile",
            )
        with reset_col:
            bulk_reset = st.checkbox(
                "Reset overrides",
                value=False,
                key="uncertainty_profile_bulk_reset",
            )
            st.caption(f"{len(match_keys)} matches")
        with apply_col:
            st.write("")
            st.write("")
            apply_matches = st.button(
                "Apply to matches",
                key="uncertainty_profile_apply_matches",
            )

        undo_col, _ = st.columns([2, 8])
        with undo_col:
            undo_bulk = st.button(
                "Undo last bulk assignment",
                key="uncertainty_profile_undo_bulk",
                disabled=not bool(st.session_state.get("_uncertainty_profile_bulk_undo")),
            )

        if undo_bulk:
            count, skipped = _restore_bulk_undo(eligible)
            if skipped:
                st.warning(
                    f"{skipped} sample(s) were changed after that assignment and "
                    "were left as they are."
                )
            if count:
                _sync_applicability_to_state_samples(eligible)
                clear_runtime_budget_cache()
                st.success(f"Restored {count} samples.")
                st.rerun()

        if apply_matches:
            matched = [sample for sample in eligible if _sample_key(sample) in match_keys]
            _store_bulk_undo(matched)
            count = apply_profile_to_samples(
                samples,
                selected_keys=match_keys,
                profile=bulk_profile,
                reset_overrides=bulk_reset,
                profiles=profiles,
                profile_defaults=profile_defaults,
                include_standards=include_standards,
            )
            if count:
                _record_bulk_undo_applied(matched)
                _sync_applicability_to_state_samples(matched)
                clear_runtime_budget_cache()
                st.success(f"Applied {bulk_profile} to {count} samples.")
                st.rerun()
                return True
            st.info("No matching eligible samples were updated.")

        with st.form("sample_contributor_applicability_form", clear_on_submit=False):
            rows = []
            for sample in eligible:
                applicability = SampleContributorApplicability.from_sample(
                    sample,
                    known_profiles=set(profiles.keys()),
                )
                row: dict = {
                    "_key": _sample_key_token(sample),
                    "Select": False,
                    "Sample": sample.name,
                    "Run": sample.run_number,
                    "Profile": applicability.profile,
                    "_initial_profile": applicability.profile,
                    "Source": applicability.source,
                    "Overrides": _override_indicator(applicability),
                    "Note": applicability.note,
                }
                for name, _label in built_in_contributors:
                    effective_enabled = _effective_builtin_enabled(
                        applicability,
                        name,
                        profile_defaults,
                    )
                    row[name] = effective_enabled
                    row[f"_initial_{name}"] = effective_enabled
                for definition in custom_contributors:
                    row[definition.name] = definition.name in applicability.custom_enabled
                rows.append(row)

            column_config = {
                "_key": None,
                "Select": st.column_config.CheckboxColumn("Select", width="small"),
                "Profile": st.column_config.SelectboxColumn(
                    "Profile",
                    options=option_names,
                    required=True,
                ),
                "Source": st.column_config.TextColumn("Source", disabled=True, width="small"),
                "Overrides": st.column_config.TextColumn(
                    "Overrides",
                    disabled=True,
                    width="small",
                    help="Row-level deviations from the profile default "
                    "(detail is kept in sample metadata).",
                ),
            }
            for name, label in built_in_contributors:
                column_config[name] = st.column_config.CheckboxColumn(label)
                column_config[f"_initial_{name}"] = None
            for definition in custom_contributors:
                column_config[definition.name] = st.column_config.CheckboxColumn(
                    getattr(definition, "display_name", definition.name)
                )
            column_config["_initial_profile"] = None

            edited = st.data_editor(
                pd.DataFrame(rows),
                hide_index=True,
                width="stretch",
                num_rows="fixed",
                disabled=["_key", "Sample", "Run", "Source", "Overrides"],
                column_config=column_config,
                key=editor_key,
            )

            selected_profile_col, selected_reset_col, _ = st.columns([3, 2, 5])
            with selected_profile_col:
                selected_profile = st.selectbox(
                    "Selected rows profile",
                    options=option_names,
                    format_func=lambda name: profile_label(profiles[name]),
                    key="uncertainty_profile_selected_rows_profile",
                )
            with selected_reset_col:
                selected_reset = st.checkbox(
                    "Reset selected overrides",
                    value=False,
                    key="uncertainty_profile_selected_rows_reset",
                )

            apply_col, discard_col, selected_col, _ = st.columns([2, 2, 3, 3])
            with apply_col:
                apply_changes = st.form_submit_button("Apply changes", type="primary")
            with discard_col:
                discard_changes = st.form_submit_button("Discard changes")
            with selected_col:
                apply_selected = st.form_submit_button("Apply profile to selected")

        if discard_changes:
            st.session_state["sample_contributor_applicability_editor_counter"] = (
                editor_counter + 1
            )
            st.rerun()

        if apply_selected:
            selected_keys = set()
            for _, row in edited.iterrows():
                if not bool(row.get("Select", False)):
                    continue
                key = _sample_key_from_editor_value(row["_key"])
                if key is not None:
                    selected_keys.add(key)
            matched = [sample for sample in eligible if _sample_key(sample) in selected_keys]
            _store_bulk_undo(matched)
            count = apply_profile_to_samples(
                samples,
                selected_keys=selected_keys,
                profile=selected_profile,
                reset_overrides=selected_reset,
                profiles=profiles,
                profile_defaults=profile_defaults,
                include_standards=include_standards,
            )
            if count:
                _record_bulk_undo_applied(matched)
                _sync_applicability_to_state_samples(matched)
                clear_runtime_budget_cache()
                st.success(f"Applied {selected_profile} to {count} selected samples.")
                st.rerun()
                return True
            st.info("No selected eligible samples were updated.")

        if apply_changes:
            by_key = {_sample_key(sample): sample for sample in eligible}
            custom_names = {definition.name for definition in custom_contributors}
            built_in_names = {name for name, _label in built_in_contributors}

            for _, row in edited.iterrows():
                key = _sample_key_from_editor_value(row["_key"])
                sample = by_key.get(key)
                if sample is None:
                    continue

                profile = str(row.get("Profile") or "full_chemistry")
                if profile not in profiles:
                    profile = "custom"

                overrides = _edited_builtin_overrides(
                    row,
                    built_in_names=built_in_names,
                    profile=profile,
                    profile_defaults=profile_defaults,
                )
                custom_enabled = [
                    name for name in custom_names
                    if bool(row.get(name, False))
                ]

                _write_applicability(
                    sample,
                    profile=profile,
                    overrides=overrides,
                    custom_enabled=custom_enabled,
                    note=str(row.get("Note") or ""),
                    source="user_assigned",
                    profile_defaults=profile_defaults.get(profile, {}),
                    owned_built_in=built_in_names,
                    owned_custom=custom_names,
                )

            _sync_applicability_to_state_samples(by_key.values())
            clear_runtime_budget_cache()
            st.success("Sample contributor applicability updated.")
            st.rerun()
            return True

    return False

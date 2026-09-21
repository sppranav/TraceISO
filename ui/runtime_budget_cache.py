from __future__ import annotations

from copy import deepcopy

import dataclasses
import hashlib
from collections import OrderedDict
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import streamlit as st

from ui.diagnostics import cache_event, timed

from config.scientific_identity import scientific_configuration, digest
from config.constants import SR_GEOREM_REFERENCE_MATERIAL
from config.contributor_names import canonical_contributor_mapping
from config.settings import ProcessingConfig, UncertaintyConfig
from domain.filters.outlier import sample_cycle_key
from domain.models import Sample, UncertaintyBudget
from domain.uncertainty.runtime import (
    RuntimeUncertaintyMap,
    compute_runtime_budget,
    make_runtime_uncertainty_key,
)


_RUNTIME_BUDGET_CACHE_KEY = "_runtime_budget_cache_v1"
#: A095: completed runtime maps, keyed by the exact request that produced them.
#: The per-budget LRU below is a *scan* cache: a session whose working set is
#: larger than its cap evicts the entry each pass needs next, so two identical
#: full-map renders recomputed every budget twice. This second, coarser cache
#: answers the repeated whole-map question directly and never consults the LRU.
_RUNTIME_MAP_CACHE_KEY = "_runtime_budget_map_cache_v1"
_RUNTIME_BUDGET_ALGORITHM_TOKEN = "runtime_budget_v9_scientific_observation_identity"
# item 89: practical session cap — evict oldest entries beyond this limit.
_MAX_CACHE_SIZE = 512
#: Memory bound on the completed-map cache. Deliberately small: each retained
#: map holds one budget per sample and ratio, so the bound is a small multiple
#: of one session, not an unbounded history. This is not a production
#: performance target — it is the smallest reuse that removes the repeated
#: scan-eviction, and a target may be selected separately.
_MAX_MAP_CACHE_SIZE = 2
_CYCLE_DATA_LAYER_NAMES = (
    "intensities",
    "corrected_intensities",
    "blank_corrected_intensities",
    "ratios",
    "corrected_ratios",
    "blank_corrected_ratios",
    "iif_corrected_ratios",
    "drift_corrected_ratios",
    "interference_corrected_intensities",
    "interference_corrected_ratios",
    "sr_standard_corrected_ratios",
    "pb_standard_corrected_ratios",
    "pb_calibrated_delta_cycles",
)


def _freeze_for_cache(value: Any) -> Any:
    """Convert nested runtime settings into hashable cache tokens."""
    if dataclasses.is_dataclass(value):
        return _freeze_for_cache(dataclasses.asdict(value))
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_for_cache(item))
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_cache(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_for_cache(item) for item in value))
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        return (
            "ndarray",
            str(arr.dtype),
            tuple(arr.shape),
            tuple(arr.ravel().tolist()),
        )
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sample_cache_token(sample: Sample) -> Tuple[Any, ...]:
    """Return a stable token for a sample object in the current session.

    Includes observation identity: identical numbers can denote independent inputs.
    """
    metadata = getattr(sample, "metadata", {}) or {}
    applicability = metadata.get("uncertainty_contributors") or {}
    return (
        str(getattr(sample, "observation_id", "")),
        _freeze_for_cache(metadata),
        _blank_references(sample),
        int(getattr(sample, "run_number", 0) or 0),
        str(getattr(sample, "name", "")),
        str(getattr(sample, "sample_type", "")),
        bool(metadata.get("excluded", False)),
        _freeze_for_cache(metadata.get("manual_exclusions") or ()),
        _freeze_for_cache(metadata.get("_original_masks") or {}),
        _freeze_for_cache(metadata.get("_applied_correction_state") or {}),
        _freeze_for_cache(metadata.get("_pb_hg_correction_state") or {}),
        _sample_mask_token(sample),
        _freeze_for_cache(applicability),
        _sample_content_digest(sample),
    )


def _sample_content_digest(sample: Sample) -> str:
    """Digest the measured numbers a runtime budget would be computed from.

    Covers every cycle-data layer plus the SSB and delta producer payloads.
    Hashing the raw buffers keeps this cheap enough to run once per sample per
    render, which matters because the session token digests every sample.
    """
    hasher = hashlib.blake2b(digest_size=16)
    for layer_name in _CYCLE_DATA_LAYER_NAMES:
        mapping = getattr(sample, layer_name, None)
        if not isinstance(mapping, dict):
            continue
        hasher.update(layer_name.encode("utf-8"))
        for key in sorted(mapping, key=str):
            cycle_data = mapping[key]
            values = getattr(cycle_data, "values", None)
            mask = getattr(cycle_data, "mask", None)
            hasher.update(str(key).encode("utf-8"))
            if values is not None:
                hasher.update(np.ascontiguousarray(values, dtype=np.float64).tobytes())
            if mask is not None:
                hasher.update(np.ascontiguousarray(mask, dtype=bool).tobytes())
    for payload_name in ("ssb_results", "delta_results"):
        payload = getattr(sample, payload_name, None)
        if not isinstance(payload, dict):
            continue
        hasher.update(payload_name.encode("utf-8"))
        hasher.update(repr(_freeze_for_cache(payload)).encode("utf-8"))
    if getattr(sample, "correction_records", None):
        # The Hg record decides whether a ratio has a final value at all, so a
        # status change must change the token even when no array moved.
        from domain.pb_correction_records import correction_records_payload

        hasher.update(b"correction_records")
        hasher.update(repr(_freeze_for_cache(correction_records_payload(sample))).encode("utf-8"))
    return hasher.hexdigest()


def _blank_references(sample: Sample) -> Tuple[Tuple[str, str, str], ...]:
    """Every recorded blank reference as ``(role, observation_id, label)``.

    Sorted by role, so the dictionary insertion order the producer happened to
    use cannot change the token, while ``before`` and ``after`` stay
    distinguishable — they are weighted differently.
    """
    ids = getattr(sample, "used_blank_ids", None) or {}
    labels = getattr(sample, "used_blanks", None) or {}
    roles = {str(role) for role in ids} | {str(role) for role in labels}
    return tuple(
        (
            role,
            str(ids.get(role, "") or ""),
            str(labels.get(role, "") or ""),
        )
        for role in sorted(roles)
    )


def _reference_index(
    session_samples: Iterable[Sample],
) -> Tuple[Dict[str, Sample], Dict[str, list]]:
    """Index one session by observation ID and by label, once per runtime pass.

    Hoisted for the same reason as the session token: a map build asks the same
    question for every sample and ratio, and rescanning the session each time
    turns a render into quadratic work.
    """
    by_id: Dict[str, Sample] = {}
    by_name: Dict[str, list] = {}
    for candidate in session_samples:
        observation_id = str(getattr(candidate, "observation_id", "") or "")
        if observation_id:
            by_id[observation_id] = candidate
        by_name.setdefault(str(getattr(candidate, "name", "")), []).append(candidate)
    return by_id, by_name


def _reference_dependency_token(
    sample: Sample,
    session_samples: Tuple[Sample, ...],
    index: Optional[Tuple[Dict[str, Sample], Dict[str, list]]] = None,
) -> Tuple[Any, ...]:
    """Digest what each recorded blank reference *denotes* in this session.

    The requesting sample's own content token cannot see this. Two blanks may
    share a mean and differ only in scatter, so switching which one a sample
    records changes its uncertainty while every array in the session stays
    exactly where it was — and re-pointing an identity at different arrays does
    the same in reverse.

    What goes in is the *resolution*, not the raw identity: a resolved
    reference contributes the referenced observation's content digest and its
    cycle-window key, so two sessions whose references denote equivalent
    observations still share one cache entry. An unresolved or ambiguous
    reference contributes the recorded text as well, because the refusal it
    produces names it.
    """
    references = _blank_references(sample)
    if not references:
        return ()

    by_id, by_name = index if index is not None else _reference_index(session_samples)

    def _denotes(candidate: Sample) -> Tuple[Any, ...]:
        return (candidate.observation_id, _sample_content_digest(candidate), sample_cycle_key(candidate))

    tokens = []
    for role, observation_id, label in references:
        if observation_id:
            found = by_id.get(observation_id)
            matches = [found] if found is not None else []
            recorded = observation_id
        elif label:
            matches = list(by_name.get(label, ()))
            recorded = label
        else:
            matches = []
            recorded = ""
        if len(matches) == 1:
            tokens.append((role, "resolved", _denotes(matches[0])))
        elif matches:
            tokens.append((
                role,
                "ambiguous",
                recorded,
                tuple(sorted(_denotes(match) for match in matches)),
            ))
        else:
            tokens.append((role, "unresolved", recorded))
    return tuple(tokens)


def _cycle_data_mapping_mask_token(mapping: Any) -> Tuple[Any, ...]:
    if not isinstance(mapping, dict):
        return ()
    items = []
    for key, cycle_data in mapping.items():
        mask = getattr(cycle_data, "mask", None)
        if mask is None:
            continue
        arr = np.asarray(mask, dtype=bool)
        items.append((str(key), tuple(arr.tolist())))
    return tuple(sorted(items, key=lambda item: item[0]))


def _ssb_mask_token(sample: Sample) -> Tuple[Any, ...]:
    items = []
    for ratio_name, payload in (getattr(sample, "ssb_results", {}) or {}).items():
        if not isinstance(payload, dict) or "ssb_mask" not in payload:
            continue
        arr = np.asarray(payload["ssb_mask"], dtype=bool)
        items.append((str(ratio_name), tuple(arr.tolist())))
    return tuple(sorted(items, key=lambda item: item[0]))


def _sample_mask_token(sample: Sample) -> Tuple[Any, ...]:
    """Return current mask state for runtime-sensitive sample layers."""
    layer_tokens = []
    for layer_name in _CYCLE_DATA_LAYER_NAMES:
        token = _cycle_data_mapping_mask_token(getattr(sample, layer_name, None))
        if token:
            layer_tokens.append((layer_name, token))
    ssb_token = _ssb_mask_token(sample)
    if ssb_token:
        layer_tokens.append(("ssb_results", ssb_token))
    return tuple(layer_tokens)


def _custom_contributor_token(library: Optional[Any]) -> Any:
    """Stable cache token for the full custom-contributor library dict."""
    all_defs: list = []
    for element_defs in (library or {}).values():
        for item in element_defs:
            all_defs.append({
                "name": getattr(item, "name", ""),
                "element_symbol": getattr(item, "element_symbol", ""),
                "u_rel_permil": getattr(item, "u_rel_permil", 0.0),
                "type_ab": getattr(item, "type_ab", ""),
                "degrees_of_freedom": getattr(item, "degrees_of_freedom", float("inf")),
                "distribution": getattr(item, "distribution", "normal"),
                "description": getattr(item, "description", ""),
                "reference": getattr(item, "reference", ""),
                "enabled": getattr(item, "enabled", True),
            })
    return _freeze_for_cache(
        sorted(all_defs, key=lambda d: (d["element_symbol"], d["name"]))
    )


def _profile_defaults_token(
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]],
) -> Tuple[Any, ...]:
    """Stable cache token for active uncertainty profile defaults."""
    if not profile_defaults:
        return ()
    return tuple(
        (
            str(profile),
            tuple(
                sorted(
                    (str(k), bool(v))
                    for k, v in canonical_contributor_mapping(defaults).items()
                )
            ),
        )
        for profile, defaults in sorted(profile_defaults.items(), key=lambda item: str(item[0]))
    )


def _certificate_contents_token(
    element_config: Any,
    processing_config: Optional[ProcessingConfig],
) -> Tuple[Any, ...]:
    """Token for the certificate numbers this budget would actually resolve.

    A017: the selected reference is identified by element symbol and material
    name, and a CRM Manager edit deliberately keeps both. Without the resolved
    contents in the key, an edited certificate reused the previous budget —
    including its u_CRM — after a library reload. Sr additionally reads the
    fixed GeoReM reference for ``u_ref_value``, which is a different record and
    is captured separately.
    """
    from config.reference_materials import resolved_reference_snapshot

    symbol = str(getattr(element_config, "symbol", "") or "")
    if not symbol:
        return ()
    rm_name = (
        getattr(processing_config, "reference_material", None)
        or getattr(element_config, "reference_material", None)
        or ""
    )
    entries: list = []
    if rm_name:
        entries.append(("selected", str(rm_name), resolved_reference_snapshot(symbol, str(rm_name))))
    if symbol == "Sr":
        entries.append((
            "ref_value",
            SR_GEOREM_REFERENCE_MATERIAL,
            resolved_reference_snapshot(symbol, SR_GEOREM_REFERENCE_MATERIAL),
        ))
    return tuple(entries)


def _processing_config_token(config: Optional[ProcessingConfig]) -> Any:
    if config is None:
        return None
    return _freeze_for_cache(dataclasses.asdict(config))


def _uncertainty_config_token(config: Optional[UncertaintyConfig]) -> Any:
    if config is None:
        return None
    return _freeze_for_cache(dataclasses.asdict(config))



def clear_runtime_budget_cache() -> None:
    """Drop the shared runtime-budget caches from session state."""
    st.session_state.pop(_RUNTIME_BUDGET_CACHE_KEY, None)
    st.session_state.pop(_RUNTIME_MAP_CACHE_KEY, None)


def _get_runtime_budget_cache() -> OrderedDict[Tuple[Any, ...], Optional[UncertaintyBudget]]:
    """Return the mutable session-scoped runtime-budget cache."""
    cache = st.session_state.get(_RUNTIME_BUDGET_CACHE_KEY)
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict(cache or {})
        st.session_state[_RUNTIME_BUDGET_CACHE_KEY] = cache
    return cache


def _get_runtime_map_cache() -> "OrderedDict[str, RuntimeUncertaintyMap]":
    """Return the mutable session-scoped completed-map cache."""
    cache = st.session_state.get(_RUNTIME_MAP_CACHE_KEY)
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict(cache or {})
        st.session_state[_RUNTIME_MAP_CACHE_KEY] = cache
    return cache


def _map_request_key(
    shared,
    ratio_list,
    per_sample,
):
    """Key one whole-map request by the same inputs its per-budget keys use.

    Assembled in linear time rather than from the per-entry keys themselves:
    every one of those embeds the whole-session token, so digesting all of them
    would be quadratic in the session size - the cost this cache exists to
    avoid. The information content is the same, so a map hit is exactly as
    sensitive to an edited input as the per-budget cache is: any change to a
    mask, a window, a setting, a certificate or a referenced blank moves a
    component and therefore moves this key.
    """
    return (shared, ratio_list, per_sample)


def _session_samples_token(samples: Iterable[Sample]) -> Tuple[Any, ...]:
    """Return the shared cache token for all samples in one runtime pass."""
    return tuple(_sample_cache_token(item) for item in samples)


def _make_runtime_budget_cache_key(
    sample: Sample,
    ratio_name: str,
    *,
    element_config,
    processing_config: Optional[ProcessingConfig],
    uncertainty_config: Optional[UncertaintyConfig],
    all_samples: Optional[Iterable[Sample]],
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]],
    filter_method: Optional[str],
    filter_threshold: Optional[float],
    drift_fit_info,
    custom_contributor_library: Optional[Any] = None,
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
    session_token: Optional[Tuple[Any, ...]] = None,
    reference_index: Optional[Tuple[Dict[str, Sample], Dict[str, list]]] = None,
) -> Tuple[Any, ...]:
    """Build a cache key for one sample/ratio runtime budget."""
    session_samples = tuple(all_samples) if all_samples is not None else ()
    resolved_session_token = (
        session_token
        if session_token is not None
        else _session_samples_token(session_samples)
    )
    return (
        _RUNTIME_BUDGET_ALGORITHM_TOKEN,
        _sample_cache_token(sample),
        str(ratio_name),
        resolved_session_token,
        _reference_dependency_token(sample, session_samples, reference_index),
        digest(scientific_configuration(element_config)),
        _certificate_contents_token(element_config, processing_config),
        _processing_config_token(processing_config),
        _uncertainty_config_token(uncertainty_config),
        _freeze_for_cache(cycle_ranges or {}),
        filter_method,
        filter_threshold,
        _freeze_for_cache(drift_fit_info or {}),
        _custom_contributor_token(custom_contributor_library),
        _profile_defaults_token(profile_defaults),
    )


@timed("Runtime budget request (includes key lookup)")
def get_cached_runtime_budget(
    sample: Sample,
    ratio_name: str,
    *,
    element_config,
    processing_config: ProcessingConfig,
    uncertainty_config: Optional[UncertaintyConfig] = None,
    all_samples: Optional[Iterable[Sample]] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    filter_method: Optional[str] = None,
    filter_threshold: Optional[float] = None,
    drift_fit_info=None,
    custom_contributor_library: Optional[Any] = None,
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
    _session_token: Optional[Tuple[Any, ...]] = None,
    _reference_index: Optional[Tuple[Dict[str, Sample], Dict[str, list]]] = None,
) -> Optional[UncertaintyBudget]:
    """Return a cached runtime budget for one sample/ratio combination."""
    # A023: materialize the session once. The parameter is declared
    # ``Iterable[Sample]``, and a one-shot iterator used to be drained by key
    # construction, leaving the engine an empty session — the cache then stored
    # that incomplete-session budget under a key describing the complete one.
    session_samples = None if all_samples is None else list(all_samples)
    cache = _get_runtime_budget_cache()
    cache_key = _make_runtime_budget_cache_key(
        sample,
        ratio_name,
        element_config=element_config,
        processing_config=processing_config,
        uncertainty_config=uncertainty_config,
        all_samples=session_samples,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
        drift_fit_info=drift_fit_info,
        custom_contributor_library=custom_contributor_library,
        profile_defaults=profile_defaults,
        session_token=_session_token,
        reference_index=_reference_index,
    )
    cache_event("Individual budgets", cache_key in cache)
    if cache_key in cache:
        cache.move_to_end(cache_key)
    else:
        cache[cache_key] = timed("Runtime budget computation")(compute_runtime_budget)(
            sample,
            ratio_name,
            element_config=element_config,
            processing_config=processing_config,
            uncertainty_config=uncertainty_config,
            all_samples=session_samples,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
            drift_fit_info=drift_fit_info,
            custom_contributor_library=custom_contributor_library,
            profile_defaults=profile_defaults,
        )
        # Cap growth using least-recently-used eviction.
        if len(cache) > _MAX_CACHE_SIZE:
            evict_count = len(cache) // 4
            for _ in range(evict_count):
                cache.popitem(last=False)
    # Budgets and their contributor rows are mutable dataclasses. Readers own
    # their returned value, never the object retained for subsequent renders.
    return deepcopy(cache[cache_key])


@timed("Runtime map request (includes budget requests)")
def build_cached_runtime_uncertainty_map(
    samples: Iterable[Sample],
    ratio_names: Iterable[str],
    *,
    element_config,
    processing_config: ProcessingConfig,
    uncertainty_config: Optional[UncertaintyConfig] = None,
    all_session_samples: Optional[Iterable[Sample]] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    filter_method: Optional[str] = None,
    filter_threshold: Optional[float] = None,
    drift_fit_info=None,
    custom_contributor_library: Optional[Any] = None,
    profile_defaults: Optional[Mapping[str, Mapping[str, bool]]] = None,
) -> RuntimeUncertaintyMap:
    """Build a runtime uncertainty map backed by the shared session cache."""
    samples_list = list(samples)
    # A072: consumed once for every sample, so a one-shot iterable must be
    # materialized before the loop or every sample after the first silently
    # receives no ratios at all.
    ratio_list = list(ratio_names)
    session_list = list(all_session_samples) if all_session_samples is not None else samples_list
    session_token = _session_samples_token(session_list)
    reference_index = _reference_index(session_list)
    session_tuple = tuple(session_list)
    budgets: RuntimeUncertaintyMap = {}

    map_cache = _get_runtime_map_cache()
    map_key = _map_request_key(
        (
            _RUNTIME_BUDGET_ALGORITHM_TOKEN,
            session_token,
            digest(scientific_configuration(element_config)),
            _certificate_contents_token(element_config, processing_config),
            _processing_config_token(processing_config),
            _uncertainty_config_token(uncertainty_config),
            _freeze_for_cache(cycle_ranges or {}),
            filter_method,
            filter_threshold,
            _freeze_for_cache(drift_fit_info or {}),
            _custom_contributor_token(custom_contributor_library),
            _profile_defaults_token(profile_defaults),
        ),
        tuple(str(name) for name in ratio_list),
        tuple(
            (
                _sample_cache_token(sample),
                _reference_dependency_token(sample, session_tuple, reference_index),
            )
            for sample in samples_list
        ),
    )
    cache_event("Complete budget maps", map_key in map_cache)
    cached_map = map_cache.get(map_key)
    if cached_map is not None:
        map_cache.move_to_end(map_key)
        # A copy, so a caller that annotates its map cannot silently rewrite
        # what the next render is handed.
        return deepcopy(cached_map)

    for sample in samples_list:
        for ratio_name in ratio_list:
            budget = get_cached_runtime_budget(
                sample,
                ratio_name,
                element_config=element_config,
                processing_config=processing_config,
                uncertainty_config=uncertainty_config,
                all_samples=session_list,
                cycle_ranges=cycle_ranges,
                filter_method=filter_method,
                filter_threshold=filter_threshold,
                drift_fit_info=drift_fit_info,
                custom_contributor_library=custom_contributor_library,
                profile_defaults=profile_defaults,
                _session_token=session_token,
                _reference_index=reference_index,
            )
            if budget is not None:
                budgets[make_runtime_uncertainty_key(sample, ratio_name)] = budget

    map_cache[map_key] = deepcopy(budgets)
    while len(map_cache) > _MAX_MAP_CACHE_SIZE:
        map_cache.popitem(last=False)

    return budgets

"""Typed application state for TraceISO."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Set, Tuple

import streamlit as st

from config.settings import ProcessingConfig, DisplayConfig, ExportConfig, UncertaintyConfig
from ui.navigation import (
    SESSION_KEY_PROCESSED_EXCLUSIONS,
    SESSION_KEY_PROCESSED_MANUAL_EXCLUSIONS,
    SESSION_KEY_PROCESSED_TYPES,
    SESSION_KEY_PROCESSED_PROCESSING_CONFIG,
    SESSION_KEY_THEME_NAME,
    SESSION_KEY_SUPPRESS_RELOAD,
)
from ui.runtime_budget_cache import clear_runtime_budget_cache

if TYPE_CHECKING:
    from domain.models import ProcessingResult, Sample
    from domain.elements.base import ElementConfig


_KEY_SAMPLES = "samples"
_KEY_ELEMENT_CONFIG = "element_config"
_KEY_ELEMENT_SYMBOL = "element_symbol"
_KEY_PROCESSING_RESULT = "processing_result"
_KEY_PROCESSING_CONFIG = "processing_config"
_KEY_DISPLAY_CONFIG = "display_config"
_KEY_EXPORT_CONFIG = "export_config"
_KEY_SELECTED_SAMPLE_IDX = "selected_sample_idx"
_KEY_LOADED_FILE = "loaded_file"
_KEY_DETECTED_ISOTOPES = "detected_isotopes"
_KEY_FILE_STRUCTURE = "file_structure"
_KEY_WARNINGS = "warnings"
_KEY_DEV_MODE = "dev_mode"
_KEY_FILE_HASH = "file_hash"
_KEY_FILE_SIZE = "file_size"
_KEY_LOADED_DATA_PREFERENCE = "loaded_data_preference"
_KEY_PLOT_CONFIG = "plot_config"
_KEY_GLOBAL_CYCLE_RANGE_ENABLED = "global_cycle_range_enabled"
_KEY_GLOBAL_CYCLE_RANGE = "global_cycle_range"
_KEY_SELECTED_RATIOS = "selected_ratios"
_KEY_ORIGINAL_SAMPLES = "_original_samples"
_KEY_UNCERTAINTY_CONFIG = "uncertainty_config"
_INIT_SENTINEL = "_traceiso_initialized"
_DATACLASS_MIGRATION_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Any]] = {}


def _migrate_dataclass_config(value: Any, cls: type, *, aliases: Optional[Dict[str, Tuple[str, ...]]] = None) -> Any:
    """Return *value* rebuilt with fields currently defined by dataclass *cls*."""
    alias_token = tuple(
        sorted((target, tuple(source_names)) for target, source_names in (aliases or {}).items())
    )
    cache_key = (id(value), cls, alias_token)
    cached = _DATACLASS_MIGRATION_CACHE.get(cache_key)
    if cached is not None and cached[0] is value:
        return cached[1]

    defaults = cls()
    if not isinstance(value, cls):
        _DATACLASS_MIGRATION_CACHE[cache_key] = (value, defaults)
        return defaults

    updates: Dict[str, Any] = {}
    changed = False
    aliases = aliases or {}

    for field in fields(cls):
        if hasattr(value, field.name):
            updates[field.name] = getattr(value, field.name)
        else:
            updates[field.name] = getattr(defaults, field.name)
            changed = True

    for target, source_names in aliases.items():
        if hasattr(value, target):
            continue
        for source_name in source_names:
            if hasattr(value, source_name):
                updates[target] = getattr(value, source_name)
                changed = True
                break

    migrated = cls(**updates) if changed else value
    _DATACLASS_MIGRATION_CACHE[cache_key] = (value, migrated)
    if len(_DATACLASS_MIGRATION_CACHE) > 64:
        _DATACLASS_MIGRATION_CACHE.pop(next(iter(_DATACLASS_MIGRATION_CACHE)))
    return migrated


def _default_plot_config() -> Dict[str, Any]:
    return {
        "height": 550,
        "marker_size": 9,
        "show_grid": True,
        "show_shaded_uncertainty": True,
        "download_format": "png",
    }


def _coerce_ratio_selection(value: Any) -> Optional[Set[str]]:
    """Normalize stored ratio-selection state."""
    if value is None:
        return None
    if isinstance(value, set):
        return set(value)
    if isinstance(value, str):
        return {value}
    try:
        return {str(item) for item in value}
    except TypeError:
        return {str(value)}


def _resolve_kappa_spec(kappa: str, specs: Dict[str, Any]):
    """Resolve a short kappa id or full key to a loader KappaSpec."""
    token = str(kappa or "").strip()
    if token in specs:
        return specs[token]
    short_map = {
        "k1": "k1_sample_decomposition",
        "k2": "k2_matrix_separation",
        "k3": "k3_procedural_blank",
        "k4": "k4_bracketing_standard_heterogeneity",
        "k6": "k6_matrix_effects",
        "k7": "k7_residual_interferences",
    }
    key = short_map.get(token)
    if key and key in specs:
        return specs[key]
    raise KeyError(f"Unknown SSB kappa default {kappa!r}.")


class AppState:
    """Typed accessor for Streamlit session state."""

    def _ensure_initialized(self) -> None:
        """Initialize all state keys with defaults if not present."""
        # item 88: fast-path — skip the entire defaults dict construction
        # when the session has already been initialized.
        if _INIT_SENTINEL in st.session_state:
            return

        had_processing_cfg = _KEY_PROCESSING_CONFIG in st.session_state
        had_global_cycle_flag = _KEY_GLOBAL_CYCLE_RANGE_ENABLED in st.session_state
        defaults = {
            _KEY_SAMPLES: None,
            _KEY_ELEMENT_CONFIG: None,
            _KEY_ELEMENT_SYMBOL: "",
            _KEY_PROCESSING_RESULT: None,
            _KEY_PROCESSING_CONFIG: ProcessingConfig(),
            _KEY_DISPLAY_CONFIG: DisplayConfig(),
            _KEY_EXPORT_CONFIG: ExportConfig(),
            _KEY_SELECTED_SAMPLE_IDX: 0,
            _KEY_LOADED_FILE: None,
            _KEY_DETECTED_ISOTOPES: set(),
            _KEY_FILE_STRUCTURE: "direct",
            _KEY_WARNINGS: [],
            _KEY_DEV_MODE: False,
            _KEY_FILE_HASH: None,
            _KEY_FILE_SIZE: None,
            _KEY_LOADED_DATA_PREFERENCE: None,
            _KEY_PLOT_CONFIG: _default_plot_config(),
            _KEY_GLOBAL_CYCLE_RANGE_ENABLED: False,
            _KEY_GLOBAL_CYCLE_RANGE: None,
            _KEY_SELECTED_RATIOS: None,
            _KEY_ORIGINAL_SAMPLES: None,
            _KEY_UNCERTAINTY_CONFIG: UncertaintyConfig(),
        }
        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

        cfg = st.session_state.get(_KEY_PROCESSING_CONFIG)
        if isinstance(cfg, ProcessingConfig):
            if not had_processing_cfg and had_global_cycle_flag:
                cfg.global_cycle_range = bool(
                    st.session_state.get(_KEY_GLOBAL_CYCLE_RANGE_ENABLED, False)
                )
            else:
                st.session_state[_KEY_GLOBAL_CYCLE_RANGE_ENABLED] = bool(
                    getattr(cfg, "global_cycle_range", False)
                )

        st.session_state[_INIT_SENTINEL] = True

    def __init__(self) -> None:
        self._ensure_initialized()


    @property
    def loaded_file(self) -> Optional[str]:
        """Name of the currently loaded file."""
        return st.session_state.get(_KEY_LOADED_FILE)

    @loaded_file.setter
    def loaded_file(self, value: Optional[str]) -> None:
        st.session_state[_KEY_LOADED_FILE] = value

    @property
    def file_hash(self) -> Optional[str]:
        """SHA-1 hash of the loaded file content."""
        return st.session_state.get(_KEY_FILE_HASH)

    @file_hash.setter
    def file_hash(self, value: Optional[str]) -> None:
        st.session_state[_KEY_FILE_HASH] = value

    @property
    def file_size(self) -> Optional[int]:
        """Byte size of the loaded file (used for fast change-detection)."""
        return st.session_state.get(_KEY_FILE_SIZE)

    @file_size.setter
    def file_size(self, value: Optional[int]) -> None:
        st.session_state[_KEY_FILE_SIZE] = value

    @property
    def loaded_data_preference(self) -> Optional[str]:
        """Data-preference mode used for the currently loaded file."""
        return st.session_state.get(_KEY_LOADED_DATA_PREFERENCE)

    @loaded_data_preference.setter
    def loaded_data_preference(self, value: Optional[str]) -> None:
        st.session_state[_KEY_LOADED_DATA_PREFERENCE] = value

    @property
    def plot_config(self) -> Dict[str, Any]:
        """Persistent plot appearance preferences."""
        return st.session_state.get(_KEY_PLOT_CONFIG, _default_plot_config())

    @plot_config.setter
    def plot_config(self, value: Dict[str, Any]) -> None:
        st.session_state[_KEY_PLOT_CONFIG] = value

    @property
    def global_cycle_range_enabled(self) -> bool:
        """Whether a single cycle window is applied to all samples."""
        cfg = st.session_state.get(_KEY_PROCESSING_CONFIG)
        if isinstance(cfg, ProcessingConfig):
            return bool(getattr(cfg, "global_cycle_range", False))
        return bool(st.session_state.get(_KEY_GLOBAL_CYCLE_RANGE_ENABLED, False))

    @global_cycle_range_enabled.setter
    def global_cycle_range_enabled(self, value: bool) -> None:
        normalized = bool(value)
        st.session_state[_KEY_GLOBAL_CYCLE_RANGE_ENABLED] = normalized
        cfg = st.session_state.get(_KEY_PROCESSING_CONFIG)
        if isinstance(cfg, ProcessingConfig):
            cfg.global_cycle_range = normalized

    @property
    def global_cycle_range(self) -> Optional[Tuple[int, int]]:
        """Global cycle range shared across samples."""
        value = st.session_state.get(_KEY_GLOBAL_CYCLE_RANGE)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return int(value[0]), int(value[1])
        return None

    @global_cycle_range.setter
    def global_cycle_range(self, value: Optional[Tuple[int, int]]) -> None:
        st.session_state[_KEY_GLOBAL_CYCLE_RANGE] = tuple(value) if value is not None else None

    @property
    def selected_ratios(self) -> Set[str]:
        """Current ratio selection."""
        value = _coerce_ratio_selection(st.session_state.get(_KEY_SELECTED_RATIOS, set()))
        return set() if value is None else value

    @selected_ratios.setter
    def selected_ratios(self, value: Set[str]) -> None:
        st.session_state[_KEY_SELECTED_RATIOS] = _coerce_ratio_selection(value)

    @property
    def has_selected_ratio_selection(self) -> bool:
        """Whether the ratio-selection key has been initialized by the UI."""
        return st.session_state.get(_KEY_SELECTED_RATIOS) is not None

    @property
    def original_samples(self) -> Optional[List["Sample"]]:
        """Original uploaded samples for reset actions."""
        return st.session_state.get(_KEY_ORIGINAL_SAMPLES)

    @original_samples.setter
    def original_samples(self, value: Optional[List["Sample"]]) -> None:
        st.session_state[_KEY_ORIGINAL_SAMPLES] = value

    @property
    def samples(self) -> Optional[List["Sample"]]:
        """Raw loaded samples (before processing)."""
        return st.session_state.get(_KEY_SAMPLES)

    @samples.setter
    def samples(self, value: Optional[List["Sample"]]) -> None:
        st.session_state[_KEY_SAMPLES] = value

    @property
    def element_config(self) -> Optional["ElementConfig"]:
        """Detected or selected element configuration."""
        return st.session_state.get(_KEY_ELEMENT_CONFIG)

    @element_config.setter
    def element_config(self, value: Optional["ElementConfig"]) -> None:
        st.session_state[_KEY_ELEMENT_CONFIG] = value
        if value:
            st.session_state[_KEY_ELEMENT_SYMBOL] = value.symbol

    @property
    def element_symbol(self) -> str:
        """Element symbol (e.g., 'Sr', 'Li')."""
        return st.session_state.get(_KEY_ELEMENT_SYMBOL, "")

    @property
    def detected_isotopes(self) -> set:
        """Set of isotope names found in the file."""
        return st.session_state.get(_KEY_DETECTED_ISOTOPES, set())

    @detected_isotopes.setter
    def detected_isotopes(self, value: set) -> None:
        st.session_state[_KEY_DETECTED_ISOTOPES] = value

    @property
    def file_structure(self) -> str:
        """HDF5 file structure: 'direct' or 'raw_corrected'."""
        return st.session_state.get(_KEY_FILE_STRUCTURE, "direct")

    @file_structure.setter
    def file_structure(self, value: str) -> None:
        st.session_state[_KEY_FILE_STRUCTURE] = value

    @property
    def has_data(self) -> bool:
        """True if samples are loaded."""
        return self.samples is not None and len(self.samples) > 0


    @property
    def result(self) -> Optional["ProcessingResult"]:
        """Processing result after running the pipeline."""
        return st.session_state.get(_KEY_PROCESSING_RESULT)

    @result.setter
    def result(self, value: Optional["ProcessingResult"]) -> None:
        st.session_state[_KEY_PROCESSING_RESULT] = value

    @property
    def has_result(self) -> bool:
        """True if processing has been run."""
        return self.result is not None

    def invalidate_processed_result(self) -> None:
        """Clear the committed processed result and its freshness snapshot."""
        st.session_state[_KEY_PROCESSING_RESULT] = None
        st.session_state.pop(SESSION_KEY_PROCESSED_EXCLUSIONS, None)
        st.session_state.pop(SESSION_KEY_PROCESSED_MANUAL_EXCLUSIONS, None)
        st.session_state.pop(SESSION_KEY_PROCESSED_TYPES, None)
        st.session_state.pop(SESSION_KEY_PROCESSED_PROCESSING_CONFIG, None)
        clear_runtime_budget_cache()

    @property
    def processing_config(self) -> ProcessingConfig:
        """Current processing settings."""
        cfg = st.session_state.get(_KEY_PROCESSING_CONFIG, ProcessingConfig())
        migrated = _migrate_dataclass_config(cfg, ProcessingConfig)
        if migrated is not cfg:
            cfg = migrated
            st.session_state[_KEY_PROCESSING_CONFIG] = cfg

        # item 92: removed per-access __post_init__() call; normalization
        # happens once at construction (in the migration branch above)
        # and in the setter below.
        return cfg

    @processing_config.setter
    def processing_config(self, value: ProcessingConfig) -> None:
        # item 92: normalize once when stored instead of on every getter access
        if isinstance(value, ProcessingConfig):
            value.__post_init__()
        st.session_state[_KEY_PROCESSING_CONFIG] = value
        st.session_state[_KEY_GLOBAL_CYCLE_RANGE_ENABLED] = bool(
            getattr(value, "global_cycle_range", False)
        )

    @property
    def warnings(self) -> List[str]:
        """Warnings from processing or loading."""
        return st.session_state.get(_KEY_WARNINGS, [])

    @warnings.setter
    def warnings(self, value: List[str]) -> None:
        st.session_state[_KEY_WARNINGS] = value


    @property
    def display_config(self) -> DisplayConfig:
        """Current display settings."""
        cfg = st.session_state.get(_KEY_DISPLAY_CONFIG, DisplayConfig())
        migrated = _migrate_dataclass_config(cfg, DisplayConfig)
        if migrated is not cfg:
            cfg = migrated
            st.session_state[_KEY_DISPLAY_CONFIG] = cfg

        return cfg

    @display_config.setter
    def display_config(self, value: DisplayConfig) -> None:
        st.session_state[_KEY_DISPLAY_CONFIG] = value

    @property
    def export_config(self) -> ExportConfig:
        """Current export settings."""
        cfg = st.session_state.get(_KEY_EXPORT_CONFIG, ExportConfig())
        migrated = _migrate_dataclass_config(cfg, ExportConfig)
        if migrated is not cfg:
            cfg = migrated
            st.session_state[_KEY_EXPORT_CONFIG] = cfg
        return cfg

    @export_config.setter
    def export_config(self, value: ExportConfig) -> None:
        st.session_state[_KEY_EXPORT_CONFIG] = value

    @property
    def uncertainty_config(self) -> UncertaintyConfig:
        """Current uncertainty configuration."""
        cfg = st.session_state.get(_KEY_UNCERTAINTY_CONFIG, UncertaintyConfig())
        migrated = _migrate_dataclass_config(
            cfg,
            UncertaintyConfig,
            aliases={
                "k2_matrix_separation_permil": ("k2_column_separation_permil",),
                "k4_bracketing_standard_heterogeneity_permil": ("k4_heterogeneity_permil",),
            },
        )
        if migrated is not cfg:
            cfg = migrated
            st.session_state[_KEY_UNCERTAINTY_CONFIG] = cfg
        return cfg

    @uncertainty_config.setter
    def uncertainty_config(self, value: UncertaintyConfig) -> None:
        st.session_state[_KEY_UNCERTAINTY_CONFIG] = value

    @property
    def custom_contributor_library(self) -> Optional[Dict]:
        """Custom uncertainty contributor library loaded from disk (lazy, cached per session)."""
        key = "_custom_contributor_library"
        error_key = "_custom_contributor_library_error"
        if key not in st.session_state:
            from config.custom_uncertainty_contributors_loader import load_custom_contributors
            try:
                st.session_state[key] = load_custom_contributors() or None
                st.session_state.pop(error_key, None)
            except Exception as exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Failed to load custom uncertainty contributors: %s",
                    exc,
                    exc_info=True,
                )
                st.session_state[key] = None
                st.session_state[error_key] = str(exc)
        return st.session_state.get(key)

    @property
    def custom_contributor_library_error(self) -> str:
        """Last custom-contributor library load error, if any."""
        _ = self.custom_contributor_library
        return str(st.session_state.get("_custom_contributor_library_error") or "")

    @property
    def global_uncertainty_values(self):
        """Global uncertainty defaults loaded from disk (lazy, cached per session)."""
        key = "_global_uncertainty_values"
        error_key = "_global_uncertainty_values_error"
        from config.global_uncertainty_values_loader import (
            GlobalUncertaintyValues,
            load_global_uncertainty_values,
        )

        cached = st.session_state.get(key)
        # Reload if uncached, or if a cached object predates a schema change /
        # module hot-reload (its class identity no longer matches the freshly
        # imported dataclass). This prevents stale-cache AttributeErrors when
        # the loader schema is edited while a session is live.
        stale = (
            cached is not None
            and not isinstance(cached, GlobalUncertaintyValues)
        )
        if key not in st.session_state or stale:
            try:
                st.session_state[key] = load_global_uncertainty_values()
                st.session_state.pop(error_key, None)
            except Exception as exc:
                # item 64: log traceback for support/debugging while keeping
                # the user-facing error string short and readable.
                import logging as _logging
                import traceback as _traceback
                _logging.getLogger(__name__).warning(
                    "Failed to load global uncertainty values: %s\n%s",
                    exc,
                    _traceback.format_exc(),
                )
                st.session_state[key] = None
                st.session_state[error_key] = str(exc)
        return st.session_state.get(key)

    @property
    def global_uncertainty_values_error(self) -> str:
        """Last global uncertainty defaults load error, if any."""
        self.global_uncertainty_values
        return str(st.session_state.get("_global_uncertainty_values_error") or "")

    def refresh_global_uncertainty_values(self) -> None:
        """Drop cached global uncertainty values and dependent runtime budgets."""
        st.session_state.pop("_global_uncertainty_values", None)
        st.session_state.pop("_global_uncertainty_values_error", None)
        st.session_state.pop("_custom_contributor_library", None)
        st.session_state.pop("_custom_contributor_library_error", None)
        clear_runtime_budget_cache()

    def resolve_kappa_default(self, kappa: str, element_symbol: str) -> float:
        """Return element-specific default u_rel_permil for an SSB kappa term."""
        from config.global_uncertainty_values_loader import KAPPA_SPECS

        spec = _resolve_kappa_spec(kappa, KAPPA_SPECS)
        values = self.global_uncertainty_values
        if values is not None:
            symbol = str(element_symbol or "").strip()
            symbol = symbol[:1].upper() + symbol[1:].lower() if symbol else ""
            element_values = values.elements.get(symbol)
            if element_values is not None:
                default = element_values.ssb_kappa_defaults.get(spec.key)
                if default is not None:
                    return float(default.u_rel_permil)
        return float(spec.default_value)

    def resolve_kappa_distribution(self, kappa: str, element_symbol: str) -> str:
        """Return element-specific default distribution for an SSB kappa term."""
        from config.global_uncertainty_values_loader import KAPPA_SPECS

        spec = _resolve_kappa_spec(kappa, KAPPA_SPECS)
        values = self.global_uncertainty_values
        if values is not None:
            symbol = str(element_symbol or "").strip()
            symbol = symbol[:1].upper() + symbol[1:].lower() if symbol else ""
            element_values = values.elements.get(symbol)
            if element_values is not None:
                default = element_values.ssb_kappa_defaults.get(spec.key)
                if default is not None:
                    return default.distribution
        return spec.default_distribution

    def resolve_kappa_drift_distribution(self, element_symbol: str) -> str:
        """Return the element-specific calculated-k5 MC distribution."""
        from config.global_uncertainty_values_loader import K5_KEY
        from config.settings import DEFAULT_KAPPA_DRIFT_DISTRIBUTION

        values = self.global_uncertainty_values
        if values is not None:
            symbol = str(element_symbol or "").strip()
            symbol = symbol[:1].upper() + symbol[1:].lower() if symbol else ""
            element_values = values.elements.get(symbol)
            if element_values is not None:
                default = element_values.ssb_kappa_defaults.get(K5_KEY)
                if default is not None:
                    return str(default.distribution or DEFAULT_KAPPA_DRIFT_DISTRIBUTION)
        return DEFAULT_KAPPA_DRIFT_DISTRIBUTION

    def global_contributor_enabled_defaults(self, element_symbol: str) -> Dict[str, bool]:
        """Return global default checkbox states for engine contributors/controls."""
        values = self.global_uncertainty_values
        if values is None:
            return {}
        from config.global_uncertainty_values_loader import contributor_enabled_defaults_for_element

        return contributor_enabled_defaults_for_element(element_symbol, values)

    def resolve_sr_engine_a_default(self, key: str, element_symbol: str) -> float:
        """Return an element-specific Sr Engine A global default value."""
        from config.global_uncertainty_values_loader import SR_ENGINE_A_DEFAULT_SPECS

        spec = SR_ENGINE_A_DEFAULT_SPECS[key]
        values = self.global_uncertainty_values
        if values is not None:
            symbol = str(element_symbol or "").strip()
            symbol = symbol[:1].upper() + symbol[1:].lower() if symbol else ""
            element_values = values.elements.get(symbol)
            if element_values is not None:
                default = element_values.sr_engine_a_defaults.get(spec.key)
                if default is not None:
                    return float(default.value)
        return float(spec.default_value)

    @property
    def uncertainty_profiles(self) -> dict:
        """Uncertainty contributor profile registry loaded lazily per session."""
        key = "_uncertainty_profiles"
        error_key = "_uncertainty_profiles_error"
        if key not in st.session_state:
            # item 91: merge builtins once during initial load, not every access
            from config.uncertainty_profiles_loader import (
                load_all_profiles,
                user_profiles_mtime,
            )
            from domain.uncertainty.contributors import BUILTIN_PROFILES
            try:
                profiles = load_all_profiles()
                st.session_state.pop(error_key, None)
            except Exception as exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Failed to load uncertainty profiles: %s",
                    exc,
                    exc_info=True,
                )
                profiles = dict(BUILTIN_PROFILES)
                st.session_state[error_key] = str(exc)
            profiles.update(BUILTIN_PROFILES)
            st.session_state[key] = profiles
            try:
                mtime = user_profiles_mtime()
            except OSError:
                mtime = -1.0
            st.session_state["_uncertainty_profiles_mtime"] = mtime
        return st.session_state[key]

    @property
    def uncertainty_profiles_error(self) -> str:
        """Last uncertainty-profile load error, if any."""
        _ = self.uncertainty_profiles
        return str(st.session_state.get("_uncertainty_profiles_error") or "")

    @property
    def uncertainty_profiles_mtime(self) -> float:
        """Mtime captured when user uncertainty profiles were loaded."""
        if "_uncertainty_profiles_mtime" not in st.session_state:
            _ = self.uncertainty_profiles
        return float(st.session_state.get("_uncertainty_profiles_mtime", -1.0))

    @property
    def uncertainty_profile_defaults(self) -> Dict[str, Dict[str, bool]]:
        """Resolver-ready profile defaults loaded lazily per session."""
        key = "_uncertainty_profile_defaults"
        if key not in st.session_state:
            # item 91: normalize once during initial load, not every access
            from config.uncertainty_profiles_loader import profile_defaults_for_resolver
            from domain.uncertainty.contributors import normalize_profile_defaults
            try:
                defaults = profile_defaults_for_resolver()
            except Exception as exc:
                import logging as _logging
                from domain.uncertainty.contributors import merge_profile_defaults
                _logging.getLogger(__name__).warning(
                    "Failed to load uncertainty profile defaults: %s",
                    exc,
                    exc_info=True,
                )
                defaults = merge_profile_defaults({})
                st.session_state["_uncertainty_profiles_error"] = str(exc)
            st.session_state[key] = normalize_profile_defaults(defaults)
        return st.session_state[key]

    def reload_uncertainty_profiles(self) -> None:
        """Drop cached uncertainty profiles and dependent runtime budgets."""
        st.session_state.pop("_uncertainty_profiles", None)
        st.session_state.pop("_uncertainty_profiles_mtime", None)
        st.session_state.pop("_uncertainty_profile_defaults", None)
        st.session_state.pop("_uncertainty_profiles_error", None)
        clear_runtime_budget_cache()

    @property
    def selected_sample_idx(self) -> int:
        """Index of currently selected sample in the inspector."""
        return st.session_state.get(_KEY_SELECTED_SAMPLE_IDX, 0)

    @selected_sample_idx.setter
    def selected_sample_idx(self, value: int) -> None:
        st.session_state[_KEY_SELECTED_SAMPLE_IDX] = value

    @property
    def selected_sample(self) -> Optional["Sample"]:
        """Currently selected sample object."""
        samples = self._active_samples()
        if samples and 0 <= self.selected_sample_idx < len(samples):
            return samples[self.selected_sample_idx]
        return None

    def _active_samples(self) -> List["Sample"]:
        """Return processed samples when available, otherwise loaded samples."""
        return self.result.samples if self.has_result else self.samples

    @property
    def dev_mode(self) -> bool:
        """Developer mode flag."""
        return st.session_state.get(_KEY_DEV_MODE, False)

    @dev_mode.setter
    def dev_mode(self, value: bool) -> None:
        st.session_state[_KEY_DEV_MODE] = value


    @property
    def standards(self) -> List["Sample"]:
        """List of standard samples."""
        samples = self._active_samples()
        if samples:
            return [s for s in samples if s.is_standard]
        return []

    @property
    def blanks(self) -> List["Sample"]:
        """List of blank samples."""
        samples = self._active_samples()
        if samples:
            return [s for s in samples if s.is_blank]
        return []

    @property
    def sample_measurements(self) -> List["Sample"]:
        """List of sample (SMP) measurements."""
        samples = self._active_samples()
        if samples:
            return [s for s in samples if s.is_sample]
        return []

    @property
    def sample_count_by_type(self) -> Dict[str, int]:
        """Count of samples by type."""
        samples = self._active_samples()
        if not samples:
            return {}
        counts: Dict[str, int] = {}
        for s in samples:
            t = s.sample_type.upper()
            counts[t] = counts.get(t, 0) + 1
        return counts


    def clear(self) -> None:
        """Reset all state to defaults, clearing ad-hoc widget keys."""
        # Keys to preserve across clear
        keep_keys = {
            _KEY_PROCESSING_CONFIG,
            _KEY_DISPLAY_CONFIG,
            _KEY_EXPORT_CONFIG,
            _KEY_DEV_MODE,
            _KEY_PLOT_CONFIG,
            _KEY_UNCERTAINTY_CONFIG,
            SESSION_KEY_THEME_NAME,
            SESSION_KEY_SUPPRESS_RELOAD,
        }
        
        # Collect keys to remove to avoid runtime definition change during iteration
        keys_to_remove = [k for k in st.session_state.keys() if k not in keep_keys]
        
        for k in keys_to_remove:
            del st.session_state[k]

        # item 88: sentinel was cleared above — _ensure_initialized will
        # rebuild defaults on the next call.
        self._ensure_initialized()
        clear_runtime_budget_cache()

    def select_sample(self, idx: int) -> None:
        """Select a sample by index (bounds-checked)."""
        samples = self.result.samples if self.has_result else self.samples
        if samples and 0 <= idx < len(samples):
            self.selected_sample_idx = idx

    def select_sample_by_name(self, name: str) -> bool:
        """Select a sample by name. Returns True if found."""
        samples = self.result.samples if self.has_result else self.samples
        if samples:
            for i, s in enumerate(samples):
                if s.name == name:
                    self.selected_sample_idx = i
                    return True
        return False


def get_state() -> AppState:
    """Get the typed state accessor."""
    return AppState()

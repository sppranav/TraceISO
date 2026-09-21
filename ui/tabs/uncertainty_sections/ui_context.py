"""UI context helpers for the Uncertainty tab redesign."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from config.settings import UncertaintyConfig
from domain.models import Sample
from domain.uncertainty.contributors import (
    ENGINE_INTERNAL_CONTRIBUTOR_NAMES,
    ENGINE_PB_TL_CONTRIBUTOR_NAMES,
    ENGINE_SSB_CONTRIBUTOR_NAMES,
)
from ui.tabs.uncertainty_sections import config_controls
from ui.tabs.uncertainty_sections import repeatability as repeatability_section


@dataclass(frozen=True)
class DiagnosticSection:
    """One selectable contributor diagnostic in the redesigned tab."""

    key: str
    label: str
    contributor_names: tuple[str, ...]


@dataclass(frozen=True)
class UncertaintyUiContext:
    """Resolved UI routing metadata for the active uncertainty view."""

    element_symbol: str
    engine_key: str
    ui_kind: str
    engine_label: str
    ratio_list: tuple[str, ...]
    default_ratio: str
    selected_ratio: str
    enabled_contributors: tuple[tuple[str, str], ...]
    diagnostic_sections: tuple[DiagnosticSection, ...]
    reset_token: tuple


def _state_element_symbol(state) -> str:
    element_symbol = getattr(state, "element_symbol", "") or ""
    if not element_symbol and getattr(state, "element_config", None) is not None:
        element_symbol = getattr(state.element_config, "symbol", "") or ""
    return str(element_symbol)


def _resolve_engine_key(
    state,
    u_config: UncertaintyConfig,
    *,
    selected_ratio: str,
    samples: Sequence[Sample],
) -> str:
    element_symbol = _state_element_symbol(state)
    processing_config = getattr(state, "processing_config", None)
    engine = u_config.resolve_engine(
        element_symbol,
        processing_config=processing_config,
    )

    return engine


def _ui_kind_for_engine(element_symbol: str, engine_key: str) -> str:
    if engine_key == "pb_tl_external_normalization":
        return "pb_tl"
    if engine_key == "internal_normalization":
        return "sr_internal"
    return "ssb_delta"


def _engine_label(ui_kind: str, u_config: UncertaintyConfig) -> str:
    if ui_kind == "pb_tl":
        return "Pb–Tl external normalization"
    if ui_kind == "sr_internal":
        return "Sr internal normalization"
    if bool(getattr(u_config, "enable_delta", False)):
        return "SSB/delta"
    if bool(getattr(u_config, "enable_ssb", False)):
        return "SSB"
    return "Absolute ratio"


def _built_in_names_for_ui_kind(ui_kind: str, state=None) -> list[str]:
    if ui_kind == "pb_tl":
        calibration = getattr(
            getattr(getattr(state, "processing_config", None), "pb_standard_calibration", None),
            "enabled",
            False,
        )
        if calibration:
            # These are the rows of the extended Engine C budget, including
            # explicit non-modelled coverage rows. They are presentation
            # metadata only; their states and values come from the budget.
            return [
                "u_prec",
                "u_pb_cal_std_precision",
                "u_pb_cal_reference",
                "u_norm_ref",
                "u_interf",
                "u_blank",
                "u_pb_cal_residual",
                "u_std_repeatability",
                "u_kappa_drift",
                "u_pb_cal_layout_mismatch",
                "u_pb_cal_sample_transfer",
                "u_bias_qc",
                "u_reprod_dig",
            ]
        return list(ENGINE_PB_TL_CONTRIBUTOR_NAMES)
    if ui_kind == "sr_internal":
        return list(ENGINE_INTERNAL_CONTRIBUTOR_NAMES)
    return list(ENGINE_SSB_CONTRIBUTOR_NAMES)


def _is_pb_tl_drift_enabled(u_config: UncertaintyConfig) -> bool:
    explicit = getattr(u_config, "contributor_enabled", {}) or {}
    if "u_kappa_drift" in explicit:
        return bool(explicit["u_kappa_drift"])
    return bool(getattr(u_config, "include_kappa_drift", False))


def built_in_contributors_for_context(
    context: UncertaintyUiContext,
    u_config: UncertaintyConfig,
    state,
) -> list[tuple[str, str]]:
    """Return matrix-ready built-in contributors for a resolved UI context."""
    element_symbol = context.element_symbol
    names = _built_in_names_for_ui_kind(context.ui_kind, state)

    def _show(name: str) -> bool:
        if context.ui_kind == "sr_internal" and name == "u_bias_ref":
            return False
        if context.ui_kind == "sr_internal" and name == "u_std_repeatability":
            return repeatability_section._is_standard_repeatability_enabled(
                u_config,
                state,
            )
        if context.ui_kind == "pb_tl" and name == "u_kappa_drift":
            return _is_pb_tl_drift_enabled(u_config)
        return u_config.is_contributor_enabled(name, element_symbol=element_symbol)

    return [
        (name, config_controls.contributor_display_label(name, ui_kind=context.ui_kind))
        for name in names
        if _show(name)
    ]


def _diagnostic_sections_for_context(
    context: UncertaintyUiContext,
    u_config: UncertaintyConfig,
) -> tuple[DiagnosticSection, ...]:
    enabled = {name for name, _label in context.enabled_contributors}
    sections: list[DiagnosticSection] = []

    if "u_prec" in enabled or "u_std" in enabled:
        sections.append(
            DiagnosticSection(
                key="precision",
                label="Measurement precision",
                contributor_names=tuple(
                    name for name in ("u_prec", "u_std") if name in enabled
                ),
            )
        )
    if "u_std_repeatability" in enabled:
        repeatability_label = {
            "sr_internal": "Reference-material repeatability",
            "pb_tl": "Reference-material repeatability",
            "ssb_delta": "Bracketing-standard repeatability",
        }.get(context.ui_kind, "Standard repeatability")
        sections.append(
            DiagnosticSection(
                key="repeatability",
                label=repeatability_label,
                contributor_names=("u_std_repeatability",),
            )
        )
    if "u_blank" in enabled:
        sections.append(
            DiagnosticSection(
                key="blank",
                label=(
                    "Measurement-blank correction"
                    if context.ui_kind == "ssb_delta"
                    else "Blank correction"
                ),
                contributor_names=("u_blank",),
            )
        )
    if "u_k5_instrumental_drift" in enabled:
        sections.append(
            DiagnosticSection(
                key="drift",
                label="Instrumental mass-bias drift",
                contributor_names=("u_k5_instrumental_drift",),
            )
        )
    if "u_crm" in enabled:
        sections.append(
            DiagnosticSection(
                key="crm_certified_value",
                label=(
                    "Certified/assigned reference ratio"
                    if context.ui_kind == "ssb_delta"
                    else "Certified/assigned reference value"
                ),
                contributor_names=("u_crm",),
            )
        )
    if "u_ref_value" in enabled:
        sections.append(
            DiagnosticSection(
                key="ref_value_certified",
                label="Reference value (literature)",
                contributor_names=("u_ref_value",),
            )
        )
    if context.ui_kind == "sr_internal":
        if "u_interf" in enabled:
            sections.append(
                DiagnosticSection(
                    key="sr_interference",
                    label="Isobaric interference",
                    contributor_names=("u_interf",),
                )
            )
        if "u_bias_qc" in enabled:
            sections.append(
                DiagnosticSection(
                    key="sr_bias",
                    label="Bias of processed QC material",
                    contributor_names=("u_bias_qc",),
                )
            )
    elif context.ui_kind == "pb_tl":
        if "u_norm_ref" in enabled:
            sections.append(
                DiagnosticSection(
                    key="pb_tl_norm_ref",
                    label="Tl normalization ratio uncertainty",
                    contributor_names=("u_norm_ref",),
                )
            )
        if "u_interf" in enabled:
            sections.append(
                DiagnosticSection(
                    key="pb_tl_interference",
                    label="204Hg interference correction",
                    contributor_names=("u_interf",),
                )
            )

    return tuple(sections)


def resolve_ui_context(
    state,
    u_config: UncertaintyConfig,
    *,
    ratio_list: Sequence[str],
    selected_ratio: str,
    samples: Sequence[Sample],
) -> UncertaintyUiContext:
    """Resolve the UI context for the active element, ratio, and session."""
    element_symbol = _state_element_symbol(state)
    engine_key = _resolve_engine_key(
        state,
        u_config,
        selected_ratio=selected_ratio,
        samples=samples,
    )
    ui_kind = _ui_kind_for_engine(element_symbol, engine_key)
    default_ratio = selected_ratio
    if getattr(state, "element_config", None) is not None:
        primary = getattr(state.element_config, "primary_ratio", None)
        if primary in ratio_list:
            default_ratio = primary

    placeholder = UncertaintyUiContext(
        element_symbol=element_symbol,
        engine_key=engine_key,
        ui_kind=ui_kind,
        engine_label=_engine_label(ui_kind, u_config),
        ratio_list=tuple(ratio_list),
        default_ratio=default_ratio,
        selected_ratio=selected_ratio,
        enabled_contributors=(),
        diagnostic_sections=(),
        reset_token=(),
    )
    enabled = tuple(built_in_contributors_for_context(placeholder, u_config, state))
    context = UncertaintyUiContext(
        element_symbol=element_symbol,
        engine_key=engine_key,
        ui_kind=ui_kind,
        engine_label=_engine_label(ui_kind, u_config),
        ratio_list=tuple(ratio_list),
        default_ratio=default_ratio,
        selected_ratio=selected_ratio,
        enabled_contributors=enabled,
        diagnostic_sections=(),
        reset_token=_build_reset_token(state, ratio_list),
    )
    return UncertaintyUiContext(
        element_symbol=context.element_symbol,
        engine_key=context.engine_key,
        ui_kind=context.ui_kind,
        engine_label=context.engine_label,
        ratio_list=context.ratio_list,
        default_ratio=context.default_ratio,
        selected_ratio=context.selected_ratio,
        enabled_contributors=context.enabled_contributors,
        diagnostic_sections=_diagnostic_sections_for_context(context, u_config),
        reset_token=context.reset_token,
    )


def _build_reset_token(state, ratio_list: Sequence[str]) -> tuple:
    processing_config = getattr(state, "processing_config", None)
    correction_mode = None
    if processing_config is not None:
        correction_mode = (
            bool(getattr(processing_config, "apply_mass_bias_correction", False)),
            bool(getattr(processing_config, "enable_ssb", False)),
            bool(getattr(processing_config, "enable_delta", False)),
            str(getattr(processing_config, "ssb_mode", "")),
        )
    return (
        getattr(state, "loaded_file", None),
        _state_element_symbol(state),
        correction_mode,
        tuple(ratio_list),
    )

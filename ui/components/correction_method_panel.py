"""Shared correction-method UI for element-specific normalization workflows.

This module provides a descriptor-driven renderer for the correction-method
section of the Session Configuration panel. It replaces the large per-element
branches in settings_panel.py with a single renderer + element descriptors.

NOTE: This replaces has_interference ONLY as the settings-panel UI branching
signal. has_interference survives in pipeline.py, inspector.py, sidebar.py,
and ratio_plot.py as the pipeline/display routing flag. Do not treat
has_interference as dead code after this UI refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

from config.settings import ProcessingConfig
from domain.elements.base import ElementConfig


@dataclass(frozen=True)
class MethodChoice:
    key: str
    label: str
    help_text: str = ""


@dataclass(frozen=True)
class MonitorToggle:
    setting_name: str
    label: str
    required_isotopes: tuple[str, ...]
    missing_message: str
    help_text: str


@dataclass(frozen=True)
class CorrectionMethodDescriptor:
    element_symbol: str
    section_caption: str
    method_label: str
    methods: tuple[MethodChoice, ...]
    default_internal_method_key: str
    widget_key_suffix: str                       # Streamlit radio key suffix — preserves session state
    default_ssb_method_key: Optional[str] = None
    default_none_method_key: Optional[str] = None
    internal_help_text: str = ""
    normalization_value_placeholder: str = ""
    normalization_value_help: str = ""
    use_ssb_delta_checkbox: bool = False
    monitor_toggles: tuple[MonitorToggle, ...] = ()
    allow_sr_anchoring: bool = False
    allow_interference_toggle: bool = False


@dataclass(frozen=True)
class CorrectionMethodUiResult:
    enable_ssb: bool
    enable_delta: bool
    apply_interference_correction: bool
    apply_mass_bias_correction: bool
    subtract_kr_blank: bool
    apply_hg_interference_correction: bool
    normalization_ratio_override: Optional[str]
    normalization_value_override: Optional[float]
    sr_session_anchoring: bool
    sr_calibration_standard_ids: tuple[str, ...] = ()


SR_CORRECTION_DESCRIPTOR = CorrectionMethodDescriptor(
    element_symbol="Sr",
    section_caption="**Sr Correction Method**",
    method_label="Sr Correction Method",
    methods=(
        MethodChoice(
            key="internal",
            label="Internal Normalization",
            help_text="Russell's exponential law using the selected Sr or Zr normalization ratio.",
        ),
        MethodChoice(
            key="none",
            label="None (No Correction)",
            help_text="No mass-bias correction applied.",
        ),
    ),
    default_internal_method_key="internal",
    default_none_method_key="none",
    widget_key_suffix="iif_method",              # preserves existing f"{key_prefix}_iif_method"
    internal_help_text="Select the isotope ratio used to estimate the Russell-law mass-bias factor.",
    normalization_value_placeholder="e.g. 0.1194",
    normalization_value_help="Accepted value of the selected normalization ratio used in Russell's exponential law.",
    allow_interference_toggle=True,
    allow_sr_anchoring=True,
)


PB_CORRECTION_DESCRIPTOR = CorrectionMethodDescriptor(
    element_symbol="Pb",
    section_caption="**Correction Method**",
    method_label="Lead Correction Method",
    methods=(
        MethodChoice(
            key="ssb",
            label="Standard-sample bracketing (SSB)",
            help_text="Correct Pb ratios against bracketing standards.",
        ),
        MethodChoice(
            key="internal",
            label="External Normalization (Pb–Tl)",
            help_text="Use admixed Tl isotope data and Russell's law to correct Pb mass bias.",
        ),
        MethodChoice(
            key="none",
            label="None",
            help_text="Apply neither SSB nor Tl normalization.",
        ),
    ),
    default_internal_method_key="internal",
    default_ssb_method_key="ssb",
    default_none_method_key="none",
    widget_key_suffix="pb_method",               # preserves existing f"{key_prefix}_pb_method"
    internal_help_text="Select the Tl ratio used for Pb–Tl external normalization with Russell's law.",
    normalization_value_placeholder="e.g. 2.38714",
    normalization_value_help="Accepted value of the selected Tl normalization ratio used in Russell's law for Pb.",
    use_ssb_delta_checkbox=True,
    monitor_toggles=(
        MonitorToggle(
            setting_name="apply_hg_interference_correction",
            label="Hg Interference Correction (202Hg \u2192 204Pb)",   # Unicode → (U+2192)
            required_isotopes=("202Hg",),
            missing_message=(
                "202Hg not present in the loaded data. "
                "Hg interference correction for 204Pb is disabled."
            ),
            help_text=(
                "Correct 204Pb for 204Hg isobaric interference using the 202Hg "
                "monitor channel. Required for metrologically defensible 206Pb/204Pb ratios. "
                "Disabled when 202Hg is absent from the data."
            ),
        ),
    ),
)


CORRECTION_METHOD_DESCRIPTORS: dict[str, CorrectionMethodDescriptor] = {
    "Sr": SR_CORRECTION_DESCRIPTOR,
    "Pb": PB_CORRECTION_DESCRIPTOR,
}


def get_correction_method_descriptor(
    element_config: ElementConfig,
) -> Optional[CorrectionMethodDescriptor]:
    """Return the UI descriptor for element-specific correction-method controls."""
    return CORRECTION_METHOD_DESCRIPTORS.get(element_config.symbol)


def resolve_initial_method_key(
    descriptor: CorrectionMethodDescriptor,
    current_config: ProcessingConfig,
) -> str:
    """Map the active processing config to a descriptor method key."""
    if current_config.apply_mass_bias_correction:
        return descriptor.default_internal_method_key
    if current_config.enable_ssb and descriptor.default_ssb_method_key is not None:
        return descriptor.default_ssb_method_key
    if descriptor.default_none_method_key is not None:
        return descriptor.default_none_method_key
    return descriptor.methods[0].key


def render_correction_method_panel(
    *,
    descriptor: CorrectionMethodDescriptor,
    element_config: ElementConfig,
    current_config: ProcessingConfig,
    key_prefix: str,
    detected_isotopes: set[str],
    supports_delta: bool,
    render_normalization_controls: Callable[[str], tuple[Optional[str], Optional[float]]],
    samples=(),
) -> CorrectionMethodUiResult:
    """
    Render a descriptor-driven correction-method panel.

    Parameters
    ----------
    descriptor:
    Element-specific UI descriptor.
    element_config:
    ElementConfig for the active element (used for delta support etc.).
    current_config:
    Current ProcessingConfig — read-only; never mutated.
    key_prefix:
    Streamlit widget key prefix shared with settings_panel.py.
    detected_isotopes:
    Normalized isotope labels present in the loaded session.
    supports_delta:
    Whether delta calculation is supported for this element.
    render_normalization_controls:
    Callable(help_text) that renders ratio/value controls and returns
    (normalization_ratio_override, normalization_value_override).
    """
    # Read current config values directly — both fields are formally declared
    # on ProcessingConfig (config/settings.py:121,124). No getattr needed.
    apply_interference = current_config.apply_interference_correction
    sr_session_anchoring = current_config.sr_session_anchoring
    sr_calibration_standard_ids = list(current_config.sr_calibration_standard_ids)
    normalization_ratio_override = current_config.normalization_ratio_override
    normalization_value_override = current_config.normalization_value_override
    # Monitor toggles (e.g. Hg) only take effect in internal-normalization mode
    # (pipeline.py:130 gates _pb_tl_correction on apply_mass_bias_correction).
    # Default to False so a stale ON state from a previous internal-mode session
    # cannot leak into SSB/None mode and produce a misleading settings summary.
    apply_hg_interference = False

    if descriptor.allow_interference_toggle:
        st.caption("**Interference Corrections**")
        apply_interference = st.checkbox(
            "Interference Correction",
            value=current_config.apply_interference_correction,
            key=f"{key_prefix}_interference",
            help=(
                "Apply natural-abundance interference corrections before internal "
                "normalization where required by the active isotope system."
            ),
        )

    st.caption(descriptor.section_caption)
    method_labels = [method.label for method in descriptor.methods]
    method_by_label = {method.label: method for method in descriptor.methods}
    initial_key = resolve_initial_method_key(descriptor, current_config)
    initial_index = next(
        (
            index
            for index, method in enumerate(descriptor.methods)
            if method.key == initial_key
        ),
        0,
    )
    selected_label = st.radio(
        descriptor.method_label,
        options=method_labels,
        index=initial_index,
        key=f"{key_prefix}_{descriptor.widget_key_suffix}",
        help="\n\n".join(
            f"**{method.label}** \u2014 {method.help_text}"
            for method in descriptor.methods
            if method.help_text
        ),
    )
    selected_method = method_by_label[selected_label]

    enable_ssb = False
    enable_delta = False
    apply_mass_bias = False

    if selected_method.key == descriptor.default_internal_method_key:
        apply_mass_bias = True
        normalization_ratio_override, normalization_value_override = render_normalization_controls(
            descriptor.internal_help_text
        )
        if descriptor.allow_sr_anchoring:
            from ui.components.sr_calibration_panel import render_sr_calibration_controls
            sr_session_anchoring, sr_calibration_standard_ids = render_sr_calibration_controls(
                current_config, samples, key_prefix=key_prefix,
            )
        for toggle in descriptor.monitor_toggles:
            missing = [isotope for isotope in toggle.required_isotopes if isotope not in detected_isotopes]
            if missing:
                st.warning(toggle.missing_message)
                if toggle.setting_name == "apply_hg_interference_correction":
                    apply_hg_interference = False
            elif toggle.setting_name == "apply_hg_interference_correction":
                apply_hg_interference = st.checkbox(
                    toggle.label,
                    value=current_config.apply_hg_interference_correction,
                    key=f"{key_prefix}_pb_hg_interference",
                    help=toggle.help_text,
                )

    elif selected_method.key == descriptor.default_ssb_method_key:
        enable_ssb = True
        # Delta is available only for descriptor-controlled SSB methods on elements that support it.
        # Guard order matches existing settings_panel.py:652 (supports_delta checked first).
        if supports_delta and descriptor.use_ssb_delta_checkbox:
            enable_delta = st.checkbox(
                "Report delta values",
                value=current_config.enable_delta,
                key=f"{key_prefix}_delta",
                help=(
                    "Calculate delta values relative to the selected reference "
                    "material after standard-sample bracketing (SSB)."
                ),
            )
        for toggle in descriptor.monitor_toggles:
            missing = [isotope for isotope in toggle.required_isotopes if isotope not in detected_isotopes]
            if missing:
                st.warning(toggle.missing_message)
                if toggle.setting_name == "apply_hg_interference_correction":
                    apply_hg_interference = False
            elif toggle.setting_name == "apply_hg_interference_correction":
                apply_hg_interference = st.checkbox(
                    toggle.label,
                    value=current_config.apply_hg_interference_correction,
                    key=f"{key_prefix}_pb_hg_interference_ssb",
                    help=toggle.help_text,
                )

    else:
        # None mode: only reset anchoring.
        # Decision D-UI-1: apply_interference is NOT reset — a user who enables
        # interference correction but disables mass-bias correction triggers
        # the Sr Rb/Kr correction without the final IIF ratio multiplication
        # (pipeline.py:925 gates IIF on apply_mass_bias_correction). This is
        # scientifically meaningful and is the existing behavior.
        sr_session_anchoring = False

    return CorrectionMethodUiResult(
        enable_ssb=enable_ssb,
        enable_delta=enable_delta,
        apply_interference_correction=apply_interference,
        apply_mass_bias_correction=apply_mass_bias,
        subtract_kr_blank=False,
        apply_hg_interference_correction=apply_hg_interference,
        normalization_ratio_override=normalization_ratio_override,
        normalization_value_override=normalization_value_override,
        sr_session_anchoring=sr_session_anchoring,
        sr_calibration_standard_ids=tuple(sr_calibration_standard_ids),
    )

"""
Processing settings panel for TraceISO.

Element-driven settings that adapt based on detected ElementConfig.
"""

from typing import Optional

import streamlit as st

from config.filtering import (
    get_filter_method_display_name,
    get_filter_method_options,
    normalize_filter_method_name,
)
from config.settings import ProcessingConfig, DriftConfig
from config.reference_materials import (
    get_crm_names,
    get_crm_ratios,
    get_crm_records,
    get_internal_normalization,
    get_natural_ratio,
)
from domain.elements.base import ElementConfig
from domain.sr_normalization_support import (
    sr_normalization_support,
    supported_sr_normalization_ratio,
)
from domain.ratio_utils import (
    element_symbol_from_isotope,
    normalize_ratio_name,
    normalize_ratio_token,
)
from ui.state import get_state
from ui.formatting import format_reference_value
from ui.utils import get_active_internal_normalization_ratio


def _get_available_normalization_ratios(
    element_config: ElementConfig,
    detected_isotopes: set[str],
) -> list[str]:
    """Return supported normalization ratios for the element."""
    options: list[str] = []

    def _add_option(ratio_name: Optional[str]) -> None:
        normalized = normalize_ratio_name(ratio_name)
        if normalized is None or normalized in options:
            return
        options.append(normalized)

    default_ratio = normalize_ratio_name(element_config.normalization_ratio)
    if default_ratio is not None:
        _add_option(default_ratio)

    if element_config.symbol == "Sr":
        _add_option("91Zr/90Zr")
    if element_config.symbol == "Pb":
        _add_option("205Tl/203Tl")

    return options


def _get_missing_isotopes_for_ratio(
    ratio_name: Optional[str],
    detected_isotopes: set[str],
) -> list[str]:
    """Return missing isotope labels for a normalization ratio."""
    normalized = normalize_ratio_name(ratio_name)
    if normalized is None or normalized.count("/") != 1:
        return []
    parts = normalized.split("/")
    return [part for part in parts if part not in detected_isotopes]


_CUSTOM_RATIO_LABEL = "Custom\u2026"


def _render_sr_normalization_ratio(
    element_config: ElementConfig,
    previous_ratio: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Render the Sr normalization pair, which the supported scope fixes.

    A039/D1: the Sr correction chain, its Kragten sensitivities and its Monte
    Carlo replay are all derived for the pair named by this element's correction
    roles, so no other choice is offered.  A pair carried in from an older
    session is shown as refused rather than silently rewritten to the supported
    one — coercing it would report numbers the user never asked for under the
    label they chose.
    """
    supported = supported_sr_normalization_ratio(element_config)

    st.text_input(
        "Normalization Ratio",
        value=supported or "",
        disabled=True,
        key="sr_norm_ratio_fixed",
        help=(
            f"Sr internal normalization is derived for {supported}. The Rb/Kr "
            "interference corrections, the K-factor recursion, the Kragten "
            "sensitivities and the Monte Carlo replay all depend on that choice, "
            "so TraceISO supports no other Sr normalization pair."
        ),
    )

    is_supported, reason = sr_normalization_support(element_config, previous_ratio)
    if not is_supported:
        st.error(
            f"This session carries an unsupported Sr normalization pair "
            f"({previous_ratio}). {reason}"
        )
        # Keep the refused override so processing refuses explicitly instead of
        # quietly producing supported-pair results under the old label.
        return previous_ratio, previous_ratio

    return supported, None


def _render_normalization_ratio_selector(
    available_ratios: list[str],
    previous_ratio: Optional[str],
    element_default_ratio: Optional[str],
    key_prefix: str,
    help_text: str,
) -> tuple[Optional[str], Optional[str]]:
    """Render a normalization-ratio selector with free-text custom input.

    Returns ``(selected_ratio, normalization_ratio_override)`` where the
    override is ``None`` when the selection matches the element default.
    """
    options = list(available_ratios) + [_CUSTOM_RATIO_LABEL]

    if previous_ratio in available_ratios:
        initial_index = available_ratios.index(previous_ratio)
    elif previous_ratio is not None:
        initial_index = len(available_ratios)  # "Custom\u2026"
    else:
        initial_index = 0

    choice = st.selectbox(
        "Normalization Ratio",
        options=options,
        index=initial_index,
        key=f"{key_prefix}_norm_ratio",
        help=help_text,
    )

    if choice == _CUSTOM_RATIO_LABEL:
        custom_default = (
            previous_ratio
            if previous_ratio is not None and previous_ratio not in available_ratios
            else ""
        )
        custom_text = st.text_input(
            "Enter ratio (e.g. 84Sr/88Sr)",
            value=custom_default,
            key=f"{key_prefix}_custom_norm_ratio",
            help="Type any isotope ratio in numerator/denominator format.",
        ).strip()
        normalized = normalize_ratio_name(custom_text)
        if normalized and normalized.count("/") == 1:
            selected = normalized
        else:
            if custom_text:
                st.error("Invalid ratio format. Use the form: 84Sr/88Sr")
            selected = available_ratios[0] if available_ratios else element_default_ratio
    else:
        selected = choice

    if selected is None:
        return None, None
    override = None if selected == element_default_ratio else selected
    return selected, override


def _get_managed_normalization_value(
    ratio_name: Optional[str],
    element_config: ElementConfig,
) -> Optional[float]:
    """Resolve the managed default value for a normalization ratio."""
    normalized = normalize_ratio_name(ratio_name)
    element_default = normalize_ratio_name(element_config.normalization_ratio)
    if normalized is None:
        return None
    if normalized == element_default and element_config.normalization_value is not None:
        return element_config.normalization_value

    numerator, denominator = normalized.split("/", 1)
    num_element = element_symbol_from_isotope(numerator)
    den_element = element_symbol_from_isotope(denominator)
    if num_element is None or den_element is None or num_element != den_element:
        return None

    internal_norm = get_internal_normalization(num_element)
    if internal_norm is not None:
        managed_ratio_name, managed_value = internal_norm
        if normalize_ratio_name(managed_ratio_name) == normalized:
            return managed_value

    natural_ratio = get_natural_ratio(num_element, normalized)
    if natural_ratio is not None:
        return natural_ratio[0]

    return None


def render_processing_settings(
    element_config: Optional[ElementConfig],
    current_config: ProcessingConfig,
    key_prefix: str = "proc",
    show_heading: bool = True,
) -> ProcessingConfig:
    """Render processing settings panel adapted to the element.

    Parameters
    ----------
    element_config:
        The detected/selected element configuration.
    current_config:
        Current processing configuration.
    key_prefix:
        Prefix for widget keys to avoid conflicts.

    Returns
    -------
    Updated ProcessingConfig based on user selections.
    """
    if show_heading:
        st.subheader("Data Reduction Parameters")

    if element_config is None:
        st.info("Upload a file to configure processing settings.")
        return current_config

    from ui.components.workspace_ui import render_group_heading

    # Full-width groups in reading order. The group containers are created
    # first and filled below, so each widget keeps its key and its position in
    # the code while landing in its group.
    state = get_state()
    # Kept as saved unless the Pb-Tl calibration controls below replace them.
    pb_calibration = current_config.pb_standard_calibration
    ssb_mode = current_config.ssb_mode
    data_group = st.container(key=f"{key_prefix}_group_data")
    corrections_group = st.container(key=f"{key_prefix}_group_corrections")
    reference_group = st.container(key=f"{key_prefix}_group_reference")
    with data_group:
        render_group_heading("Data & cycles")
    with corrections_group:
        render_group_heading("Corrections")
    with reference_group:
        render_group_heading("Reference material")

    with corrections_group:
        blank_mode = st.selectbox(
            "Blank Correction",
            options=["none", "before", "before_and_after"],
            index=["none", "before", "before_and_after"].index(current_config.blank_mode),
            format_func={
                "none": "No blank correction",
                "before": "Preceding blank",
                "before_and_after": "Mean of adjacent blanks",
            }.get,
            key=f"{key_prefix}_blank_mode",
            help=(
                "Choose the procedural blank used for subtraction. The adjacent-blank "
                "option averages the nearest preceding and following blanks when both "
                "are available."
            ),
        )

    with data_group:
        filter_options = get_filter_method_options(current_config.filter_method)
        current_filter_method = normalize_filter_method_name(current_config.filter_method)
        if current_filter_method not in filter_options:
            current_filter_method = "Standard deviation"

        filter_method = st.selectbox(
            "Outlier Rejection Method",
            options=filter_options,
            index=filter_options.index(current_filter_method),
            format_func=get_filter_method_display_name,
            key=f"{key_prefix}_filter",
            help=(
                "Optional cycle rejection before averaging. Routine workflows "
                "use no rejection, a standard-deviation cutoff, robust MAD, or IQR."
            ),
        )

        filter_threshold = current_config.filter_threshold
        if filter_method != "None":
            if filter_method == "MAD":
                threshold_help = (
                    "Median absolute deviation multiplier used to reject "
                    "spike-like cycle anomalies around the median."
                )
            elif filter_method == "IQR":
                threshold_help = (
                    "Tukey-style IQR fence multiplier. TraceISO's default is 2.0×IQR; "
                    "the classic mild-outlier Tukey fence is 1.5×IQR."
                )
            else:
                threshold_help = (
                    "Symmetric cycle rejection threshold in standard deviation units; "
                    "for example, 2 means two standard deviations from the mean."
                )
            filter_threshold = st.number_input(
                "Outlier Rejection Threshold",
                min_value=0.1,
                max_value=10.0,
                value=current_config.filter_threshold,
                step=0.5,
                key=f"{key_prefix}_threshold",
                help=threshold_help,
            )

        data_pref = st.selectbox(
            "Data Preference",
            options=["auto", "raw", "corrected"],
            index=["auto", "raw", "corrected"].index(current_config.data_preference),
            format_func={
                "auto": "Automatic",
                "raw": "Raw instrument data",
                "corrected": "Instrument-corrected data",
            }.get,
            key=f"{key_prefix}_data_pref",
            help=(
                "Choose the HDF5 data stream used for ratios and intensities. "
                "Automatic uses instrument-corrected data when available and otherwise "
                "uses the raw stream."
            ),
        )

        global_cycle = st.checkbox(
            "Use Global Cycle Range",
            value=current_config.global_cycle_range,
            key=f"{key_prefix}_global_cycle",
            help="Use one shared acquisition window for all samples when the analytical plateau occurs over the same cycle interval across the session.",
        )
        state.global_cycle_range_enabled = global_cycle

    with reference_group:
        # Reference Material dropdown
        crm_names = get_crm_names(element_config.symbol)
        if crm_names:
            current_rm = element_config.reference_material or crm_names[0]
            if element_config.symbol == "Pb":
                # Preserve the active K reference when consolidating older sessions
                # that carried two different Pb reference selections.
                current_rm = (
                    current_config.pb_standard_calibration.reference_material
                    if current_config.apply_mass_bias_correction
                    and current_config.pb_standard_calibration.enabled
                    and current_config.pb_standard_calibration.reference_material
                    else current_config.reference_material or current_rm
                )
            try:
                rm_idx = crm_names.index(current_rm)
            except ValueError:
                rm_idx = 0

            selected_rm = st.selectbox(
                (
                    "Pb reference material"
                    if element_config.symbol == "Pb"
                    else "Bracketing reference material"
                    if current_config.enable_ssb
                    else "Reference material"
                ),
                options=crm_names,
                index=rm_idx,
                key=(f"{key_prefix}_pb_reference_material" if element_config.symbol == "Pb"
                     else f"{key_prefix}_ref_material"),
                help=(
                    "Shared Pb reference for certified-value displays and Pb-standard calibration "
                    "(session K or local bracketing). The accepted Tl normalization ratio is set separately."
                    if element_config.symbol == "Pb" else
                    "Certified or assigned reference material used to anchor isotope-ratio "
                    "correction or standard-sample bracketing and supply its Type B uncertainty."
                ),
            )
            st.caption(
                "Reference-material values and uncertainty provenance are shown below."
            )

            # Show certified values for selected RM
            crm_ratios = get_crm_ratios(element_config.symbol, selected_rm)
            direct_records = {
                record.ratio_name: record
                for record in get_crm_records(element_config.symbol, selected_rm)
            }
            for ratio_name, record in direct_records.items():
                crm_ratios.setdefault(
                    ratio_name,
                    (record.ratio, record.uncertainty, record.k),
                )
            if crm_ratios:
                with st.expander("Certified/assigned values", expanded=False):
                    for ratio_name, ratio_payload in crm_ratios.items():
                        # Backward/forward compatible with CRM tuple shapes:
                        # (value, unc) or (value, unc, k)
                        try:
                            value = ratio_payload[0]
                            unc = ratio_payload[1]
                            coverage_factor = ratio_payload[2] if len(ratio_payload) >= 3 else 1.0
                        except (IndexError, TypeError):
                            continue
                        record = direct_records.get(ratio_name)
                        semantics = (
                            record.uncertainty_semantics
                            if record is not None
                            else "standard_uncertainty"
                        )
                        st.caption(
                            f"{ratio_name}: "
                            f"{format_reference_value(value, unc, coverage_factor, semantics)}"
                        )
        else:
            selected_rm = element_config.reference_material

    with corrections_group:
        # Element-specific options based on capabilities
        enable_ssb = current_config.enable_ssb
        enable_delta = current_config.enable_delta
        apply_interference = current_config.apply_interference_correction
        apply_mass_bias = current_config.apply_mass_bias_correction
        interference_monitor_overrides = dict(current_config.interference_monitors_enabled)
        # Hg is a staged request across Pb method switches.  A disabled/missing
        # monitor control must not erase the operator's saved request; the
        # processing record distinguishes requested from applied state.
        apply_hg_interference = current_config.apply_hg_interference_correction
        normalization_ratio_override = current_config.normalization_ratio_override
        normalization_value_override = current_config.normalization_value_override
        detected_isotopes = {
            normalize_ratio_token(isotope)
            for isotope in (state.detected_isotopes or set())
        }
        available_norm_ratios = _get_available_normalization_ratios(
            element_config,
            detected_isotopes,
        )
        element_default_ratio = normalize_ratio_name(element_config.normalization_ratio)
        previous_norm_ratio = normalize_ratio_name(
            current_config.normalization_ratio_override or element_config.normalization_ratio
        )
        sr_session_anchoring = getattr(current_config, "sr_session_anchoring", False)
        sr_calibration_standard_ids = list(current_config.sr_calibration_standard_ids)

        if element_config.has_interference:
            # Sr-specific settings
            st.caption("**Interference Corrections**")
            apply_interference = st.checkbox(
                "Interference Correction",
                value=current_config.apply_interference_correction,
                key=f"{key_prefix}_interference",
                help="Apply natural-abundance interference corrections for 87Rb on 87Sr and krypton on 84Sr/86Sr before internal normalisation.",
            )

            if element_config.monitors:
                st.caption("Select interferences to apply")
                monitor_states: dict[str, bool] = {}
                for spec in element_config.monitors:
                    enabled = st.checkbox(
                        f"{spec.interfering_isotope} -> {spec.corrected_isotope}",
                        value=current_config.is_monitor_configured_enabled(
                            spec.interfering_isotope
                        ),
                        key=f"{key_prefix}_interf_{spec.interfering_isotope}",
                        disabled=not apply_interference,
                        help=(
                            "Controlled by the Interference Correction master switch."
                        ),
                    )
                    monitor_states[spec.interfering_isotope] = enabled
                interference_monitor_overrides = {
                    interferent: enabled
                    for interferent, enabled in monitor_states.items()
                    if not enabled
                }

            st.caption("**Sr Correction Method**")
            sr_method_options = [
                "Internal Normalization",
                "None (No Correction)",
            ]
            sr_method_idx = 0 if current_config.apply_mass_bias_correction else 1

            sr_method = st.radio(
                "Sr Correction Method",
                options=sr_method_options,
                index=sr_method_idx,
                key=f"{key_prefix}_iif_method",
                help=(
                    "**Internal Normalization** — Russell's exponential law for instrumental "
                    "isotope fractionation using the selected normalization ratio.\n\n"
                    "**None** — No mass bias correction applied."
                ),
            )
            apply_mass_bias = sr_method == "Internal Normalization"
            enable_ssb = False
            sr_session_anchoring = False

            if apply_mass_bias:
                # Internal Normalization controls
                normalization_controls_active = apply_interference or apply_mass_bias
                selected_norm_ratio = previous_norm_ratio

                if normalization_controls_active and (available_norm_ratios or selected_norm_ratio is not None):
                    # A039/D1: Sr has exactly one supported pair, so it is shown
                    # fixed rather than offered as a free selection.
                    selected_norm_ratio, normalization_ratio_override = _render_sr_normalization_ratio(
                        element_config,
                        previous_norm_ratio,
                    )
                    missing_norm_isotopes = _get_missing_isotopes_for_ratio(
                        selected_norm_ratio,
                        detected_isotopes,
                    )
                    if missing_norm_isotopes:
                        st.warning(
                            "Selected normalization ratio requires isotope(s) not present in the loaded data: "
                            + ", ".join(missing_norm_isotopes)
                        )
                    managed_norm_value = _get_managed_normalization_value(
                        selected_norm_ratio,
                        element_config,
                    )
                    ratio_changed = selected_norm_ratio != previous_norm_ratio
                    effective_norm_value = (
                        current_config.normalization_value_override
                        if current_config.normalization_value_override is not None and not ratio_changed
                        else managed_norm_value
                    )

                    if effective_norm_value is None:
                        st.warning(
                            "Internal normalization value is not configured. "
                            "Enter a value below to process with the selected ratio."
                        )
                        norm_text = st.text_input(
                            f"Internal Normalization Value ({selected_norm_ratio})",
                            value="",
                            placeholder="e.g. 0.1194",
                            key=f"{key_prefix}_norm_value_text",
                            help="Accepted value of the selected normalization ratio used in Russell's exponential law.",
                        ).strip()
                        if norm_text:
                            try:
                                parsed_norm = float(norm_text)
                                if parsed_norm > 0:
                                    normalization_value_override = parsed_norm
                                else:
                                    normalization_value_override = None
                                    st.error("Normalization value must be greater than 0.")
                            except ValueError:
                                normalization_value_override = None
                                st.error("Normalization value must be numeric.")
                        else:
                            normalization_value_override = None
                    else:
                        entered_norm = st.number_input(
                            f"Internal Normalization Value ({selected_norm_ratio})",
                            min_value=0.000000000001,
                            value=float(effective_norm_value),
                            step=0.000001,
                            format="%.12f",
                            key=f"{key_prefix}_norm_value_num",
                            help="Accepted value of the selected normalization ratio used in Russell's exponential law.",
                        )
                        if managed_norm_value is not None and abs(entered_norm - managed_norm_value) < 1e-15:
                            normalization_value_override = None
                        else:
                            normalization_value_override = entered_norm

                    active_norm_value = (
                        normalization_value_override
                        if normalization_value_override is not None
                        else managed_norm_value
                    )
                    active_norm_text = (
                        f"{active_norm_value:.12g}"
                        if active_norm_value is not None
                        else "Not set"
                    )
                    st.caption(
                        f"Using Russell's law: {selected_norm_ratio} = "
                        f"{active_norm_text}"
                    )

                from ui.components.sr_calibration_panel import render_sr_calibration_controls
                sr_session_anchoring, sr_calibration_standard_ids = render_sr_calibration_controls(
                    current_config, getattr(state, "samples", None) or [], key_prefix=key_prefix,
                )

            else:
                # None selected
                apply_mass_bias = False
                enable_ssb = False
                enable_delta = False

        elif element_config.symbol == "Pb" and available_norm_ratios:
            st.caption("**Correction Method**")
            pb_method = st.radio(
                "Lead Correction Method",
                options=[
                    "Standard-sample bracketing (SSB)",
                    "External Normalization (Pb–Tl)",
                    "None",
                ],
                index=(
                    1 if current_config.apply_mass_bias_correction
                    else 0 if current_config.enable_ssb
                    else 2
                ),
                key=f"{key_prefix}_pb_method",
                help=(
                    "**SSB** — correct Pb ratios against bracketing standards.\n\n"
                    "**External Normalization (Pb–Tl)** — use admixed Tl isotope data and Russell's law to correct Pb mass bias.\n\n"
                    "**None** — apply neither SSB nor Tl normalization."
                ),
            )

            if pb_method == "External Normalization (Pb–Tl)":
                apply_mass_bias = True
                enable_ssb = False
                enable_delta = False

                selected_norm_ratio, normalization_ratio_override = _render_normalization_ratio_selector(
                    available_norm_ratios,
                    previous_norm_ratio,
                    element_default_ratio,
                    key_prefix,
                    "Select the Tl ratio used to estimate the Russell-law mass-bias factor for Pb.",
                )
                missing_norm_isotopes = _get_missing_isotopes_for_ratio(
                    selected_norm_ratio,
                    detected_isotopes,
                )
                if missing_norm_isotopes:
                    st.warning(
                        "Selected normalization ratio requires isotope(s) not present in the loaded data: "
                        + ", ".join(missing_norm_isotopes)
                    )
                managed_norm_value = _get_managed_normalization_value(
                    selected_norm_ratio,
                    element_config,
                )
                ratio_changed = selected_norm_ratio != previous_norm_ratio
                effective_norm_value = (
                    current_config.normalization_value_override
                    if current_config.normalization_value_override is not None and not ratio_changed
                    else managed_norm_value
                )

                if effective_norm_value is None:
                    st.warning(
                        "External-normalization value is not configured. "
                        "Enter a value below to process with Tl normalization."
                    )
                    norm_text = st.text_input(
                        f"External Normalization Value ({selected_norm_ratio})",
                        value="",
                        placeholder="e.g. 2.38714",
                        key=f"{key_prefix}_norm_value_text",
                        help="Accepted value of the selected Tl normalization ratio used in Russell's law for Pb.",
                    ).strip()
                    if norm_text:
                        try:
                            parsed_norm = float(norm_text)
                            if parsed_norm > 0:
                                normalization_value_override = parsed_norm
                            else:
                                normalization_value_override = None
                                st.error("Normalization value must be greater than 0.")
                        except ValueError:
                            normalization_value_override = None
                            st.error("Normalization value must be numeric.")
                    else:
                        normalization_value_override = None
                else:
                    entered_norm = st.number_input(
                        f"External Normalization Value ({selected_norm_ratio})",
                        min_value=0.000000000001,
                        value=float(effective_norm_value),
                        step=0.000001,
                        format="%.12f",
                        key=f"{key_prefix}_norm_value_num",
                        help="Accepted value of the selected Tl normalization ratio used in Russell's law for Pb.",
                    )
                    if managed_norm_value is not None and abs(entered_norm - managed_norm_value) < 1e-15:
                        normalization_value_override = None
                    else:
                        normalization_value_override = entered_norm

                active_norm_value = (
                    normalization_value_override
                    if normalization_value_override is not None
                    else managed_norm_value
                )
                active_norm_text = (
                    f"{active_norm_value:.12g}"
                    if active_norm_value is not None
                    else "Not set"
                )
                st.caption(
                    f"Using Russell's law: {selected_norm_ratio} = "
                    f"{active_norm_text}"
                )

                # Hg interference correction (requires 202Hg channel)
                if "202Hg" in detected_isotopes:
                    apply_hg_interference = st.checkbox(
                        "Hg Interference Correction (202Hg \u2192 204Pb)",
                        value=current_config.apply_hg_interference_correction,
                        key=f"{key_prefix}_pb_hg_interference",
                        help=(
                            "Subtract the 204Hg isobaric interference from 204Pb using the "
                            "202Hg monitor channel before applying Tl external normalization."
                        ),
                    )
                else:
                    st.caption(
                        "\u26a0 202Hg not detected \u2014 Hg correction for 204Pb is unavailable."
                    )

                from ui.components.pb_calibration_panel import render_pb_calibration_controls

                pb_calibration, ssb_mode = render_pb_calibration_controls(
                    current_config.pb_standard_calibration,
                    current_ssb_mode=current_config.ssb_mode,
                    samples=getattr(state, "samples", None) or [],
                    key_prefix=key_prefix,
                    reference_material=selected_rm,
                )

            elif pb_method == "Standard-sample bracketing (SSB)":
                apply_mass_bias = False
                enable_ssb = True
                apply_hg_interference = st.checkbox(
                    "Hg Interference Correction (202Hg \u2192 204Pb)",
                    value=current_config.apply_hg_interference_correction,
                    key=f"{key_prefix}_pb_hg_interference",
                    disabled="202Hg" not in detected_isotopes,
                    help=(
                        "Subtract 204Hg from 204Pb before SSB. When both Tl channels "
                        "are usable, their measured mass bias is used automatically; "
                        "when Tl is absent, the managed natural Hg ratio is used."
                    ),
                )
                if "202Hg" not in detected_isotopes:
                    st.caption(
                        "\u26a0 202Hg not detected \u2014 Hg correction is unavailable. "
                        "The saved request is retained until data with the monitor are loaded."
                    )
                elif {"203Tl", "205Tl"}.issubset(detected_isotopes):
                    st.caption(
                        "Tl-assisted Hg subtraction is used automatically for observations "
                        "with usable Tl; this remains the SSB route and does not select Engine C."
                    )
                    selected_norm_ratio, normalization_ratio_override = _render_normalization_ratio_selector(
                        available_norm_ratios,
                        previous_norm_ratio,
                        element_default_ratio,
                        key_prefix,
                        "Select the accepted Tl ratio used only to mass-bias-correct the Hg subtraction.",
                    )
                    managed_norm_value = _get_managed_normalization_value(
                        selected_norm_ratio, element_config
                    )
                    ratio_changed = selected_norm_ratio != previous_norm_ratio
                    effective_norm_value = (
                        current_config.normalization_value_override
                        if current_config.normalization_value_override is not None and not ratio_changed
                        else managed_norm_value
                    )
                    if effective_norm_value is None:
                        st.warning(
                            "The Tl reference value is unresolved; Tl-bearing observations "
                            "will make the requested Hg correction unavailable."
                        )
                    else:
                        entered_norm = st.number_input(
                            f"Tl Reference Value for Hg Correction ({selected_norm_ratio})",
                            min_value=0.000000000001,
                            value=float(effective_norm_value),
                            step=0.000001,
                            format="%.12f",
                            key=f"{key_prefix}_norm_value_num",
                            help=(
                                "Accepted Tl ratio used for the Hg mass-bias factor only; "
                                "Pb remains corrected by standard-sample bracketing."
                            ),
                        )
                        normalization_value_override = (
                            None
                            if managed_norm_value is not None
                            and abs(entered_norm - managed_norm_value) < 1e-15
                            else entered_norm
                        )
                elif not any(str(iso).endswith("Tl") for iso in detected_isotopes):
                    st.caption(
                        "Tl is absent, so Hg subtraction uses the managed natural 204Hg/202Hg ratio."
                    )
                else:
                    st.warning(
                        "Only part of the Tl monitor pair is present. This is a failed Tl-assisted "
                        "case, not a natural-ratio fallback; the corrected result will be unavailable."
                    )
                if element_config.supports_delta:
                    enable_delta = st.checkbox(
                        "Report delta values",
                        value=current_config.enable_delta,
                        key=f"{key_prefix}_delta",
                        help="Calculate δ values relative to the selected reference material after standard-sample bracketing (SSB).",
                    )
            else:
                apply_mass_bias = False
                enable_ssb = False
                enable_delta = False

        else:
            # SSB/Delta elements (Li, B, Mg, Cd, Pb without Tl)
            apply_mass_bias = False
            normalization_ratio_override = None
            normalization_value_override = None
            if element_config.supports_ssb:
                enable_ssb = st.checkbox(
                    "Standard-sample bracketing (SSB)",
                    value=current_config.enable_ssb,
                    key=f"{key_prefix}_ssb",
                    help="Apply standard-sample bracketing (SSB) so unknowns are corrected against surrounding reference standards measured through the same session.",
                )

            if element_config.supports_delta:
                enable_delta = st.checkbox(
                    "Report delta values",
                    value=current_config.enable_delta,
                    key=f"{key_prefix}_delta",
                    help=(
                        "Calculate δ values relative to the bracketing standard mean. "
                        "When SSB is also enabled, the SSB-corrected ratios are used as input."
                    ),
                )

    # Drift correction (all elements)
    with corrections_group:
        st.caption("**Drift Correction**")
        drift_locked = bool(
            element_config.symbol == "Pb" and apply_mass_bias and pb_calibration.enabled
        )
        enable_drift = st.checkbox(
            "Enable Drift Correction",
            value=current_config.drift.enabled,
            key=f"{key_prefix}_drift",
            help="Enable empirical session-drift correction fitted to standards in the Instrumental Drift tab before final reporting.",
            disabled=drift_locked,
        )
        if drift_locked:
            enable_drift = current_config.drift.enabled
            st.caption(
                "Separate drift correction is not applied while Pb-standard calibration is "
                "enabled. The saved drift setting is kept and reported as requested but not applied."
            )

    # Build updated drift config (preserve all fields, only update enabled flag)
    updated_drift = DriftConfig(
        enabled=enable_drift,
        method=current_config.drift.method,
        degree=current_config.drift.degree,
        x_axis=current_config.drift.x_axis,
        apply_outlier_filter=current_config.drift.apply_outlier_filter,
        outlier_threshold=current_config.drift.outlier_threshold,
        outlier_method=current_config.drift.outlier_method,
        ratio_name=current_config.drift.ratio_name,
        norm_mode=current_config.drift.norm_mode,
        norm_standard=current_config.drift.norm_standard,
        fit_info=current_config.drift.fit_info,
    )

    return ProcessingConfig(
        blank_mode=blank_mode,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
        enable_ssb=enable_ssb,
        enable_delta=enable_delta,
        enable_post_ssb_outliers=current_config.enable_post_ssb_outliers,
        post_ssb_outlier_threshold=current_config.post_ssb_outlier_threshold,
        apply_interference_correction=apply_interference,
        apply_mass_bias_correction=apply_mass_bias,
        apply_hg_interference_correction=apply_hg_interference,
        interference_monitors_enabled=interference_monitor_overrides,
        reference_material=selected_rm,
        normalization_ratio_override=normalization_ratio_override,
        normalization_value_override=normalization_value_override,
        sr_session_anchoring=sr_session_anchoring,
        sr_calibration_standard_ids=sr_calibration_standard_ids,
        include_certified_uncertainty=current_config.include_certified_uncertainty,
        data_preference=data_pref,
        global_cycle_range=global_cycle,
        drift=updated_drift,
        ssb_mode=ssb_mode,
        pb_standard_calibration=pb_calibration,
    )


def render_settings_summary(config: ProcessingConfig, element_config: Optional[ElementConfig]) -> None:
    """Render a compact summary of current settings."""
    if element_config is None:
        return

    st.caption("**Current Settings:**")

    parts = [f"Blank: {config.blank_mode}"]

    if config.filter_method != "None":
        parts.append(
            f"Outlier rejection: {config.filter_method} ({config.format_filter_parameter()})"
        )

    if element_config.has_interference:
        if config.apply_interference_correction:
            if element_config.monitors:
                active_interferents = [
                    spec.interfering_isotope
                    for spec in element_config.monitors
                    if config.is_monitor_enabled(spec.interfering_isotope)
                ]
                if active_interferents:
                    parts.append(f"Interference: {', '.join(active_interferents)}")
                else:
                    parts.append("Interference: none")
            else:
                parts.append("Interference: ON")
        if config.enable_ssb:
            parts.append("SSB: ON")
        elif config.apply_mass_bias_correction:
            active_ratio = get_active_internal_normalization_ratio(config, element_config)
            parts.append(
                f"Internal normalization: ON ({active_ratio})"
                if active_ratio
                else "Internal normalization: ON"
            )
            if config.sr_session_anchoring:
                parts.append("Sr-standard calibration: ON")
    else:
        active_ratio = get_active_internal_normalization_ratio(config, element_config)
        if active_ratio:
            normalization_label = (
                "External normalization"
                if element_config.symbol == "Pb"
                else "Internal normalization"
            )
            parts.append(f"{normalization_label}: ON ({active_ratio})")
        if getattr(config, "apply_hg_interference_correction", False):
            parts.append("Interference: Hg on 204Pb")
        if config.enable_ssb:
            parts.append("Standard-sample bracketing (SSB): ON")
        if config.enable_delta:
            parts.append("Delta: ON")

    st.caption(" | ".join(parts))

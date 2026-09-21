"""
Sample Inspector tab for TraceISO.

Provides detailed view of individual samples with intensity/ratio plots,
cycle controls, and correction summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, MutableMapping, Optional

import numpy as np
import streamlit as st

import html

from config.constants import DEFAULT_OUTLIER_THRESHOLD_SD
from domain.filters.outlier import get_filtered_values, get_runtime_mask
from domain.corrections.blank import KR_ISOTOPES
from domain.layers import cycle_data_equal
from domain.models import Sample
from domain.ratio_selection import (
    INTERFERENCE_LAYER_LABEL,
    get_processing_ratio_data,
    governed_interference_ratio_data,
)
from domain.runtime_delta import RuntimeDeltaResult, compute_runtime_delta
from config.settings import DisplayConfig, is_russell_law_normalization_engine
from ui.edit_actions import commit_manual_exclusions
from ui.state import get_state
from ui.navigation import SESSION_KEY_LAST_EDITED_SAMPLE_NAME
from ui.components.workspace_ui import (
    render_compact_metadata,
    render_panel_heading,
    render_statistics_table,
)
from ui.components.intensity_plot import create_single_isotope_plot
from ui.components.ratio_plot import create_single_ratio_plot, create_delta_cycle_plot
from ui.config_plotly import (
    PLOTLY_ANNOTATION_FONT_SIZE,
    PLOTLY_AXIS_TITLE_FONT_SIZE,
    PLOTLY_BASE_FONT_SIZE,
    PLOTLY_TITLE_FONT_SIZE,
    get_plotly_config,
)
from ui.runtime_budget_cache import get_cached_runtime_budget
from ui.utils import (
    clamp_cycle_range,
    _CYCLE_RANGES_STATE_KEY,
    format_isotope_label as format_ratio_label,
    get_active_internal_normalization_ratio,
    get_cycle_ranges as get_shared_cycle_ranges,
    render_runtime_budget_error,
    get_sample_state_key,
    current_pb_calibration_freshness,
)
from ui.formatting import (
    format_value_with_uncertainty,
    format_uncertainty,
    get_k_footnote,
)


_INSPECTOR_PLOT_TICK_FONT_SIZE = PLOTLY_BASE_FONT_SIZE + 3
_INSPECTOR_PLOT_AXIS_TITLE_FONT_SIZE = PLOTLY_AXIS_TITLE_FONT_SIZE + 4
_INSPECTOR_PLOT_TITLE_FONT_SIZE = PLOTLY_TITLE_FONT_SIZE + 3
_INSPECTOR_PLOT_LEGEND_FONT_SIZE = PLOTLY_BASE_FONT_SIZE + 1
_INSPECTOR_PLOT_ANNOTATION_FONT_SIZE = PLOTLY_ANNOTATION_FONT_SIZE + 3
_INSPECTOR_PLOT_EXPORT_WIDTH = 1200
_INSPECTOR_PLOT_EXPORT_HEIGHT = 700
_INSPECTOR_LEGEND_TOP_MARGIN = 210


def _get_cycle_ranges() -> dict:
    """Compatibility wrapper for shared cycle-range state."""
    return get_shared_cycle_ranges(get_state())


def _resolve_active_cycle_ranges(
    sample: Sample,
    cycle_range: Optional[tuple] = None,
    cycle_ranges: Optional[dict] = None,
) -> Optional[dict]:
    """Return the active range map for this sample and any linked standards."""
    if cycle_ranges:
        return cycle_ranges
    if cycle_range:
        sample_key = get_sample_state_key(sample)
        return {sample_key: cycle_range}
    return None


def _extract_runtime_valid_values(
    values: np.ndarray,
    mask: np.ndarray,
    sample_name: str,
    *,
    sample_key: Optional[str] = None,
    cycle_range: Optional[tuple] = None,
    cycle_ranges: Optional[dict] = None,
    filter_method: str = "None",
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD,
) -> np.ndarray:
    """Return valid values using the same range/filter path as runtime budgets."""
    active_cycle_ranges = cycle_ranges
    if active_cycle_ranges is None and cycle_range is not None:
        active_cycle_ranges = {sample_key or sample_name: cycle_range}

    return get_filtered_values(
        np.asarray(values, dtype=np.float64),
        np.asarray(mask, dtype=bool),
        sample_name,
        cycle_ranges=active_cycle_ranges,
        sample_key=sample_key,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
    )


def _blank_correction_active(sample: Sample) -> bool:
    """Return True when the current UI context says blank correction is active."""
    try:
        state = get_state()
        processing_config = getattr(state, "processing_config", None)
        if processing_config is not None:
            return getattr(processing_config, "blank_mode", "before") != "none"
    except Exception:
        pass
    return bool(sample.used_blanks) or bool(sample.blank_corrected_ratios)


def _select_visible_ratio_layer(
    sample: Sample,
    ratio_name: str,
    display_config: DisplayConfig,
    *,
    element_cfg=None,
):
    """Return the ratio layer currently represented by the Inspector plot/stats."""
    from domain.ratio_selection import (
        get_ssb_cycle_data,
        governed_pb_standard_ratio_data,
        pb_standard_layer_label,
    )

    pb_standard_cd = governed_pb_standard_ratio_data(sample, ratio_name)
    if display_config.show_pb_standard_corrected and pb_standard_cd is not None:
        return pb_standard_cd, pb_standard_layer_label(sample, ratio_name)

    if (
        display_config.show_drift_corrected
        and sample.drift_corrected_ratios
        and ratio_name in sample.drift_corrected_ratios
    ):
        return sample.drift_corrected_ratios[ratio_name], "Drift-corrected"

    from domain.ratio_selection import SR_STANDARD_LAYER_LABEL
    sr_standard_cd = sample.sr_standard_corrected_ratios.get(ratio_name)
    if display_config.show_sr_standard_corrected and sr_standard_cd is not None:
        return sr_standard_cd, SR_STANDARD_LAYER_LABEL

    if display_config.show_iif_corrected:
        if sample.iif_corrected_ratios and ratio_name in sample.iif_corrected_ratios:
            return sample.iif_corrected_ratios[ratio_name], _get_iif_like_plot_label(sample, ratio_name)
        if sample.ssb_results and ratio_name in sample.ssb_results:
            ssb_cd = get_ssb_cycle_data(sample, ratio_name)
            if ssb_cd is not None:
                return ssb_cd, "SSB-corrected"

    if (
        display_config.show_interference_corrected
        and sample.interference_corrected_ratios
        and ratio_name in sample.interference_corrected_ratios
    ):
        # Labelled inspection of the saved Hg interference-only ratio (both Pb
        # routes). Whether it is final is decided in the domain, not here.
        return sample.interference_corrected_ratios[ratio_name], INTERFERENCE_LAYER_LABEL

    if (
        display_config.show_interference_corrected
        and element_cfg is not None
        and element_cfg.has_interference
        and sample.corrected_ratios
        and ratio_name in sample.corrected_ratios
    ):
        return sample.corrected_ratios[ratio_name], "Interference-corrected"

    if display_config.show_corrected:
        if _blank_correction_active(sample) and sample.blank_corrected_ratios and ratio_name in sample.blank_corrected_ratios:
            blank_cd = sample.blank_corrected_ratios[ratio_name]
            raw_cd = sample.ratios.get(ratio_name) if sample.ratios else None
            if not (display_config.show_raw and cycle_data_equal(blank_cd, raw_cd)):
                return blank_cd, "Blank-corrected"
        if _blank_correction_active(sample) and sample.corrected_ratios and ratio_name in sample.corrected_ratios:
            label = "Interference-corrected" if (element_cfg and element_cfg.has_interference) else "Blank-corrected"
            corrected_cd = sample.corrected_ratios[ratio_name]
            raw_cd = sample.ratios.get(ratio_name) if sample.ratios else None
            if not (display_config.show_raw and cycle_data_equal(corrected_cd, raw_cd)):
                return corrected_cd, label

    if display_config.show_raw and sample.ratios and ratio_name in sample.ratios:
        return sample.ratios[ratio_name], "Raw"

    # Fallback: preserve final-layer priority if no visible layer is enabled.
    if sample.drift_corrected_ratios and ratio_name in sample.drift_corrected_ratios:
        return sample.drift_corrected_ratios[ratio_name], "Drift-corrected"
    if sr_standard_cd is not None:
        return sr_standard_cd, SR_STANDARD_LAYER_LABEL
    if sample.iif_corrected_ratios and ratio_name in sample.iif_corrected_ratios:
        return sample.iif_corrected_ratios[ratio_name], "IIF-corrected"
    if sample.ssb_results and ratio_name in sample.ssb_results:
        ssb_cd = get_ssb_cycle_data(sample, ratio_name)
        if ssb_cd is not None:
            return ssb_cd, "SSB-corrected"
    governed_hg_cd = governed_interference_ratio_data(sample, ratio_name)
    if governed_hg_cd is not None:
        return governed_hg_cd, INTERFERENCE_LAYER_LABEL
    if _blank_correction_active(sample) and sample.blank_corrected_ratios and ratio_name in sample.blank_corrected_ratios:
        blank_cd = sample.blank_corrected_ratios[ratio_name]
        raw_cd = sample.ratios.get(ratio_name) if sample.ratios else None
        if not (display_config.show_raw and cycle_data_equal(blank_cd, raw_cd)):
            return blank_cd, "Blank-corrected"
    if _blank_correction_active(sample) and sample.corrected_ratios and ratio_name in sample.corrected_ratios:
        label = "Interference-corrected" if (element_cfg and element_cfg.has_interference) else "Blank-corrected"
        corrected_cd = sample.corrected_ratios[ratio_name]
        raw_cd = sample.ratios.get(ratio_name) if sample.ratios else None
        if not (display_config.show_raw and cycle_data_equal(corrected_cd, raw_cd)):
            return corrected_cd, label
    if sample.ratios and ratio_name in sample.ratios:
        return sample.ratios[ratio_name], "Raw"
    return None, "Raw"


def _hg_correction_caption(sample: Sample, ratio_name: str) -> str:
    """Plain-language status of the Hg record for the selected ratio, or ''.

    Presentation only: every fact comes from the domain record.
    """
    from domain.pb_correction_records import hg_records

    record = hg_records(sample).get(ratio_name)
    if record is None:
        return ""
    if not record.governs_final:
        if record.status == "unavailable":
            return (
                f"Hg interference-only intermediate not saved ({record.reason_code}). "
                "The Tl-normalized result keeps its historical behaviour."
            )
        return (
            "Hg interference-corrected intermediate saved before Tl normalization "
            "(diagnostic only; it does not change the Tl-normalized result)."
        )
    if record.status != "applied":
        return (
            f"Hg interference correction unavailable for {ratio_name} "
            f"({record.reason_code}): {record.reason} No corrected value is reported."
        )
    text = f"Hg interference correction applied (source: {record.source.replace('_', ' ')})."
    if record.excluded_cycles:
        reasons = ", ".join(f"cycle {cycle}: {reason}" for cycle, reason in record.excluded_cycles)
        fraction = record.excluded_fraction or 0.0
        text += (
            f" {record.n_excluded} of {record.support_n_valid} cycle(s) excluded as invalid "
            f"({fraction:.1%}; {reasons}). Flagged for review: the exclusion does not make "
            "the remaining result reliable."
        )
    return text


def _pb_calibration_caption(
    sample: Sample, ratio_name: str, freshness: Optional[Dict[str, Any]] = None,
) -> str:
    """Plain-language status of the Pb-standard calibration for the selected ratio, or ''.

    Presentation only: every fact comes from the domain records.
    """
    from domain.pb_calibration_records import calibrated_delta_record, calibration_records

    record = calibration_records(sample).get(ratio_name)
    if record is None:
        return ""
    if freshness is not None:
        from domain.calibration_dependencies import effective_calibration_availability
        availability = effective_calibration_availability(sample, ratio_name, freshness)
        if availability.status == "stale":
            return (
                "Historical Pb-standard calibration (stale): the displayed calibrated trace is "
                "retained only for inspection and is not a current final result. Reprocess before reporting."
            )
    if not record.governs_final:
        return (
            f"Not calibrated ({record.reason_code}): the Tl-normalized value is shown as a "
            "diagnostic and is not validated by its own calibration."
        )
    if record.status != "applied":
        return (
            f"Pb-standard calibration unavailable for {ratio_name} ({record.reason_code}): "
            f"{record.reason} No final value is reported."
        )
    text = (
        f"{record.final_layer_label}: K = {record.k_applied:.10g} from "
        f"{len(record.members)} standard observation(s) of {record.reference.get('material', '')}."
    )
    if record.skipped:
        text += " Skipped: " + ", ".join(f"{s.label} ({s.reason_code})" for s in record.skipped) + "."
    if record.excluded_cycles:
        text += (
            f" {record.n_excluded} of {record.support_n_valid} cycle(s) excluded as invalid; "
            "flagged for review."
        )
    delta = calibrated_delta_record(sample, ratio_name)
    if delta is not None and delta.status == "applied":
        text += (
            f" Calibrated δ = {delta.delta_mean:.4f} ‰ (stored support); combined uncertainty: Not calculated."
        )
    text += " Absolute-ratio uncertainty is not modelled yet (unavailable)."
    return text


def _get_iif_like_plot_label(sample: Sample, ratio_name: str) -> str:
    """Label the downstream correction trace from the layer actually plotted."""
    from domain.pb_calibration_records import calibration_records

    from domain.sr_standard_calibration import sr_calibration_record
    if sr_calibration_record(sample, ratio_name):
        return "Internally normalized"
    if ratio_name in calibration_records(sample) and ratio_name in (sample.iif_corrected_ratios or {}):
        # On the calibrated route this trace is the Tl-only intermediate.
        return "Tl-normalized"
    if sample.iif_corrected_ratios and ratio_name in sample.iif_corrected_ratios:
        return "IIF-corrected"
    if sample.ssb_results and ratio_name in sample.ssb_results:
        return "SSB-corrected"
    return "IIF-corrected"


def _select_visible_intensity_layer(
    sample: Sample,
    isotope: str,
    display_config: DisplayConfig,
):
    """Return the intensity layer currently represented by the Inspector."""
    blank_active = _blank_correction_active(sample)
    blank_cd = None
    interference_cd = None

    hg_intensity_cd = (sample.interference_corrected_intensities or {}).get(isotope)
    if display_config.show_interference_corrected and hg_intensity_cd is not None:
        return hg_intensity_cd, INTERFERENCE_LAYER_LABEL

    if blank_active and sample.blank_corrected_intensities:
        blank_cd = sample.blank_corrected_intensities.get(isotope)
        raw_cd = sample.intensities.get(isotope) if sample.intensities else None
        if isotope in KR_ISOTOPES and cycle_data_equal(blank_cd, raw_cd):
            blank_cd = None

    if sample.corrected_intensities:
        corrected_cd = sample.corrected_intensities.get(isotope)
        if corrected_cd is not None:
            if blank_cd is not None:
                if (
                    not np.array_equal(corrected_cd.values, blank_cd.values)
                    or not np.array_equal(corrected_cd.mask, blank_cd.mask)
                ):
                    interference_cd = corrected_cd
            elif isotope in {"84Sr", "86Sr", "87Sr"}:
                raw_cd = sample.intensities.get(isotope) if sample.intensities else None
                if raw_cd is None or (
                    not np.array_equal(corrected_cd.values, raw_cd.values)
                    or not np.array_equal(corrected_cd.mask, raw_cd.mask)
                ):
                    interference_cd = corrected_cd
            elif blank_active and blank_cd is None:
                blank_cd = corrected_cd

    if display_config.show_interference_corrected and interference_cd is not None:
        return interference_cd, "Interference-corrected"

    if display_config.show_corrected and blank_cd is not None:
        return blank_cd, "Blank-corrected"

    if display_config.show_raw and sample.intensities and isotope in sample.intensities:
        return sample.intensities[isotope], "Raw"

    if interference_cd is not None:
        return interference_cd, "Interference-corrected"

    if blank_cd is not None:
        return blank_cd, "Blank-corrected"

    if sample.intensities and isotope in sample.intensities:
        return sample.intensities[isotope], "Raw"

    return None, "Raw"


def _count_exclusions_in_range(excluded_cycles: list[int], cycle_range: Optional[tuple], n_cycles: int) -> tuple[int, int]:
    """Return total visible cycles and excluded cycles within that visible window."""
    if cycle_range:
        start, end = cycle_range
        visible_cycles = max(0, end - start + 1)
        excluded_in_range = sum(start <= cycle <= end for cycle in excluded_cycles)
        return visible_cycles, excluded_in_range
    return n_cycles, len(excluded_cycles)


def _get_uncertainty_budget_ratios(
    sample: Sample,
    selected_ratio: Optional[str] = None,
) -> List[str]:
    """Return the ratio list the Inspector uncertainty budget should show."""
    if not sample.ratios:
        return []
    if selected_ratio and selected_ratio in sample.ratios:
        return [selected_ratio]
    try:
        state = get_state()
        primary = (
            state.element_config.primary_ratio
            if state.element_config and state.element_config.primary_ratio
            else None
        )
    except Exception:
        primary = None
    if primary and primary in sample.ratios:
        return [primary]
    return [next(iter(sample.ratios.keys()))]


def render_inspector_tab() -> None:
    """Render the Sample Inspector tab."""
    state = get_state()

    if not state.has_data:
        st.info("Upload an HDF5 file to inspect samples.")
        return

    # Auto-navigate to last edited sample (from Overview tab data editor)
    last_edited = st.session_state.get(SESSION_KEY_LAST_EDITED_SAMPLE_NAME)
    if last_edited:
        # Try to select the last edited sample by name
        if state.select_sample_by_name(last_edited):
            # Clear the flag after successful navigation
            st.session_state.pop(SESSION_KEY_LAST_EDITED_SAMPLE_NAME, None)

    # Get selected sample
    sample = state.selected_sample
    if sample is None:
        st.info("Select a sample from the sidebar.")
        return

    # When the user navigates to a different sample, discard any staged-but-not-committed
    # exclusions for the incoming sample and reset the chart version so stale Plotly
    # point-selection state is cleared.  Committed exclusions (in sample.metadata) are
    # unaffected — only the un-applied multiselect staging is discarded.
    _nav_key = "_insp_last_sample"
    _cur_key = get_sample_state_key(sample)
    if st.session_state.get(_nav_key) not in (None, _cur_key):
        st.session_state.pop(f"excluded_{_cur_key}", None)
        st.session_state.pop(f"excluded_{sample.name}", None)
        _cv = f"chart_ver_{_cur_key}"
        st.session_state[_cv] = st.session_state.get(_cv, 0) + 1
    st.session_state[_nav_key] = _cur_key

    # Compact review strip: navigation header, then the isotope/ratio selectors
    # beside the active cycle-window and exclusion summary, then the staged
    # cycle and exclusion forms. The header and summary slots are filled once
    # those forms have resolved the current window and staged exclusions.
    header_slot = st.container()
    with st.container(key="inspector_selectors"):
        selector_col, summary_col = st.columns([1.2, 1], gap="medium")
    with summary_col:
        summary_slot = st.container()
    with selector_col:
        col_sel1, col_sel2 = st.columns(2)

    with col_sel1:
        isotopes = [
            isotope
            for isotope in (list(sample.intensities.keys()) if sample.intensities else [])
            if isotope.strip().lower() != "cycle"
        ]

        sample_key = get_sample_state_key(sample)
        isotope_pref_key = f"inspector_last_isotope::{sample_key}"
        # Persist isotope selection per sample identity
        last_selected_isotope = st.session_state.get(
            isotope_pref_key,
            st.session_state.get("inspector_last_isotope"),
        )
        default_iso_idx = 0
        if last_selected_isotope and last_selected_isotope in isotopes:
            default_iso_idx = isotopes.index(last_selected_isotope)

        selected_isotope = st.selectbox(
            "Select Isotope",
            options=isotopes,
            index=default_iso_idx if isotopes else None,
            format_func=format_ratio_label,
            key="inspector_isotope_select",
        ) if isotopes else None

        # Store selected isotope for this sample
        if selected_isotope:
            st.session_state[isotope_pref_key] = selected_isotope
            st.session_state["inspector_last_isotope"] = selected_isotope

    with col_sel2:
        from ui.components.custom_ratio import get_selected_ratios

        # the inspector uses (post-processing if available)
        effective_samples = state.result.samples if state.has_result else state.samples
        selected_ratios = get_selected_ratios(effective_samples)

        # Filter sample's ratios to only show selected ones
        sample_ratios = set(sample.ratios.keys()) if sample.ratios else set()
        ratios = sorted(selected_ratios & sample_ratios)

        ratio_pref_key = f"inspector_last_ratio::{sample_key}"
        # Persist ratio selection per sample identity
        last_selected_ratio = st.session_state.get(
            ratio_pref_key,
            st.session_state.get("inspector_last_ratio"),
        )
        default_idx = 0
        if last_selected_ratio and last_selected_ratio in ratios:
            default_idx = ratios.index(last_selected_ratio)

        selected_ratio = st.selectbox(
            "Select Ratio",
            options=ratios,
            index=default_idx if ratios else None,
            format_func=format_ratio_label,
            key="inspector_ratio_select",
        ) if ratios else None

        # Store selected ratio for this sample
        if selected_ratio:
            st.session_state[ratio_pref_key] = selected_ratio
            st.session_state["inspector_last_ratio"] = selected_ratio

    # Cycle Settings and Manual Exclusions keep their staged widgets; they sit
    # side by side and stack when the workspace is narrow.
    with st.container(key="inspector_controls"):
        cycle_col, exclusion_col = st.columns(2, gap="medium")
    with cycle_col:
        cycle_range = _render_cycle_settings(sample)
    cycle_ranges = _get_cycle_ranges()
    with exclusion_col:
        manual_exclusion_mode = _render_manual_exclusions(sample, cycle_range)
    with header_slot:
        _render_sample_header(
            sample,
            display_config=state.display_config,
            selected_ratio=selected_ratio,
            cycle_range=cycle_range,
            cycle_ranges=cycle_ranges,
        )
    with summary_slot:
        _render_review_context_summary(sample, cycle_range)

    st.divider()

    with st.expander("Plot display", expanded=False):
        # These controls affect only the Inspector plots below. Their values
        # remain owned by DisplayConfig while this expander or tab is hidden.
        from ui.sidebar import render_plot_display_options

        render_plot_display_options()

    # The two diagnostics sit side by side on wide workspaces and stack when
    # width is limited. The ratio chart explicitly uses responsive sizing so its
    # interactive wrapper fills its column. Ratio tick labels carry more decimal
    # places and therefore consume more internal left margin; a slight width
    # bias keeps the visible plot frames approximately equal.
    with st.container(key="inspector_plots"):
        col1, col2 = st.columns([0.48, 0.52])
    with col1:
        _render_intensity_section(
            sample,
            state.display_config,
            selected_isotope,
            selected_ratio,
            cycle_range,
        )
    with col2:
        _render_ratio_section(
            sample,
            state.display_config,
            selected_ratio,
            cycle_range,
            manual_exclusion_mode=manual_exclusion_mode,
        )

    runtime_delta = None
    from domain.pb_standard_calibration import calibration_requested, calibrated_delta_requested
    delta_requested = bool(
        state.element_config and state.processing_config and state.element_config.supports_delta
        and (
            calibrated_delta_requested(getattr(state.element_config, "symbol", ""), state.processing_config)
            if calibration_requested(getattr(state.element_config, "symbol", ""), state.processing_config)
            else state.processing_config.enable_delta
        )
    )
    if selected_ratio and delta_requested:
        state_samples = state.result.samples if state.has_result else (state.samples or [])
        runtime_delta = compute_runtime_delta(
            sample,
            selected_ratio,
            state_samples,
            cycle_ranges=cycle_ranges,
            filter_method=state.processing_config.filter_method,
            filter_threshold=state.processing_config.get_active_filter_threshold(),
            processing_config=state.processing_config,
            element_config=state.element_config,
            calibration_freshness=current_pb_calibration_freshness(state),
        )

    st.subheader("Summary Statistics")

    intensity_stats = _render_intensity_stats(
        sample,
        state.display_config,
        selected_isotope,
        selected_ratio,
        cycle_range,
    )
    quick_stats_kwargs = (
        {"runtime_delta": runtime_delta}
        if runtime_delta is not None
        else {}
    )
    ratio_stats = _render_quick_stats(
        sample,
        state.display_config,
        selected_ratio,
        cycle_range,
        cycle_ranges,
        **quick_stats_kwargs,
    )
    _render_statistics_table(intensity_stats, ratio_stats)

    if runtime_delta is not None and selected_ratio:
        st.divider()
        _render_delta_section(
            sample,
            selected_ratio,
            runtime_delta,
            state.display_config,
        )

    # Corrections and uncertainty (if processed)
    if state.has_result:
        st.divider()
        _render_correction_summary(sample, selected_ratio, cycle_range, cycle_ranges)


def _render_cycle_settings(sample: Sample) -> tuple:
    """Render cycle range settings with global/per-sample option.

    Returns
    -------
    Tuple of (start_cycle, end_cycle) for the current sample.
    """
    state = get_state()
    n_cycles = sample.n_cycles

    with st.expander("Cycle Settings", expanded=False):
        # Global vs per-sample toggle
        apply_global = st.checkbox(
            "Apply same cycle range to all samples",
            value=state.global_cycle_range_enabled,
            key="global_cycle_range_toggle",
        )
        if state.global_cycle_range_enabled != apply_global:
            state.processing_config.global_cycle_range = apply_global
            state.global_cycle_range_enabled = apply_global

        if apply_global:
            # Use the minimum cycle count so the range is valid for every
            # sample; using the largest count would exceed shorter runs.
            all_counts = [s.n_cycles for s in state.samples] if state.samples else [n_cycles]
            common_max_cycles = min(all_counts) if all_counts else n_cycles
            # Safety: must be at least 1
            common_max_cycles = max(common_max_cycles, 1)

            raw_global_range = state.global_cycle_range or (1, common_max_cycles)
            global_range = clamp_cycle_range(raw_global_range, max_cycles=common_max_cycles)

            cycle_range = (1, 1) if common_max_cycles == 1 else st.slider(
                "Cycle Range (All Samples)",
                min_value=1,
                max_value=common_max_cycles,
                value=global_range,
                key="global_cycle_range_slider",
            )
            state.global_cycle_range = cycle_range

            st.caption(f"Applied to all {len(state.samples)} samples")
        else:
            # Per-sample cycle range - stored in session state by stable sample key.
            sample_key = get_sample_state_key(sample)
            range_key = f"cycle_range_{sample_key}"
            legacy_range_key = f"cycle_range_{sample.name}"

            n_cycles_safe = max(1, n_cycles)
            raw_existing_range = st.session_state.get(
                range_key,
                st.session_state.get(legacy_range_key, (1, n_cycles_safe)),
            )
            existing_range = clamp_cycle_range(raw_existing_range, max_cycles=n_cycles_safe)

            cycle_range = (1, 1) if n_cycles_safe == 1 else st.slider(
                f"Cycle Range ({sample.name} | Run {sample.run_number})",
                min_value=1,
                max_value=n_cycles_safe,
                value=existing_range,
                key=f"cycle_range_slider_{sample_key}",
            )
            st.session_state[range_key] = cycle_range
            st.session_state[legacy_range_key] = cycle_range
            # Structural path avoids legacy key-prefix collisions for sample
            # names that look like widget keys, e.g. "slider_*".
            structural_ranges = st.session_state.setdefault(_CYCLE_RANGES_STATE_KEY, {})
            structural_ranges[sample_key] = cycle_range

    return cycle_range


def _hide_plot_grids(fig):
    """Force Inspector diagnostic plots to stay gridless."""
    if fig is None:
        return None
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def _apply_inspector_plot_sizing(fig) -> None:
    """Apply consistent screen and export sizing to Inspector figures."""
    if fig.layout.legend.orientation == "h":
        # Ratio layers and their statistical guides can require four or more
        # legend rows. Reserve a fixed header rather than constraining the
        # legend's height, which clips later rows behind the plot frame.
        # The legend bottom still remains 12 px above the frame.
        plot_area_height = max(
            float(fig.layout.height or 550) - _INSPECTOR_LEGEND_TOP_MARGIN - 85,
            1,
        )
        fig.update_layout(
            legend=dict(
                entrywidth=0, entrywidthmode="pixels",
                yref="paper", y=1 + 12 / plot_area_height, yanchor="bottom",
            ),
            title=dict(y=0.99, yanchor="top", yref="container"),
            margin=dict(t=_INSPECTOR_LEGEND_TOP_MARGIN, b=85, autoexpand=False),
        )
        fig.layout.legend.maxheight = None
    fig.update_layout(
        autosize=True,
        font=dict(size=_INSPECTOR_PLOT_TICK_FONT_SIZE),
        title=dict(font=dict(size=_INSPECTOR_PLOT_TITLE_FONT_SIZE)),
        legend=dict(font=dict(size=_INSPECTOR_PLOT_LEGEND_FONT_SIZE)),
    )
    fig.update_xaxes(
        tickfont=dict(size=_INSPECTOR_PLOT_TICK_FONT_SIZE),
        title_font=dict(size=_INSPECTOR_PLOT_AXIS_TITLE_FONT_SIZE),
    )
    fig.update_yaxes(
        tickfont=dict(size=_INSPECTOR_PLOT_TICK_FONT_SIZE),
        title_font=dict(size=_INSPECTOR_PLOT_AXIS_TITLE_FONT_SIZE),
    )
    fig.update_annotations(font=dict(size=_INSPECTOR_PLOT_ANNOTATION_FONT_SIZE))


def _navigate_to_sample(state, target_idx: int) -> None:
    """Synchronize Inspector and sidebar selection before the interaction rerun."""
    samples = (state.result.samples if state.has_result else state.samples) or []
    if not 0 <= target_idx < len(samples):
        return
    state.select_sample(target_idx)
    st.session_state["sample_selectbox_picker"] = samples[target_idx].observation_id


def _effective_ratio_values(
    sample: Sample,
    display_config: DisplayConfig,
    selected_ratio: Optional[str],
    cycle_range: Optional[tuple],
    cycle_ranges: Optional[dict],
) -> tuple[np.ndarray, str]:
    """Return the values used by the selected-ratio Inspector statistics."""
    if selected_ratio is None:
        return np.asarray([], dtype=float), ""
    state = get_state()
    data, label = _select_visible_ratio_layer(
        sample,
        selected_ratio,
        display_config,
        element_cfg=state.element_config,
    )
    if data is None:
        return np.asarray([], dtype=float), label
    active_cycle_ranges = _resolve_active_cycle_ranges(
        sample,
        cycle_range=cycle_range,
        cycle_ranges=cycle_ranges,
    )
    processing_config = state.processing_config
    return (
        _extract_runtime_valid_values(
            data.values,
            data.mask,
            sample.name,
            sample_key=get_sample_state_key(sample),
            cycle_range=cycle_range,
            cycle_ranges=active_cycle_ranges,
            filter_method=(processing_config.filter_method if processing_config else "None"),
            filter_threshold=(processing_config.get_active_filter_threshold() if processing_config else 2.0),
        ),
        label,
    )


def _render_sample_header(
    sample: Sample,
    *,
    display_config: Optional[DisplayConfig] = None,
    selected_ratio: Optional[str] = None,
    cycle_range: Optional[tuple] = None,
    cycle_ranges: Optional[dict] = None,
) -> None:
    """Render the sample header with type badge, excluded state, valid/total,
    and Prev/Next navigation buttons."""
    state = get_state()
    all_samples = (state.result.samples if state.has_result else state.samples) or []
    current_idx = state.selected_sample_idx
    n_total = len(all_samples)

    stype = sample.sample_type.upper()
    badge_class = f"badge-{stype.lower()[:3]}"
    is_excluded = bool(sample.metadata.get("excluded", False))
    excluded_html = (
        ' <span class="badge-exc">EXCLUDED</span>'
        if is_excluded else ""
    )

    valid_count = None
    accepted_label = "Accepted"
    if display_config is not None and selected_ratio is not None:
        valid_values, _ = _effective_ratio_values(
            sample, display_config, selected_ratio, cycle_range, cycle_ranges
        )
        valid_count = len(valid_values)
        accepted_label = f"Accepted · {selected_ratio}"

    col_name, col_run, col_cyc, col_valid, col_prev, col_next = st.columns(
        [3, 0.8, 0.8, 0.9, 0.65, 0.65]
    )

    with col_name:
        st.markdown(
            f'<h3 style="margin-bottom: 0.2rem;">{html.escape(sample.name)} '
            f'<span class="{badge_class}">{stype}</span>'
            f'{excluded_html}</h3>',
            unsafe_allow_html=True,
        )

    with col_run:
        st.metric("Run", sample.run_number)

    with col_cyc:
        st.metric("Cycles", sample.n_cycles)

    with col_valid:
        if valid_count is not None:
            n_in_range = (
                cycle_range[1] - cycle_range[0] + 1
                if cycle_range is not None
                else sample.n_cycles
            )
            st.metric(accepted_label, f"{valid_count}/{n_in_range}")
        else:
            st.metric(accepted_label, "—")

    with col_prev:
        # Mirrors the established selection pattern (sidebar.py:819) — commit the
        # index through the bounds-checked setter, then rerun so the tab body
        # re-reads state.selected_sample from the top.
        if st.button(
            "◀",
            key="_insp_nav_prev_btn",
            disabled=(current_idx <= 0),
            help="Previous sample",
            width="stretch",
            on_click=_navigate_to_sample,
            args=(state, current_idx - 1),
        ):
            pass

    with col_next:
        if st.button(
            "▶",
            key="_insp_nav_next_btn",
            disabled=(current_idx >= n_total - 1),
            help="Next sample",
            width="stretch",
            on_click=_navigate_to_sample,
            args=(state, current_idx + 1),
        ):
            pass


def _render_intensity_section(
    sample: Sample,
    display_config: DisplayConfig,
    selected_isotope: Optional[str] = None,
    selected_ratio: Optional[str] = None,
    cycle_range: tuple = None,
) -> None:
    """Render intensity plots section."""
    st.subheader("Intensity")

    if not sample.intensities:
        st.caption("No intensity data available.")
        return

    if selected_isotope is None:
        selected_isotope = list(sample.intensities.keys())[0]

    plot_height = min(int(get_state().plot_config.get("height", 650)), 550)

    # Single isotope plot (larger, focused) with cycle range
    # Note: show_threshold=False for intensity (threshold lines only for ratios)
    fig = create_single_isotope_plot(
        sample,
        selected_isotope,
        show_raw=display_config.show_raw,
        show_corrected=display_config.show_corrected,
        show_interference=display_config.show_interference_corrected,
        show_outliers=display_config.show_outliers,
        show_threshold=False,
        height=plot_height,
        cycle_range=cycle_range,
        selected_ratio=selected_ratio,
    )
    _hide_plot_grids(fig)
    _apply_inspector_plot_sizing(fig)
    isotope_slug = selected_isotope.replace("/", "_")
    sample_slug = sample.name.replace("/", "_").replace("\\", "_")
    intensity_plot_config = get_plotly_config(
        f"traceiso_{sample_slug}_{isotope_slug}_intensity",
        width=_INSPECTOR_PLOT_EXPORT_WIDTH,
        height=_INSPECTOR_PLOT_EXPORT_HEIGHT,
    )
    intensity_plot_config["responsive"] = True
    st.plotly_chart(
        fig,
        width="stretch",
        height=plot_height,
        key=f"inspector_intensity_main_{selected_isotope}",
        config=intensity_plot_config,
    )


def _render_ratio_section(
    sample: Sample,
    display_config: DisplayConfig,
    selected_ratio: Optional[str] = None,
    cycle_range: tuple = None,
    *,
    manual_exclusion_mode: bool = False,
) -> None:
    """Render ratio plots section."""
    st.subheader("Ratio")

    if not sample.ratios:
        st.caption("No ratio data available.")
        return

    if selected_ratio is None:
        selected_ratio = list(sample.ratios.keys())[0]

    state = get_state()
    iif_label = _get_iif_like_plot_label(sample, selected_ratio)

    # Compute sample key and staged exclusions before building the figure so
    # staged cycles can be overlaid on the plot (I2).
    sample_key = get_sample_state_key(sample)
    widget_key = f"excluded_{sample_key}"
    staged_cycles_raw = st.session_state.get(widget_key, [])
    staged_cycles = list(staged_cycles_raw) if staged_cycles_raw else None

    plot_height = min(int(state.plot_config.get("height", 650)), 550)

    # Single ratio plot (no CRM line — that belongs in Results Overview)
    fig = create_single_ratio_plot(
        sample,
        selected_ratio,
        show_raw=display_config.show_raw,
        show_corrected=display_config.show_corrected,
        show_drift=display_config.show_drift_corrected,
        show_interference=(
            bool(
                (state.element_config and state.element_config.has_interference)
                or sample.interference_corrected_ratios
            )
            and display_config.show_interference_corrected
        ),
        show_iif=display_config.show_iif_corrected,
        show_pb_standard=display_config.show_pb_standard_corrected,
        show_sr_standard=display_config.show_sr_standard_corrected,
        show_filtered_raw=display_config.show_filtered_raw,
        show_filtered_corrected=display_config.show_filtered_corrected,
        show_threshold=display_config.show_threshold_lines,
        show_outliers=display_config.show_outliers,
        threshold_basis=display_config.threshold_line_basis,
        show_stats_box=display_config.show_ratio_stats_box,
        stats_statistic=display_config.ratio_stats_statistic,
        iif_label=iif_label,
        height=plot_height,
        cycle_range=cycle_range,
        staged_cycles=staged_cycles,
    )
    _hide_plot_grids(fig)
    _apply_inspector_plot_sizing(fig)
    sync_key = f"{widget_key}__sync"
    reset_key = f"ratio_view_reset_{sample_key}_{selected_ratio}"
    # chart_ver is incremented by Apply/Revert so the chart gets a fresh key
    # (and therefore no stale Plotly selection state) after each commit.
    chart_ver = st.session_state.get(f"chart_ver_{sample_key}", 0)
    selection_ver_key = f"selection_ver_{sample_key}"
    selection_ver = st.session_state.get(selection_ver_key, 0)
    reset_ver = st.session_state.get(reset_key, 0)
    chart_key = (
        f"inspector_ratio_main_{sample_key}_{selected_ratio}_cv{chart_ver}"
        f"_manual{int(manual_exclusion_mode)}_sv{selection_ver}_rv{reset_ver}"
    )

    ratio_slug = selected_ratio.replace("/", "_")
    sample_slug = sample.name.replace("/", "_").replace("\\", "_")
    ratio_plot_config = get_plotly_config(
        f"traceiso_{sample_slug}_{ratio_slug}",
        width=_INSPECTOR_PLOT_EXPORT_WIDTH,
        height=_INSPECTOR_PLOT_EXPORT_HEIGHT,
    )
    ratio_plot_config["responsive"] = True
    if manual_exclusion_mode:
        for trace in fig.data:
            if "markers" in str(getattr(trace, "mode", "")):
                trace.selected.marker.opacity = 1.0
                trace.unselected.marker.opacity = 1.0
        legacy_widget_key = f"excluded_{sample.name}"

        def _on_ratio_point_select() -> None:
            _stage_cycles_from_selection(
                st.session_state,
                chart_key=chart_key,
                widget_key=widget_key,
                selection_ver_key=selection_ver_key,
                legacy_widget_key=legacy_widget_key,
                sync_key=sync_key,
            )

        st.plotly_chart(
            fig,
            width="stretch",
            height=plot_height,
            key=chart_key,
            config=ratio_plot_config,
            on_select=_on_ratio_point_select,
            selection_mode="points",
        )
    else:
        st.plotly_chart(
            fig,
            width="stretch",
            height=plot_height,
            key=chart_key,
            config=ratio_plot_config,
        )
    if st.button(
        "Reset ratio view",
        key=f"{reset_key}_button",
        help="Restore the full cycle and ratio ranges without changing exclusions or data.",
    ):
        st.session_state[reset_key] = st.session_state.get(reset_key, 0) + 1
        st.rerun()
    from domain.sr_standard_calibration import sr_calibration_message
    sr_caption = sr_calibration_message(sample, selected_ratio)
    if sr_caption:
        st.warning(sr_caption)
    hg_caption = _hg_correction_caption(sample, selected_ratio)
    if hg_caption:
        st.caption(hg_caption)
    calibration_caption = _pb_calibration_caption(
        sample, selected_ratio, current_pb_calibration_freshness(state),
    )
    if calibration_caption:
        st.caption(calibration_caption)
    if manual_exclusion_mode:
        st.caption(
            "Manual exclusion mode is active. Click a point to stage/un-stage it, "
            "then use Apply Exclusions above the plots to commit."
        )


def _apply_selected_ratio_mask(
    sample: Sample,
    base_mask: np.ndarray,
    selected_ratio: Optional[str] = None,
    *,
    cycle_range: Optional[tuple] = None,
    cycle_ranges: Optional[dict] = None,
) -> np.ndarray:
    """Return *base_mask* constrained by the selected ratio's accepted cycles."""
    mask = np.asarray(base_mask, dtype=bool).copy()
    if not selected_ratio:
        return mask

    ratio_cd = get_processing_ratio_data(sample, selected_ratio)
    if ratio_cd is None:
        return mask

    state = get_state()
    processing_config = getattr(state, "processing_config", None)
    filter_method = processing_config.filter_method if processing_config else "None"
    filter_threshold = (
        processing_config.get_active_filter_threshold()
        if processing_config
        else DEFAULT_OUTLIER_THRESHOLD_SD
    )
    active_cycle_ranges = _resolve_active_cycle_ranges(
        sample,
        cycle_range=cycle_range,
        cycle_ranges=cycle_ranges,
    )
    ratio_mask = get_runtime_mask(
        ratio_cd.values,
        ratio_cd.mask,
        sample.name,
        cycle_ranges=active_cycle_ranges,
        sample_key=get_sample_state_key(sample),
        filter_method=filter_method,
        filter_threshold=filter_threshold,
    )
    n = min(len(mask), len(ratio_mask))
    if n > 0:
        mask[:n] = mask[:n] & ratio_mask[:n]
    return mask


@dataclass(frozen=True)
class InspectorStatsColumn:
    """One series (intensity or ratio) of already-formatted Inspector statistics."""

    heading: str
    cells: Dict[str, str] = field(default_factory=dict)
    note: Optional[str] = None


# Row order of the shared statistics table. A series omits rows it does not
# report; those cells read "not reported" rather than a value-like placeholder,
# so they cannot be mistaken for an undefined or unavailable statistic.
_STATISTIC_ROWS = (
    "Mean ± 2SD",
    "SD",
    "SE",
    "RSD",
    "RSE",
    "n accepted",
    "SSB-corrected mean",
    "SSB K factor",
    "δ ± 2SD",
)
_NOT_REPORTED = "not reported"


def _render_statistics_table(*columns: Optional[InspectorStatsColumn]) -> None:
    """Render the intensity and ratio statistics as one aligned table."""
    series = [column for column in columns if isinstance(column, InspectorStatsColumn)]
    if not series:
        return
    rows = [
        (name, [column.cells.get(name, _NOT_REPORTED) for column in series])
        for name in _STATISTIC_ROWS
        if any(name in column.cells for column in series)
    ]
    notes = [column.note for column in series if column.note]
    render_statistics_table(
        ["Statistic", *[column.heading for column in series]],
        rows,
        caption=" ".join(notes) or None,
    )


def _render_review_context_summary(sample: Sample, cycle_range: Optional[tuple]) -> None:
    """Show the active cycle window and exclusion counts beside the selectors."""
    state = get_state()
    render_compact_metadata(
        _review_context_items(
            sample,
            cycle_range,
            apply_global=bool(getattr(state, "global_cycle_range_enabled", False)),
            staged_exclusions=st.session_state.get(f"excluded_{get_sample_state_key(sample)}"),
        )
    )


def _review_context_items(
    sample: Sample,
    cycle_range: Optional[tuple],
    *,
    apply_global: bool,
    staged_exclusions: Optional[list] = None,
) -> list[tuple[str, str]]:
    """Describe the active cycle window and committed manual exclusions.

    Counts come from the committed exclusion list and the resolved window, the
    same inputs the Manual Exclusions form reports; staged edits are flagged,
    never counted as applied.
    """
    n_cycles = sample.n_cycles
    persisted = list(sample.metadata.get("manual_exclusions", []) or [])
    in_window, excluded_in_window = _count_exclusions_in_range(persisted, cycle_range, n_cycles)
    start, end = cycle_range if cycle_range else (1, n_cycles)
    scope = "all samples" if apply_global else "this sample"
    excluded = str(excluded_in_window)
    if len(persisted) != excluded_in_window:
        excluded += f" ({len(persisted)} in sample)"
    items = [
        ("Cycle window", f"{start}–{end} ({in_window} of {n_cycles}) · {scope}"),
        ("Manual exclusions in window", excluded),
    ]
    if staged_exclusions is not None and sorted(staged_exclusions) != sorted(persisted):
        items.append(("Pending", "staged exclusions not applied"))
    return items


def _render_intensity_stats(
    sample: Sample,
    display_config: DisplayConfig,
    selected_isotope: Optional[str] = None,
    selected_ratio: Optional[str] = None,
    cycle_range: tuple = None,
) -> Optional[InspectorStatsColumn]:
    """Collect the selected isotope's intensity statistics for the shared table."""
    if selected_isotope is None or not (
        sample.intensities or sample.corrected_intensities or sample.blank_corrected_intensities
    ):
        return

    src_cd, label = _select_visible_intensity_layer(sample, selected_isotope, display_config)
    if src_cd is None:
        return

    sample_key = get_sample_state_key(sample)
    active_cycle_ranges = _resolve_active_cycle_ranges(sample, cycle_range=cycle_range)
    governed_mask = _apply_selected_ratio_mask(
        sample,
        src_cd.mask,
        selected_ratio,
        cycle_range=cycle_range,
        cycle_ranges=active_cycle_ranges,
    )
    valid = _extract_runtime_valid_values(
        src_cd.values,
        governed_mask,
        sample.name,
        sample_key=sample_key,
        cycle_range=cycle_range,
        cycle_ranges=active_cycle_ranges,
        filter_method="None",
        filter_threshold=DEFAULT_OUTLIER_THRESHOLD_SD,
    )
    n_valid = len(valid)
    if n_valid == 0:
        return

    mean_val = float(np.mean(valid))
    sd_val = float(np.std(valid, ddof=1)) if n_valid > 1 else np.nan
    se_val = sd_val / np.sqrt(n_valid) if n_valid > 1 else np.nan
    rsd_pct = (sd_val / mean_val * 100) if (n_valid > 1 and mean_val != 0) else np.nan
    rse_pct = (se_val / mean_val * 100) if (n_valid > 1 and mean_val != 0) else np.nan

    mean_2s_str = format_value_with_uncertainty(mean_val, 2 * sd_val, unit="V")
    sd_str = format_uncertainty(sd_val, unit="V")
    se_str = format_uncertainty(se_val, unit="V")
    rsd_intensity_str = f"{rsd_pct:.3f}%" if np.isfinite(rsd_pct) else "N/A"
    rse_intensity_str = f"{rse_pct:.3f}%" if np.isfinite(rse_pct) else "N/A"
    return InspectorStatsColumn(
        heading=f"{format_ratio_label(selected_isotope)} ({label})",
        cells={
            "Mean \u00b1 2SD": mean_2s_str,
            "SD": sd_str,
            "SE": se_str,
            "RSD": rsd_intensity_str,
            "RSE": rse_intensity_str,
            "n accepted": str(n_valid),
            "\u03b4 \u00b1 2SD": "",
        },
        note=(
            "Intensity n counts the cycles accepted by the selected ratio."
            if selected_ratio
            else None
        ),
    )


def _render_delta_section(
    sample: Sample,
    selected_ratio: str,
    runtime_delta: Optional[RuntimeDeltaResult],
    display_config: DisplayConfig,
) -> None:
    """Render per-cycle delta scatter plot below ratio plot when SSB/delta is enabled."""
    if runtime_delta is None:
        return

    from domain.pb_calibration_records import calibrated_delta_record
    calibrated = calibrated_delta_record(sample, selected_ratio)
    if calibrated is not None:
        st.caption(
            f"Calibration reference: {calibrated.reference_material} | "
            f"Runtime calibration denominator: {runtime_delta.std_mean:.6f} | "
            "Combined uncertainty and Monte Carlo: Not calculated"
        )
    else:
        st.caption(
            f"Bracketing STDs: {runtime_delta.prev_std} -> {runtime_delta.next_std} | "
            f"Runtime STD mean: {runtime_delta.std_mean:.6f}"
        )

    # CRM-anchored delta: strip "(certified)" suffix for the chart title.
    # Classic bracketing: caption already lists both standards, so use a generic label.
    if calibrated is not None:
        ref_for_chart = calibrated.reference_material
    elif runtime_delta.prev_std.endswith("(certified)"):
        ref_for_chart = runtime_delta.prev_std.replace(" (certified)", "").strip()
    else:
        ref_for_chart = "Bracketing STDs"

    plot_height = min(int(get_state().plot_config.get("height", 650)), 500)
    delta_fig = create_delta_cycle_plot(
        sample,
        selected_ratio,
        show_threshold=False,
        show_stats_box=display_config.show_ratio_stats_box,
        stats_statistic=display_config.ratio_stats_statistic,
        height=plot_height,
        delta_cycle_data=runtime_delta.cycle_data,
        delta_summary={
            "delta": runtime_delta.delta,
            "delta_sd": runtime_delta.delta_sd,
        },
        reference_name=ref_for_chart,
    )
    if delta_fig:
        _hide_plot_grids(delta_fig)
        _apply_inspector_plot_sizing(delta_fig)
        ratio_slug = selected_ratio.replace("/", "_")
        sample_slug = sample.name.replace("/", "_").replace("\\", "_")
        delta_plot_config = get_plotly_config(
            f"traceiso_{sample_slug}_{ratio_slug}_delta",
            width=_INSPECTOR_PLOT_EXPORT_WIDTH,
            height=_INSPECTOR_PLOT_EXPORT_HEIGHT,
        )
        delta_plot_config["responsive"] = True
        # Use the same width as the Ratio plot, positioned in the left column.
        delta_col, _ = st.columns([0.52, 0.48])
        delta_col.plotly_chart(
            delta_fig,
            width="stretch",
            height=plot_height,
            key=f"inspector_delta_chart_{selected_ratio}",
            config=delta_plot_config,
        )


def _render_quick_stats(
    sample: Sample,
    display_config: DisplayConfig,
    selected_ratio: Optional[str] = None,
    cycle_range: tuple = None,
    cycle_ranges: Optional[dict] = None,
    runtime_delta: Optional[RuntimeDeltaResult] = None,
) -> Optional[InspectorStatsColumn]:
    """Collect the selected ratio's statistics for the shared table.

    Renders only the unavailable-state notices; the values are returned so the
    caller can align them with the intensity series in one table.
    """
    from domain.ratio_selection import get_ssb_cycle_data

    if not sample.ratios:
        st.caption("Process data to see statistics.")
        return

    # Use selected ratio or first available
    if selected_ratio is None:
        selected_ratio = list(sample.ratios.keys())[0]

    state = get_state()
    valid_values, label = _effective_ratio_values(
        sample, display_config, selected_ratio, cycle_range, cycle_ranges
    )
    if not label:
        st.warning("No ratio data available for the selected layer.")
        return
    active_cycle_ranges = _resolve_active_cycle_ranges(sample, cycle_range=cycle_range, cycle_ranges=cycle_ranges)
    filter_method = state.processing_config.filter_method if state.processing_config else "None"
    filter_threshold = state.processing_config.get_active_filter_threshold() if state.processing_config else 2.0
    n_valid = len(valid_values)
    n_in_range = cycle_range[1] - cycle_range[0] + 1 if cycle_range else sample.n_cycles

    if n_valid == 0:
        st.warning("No valid cycles in selected range.")
        return

    mean_val = np.nanmean(valid_values)
    std_val = np.nanstd(valid_values, ddof=1) if n_valid > 1 else np.nan
    se_val = std_val / np.sqrt(n_valid) if n_valid > 1 else np.nan
    rsd_pct = (std_val / mean_val * 100) if (n_valid > 1 and mean_val != 0) else np.nan
    rse_pct = (se_val / mean_val * 100) if (n_valid > 1 and mean_val != 0) else np.nan

    mean_2s_str = format_value_with_uncertainty(mean_val, 2 * std_val)
    se_str = format_uncertainty(se_val)
    sd_str = format_uncertainty(std_val)
    rsd_str = f"{rsd_pct:.4f}%" if np.isfinite(rsd_pct) else "N/A"
    rse_str = f"{rse_pct:.4f}%" if np.isfinite(rse_pct) else "N/A"

    cells = {
        "Mean \u00b1 2SD": mean_2s_str,
        "SD": sd_str,
        "SE": se_str,
        "RSD": rsd_str,
        "RSE": rse_str,
        "n accepted": f"{n_valid}/{n_in_range}",
    }

    # Standard-sample bracketing (SSB)-corrected ratio and delta values if available
    if sample.ssb_results and selected_ratio and label != "SSB-corrected":
        ssb_data = sample.ssb_results.get(selected_ratio, {})
        ssb_cd = get_ssb_cycle_data(sample, selected_ratio)
        if ssb_data and ssb_cd is not None:
            ssb_values = _extract_runtime_valid_values(
                ssb_cd.values,
                ssb_cd.mask,
                sample.name,
                sample_key=get_sample_state_key(sample),
                cycle_range=cycle_range,
                cycle_ranges=active_cycle_ranges,
                filter_method=filter_method,
                filter_threshold=filter_threshold,
            )
            if len(ssb_values) > 0:
                ssb_mean = float(np.nanmean(ssb_values))
                k_factor = ssb_data.get("k_factor", None)
                cells["SSB-corrected mean"] = f"{ssb_mean:.6f}"
                if k_factor is not None:
                    cells["SSB K factor"] = f"{k_factor:.6f}"

    if selected_ratio and runtime_delta is not None:
        from domain.pb_calibration_records import calibrated_delta_record
        calibrated = calibrated_delta_record(sample, selected_ratio)
        if calibrated is not None:
            statistic = state.processing_config.pb_standard_calibration.delta_precision_statistic
            cells["\u03b4"] = f"{runtime_delta.delta:.2f} \u2030"
            if statistic == "sd" and runtime_delta.n >= 2:
                cells["Cycle scatter (SD)"] = f"{runtime_delta.delta_sd:.2f} \u2030"
            elif statistic == "se" and runtime_delta.n >= 2:
                cells["Precision of the mean (SE)"] = f"{runtime_delta.delta_se:.2f} \u2030"
            cells["Combined uncertainty / MC"] = "Not calculated"
        else:
            cells["\u03b4 \u00b1 2SD"] = (
                f"{runtime_delta.delta:.2f} \u00b1 {2 * runtime_delta.delta_sd:.2f} \u2030"
            )

    return InspectorStatsColumn(
        heading=f"{format_ratio_label(selected_ratio)} ({label})",
        cells=cells,
    )


def _commit_manual_exclusions(sample: Sample, excluded_cycles: list[int]) -> None:
    """Invoke the exclusion state transition for the current session.

    A100: the edit itself lives in :mod:`ui.edit_actions`. This render module
    supplies the values the user chose and calls the action; it does not touch
    the committed samples. Retained as the Inspector's own entry point so the
    button branches read as actions rather than reaching across modules.
    """
    commit_manual_exclusions(get_state(), sample, excluded_cycles)


def _toggle_cycle_from_click(
    cycle_number: int,
    widget_key: str,
    session_state: dict,
) -> list[int]:
    """Toggle cycle_number in the staged exclusion list at session_state[widget_key].

    Adds the cycle if absent; removes it if already staged. The multiselect
    widget reads the same key, so it will reflect the change on the next render
    without requiring an extra rerun.

    Returns the updated sorted list.
    """
    current = list(session_state.get(widget_key, []))
    if cycle_number in current:
        current.remove(cycle_number)
    else:
        current.append(cycle_number)
    result = sorted(current)
    session_state[widget_key] = result
    return result


def _stage_cycles_from_selection(
    session_state: MutableMapping[str, Any],
    *,
    chart_key: str,
    widget_key: str,
    selection_ver_key: str,
    legacy_widget_key: Optional[str] = None,
    sync_key: Optional[str] = None,
) -> Optional[List[int]]:
    """Stage Plotly-selected cycles during its pre-render callback."""
    if sync_key is not None and sync_key in session_state:
        return None

    chart_state = session_state.get(chart_key) or {}
    if not hasattr(chart_state, "get"):
        return None
    selection = chart_state.get("selection") or {}
    if not hasattr(selection, "get"):
        return None
    points = selection.get("points") or []
    if not points:
        return None

    cycles: set[int] = set()
    for point in points:
        if not hasattr(point, "get") or point.get("x") is None:
            continue
        try:
            numeric_x = float(point.get("x"))
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite(numeric_x) or not numeric_x.is_integer():
            continue
        cycles.add(int(numeric_x))

    if not cycles:
        return None

    for cycle_number in sorted(cycles):
        _toggle_cycle_from_click(cycle_number, widget_key, session_state)

    staged = list(session_state.get(widget_key, []))
    if legacy_widget_key and legacy_widget_key != widget_key:
        session_state[legacy_widget_key] = list(staged)

    session_state[selection_ver_key] = (
        int(session_state.get(selection_ver_key, 0)) + 1
    )
    return staged


def _render_manual_exclusions(sample: Sample, cycle_range: tuple = None) -> bool:
    """Render manual exclusion controls and return whether edit mode is active."""
    if not sample.ratios:
        return False

    n_cycles = sample.n_cycles
    sample_key = get_sample_state_key(sample)

    # Load persisted exclusions (if any) as the default for the widget
    persisted = sample.metadata.get("manual_exclusions", [])
    widget_key = f"excluded_{sample_key}"
    legacy_widget_key = f"excluded_{sample.name}"
    sync_key = f"{widget_key}__sync"
    mode_key = f"manual_exclusion_mode_{sample_key}"
    mode_reset_key = f"{mode_key}__reset"
    if sync_key in st.session_state:
        st.session_state[widget_key] = st.session_state.pop(sync_key)
        st.session_state[legacy_widget_key] = st.session_state[widget_key]
    elif legacy_widget_key in st.session_state and widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state[legacy_widget_key]
    elif widget_key not in st.session_state:
        st.session_state[widget_key] = persisted
        st.session_state[legacy_widget_key] = persisted
    if st.session_state.pop(mode_reset_key, False):
        st.session_state[mode_key] = False

    pending_changes = sorted(st.session_state[widget_key]) != sorted(persisted)
    with st.expander(
        "Manual Exclusions",
        expanded=bool(st.session_state.get(mode_key, False) or pending_changes),
    ):
        manual_mode = st.checkbox(
            "Enable manual exclusion mode",
            value=False,
            key=mode_key,
            help=(
                "When enabled, clicking a point in the Ratio plot stages or "
                "unstages that cycle for exclusion."
            ),
        )

        # Manual exclusions multiselect
        excluded_cycles = st.multiselect(
            "Exclude specific cycles",
            options=list(range(1, n_cycles + 1)),
            key=widget_key,
            disabled=not manual_mode,
        )

        # Summary
        in_range, excluded_in_range = _count_exclusions_in_range(
            excluded_cycles,
            cycle_range,
            n_cycles,
        )

        st.caption(
            f"**Visible:** {in_range} cycles | "
            f"**Excluded in view:** {excluded_in_range}"
        )
        if len(excluded_cycles) != excluded_in_range:
            st.caption(f"Total excluded across the sample: {len(excluded_cycles)}")

        # Apply / discard-staged / clear-committed are intentionally distinct.
        col_a, col_b, col_c = st.columns(3)

        has_changes = sorted(excluded_cycles) != sorted(persisted)
        has_exclusions = len(persisted) > 0

        with col_a:
            if st.button(
                "Apply Exclusions",
                type="primary" if has_changes else "secondary",
                width="stretch",
                disabled=not manual_mode or not has_changes,
                key=f"apply_excl_{sample_key}",
            ):
                _commit_manual_exclusions(sample, excluded_cycles)
                st.session_state[sync_key] = sorted(excluded_cycles)
                st.session_state[legacy_widget_key] = sorted(excluded_cycles)
                st.session_state[mode_reset_key] = True
                # Bump chart version so the ratio plot gets a fresh key on rerun,
                # clearing any stale Plotly point-selection state.
                chart_ver_key = f"chart_ver_{sample_key}"
                st.session_state[chart_ver_key] = st.session_state.get(chart_ver_key, 0) + 1
                st.rerun()

        with col_b:
            if st.button(
                "Discard pending changes",
                type="secondary",
                width="stretch",
                disabled=not has_changes,
                key=f"discard_excl_{sample_key}",
            ):
                restored = sorted(persisted)
                st.session_state[sync_key] = restored
                st.session_state[legacy_widget_key] = restored
                chart_ver_key = f"chart_ver_{sample_key}"
                st.session_state[chart_ver_key] = st.session_state.get(chart_ver_key, 0) + 1
                st.rerun()

        with col_c:
            if st.button(
                "Clear applied exclusions",
                type="secondary",
                width="stretch",
                disabled=not manual_mode or not has_exclusions,
                key=f"clear_excl_{sample_key}",
            ):
                _commit_manual_exclusions(sample, [])
                st.session_state[sync_key] = []
                st.session_state[legacy_widget_key] = []
                st.session_state[mode_reset_key] = True
                chart_ver_key = f"chart_ver_{sample_key}"
                st.session_state[chart_ver_key] = st.session_state.get(chart_ver_key, 0) + 1
                st.rerun()

    return bool(manual_mode)


def _render_correction_summary(
    sample: Sample,
    selected_ratio: Optional[str] = None,
    cycle_range: tuple = None,
    cycle_ranges: Optional[Dict] = None,
) -> None:
    """Render summary of corrections applied and the uncertainty budget."""
    render_panel_heading("Provenance & Budget")
    with st.expander("Correction Chain Provenance", expanded=False):
        state = get_state()
        config = state.processing_config

        corrections = []

        # Blank correction
        if config.blank_mode != "none":
            corrections.append(f"Blank: {config.blank_mode}")

        # Outlier rejection
        if config.filter_method != "None":
            corrections.append(
                f"Outlier rejection: {config.filter_method} ({config.format_filter_parameter()})"
            )

        has_iif_result = bool(sample.iif_corrected_ratios)

        # Element-specific
        if state.element_config:
            if state.element_config.has_interference:
                if config.apply_interference_correction:
                    corrections.append("Interference correction (Rb/Kr)")
                if has_iif_result:
                    ratio_name = get_active_internal_normalization_ratio(
                        config,
                        state.element_config,
                    )
                    if ratio_name:
                        corrections.append(
                            f"Mass bias correction via instrumental isotope fractionation (IIF; Russell's law, {ratio_name})"
                        )
                    else:
                        corrections.append(
                            "Mass bias correction via instrumental isotope fractionation (IIF; Russell's law)"
                        )
                    if getattr(config, "sr_session_anchoring", False):
                        corrections.append("Sr-standard calibration")
            else:
                if has_iif_result:
                    ratio_name = get_active_internal_normalization_ratio(
                        config,
                        state.element_config,
                    )
                    normalization_label = (
                        "External normalization (Pb–Tl)"
                        if state.element_config.symbol == "Pb"
                        else "Internal normalization"
                    )
                    if ratio_name:
                        corrections.append(
                            f"{normalization_label} (Russell's law, {ratio_name})"
                        )
                    else:
                        corrections.append(f"{normalization_label} (Russell's law)")
                if sample.ssb_results:
                    corrections.append("Standard-sample bracketing (SSB)")
                if sample.delta_results:
                    corrections.append("Delta calculation")

        if corrections:
            for c in corrections:
                st.markdown(f"- {c}")
        else:
            st.caption("No corrections applied.")

    # Uncertainty budget (runtime, based on current cycle window/masks)
    if sample.ratios:
        with st.expander("Uncertainty Budget", expanded=False):

            state = get_state()
            ratio_names = _get_uncertainty_budget_ratios(sample, selected_ratio)
            cycle_ranges = _resolve_active_cycle_ranges(
                sample, cycle_range=cycle_range, cycle_ranges=cycle_ranges,
            )

            shown_any = False
            for ratio_name in ratio_names:
                all_session = (
                    state.result.samples if state.has_result
                    else (state.samples or [])
                )
                try:
                    budget = get_cached_runtime_budget(
                        sample,
                        ratio_name,
                        element_config=state.element_config,
                        processing_config=state.processing_config,
                        uncertainty_config=state.uncertainty_config,
                        all_samples=all_session,
                        cycle_ranges=cycle_ranges,
                        filter_method=(
                            state.processing_config.filter_method
                            if state.processing_config
                            else "None"
                        ),
                        filter_threshold=(
                            state.processing_config.get_active_filter_threshold()
                            if state.processing_config
                            else 2.0
                        ),
                        drift_fit_info=(
                            state.result.quality_metrics.get("drift_fit_info")
                            if state.has_result and getattr(state.result, "quality_metrics", None)
                            else None
                        ),
                        custom_contributor_library=state.custom_contributor_library,
                        profile_defaults=getattr(state, "uncertainty_profile_defaults", None),
                    )
                except Exception as exc:
                    render_runtime_budget_error(
                        f"Uncertainty Budget ({ratio_name})",
                        exc,
                    )
                    continue
                if budget is None:
                    continue
                shown_any = True
                st.markdown(f"**{ratio_name}**")

                components = []
                # Engine A (Sr / internal normalisation) always displays in
                # absolute units; Engine B always displays in permil.
                # Do NOT follow state.uncertainty_config.output_mode here:
                # that field is auto-synced by the Uncertainty tab and would
                # silently change units after the user visits that tab.
                _element_symbol = (
                    getattr(state.element_config, "symbol", "")
                    if state.element_config else ""
                )
                _engine = budget.engine or (
                    state.uncertainty_config.resolve_engine(
                        _element_symbol,
                        processing_config=getattr(state, "processing_config", None),
                    )
                    if state.uncertainty_config and _element_symbol
                    else "ssb_delta"
                )
                is_abs_mode = is_russell_law_normalization_engine(_engine)

                # Show individual contributors from the new engine
                if budget.contributors:
                    for contrib in budget.contributors:
                        if contrib.is_active and contrib.value_abs > 0:
                            pct = (
                                f" ({contrib.percentage_contribution:.1f}% share)"
                                if contrib.percentage_contribution > 0
                                else ""
                            )
                            if is_abs_mode:
                                val_str = format_uncertainty(contrib.value_abs)
                            else:
                                val_str = format_uncertainty(contrib.value_rel_permil, unit="\u2030")
                            dof = contrib.degrees_of_freedom
                            dof_tag = " [\u03bd=\u221e]" if dof == float('inf') else f" [\u03bd={int(dof)}]"
                            components.append((
                                f"{contrib.display_name}{pct}{dof_tag}",
                                val_str,
                            ))
                else:
                    # Fallback: old-style display
                    u_precision = budget.contributor_value_abs("u_prec")
                    u_blank = budget.contributor_value_abs("u_blank")
                    if u_precision > 0:
                        components.append(("Precision (SE)", format_uncertainty(u_precision)))
                    if u_blank > 0:
                        components.append(("Blank", format_uncertainty(u_blank)))

                if budget.u_combined_abs > 0:
                    k = budget.coverage_factor_k
                    eff_dof = getattr(budget, 'effective_dof', float('inf'))
                    dof_str = f", \u03bd_eff={eff_dof:.0f}" if eff_dof < 1000 else ""
                    if is_abs_mode:
                        components.append((
                            f"**Combined u_c (k=1{dof_str})**",
                            f"**{format_uncertainty(budget.u_combined_abs)}**",
                        ))
                    else:
                        _uc_str = format_uncertainty(budget.u_combined_rel_permil, unit="\u2030")
                        components.append((
                            f"**Combined u_c (k=1{dof_str})**",
                            f"**{_uc_str}**",
                        ))
                if budget.expanded_abs > 0:
                    k = budget.coverage_factor_k
                    if is_abs_mode:
                        _ue_str = format_uncertainty(budget.expanded_abs)
                        components.append((
                            f"**Expanded U (k={k:.2f})**",
                            f"**{_ue_str}**",
                        ))
                    else:
                        _ue_str = format_uncertainty(budget.expanded_rel_permil, unit="\u2030")
                        components.append((
                            f"**Expanded U (k={k:.2f})**",
                            f"**{_ue_str}**",
                        ))

                if components:
                    detail = [(n, v) for n, v in components if not n.startswith("**")]
                    totals = [(n, v) for n, v in components if n.startswith("**")]

                    if detail:
                        rows = "".join(
                            f"<div style='padding:3px 0;border-bottom:1px solid rgba(128,128,128,0.12)'>"
                            f"{n}: {v}</div>"
                            for n, v in detail
                        )
                        st.markdown(
                            f"<div style='font-size:var(--t-size-supporting, 13px);line-height:1.75;"
                            f"margin:4px 0 8px 0'>{rows}</div>",
                            unsafe_allow_html=True,
                        )

                    for n, v in totals:
                        st.markdown(f"{n}: {v}")

                    st.caption(
                        get_k_footnote(
                            coverage_method=get_state().uncertainty_config.coverage_method,
                            k=float(budget.coverage_factor_k),
                            sample_specific=(
                                get_state().uncertainty_config.coverage_method
                                == "welch_satterthwaite"
                            ),
                        )
                    )
                else:
                    st.caption("  No uncertainty data available.")

            if not shown_any:
                st.caption("  No uncertainty data available for the current cycle selection.")

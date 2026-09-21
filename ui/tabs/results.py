"""Results & Statistics tab for TraceISO."""

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple

import numpy as np
import streamlit as st

from config.constants import DEFAULT_OUTLIER_THRESHOLD_SD
from domain.filters.outlier import get_filtered_values
from domain.ratio_selection import select_best_ratio_layer
from ui.components.workspace_ui import (
    render_compact_metadata,
    render_next_step,
    render_subview_nav,
)
from ui.navigation import request_main_section, resolve_subview_selection
from ui.state import get_state
from ui.utils import (
    format_delta_label,
    format_isotope_label,
    get_cycle_ranges,
    get_filtered_sample_caption,
    get_sample_state_key,
    current_pb_calibration_freshness,
    settings_changed_since_processing,
    sticky_type_multiselect,
)
from ui.config_plotly import (
    PLOTLY_ANNOTATION_FONT_SIZE,
    PLOTLY_AXIS_TITLE_FONT_SIZE,
    PLOTLY_BASE_FONT_SIZE,
    PLOTLY_TITLE_FONT_SIZE,
)
_SELECTED_PLOT_EXPORT_WIDTH = 1200
_SELECTED_PLOT_EXPORT_HEIGHT = 800
_OVERVIEW_PLOT_EXPORT_WIDTH = 1200
_OVERVIEW_PLOT_EXPORT_HEIGHT = 800
_RESULTS_OVERVIEW_DISPLAY_HEIGHT = 440
_RESULTS_SELECTED_DISPLAY_HEIGHT = 460
_RESULTS_ENLARGED_DISPLAY_HEIGHT = 650
_LONG_SESSION_RUN_ORDER_THRESHOLD = 24

# Shared "Sample axis" widget key, consumed by the Overview and Summary Table
# subviews.  The `sticky_` prefix is load-bearing: the radio is not instantiated
# on every run (it is hidden in the subviews that ignore it, and the whole tab is
# lazily rendered), and Streamlit evicts session state for widgets it did not
# render.  app_shell._keep_sticky_widgets_alive reasserts sticky keys instead.
RESULTS_AXIS_STATE_KEY = "sticky_results_sample_axis"


def _default_results_sample_axis(sample_count: int) -> str:
    """Prefer uncluttered run order once a session is long enough."""
    return (
        "Run order"
        if sample_count >= _LONG_SESSION_RUN_ORDER_THRESHOLD
        else "Names"
    )


def _results_marker_size(fig, *, profile: Literal["screen", "export"]) -> int:
    """Choose a legible marker size without hiding dense-session error bars."""
    point_count = 0
    for trace in fig.data:
        if "markers" not in str(getattr(trace, "mode", "")):
            continue
        x_values = getattr(trace, "x", None)
        point_count += len(x_values) if x_values is not None else 0
    bands = (
        ((20, 10), (50, 8), (100, 6), (float("inf"), 5))
        if profile == "screen"
        else ((20, 12), (50, 10), (100, 8), (float("inf"), 6))
    )
    return next(size for limit, size in bands if point_count <= limit)


def _apply_results_plot_sizing(fig, theme, *, profile: Literal["screen", "export"] = "screen") -> None:
    """Apply independent screen/export typography and density-aware markers."""
    theme.apply_to_figure(fig, profile=f"{profile}_overview")
    marker_size = _results_marker_size(fig, profile=profile)
    for trace in fig.data:
        if "markers" in str(getattr(trace, "mode", "")):
            trace.marker.size = marker_size


def _build_results_export_figure(fig, theme, *, height: int):
    """Return a publication-styled copy; never mutate the on-screen figure."""
    import plotly.graph_objects as go

    export_fig = go.Figure(fig)
    export_fig.update_layout(height=height)
    _apply_results_plot_sizing(export_fig, theme, profile="export")
    return export_fig


def _render_results_figure(
    fig,
    *,
    theme,
    key: str,
    filename: str,
    display_height: int,
    export_width: int,
    export_height: int,
    publication=None,
) -> None:
    """Render the compact screen figure and an opt-in enlarged export figure."""
    from ui.config_plotly import get_plotly_config

    _apply_results_plot_sizing(fig, theme, profile="screen")
    screen_config = get_plotly_config(filename, width=export_width, height=export_height)
    screen_config["modeBarButtonsToRemove"] = [
        *screen_config["modeBarButtonsToRemove"],
        "toImage",
    ]
    st.plotly_chart(
        fig,
        width="stretch",
        height=display_height,
        key=key,
        config=screen_config,
    )
    if st.checkbox(
        "Enlarged export view",
        value=False,
        key=f"{key}_export_view",
        help="Open the independently styled publication figure; use its camera button for PNG or SVG.",
    ):
        export_fig = _build_results_export_figure(fig, theme, height=export_height)
        export_config = get_plotly_config(filename, width=export_width, height=export_height)
        if publication is not None:
            from ui.components.group_plot_publication import apply_publication_sizing
            apply_publication_sizing(export_fig, publication)
            export_config = get_plotly_config(filename, format=publication["format"],
                width=export_fig.layout.width, height=export_fig.layout.height)
            if publication["format"] == "png":
                export_config["toImageButtonOptions"]["scale"] = publication["dpi"] / 96
        st.plotly_chart(
            export_fig,
            width=export_fig.layout.width if publication else "stretch",
            height=export_fig.layout.height if publication else _RESULTS_ENLARGED_DISPLAY_HEIGHT,
            key=f"{key}_export",
            config=export_config,
        )


@dataclass(frozen=True)
class CertifiedOverlay:
    """Certified chart-overlay value using its displayed coverage convention."""

    value: float
    band_uncertainty: Optional[float]
    coverage_factor_k: float
    source: str
    derived: bool = False


def _samples_eligible_for_reported_uncertainty_bars(samples, state):
    """Return samples that may carry reported uncertainty-budget bars."""
    u_config = getattr(state, "uncertainty_config", None)
    processing_config = getattr(state, "processing_config", None)
    engine = ""
    if u_config is not None:
        resolver = getattr(u_config, "resolve_engine", None)
        element_symbol = getattr(state, "element_symbol", "") or ""
        if not element_symbol and getattr(state, "element_config", None) is not None:
            element_symbol = getattr(state.element_config, "symbol", "")
        engine = (
            resolver(element_symbol, processing_config=processing_config)
            if callable(resolver)
            else getattr(u_config, "engine", "")
        )

    # For SSB/delta, the reported expanded uncertainty is a per-unknown
    # quantity: the sample's corrected ratio / delta relative to the
    # bracketing standards. Standards are the calibration anchor and blanks
    # are not measurands, so neither carries a reported U — independent of
    # whether SSB or delta-only bracketing is active.
    if engine == "ssb_delta":
        return [sample for sample in samples if not sample.is_standard and not sample.is_blank]
    return list(samples)


def _eligible_reported_uncertainty_overrides(overrides, samples, state):
    """Restrict a runtime uncertainty map to samples eligible for reported-U bars.

    The shared map (also feeding the Summary Table) may include standards and
    blanks; this drops their keys so SSB/delta plots never draw a reported
    expanded uncertainty on the calibration anchor.
    """
    if not overrides:
        return overrides
    eligible = _samples_eligible_for_reported_uncertainty_bars(samples, state)
    eligible_ids = {(s.run_number, s.name) for s in eligible}
    return {
        key: budget
        for key, budget in overrides.items()
        if isinstance(key, tuple) and len(key) >= 2 and (key[0], key[1]) in eligible_ids
    }


def _build_results_runtime_delta_map(
    samples,
    ratio_names,
    *,
    cycle_ranges=None,
    filter_method="None",
    filter_threshold=DEFAULT_OUTLIER_THRESHOLD_SD,
):
    """Build runtime delta overrides once for a results render path."""
    state = get_state()
    if not _delta_enabled(state):
        return {}
    from domain.runtime_delta import build_runtime_delta_map

    return build_runtime_delta_map(
        samples,
        ratio_names,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
        processing_config=state.processing_config,
        element_config=state.element_config,
        calibration_freshness=current_pb_calibration_freshness(state),
    )


def _render_manual_certified_controls(
    key_prefix: str,
    *,
    checkbox_label: str,
    value_label: str,
    uncertainty_label: str,
    value_format: str,
    default_value: Optional[float] = None,
    default_uncertainty: Optional[float] = None,
    uncertainty_help: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Render manual certified/reference overlay controls.

    R1: When default_value / default_uncertainty are supplied (from the active CRM),
    pre-populate the number inputs so users don't need to look up the reference value.
    """
    show_overlay = st.checkbox(
        checkbox_label,
        value=False,
        key=f"{key_prefix}_show",
    )
    if not show_overlay:
        return None, None

    col_value, col_uncertainty = st.columns(2)
    with col_value:
        certified_value = st.number_input(
            value_label,
            value=default_value,
            format=value_format,
            key=f"{key_prefix}_value",
        )
    with col_uncertainty:
        certified_uncertainty = st.number_input(
            uncertainty_label,
            min_value=0.0,
            value=default_uncertainty,
            format=value_format,
            key=f"{key_prefix}_uncertainty",
            help=uncertainty_help,
        )

    value = float(certified_value) if certified_value is not None else None
    uncertainty = (
        float(certified_uncertainty)
        if certified_uncertainty is not None
        else None
    )
    return value, uncertainty


def render_results_tab() -> None:
    """Render the Results & Statistics tab."""
    state = get_state()

    if not state.has_result:
        if render_next_step(
            "Results are not available yet",
            "Configure the session and execute data reduction before reviewing results.",
            action_label="Open Session Configuration",
            action_key="results_open_session_configuration",
        ):
            request_main_section("Session Configuration")
            st.rerun()
        return

    result = state.result
    samples = result.samples
    from ui.components.sr_calibration_panel import render_sr_calibration_notice
    render_sr_calibration_notice(samples)


    if not samples:
        st.warning("No samples in processing result.")
        return

    calibration_freshness = current_pb_calibration_freshness(state)
    if calibration_freshness and calibration_freshness.get("status") == "stale":
        st.error(
            "Pb-standard calibration is stale for the current session. Final absolute ratios, "
            "calibrated delta, uncertainty and Monte Carlo results are withheld until data reduction is rerun."
        )
        return

    # Check if cycle selections, exclusions, types, or settings changed.
    if settings_changed_since_processing(state):
        st.warning(
            "Cycle selections, sample exclusions, types, or processing settings "
            "have changed since last processing. "
            "Go to **Session Configuration** and click **Execute Data Reduction** to update results."
        )

    # Ratio selector (shared across all sub-tabs) - use selected ratios from Session Configuration
    from ui.components.custom_ratio import get_selected_ratios

    selected_ratios = get_selected_ratios(state.samples)
    ratio_list = sorted(selected_ratios)
    if not ratio_list:
        st.warning("No ratio data available.")
        return

    # Default to primary ratio (has certified value / standard-sample bracketing / uncertainty)
    default_idx = 0
    if state.element_config and state.element_config.primary_ratio:
        primary = state.element_config.primary_ratio
        if primary in ratio_list:
            default_idx = ratio_list.index(primary)

    if st.session_state.get("results_ratio_select") not in ratio_list:
        st.session_state.pop("results_ratio_select", None)

    # Migrate stale per-section axis keys to the single shared key (one-time,
    # before the widget is instantiated so no widget state is overwritten).
    _SHARED_AXIS_KEY = RESULTS_AXIS_STATE_KEY
    for _legacy in (
        "overview_sample_axis",
        "results_summary_plot_axis",
        "results_sample_axis",
    ):
        if _legacy in st.session_state and _SHARED_AXIS_KEY not in st.session_state:
            st.session_state[_SHARED_AXIS_KEY] = st.session_state[_legacy]
        st.session_state.pop(_legacy, None)

    # The "Customize visible metrics" checkbox was removed in favour of an
    # always-on popover.  Its per-column keys used to be ignored while the
    # checkbox was off, so a stale unchecked set must not silently become
    # authoritative — clear both the flag and the column keys exactly once.
    if st.session_state.pop("results_table_customize_columns", None) is not None:
        for _stale in [k for k in st.session_state if k.startswith("results_table_cols_")]:
            st.session_state.pop(_stale, None)

    # Declared before the control row so the axis radio can resolve the active
    # subview against the same option list render_subview_nav validates against.
    sub_views = [
        "Overview",
        "Summary Table",
        "Precision Comparison",
        "Quality Control Summary",
    ]

    col_ratio, col_filter, col_axis = st.columns([2, 3, 2])
    with col_ratio:
        selected_ratio = st.selectbox(
            "Ratio",
            options=ratio_list,
            index=default_idx,
            format_func=format_isotope_label,
            key="results_ratio_select",
        )

    with col_filter:
        # Sample type filter
        available_types = sorted(set(s.sample_type.upper() for s in samples))
        selected_types = sticky_type_multiselect(
            "Sample Types",
            options=available_types,
            key="sticky_results_type_filter",
        )

    # "Sample axis" only affects Overview and the Summary Table's plot section;
    # hide it in the other subviews where it would be a live control with no
    # effect.  The persisted subview is normalized the same way
    # render_subview_nav normalizes it below, so a stale or evicted value cannot
    # hide the radio on a pass where Overview actually renders.
    _AXIS_VIEWS = ("Overview", "Summary Table")
    _active_view = resolve_subview_selection(
        key="results_subview_nav", options=sub_views, default="Overview",
    )
    with col_axis:
        if _active_view in _AXIS_VIEWS:
            if _SHARED_AXIS_KEY not in st.session_state:
                st.session_state[_SHARED_AXIS_KEY] = _default_results_sample_axis(len(samples))
            axis_choice = st.radio(
                "Sample axis",
                options=["Names", "Run order"],
                horizontal=True,
                key=_SHARED_AXIS_KEY,
            )
        else:
            # Unread for these subviews — only the Overview and Summary Table
            # renderers accept `axis_choice` — but kept bound so the renderer
            # lambdas below always close over a defined name.
            axis_choice = st.session_state.get(_SHARED_AXIS_KEY, "Names")

    filtered_samples = [
        s for s in samples
        if s.sample_type.upper() in selected_types
        and not s.metadata.get("excluded", False)
    ]

    if not filtered_samples:
        st.warning("No samples match the selected filters.")
        return

    # Compact metadata line — combines filter caption and layer description.
    layer_label = _get_results_layer_label(filtered_samples, selected_ratio, state)
    meta_items = [("Layer", layer_label)]
    filter_caption = get_filtered_sample_caption(
        showing=len(filtered_samples),
        total=len(samples),
    )
    if filter_caption is not None:
        meta_items.append(("Samples shown", filter_caption))
    render_compact_metadata(meta_items)

    # Get cycle ranges from Inspector tab (session state)
    cycle_ranges = get_cycle_ranges(state)

    # Get filter settings for re-filtering within cycle ranges
    filter_method = state.processing_config.filter_method if state.processing_config else "None"
    filter_threshold = (
        state.processing_config.get_active_filter_threshold()
        if state.processing_config
        else 2.0
    )

    # Lazy sub-view selector (`sub_views` is declared above the control row)
    selected_view = render_subview_nav(
        "Results View",
        sub_views,
        key="results_subview_nav",
        default="Overview",
        current=resolve_subview_selection(
            key="results_subview_nav", options=sub_views, default="Overview",
        ),
    )

    # Precompute once per rerun for overview/rsd views.
    ratio_metrics = None
    if selected_view in ("Overview", "Precision Comparison"):
        from ui.components.stats_charts import build_ratio_metrics

        ratio_metrics = build_ratio_metrics(
            filtered_samples,
            selected_ratio,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
        )

    view_renderers: Dict[str, Callable[[], None]] = {
        "Overview": lambda: _render_overview(
            filtered_samples, selected_ratio,
            cycle_ranges, filter_method, filter_threshold,
            ratio_metrics=ratio_metrics,
            all_session_samples=result.samples,
            drift_fit_info=result.quality_metrics.get("drift_fit_info"),
            axis_choice=axis_choice,
        ),
        "Summary Table": lambda: _render_summary_table(
            result, filtered_samples, ratio_list, selected_ratio, selected_types,
            cycle_ranges, filter_method, filter_threshold,
            axis_choice=axis_choice,
        ),
        "Precision Comparison": lambda: _render_rsd_comparison(
            filtered_samples, selected_ratio,
            cycle_ranges, filter_method, filter_threshold,
            ratio_metrics=ratio_metrics,
        ),
        "Quality Control Summary": lambda: _render_qc_dashboard(
            result,
            selected_ratio,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
        ),
    }
    _render_selected_results_view(selected_view, view_renderers)


def _render_selected_results_view(
    selected_view: str,
    view_renderers: Dict[str, Callable[[], None]],
) -> None:
    """Render only the selected results sub-view."""
    renderer = view_renderers.get(selected_view)
    if renderer is not None:
        renderer()


def _get_certified_value(ratio_name: str) -> Optional[float]:
    """Get certified value for a ratio, respecting user's CRM selection."""
    overlay = _get_certified_overlay(ratio_name)
    return overlay.value if overlay is not None else None


def _get_certified_overlay(ratio_name: str) -> Optional[CertifiedOverlay]:
    """Return the active certified value and its Results-plot band."""
    from config.reference_materials import (
        derive_certified_value,
        get_crm_ratios,
    )

    state = get_state()
    if not state.element_config:
        return None

    rm_name = None
    if state.processing_config and state.processing_config.reference_material:
        rm_name = state.processing_config.reference_material
        direct_crm_ratios = get_crm_ratios(
            state.element_config.symbol,
            rm_name,
            derive=False,
        )
        crm_ratios = get_crm_ratios(
            state.element_config.symbol,
            rm_name,
            derive=True,
        )
        if ratio_name in crm_ratios:
            value, stated_uncertainty, stated_k = crm_ratios[ratio_name]
            is_derived = ratio_name not in direct_crm_ratios
            band_uncertainty, coverage_factor_k = _overlay_band_values(
                stated_uncertainty,
                stated_k,
                derived=is_derived,
            )
            return CertifiedOverlay(
                value=float(value),
                band_uncertainty=band_uncertainty,
                coverage_factor_k=coverage_factor_k,
                source=rm_name,
                derived=is_derived,
            )

    # Fall back to element config defaults
    if state.element_config.certified_values:
        cv = state.element_config.certified_values.get(ratio_name)
        if cv:
            band_uncertainty, coverage_factor_k = _overlay_band_values(
                getattr(cv, "uncertainty", None),
                getattr(cv, "k", 1.0),
                derived=False,
            )
            return CertifiedOverlay(
                value=float(cv.value),
                band_uncertainty=band_uncertainty,
                coverage_factor_k=coverage_factor_k,
                source=str(getattr(cv, "source", "") or rm_name or "Element config"),
                derived=False,
            )

    # Fall back to derived-ratio solver
    derived = derive_certified_value(
        state.element_config.symbol,
        ratio_name,
        crm_name=rm_name,
    )
    if derived is not None:
        band_uncertainty, coverage_factor_k = _overlay_band_values(
            derived[1],
            1.0,
            derived=True,
        )
        return CertifiedOverlay(
            value=float(derived[0]),
            band_uncertainty=band_uncertainty,
            coverage_factor_k=coverage_factor_k,
            source=rm_name or "Derived CRM ratio",
            derived=True,
        )

    return None


def _overlay_band_values(
    uncertainty: object,
    coverage_factor: object,
    *,
    derived: bool,
) -> Tuple[Optional[float], float]:
    """Validate a plot-band uncertainty and apply the derived-ratio convention."""
    try:
        uncertainty_value = float(uncertainty)
        k_value = float(coverage_factor)
    except (TypeError, ValueError, OverflowError):
        return None, 2.0
    if (
        not np.isfinite(uncertainty_value)
        or uncertainty_value < 0
        or not np.isfinite(k_value)
        or k_value <= 0
    ):
        return None, 2.0
    if derived:
        # The derivation solver returns propagated standard uncertainty (k=1).
        # Expand it only at the Results-overlay boundary.
        return uncertainty_value * 2.0, 2.0
    return uncertainty_value, k_value


def _certified_uncertainty_label(
    overlay: Optional[CertifiedOverlay],
    *,
    unit: Optional[str] = None,
) -> str:
    """Return a truthful label for a certified comparison-band input."""
    k_value = overlay.coverage_factor_k if overlay is not None else 2.0
    quantity = (
        "Standard uncertainty u"
        if np.isclose(k_value, 1.0)
        else "Expanded uncertainty U"
    )
    suffix = f", {unit}" if unit else ""
    return f"{quantity} (k={k_value:g}{suffix})"


def _certified_overlay_help(overlay: Optional[CertifiedOverlay]) -> str:
    """Explain the coverage convention for a certified comparison band."""
    if overlay is None:
        return "Enter an expanded uncertainty U at k=2."
    if overlay.derived:
        return (
            "Derived certified-ratio uncertainty for "
            f"{overlay.source}, propagated at k=1 and expanded to k=2 for this band."
        )
    return (
        f"Uncertainty retained from the certificate for {overlay.source} "
        f"at its stated coverage factor k={overlay.coverage_factor_k:g}."
    )


def _get_results_layer_label(samples, ratio_name: str, state) -> str:
    """Describe which correction layer is shown for the selected ratio."""
    del state  # kept in the signature for stable callers
    selected = []
    for sample in samples:
        layer = select_best_ratio_layer(sample, ratio_name)
        if layer is not None:
            selected.append((sample, layer))

    if not selected:
        return f"No ratio layer is available for `{ratio_name}`."

    counts = Counter(layer.key for _sample, layer in selected)
    labels = {layer.key: layer.label for _sample, layer in selected}

    # Display-level label refinement: when the "corrected" layer is just a
    # copy of "blank_corrected" (common for elements like B that have no
    # interference correction), use the more descriptive label for the user.
    if "corrected" in labels:
        from domain.layers import cycle_data_nearly_equal
        all_identical = True
        for sample, layer in selected:
            if layer.key != "corrected":
                continue
            bc = sample.blank_corrected_ratios.get(ratio_name) if sample.blank_corrected_ratios else None
            if bc is None or not cycle_data_nearly_equal(layer.data, bc):
                all_identical = False
                break
        if all_identical:
            labels["corrected"] = "Blank-corrected"

    if len(counts) == 1:
        key = next(iter(counts))
        caption = f"Showing {labels[key]} ratios for `{ratio_name}`."
    else:
        total = len(selected)
        details = "; ".join(
            f"{labels[key]} {count}/{total}"
            for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )
        caption = f"Showing mixed ratio layers for `{ratio_name}`: {details}."

    if "ssb" in counts and any(key != "ssb" for key in counts):
        caption += (
            " This combines certified-scale SSB points with measured-scale points; "
            "do not interpret the overview as one homogeneous ratio scale."
        )
    return caption




def _render_overview(
    samples, ratio_name,
    cycle_ranges=None, filter_method="None", filter_threshold=DEFAULT_OUTLIER_THRESHOLD_SD,
    ratio_metrics=None,
    all_session_samples=None,
    drift_fit_info=None,
    axis_choice: str = "Names",
) -> None:
    """Render the reported output first, with optional aligned diagnostics."""
    state = get_state()
    x_axis_mode = "run_order" if axis_choice == "Run order" else "category"
    common = {
        "all_session_samples": all_session_samples or samples,
        "cycle_ranges": cycle_ranges,
        "filter_method": filter_method,
        "filter_threshold": filter_threshold,
        "drift_fit_info": drift_fit_info,
    }

    if _delta_enabled(state):
        _render_delta_overview(samples, ratio_name, x_axis_mode=x_axis_mode, **common)
        if st.checkbox(
            "Show aligned absolute-ratio comparison",
            value=False,
            key="sticky_results_show_ratio_comparison",
            help="Uses the same samples, order and filters as the delta view; each chart keeps its own labelled axis and unit.",
        ):
            st.divider()
            _render_ratio_overview(
                samples, ratio_name, cycle_ranges, filter_method, filter_threshold,
                ratio_metrics=ratio_metrics,
                all_session_samples=all_session_samples,
                drift_fit_info=drift_fit_info,
                axis_choice=axis_choice,
            )
    else:
        _render_ratio_overview(
            samples, ratio_name, cycle_ranges, filter_method, filter_threshold,
            ratio_metrics=ratio_metrics,
            all_session_samples=all_session_samples,
            drift_fit_info=drift_fit_info,
            axis_choice=axis_choice,
        )

    if st.checkbox(
        "Show intensity diagnostics",
        value=False,
        key="sticky_results_show_intensity_diagnostics",
    ):
        st.divider()
        st.subheader("Intensity diagnostics")
        st.caption("Mean intensity per sample for each isotope, using the active cycle window and re-filtering.")
        _render_intensity_overview(
            samples,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
            axis_choice=axis_choice,
        )


def _render_ratio_overview(
    samples, ratio_name,
    cycle_ranges=None, filter_method="None", filter_threshold=DEFAULT_OUTLIER_THRESHOLD_SD,
    ratio_metrics=None,
    all_session_samples=None,
    drift_fit_info=None,
    axis_choice: str = "Names",
) -> None:
    """Render the absolute-ratio overview and its own controls."""
    from ui.runtime_budget_cache import build_cached_runtime_uncertainty_map
    from ui.components.stats_charts import build_ratio_metrics, create_ratio_overview_chart
    from ui.theme import get_theme

    st.subheader("Ratio Overview")
    st.caption("Mean ratio values for the active sample-type and ratio filters.")

    col_opt1, col_opt2 = st.columns([2, 4])
    with col_opt1:
        error_choice = st.radio(
            "Error bars",
            options=["Off", "2 SE", "2 SD", "Reported U (k = actual)"],
            index=1,
            horizontal=True,
            key="overview_error_mode",
        )
    with col_opt2:
        crm_overlay = _get_certified_overlay(ratio_name)
        certified_value, certified_uncertainty = _render_manual_certified_controls(
            f"overview_certified_ratio_v2_{ratio_name}",
            checkbox_label="Show certified value",
            value_label="Certified value",
            uncertainty_label=_certified_uncertainty_label(crm_overlay),
            value_format="%.6f",
            default_value=crm_overlay.value if crm_overlay else None,
            default_uncertainty=(
                crm_overlay.band_uncertainty if crm_overlay else None
            ),
            uncertainty_help=_certified_overlay_help(crm_overlay),
        )

    error_mode_by_label = {
        "2 SE": "2SE",
        "2 SD": "2SD",
        "Reported U (k = actual)": "U_reported",
    }
    show_errors = error_choice != "Off"
    error_mode = error_mode_by_label.get(error_choice, "2SE")
    x_axis_mode = "run_order" if axis_choice == "Run order" else "category"
    uncertainty_overrides = None
    chart_ratio_metrics = ratio_metrics

    if error_mode == "U_reported":
        state = get_state()
        if state.element_config and state.processing_config:
            uncertainty_samples = _samples_eligible_for_reported_uncertainty_bars(samples, state)
            uncertainty_overrides = build_cached_runtime_uncertainty_map(
                uncertainty_samples,
                [ratio_name],
                element_config=state.element_config,
                processing_config=state.processing_config,
                uncertainty_config=state.uncertainty_config,
                all_session_samples=all_session_samples or samples,
                cycle_ranges=cycle_ranges,
                filter_method=filter_method,
                filter_threshold=filter_threshold,
                drift_fit_info=drift_fit_info,
                custom_contributor_library=state.custom_contributor_library,
                profile_defaults=getattr(state, "uncertainty_profile_defaults", None),
            )
            chart_ratio_metrics = build_ratio_metrics(
                samples,
                ratio_name,
                cycle_ranges=cycle_ranges,
                filter_method=filter_method,
                filter_threshold=filter_threshold,
                uncertainty_overrides=uncertainty_overrides,
            )

    fig = create_ratio_overview_chart(
        samples,
        ratio_name,
        show_error_bars=show_errors,
        error_mode=error_mode,
        show_certified=certified_value,
        certified_uncertainty=certified_uncertainty,
        certified_coverage_factor=(
            crm_overlay.coverage_factor_k if crm_overlay else 2.0
        ),
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
        ratio_metrics=chart_ratio_metrics,
        uncertainty_overrides=uncertainty_overrides,
        height=_RESULTS_OVERVIEW_DISPLAY_HEIGHT,
        x_axis_mode=x_axis_mode,
    )
    _render_results_figure(
        fig,
        theme=get_theme(),
        key=f"results_overview_chart_{ratio_name}",
        filename=f"traceiso_{ratio_name.replace('/', '-')}_overview",
        display_height=_RESULTS_OVERVIEW_DISPLAY_HEIGHT,
        export_width=_OVERVIEW_PLOT_EXPORT_WIDTH,
        export_height=_OVERVIEW_PLOT_EXPORT_HEIGHT,
    )

def _delta_enabled(state) -> bool:
    """Route-aware delta request; calibrated Pb never inherits the legacy flag."""
    if not state.element_config or not state.processing_config or not state.element_config.supports_delta:
        return False
    from domain.pb_standard_calibration import calibration_requested, calibrated_delta_requested
    symbol = getattr(state.element_config, "symbol", "")
    if symbol and calibration_requested(symbol, state.processing_config):
        return calibrated_delta_requested(symbol, state.processing_config)
    return bool(state.processing_config.enable_delta)


def _delta_reference_name(state) -> str:
    """Return the reference label for delta chart titles.

    SSB+CRM active → the CRM name (e.g. 'BAM-I012').
    Delta-only / no CRM → 'Bracketing STDs'.
    """
    from domain.pb_standard_calibration import calibration_requested
    if (
        state.element_config and state.processing_config
        and calibration_requested(state.element_config.symbol, state.processing_config)
    ):
        return state.processing_config.pb_standard_calibration.reference_material
    if (
        state.processing_config
        and state.processing_config.enable_ssb
        and state.element_config
    ):
        crm_name = (
            state.processing_config.reference_material
            or state.element_config.reference_material
        )
        if crm_name:
            return crm_name
    return "Bracketing STDs"


def _render_delta_overview(
    samples,
    ratio_name,
    *,
    all_session_samples=None,
    cycle_ranges=None,
    filter_method="None",
    filter_threshold=DEFAULT_OUTLIER_THRESHOLD_SD,
    drift_fit_info=None,
    x_axis_mode="category",
) -> None:
    """Render the cross-sample delta overview below the Ratio Overview."""
    state = get_state()
    if not _delta_enabled(state):
        return

    from ui.runtime_budget_cache import build_cached_runtime_uncertainty_map
    from ui.components.stats_charts import create_delta_overview_chart
    from ui.theme import get_theme

    delta_overrides = _build_results_runtime_delta_map(
        all_session_samples or samples,
        [ratio_name],
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
    )

    reference_name = _delta_reference_name(state)
    delta_heading = format_delta_label(ratio_name, reference="")
    if reference_name == "Bracketing STDs":
        st.subheader(f"{delta_heading} obtained by standard bracketing")
    else:
        st.subheader(format_delta_label(ratio_name, reference=reference_name))
    st.caption(
        "Mean delta values (‰) per sample, using the active cycle window and "
        "re-filtering. “Reported U” is the full Engine B GUM expanded "
        "uncertainty of the delta."
    )

    delta_col_opt1, delta_col_opt2, delta_col_opt3 = st.columns([2, 1, 3])
    with delta_col_opt1:
        delta_error_choice = st.radio(
            "Error bars",
            options=["Off", "2 SE", "2 SD", "Reported U (k = actual)"],
            index=1,
            horizontal=True,
            key="overview_delta_error_mode",
        )
    with delta_col_opt2:
        show_delta_zero_line = st.checkbox(
            "Show 0 permil line",
            value=True,
            key="overview_delta_zero_line",
        )
    with delta_col_opt3:
        certified_delta, certified_delta_uncertainty = _render_manual_certified_controls(
            f"overview_certified_delta_v2_{ratio_name}",
            checkbox_label="Show certified delta",
            value_label="Certified delta (permil)",
            uncertainty_label=_certified_uncertainty_label(None, unit="permil"),
            value_format="%.4f",
            uncertainty_help=_certified_overlay_help(None),
        )
    delta_error_by_label = {
        "2 SE": "2SE",
        "2 SD": "2SD",
        "Reported U (k = actual)": "U_reported",
    }
    delta_error_mode = delta_error_by_label.get(delta_error_choice, "2SE")

    delta_uncertainty_overrides = None
    if delta_error_mode == "U_reported":
        uncertainty_samples = _samples_eligible_for_reported_uncertainty_bars(samples, state)
        delta_uncertainty_overrides = build_cached_runtime_uncertainty_map(
            uncertainty_samples,
            [ratio_name],
            element_config=state.element_config,
            processing_config=state.processing_config,
            uncertainty_config=state.uncertainty_config,
            all_session_samples=all_session_samples or samples,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
            drift_fit_info=drift_fit_info,
            custom_contributor_library=state.custom_contributor_library,
            profile_defaults=getattr(state, "uncertainty_profile_defaults", None),
        )

    fig = create_delta_overview_chart(
        samples,
        ratio_name,
        show_error_bars=delta_error_choice != "Off",
        error_mode=delta_error_mode,
        show_zero_line=show_delta_zero_line,
        certified_delta=certified_delta,
        certified_delta_uncertainty=certified_delta_uncertainty,
        certified_coverage_factor=2.0,
        cycle_ranges=cycle_ranges,
        delta_overrides=delta_overrides,
        uncertainty_overrides=delta_uncertainty_overrides,
        height=_RESULTS_OVERVIEW_DISPLAY_HEIGHT,
        reference_name=reference_name,
        x_axis_mode=x_axis_mode,
    )
    _render_results_figure(
        fig,
        theme=get_theme(),
        key=f"results_overview_delta_chart_{ratio_name}",
        filename=f"traceiso_{ratio_name.replace('/', '-')}_delta",
        display_height=_RESULTS_OVERVIEW_DISPLAY_HEIGHT,
        export_width=_OVERVIEW_PLOT_EXPORT_WIDTH,
        export_height=_OVERVIEW_PLOT_EXPORT_HEIGHT,
    )


def _render_intensity_overview(
    samples,
    cycle_ranges=None,
    filter_method="None",
    filter_threshold=DEFAULT_OUTLIER_THRESHOLD_SD,
    axis_choice: str = "Names",
) -> None:
    """Render mean intensity scatter chart per isotope."""
    import plotly.graph_objects as go
    from ui.theme import get_theme
    from ui.utils import format_name

    all_isotopes = set()
    for s in samples:
        if s.intensities:
            all_isotopes.update(s.intensities.keys())
        if s.corrected_intensities:
            all_isotopes.update(s.corrected_intensities.keys())

    if not all_isotopes:
        st.caption("No intensity data available.")
        return

    isotope_list = sorted(all_isotopes)
    selected_isotope = st.selectbox(
        "Isotope", options=isotope_list, index=0,
        format_func=format_isotope_label, key="overview_isotope_select",
    )

    theme = get_theme()
    palette = theme.palette

    # Group by type, sorted by run number
    sorted_samples = sorted(
        [s for s in samples if not s.metadata.get("excluded", False)],
        key=lambda s: s.run_number,
    )

    type_data = {}
    category_order = []
    for s in sorted_samples:
        # Use corrected if available, else raw
        src = s.corrected_intensities if s.corrected_intensities else s.intensities
        if not src or selected_isotope not in src:
            continue
        cd = src[selected_isotope]
        valid = get_filtered_values(
            cd.values,
            cd.mask,
            s.name,
            cycle_ranges=cycle_ranges,
            sample_key=get_sample_state_key(s),
            filter_method=filter_method,
            filter_threshold=filter_threshold,
        )
        if len(valid) == 0:
            continue
        stype = s.sample_type.upper()
        if stype not in type_data:
            type_data[stype] = {"names": [], "means": [], "runs": [], "n": []}
        type_data[stype]["names"].append(s.name)
        type_data[stype]["means"].append(float(np.nanmean(valid)))
        type_data[stype]["runs"].append(s.run_number)
        type_data[stype]["n"].append(int(len(valid)))
        category_order.append(s.name)

    has_blanks = "BLK" in type_data
    has_non_blanks = any(stype != "BLK" for stype in type_data)
    separate_blank_panel = has_blanks and has_non_blanks
    if separate_blank_panel:
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
        )
    else:
        fig = go.Figure()
    use_run_order = axis_choice == "Run order"
    for stype, data in type_data.items():
        customdata = (
            np.column_stack([data["names"], data["n"]])
            if use_run_order
            else np.column_stack([data["runs"], data["n"]])
        )
        hover_heading = (
            "<b>%{customdata[0]}</b><br>Run: %{x}<br>"
            if use_run_order
            else "<b>%{x}</b><br>Run: %{customdata[0]}<br>"
        )
        trace = go.Scatter(
            x=data["runs"] if use_run_order else data["names"], y=data["means"],
            mode="markers",
            marker=dict(
                color=theme.sample_color(stype),
                size=10,
                symbol=theme.sample_symbol(stype),
                line=dict(color=palette.figure_axis, width=0.8),
            ),
            name=stype,
            customdata=customdata,
            hovertemplate=(
                hover_heading
                + "Mean: %{y:.6g} V<br>"
                "n: %{customdata[1]}<br>"
                f"Type: {stype}<extra></extra>"
            ),
        )
        if separate_blank_panel:
            fig.add_trace(trace, row=2 if stype == "BLK" else 1, col=1)
        else:
            fig.add_trace(trace)

    fig.update_layout(
        title=dict(text=format_name(selected_isotope), font=dict(size=PLOTLY_TITLE_FONT_SIZE)),
        yaxis=dict(
            title=dict(text="Blank Intensity (V)" if has_blanks and not has_non_blanks else "Intensity (V)"),
            showgrid=False,
            zeroline=False,
        ),
        height=_RESULTS_OVERVIEW_DISPLAY_HEIGHT,
        margin=dict(l=80, r=30, t=70, b=75 if use_run_order else 110),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
    )
    if separate_blank_panel:
        fig.update_yaxes(title_text="Blank Intensity (V)", row=2, col=1)
    fig.update_xaxes(
        title_text="Run order" if use_run_order else "",
        type="linear" if use_run_order else "category",
        categoryorder=None if use_run_order else "array",
        categoryarray=None if use_run_order else category_order,
        tickangle=0 if use_run_order else -45,
        showgrid=False,
        zeroline=False,
    )
    _render_results_figure(
        fig,
        theme=theme,
        key=f"results_intensity_overview_{selected_isotope}",
        filename=f"traceiso_{selected_isotope.replace('/', '-')}_intensity_overview",
        display_height=_RESULTS_OVERVIEW_DISPLAY_HEIGHT,
        export_width=_OVERVIEW_PLOT_EXPORT_WIDTH,
        export_height=_OVERVIEW_PLOT_EXPORT_HEIGHT,
    )


def _render_summary_table(
    result, filtered_samples, available_ratios, ratio_name, selected_types,
    cycle_ranges=None, filter_method="None", filter_threshold=DEFAULT_OUTLIER_THRESHOLD_SD,
    axis_choice: str = "Names",
) -> None:
    """Render the configurable summary table."""
    from ui.components.summary_table import (
        infer_available_columns,
        render_column_selector,
        render_summary_table,
    )
    from ui.runtime_budget_cache import build_cached_runtime_uncertainty_map

    st.markdown(
        """
        <style>
        .results-summary-table-title {
            color: var(--t-ink, inherit);
            font-size: var(--t-size-section, 18px);
            font-weight: 700;
            line-height: 1.25;
            margin: 0.2rem 0 0.45rem 0;
        }
        </style>
        <div class="results-summary-table-title">Summary Table</div>
        """,
        unsafe_allow_html=True,
    )

    selected_table_ratios = st.multiselect(
        "Ratios in table",
        options=available_ratios,
        default=[ratio_name] if ratio_name in available_ratios else available_ratios[:1],
        key="results_table_ratio_select",
    )
    if not selected_table_ratios:
        st.warning("Select at least one ratio for the summary table.")
        return

    state = get_state()
    uncertainty_overrides = None
    delta_overrides = None
    if state.element_config and state.processing_config:
        uncertainty_overrides = build_cached_runtime_uncertainty_map(
            filtered_samples,
            selected_table_ratios,
            element_config=state.element_config,
            processing_config=state.processing_config,
            uncertainty_config=state.uncertainty_config,
            all_session_samples=result.samples,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
            drift_fit_info=result.quality_metrics.get("drift_fit_info"),
            custom_contributor_library=state.custom_contributor_library,
            profile_defaults=getattr(state, "uncertainty_profile_defaults", None),
        )
        if _delta_enabled(state):
            delta_overrides = _build_results_runtime_delta_map(
                result.samples,
                selected_table_ratios,
                cycle_ranges=cycle_ranges,
                filter_method=filter_method,
                filter_threshold=filter_threshold,
            )

    available_columns = infer_available_columns(
        filtered_samples,
        selected_table_ratios,
        uncertainty_overrides=uncertainty_overrides,
        delta_overrides=delta_overrides,
    )
    # Column configuration popover: st.popover always executes its body so
    # render_column_selector always populates `columns` via its widget key.
    with st.popover("Configure columns \u2026", width="content"):
        columns = render_column_selector(
            default_columns=None,
            key="results_table_cols",
            available_columns=available_columns,
        )

    summary_df = render_summary_table(
        result,
        ratio_names=selected_table_ratios,
        columns=columns,
        filter_types=selected_types,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
        key="results_summary",
        uncertainty_overrides=uncertainty_overrides,
        delta_overrides=delta_overrides,
    )

    visible_names = summary_df["Sample"].tolist() if not summary_df.empty and "Sample" in summary_df.columns else []
    visible_samples = [s for s in filtered_samples if s.name in visible_names]
    if visible_samples:
        st.divider()
        _render_summary_plot_section(
            visible_samples,
            selected_table_ratios,
            ratio_name,
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
            uncertainty_overrides=uncertainty_overrides,
            delta_overrides=delta_overrides,
            axis_choice=axis_choice,
        )


def _render_rsd_comparison(
    samples, ratio_name,
    cycle_ranges=None, filter_method="None", filter_threshold=DEFAULT_OUTLIER_THRESHOLD_SD,
    ratio_metrics=None,
) -> None:
    """Render RSD comparison chart."""
    from ui.components.stats_charts import create_rsd_comparison_chart
    from ui.theme import get_theme

    st.subheader("Precision Comparison")

    fkw = dict(
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
        ratio_metrics=ratio_metrics,
        height=500,
    )
    fig = create_rsd_comparison_chart(samples, ratio_name, **fkw)
    _render_results_figure(
        fig,
        theme=get_theme(),
        key=f"dist_rsd_{ratio_name}",
        filename=f"traceiso_{ratio_name.replace('/', '-')}_precision",
        display_height=int(fig.layout.height or _RESULTS_OVERVIEW_DISPLAY_HEIGHT),
        export_width=_OVERVIEW_PLOT_EXPORT_WIDTH,
        export_height=_OVERVIEW_PLOT_EXPORT_HEIGHT,
    )


def _render_summary_plot_section(
    samples,
    ratio_names,
    default_ratio,
    cycle_ranges=None,
    filter_method="None",
    filter_threshold=DEFAULT_OUTLIER_THRESHOLD_SD,
    uncertainty_overrides=None,
    delta_overrides=None,
    axis_choice: str = "Names",
) -> None:
    """Render a selectable comparison plot below the summary table."""
    import plotly.graph_objects as go
    from ui.components.stats_charts import build_ratio_metrics, create_ratio_overview_chart
    from ui.theme import get_theme

    from ui.components.sample_plot_groups import (
        apply_sample_plot_groups, render_sample_plot_groups,
    )

    from ui.components.group_plot_publication import (
        render_group_publication_controls, add_group_references, apply_group_axes,
    )

    st.subheader("Selected Samples Plot")
    theme = get_theme()
    palette = theme.palette

    if st.session_state.get("results_summary_plot_ratio") not in ratio_names:
        st.session_state.pop("results_summary_plot_ratio", None)

    plot_ratio = st.selectbox(
        "Plot ratio",
        options=ratio_names,
        index=ratio_names.index(default_ratio) if default_ratio in ratio_names else 0,
        format_func=format_isotope_label,
        key="results_summary_plot_ratio",
    )

    error_mode_by_label = {
        "2 SE": "2SE",
        "2 SD": "2SD",
        "Reported U (k = actual)": "U_reported",
    }

    available_types = sorted({s.sample_type.upper() for s in samples})
    col_type, col_error, col_mean, col_legend, col_certified = st.columns(
        [3, 2, 1, 1.5, 3]
    )
    with col_type:
        plot_types = st.multiselect(
            "Sample types in plot",
            options=available_types,
            default=available_types,
            key="results_summary_plot_types",
        )
    with col_error:
        error_choice = st.radio(
            "Error bars",
            options=list(error_mode_by_label.keys()),
            index=0,
            horizontal=True,
            key="results_summary_plot_error_mode",
        )
    with col_mean:
        show_mean_line = st.checkbox(
            "Add mean line",
            value=False,
            key="results_summary_plot_mean_line",
        )
    with col_legend:
        show_sample_type_legend = st.checkbox(
            "Sample types in legend",
            value=True,
            key="results_summary_plot_sample_type_legend",
            help="Show or hide sample-type legend entries in the original plot without hiding data.",
        )
    with col_certified:
        crm_overlay = _get_certified_overlay(plot_ratio)
        certified_value, certified_uncertainty = _render_manual_certified_controls(
            f"summary_certified_ratio_v2_{plot_ratio}",
            checkbox_label="Show certified value",
            value_label="Certified value",
            uncertainty_label=_certified_uncertainty_label(crm_overlay),
            value_format="%.6f",
            default_value=crm_overlay.value if crm_overlay else None,
            default_uncertainty=(
                crm_overlay.band_uncertainty if crm_overlay else None
            ),
            uncertainty_help=_certified_overlay_help(crm_overlay),
        )

    candidate_samples = [
        s for s in samples
        if s.sample_type.upper() in plot_types
    ]
    from ui.components.stats_charts import _sample_display_labels

    candidate_labels = _sample_display_labels(candidate_samples)
    candidate_names = list(candidate_labels)
    selection_key = "results_summary_plot_samples_v2"
    if selection_key in st.session_state:
        st.session_state[selection_key] = [
            sid for sid in st.session_state[selection_key] if sid in candidate_labels
        ]
    selected_names = st.multiselect(
        "Samples to include",
        options=candidate_names,
        default=candidate_names,
        key=selection_key,
        format_func=candidate_labels.get,
    )
    plot_samples = [s for s in candidate_samples if s.observation_id in selected_names]

    if not plot_samples:
        st.info("Select at least one sample to plot.")
        return

    # The shared uncertainty map (built for the Summary Table) may include
    # standards/blanks. Reported-U bars are a per-unknown quantity, so drop
    # ineligible samples before they reach the plot.
    plot_overrides = _eligible_reported_uncertainty_overrides(
        uncertainty_overrides, plot_samples, get_state(),
    )

    ratio_metrics = build_ratio_metrics(
        plot_samples,
        plot_ratio,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
        uncertainty_overrides=plot_overrides,
    )
    def render_ratio(plot_samples, ratio_metrics, *, grouped=False):
        fig = create_ratio_overview_chart(
            plot_samples,
            plot_ratio,
            show_error_bars=True,
            error_mode=error_mode_by_label[error_choice],
            show_certified=None if grouped else certified_value,
            certified_uncertainty=certified_uncertainty,
            certified_coverage_factor=(
                crm_overlay.coverage_factor_k if crm_overlay else 2.0
            ),
            cycle_ranges=cycle_ranges,
            filter_method=filter_method,
            filter_threshold=filter_threshold,
            ratio_metrics=ratio_metrics,
            uncertainty_overrides=plot_overrides,
            height=_RESULTS_SELECTED_DISPLAY_HEIGHT,
            x_axis_mode=(
                "run_order" if axis_choice == "Run order" else "category"
            ),
        )
        for trace in fig.data:
            if trace.name in plot_types:
                trace.showlegend = show_sample_type_legend

        finite_metrics = [
            metric for metric in ratio_metrics.values() if np.isfinite(metric["mean"])
        ]
        primary_metrics = [
            metric for metric in finite_metrics if metric["sample_type"] != "BLK"
        ]
        # Blank ratios use a separate y-axis and scale, so never mix them into
        # primary-axis summary statistics when non-blank samples are present.
        means = [
            metric["mean"] for metric in (primary_metrics or finite_metrics)
        ]
        if means:
            mean_val = float(np.mean(means))
            std_val = float(np.std(means, ddof=1)) if len(means) > 1 else np.nan
            if show_mean_line:
                ordered_x = [
                    (
                        metric["run_number"]
                        if axis_choice == "Run order"
                        else metric.get("display_label", metric["sample_name"])
                    )
                    for metric in sorted(
                        ratio_metrics.values(),
                        key=lambda metric: metric["run_number"],
                    )
                ]
                fig.add_trace(
                    go.Scatter(
                        x=ordered_x,
                        y=[mean_val] * len(ordered_x),
                        mode="lines",
                        line=dict(color=palette.guide_bounds, dash="dash", width=1.2),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.02,
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                text=f"Mean \u00b1 2SD: {mean_val:.5f} \u00b1 {2 * std_val:.5f}",
                font=dict(size=PLOTLY_ANNOTATION_FONT_SIZE, color=palette.annotation_text_color),
                bgcolor=palette.annotation_bg,
                bordercolor=palette.annotation_border,
                borderwidth=1,
            )

        if grouped:
            apply_sample_plot_groups(fig, groups)
            add_group_references(fig, groups, plot_ratio, "ratio")
            apply_group_axes(fig, publication, plot_ratio, "ratio")
        suffix = "_groups" if grouped else ""
        _render_results_figure(
            fig,
            theme=theme,
            key=f"results_summary_plot_{plot_ratio}{suffix}",
            filename=f"traceiso_{plot_ratio.replace('/', '-')}_summary{suffix}",
            display_height=_RESULTS_SELECTED_DISPLAY_HEIGHT,
            export_width=_SELECTED_PLOT_EXPORT_WIDTH,
            export_height=_SELECTED_PLOT_EXPORT_HEIGHT,
            publication=publication if grouped else None,
        )

    render_ratio(plot_samples, ratio_metrics)

    state = get_state()
    if _delta_enabled(state):
        from ui.components.stats_charts import create_delta_overview_chart

        delta_zero_col, delta_certified_col = st.columns([1, 3])
        with delta_zero_col:
            show_summary_delta_zero_line = st.checkbox(
                "Show 0 permil line",
                value=True,
                key="summary_delta_zero_line",
            )
        with delta_certified_col:
            certified_delta, certified_delta_uncertainty = _render_manual_certified_controls(
                f"summary_certified_delta_v2_{plot_ratio}",
                checkbox_label="Show certified delta",
                value_label="Certified delta (permil)",
                uncertainty_label=_certified_uncertainty_label(None, unit="permil"),
                value_format="%.4f",
                uncertainty_help=_certified_overlay_help(None),
            )

        def render_delta(plot_samples, *, grouped=False):
            delta_fig = create_delta_overview_chart(
                plot_samples,
                plot_ratio,
                show_error_bars=True,
                error_mode=error_mode_by_label[error_choice],
                show_zero_line=show_summary_delta_zero_line,
                certified_delta=None if grouped else certified_delta,
                certified_delta_uncertainty=certified_delta_uncertainty,
                certified_coverage_factor=2.0,
                cycle_ranges=cycle_ranges,
                delta_overrides=delta_overrides,
                uncertainty_overrides=plot_overrides,
                height=_RESULTS_SELECTED_DISPLAY_HEIGHT,
                reference_name=_delta_reference_name(state),
                x_axis_mode=(
                    "run_order" if axis_choice == "Run order" else "category"
                ),
            )
            for trace in delta_fig.data:
                if trace.name in plot_types:
                    trace.showlegend = show_sample_type_legend
            if grouped:
                apply_sample_plot_groups(delta_fig, groups)
                add_group_references(delta_fig, groups, plot_ratio, "delta")
                apply_group_axes(delta_fig, publication, plot_ratio, "delta")
            suffix = "_groups" if grouped else ""
            _render_results_figure(
                delta_fig,
                theme=theme,
                key=f"results_summary_delta_plot_{plot_ratio}{suffix}",
                filename=f"traceiso_{plot_ratio.replace('/', '-')}_delta_summary{suffix}",
                display_height=_RESULTS_SELECTED_DISPLAY_HEIGHT,
                export_width=_SELECTED_PLOT_EXPORT_WIDTH,
                export_height=_SELECTED_PLOT_EXPORT_HEIGHT,
                publication=publication if grouped else None,
            )

        render_delta(plot_samples)

    st.divider()
    st.subheader("Grouped Samples Plot")
    group_samples = state.result.samples if state.has_result else samples
    groups = render_sample_plot_groups(
        group_samples, session_id=getattr(state, "file_hash", None),
    )
    member_ids = {sid for group in groups for sid in group["members"]}
    grouped_samples = [s for s in plot_samples if s.observation_id in member_ids]
    if not grouped_samples:
        st.info("Add samples to a group to display them here. The sample filters above also apply.")
        return
    st.caption("Only assigned group members are shown. Ratio, sample filters and error-bar settings follow the plot above.")
    publication = render_group_publication_controls(
        groups, plot_ratio, delta_enabled=_delta_enabled(state),
        run_order=axis_choice == "Run order", has_blanks=any(s.is_blank for s in grouped_samples),
    )
    grouped_metrics = {sid: metric for sid, metric in ratio_metrics.items() if sid in member_ids}
    render_ratio(grouped_samples, grouped_metrics, grouped=True)
    if _delta_enabled(state):
        render_delta(grouped_samples, grouped=True)



def _render_qc_dashboard(
    result,
    ratio_name,
    *,
    cycle_ranges=None,
    filter_method="None",
    filter_threshold=DEFAULT_OUTLIER_THRESHOLD_SD,
) -> None:
    """Render the quality control summary."""
    from ui.components.qc_panel import render_qc_dashboard

    st.subheader("Quality Control Summary")
    with st.expander("QC Thresholds (%)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            std_repro_threshold = st.number_input(
                "STD reproducibility",
                min_value=0.0,
                value=0.1,
                step=0.01,
                key="qc_thr_std_repro",
            )
        with col2:
            assess_blank_ratio_change = st.checkbox(
                "Assess signed blank ratio change",
                value=False,
                key="qc_assess_blank_ratio_change",
                help="Optional laboratory criterion; absolute magnitude is assessed while the sign remains displayed.",
            )
            blank_contribution_threshold = (
                st.number_input(
                    "Ratio-change criterion (%)",
                    min_value=0.0,
                    value=0.1,
                    step=0.01,
                    key="qc_thr_blank_contrib",
                )
                if assess_blank_ratio_change else None
            )
        with col3:
            drift_threshold = st.number_input(
                "Session drift",
                min_value=0.0,
                value=0.1,
                step=0.01,
                key="qc_thr_session_drift",
            )

    render_qc_dashboard(
        result,
        ratio_name,
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
        std_repro_threshold=std_repro_threshold,
        blank_contribution_threshold=blank_contribution_threshold,
        drift_threshold=drift_threshold,
    )


def _settings_changed_since_processing(state) -> bool:
    """Compatibility wrapper for the shared settings-change helper."""
    return settings_changed_since_processing(state)

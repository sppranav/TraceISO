"""Blank-correction diagnostics for the uncertainty tab."""

from __future__ import annotations

from html import escape
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config.constants import DEFAULT_OUTLIER_THRESHOLD_SD
from config.settings import UncertaintyConfig, is_russell_law_normalization_engine
from domain.filters.outlier import get_filtered_values, get_runtime_mask
from domain.models import Sample
from ui.state import get_state
from ui.tabs.uncertainty_sections.common import (
    _render_compact_summary,
    _render_equation_note_block,
)
from ui.config_plotly import (
    get_plotly_config,
    PLOTLY_ANNOTATION_FONT_SIZE,
    PLOTLY_AXIS_TITLE_FONT_SIZE,
    PLOTLY_BASE_FONT_SIZE,
)
from ui.theme import get_theme
from ui.utils import (
    format_sample_display_label,
    format_isotope_label,
    format_name,
    get_cycle_ranges,
    get_sample_state_key,
)

_BLANK_CORR_LABELS = {
    "pearson_from_data": "Pearson from data",
    "fixed_value": "Fixed r",
    "uncorrelated": "Uncorrelated",
}


def _render_blank_correction_section(
    all_samples: list,
    ratio_name: str,
    u_config: UncertaintyConfig,
    state,
) -> None:
    """Render a blank-correction diagnostic section for the selected ratio."""
    from domain.uncertainty.blank import resolve_blank_correction_mode

    element_symbol = getattr(state, "element_symbol", "") or ""
    if not element_symbol and getattr(state, "element_config", None) is not None:
        element_symbol = getattr(state.element_config, "symbol", "") or ""
    if not u_config.is_contributor_enabled("u_blank", element_symbol=element_symbol):
        return

    st.divider()
    st.subheader("Blank Correction (u_blank)")

    ratio_def = (
        state.element_config.default_ratios.get(ratio_name)
        if state.element_config and getattr(state.element_config, "default_ratios", None)
        else None
    )
    if ratio_def is None:
        st.info(
            "Blank correction diagnostics are available for configured default ratios only."
        )
        return

    num_isotope, den_isotope = ratio_def
    _strip_elem = lambda iso: iso.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") if iso else iso
    engine = u_config.resolve_engine(
        element_symbol,
        processing_config=getattr(state, "processing_config", None),
    )
    is_normalization_engine = is_russell_law_normalization_engine(engine)
    active_nonblank = [
        sample for sample in all_samples
        if not sample.is_blank
        and not sample.metadata.get("excluded", False)
        and (is_normalization_engine or not sample.is_standard)
    ]
    if not active_nonblank:
        st.info("No active standards or samples available for blank-correction diagnostics.")
        return

    sample_labels = {
        get_sample_state_key(sample): format_sample_display_label(sample)
        for sample in active_nonblank
    }
    sample_options = [get_sample_state_key(sample) for sample in active_nonblank]
    widget_key = "uncertainty_blank_sample"
    if st.session_state.get(widget_key) not in sample_options:
        st.session_state[widget_key] = sample_options[0]
    selected_sample_key = st.selectbox(
        "Sample for blank diagnostics",
        options=sample_options,
        format_func=lambda key: sample_labels.get(key, str(key)),
        key=widget_key,
    )
    selected_sample = next(
        (sample for sample in active_nonblank if get_sample_state_key(sample) == selected_sample_key),
        None,
    )
    if selected_sample is None:
        st.info("Select a sample to inspect its assigned blank file(s).")
        return

    configured_mode = state.processing_config.blank_mode if state.processing_config else None
    blank_mode = resolve_blank_correction_mode(selected_sample, configured_mode)
    cycle_ranges = get_cycle_ranges(state)
    blank_samples, used_fallback = _resolve_effective_blank_samples(
        selected_sample,
        all_samples,
        configured_mode=configured_mode,
    )

    unresolved_note = describe_unresolved_blank_references(selected_sample, all_samples)
    if unresolved_note:
        st.warning(f"Blank selection is incomplete: {unresolved_note}.")

    if not blank_samples:
        st.info("No blank samples are available for the selected sample.")
        return

    num_corrected_mean = _get_blank_sensitivity_mean(
        selected_sample,
        num_isotope,
        ratio_name=ratio_name,
        cycle_ranges=cycle_ranges,
    )
    den_corrected_mean = _get_blank_sensitivity_mean(
        selected_sample,
        den_isotope,
        ratio_name=ratio_name,
        cycle_ranges=cycle_ranges,
    )
    blank_result = _compute_runtime_blank_result(
        sample=selected_sample,
        blank_samples=blank_samples,
        all_samples=all_samples,
        ratio_name=ratio_name,
        num_isotope=num_isotope,
        den_isotope=den_isotope,
        num_corrected_mean=num_corrected_mean,
        den_corrected_mean=den_corrected_mean,
        blank_mode=blank_mode,
        u_config=u_config,
        state=state,
        cycle_ranges=cycle_ranges,
    )
    blank_df, blank_stats_rows = _build_blank_covariation_dataframe(
        blank_samples,
        num_isotope,
        den_isotope,
        aux_isotope=(
            blank_result.aux_isotope
            if blank_result.model_dimension == 3 and blank_result.aux_isotope
            else None
        ),
        correlation_method=u_config.blank_correlation_method,
        fixed_r=u_config.blank_fixed_r,
        blank_uncertainty_input=getattr(u_config, "blank_uncertainty_input", "sd"),
        cycle_ranges=cycle_ranges,
    )

    _render_compact_summary(
        [
            ("Selected sample", selected_sample.name),
            (
                "Blank model",
                (
                    f"3-variable ({_strip_elem(num_isotope)}, {_strip_elem(den_isotope)}, {_strip_elem(blank_result.aux_isotope)})"
                    if blank_result.model_dimension == 3
                    else f"2-variable ({_strip_elem(num_isotope)}, {_strip_elem(den_isotope)})"
                ),
            ),
            ("Blank mode", blank_result.blank_mode),
            ("Blank input", blank_result.blank_uncertainty_input.upper()),
            ("Blank files used", str(blank_result.n_blanks_used)),
            ("Paired cycles", str(blank_result.n_blank_cycles)),
            ("u_blank (abs)", f"{float(blank_result.u_blank_abs):.8f}"),
            (
                "ν_blank",
                "∞" if blank_result.degrees_of_freedom == float("inf")
                else f"{float(blank_result.degrees_of_freedom):.1f}",
            ),
        ]
    )

    blank_names = ", ".join(sample.name for sample in blank_samples)
    st.caption(f"Assigned blank file(s) for {selected_sample.name}: {blank_names}")
    if used_fallback:
        st.caption(
            "No explicit blank assignment was found for this sample. "
            "The diagnostic mirrors the engine fallback selection."
        )
    if blank_result.correlation_warning:
        st.caption(blank_result.correlation_warning)

    overview_tab, detail_tab = st.tabs(["Overview", "Detail"])

    with overview_tab:
        if blank_df.empty:
            st.info("Need at least 2 overlapping blank cycles to show the selected blank-file covariation.")
        else:
            if blank_result.model_dimension == 3 and blank_result.aux_isotope:
                blank_figs = _build_blank_covariation_3var_figures(
                    blank_df,
                    num_isotope,
                    den_isotope,
                    blank_result.aux_isotope,
                    blank_stats_rows,
                    selected_sample_name=selected_sample.name,
                )
                for panel_fig, panel_filename in blank_figs:
                    st.plotly_chart(
                        panel_fig,
                        width="stretch",
                        config=get_plotly_config(filename=panel_filename),
                    )
            else:
                blank_fig = _build_blank_covariation_figure(
                    blank_df,
                    num_isotope,
                    den_isotope,
                    blank_stats_rows,
                    selected_sample_name=selected_sample.name,
                )
                st.plotly_chart(blank_fig, width="stretch", config=get_plotly_config())

        if blank_stats_rows:
            stats_df = _format_blank_stats_dataframe(
                blank_stats_rows,
                num_isotope,
                den_isotope,
                blank_result.aux_isotope
                if blank_result.model_dimension == 3 and blank_result.aux_isotope
                else None,
            )
            st.markdown("**Assigned Blank File Statistics**")
            _render_blank_html_table(stats_df)

        st.markdown("**Blank-Correction Equations**")
        col_corr, col_model = st.columns(2, gap="large")
        with col_corr:
            if blank_result.model_dimension == 3 and blank_result.aux_isotope:
                corr_lines = [
                    r"r_{ij} = \frac{\mathrm{cov}(V_i, V_j)}{s_i\,s_j}",
                    r"r_{87,86},\ r_{87,88},\ r_{86,88}\ \mathrm{are\ computed\ from\ the\ same\ paired\ blank\ cycles}",
                ]
                corr_note = (
                    "Empirical r is measured separately for every isotope pair within each blank file. "
                    "Applied r is the value used in the covariance matrix after the selected correlation "
                    "mode (Pearson from data, fixed r, or uncorrelated = 0). "
                    "The three pairwise r values map directly to the three covariance terms in the "
                    "3-variable blank model."
                )
            else:
                corr_lines = [
                    r"r = \frac{\mathrm{cov}(V_{\mathrm{num}}, V_{\mathrm{den}})}{s_{\mathrm{num}}\,s_{\mathrm{den}}}",
                    r"= \frac{\sum \left[(V_{\mathrm{num},i} - \overline{V}_{\mathrm{num}})(V_{\mathrm{den},i} - \overline{V}_{\mathrm{den}})\right]}{(n - 1)\,s_{\mathrm{num}}\,s_{\mathrm{den}}}",
                ]
                corr_note = (
                    "Empirical r is measured within each blank file from paired blank cycles. "
                    "Applied r is the model value used in u_blank after the selected correlation mode "
                    "(Pearson from data, fixed r, or uncorrelated = 0). "
                    "Parameters: V_num and V_den are paired blank voltages, "
                    "s_num and s_den are their sample SDs, and n is the number of paired blank cycles."
                )
            _render_equation_note_block(
                corr_lines,
                title="Correlation Term",
                note=corr_note,
            )

        with col_model:
            if blank_result.model_dimension == 3:
                _render_equation_note_block(
                    [
                        r"u_{\mathrm{blank}}^2 = \mathbf{c}^{\mathsf{T}} \mathbf{S}_{\mathrm{blank}} \mathbf{c}",
                        r"\mathbf{c} = [c_{87}, c_{86}, c_{88}]",
                        r"= (c_{87}u_{87})^2 + (c_{86}u_{86})^2 + (c_{88}u_{88})^2",
                        r"\quad + 2c_{87}c_{86}u_{87}u_{86}r_{87,86} + 2c_{87}c_{88}u_{87}u_{88}r_{87,88} + 2c_{86}c_{88}u_{86}u_{88}r_{86,88}",
                        r"c_k = \frac{R(B_k + s_k) - R(B_k - s_k)}{2\,s_k}",
                    ],
                    title="Blank Model",
                    note=(
                        "All three sensitivity coefficients are computed by Kragten perturbation: "
                        "the blank at each mass (87Sr, 86Sr, 88Sr) is shifted by plus/minus 1 SD and the "
                        "full Sr correction chain (IIF + interference) is replayed to obtain the "
                        "numerical derivative dR/dB_k. "
                        "The blank covariance matrix is the full 3x3 matrix from paired blank cycles. "
                        "Parameters: R is the final processed ratio, B_k is the blank mean at isotope k, "
                        "s_k is the 1 SD blank uncertainty at isotope k, c_k is the sensitivity dR/dB_k, "
                        "and c is the sensitivity vector [c87, c86, c88]."
                    ),
                )
            else:
                _render_equation_note_block(
                    [
                        r"u_{\mathrm{blank}}^2 = (c_{\mathrm{num}}\,u_{\mathrm{num}})^2 + (c_{\mathrm{den}}\,u_{\mathrm{den}})^2 + 2\,c_{\mathrm{num}}\,c_{\mathrm{den}}\,u_{\mathrm{num}}\,u_{\mathrm{den}}\,r",
                        r"c_{\mathrm{num}} = -\frac{1}{I_{\mathrm{den}}}",
                        r"c_{\mathrm{den}} = \frac{I_{\mathrm{num}}}{I_{\mathrm{den}}^2}",
                    ],
                    title="Blank Model",
                    note=(
                        "Parameters: I_num and I_den are the blank-corrected sample intensities used "
                        "for the sensitivity coefficients; u_num and u_den are the 1 SD blank "
                        "uncertainties of the numerator and denominator blank voltages; c_num and "
                        "c_den are the ratio sensitivities to those two blank terms; r is the applied "
                        "blank correlation coefficient."
                    ),
                )
            if blank_result.blank_mode == "before_and_after" and len(blank_samples) >= 2:
                _render_equation_note_block(
                    [
                        r"u_{\mathrm{blank,combined}}^2 = \frac{u_{\mathrm{before}}^2 + u_{\mathrm{after}}^2}{4}",
                        r"u_{\mathrm{blank,combined}} = \sqrt{\left(\frac{u_{\mathrm{before}}}{2}\right)^2 + \left(\frac{u_{\mathrm{after}}}{2}\right)^2}",
                    ],
                    title="Before-and-After Combination",
                    note=(
                        "Each blank file keeps its own paired-cycle covariance term. "
                        "The two blank uncertainties are then halved and combined in quadrature. "
                        "Parameters: u_before and u_after are the single-blank standard uncertainties "
                        "from the assigned before and after blank files."
                    ),
                )

    detail_df = _build_blank_detail_dataframe(
        active_nonblank,
        all_samples,
        ratio_name,
        state.element_config,
        u_config,
        processing_config=state.processing_config,
        cycle_ranges=cycle_ranges,
    )

    with detail_tab:
        if detail_df.empty:
            st.info("No per-sample blank uncertainty values could be computed for the selected ratio.")
            return

        _render_compact_summary(
            [
                ("Samples evaluated", str(len(detail_df))),
                ("Median u_blank (abs)", f"{float(detail_df['u_blank'].median()):.8f}"),
                ("Max u_blank (abs)", f"{float(detail_df['u_blank'].max()):.8f}"),
            ]
        )
        _render_blank_html_table(_format_blank_detail_dataframe(detail_df))
        _render_blank_detail_download(detail_df, ratio_name)


def describe_unresolved_blank_references(sample: Sample, all_samples: list) -> str:
    """Describe blank references this session cannot resolve to one observation.

    A legacy record that carries a label without an observation ID can be
    ambiguous; a record whose observation ID names nothing in this session is
    simply gone. Rather than silently resolving either to the first same-named
    observation — which is what produced the wrong blank in the first place —
    the selection drops it, and this sentence says so.
    """
    from domain.uncertainty.blank import resolve_blank_selection

    selection = resolve_blank_selection(sample, all_samples)
    parts = []
    if selection.ambiguous_names:
        parts.append(
            "blank label(s) "
            + ", ".join(f"'{name}'" for name in selection.ambiguous_names)
            + " match more than one observation and carry no recorded "
            "observation identity, so no blank could be identified for them"
        )
    if selection.unresolved_names:
        parts.append(
            "blank label(s) "
            + ", ".join(f"'{name}'" for name in selection.unresolved_names)
            + " are not present in this session"
        )
    if selection.unresolved_ids:
        parts.append(
            "recorded blank observation(s) "
            + ", ".join(f"'{value}'" for value in selection.unresolved_ids)
            + " are not present in this session"
        )
    return "; ".join(parts)


def _resolve_effective_blank_samples(
    sample: Sample,
    all_samples: list,
    *,
    configured_mode: str | None = None,
) -> Tuple[list, bool]:
    """Resolve the exact blank file(s) used by the blank uncertainty engine."""
    from domain.uncertainty.blank import (
        resolve_blank_correction_mode,
        resolve_blank_samples_for_uncertainty,
    )

    blank_mode = resolve_blank_correction_mode(sample, configured_mode)
    explicit_blanks, _selection = resolve_blank_samples_for_uncertainty(sample, all_samples)
    used_fallback = False

    if explicit_blanks is None:
        # A recorded reference that could not be resolved. The engine
        # propagates nothing here, so the panel must not name a blank either.
        return [], False

    candidate_blanks = explicit_blanks
    if not candidate_blanks:
        candidate_blanks = [blank_sample for blank_sample in all_samples if blank_sample.is_blank]
        used_fallback = True

    if blank_mode == "none" or not candidate_blanks:
        return [], used_fallback

    if blank_mode == "before_and_after" and len(candidate_blanks) >= 2:
        return [candidate_blanks[0], candidate_blanks[-1]], used_fallback

    return [candidate_blanks[0]], used_fallback


def _compute_runtime_blank_result(
    *,
    sample: Sample,
    blank_samples: list,
    all_samples: list,
    ratio_name: str,
    num_isotope: str,
    den_isotope: str,
    num_corrected_mean: float,
    den_corrected_mean: float,
    blank_mode: str,
    u_config: UncertaintyConfig,
    state,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
):
    """Compute the exact blank result used by the active engine when possible."""
    from domain.uncertainty.blank import compute_blank_uncertainty
    from domain.uncertainty.engine_external_pb_tl import (
        _compute_blank_contribution as _compute_pb_tl_blank_contribution,
    )
    from domain.uncertainty.engine_internal_sr import (
        _compute_blank_contribution as _compute_sr_blank_contribution,
    )

    element_symbol = state.element_symbol or ""
    if not element_symbol and state.element_config is not None:
        element_symbol = getattr(state.element_config, "symbol", "")
    engine = u_config.resolve_engine(
        element_symbol,
        processing_config=getattr(state, "processing_config", None),
    )

    if (
        engine == "internal_normalization"
        and ratio_name == "87Sr/86Sr"
        and bool(getattr(u_config, "sr_blank_3var", False))
        and state.processing_config is not None
        and state.element_config is not None
    ):
        return _compute_sr_blank_contribution(
            sample,
            ratio_name,
            all_samples,
            state.element_config,
            u_config,
            ratio_mean=0.0,
            processing_config=state.processing_config,
            cycle_ranges=cycle_ranges,
        )

    if (
        engine == "pb_tl_external_normalization"
        and state.processing_config is not None
        and state.element_config is not None
    ):
        return _compute_pb_tl_blank_contribution(
            sample=sample,
            ratio_name=ratio_name,
            all_samples=all_samples,
            element_config=state.element_config,
            uncertainty_config=u_config,
            ratio_mean=0.0,
            processing_config=state.processing_config,
            cycle_ranges=cycle_ranges,
        )

    return compute_blank_uncertainty(
        blank_samples=blank_samples,
        num_isotope=num_isotope,
        den_isotope=den_isotope,
        num_corrected_mean=num_corrected_mean,
        den_corrected_mean=den_corrected_mean,
        correlation_method=u_config.blank_correlation_method,
        fixed_r=u_config.blank_fixed_r,
        blank_correction_mode=blank_mode,
        blank_uncertainty_input=getattr(u_config, "blank_uncertainty_input", "sd"),
        cycle_ranges=cycle_ranges,
    )



def _build_blank_covariation_dataframe(
    blank_samples: list,
    num_isotope: str,
    den_isotope: str,
    *,
    aux_isotope: Optional[str] = None,
    correlation_method: str,
    fixed_r: float,
    blank_uncertainty_input: str = "sd",
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, float | int | str]]]:
    """Build per-blank paired data and statistics for the selected sample."""
    from domain.uncertainty.blank import (
        _compute_correlation,
        _get_paired_blank_matrix,
        blank_input_sigma,
        compute_blank_stats_3var,
        get_paired_blank_voltages,
        normalize_blank_uncertainty_input,
    )

    blank_input_mode = normalize_blank_uncertainty_input(blank_uncertainty_input)
    rows = []
    stats_rows: List[Dict[str, float | int | str]] = []
    n_blanks = len(blank_samples)

    for index, blank_sample in enumerate(blank_samples):
        if n_blanks == 2:
            blank_role = "Before" if index == 0 else "After"
        else:
            blank_role = "Assigned"

        display_label = f"{blank_sample.name} ({blank_role})"
        if aux_isotope:
            paired = _get_paired_blank_matrix(
                blank_sample,
                (num_isotope, den_isotope, aux_isotope),
                cycle_ranges=cycle_ranges,
            )
            if paired is not None:
                num_values, den_values, aux_values = paired
                n_pairs = int(paired.shape[1])
                stats_3var = compute_blank_stats_3var(
                    blank_sample,
                    (num_isotope, den_isotope, aux_isotope),
                    correlation_method=correlation_method,
                    fixed_r=fixed_r,
                    blank_uncertainty_input=blank_input_mode,
                    cycle_ranges=cycle_ranges,
                )
            else:
                num_values, den_values = get_paired_blank_voltages(
                    blank_sample,
                    num_isotope,
                    den_isotope,
                    cycle_ranges=cycle_ranges,
                )
                aux_values = np.array([], dtype=np.float64)
                n_pairs = len(num_values)
                stats_3var = None
        else:
            num_values, den_values = get_paired_blank_voltages(
                blank_sample,
                num_isotope,
                den_isotope,
                cycle_ranges=cycle_ranges,
            )
            aux_values = np.array([], dtype=np.float64)
            n_pairs = len(num_values)
            stats_3var = None

        empirical_r, empirical_warning = _compute_correlation(
            num_values, den_values, "pearson_from_data", 0.0,
        )
        applied_r, applied_warning = _compute_correlation(
            num_values, den_values, correlation_method, fixed_r,
        )
        pair_stats = None
        if aux_isotope and stats_3var is not None:
            isotopes = (num_isotope, den_isotope, aux_isotope)
            pair_stats = []
            for i, j in ((0, 1), (0, 2), (1, 2)):
                pair_stats.append(
                    {
                        "x_isotope": isotopes[i],
                        "y_isotope": isotopes[j],
                        "pair_label": f"{isotopes[i]} vs {isotopes[j]}",
                        "u_x_sd": float(stats_3var.sds[isotopes[i]]),
                        "u_y_sd": float(stats_3var.sds[isotopes[j]]),
                        "u_x_input": float(stats_3var.input_sds[isotopes[i]]),
                        "u_y_input": float(stats_3var.input_sds[isotopes[j]]),
                        "empirical_r": float(stats_3var.empirical_correlation_matrix[i, j]),
                        "applied_r": float(stats_3var.applied_correlation_matrix[i, j]),
                    }
                )

        stats_rows.append(
            {
                "blank_name": blank_sample.name,
                "blank_role": blank_role,
                "display_label": display_label,
                "n_pairs": n_pairs,
                "u_num_sd": float(np.std(num_values, ddof=1)) if len(num_values) >= 2 else 0.0,
                "u_den_sd": float(np.std(den_values, ddof=1)) if len(den_values) >= 2 else 0.0,
                "u_aux_sd": float(np.std(aux_values, ddof=1)) if len(aux_values) >= 2 else 0.0,
                "u_num_input": blank_input_sigma(
                    float(np.std(num_values, ddof=1)) if len(num_values) >= 2 else 0.0,
                    n_pairs,
                    blank_input_mode,
                ),
                "u_den_input": blank_input_sigma(
                    float(np.std(den_values, ddof=1)) if len(den_values) >= 2 else 0.0,
                    n_pairs,
                    blank_input_mode,
                ),
                "u_aux_input": blank_input_sigma(
                    float(np.std(aux_values, ddof=1)) if len(aux_values) >= 2 else 0.0,
                    n_pairs,
                    blank_input_mode,
                ),
                "blank_uncertainty_input": blank_input_mode,
                "empirical_r": empirical_r,
                "applied_r": applied_r,
                "empirical_warning": empirical_warning,
                "applied_warning": applied_warning,
                "pair_stats": pair_stats,
            }
        )

        if n_pairs < 2:
            continue

        for pair_index in range(n_pairs):
            rows.append(
                {
                    "Blank sample": blank_sample.name,
                    "Blank role": blank_role,
                    "Display label": display_label,
                    "Pair": pair_index + 1,
                    num_isotope: num_values[pair_index],
                    den_isotope: den_values[pair_index],
                    **(
                        {aux_isotope: aux_values[pair_index]}
                        if aux_isotope and len(aux_values) > pair_index
                        else {}
                    ),
                }
            )

    return pd.DataFrame(rows), stats_rows


def _format_blank_stats_dataframe(
    blank_stats_rows: List[Dict[str, float | int | str]],
    num_isotope: str,
    den_isotope: str,
    aux_isotope: Optional[str] = None,
) -> pd.DataFrame:
    """Format per-blank statistics for display."""
    if aux_isotope:
        rows = []
        for stats_row in blank_stats_rows:
            pair_stats = stats_row.get("pair_stats")
            if not pair_stats:
                rows.append(
                    {
                        "Blank file": stats_row["display_label"],
                        "Pair": f"{num_isotope} vs {den_isotope}",
                        "Paired cycles": stats_row["n_pairs"],
                        "Input": stats_row["blank_uncertainty_input"],
                        "u_x SD": stats_row["u_num_sd"],
                        "u_y SD": stats_row["u_den_sd"],
                        "u_x input": stats_row["u_num_input"],
                        "u_y input": stats_row["u_den_input"],
                        "Empirical r": stats_row["empirical_r"],
                        "Applied r": stats_row["applied_r"],
                    }
                )
                continue
            for pair_row in pair_stats:
                rows.append(
                    {
                        "Blank file": stats_row["display_label"],
                        "Pair": pair_row["pair_label"],
                        "Paired cycles": stats_row["n_pairs"],
                        "Input": stats_row["blank_uncertainty_input"],
                        "u_x SD": pair_row["u_x_sd"],
                        "u_y SD": pair_row["u_y_sd"],
                        "u_x input": pair_row["u_x_input"],
                        "u_y input": pair_row["u_y_input"],
                        "Empirical r": pair_row["empirical_r"],
                        "Applied r": pair_row["applied_r"],
                    }
                )
        stats_df = pd.DataFrame(rows)
        for column in ("u_x SD", "u_y SD", "u_x input", "u_y input", "Empirical r", "Applied r"):
            stats_df[column] = stats_df[column].map(lambda value: f"{float(value):.6f}")
        return stats_df

    stats_df = pd.DataFrame(blank_stats_rows)
    stats_df = stats_df.rename(
        columns={
            "display_label": "Blank file",
            "n_pairs": "Paired cycles",
            "blank_uncertainty_input": "Input",
            "u_num_sd": "u_num SD",
            "u_den_sd": "u_den SD",
            "u_num_input": "u_num input",
            "u_den_input": "u_den input",
            "empirical_r": "Empirical r",
            "applied_r": "Applied r",
        }
    )
    stats_df = stats_df[
        [
            "Blank file",
            "Paired cycles",
            "Input",
            "u_num SD",
            "u_den SD",
            "u_num input",
            "u_den input",
            "Empirical r",
            "Applied r",
        ]
    ]
    for column in ("u_num SD", "u_den SD", "u_num input", "u_den input", "Empirical r", "Applied r"):
        stats_df[column] = stats_df[column].map(lambda value: f"{float(value):.6f}")
    return stats_df


def _build_blank_covariation_figure(
    blank_df: pd.DataFrame,
    num_isotope: str,
    den_isotope: str,
    blank_stats_rows: List[Dict[str, float | int | str]],
    *,
    selected_sample_name: str,
):
    """Build a covariation figure for the selected sample's blank file(s)."""
    import plotly.graph_objects as go

    theme = get_theme()
    palette = theme.palette
    fig = go.Figure()

    color_scale = palette.sequence
    annotation_lines = []
    for index, stats_row in enumerate(blank_stats_rows):
        display_label = str(stats_row["display_label"])
        subset = blank_df[blank_df["Display label"] == display_label]
        color = color_scale[index % len(color_scale)]

        fig.add_trace(
            go.Scatter(
                x=subset[num_isotope] if not subset.empty else [],
                y=subset[den_isotope] if not subset.empty else [],
                mode="markers",
                marker=dict(size=8, color=color),
                name=display_label,
                hovertemplate=(
                    "%{fullData.name}<br>"
                    "Pair %{customdata}<br>"
                    f"{format_isotope_label(num_isotope)}: "
                    "%{x:.6f} V<br>"
                    f"{format_isotope_label(den_isotope)}: "
                    "%{y:.6f} V<extra></extra>"
                ),
                customdata=subset["Pair"] if not subset.empty else [],
            )
        )

        if len(subset) >= 2 and subset[num_isotope].nunique(dropna=True) >= 2:
            coeffs = np.polyfit(subset[num_isotope], subset[den_isotope], 1)
            x_line = np.linspace(subset[num_isotope].min(), subset[num_isotope].max(), 100)
            y_line = coeffs[0] * x_line + coeffs[1]
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color=color, dash="dash"),
                    name=f"{display_label} trend",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        annotation_lines.append(
            f"{display_label}: r_emp = {float(stats_row['empirical_r']):.3f}, "
            f"r_used = {float(stats_row['applied_r']):.3f}"
        )

    fig.update_layout(
        title=(
            f"Blank Covariation for {selected_sample_name} "
            f"({format_name(num_isotope)} vs {format_name(den_isotope)})"
        ),
        xaxis_title=f"{format_isotope_label(num_isotope)} blank intensity (V)",
        yaxis_title=f"{format_isotope_label(den_isotope)} blank intensity (V)",
        height=320,
        margin=dict(l=80, r=20, t=60, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0,
        ),
        annotations=[
            dict(
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                showarrow=False,
                text="<br>".join(annotation_lines),
                bgcolor=palette.annotation_bg,
                bordercolor=palette.annotation_border,
                borderwidth=1,
                font=dict(color=palette.annotation_text_color),
            )
        ],
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        tickfont=dict(size=PLOTLY_BASE_FONT_SIZE),
        title_font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        tickfont=dict(size=PLOTLY_BASE_FONT_SIZE),
        title_font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
    )
    theme.apply_to_figure(fig, profile="diagnostic")
    return fig


def _build_blank_covariation_3var_figures(
    blank_df: pd.DataFrame,
    num_isotope: str,
    den_isotope: str,
    aux_isotope: str,
    blank_stats_rows: List[Dict[str, float | int | str]],
    *,
    selected_sample_name: str,
) -> List[tuple]:
    """Build three individual pairwise covariation figures for the Sr 3-variable blank model.

    Returns a list of (figure, download_filename) tuples — one per isotope pair —
    so each panel can be rendered at full Streamlit width and downloaded independently.
    """
    import plotly.graph_objects as go

    theme = get_theme()
    palette = theme.palette
    pairs = [
        (num_isotope, den_isotope),
        (num_isotope, aux_isotope),
        (den_isotope, aux_isotope),
    ]
    color_scale = palette.sequence

    # Build a safe sample name fragment for download file names.
    safe_sample = selected_sample_name.replace(" ", "_").replace("/", "-")

    results = []
    for x_iso, y_iso in pairs:
        fig = go.Figure()
        annotation_lines = []

        for blank_index, stats_row in enumerate(blank_stats_rows):
            display_label = str(stats_row["display_label"])
            subset = blank_df[blank_df["Display label"] == display_label]
            color = color_scale[blank_index % len(color_scale)]

            if x_iso not in subset.columns or y_iso not in subset.columns:
                x_values: list = []
                y_values: list = []
            else:
                x_values = subset[x_iso]
                y_values = subset[y_iso]

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="markers",
                    marker=dict(size=9, color=color),
                    name=display_label,
                    hovertemplate=(
                        "%{fullData.name}<br>"
                        "Pair %{customdata}<br>"
                        f"{format_isotope_label(x_iso)}: "
                        "%{x:.6f} V<br>"
                        f"{format_isotope_label(y_iso)}: "
                        "%{y:.6f} V<extra></extra>"
                    ),
                    customdata=subset["Pair"] if not subset.empty else [],
                )
            )

            has_data = not isinstance(x_values, list)
            if (
                has_data
                and len(subset) >= 2
                and x_iso in subset.columns
                and y_iso in subset.columns
                and subset[x_iso].nunique(dropna=True) >= 2
            ):
                coeffs = np.polyfit(subset[x_iso], subset[y_iso], 1)
                x_line = np.linspace(subset[x_iso].min(), subset[x_iso].max(), 100)
                y_line = coeffs[0] * x_line + coeffs[1]
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(color=color, dash="dash", width=2),
                        name=f"{display_label} trend",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

            pair_stat = _find_pair_stat(stats_row, x_iso, y_iso)
            if pair_stat is not None:
                empirical_r = float(pair_stat["empirical_r"])
                applied_r = float(pair_stat["applied_r"])
            elif x_iso == num_isotope and y_iso == den_isotope:
                empirical_r = float(stats_row["empirical_r"])
                applied_r = float(stats_row["applied_r"])
            else:
                empirical_r = 0.0
                applied_r = 0.0
            n_pairs = int(stats_row["n_pairs"])
            annotation_lines.append(
                f"{display_label}:  "
                f"r<sub>emp</sub> = {empirical_r:.3f}   "
                f"r<sub>used</sub> = {applied_r:.3f}   "
                f"n = {n_pairs}"
            )

        fig.add_annotation(
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            showarrow=False,
            text="<br>".join(annotation_lines),
            bgcolor=palette.annotation_bg,
            bordercolor=palette.annotation_border,
            borderwidth=1,
            font=dict(color=palette.annotation_text_color, size=PLOTLY_ANNOTATION_FONT_SIZE),
        )

        fig.update_layout(
            title=(
                f"{format_name(x_iso)} vs {format_name(y_iso)}  \u2014  "
                f"Blank Covariation for {selected_sample_name}"
            ),
            xaxis_title=f"{format_isotope_label(x_iso)} blank intensity (V)",
            yaxis_title=f"{format_isotope_label(y_iso)} blank intensity (V)",
            height=420,
            margin=dict(l=90, r=30, t=70, b=70),
            showlegend=False,
        )
        fig.update_xaxes(
            showgrid=False,
            zeroline=False,
            showline=True,
            tickfont=dict(size=PLOTLY_BASE_FONT_SIZE),
            title_font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
        )
        fig.update_yaxes(
            showgrid=False,
            zeroline=False,
            showline=True,
            automargin=True,
            tickfont=dict(size=PLOTLY_BASE_FONT_SIZE),
            title_font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE),
            title_standoff=8,
        )
        theme.apply_to_figure(fig, profile="diagnostic")

        safe_x = x_iso.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") if x_iso else x_iso
        safe_y = y_iso.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") if y_iso else y_iso
        filename = f"blank_covariation_{safe_sample}_{safe_x}_vs_{safe_y}"
        results.append((fig, filename))

    return results


def _find_pair_stat(
    stats_row: Dict[str, float | int | str],
    x_iso: str,
    y_iso: str,
) -> Optional[Dict[str, float | int | str]]:
    """Return stored pair statistics for either isotope order."""
    pair_stats = stats_row.get("pair_stats")
    if not pair_stats:
        return None
    for pair_row in pair_stats:
        if {
            str(pair_row.get("x_isotope")),
            str(pair_row.get("y_isotope")),
        } == {x_iso, y_iso}:
            return pair_row
    return None


def _build_blank_detail_dataframe(
    samples: list,
    all_samples: list,
    ratio_name: str,
    element_config,
    u_config: UncertaintyConfig,
    *,
    processing_config=None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """Build the per-sample blank uncertainty detail table."""
    from domain.ratio_selection import get_best_ratio_data
    from domain.uncertainty.blank import resolve_blank_correction_mode

    ratio_def = (
        element_config.default_ratios.get(ratio_name)
        if element_config and getattr(element_config, "default_ratios", None)
        else None
    )
    if ratio_def is None:
        return pd.DataFrame()

    num_isotope, den_isotope = ratio_def
    rows = []

    for sample in samples:
        ratio_cd = get_best_ratio_data(sample, ratio_name)
        if ratio_cd is None or ratio_cd.n_valid < 2:
            continue

        configured_mode = (
            processing_config.blank_mode if processing_config is not None else None
        )
        blank_mode = resolve_blank_correction_mode(sample, configured_mode)
        blank_samples, used_fallback = _resolve_effective_blank_samples(
            sample,
            all_samples,
            configured_mode=configured_mode,
        )
        if not blank_samples:
            continue

        num_corrected_mean = _get_blank_sensitivity_mean(
            sample,
            num_isotope,
            ratio_name=ratio_name,
            cycle_ranges=cycle_ranges,
        )
        den_corrected_mean = _get_blank_sensitivity_mean(
            sample,
            den_isotope,
            ratio_name=ratio_name,
            cycle_ranges=cycle_ranges,
        )
        state_like = type("BlankState", (), {
            "element_symbol": getattr(element_config, "symbol", ""),
            "element_config": element_config,
            "processing_config": processing_config,
        })()
        blank_result = _compute_runtime_blank_result(
            sample=sample,
            blank_samples=blank_samples,
            all_samples=all_samples,
            ratio_name=ratio_name,
            num_isotope=num_isotope,
            den_isotope=den_isotope,
            num_corrected_mean=num_corrected_mean,
            den_corrected_mean=den_corrected_mean,
            blank_mode=blank_mode,
            u_config=u_config,
            state=state_like,
            cycle_ranges=cycle_ranges,
        )
        runtime_ratio_values = get_filtered_values(
            ratio_cd.values,
            ratio_cd.mask,
            sample.name,
            sample_key=get_sample_state_key(sample),
            cycle_ranges=cycle_ranges,
            filter_method=(
                processing_config.filter_method
                if processing_config is not None else "None"
            ),
            filter_threshold=(
                processing_config.get_active_filter_threshold()
                if processing_config is not None else DEFAULT_OUTLIER_THRESHOLD_SD
            ),
        )
        ratio_mean = (
            float(np.mean(runtime_ratio_values))
            if len(runtime_ratio_values) > 0 else float("nan")
        )
        u_blank_rel = (
            blank_result.u_blank_abs / ratio_mean * 1000.0
            if ratio_mean != 0.0 and np.isfinite(ratio_mean)
            else float("nan")
        )
        u_blank_uncorrelated_abs = _get_u_blank_uncorrelated_abs(blank_result)
        correlation_term_abs2 = _get_u_blank_correlation_term_abs2(
            blank_result,
            u_blank_uncorrelated_abs,
        )
        delta_u_from_r = _get_delta_u_from_correlation(
            blank_result,
            u_blank_uncorrelated_abs,
        )
        u_blank_reduction_from_r_pct = _get_u_blank_reduction_from_correlation_percent(
            delta_u_from_r,
            u_blank_uncorrelated_abs,
        )

        rows.append(
            {
                "Sample": sample.name,
                "Type": sample.sample_type.upper(),
                "Blanks used": ", ".join(blank.name for blank in blank_samples),
                "Blank source": "Fallback" if used_fallback else "Assigned",
                "Model": "3-variable" if blank_result.model_dimension == 3 else "2-variable",
                "Blank input": blank_result.blank_uncertainty_input,
                "Paired cycles": blank_result.n_blank_cycles,
                "I_num": blank_result.num_corrected_mean,
                "I_den": blank_result.den_corrected_mean,
                "I_aux": blank_result.aux_corrected_mean,
                "u_num SD": blank_result.u_num_sd,
                "u_den SD": blank_result.u_den_sd,
                "u_aux SD": blank_result.u_aux_sd,
                "u_num input": blank_result.u_num_input,
                "u_den input": blank_result.u_den_input,
                "u_aux input": blank_result.u_aux_input,
                "Applied r": blank_result.correlation,
                "u_blank (r=0)": u_blank_uncorrelated_abs,
                "u_blank (with r)": blank_result.u_blank_abs,
                "u_blank reduction from r (%)": u_blank_reduction_from_r_pct,
                "Covariance term (u_blank^2)": correlation_term_abs2,
                "Δ r": delta_u_from_r,
                "u_blank": blank_result.u_blank_abs,
                "u_blank (‰)": u_blank_rel,
                "ν_blank": blank_result.degrees_of_freedom,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["Type", "Sample"]).reset_index(drop=True)


def _get_blank_sensitivity_mean(
    sample,
    isotope: str,
    *,
    ratio_name: Optional[str] = None,
    cycle_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> float:
    """Return the mean blank-corrected intensity used as the sensitivity coefficient."""
    from domain.ratio_selection import get_processing_ratio_data

    src = (
        sample.blank_corrected_intensities
        if sample.blank_corrected_intensities
        else sample.corrected_intensities
        if sample.corrected_intensities
        else sample.intensities
    )
    cd = src.get(isotope)
    if cd is not None and cd.n_valid > 0:
        mask = np.asarray(cd.mask, dtype=bool).copy()
        if ratio_name:
            ratio_cd = get_processing_ratio_data(sample, ratio_name)
            if ratio_cd is not None:
                state = get_state()
                processing_config = getattr(state, "processing_config", None)
                filter_method = processing_config.filter_method if processing_config else "None"
                filter_threshold = (
                    processing_config.get_active_filter_threshold()
                    if processing_config
                    else DEFAULT_OUTLIER_THRESHOLD_SD
                )
                ratio_mask = get_runtime_mask(
                    ratio_cd.values,
                    ratio_cd.mask,
                    sample.name,
                    cycle_ranges=cycle_ranges,
                    sample_key=get_sample_state_key(sample),
                    filter_method=filter_method,
                    filter_threshold=filter_threshold,
                )
                n = min(len(mask), len(ratio_mask))
                if n > 0:
                    mask[:n] = mask[:n] & ratio_mask[:n]
        valid_values = get_filtered_values(
            cd.values,
            mask,
            sample.name,
            sample_key=get_sample_state_key(sample),
            cycle_ranges=cycle_ranges,
            filter_method="None",
            filter_threshold=DEFAULT_OUTLIER_THRESHOLD_SD,
        )
        finite_values = np.asarray(valid_values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if len(finite_values) > 0:
            return float(np.mean(finite_values))
    return 0.0


def _get_u_blank_uncorrelated_abs(blank_result) -> float:
    """Return the no-correlation blank uncertainty, falling back to final u_blank."""
    final_value = float(getattr(blank_result, "u_blank_abs", 0.0) or 0.0)
    value = getattr(blank_result, "u_blank_uncorrelated_abs", None)
    if value is None:
        return final_value

    value = float(value)
    if not np.isfinite(value):
        return final_value

    if value == 0.0 and final_value > 0.0:
        return final_value
    return value


def _get_u_blank_correlation_term_abs2(
    blank_result,
    u_blank_uncorrelated_abs: float,
) -> float:
    """Return the applied covariance term in u_blank variance space."""
    final_value = float(getattr(blank_result, "u_blank_abs", 0.0) or 0.0)
    value = getattr(blank_result, "u_blank_correlation_term_abs2", None)
    if value is not None:
        value = float(value)
        if np.isfinite(value):
            return value
    return final_value ** 2 - float(u_blank_uncorrelated_abs) ** 2


def _get_delta_u_from_correlation(
    blank_result,
    u_blank_uncorrelated_abs: float,
) -> float:
    """Return the change in standard uncertainty caused by correlation."""
    final_value = float(getattr(blank_result, "u_blank_abs", 0.0) or 0.0)
    delta = float(u_blank_uncorrelated_abs) - final_value
    return delta if np.isfinite(delta) else 0.0


def _get_u_blank_reduction_from_correlation_percent(
    delta_u_from_r: float,
    u_blank_uncorrelated_abs: float,
) -> float:
    """Return the percent reduction in u_blank caused by correlation."""
    no_r_value = float(u_blank_uncorrelated_abs)
    if no_r_value <= 0.0 or not np.isfinite(no_r_value):
        return 0.0
    pct = float(delta_u_from_r) / no_r_value * 100.0
    return pct if np.isfinite(pct) else 0.0


_BLANK_DETAIL_COLUMN_CONFIG = {
    "I_num": st.column_config.NumberColumn("I_num", format="%.4f"),
    "I_den": st.column_config.NumberColumn("I_den", format="%.4f"),
    "I_aux": st.column_config.NumberColumn("I_aux", format="%.4f"),
    "u_num SD": st.column_config.NumberColumn("u_num SD", format="%.8f"),
    "u_den SD": st.column_config.NumberColumn("u_den SD", format="%.8f"),
    "u_aux SD": st.column_config.NumberColumn("u_aux SD", format="%.8f"),
    "u_num input": st.column_config.NumberColumn("u_num input", format="%.8f"),
    "u_den input": st.column_config.NumberColumn("u_den input", format="%.8f"),
    "u_aux input": st.column_config.NumberColumn("u_aux input", format="%.8f"),
    "Paired cycles": st.column_config.NumberColumn("Paired cycles", format="%d"),
    "u_blank (r=0)": st.column_config.NumberColumn("u_blank (r=0)", format="%.8f"),
    "Covariance term (u_blank^2)": st.column_config.TextColumn(
        "Covariance term (u_blank^2)"
    ),
    "u_blank reduction from r (%)": st.column_config.NumberColumn(
        "u_blank reduction from r (%)",
        format="%.2f",
    ),
    "u_blank (with r)": st.column_config.NumberColumn("u_blank (with r)", format="%.8f"),
    "u_blank": st.column_config.NumberColumn("u_blank", format="%.8f"),
    "Applied r": st.column_config.NumberColumn("Applied r", format="%.3f"),
    "Δ r": st.column_config.NumberColumn("Δ r", format="%.8f"),
    "u_blank (‰)": st.column_config.NumberColumn("u_blank (‰)", format="%.5f"),
    "ν_blank": st.column_config.TextColumn("ν"),
}


def _format_blank_detail_dataframe(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Format the blank detail table for display."""
    display_df = detail_df.copy()
    if "u_blank (with r)" in display_df.columns and "u_blank" in display_df.columns:
        display_df = display_df.drop(columns=["u_blank"])
    if (
        "Model" in display_df.columns
        and not display_df["Model"].astype(str).eq("3-variable").any()
    ):
        display_df = display_df.drop(
            columns=["I_aux", "u_aux SD", "u_aux input"],
            errors="ignore",
        )
    if "Covariance term (u_blank^2)" in display_df.columns:
        display_df["Covariance term (u_blank^2)"] = display_df[
            "Covariance term (u_blank^2)"
        ].map(_format_correlation_term_abs2)
    if "u_blank reduction from r (%)" in display_df.columns:
        display_df["u_blank reduction from r (%)"] = display_df[
            "u_blank reduction from r (%)"
        ].map(lambda value: _format_fixed_decimal(value, 2))
    for column in ("I_num", "I_den", "I_aux"):
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda value: _format_fixed_decimal(value, 4)
            )
    for column in (
        "u_num SD",
        "u_den SD",
        "u_aux SD",
        "u_num input",
        "u_den input",
        "u_aux input",
        "u_blank (r=0)",
        "u_blank (with r)",
        "Δ r",
    ):
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda value: _format_fixed_decimal(value, 8)
            )
    if "Applied r" in display_df.columns:
        display_df["Applied r"] = display_df["Applied r"].map(
            lambda value: _format_fixed_decimal(value, 3)
        )
    if "u_blank (‰)" in display_df.columns:
        display_df["u_blank (‰)"] = display_df["u_blank (‰)"].map(
            lambda value: _format_fixed_decimal(value, 5)
        )
    display_df["ν_blank"] = display_df["ν_blank"].map(
        lambda value: "∞" if value == float("inf") else str(int(value))
    )
    return display_df


def _render_blank_html_table(display_df: pd.DataFrame) -> None:
    """Render a blank diagnostic table with controllable header styling."""
    header_style = (
        "color:#000000 !important;"
        "-webkit-text-fill-color:#000000 !important;"
        "background:#F4F5F7;"
        "font-weight:700;"
        "text-align:left;"
    )
    header_cells = "".join(
        f'<th style="{header_style}">{escape(str(column))}</th>'
        for column in display_df.columns
    )
    body_rows = []
    text_columns = {
        "Sample",
        "Type",
        "Blanks used",
        "Blank source",
        "Model",
    }
    for _row_index, row in display_df.iterrows():
        cells = []
        for column in display_df.columns:
            align = "left" if column in text_columns else "right"
            cells.append(
                f'<td style="text-align:{align};">'
                f'{escape(str(row[column]))}</td>'
            )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    table_html = (
        '<table class="blank-detail-table">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )
    st.markdown(
        f"""
        <style>
        .blank-detail-table-wrap {{
            overflow-x: auto;
            border: 1px solid var(--t-line, #DCDFE5);
            border-radius: 6px;
            background: var(--t-bg, #FFFFFF);
        }}
        .blank-detail-table {{
            border-collapse: separate;
            border-spacing: 0;
            min-width: max-content;
            width: 100%;
            font-family: var(--t-mono, "Consolas", monospace);
            font-size: 12px;
            font-feature-settings: "tnum";
        }}
        .blank-detail-table th,
        .blank-detail-table td {{
            padding: 0.42rem 0.62rem;
            white-space: nowrap;
            border-bottom: 1px solid var(--t-line, #DCDFE5);
            color: var(--t-ink, #0B0F17);
            text-align: right;
        }}
        .blank-detail-table thead th {{
            position: sticky;
            top: 0;
            z-index: 2;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        .blank-detail-table tbody tr:nth-child(even) td {{
            background: var(--t-bg-2, #F4F5F7);
        }}
        </style>
        <div class="blank-detail-table-wrap">{table_html}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_blank_detail_download(detail_df: pd.DataFrame, ratio_name: str) -> None:
    """Render a full-precision CSV download for blank detail diagnostics."""
    safe_ratio = (
        str(ratio_name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )
    csv_text = detail_df.to_csv(index=False, float_format="%.17g")
    st.download_button(
        "Download blank detail CSV",
        csv_text,
        file_name=f"blank_detail_{safe_ratio}.csv",
        mime="text/csv",
        key=f"blank_detail_csv_{safe_ratio}",
    )


def _format_fixed_decimal(value: object, decimals: int) -> str:
    """Format a numeric value as fixed decimal text."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric_value):
        return ""
    return f"{numeric_value:.{decimals}f}"


def _format_correlation_term_abs2(value: object) -> str:
    """Format a variance-space covariance term without scientific notation."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric_value):
        return ""
    return f"{numeric_value:.8f}"

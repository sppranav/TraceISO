"""Sr-specific diagnostics for the uncertainty tab."""

from __future__ import annotations

from dataclasses import replace
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import UncertaintyConfig
from ui.formatting import format_uncertainty
from ui.tabs.uncertainty_sections.common import (
    _render_compact_summary,
    _render_equation_note_block,
)
from ui.utils import format_sample_display_label, get_cycle_ranges


def _enabled_sr_interferents(processing_config, element_config) -> set[str]:
    """Return Sr interferents enabled by the current session configuration."""
    if (
        processing_config is None
        or element_config is None
        or not getattr(processing_config, "apply_interference_correction", False)
    ):
        return set()

    enabled = set()
    for spec in getattr(element_config, "monitors", ()) or ():
        interfering_isotope = getattr(spec, "interfering_isotope", None)
        if interfering_isotope and processing_config.is_monitor_enabled(interfering_isotope):
            enabled.add(str(interfering_isotope))
    return enabled


# Interference section

def _render_interference_section(
    all_samples: list,
    ratio_name: str,
    u_config: UncertaintyConfig,
    state,
) -> None:
    """Render Engine A isobaric-interference diagnostics for the selected ratio."""
    from domain.uncertainty.engine_internal_sr import (
        _get_engine_a_chain_source,
        _get_runtime_ratio_mask,
        _resolve_active_sr_normalization_value,
        _resolve_sr_interference_reference_inputs,
    )
    from domain.uncertainty.kragten import compute_interference_uncertainty

    st.divider()
    st.subheader("Isobaric Interference (u_interf)")

    if ratio_name != "87Sr/86Sr":
        st.info("Interference diagnostics are currently available for 87Sr/86Sr only.")
        return
    if state.processing_config is None or state.element_config is None:
        st.info("Processing settings are required for interference diagnostics.")
        return

    processing_config = state.processing_config
    if not processing_config.apply_interference_correction:
        st.info("Interference correction is disabled in the processing settings.")
        return

    norm_value = _resolve_active_sr_normalization_value(
        processing_config,
        state.element_config,
    )
    if norm_value is None or norm_value <= 0:
        st.info("A valid Sr normalization value is required for interference diagnostics.")
        return

    enabled_interferents = _enabled_sr_interferents(
        processing_config,
        state.element_config,
    )
    if not enabled_interferents:
        st.info("No Sr interference monitors are enabled in the session configuration.")
        return

    active_measurements = [
        sample for sample in all_samples
        if (
            not sample.is_blank
            and not sample.is_standard
            and not sample.metadata.get("excluded", False)
        )
    ]
    if not active_measurements:
        st.info("No active non-standard samples available for interference diagnostics.")
        return

    cycle_ranges = get_cycle_ranges(state)

    def _compute_for_sample(sample_obj):
        src = _get_engine_a_chain_source(sample_obj)
        reference_inputs = _resolve_sr_interference_reference_inputs(
            processing_config,
            state.element_config,
            sample_obj,
        )
        included_isotopes = {
            "87Sr",
            "86Sr",
            reference_inputs.normalization_numerator,
            reference_inputs.normalization_denominator,
        }
        required = set(included_isotopes)
        if "87Rb" in enabled_interferents:
            included_isotopes.add("85Rb")
            required.add("85Rb")
        if {"84Kr", "86Kr"} & enabled_interferents:
            included_isotopes.add("83Kr")
            required.add("83Kr")
        if "84Kr" in enabled_interferents:
            included_isotopes.add("84Sr")
        intensities = {
            isotope: cycle_data.values.copy()
            for isotope, cycle_data in src.items()
            if isotope in included_isotopes
        }
        if not required.issubset(intensities):
            return None
        mask = _get_runtime_ratio_mask(
            sample_obj,
            ratio_name,
            cycle_ranges=cycle_ranges,
        )
        return compute_interference_uncertainty(
            intensities=intensities,
            normalization_value=float(norm_value),
            mask=mask,
            reference_inputs=reference_inputs,
            apply_iif=processing_config.apply_mass_bias_correction,
            enabled_interferents=enabled_interferents,
        )

    rows = []
    for sample_obj in active_measurements:
        result = _compute_for_sample(sample_obj)
        if result is None:
            continue
        u_rb_abs = float(result.u_rb_abs)
        u_kr86_abs = float(result.u_kr86_abs)
        u_kr84_abs = float(result.u_kr84_abs)
        u_rb_rel = float(result.u_rb_rel_permil)
        u_kr86_rel = float(result.u_kr86_rel_permil)
        u_kr84_rel = float(result.u_kr84_rel_permil)
        rows.append(
            {
                "Sample": sample_obj.name,
                # Selection and display are resolved by identity, not by name:
                # two runs may share a name, and every number in this row
                # belongs to exactly one of them.
                "_observation_id": sample_obj.observation_id,
                "_sample_label": format_sample_display_label(sample_obj),
                "n": int(sample_obj.n_cycles),
                "u_interf": float(result.u_interf_abs),
                "u_interf (‰)": float(result.u_interf_rel_permil),
                "u_Rb": u_rb_abs,
                "u_Rb^2": u_rb_abs**2,
                "u_Rb (‰)": u_rb_rel,
                "u_Rb^2 (‰^2)": u_rb_rel**2,
                "u_Kr86": u_kr86_abs,
                "u_Kr86^2": u_kr86_abs**2,
                "u_Kr86 (‰)": u_kr86_rel,
                "u_Kr86^2 (‰^2)": u_kr86_rel**2,
                "u_Kr84": u_kr84_abs,
                "u_Kr84^2": u_kr84_abs**2,
                "u_Kr84 (‰)": u_kr84_rel,
                "u_Kr84^2 (‰^2)": u_kr84_rel**2,
                "Sum u_i^2": u_rb_abs**2 + u_kr86_abs**2 + u_kr84_abs**2,
                "Sum u_i^2 (‰^2)": u_rb_rel**2 + u_kr86_rel**2 + u_kr84_rel**2,
            }
        )

    if not rows:
        st.info("No interference results are available for the selected ratio.")
        return

    detail_df = (
        pd.DataFrame(rows)
        .sort_values(["Sample", "_sample_label"])
        .reset_index(drop=True)
    )
    _interf_options = detail_df["_observation_id"].tolist()
    _interf_labels = dict(
        zip(detail_df["_observation_id"], detail_df["_sample_label"])
    )
    if st.session_state.get("uncertainty_interference_sample") not in _interf_options:
        st.session_state.pop("uncertainty_interference_sample", None)
    selected_observation_id = st.selectbox(
        "Sample for interference diagnostics",
        options=_interf_options,
        format_func=lambda key: _interf_labels.get(key, str(key)),
        key="uncertainty_interference_sample",
    )
    selected_row = detail_df.loc[
        detail_df["_observation_id"] == selected_observation_id
    ].iloc[0]
    selected_sample_name = str(selected_row["_sample_label"])
    show_rb = "87Rb" in enabled_interferents
    show_kr86 = "86Kr" in enabled_interferents
    show_kr84 = (
        "84Kr" in enabled_interferents
        and bool(np.any(np.abs(detail_df["u_Kr84"].to_numpy(dtype=np.float64)) > 0.0))
    )

    is_absolute = u_config.output_mode == "absolute_ratio"
    if is_absolute:
        summary_values = {
            "u_interf": format_uncertainty(float(selected_row["u_interf"])),
            "u_Rb": format_uncertainty(float(selected_row["u_Rb"])),
            "u_Kr86": format_uncertainty(float(selected_row["u_Kr86"])),
            "u_Kr84": format_uncertainty(float(selected_row["u_Kr84"])),
        }
    else:
        summary_values = {
            "u_interf": format_uncertainty(
                float(selected_row["u_interf (‰)"]),
                unit="‰",
            ),
            "u_Rb": format_uncertainty(
                float(selected_row["u_Rb (‰)"]),
                unit="‰",
            ),
            "u_Kr86": format_uncertainty(
                float(selected_row["u_Kr86 (‰)"]),
                unit="‰",
            ),
            "u_Kr84": format_uncertainty(
                float(selected_row["u_Kr84 (‰)"]),
                unit="‰",
            ),
        }

    overview_tab, detail_tab = st.tabs(["Overview", "Detail"])

    with overview_tab:
        summary_rows = [
            ("Selected sample", selected_sample_name),
            ("u_interf", summary_values["u_interf"]),
        ]
        if show_rb:
            summary_rows.append(("u_Rb", summary_values["u_Rb"]))
        if show_kr86:
            summary_rows.append(("u_Kr86", summary_values["u_Kr86"]))
        if show_kr84:
            summary_rows.append(("u_Kr84", summary_values["u_Kr84"]))
        _render_compact_summary(summary_rows)
        active_terms = []
        equation_terms = []
        if show_rb:
            active_terms.append("Rb")
            equation_terms.append(r"u_{\mathrm{Rb}}^2")
        if show_kr86:
            active_terms.append("Kr86")
            equation_terms.append(r"u_{\mathrm{Kr86}}^2")
        if show_kr84:
            active_terms.append("Kr84")
            equation_terms.append(r"u_{\mathrm{Kr84}}^2")
        active_terms_text = ", ".join(active_terms) if active_terms else "none"
        st.caption(
            "Kragten perturbation is applied contributor by contributor. "
            "The app perturbs one interference reference input upward by its standard uncertainty, "
            "reruns the full Sr correction chain with the same cycle mask and normalization value, "
            "then perturbs it downward and reruns the chain again. Half of the resulting ratio spread "
            f"is taken as that contributor's standard uncertainty. Active terms: {active_terms_text}. "
            "The active terms are combined in quadrature."
        )
        rss_equation = (
            r"u_{\mathrm{interf}} = \sqrt{" + " + ".join(equation_terms) + "}"
            if equation_terms
            else r"u_{\mathrm{interf}} = 0"
        )
        _render_equation_note_block(
            [
                r"R_{\mathrm{up}} = f(x_i + u_i), \quad R_{\mathrm{down}} = f(x_i - u_i)",
                r"u_i = \frac{|R_{\mathrm{up}} - R_{\mathrm{down}}|}{2}",
                rss_equation,
            ],
            title="Kragten Perturbation",
        )

    with detail_tab:
        _render_interference_html_table(
            _format_interference_detail_dataframe(
                detail_df,
                is_absolute=is_absolute,
                enabled_interferents=enabled_interferents,
            )
        )
        _render_interference_detail_download(
            detail_df,
            ratio_name=ratio_name,
            is_absolute=is_absolute,
            enabled_interferents=enabled_interferents,
        )


def _format_interference_detail_dataframe(
    detail_df: pd.DataFrame,
    *,
    is_absolute: bool,
    enabled_interferents: set[str] | None = None,
) -> pd.DataFrame:
    """Format the interference detail table for display."""
    if detail_df.empty:
        return detail_df

    working_df = _with_interference_variance_columns(
        _zero_disabled_interference_columns(detail_df, enabled_interferents),
    )
    value_columns = _interference_detail_value_columns(
        working_df,
        is_absolute=is_absolute,
        enabled_interferents=enabled_interferents,
    )
    display_df = working_df[["Sample", "n"] + value_columns].copy()
    # Show the collision-safe label so two runs sharing a name are not two
    # identical-looking rows. Absent for a caller that built the frame itself.
    if "_sample_label" in working_df.columns:
        display_df["Sample"] = working_df["_sample_label"]

    if "n" in display_df.columns:
        display_df["n"] = display_df["n"].map(_format_interference_integer)
    component_columns = [
        column
        for column in value_columns
        if "^2" not in column and column != "Sum u_i^2"
    ]
    variance_columns = [column for column in value_columns if "^2" in column]
    component_decimals = 8 if is_absolute else 5
    variance_decimals = 14 if is_absolute else 8
    for column in component_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda value: _format_interference_fixed_decimal(
                    value,
                    component_decimals,
                )
            )
    for column in variance_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                lambda value: _format_interference_fixed_decimal(
                    value,
                    variance_decimals,
                )
            )
    return display_df


def _interference_detail_value_columns(
    working_df: pd.DataFrame,
    *,
    is_absolute: bool,
    enabled_interferents: set[str] | None,
) -> list[str]:
    """Return active Sr interference detail columns for display/export."""
    show_rb = enabled_interferents is None or "87Rb" in enabled_interferents
    show_kr86 = enabled_interferents is None or "86Kr" in enabled_interferents
    show_kr84 = (
        enabled_interferents is None or "84Kr" in enabled_interferents
    ) and bool(np.any(np.abs(working_df["u_Kr84"].to_numpy(dtype=np.float64)) > 0.0))
    value_columns = []
    if is_absolute:
        if show_rb:
            value_columns.extend(["u_Rb", "u_Rb^2"])
        if show_kr86:
            value_columns.extend(["u_Kr86", "u_Kr86^2"])
        if show_kr84:
            value_columns.extend(["u_Kr84", "u_Kr84^2"])
        value_columns.extend(["Sum u_i^2", "u_interf"])
    else:
        if show_rb:
            value_columns.extend(["u_Rb (‰)", "u_Rb^2 (‰^2)"])
        if show_kr86:
            value_columns.extend(["u_Kr86 (‰)", "u_Kr86^2 (‰^2)"])
        if show_kr84:
            value_columns.extend(["u_Kr84 (‰)", "u_Kr84^2 (‰^2)"])
        value_columns.extend(["Sum u_i^2 (‰^2)", "u_interf (‰)"])
    return value_columns


def _zero_disabled_interference_columns(
    detail_df: pd.DataFrame,
    enabled_interferents: set[str] | None,
) -> pd.DataFrame:
    """Zero disabled contributors before display sums are recalculated."""
    if enabled_interferents is None:
        return detail_df

    working_df = detail_df.copy()
    disabled_prefixes = []
    if "87Rb" not in enabled_interferents:
        disabled_prefixes.append("u_Rb")
    if "86Kr" not in enabled_interferents:
        disabled_prefixes.append("u_Kr86")
    if "84Kr" not in enabled_interferents:
        disabled_prefixes.append("u_Kr84")

    for column in working_df.columns:
        if any(column.startswith(prefix) for prefix in disabled_prefixes):
            working_df[column] = 0.0
    return working_df.drop(
        columns=["Sum u_i^2", "Sum u_i^2 (‰^2)"],
        errors="ignore",
    )


def _with_interference_variance_columns(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the interference detail data has RSS variance terms."""
    working_df = detail_df.copy()
    for source, target in (
        ("u_Rb", "u_Rb^2"),
        ("u_Kr86", "u_Kr86^2"),
        ("u_Kr84", "u_Kr84^2"),
        ("u_Rb (‰)", "u_Rb^2 (‰^2)"),
        ("u_Kr86 (‰)", "u_Kr86^2 (‰^2)"),
        ("u_Kr84 (‰)", "u_Kr84^2 (‰^2)"),
    ):
        if target not in working_df.columns and source in working_df.columns:
            values = _interference_numeric_series(working_df, source)
            working_df[target] = values**2

    if "Sum u_i^2" not in working_df.columns:
        absolute_terms = [
            _interference_numeric_series(working_df, column)
            for column in ("u_Rb^2", "u_Kr86^2", "u_Kr84^2")
        ]
        working_df["Sum u_i^2"] = sum(absolute_terms)
    if "Sum u_i^2 (‰^2)" not in working_df.columns:
        relative_terms = [
            _interference_numeric_series(working_df, column)
            for column in ("u_Rb^2 (‰^2)", "u_Kr86^2 (‰^2)", "u_Kr84^2 (‰^2)")
        ]
        working_df["Sum u_i^2 (‰^2)"] = sum(relative_terms)
    return working_df


def _interference_numeric_series(detail_df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric diagnostic column, defaulting missing values to zero."""
    if column not in detail_df.columns:
        return pd.Series(0.0, index=detail_df.index)
    return pd.to_numeric(detail_df[column], errors="coerce").fillna(0.0)


def _render_interference_html_table(display_df: pd.DataFrame) -> None:
    """Render the interference diagnostic table with stable header styling."""
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
    text_columns = {"Sample"}
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
        '<table class="interference-detail-table">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )
    st.markdown(
        f"""
        <style>
        .interference-detail-table-wrap {{
            overflow-x: auto;
            border: 1px solid var(--t-line, #DCDFE5);
            border-radius: 6px;
            background: var(--t-bg, #FFFFFF);
        }}
        .interference-detail-table {{
            border-collapse: separate;
            border-spacing: 0;
            min-width: max-content;
            width: 100%;
            font-family: var(--t-mono, "Consolas", monospace);
            font-size: 12px;
            font-feature-settings: "tnum";
        }}
        .interference-detail-table th,
        .interference-detail-table td {{
            padding: 0.42rem 0.62rem;
            white-space: nowrap;
            border-bottom: 1px solid var(--t-line, #DCDFE5);
            color: var(--t-ink, #0B0F17);
            text-align: right;
        }}
        .interference-detail-table thead th {{
            position: sticky;
            top: 0;
            z-index: 2;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }}
        .interference-detail-table tbody tr:nth-child(even) td {{
            background: var(--t-bg-2, #F4F5F7);
        }}
        </style>
        <div class="interference-detail-table-wrap">{table_html}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_interference_detail_download(
    detail_df: pd.DataFrame,
    *,
    ratio_name: str,
    is_absolute: bool,
    enabled_interferents: set[str] | None = None,
) -> None:
    """Render a full-precision CSV download for Sr interference diagnostics."""
    safe_ratio = (
        str(ratio_name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )
    mode = "absolute" if is_absolute else "permil"
    working_df = _with_interference_variance_columns(
        _zero_disabled_interference_columns(detail_df, enabled_interferents),
    )
    value_columns = _interference_detail_value_columns(
        working_df,
        is_absolute=is_absolute,
        enabled_interferents=enabled_interferents,
    )
    download_df = working_df[["Sample", "n"] + value_columns].copy()
    if "_sample_label" in working_df.columns:
        # Same collision-safe label as the on-screen table, so an exported row
        # can be traced back to one observation.
        download_df.insert(1, "Run", working_df["_sample_label"])
    csv_text = download_df.to_csv(index=False, float_format="%.17g")
    st.download_button(
        "Download interference detail CSV",
        csv_text,
        file_name=f"interference_detail_{safe_ratio}_{mode}.csv",
        mime="text/csv",
        key=f"interference_detail_csv_{safe_ratio}_{mode}",
    )


def _format_interference_integer(value: object) -> str:
    """Format an integer-like diagnostic value."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric_value):
        return ""
    return str(int(numeric_value))


def _format_interference_fixed_decimal(
    value: object,
    decimals: int,
) -> str:
    """Format a numeric value as fixed decimal text without scientific notation."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric_value):
        return ""
    text = f"{numeric_value:.{decimals}f}"
    if numeric_value > 0.0 and float(text) == 0.0:
        return f"<{10 ** -decimals:.{decimals}f}"
    return text


def _render_bias_checks_section(
    all_samples: list,
    ratio_name: str,
    u_config: UncertaintyConfig,
    state,
) -> None:
    """Render optional top-down bias diagnostics for Sr Engine A."""
    from domain.uncertainty.engine_internal_sr import (
        AUTOMATIC_REFERENCE_BIAS_DISABLED_REASON,
        _get_iif_or_best_ratio,
        compute_qc_bias_term,
        compute_reference_bias_term,
    )
    from domain.uncertainty.reprod import compute_reprod
    from domain.uncertainty.runtime import _resolve_crm_certified_value

    st.divider()
    st.subheader("Reference Bias (Δ_ref)")

    if ratio_name != "87Sr/86Sr":
        st.info("Bias checks are currently available for 87Sr/86Sr only.")
        return
    if state.element_config is None or state.processing_config is None:
        st.info("Processing settings are required for Sr bias checks.")
        return

    reprod_result = compute_reprod(
        all_samples=all_samples,
        ratio_name=ratio_name,
        uncertainty_config=u_config,
        element_config=state.element_config,
        ratio_extractor=_get_iif_or_best_ratio,
    )
    certified_value = _resolve_crm_certified_value(
        state.element_config,
        state.processing_config,
        ratio_name,
    )
    default_ref_value = float(certified_value.value) if certified_value is not None else 0.0
    configured_ref_value = getattr(u_config, "sr_reference_bias_ref_value", None)
    try:
        configured_ref_value = float(configured_ref_value)
    except (TypeError, ValueError):
        configured_ref_value = 0.0
    ref_value = (
        configured_ref_value
        if np.isfinite(configured_ref_value) and configured_ref_value > 0.0
        else default_ref_value
    )
    ref_value = st.number_input(
        "Reference value R_ref for NIST SRM 987 (87Sr/86Sr)",
        min_value=0.0,
        value=float(ref_value),
        step=0.000001,
        format="%.8f",
        key="uc_sr_reference_bias_ref_value",
        help=(
            "Reference value the session standard mean is compared against. "
            "The default is the currently resolved SRM 987 reference value. "
            "The resulting offset is a diagnostic: it does not become an "
            "uncertainty contribution in this release."
        ),
    )
    u_config = replace(u_config, sr_reference_bias_ref_value=float(ref_value))
    try:
        state.uncertainty_config = u_config
    except Exception:
        pass
    ratio_scale = ref_value if ref_value > 0 else 0.0

    u_bias_ref_abs, _, bias_ref_stats = compute_reference_bias_term(
        reprod_result=reprod_result,
        certified_value=certified_value,
        element_config=state.element_config,
        ratio_name=ratio_name,
        ratio_mean=ratio_scale,
        reference_value_override=ref_value,
    )
    u_bias_qc_abs, u_bias_qc_rel_permil, qc_stats = compute_qc_bias_term(
        observed_bias_abs=float(getattr(u_config, "sr_qc_bias_abs", 0.0)),
        ratio_mean=ratio_scale,
        qc_cert_value=getattr(u_config, "sr_qc_cert_value", 0.0),
    )

    ref_enabled = u_config.is_contributor_enabled("u_bias_ref", element_symbol="Sr")
    qc_enabled = u_config.is_contributor_enabled("u_bias_qc", element_symbol="Sr")

    if not ref_enabled and not qc_enabled:
        st.info(
            "No processed-QC bias contributor is enabled. Enable "
            "'Bias of processed QC material' in Budget contributors to see results here."
        )
        return

    if ref_enabled and qc_enabled:
        left_col, right_col = st.columns(2, gap="large")
    else:
        left_col = st.container()
        right_col = st.container()

    if ref_enabled:
        with left_col:
            st.markdown("**Reference offset (Δ_ref) - diagnostic only**")
            st.warning(AUTOMATIC_REFERENCE_BIAS_DISABLED_REASON)
            _render_compact_summary(
                [
                    ("Included standards", str(int(bias_ref_stats.get("n_included", 0.0)))),
                    ("Session standard mean", f"{bias_ref_stats.get('session_mean', 0.0):.8f}"),
                    ("Reference value", f"{bias_ref_stats.get('reference_value', 0.0):.8f}"),
                    ("Δ_ref", f"{bias_ref_stats.get('delta_ref', 0.0):+.8f}"),
                    ("u_bias_ref", "unavailable - no approved model"),
                ],
            )
            _render_equation_note_block(
                [
                    r"\Delta_{\mathrm{ref}} = \mathrm{mean}(R_{\mathrm{std,session}}) - R_{\mathrm{ref}}",
                ],
                note=(
                    "The offset is reported as measured. Converting it into a "
                    "standard uncertainty requires a bias model that this "
                    "release does not assume."
                ),
            )

    if qc_enabled:
        target_col = right_col if ref_enabled and qc_enabled else left_col
        with target_col:
            st.markdown("**Bias of processed QC material**")
            _render_compact_summary(
                [
                    ("QC uncertainty numerator", f"{qc_stats.get('qc_input_abs', 0.0):.8f}"),
                    ("QC certified ratio", f"{qc_stats.get('qc_cert_value', 0.0):.8f}"),
                    ("Relative u_bias_qc", f"{u_bias_qc_rel_permil:.6f} \u2030"),
                    ("u_bias_qc, abs.", format_uncertainty(u_bias_qc_abs)),
                ],
            )
            _render_equation_note_block(
                [
                    r"u_{\mathrm{qc,input}} = \mathrm{user\ supplied\ standard\ uncertainty}",
                    r"u_{\mathrm{bias,qc,rel}} = \frac{u_{\mathrm{qc,input}}}{R_{\mathrm{qc,cert}}}",
                    r"u_{\mathrm{bias,qc,abs}} = R_{\mathrm{sample}} u_{\mathrm{bias,qc,rel}}",
                ],
                note=(
                    "The square-sum contribution is the relative term. The absolute "
                    "value is only the same term expressed on the sample ratio scale "
                    "for the internal budget."
                ),
            )

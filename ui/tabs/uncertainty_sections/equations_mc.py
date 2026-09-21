"""Equation and Monte Carlo helpers for the uncertainty tab."""

from __future__ import annotations

import dataclasses
import zlib
from typing import Optional

import numpy as np
import streamlit as st

from config.settings import (
    DEFAULT_MC_ITERATIONS_STANDARD,
    DEFAULT_K1_SAMPLE_DECOMPOSITION_PERMIL,
    DEFAULT_K2_MATRIX_SEPARATION_PERMIL,
    DEFAULT_K3_PROCEDURAL_BLANK_PERMIL,
    DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_PERMIL,
    DEFAULT_K6_MATRIX_EFFECTS_PERMIL,
    DEFAULT_K7_RESIDUAL_INTERFERENCES_DISTRIBUTION,
    DEFAULT_K7_RESIDUAL_INTERFERENCES_PERMIL,
)
from config.global_uncertainty_values_loader import KAPPA_SPECS
from domain.filters.outlier import resolve_cycle_range
from domain.uncertainty.monte_carlo import FIXED_NON_NORMAL_CONTRIBUTOR_DISTRIBUTIONS
from ui.formatting import (
    format_mc_summary_rows,
    format_uncertainty,
)
from ui.config_plotly import (
    get_plotly_config,
    PLOTLY_ANNOTATION_FONT_SIZE,
    PLOTLY_AXIS_TITLE_FONT_SIZE,
    PLOTLY_BASE_FONT_SIZE,
    PLOTLY_LEGEND_FONT_SIZE,
    PLOTLY_TITLE_FONT_SIZE,
)
from ui.theme import get_theme
from ui.utils import format_delta_html, format_name, get_cycle_ranges, get_sample_state_key


_MC_HISTOGRAM_HEIGHT = 520
_MC_TICK_FONT_SIZE = PLOTLY_BASE_FONT_SIZE + 4
_MC_AXIS_TITLE_FONT_SIZE = PLOTLY_AXIS_TITLE_FONT_SIZE + 5
_MC_TITLE_FONT_SIZE = PLOTLY_TITLE_FONT_SIZE + 1
_MC_LEGEND_FONT_SIZE = PLOTLY_LEGEND_FONT_SIZE + 4
_MC_ANNOTATION_FONT_SIZE = PLOTLY_ANNOTATION_FONT_SIZE + 1


def _deterministic_mc_seed(token: object) -> int:
    """Return a stable unsigned seed for an MC validation cache token."""
    return zlib.crc32(repr(token).encode("utf-8"))


def _equation_contributor_symbol(name: str) -> str:
    """Equation-only symbol mapping for contributor names."""
    if name == "u_std_repeatability":
        return "u_std_repeatability_sd"
    return name


def _finite_positive_float(value) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(numeric) and numeric > 0.0:
        return numeric
    return None


def _coerce_mc_iterations(value) -> int:
    """Return a UI-safe Monte Carlo iteration count."""
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = DEFAULT_MC_ITERATIONS_STANDARD
    return min(1_000_000, max(1_000, numeric))


def mc_display_cache_key(sample, ratio_name: str, config_hash: int, budget) -> str:
    """Session key for one cached Monte Carlo panel result.

    A031: three separate things decide whether a cached panel result may be
    reused, and the key needs all three, kept distinct:

    - the **observation** it belongs to - a name and a run number both repeat
      within one session, so the observation ID is what separates two of them;
    - the **numerical content** it was computed against, as the digest of the
      budget in hand - a cycle mask, a cycle window or an outlier-filter edit
      moves it without touching any setting;
    - the **transient generation**, which a committed data reduction advances
      and which carries no scientific meaning and is never exported.

    Conflating them is how a second reduction kept displaying the first
    reduction's analytical centre and expanded uncertainty as current.
    """
    from domain.uncertainty.monte_carlo import engine_b_budget_digest
    from domain.uncertainty.sr_chain_identity import active_engine_a_semantics
    from ui.invalidation import mc_display_generation

    identity_hash = _deterministic_mc_seed((
        str(getattr(sample, "observation_id", "") or ""),
        engine_b_budget_digest(budget),
        active_engine_a_semantics(sample),
        int(mc_display_generation()),
    ))
    return (
        f"_mc_result_{getattr(sample, 'run_number', 0)}_{getattr(sample, 'name', '')}"
        f"_{ratio_name}_{config_hash}_{identity_hash}"
    )


def _clear_mc_cache_for_rerun(session_state, cache_key: str) -> None:
    """Remove the exact cached result before a cross-check rerun starts."""
    session_state.pop(cache_key, None)


def _delta_display_values(budget) -> tuple[Optional[float], float, float, float]:
    """Return reference, scale, combined u, and expanded U for delta display."""
    ratio_mean = _finite_positive_float(getattr(budget, "basis_ratio_value", None))
    if ratio_mean is None:
        ratio_mean = _finite_positive_float(getattr(budget, "ratio_value", None))
    ref_value = _finite_positive_float(getattr(budget, "delta_reference_value", None))
    scale = _finite_positive_float(getattr(budget, "delta_scale_factor", None))
    if scale is None:
        scale = (ratio_mean / ref_value) if ratio_mean is not None and ref_value is not None else 1.0

    u_delta = float(getattr(budget, "u_combined_rel_permil", 0.0) or 0.0) * scale
    expanded_delta = float(getattr(budget, "coverage_factor_k", 0.0) or 0.0) * u_delta
    return ref_value, scale, u_delta, expanded_delta


def _build_rss_equation_lines(budget, *, is_absolute: bool = False) -> list[str]:
    """Build the independent-contributor RSS equation block."""
    active = [c for c in budget.contributors if c.is_active]
    report_ratio = float(getattr(budget, "ratio_value", 0.0) or 0.0)
    basis_ratio = float(getattr(budget, "basis_ratio_value", None) or report_ratio or 0.0)

    def _sample_basis_rel_permil(contributor) -> float:
        if basis_ratio > 0.0 and float(getattr(contributor, "value_abs", 0.0)) > 0.0:
            return (float(contributor.value_abs) / basis_ratio) * 1000.0
        return float(getattr(contributor, "value_rel_permil", 0.0) or 0.0)

    rss_terms = " + ".join(f"{_equation_contributor_symbol(c.name)}\u00b2" for c in active)
    rss_values = " + ".join(f"{_sample_basis_rel_permil(c):.4f}\u00b2" for c in active)
    rss_nums = " + ".join(f"{_sample_basis_rel_permil(c)**2:.6f}" for c in active)
    u_c = budget.u_combined_rel_permil
    uc_str = format_uncertainty(u_c, unit="\u2030")

    lines = [
        "RSS Combination (GUM \u00a75.1)",
        "=" * 40,
        f"u_c,rel = sqrt({rss_terms})",
        f"    = sqrt({rss_values})",
        f"    = sqrt({rss_nums})",
        f"    = {uc_str}",
    ]

    is_delta = getattr(budget, "output_mode", "") == "delta" and not is_absolute
    if is_absolute:
        uc_abs_str = format_uncertainty(budget.u_combined_abs)
        ref_value = _finite_positive_float(getattr(budget, "delta_reference_value", None))
        certified_ref = _finite_positive_float(getattr(budget, "certified_reference_value", None))
        absolute_scale = _finite_positive_float(getattr(budget, "absolute_scale_factor", None))
        anchored_absolute = (
            ref_value is not None
            and certified_ref is not None
            and absolute_scale is not None
            and (not np.isclose(absolute_scale, 1.0) or not np.isclose(basis_ratio, report_ratio))
        )
        if anchored_absolute:
            lines.extend(
                [
                    "",
                    "Absolute ratio anchoring",
                    "R_abs = (R_basis_sample / R_delta_ref) x R_cert",
                    (
                        f"      = ({basis_ratio:.10g} / {ref_value:.10g}) "
                        f"x {certified_ref:.10g}"
                    ),
                    f"      = {report_ratio:.10g}",
                    "u_abs = (u_c,rel / 1000) x R_abs",
                    f"      = ({u_c:.6f} / 1000) x {report_ratio:.10g}",
                    f"      = {uc_abs_str} (absolute ratio)",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Absolute ratio conversion",
                    "R_abs = R_report",
                    f"      = {report_ratio:.10g}",
                    "u_abs = (u_c,rel / 1000) x R_abs",
                    f"      = ({u_c:.6f} / 1000) x {report_ratio:.10g}",
                    f"      = {uc_abs_str} (absolute ratio)",
                ]
            )
    elif is_delta:
        ref_value, scale, u_delta, _expanded_delta = _delta_display_values(budget)
        uc_delta_str = format_uncertainty(u_delta, unit="\u2030")
        lines.extend(
            [
                "",
                "Delta uncertainty scaling",
                "f_delta = R_basis_sample / R_delta_ref = delta/1000 + 1",
            ]
        )
        if ref_value is not None:
            lines.append(f"        = {basis_ratio:.10g} / {ref_value:.10g} = {scale:.10g}")
        else:
            lines.append("        R_delta_ref unresolved; using f_delta = 1")
        lines.extend(
            [
                "u(delta) = u_c,rel x f_delta",
                f"         = {uc_str} x {scale:.10g}",
                f"         = {uc_delta_str}",
            ]
        )

    return lines


def _render_equations(budget, *, is_absolute: bool = False) -> None:
    """Display RSS, Welch-Satterthwaite, and expanded uncertainty."""
    from domain.uncertainty.scope import is_invalid_budget_scope
    if is_invalid_budget_scope(budget):
        st.info("Uncertainty unavailable: " + str(getattr(budget, 'scope_note', '')))
        return
    contributors = budget.contributors
    active = [c for c in contributors if c.is_active]

    if not active:
        st.info("No active contributors.")
        return

    st.code(
        "\n".join(_build_rss_equation_lines(budget, is_absolute=is_absolute)),
        language=None,
    )

    nu_eff = budget.effective_dof
    k = budget.coverage_factor_k

    ws_terms = []
    for c in active:
        dof = c.degrees_of_freedom
        dof_str = "∞" if dof == float("inf") else f"{dof:.0f}"
        ws_terms.append(f"  {c.name}: u={c.value_rel_permil:.4f}, v={dof_str}")

    nu_eff_str = f"{nu_eff:.1f}" if nu_eff != float("inf") else "∞"
    ws_lines = [
        "Welch-Satterthwaite Effective DoF (GUM \u00a7G.4)",
        "=" * 40,
        "v_eff = u_c^4 / sum(u_i^4 / v_i)",
        "",
        "Contributors:",
    ] + ws_terms + [
        "",
        f"v_eff = {nu_eff_str}",
        f"k     = t(v_eff, 95%) = {k:.3f}",
    ]
    method = getattr(budget, 'coverage_method', 'unknown')
    if method == 'fixed_k':
        ws_lines = ['Configured fixed coverage factor', f'k = {k:.3f}',
                    'No coverage probability is established by this fixed factor.',
                    'Effective DoF is not calculated in fixed-k mode.']
    elif method != 'welch_satterthwaite':
        ws_lines = ['Historical coverage method: unknown', f'Recorded k = {k:.3f}']
    elif nu_eff == float('inf'):
        ws_lines[-1] = f'k = {k:.3f}: TraceISO infinite-DoF convention (approximately 95% under a normal model).'
    st.code("\n".join(ws_lines), language=None)

    u_c = budget.u_combined_rel_permil
    is_delta = getattr(budget, "output_mode", "") == "delta" and not is_absolute
    uc_final = format_uncertainty(u_c, unit="\u2030")
    final_lines = [
        "Expanded Uncertainty (GUM \u00a76.2)",
        "=" * 40,
    ]
    if is_absolute:
        uexp_abs = format_uncertainty(budget.expanded_abs)
        final_lines.extend(
            [
                "U_abs = k x u_abs",
                f"      = {k:.3f} x {format_uncertainty(budget.u_combined_abs)}",
                f"      = {uexp_abs} (absolute ratio, k = {k:.2f})",
            ]
        )
    elif is_delta:
        _ref_value, _scale, u_delta, expanded_delta = _delta_display_values(budget)
        permil_unit = "\u2030"
        final_lines.extend(
            [
                "U(delta) = k x u(delta)",
                f"         = {k:.3f} x {format_uncertainty(u_delta, unit=permil_unit)}",
                f"         = {format_uncertainty(expanded_delta, unit=permil_unit)} (k = {k:.2f})",
            ]
        )
    else:
        u_exp = budget.expanded_rel_permil
        uexp_final = format_uncertainty(u_exp, unit="\u2030")
        final_lines.extend(
            [
                "U = k x u_c",
                f"  = {k:.3f} x {uc_final}",
                f"  = {uexp_final} (k = {k:.2f})",
            ]
        )

    st.code("\n".join(final_lines), language=None)


# Monte Carlo cross-check panel

def _render_mc_cross_check(budget, sample, ratio_name: str, u_config, state) -> None:
    """Render the fixed-size Monte Carlo cross-check panel."""
    from domain.uncertainty.mc_result import (
        attach_mc_result,
        build_mc_result_record,
        get_mc_result,
    )
    from domain.uncertainty.monte_carlo import (
        MCValidationResult,
        build_mc_debug_report,
        engine_b_budget_digest,
        engine_b_config_digest,
        monte_carlo_validate,
    )

    element_sym = state.element_config.symbol if state.element_config else ""
    resolved_engine = u_config.resolve_engine(
        element_sym, processing_config=state.processing_config
    )
    is_engine_a = resolved_engine == "internal_normalization"
    panel_title = "Monte Carlo cross-check"

    with st.expander(panel_title, expanded=False):
        if is_engine_a:
            st.caption(
                "The Monte Carlo cross-check propagates the active uncertainty "
                f"contributors for the selected {ratio_name} uncertainty budget."
            )
        else:
            st.caption(
                "The Monte Carlo cross-check propagates the active uncertainty "
                "contributors for the selected uncertainty budget."
            )

        col_iter, col_btn = st.columns([1.6, 1.0])

        validation_mode = "custom"

        with col_iter:
            n_iter = st.number_input(
                "Iterations",
                min_value=1_000,
                max_value=1_000_000,
                value=_coerce_mc_iterations(
                    getattr(u_config, "mc_iterations", DEFAULT_MC_ITERATIONS_STANDARD)
                ),
                step=1_000,
                key="mc_n_iter",
            )

        effective_n_iter = u_config.resolve_mc_iterations(
            override_mode=validation_mode,
            override_custom_iterations=int(n_iter),
        )
        # Always base the update on the CURRENT state, not the snapshot passed
        # in as u_config — avoids overwriting fields (e.g. segment_assignments)
        # that were changed after this render's u_config snapshot was taken.
        _current = state.uncertainty_config
        if (
            _current.mc_validation_mode != validation_mode
            or _current.mc_iterations != int(n_iter)
        ):
            new_config = dataclasses.replace(
                _current,
                mc_validation_mode=validation_mode,
                mc_iterations=int(n_iter),
            )
            state.uncertainty_config = new_config
            u_config = new_config
        else:
            u_config = _current

        st.caption("Custom normal/rectangular PDFs retain their declared shape; finite estimation DoF is recorded separately. Built-in finite-DoF Student-t semantics are unchanged. Comparisons are descriptive.")
        _display_mc_contributors_used(
            budget, u_config, state.custom_contributor_library, element_sym
        )

        processing_cfg = state.processing_config
        if processing_cfg is not None:
            processing_hash = (
                processing_cfg.blank_mode,
                processing_cfg.filter_method,
                processing_cfg.filter_threshold,
                processing_cfg.get_active_filter_threshold(),
                processing_cfg.enable_ssb,
                processing_cfg.enable_delta,
                processing_cfg.ssb_mode,
                processing_cfg.enable_post_ssb_outliers,
                processing_cfg.post_ssb_outlier_threshold,
                processing_cfg.apply_interference_correction,
                tuple(
                    sorted(
                        (str(k), bool(v))
                        for k, v in processing_cfg.interference_monitors_enabled.items()
                    )
                ),
                processing_cfg.apply_mass_bias_correction,
                processing_cfg.reference_material,
                processing_cfg.normalization_ratio_override,
                processing_cfg.normalization_value_override,
                processing_cfg.include_certified_uncertainty,
                processing_cfg.data_preference,
                processing_cfg.drift.enabled,
                processing_cfg.drift.ratio_name,
                processing_cfg.drift.method,
                processing_cfg.drift.degree,
                processing_cfg.drift.x_axis,
                processing_cfg.drift.apply_outlier_filter,
                processing_cfg.drift.outlier_threshold,
                processing_cfg.drift.norm_mode,
                processing_cfg.drift.norm_standard,
            )
        else:
            processing_hash = ("no_processing_config",)

        # Cache key — include uncertainty + processing-sensitive fields so
        # cached MC results are invalidated when the runtime model changes.
        _cfg_token = (
            u_config.output_mode, u_config.coverage_method, u_config.coverage_k,
            u_config.blank_correlation_method, u_config.blank_fixed_r,
            u_config.reprod_method, u_config.include_kappa_drift,
            getattr(u_config, "sr_iif_mode", "A"),
            getattr(u_config, "sr_norm_ratio_u_abs", 0.0),
            getattr(u_config, "sr_blank_3var", False),
            getattr(u_config, "k1_sample_decomposition_permil", DEFAULT_K1_SAMPLE_DECOMPOSITION_PERMIL),
            getattr(u_config, "k2_matrix_separation_permil", DEFAULT_K2_MATRIX_SEPARATION_PERMIL),
            getattr(u_config, "k3_procedural_blank_permil", DEFAULT_K3_PROCEDURAL_BLANK_PERMIL),
            getattr(
                u_config,
                "k4_bracketing_standard_heterogeneity_permil",
                DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_PERMIL,
            ),
            getattr(u_config, "k6_matrix_effects_permil", DEFAULT_K6_MATRIX_EFFECTS_PERMIL),
            getattr(u_config, "k7_residual_interferences_permil", DEFAULT_K7_RESIDUAL_INTERFERENCES_PERMIL),
            getattr(u_config, "kappa_drift_distribution", "normal"),
            getattr(u_config, "k1_sample_decomposition_distribution", "rectangular"),
            getattr(u_config, "k2_matrix_separation_distribution", "normal"),
            getattr(u_config, "k3_procedural_blank_distribution", "rectangular"),
            getattr(u_config, "k4_bracketing_standard_heterogeneity_distribution", "rectangular"),
            getattr(u_config, "k6_matrix_effects_distribution", "normal"),
            getattr(
                u_config,
                "k7_residual_interferences_distribution",
                DEFAULT_K7_RESIDUAL_INTERFERENCES_DISTRIBUTION,
            ),
            tuple(sorted(u_config.excluded_standards)),
            tuple(sorted(getattr(u_config, "contributor_enabled", {}).items())),
            tuple(sorted(getattr(u_config, "control_enabled", {}).items())),
            getattr(u_config, "mc_validation_mode", "standard"),
            int(effective_n_iter),
            processing_hash,
            str(get_cycle_ranges(state)),
        )
        from config.scientific_identity import digest, scientific_configuration
        from ui.runtime_budget_cache import _session_samples_token, _processing_config_token, _custom_contributor_token, _uncertainty_config_token
        all_samples = state.result.samples if state.has_result else []
        _cfg_token = (_cfg_token, digest(scientific_configuration(state.element_config)),
                      _processing_config_token(state.processing_config),
                      _uncertainty_config_token(u_config),
                      _session_samples_token(all_samples),
                      _custom_contributor_token(state.custom_contributor_library),
                      repr(getattr(state, "uncertainty_profile_defaults", None)))
        _cfg_hash = _deterministic_mc_seed(_cfg_token)
        mc_cache_key = mc_display_cache_key(sample, ratio_name, _cfg_hash, budget)
        # The seed stays bound to the configuration alone, so an unchanged
        # configuration still reproduces the same draws.
        mc_seed = _cfg_hash

        from domain.pb_calibration_records import governing_calibration_record

        from domain.uncertainty.scope import is_invalid_budget_scope

        calibration_record = governing_calibration_record(sample, ratio_name)
        calibrated_mc_refusal = ""
        if calibration_record is not None:
            if "delta" in (
                str(getattr(u_config, "output_mode", "") or "").strip().lower(),
                str(getattr(budget, "output_mode", "") or "").strip().lower(),
            ):
                calibrated_mc_refusal = (
                    "Monte Carlo of calibrated Pb delta values is not calculated "
                    "(not_implemented_for_calibrated_pb_delta). The absolute-ratio "
                    "cross-check is available in absolute-ratio output."
                )
            elif calibration_record.status != "applied" or budget is None or is_invalid_budget_scope(budget):
                calibrated_mc_refusal = (
                    "Monte Carlo is not available: this Pb-standard-calibrated result has no "
                    "evaluated absolute-ratio budget."
                )
        with col_btn:
            run_mc = st.button(
                "Run Monte Carlo cross-check",
                key="mc_run_btn",
                disabled=bool(calibrated_mc_refusal),
            )
        if calibrated_mc_refusal:
            run_mc = False
            st.caption(calibrated_mc_refusal)

        if run_mc:
            # Clear stale render state before work starts. A failed rerun must
            # never leave the previous result looking current.
            _clear_mc_cache_for_rerun(st.session_state, mc_cache_key)
            # Resolve certified value
            cv = _resolve_mc_certified_value(state, ratio_name)

            all_samples = state.result.samples if state.has_result else []
            _mc_ratio_values, _mc_ratio_mean, _mc_ratio_mask = _get_runtime_mc_ratio_inputs(
                sample,
                ratio_name,
                state,
            )

            progress_bar = st.progress(0.0, text="Running Monte Carlo...")

            def _progress(frac: float) -> None:
                progress_bar.progress(frac, text=f"Monte Carlo: {frac:.0%}")

            try:
                mc_result = monte_carlo_validate(
                    sample,
                    ratio_name,
                    all_samples=all_samples,
                    element_config=state.element_config,
                    uncertainty_config=u_config,
                    processing_config=state.processing_config,
                    gum_budget=budget,
                    certified_value=cv,
                    ratio_values=_mc_ratio_values,
                    ratio_mean=_mc_ratio_mean,
                    ratio_mask=_mc_ratio_mask,
                    cycle_ranges=get_cycle_ranges(state),
                    n_iter=int(effective_n_iter),
                    seed=mc_seed,
                    validation_mode=validation_mode,
                    progress_callback=_progress,
                    custom_contributor_library=state.custom_contributor_library,
                    profile_defaults=getattr(state, "uncertainty_profile_defaults", None),
                )
                # Ownership: the completed result belongs to sample/ratio
                # scientific state. This is the only place a record is
                # written; rendering below only reads it.
                attach_mc_result(
                    sample,
                    ratio_name,
                    build_mc_result_record(
                        mc_result,
                        sample=sample,
                        ratio_name=ratio_name,
                        requested_draws=int(effective_n_iter),
                    ),
                )
                # The session entry is a render convenience only: it retains
                # the draw array for the histogram, which is deliberately not
                # part of the durable record.
                st.session_state[mc_cache_key] = mc_result
                # Keep only a small per-session history; each entry may retain
                # a full Monte Carlo sample array for histogram rendering.
                mc_keys = [
                    key for key in st.session_state
                    if str(key).startswith("_mc_result_") and key != mc_cache_key
                ]
                for stale_key in mc_keys[:-2]:
                    st.session_state.pop(stale_key, None)
                progress_bar.empty()
            except ValueError as e:
                progress_bar.empty()
                st.error(str(e))
                return

        # The durable record is the authoritative stored result. It survives
        # a session-cache eviction and a configuration change, and it is what
        # every export writes. Reading it must not modify it, so nothing in
        # this block assigns to the record or to ``sample.mc_results``.
        from domain.uncertainty.sr_chain_identity import active_engine_a_semantics

        mc_record = get_mc_result(sample, ratio_name)
        if mc_record is not None:
            _display_durable_mc_record(
                mc_record,
                budget=budget,
                uncertainty_config=u_config,
                sample=sample,
                element_config=state.element_config,
                current_input_digest=resolve_live_mc_digest(mc_record, sample, ratio_name, state, budget, u_config),
                current_config_digest=engine_b_config_digest(u_config),
                # The budget in hand was recomputed from the cycle masks,
                # cycle range and filter that are active now, so comparing
                # against it is what catches a data-only change — the case a
                # configuration digest cannot see.
                current_budget_digest=engine_b_budget_digest(budget),
                # Sr records are current only for the correction method in force.
                current_semantics_version=active_engine_a_semantics(
                    sample, state.element_config
                ),
            )

        # Cached in-session result, retained only for the draw-array views.
        mc_result: Optional[MCValidationResult] = st.session_state.get(mc_cache_key)
        if mc_result is None:
            if mc_record is None:
                st.caption("Click the button above to run the Monte Carlo cross-check.")
        else:
            if getattr(mc_result, "status", "completed") == "not_supported":
                st.warning(
                    f"Monte Carlo not supported: {mc_result.scope_note} "
                    f"({mc_result.reason_code})"
                )
            else:
                _display_mc_result(mc_result, budget, ratio_name=ratio_name)
            if getattr(mc_result, "scope_note", ""):
                st.warning(f"MC approximation note: {mc_result.scope_note}")
            if state.dev_mode:
                _mc_ratio_values, _mc_ratio_mean, _mc_ratio_mask = _get_runtime_mc_ratio_inputs(
                    sample,
                    ratio_name,
                    state,
                )
                try:
                    debug_report = build_mc_debug_report(
                        sample,
                        ratio_name,
                        all_samples=state.result.samples if state.has_result else [],
                        element_config=state.element_config,
                        uncertainty_config=u_config,
                        gum_budget=budget,
                        certified_value=_resolve_mc_certified_value(state, ratio_name),
                        processing_config=state.processing_config,
                        ratio_values=_mc_ratio_values,
                        ratio_mean=_mc_ratio_mean,
                        ratio_mask=_mc_ratio_mask,
                        cycle_ranges=get_cycle_ranges(state),
                        mc_result=mc_result,
                        seed=mc_seed,
                    )
                except ValueError as exc:
                    st.caption(f"MC debug unavailable: {exc}")
                else:
                    with st.expander("MC Debug Values (Dev mode)", expanded=False):
                        st.caption(
                            "Copy this block if you want to compare the exact GUM and Monte Carlo inputs."
                        )
                        from ui.diagnostics import capture_debug_report
                        capture_debug_report("MC", debug_report)
                        st.code(debug_report, language=None)


# One-release call compatibility for extensions importing the former private
# helper. Both names render the same cross-check-only surface.
_render_mc_validation = _render_mc_cross_check


_KAPPA_CONTRIBUTOR_DISTRIBUTION_ATTR = {
    spec.contributor_name: spec.distribution_attr for spec in KAPPA_SPECS.values()
}


def _resolve_mc_contributor_distribution(
    name: str,
    u_config,
    custom_contributor_library: Optional[dict],
    element_symbol: str,
) -> tuple[str, bool]:
    """Return (distribution, is_configurable) actually used by MC for *name*.

    Mirrors the lookups in domain/uncertainty/monte_carlo.py: only the SSB
    kappa terms (k1-k7), instrumental drift, and custom contributors have a
    user-configurable distribution there. A small set of contributors are
    hardcoded to a non-normal distribution (see
    ``FIXED_NON_NORMAL_CONTRIBUTOR_DISTRIBUTIONS``, e.g. u_bias_ref is always
    rectangular); every remaining contributor is drawn as normal
    unconditionally.
    """
    if name == "u_prec":
        return "scaled Student-t (SE scale, finite DoF)", False

    distribution_attr = _KAPPA_CONTRIBUTOR_DISTRIBUTION_ATTR.get(name)
    if distribution_attr is not None:
        return str(getattr(u_config, distribution_attr, "normal") or "normal"), True

    if name in ("u_k5_instrumental_drift", "u_kappa_drift"):
        return str(getattr(u_config, "kappa_drift_distribution", "normal") or "normal"), True

    if name.startswith("u_custom_"):
        for definition in (custom_contributor_library or {}).get(element_symbol, []):
            if definition.name == name:
                return str(getattr(definition, "distribution", "normal") or "normal"), True
        return "normal", True

    fixed_distribution = FIXED_NON_NORMAL_CONTRIBUTOR_DISTRIBUTIONS.get(name)
    if fixed_distribution is not None:
        return fixed_distribution, False

    return "normal", False


def _build_mc_contributor_rows(
    budget,
    u_config,
    custom_contributor_library: Optional[dict],
    element_symbol: str,
) -> list[dict]:
    """Return active budget contributors displayed as MC input rows."""
    rows: list[dict] = []
    for contributor in getattr(budget, "contributors", []) or []:
        if not getattr(contributor, "is_active", False):
            continue
        distribution, is_configurable = _resolve_mc_contributor_distribution(
            contributor.name,
            u_config,
            custom_contributor_library,
            element_symbol,
        )
        rows.append(
            {
                "Contributor": contributor.display_name or contributor.name,
                "Symbol": contributor.name,
                "Type": contributor.type_ab,
                "u_abs": float(getattr(contributor, "value_abs", 0.0) or 0.0),
                "u_rel_permil": float(
                    getattr(contributor, "value_rel_permil", 0.0) or 0.0
                ),
                "Share_%": float(
                    getattr(contributor, "percentage_contribution", 0.0) or 0.0
                ),
                "Distribution": distribution if is_configurable else f"{distribution} (fixed)",
                "DoF": float(getattr(contributor, "degrees_of_freedom", float("inf"))),
            }
        )
    return rows


def _mc_cross_check_method_note(budget) -> str:
    """Describe the fixed-size method without assigning a verdict."""
    has_finite_type_a = any(
        getattr(contributor, "is_active", False)
        and str(getattr(contributor, "type_ab", "")).upper() == "A"
        and np.isfinite(float(getattr(contributor, "degrees_of_freedom", float("inf"))))
        for contributor in getattr(budget, "contributors", []) or []
    )
    if has_finite_type_a:
        return (
            "Fixed-size Monte Carlo with an empirical central 95% interval. "
            "Finite-DoF Type A contributors use Student-t tails."
        )
    return "Fixed-size Monte Carlo with an empirical central 95% interval."


def _display_mc_contributors_used(
    budget,
    u_config,
    custom_contributor_library: Optional[dict],
    element_symbol: str,
) -> None:
    """Show the active GUM contributors used by MC for this sample budget."""
    import pandas as pd

    rows = _build_mc_contributor_rows(
        budget, u_config, custom_contributor_library, element_symbol
    )
    with st.expander("Monte Carlo contributors", expanded=False):
        if not rows:
            st.caption("No active uncertainty contributors are available for this budget.")
            return

        st.caption("Distributions shown are those used in Monte Carlo sampling.")
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Distribution": st.column_config.TextColumn(
                    "Distribution",
                    help=(
                        "'fixed' identifies a contributor whose sampling distribution "
                        "is defined by its uncertainty model rather than a session setting."
                    ),
                ),
                "u_abs": st.column_config.NumberColumn("u abs", format="%.6g"),
                "u_rel_permil": st.column_config.NumberColumn(
                    "u rel (per mil)",
                    format="%.4g",
                ),
                "Share_%": st.column_config.NumberColumn("Share %", format="%.2f"),
                "DoF": st.column_config.NumberColumn("DoF", format="%.4g"),
            },
        )


def resolve_live_mc_digest(record, sample, ratio_name, state, budget, u_config=None):
    """Resolve the same effective inputs as the Run MC action, without a draw."""
    if record.engine not in {"internal_normalization", "pb_tl_external_normalization"}:
        return ""
    if "pb_standard_calibration" in record.semantics_version:
        return ""
    from domain.uncertainty.mc_freshness import resolve_current_mc_input_digest
    values, mean, mask = _get_runtime_mc_ratio_inputs(sample, ratio_name, state)
    try:
        return resolve_current_mc_input_digest(
            record, sample, ratio_name, all_samples=state.result.samples,
            element_config=state.element_config,
            uncertainty_config=u_config if u_config is not None else state.uncertainty_config,
            processing_config=state.processing_config, gum_budget=budget,
            certified_value=_resolve_mc_certified_value(state, ratio_name),
            ratio_values=values, ratio_mean=mean, ratio_mask=mask,
            cycle_ranges=get_cycle_ranges(state),
            custom_contributor_library=state.custom_contributor_library,
            profile_defaults=getattr(state, "uncertainty_profile_defaults", None),
        )
    except (ValueError, TypeError, KeyError):
        # Complete historical evidence was checked before extraction. A live
        # input that can no longer support replay is a known mismatch.
        return "current-replay-unavailable"


def _get_runtime_mc_ratio_inputs(sample, ratio_name: str, state) -> tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    """Return runtime-filtered ratio values, mean, and inclusion mask for MC."""
    from domain.filters.outlier import apply_filter, get_filtered_values
    from domain.ratio_selection import get_best_ratio_data as _get_ratio

    ratio_cd = _get_ratio(sample, ratio_name)
    if ratio_cd is None:
        return None, None, None

    cycle_ranges = get_cycle_ranges(state)
    filter_method = (
        state.processing_config.filter_method
        if state.processing_config else "None"
    )
    filter_threshold = (
        state.processing_config.get_active_filter_threshold()
        if state.processing_config else 2.0
    )
    ratio_values = get_filtered_values(
        ratio_cd.values,
        ratio_cd.mask,
        sample.name,
        sample_key=get_sample_state_key(sample),
        cycle_ranges=cycle_ranges,
        filter_method=filter_method,
        filter_threshold=filter_threshold,
    )
    runtime_mask = np.asarray(ratio_cd.mask, dtype=bool).copy()
    cycle_range = resolve_cycle_range(
        cycle_ranges,
        sample_name=sample.name,
        sample_key=get_sample_state_key(sample),
    )
    if cycle_range is not None:
        start_idx = max(int(cycle_range[0]) - 1, 0)
        end_idx = min(int(cycle_range[1]), len(runtime_mask))
        range_mask = np.zeros(len(runtime_mask), dtype=bool)
        if end_idx > start_idx:
            range_mask[start_idx:end_idx] = True
        runtime_mask &= range_mask
        if not (int(cycle_range[0]) <= 1 and int(cycle_range[1]) >= len(ratio_cd.values)):
            finite_visible_mask = runtime_mask & np.isfinite(ratio_cd.values)
            valid_indices = np.where(finite_visible_mask)[0]
            if len(valid_indices) >= 3 and filter_method != "None":
                filtered = apply_filter(
                    ratio_cd.values[valid_indices],
                    filter_method,
                    filter_threshold,
                )
                runtime_mask[valid_indices] = filtered.mask
    if len(ratio_values) < 2:
        return ratio_values, None, runtime_mask

    finite = ratio_values[np.isfinite(ratio_values)]
    ratio_mean = float(np.mean(finite)) if len(finite) > 0 else None
    return ratio_values, ratio_mean, runtime_mask


def _resolve_mc_certified_value(state, ratio_name: str):
    """Use the runtime authority, including refusal of unresolved explicit CRMs."""
    from domain.uncertainty.runtime import _resolve_crm_certified_value

    if state.element_config is None or state.processing_config is None:
        return None

    return _resolve_crm_certified_value(
        state.element_config, state.processing_config, ratio_name,
    )


def _display_durable_mc_record(
    record,
    *,
    budget=None,
    uncertainty_config=None,
    sample=None,
    element_config=None,
    current_config_digest: str = "",
    current_budget_digest: str = "",
    current_semantics_version: str = "",
    current_input_digest: str = "",
) -> None:
    """Render the stored Monte Carlo record.

    Read-only by construction: it takes the frozen record, formats it through
    the shared presentation helpers, and writes nothing back. The canonical
    machine values in the record are never replaced by their display strings.
    """
    from domain.uncertainty.mc_result import (
        MC_FRESHNESS_CURRENT,
        MC_FRESHNESS_UNKNOWN,
        mc_freshness_label,
        mc_record_freshness,
        mc_record_freshness_for_budget,
    )

    st.markdown("**Stored Monte Carlo result**")

    if getattr(record, "status", "completed") == "not_supported":
        st.warning(
            f"Monte Carlo not supported for this engine/ratio: "
            f"{record.scope_note} ({record.reason_code})"
        )
        return

    if record.is_chain_replay:
        st.caption("Monte Carlo cross-check result.")
    elif not record.is_engine_b:
        st.caption("Monte Carlo cross-check result.")
    else:
        st.warning(
            "This stored result uses a legacy Monte Carlo method. Rerun the "
            "cross-check to update it."
        )

    # Freshness is tested on two independent axes. The configuration digest
    # catches a settings change; the budget digest catches a data-only change
    # — a cycle mask, cycle range or outlier-filter edit that leaves every
    # setting identical while moving the measured inputs. A result that fails
    # either test is stale, and saying so is the whole point: the analytical
    # budget beside it has already been recomputed, this one has not.
    freshness = (
        mc_record_freshness_for_budget(
            record, budget, uncertainty_config, sample=sample, element_config=element_config,
            current_input_digest=current_input_digest,
        )
        if budget is not None
        else mc_record_freshness(
            record,
            current_config_digest=current_config_digest,
            current_budget_digest=current_budget_digest,
            current_semantics_version=current_semantics_version,
        )
    )
    if freshness == MC_FRESHNESS_CURRENT:
        st.caption(f"Freshness: {mc_freshness_label(freshness)}")
    elif freshness == MC_FRESHNESS_UNKNOWN:
        st.info(f"Freshness: {mc_freshness_label(freshness)}")
    else:
        st.warning(mc_freshness_label(freshness))

    summary_lines = [
        f"{label}: {value}" for label, value in format_mc_summary_rows(record)
    ]
    st.code("\n".join(summary_lines), language=None)

    identity_lines = [
        f"Execution ID:      {record.execution_id}",
        f"Semantics:         {record.semantics_version}",
        f"Result space:      {record.effective_result_space or record.result_space}",
        f"Iteration space:   {record.iteration_return_space}",
        f"Post-loop transform applied: "
        f"{'yes' if record.post_loop_transform_applied else 'no'}",
        f"Sample / ratio:    {record.sample_name} / {record.ratio_name}",
        f"Bracket:           {record.prev_std_label or '-'} .. "
        f"{record.next_std_label or '-'} ({record.bracket_mode or 'n/a'})",
        f"Output mode:       {record.output_mode or 'n/a'}",
        f"Draws:             {record.completed_draws:,} completed of "
        f"{record.requested_draws:,} requested",
        f"Seed / RNG:        {record.seed} / {record.bit_generator}",
        f"Interval:          {record.coverage_probability:.4g} "
        f"{record.interval_convention}, method={record.percentile_method}",
        f"Config / input:    {record.config_digest} / {record.input_digest}",
        f"Software:          TraceISO {record.app_version}, "
        f"NumPy {record.numpy_version}, Python {record.python_version}",
        f"Schema:            {record.schema_name} v{record.schema_version}",
    ]
    with st.expander("Stored result provenance", expanded=False):
        st.code("\n".join(identity_lines), language=None)
        if record.contributors:
            placement_lines = [
                f"{item.name}: placement={item.placement}"
                + (f", distribution={item.distribution}" if item.distribution else "")
                for item in record.contributors
            ]
            st.code("\n".join(placement_lines), language=None)

    for warning_text in record.warnings:
        st.warning(warning_text)
    if record.scope_note:
        st.info(f"MC approximation note: {record.scope_note}")


def _mc_moment_note(mc_result) -> str:
    """Explain why a summary moment is withheld, when it is."""
    status = str(getattr(mc_result, "moment_status", "") or "")
    dof = getattr(mc_result, "min_type_a_dof", None)
    suffix = f" (lowest Type A DoF: {float(dof):g})" if dof is not None else ""
    if status == "VARIANCE_UNDEFINED":
        return (
            "The Student-t variance is undefined at these degrees of freedom, so "
            "the MC standard deviation is not reported as a dispersion estimate"
            + suffix
            + ". The central 95% percentile interval remains valid."
        )
    if status == "MEAN_AND_VARIANCE_UNDEFINED":
        return (
            "The Student-t mean and variance are both undefined at these degrees "
            "of freedom, so neither is reported as a scientific value"
            + suffix
            + ". The central 95% percentile interval remains valid."
        )
    return ""


def _format_optional_value(value: Optional[float], spec: str = ".6f") -> str:
    """Render a suppressed summary moment without inventing a number."""
    if value is None:
        return "not reported"
    return format(float(value), spec)


def _format_optional_uncertainty(value: Optional[float]) -> str:
    if value is None:
        return "not reported"
    return format_uncertainty(value)


def _display_mc_result(mc_result, budget, *, ratio_name: Optional[str] = None) -> None:
    """Display descriptive cross-check diagnostics and the sample histogram."""

    std_difference_pct = getattr(mc_result, "std_difference_pct", None)
    st.info(
        "Analytical and Monte Carlo results are shown for comparison; "
        "no pass/fail criterion is applied."
    )
    if std_difference_pct is not None:
        st.markdown(
            f"MC SD difference from analytical $u_c$: **{std_difference_pct:.1f} %**"
        )
    moment_note = _mc_moment_note(mc_result)
    if moment_note:
        st.warning(moment_note)
    for warning_text in getattr(mc_result, "warnings", ()) or ():
        st.warning(warning_text)
    st.caption(_mc_cross_check_method_note(budget))

    # Comparison table
    is_delta = getattr(budget, "output_mode", "") == "delta"
    # For the histogram / MC column, use the MC mean as the reference centre.
    # The GUM centre is the midpoint of its own interval, not the MC mean.
    # These centres may differ; the comparison remains descriptive.
    gum_centre = (mc_result.gum_lower + mc_result.gum_upper) / 2.0
    if is_delta:
        ratio_mean = mc_result.mc_mean
    else:
        ratio_mean = budget.ratio_value if budget.ratio_value else mc_result.mc_mean
    if ratio_mean is None:
        # The MC mean is withheld at low degrees of freedom; fall back to the
        # analytical centre rather than plotting an invented value.
        ratio_mean = gum_centre
    k = budget.coverage_factor_k

    u_c = float(getattr(mc_result, "gum_u_c", 0.0) or 0.0)

    st.markdown("**Results**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Analytical propagation**")
        st.code(
            f"Result:           {gum_centre:.6f}\n"
            f"u_c:              {format_uncertainty(u_c)}\n"
            f"U (k = {k:.2f}):     {format_uncertainty(mc_result.gum_u_expanded)}\n"
            f"Expanded interval (k = {k:.2f}):\n"
            f"[{mc_result.gum_lower:.6f}, {mc_result.gum_upper:.6f}]",
            language=None,
        )
    with col2:
        st.markdown(
            f"**Monte Carlo** ({mc_result.n_iter:,} iterations)"
        )
        mc_std_shown = mc_result.mc_std
        if mc_std_shown is None:
            mc_std_shown = getattr(mc_result, "mc_std_diagnostic", None)
            std_label = "SD:       " if mc_std_shown is None else "SD*:      "
        else:
            std_label = "SD:       "
        st.code(
            f"Mean:     {_format_optional_value(mc_result.mc_mean)}\n"
            f"{std_label}{_format_optional_uncertainty(mc_std_shown)}\n"
            f"Empirical central 95% interval:\n"
            f"[{mc_result.mc_lower_95:.6f}, {mc_result.mc_upper_95:.6f}]",
            language=None,
        )
        if mc_result.mc_std is None and mc_std_shown is not None:
            st.caption(
                "* diagnostic only — not a dispersion estimate at these degrees of freedom."
            )

    n_dropped = int(getattr(mc_result, "n_dropped", 0) or 0)
    if n_dropped:
        st.warning(
            f"Excluded {n_dropped:,} non-finite Monte Carlo iteration(s) from summary statistics."
        )

    st.markdown("**Convergence diagnostics**")
    st.code(
        f"Mean stability: {mc_result.convergence_mean_pct:.2f}%\n"
        f"SD stability: {mc_result.convergence_std_pct:.2f}%\n"
        f"95% half-width stability: {mc_result.convergence_half_width_pct:.2f}%\n"
        f"2.5% quantile stability: {mc_result.convergence_q025_pct:.2f}%\n"
        f"97.5% quantile stability: {mc_result.convergence_q975_pct:.2f}%",
        language=None,
    )

    # Histogram of MC distribution with GUM overlay. Uses the stored MC samples
    # when available; falls back to a Gaussian approximation from (mc_mean,
    # mc_std) otherwise.
    _render_mc_histogram(
        mc_result,
        ratio_mean,
        is_delta=is_delta,
        ratio_name=ratio_name,
        coverage_factor_k=k,
    )

    if mc_result.convergence_checkpoints:
        import pandas as pd

        checkpoint_df = pd.DataFrame(
            [
                {
                    "n_iter": count,
                    "mean": mean_val,
                    "sd": std_val,
                    "q2.5": q025,
                    "q97.5": q975,
                    "half_width": (q975 - q025) / 2.0,
                }
                for count, mean_val, std_val, q025, q975 in mc_result.convergence_checkpoints
            ]
        )
        with st.expander("Cumulative Monte Carlo checkpoints", expanded=False):
            st.caption(
                "Stability values above compare the earlier checkpoints against "
                "the final run."
            )
            st.dataframe(
                checkpoint_df,
                width="stretch",
                hide_index=True,
            )


def _render_mc_histogram(
    mc_result,
    ratio_mean: float,
    *,
    is_delta: bool = False,
    ratio_name: Optional[str] = None,
    coverage_factor_k: Optional[float] = None,
) -> None:
    """Render a schematic histogram of the MC distribution using normal approximation."""
    fig = _build_mc_histogram_figure(
        mc_result,
        ratio_mean,
        is_delta=is_delta,
        ratio_name=ratio_name,
        coverage_factor_k=coverage_factor_k,
    )
    st.plotly_chart(fig, width="stretch", config=get_plotly_config())


def _build_mc_histogram_figure(
    mc_result,
    ratio_mean: float,
    *,
    is_delta: bool = False,
    ratio_name: Optional[str] = None,
    coverage_factor_k: Optional[float] = None,
):
    """Build the MC distribution figure.

    Uses a real histogram when mc_samples are available; falls back to a
    Gaussian approximation from (mc_mean, mc_std) otherwise.
    """
    import plotly.graph_objects as go

    theme = get_theme()
    palette = theme.palette
    mc = mc_result

    fig = go.Figure()

    samples = getattr(mc_result, "mc_samples", None)
    if samples is not None and len(samples) >= 2:
        # Real histogram from stored MC samples
        counts, bin_edges = np.histogram(samples, bins="auto", density=True)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        fig.add_trace(go.Bar(
            x=bin_centres,
            y=counts,
            width=float(bin_edges[1] - bin_edges[0]),
            marker_color=palette.layer_blank_light,
            marker_line=dict(color=palette.layer_blank, width=1),
            name="MC distribution",
            hovertemplate="Value: %{x:.6g}<br>Density: %{y:.4g}<extra></extra>",
        ))
        y_top = float(np.max(counts) * 1.05) if len(counts) else 1.0
    elif mc.mc_mean is not None and mc.mc_std:
        # Gaussian approximation fallback (mc_samples not available)
        x = np.linspace(mc.mc_mean - 4 * mc.mc_std, mc.mc_mean + 4 * mc.mc_std, 200)
        y = np.exp(-0.5 * ((x - mc.mc_mean) / mc.mc_std) ** 2) / (mc.mc_std * np.sqrt(2 * np.pi))
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            fill="tozeroy",
            fillcolor=palette.layer_blank_light,
            line=dict(color=palette.layer_blank, width=2),
            name="MC distribution (Gaussian approx.)",
        ))
        y_top = float(np.max(y) * 1.05) if len(y) else 1.0
    else:
        # Neither stored draws nor a reportable mean/SD: show the interval only.
        y_top = 1.0

    def _add_interval_line(
        bound: float,
        label: str,
        *,
        color: str,
        dash: str,
        legendgroup: Optional[str] = None,
        showlegend: bool = True,
        hover_label: Optional[str] = None,
    ) -> None:
        fig.add_trace(
            go.Scatter(
                x=[bound, bound],
                y=[0.0, y_top],
                mode="lines",
                line=dict(color=color, width=2, dash=dash),
                name=label,
                legendgroup=legendgroup,
                showlegend=showlegend,
                hovertemplate=f"{hover_label or label}: %{{x:.6g}}<extra></extra>",
            )
        )

    # Keep interval labels in the legend/hover layer; in-plot annotations collide
    # when GUM and MC bounds are close.
    _add_interval_line(
        mc.mc_lower_95,
        "MC 95% interval",
        color=palette.layer_blank,
        dash="dash",
        legendgroup="mc_interval",
        hover_label="MC 2.5%",
    )
    _add_interval_line(
        mc.mc_upper_95,
        "MC 95% interval",
        color=palette.layer_blank,
        dash="dash",
        legendgroup="mc_interval",
        showlegend=False,
        hover_label="MC 97.5%",
    )

    k_label = (
        f"k = {coverage_factor_k:.3g}"
        if coverage_factor_k is not None and np.isfinite(coverage_factor_k)
        else "k unavailable"
    )
    analytical_label = f"Analytical expanded interval ({k_label})"
    _add_interval_line(
        mc.gum_lower,
        analytical_label,
        color=palette.layer_interference,
        dash="dash",
        legendgroup="analytical_interval",
        hover_label="Analytical lower",
    )
    _add_interval_line(
        mc.gum_upper,
        analytical_label,
        color=palette.layer_interference,
        dash="dash",
        legendgroup="analytical_interval",
        showlegend=False,
        hover_label="Analytical upper",
    )

    if mc.mc_mean is not None:
        _add_interval_line(
            mc.mc_mean,
            "MC mean",
            color=palette.figure_ink,
            dash="solid",
        )

    unit_suffix = "‰" if is_delta else ""
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.96,
        xanchor="right",
        yanchor="top",
        align="left",
        showarrow=False,
        text=(
            f"n = {mc.n_iter:,}<br>"
            f"MC SD = {_format_optional_uncertainty(mc.mc_std)}{unit_suffix}<br>"
            f"u<sub>c</sub> = "
            f"{format_uncertainty(float(getattr(mc, 'gum_u_c', 0.0) or 0.0))}"
            f"{unit_suffix}"
        ),
        font=dict(size=_MC_ANNOTATION_FONT_SIZE, color=palette.annotation_text_color),
        bgcolor=palette.annotation_bg,
        bordercolor=palette.annotation_border,
        borderwidth=1,
    )

    if ratio_name:
        x_axis_title = (
            format_delta_html(ratio_name) if is_delta else format_name(ratio_name)
        )
    else:
        x_axis_title = "Delta value (‰)" if is_delta else "Ratio value"

    fig.update_layout(
        title="Monte Carlo cross-check" if getattr(mc, "semantics_version", "") else None,
        xaxis_title=x_axis_title,
        yaxis_title="Density",
        height=_MC_HISTOGRAM_HEIGHT,
        margin=dict(l=80, r=80, t=60, b=70),
        showlegend=True,
        title_font=dict(size=_MC_TITLE_FONT_SIZE),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor=palette.annotation_bg,
            bordercolor=palette.annotation_border,
            borderwidth=1,
            font=dict(size=_MC_LEGEND_FONT_SIZE),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showline=True,
        tickfont=dict(size=_MC_TICK_FONT_SIZE),
        title_font=dict(size=_MC_AXIS_TITLE_FONT_SIZE),
    )
    fig.update_yaxes(
        range=[0, y_top],
        showgrid=False,
        zeroline=False,
        showline=True,
        tickfont=dict(size=_MC_TICK_FONT_SIZE),
        title_font=dict(size=_MC_AXIS_TITLE_FONT_SIZE),
    )

    theme.apply_to_figure(fig, profile="distribution")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(range=[0, y_top], autorange=False, showgrid=False)
    return fig

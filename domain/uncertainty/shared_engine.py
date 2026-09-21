"""Shared scaffolding and combination logic for uncertainty engines."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from config.settings import UncertaintyConfig
from domain.models import UncertaintyBudget, UncertaintyContributor
from domain.uncertainty.welch_satterthwaite import (
    coverage_factor as ws_coverage_factor,
    effective_dof_from_contributors,
    effective_dof_from_contributors_permil,
)


def combine_and_build_budget_shared(
    *,
    engine: str,
    contributors: List[UncertaintyContributor],
    ratio_mean: float,
    n_cycles: int,
    uncertainty_config: UncertaintyConfig,
    delta_reference_value: Optional[float] = None,
    certified_reference_value: Optional[float] = None,
    anchor_absolute_to_certified: bool = False,
    use_abs_dof: bool = False,
    **extra_fields,
) -> UncertaintyBudget:
    """Centralized RSS combination, Welch-Satterthwaite, and budget assembly for all engines."""
    active = [c for c in contributors if c.is_active]

    invalid = [c.name for c in active if not np.isfinite(c.value_abs) or not np.isfinite(c.value_rel_permil) or c.value_abs < 0 or c.value_rel_permil < 0]
    # Bound squaring/fourth powers used by RSS and Welch-Satterthwaite.
    if invalid or not np.isfinite(ratio_mean) or any(max(abs(c.value_abs), abs(c.value_rel_permil)) > np.finfo(float).max ** 0.25 for c in active):
        from domain.uncertainty.eligibility import required_contributor_unavailable_budget
        budget = required_contributor_unavailable_budget(engine=engine, output_mode=uncertainty_config.output_mode,
            contributor_name=", ".join(invalid) or "numerical combination", reason="nonfinite, negative or overflowing required uncertainty input",
            basis_ratio_value=ratio_mean, ratio_value=ratio_mean, n_cycles=n_cycles)
        budget.contributors = contributors
        return budget

    if use_abs_dof:
        # RSS in absolute units (Engine A Sr convention)
        u_c_sq_abs = sum(c.value_abs ** 2 for c in active)
        u_c_abs = np.sqrt(u_c_sq_abs) if u_c_sq_abs > 0 else 0.0
        u_c_rel_permil = (u_c_abs / ratio_mean) * 1000.0 if ratio_mean else 0.0

        # Variance-share percentages in absolute space
        if u_c_sq_abs > 0:
            for c in contributors:
                c.percentage_contribution = (c.value_abs ** 2 / u_c_sq_abs) * 100.0 if c.is_active else 0.0
        else:
            for c in contributors:
                c.percentage_contribution = 0.0

        # Welch-Satterthwaite in absolute space
        if uncertainty_config.coverage_method == "welch_satterthwaite":
            nu_eff = effective_dof_from_contributors(active)
            k = ws_coverage_factor(nu_eff)
        else:
            nu_eff = float("inf")
            k = uncertainty_config.coverage_k

        u_exp_rel_permil = k * u_c_rel_permil
        u_exp_abs = k * u_c_abs
        report_ratio_value = ratio_mean

    else:
        # RSS in relative permil space (Engine B & C convention)
        u_c_sq = sum(c.value_rel_permil ** 2 for c in active)
        u_c_rel_permil = np.sqrt(u_c_sq) if u_c_sq > 0 else 0.0

        delta_scale_factor: Optional[float] = None
        absolute_scale_factor: Optional[float] = None
        report_ratio_value = ratio_mean

        if engine == "ssb_delta" and uncertainty_config.output_mode == "delta":
            if delta_reference_value is not None and delta_reference_value > 0.0:
                delta_scale_factor = ratio_mean / delta_reference_value
            else:
                delta_scale_factor = 1.0
            u_c_abs = u_c_rel_permil * abs(delta_scale_factor)
        else:
            absolute_scale_factor = 1.0
            if (
                anchor_absolute_to_certified
                and delta_reference_value is not None
                and certified_reference_value is not None
                and delta_reference_value > 0.0
            ):
                absolute_scale_factor = certified_reference_value / delta_reference_value
                report_ratio_value = ratio_mean * absolute_scale_factor
            u_c_abs = (u_c_rel_permil / 1000.0) * abs(report_ratio_value) if report_ratio_value else 0.0

        # Percentage contributions in relative space
        if u_c_sq > 0:
            for c in contributors:
                c.percentage_contribution = (c.value_rel_permil ** 2 / u_c_sq) * 100.0 if c.is_active else 0.0
        else:
            for c in contributors:
                c.percentage_contribution = 0.0

        # Welch-Satterthwaite in relative space
        if uncertainty_config.coverage_method == "welch_satterthwaite":
            nu_eff = effective_dof_from_contributors_permil(active)
            k = ws_coverage_factor(nu_eff)
        else:
            nu_eff = float("inf")
            k = uncertainty_config.coverage_k

        u_exp_rel_permil = k * u_c_rel_permil
        u_exp_abs = k * u_c_abs

        # Populate Engine B specific scaling fields
        extra_fields.update({
            "delta_scale_factor": delta_scale_factor,
            "absolute_scale_factor": absolute_scale_factor,
            "delta_reference_value": delta_reference_value,
            "certified_reference_value": certified_reference_value,
        })

    if not all(np.isfinite(v) for v in (u_c_abs, u_c_rel_permil, u_exp_abs, u_exp_rel_permil, k)):
        from domain.uncertainty.eligibility import required_contributor_unavailable_budget
        return required_contributor_unavailable_budget(engine=engine, output_mode=uncertainty_config.output_mode,
            contributor_name="combination", reason="nonfinite combined uncertainty", basis_ratio_value=ratio_mean,
            ratio_value=ratio_mean, n_cycles=n_cycles)

    variance_total = u_c_sq_abs if use_abs_dof else u_c_sq
    dominant = (
        max(active, key=lambda c: (c.percentage_contribution, c.name)).name
        if active and variance_total > 0
        else ""
    )

    budget = UncertaintyBudget(
        contributors=contributors,
        u_combined_abs=u_c_abs,
        u_combined_rel_permil=u_c_rel_permil,
        expanded_abs=u_exp_abs,
        expanded_rel_permil=u_exp_rel_permil,
        effective_dof=nu_eff,
        coverage_factor_k=k,
        coverage_method=uncertainty_config.coverage_method,
        coverage_probability=0.95 if uncertainty_config.coverage_method == "welch_satterthwaite" and np.isfinite(nu_eff) else None,
        coverage_factor_rule=("configured_fixed" if uncertainty_config.coverage_method == "fixed_k"
                              else "student_t_two_sided" if np.isfinite(nu_eff)
                              else "infinite_dof_k2_convention"),
        coverage_semantics_version="traceiso.budget_coverage.v1",
        dominant_contributor=dominant,
        engine=engine,
        output_mode=uncertainty_config.output_mode if engine == "ssb_delta" else "absolute_ratio",
        ratio_value=report_ratio_value,
        basis_ratio_value=ratio_mean,
        n_cycles=n_cycles,
        **extra_fields,
    )

    return budget

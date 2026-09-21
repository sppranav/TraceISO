"""JSON export for TraceISO."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from config.constants import APP_VERSION
from config.settings import ProcessingConfig
from domain.layers import cycle_data_equal
from domain.models import CycleData, ProcessingResult, Sample, UncertaintyBudget, UncertaintyContributor
from domain.uncertainty.mc_result import (
    MC_RESULT_SCHEMA_NAME,
    MC_RESULT_SCHEMA_VERSION,
    has_mc_results,
    log_mc_results_exported,
    mc_record_freshness_for_budget,
)
from domain.uncertainty.scope import (
    budget_scope_label,
    budget_scope_note,
    is_invalid_budget_scope,
    uncertainty_scope_payload,
)
from file_io.sanitize import safe_float
from domain.provenance import normalize_provenance

#: Format token used in the ``engine_b_mc.exported`` lifecycle record. It is a
#: fixed word, never a filename or a path.
_MC_EXPORT_FORMAT = "json"


def export_to_json(
    result: ProcessingResult,
    *,
    include_raw: bool = True,
    include_corrected: bool = True,
    include_cycle_data: bool = False,
    include_uncertainty: bool = True,
    indent: int = 2,
    provenance: Optional[Dict[str, Any]] = None,
    uncertainty_config: Optional[object] = None,
) -> str:
    """Export a ProcessingResult to a JSON string."""
    data = _serialize_result(
        result,
        include_raw=include_raw,
        include_corrected=include_corrected,
        include_cycle_data=include_cycle_data,
        include_uncertainty=include_uncertainty,
        provenance=provenance,
        uncertainty_config=uncertainty_config,
    )
    # item 58: wrap allow_nan=False so non-serialisable values produce a
    # diagnosable message instead of a bare ValueError.
    try:
        return json.dumps(data, indent=indent if indent > 0 else None, allow_nan=False)
    except (ValueError, TypeError) as _exc:
        # Locate the first non-serialisable leaf for the error message.
        _path = _find_non_serialisable_path(data)
        _hint = f" (first problematic key path: {_path})" if _path else ""
        raise ValueError(
            f"JSON serialisation failed: {_exc}{_hint}. "
            f"Check for NaN/Infinity floats or non-JSON-serialisable types."
        ) from _exc


def result_to_dict(
    result: ProcessingResult,
    *,
    include_raw: bool = True,
    include_corrected: bool = True,
    include_cycle_data: bool = False,
    include_uncertainty: bool = True,
    provenance: Optional[Dict[str, Any]] = None,
    uncertainty_config: Optional[object] = None,
) -> Dict[str, Any]:
    """Convert a ProcessingResult to a JSON-serializable dictionary."""
    return _serialize_result(
        result,
        include_raw=include_raw,
        include_corrected=include_corrected,
        include_cycle_data=include_cycle_data,
        include_uncertainty=include_uncertainty,
        provenance=provenance,
        uncertainty_config=uncertainty_config,
    )


def _serialize_result(
    result: ProcessingResult,
    include_raw: bool,
    include_corrected: bool,
    include_cycle_data: bool,
    include_uncertainty: bool,
    provenance: Optional[Dict[str, Any]] = None,
    uncertainty_config: Optional[object] = None,
) -> Dict[str, Any]:
    """Build the JSON-serializable dictionary from ProcessingResult."""
    out: Dict[str, Any] = {
        "export_info": {
            "version": APP_VERSION,
            "export_date": datetime.now(timezone.utc).isoformat(),
            "options": {
                "include_raw": include_raw,
                "include_corrected": include_corrected,
                "include_cycle_data": include_cycle_data,
                "include_uncertainty": include_uncertainty,
            },
        },
        "element": result.element_symbol,
        "sample_counts": result.count_by_type,
        "warnings": result.warnings,
        "quality_metrics": _sanitize_dict(result.quality_metrics),
        "samples": [
            _serialize_sample(
                s,
                include_raw=include_raw,
                include_corrected=include_corrected,
                include_cycle_data=include_cycle_data,
                include_uncertainty=include_uncertainty,
                uncertainty_config=uncertainty_config,
            )
            for s in result.samples
        ],
    }
    if provenance is not None:
        normalized_provenance = normalize_provenance(provenance)
        out["provenance"] = _sanitize_value(normalized_provenance)
        # Signed configuration payloads are already canonical JSON. Removing
        # private-looking scientific metadata would invalidate their digest.
        if normalized_provenance.get("effective_configuration"):
            out["provenance"]["effective_configuration"] = normalized_provenance["effective_configuration"]
    if include_corrected and any(getattr(s, "correction_records", None) for s in result.samples):
        from domain.pb_calibration_records import RECORD_SCHEMAS
        from domain.pb_correction_records import HG_RECORD_SCHEMA_NAME, HG_RECORD_SCHEMA_VERSION

        schemas = {"hg": {"name": HG_RECORD_SCHEMA_NAME, "version": HG_RECORD_SCHEMA_VERSION}}
        families = {f for s in result.samples for f in (getattr(s, "correction_records", None) or {})}
        for family, (name, version) in RECORD_SCHEMAS.items():
            if family in families:
                schemas[family] = {"name": name, "version": version}
        out["export_info"]["correction_record_schemas"] = schemas
    if include_uncertainty and any(sample.uncertainty for sample in result.samples):
        out["uncertainty_scope"] = uncertainty_scope_payload()
    if include_uncertainty and has_mc_results(result.samples):
        # Schema identity is stated once at the top level so a reader does not
        # have to open a sample to learn what the per-ratio records are.
        out["export_info"]["mc_result_schema"] = {
            "name": MC_RESULT_SCHEMA_NAME,
            "version": MC_RESULT_SCHEMA_VERSION,
        }
        log_mc_results_exported(result.samples, export_format=_MC_EXPORT_FORMAT)
    return out


def _serialize_sample(
    sample: Sample,
    include_raw: bool,
    include_corrected: bool,
    include_cycle_data: bool,
    include_uncertainty: bool,
    uncertainty_config: Optional[object] = None,
) -> Dict[str, Any]:
    """Serialize a single Sample to a dictionary."""
    data: Dict[str, Any] = {
        "name": sample.name,
        "observation_id": sample.observation_id,
        "type": sample.sample_type,
        "run_number": _sanitize_value(sample.run_number),
        "n_cycles": sample.n_cycles,
        "metadata": _sanitize_dict(sample.metadata),
        "warnings": list(sample.warnings),
    }
    sr_chain = _serialize_sr_correction_chain(
        sample.metadata, include_cycle_data=include_cycle_data,
    )
    if sr_chain is not None:
        data["sr_correction_chain"] = sr_chain

    # Raw ratios summary
    if include_raw and sample.ratios:
        data["raw_ratios"] = {
            name: _serialize_cycle_stats(cd) for name, cd in sample.ratios.items()
        }

    # Corrected ratios summary. A corrected layer is suppressed only when
    # the raw layer it duplicates is actually present in this file: with
    # include_raw=False the deduplication would remove the only copy of the
    # requested data, and a no-op correction (blank_mode="none", or a
    # correction that happens to cancel) is an ordinary reachable state.
    def _is_redundant_copy(cd, raw_cd) -> bool:
        return include_raw and cycle_data_equal(cd, raw_cd)

    if include_corrected:
        if sample.blank_corrected_ratios:
            payload = {
                name: _serialize_cycle_stats(cd)
                for name, cd in sample.blank_corrected_ratios.items()
                if not _is_redundant_copy(cd, sample.ratios.get(name))
            }
            if payload:
                data["blank_corrected_ratios"] = payload
        if sample.corrected_ratios:
            payload = {
                name: _serialize_cycle_stats(cd)
                for name, cd in sample.corrected_ratios.items()
                if not _is_redundant_copy(cd, sample.ratios.get(name))
            }
            if payload:
                data["corrected_ratios"] = payload
        if sample.iif_corrected_ratios:
            payload = {
                name: _serialize_cycle_stats(cd)
                for name, cd in sample.iif_corrected_ratios.items()
                if not _is_redundant_copy(cd, sample.ratios.get(name))
            }
            if payload:
                data["iif_corrected_ratios"] = payload
        if sample.drift_corrected_ratios:
            payload = {
                name: _serialize_cycle_stats(cd)
                for name, cd in sample.drift_corrected_ratios.items()
                if not _is_redundant_copy(cd, sample.ratios.get(name))
            }
            if payload:
                data["drift_corrected_ratios"] = payload
        if sample.interference_corrected_ratios:
            payload = {
                name: _serialize_cycle_stats(cd)
                for name, cd in sample.interference_corrected_ratios.items()
                if not _is_redundant_copy(cd, sample.ratios.get(name))
            }
            if payload:
                data["interference_corrected_ratios"] = payload
        if getattr(sample, "sr_standard_corrected_ratios", None):
            data["sr_standard_corrected_ratios"] = {
                name: _serialize_cycle_stats(cd)
                for name, cd in sample.sr_standard_corrected_ratios.items()
            }
        if getattr(sample, "pb_standard_corrected_ratios", None):
            data["pb_standard_corrected_ratios"] = {
                name: _serialize_cycle_stats(cd)
                for name, cd in sample.pb_standard_corrected_ratios.items()
            }
        if getattr(sample, "pb_calibrated_delta_cycles", None):
            from domain.pb_calibration_records import calibrated_delta_record

            summary = {}
            for name, cd in sample.pb_calibrated_delta_cycles.items():
                record = calibrated_delta_record(sample, name)
                summary[name] = {
                    **_serialize_cycle_stats(cd),
                    "unit": "permil",
                    "result_identity": record.result_identity if record is not None else "",
                    # The deferred combined uncertainty is absent, never zero.
                    "combined_uncertainty": None,
                    "uncertainty_status": record.uncertainty_status if record is not None else "not_calculated",
                    "uncertainty_reason_code": (
                        record.uncertainty_reason_code if record is not None else ""
                    ),
                    "selected_precision_statistic": (
                        record.precision_statistic if record is not None else "none"
                    ),
                    "selected_precision_value": (
                        record.precision_value if record is not None else None
                    ),
                    "selected_precision_label": (
                        record.precision_label if record is not None else ""
                    ),
                }
            data["pb_calibrated_delta"] = summary
        if sample.correction_records:
            # Canonical record payloads are finite by construction; written
            # through unchanged so strict readers can rebuild them exactly.
            from domain.pb_correction_records import correction_records_payload

            data["correction_records"] = correction_records_payload(sample)

    # SSB results
    if include_corrected and sample.ssb_results:
        data["ssb_results"] = _sanitize_dict(sample.ssb_results)

    if include_corrected and sample.delta_results:
        data["delta_results"] = _sanitize_dict(sample.delta_results)

    if include_uncertainty and sample.uncertainty:
        data["uncertainty"] = {
            name: _serialize_uncertainty(budget)
            for name, budget in sample.uncertainty.items()
        }

    # Durable Monte Carlo cross-check records. ``to_dict`` already returns the
    # canonical JSON-safe payload at full precision, so it is written through
    # unchanged: passing it to ``_sanitize_value`` would drop the non-finite
    # tokens the schema deliberately preserves.
    if include_uncertainty and getattr(sample, "mc_results", None):
        data["mc_results"] = {
            name: record.to_dict() for name, record in sample.mc_results.items()
        }
        # Freshness is a relationship between the stored record and the budget
        # exported alongside it, not a property of the record, so it is a
        # sibling key rather than a field inside the canonical payload. It is
        # written because a reader must not have to assume a Monte Carlo
        # result still corresponds to the data in the same export.
        budgets = getattr(sample, "uncertainty", None) or {}
        data["mc_results_freshness"] = {
            name: mc_record_freshness_for_budget(
                record, budgets.get(name), uncertainty_config, sample=sample
            )
            for name, record in sample.mc_results.items()
        }

    # Full cycle data (optional, can be large)
    if include_cycle_data:
        if sample.intensities:
            data["intensities"] = {
                name: _serialize_cycle_data(cd)
                for name, cd in sample.intensities.items()
            }
        if include_corrected and sample.corrected_intensities:
            payload = {
                name: _serialize_cycle_data(cd)
                for name, cd in sample.corrected_intensities.items()
                if not cycle_data_equal(cd, sample.intensities.get(name))
            }
            if payload:
                data["corrected_intensities"] = payload
        if include_corrected and sample.interference_corrected_intensities:
            data["interference_corrected_intensities"] = {
                name: _serialize_cycle_data(cd)
                for name, cd in sample.interference_corrected_intensities.items()
            }
        if include_raw and sample.ratios:
            data["ratio_cycles"] = {
                name: _serialize_cycle_data(cd) for name, cd in sample.ratios.items()
            }
        if include_corrected:
            corrected_cycle_layers = (
                ("blank_corrected_ratio_cycles", sample.blank_corrected_ratios),
                ("corrected_ratio_cycles", sample.corrected_ratios),
                ("iif_corrected_ratio_cycles", sample.iif_corrected_ratios),
                ("drift_corrected_ratio_cycles", sample.drift_corrected_ratios),
                ("interference_corrected_ratio_cycles", sample.interference_corrected_ratios),
                ("sr_standard_corrected_ratio_cycles", sample.sr_standard_corrected_ratios),
                ("pb_standard_corrected_ratio_cycles", sample.pb_standard_corrected_ratios),
                ("pb_calibrated_delta_cycles", sample.pb_calibrated_delta_cycles),
            )
            for output_key, layer in corrected_cycle_layers:
                if not layer:
                    continue
                payload = {
                    name: _serialize_cycle_data(cd)
                    for name, cd in layer.items()
                    if not _is_redundant_copy(cd, sample.ratios.get(name))
                }
                if payload:
                    data[output_key] = payload

    return data


def _serialize_sr_correction_chain(
    metadata: Dict[str, Any],
    *,
    include_cycle_data: bool,
) -> Optional[Dict[str, Any]]:
    """Method identity and refinement stability of the Sr correction chain (A025).

    The pipeline keeps these under private metadata keys, which the generic
    metadata export skips, so they are written here explicitly. The stability
    figures are a numerical diagnostic, not an uncertainty or accuracy claim.
    Per-cycle movements follow ``include_cycle_data``; an unavailable cycle is
    exported as null, never as zero.
    """
    method = metadata.get("_sr_chain_method")
    if not method:
        return None
    stability: Dict[str, Any] = {
        "threshold_rel": _sanitize_value(metadata.get("_sr_refinement_threshold_rel")),
        "n_compared": _sanitize_value(metadata.get("_sr_refinement_n_compared")),
        "max_abs_change": _sanitize_value(metadata.get("_sr_refinement_max_abs_change")),
        "max_rel_change": _sanitize_value(metadata.get("_sr_refinement_max_rel_change")),
        "exceeds_threshold": _sanitize_value(metadata.get("_sr_refinement_exceeds_threshold")),
    }
    if include_cycle_data:
        for key, field_name in (
            ("_sr_refinement_change_abs", "change_abs"),
            ("_sr_refinement_change_rel", "change_rel"),
        ):
            if metadata.get(key) is not None:
                stability[field_name] = _sanitize_value(metadata[key])
    return {
        "method": str(method),
        "initialization_passes": _sanitize_value(metadata.get("_sr_chain_initialization_passes")),
        "refinement_passes": _sanitize_value(metadata.get("_sr_chain_refinement_passes")),
        "refinement_stability": stability,
    }


def _serialize_cycle_stats(cd: CycleData) -> Dict[str, Any]:
    """Serialize CycleData summary statistics (not full arrays)."""
    return {
        "mean": safe_float(cd.mean),
        "sd": safe_float(cd.sd),
        "se": safe_float(cd.se),
        "rsd_percent": safe_float(cd.rsd_percent),
        "n_valid": cd.n_valid,
        "n_total": cd.n_total,
    }


def _serialize_cycle_data(cd: CycleData) -> Dict[str, Any]:
    """Serialize full CycleData including arrays."""
    return {
        "values": _array_to_list(cd.values),
        "mask": cd.mask.tolist(),
        "mean": safe_float(cd.mean),
        "sd": safe_float(cd.sd),
        "se": safe_float(cd.se),
        "n_valid": cd.n_valid,
        "n_total": cd.n_total,
    }


def _reported_contributor_abs(budget: UncertaintyBudget, c: UncertaintyContributor) -> float:
    """Resolve Engine B absolute-output contributions without altering MC inputs.

    The producer's CRM value_abs is u(C), whereas its relative field is u(C)/C.
    The shared combiner uses the latter on the reported ratio scale. Preserve
    that producer operand and use the same dimensional conversion for reporting.
    """
    if budget.engine == "ssb_delta" and budget.output_mode == "absolute_ratio":
        return abs(float(budget.ratio_value)) * float(c.value_rel_permil) / 1000.0
    return float(c.value_abs)


def _serialize_contributor(c: UncertaintyContributor, budget: UncertaintyBudget) -> Dict[str, Any]:
    """Serialize a single UncertaintyContributor to a dictionary."""
    payload = {
        "name": c.name,
        "display_name": c.display_name,
        "value_abs": safe_float(c.value_abs),
        "value_rel_permil": safe_float(c.value_rel_permil),
        "type_ab": c.type_ab,
        "degrees_of_freedom": safe_float(c.degrees_of_freedom),
        "percentage_contribution": safe_float(c.percentage_contribution),
        "description": c.description,
        "reference": getattr(c, "reference", "") or "",
        "is_active": c.is_active,
        "state": getattr(c, "state", "") or "",
        "inactive_reason": getattr(c, "inactive_reason", "") or "",
        "distribution": getattr(c, "distribution", "") or "",
    }
    if budget.engine == "ssb_delta" and budget.output_mode == "absolute_ratio":
        payload.update({
            "value_abs_basis": "producer basis; use reported_value_abs for output-unit contribution",
            "reported_value_abs": safe_float(_reported_contributor_abs(budget, c)),
            "reported_unit": "ratio",
        })
    return payload


def _compat_budget_view(budget: UncertaintyBudget) -> Dict[str, Any]:
    """Build legacy flat uncertainty fields from canonical data when needed."""
    ratio_value = safe_float(budget.ratio_value)
    n_cycles = int(getattr(budget, "n_cycles", 0) or 0)

    contributors = list(getattr(budget, "contributors", []) or [])

    def _contrib_value(name: str) -> float:
        contributor = next((c for c in contributors if c.name == name), None)
        return _reported_contributor_abs(budget, contributor) if contributor is not None else 0.0

    def _contrib_ppm(name: str) -> Optional[float]:
        contributor = next((c for c in contributors if c.name == name), None)
        if contributor is None:
            return None
        return float(contributor.value_rel_permil) * 1000.0

    u_precision = _contrib_value("u_prec")
    u_blank = _contrib_value("u_blank")
    u_type_a = budget.type_ab_abs("A")
    u_type_b = budget.type_ab_abs("B")
    u_combined = float(budget.u_combined_abs)
    u_expanded = float(budget.expanded_abs)
    coverage_factor = float(budget.coverage_factor_k)

    components = budget.compat_components_abs()
    if budget.engine == "ssb_delta" and budget.output_mode == "absolute_ratio":
        # Match the canonical relative-space RSS, including non-unit CRM scaling.
        u_type_a = float(np.sqrt(sum(_reported_contributor_abs(budget, c) ** 2
                                    for c in contributors if c.is_active and c.type_ab == "A")))
        u_type_b = float(np.sqrt(sum(_reported_contributor_abs(budget, c) ** 2
                                    for c in contributors if c.is_active and c.type_ab == "B")))
        components = {key: abs(float(budget.ratio_value)) * ppm / 1_000_000.0
                      for key, ppm in budget.compat_components_ppm().items()}

    def _ppm(value: float) -> Optional[float]:
        if ratio_value in (None, 0.0):
            return None
        return safe_float((value / ratio_value) * 1_000_000.0)

    components_ppm = budget.compat_components_ppm()
    if not components_ppm and ratio_value not in (None, 0.0):
        components_ppm = {
            key: (float(val) / ratio_value) * 1_000_000.0
            for key, val in components.items()
        }

    def _ppm_or_contrib(value: float, contributor_name: str) -> Optional[float]:
        if budget.output_mode != "delta":
            direct = _ppm(value)
            if direct is not None:
                return direct
        # Absolute delta uncertainty is in permil of delta, not amount ratio.
        # The contributor's canonical relative field supplies the unit-safe
        # conversion for both delta and absolute-ratio budgets.
        contrib_ppm = _contrib_ppm(contributor_name)
        if contrib_ppm is not None:
            return safe_float(contrib_ppm)
        return _ppm(value)

    def _aggregate_ppm(rel_permil: float, value: float) -> Optional[float]:
        """Return an aggregate legacy ppm field: a *relative* uncertainty.

        The canonical relative field is the only correct basis. Dividing the
        aggregate *absolute* field by the amount ratio happens to agree for an
        absolute-ratio budget, but for an Engine B delta budget
        ``u_combined_abs`` is the absolute uncertainty of the delta itself, in
        permil of delta, and its declared relationship to the ratio is
        ``delta_scale_factor`` -- not the ratio value. Dividing it by the ratio
        produced a number in no unit at all (1 permil on a ratio of 4 was
        exported as 250,000 ppm instead of 1,000 ppm). Converting the canonical
        relative permil field is exact in both output modes.
        """
        if rel_permil is not None:
            return safe_float(float(rel_permil) * 1000.0)
        return _ppm(value)

    def _ppm_or_type_ab_ppm(value: float, type_ab: str) -> Optional[float]:
        if budget.output_mode != "delta":
            direct = _ppm(value)
            if direct is not None:
                return direct
        val_ppm = budget.type_ab_ppm(type_ab)
        return safe_float(val_ppm)

    return {
        "ratio_value": ratio_value,
        "n_cycles": n_cycles,
        "u_precision": safe_float(u_precision),
        "u_precision_ppm": _ppm_or_contrib(u_precision, "u_prec"),
        "u_blank": safe_float(u_blank),
        "u_blank_ppm": _ppm_or_contrib(u_blank, "u_blank"),
        "u_type_a": safe_float(u_type_a),
        "u_type_a_ppm": _ppm_or_type_ab_ppm(u_type_a, "A"),
        "u_type_b": safe_float(u_type_b),
        "u_type_b_ppm": _ppm_or_type_ab_ppm(u_type_b, "B"),
        "u_combined": safe_float(u_combined),
        "u_combined_ppm": _aggregate_ppm(budget.u_combined_rel_permil, u_combined),
        "u_expanded": safe_float(u_expanded),
        "u_expanded_ppm": _aggregate_ppm(budget.expanded_rel_permil, u_expanded),
        "coverage_factor": safe_float(coverage_factor),
        "components": {k: safe_float(v) for k, v in components.items()},
        "components_ppm": {k: safe_float(v) for k, v in components_ppm.items()},
    }


def _serialize_uncertainty(budget: UncertaintyBudget) -> Dict[str, Any]:
    """Serialize an UncertaintyBudget to a dictionary."""
    compat = _compat_budget_view(budget)
    payload = {
        "contributors": [_serialize_contributor(c, budget) for c in budget.contributors],
        "u_combined_abs": safe_float(budget.u_combined_abs),
        "u_combined_rel_permil": safe_float(budget.u_combined_rel_permil),
        "expanded_abs": safe_float(budget.expanded_abs),
        "expanded_rel_permil": safe_float(budget.expanded_rel_permil),
        "effective_dof": safe_float(budget.effective_dof),
        "coverage_factor_k": safe_float(budget.coverage_factor_k),
        "coverage_method": budget.coverage_method,
        "coverage_probability": budget.coverage_probability,
        "coverage_factor_rule": budget.coverage_factor_rule,
        "coverage_semantics_version": budget.coverage_semantics_version,
        "dominant_contributor": budget.dominant_contributor,
        "engine": budget.engine,
        "output_mode": budget.output_mode,
        "budget_scope": budget.budget_scope,
        "scope_note": budget.scope_note,
        "coverage_limitations": list(getattr(budget, "coverage_limitations", ()) or ()),
        "replay_input_digest": str(getattr(budget, "replay_input_digest", "") or ""),
        "reprod_result": _sanitize_value(budget.reprod_result),
        "budget_state": budget_scope_label(budget),
        "mc_lower_95": safe_float(budget.mc_lower_95),
        "mc_upper_95": safe_float(budget.mc_upper_95),
        "mc_gum_agreement": safe_float(budget.mc_gum_agreement),
        "ratio_value": compat["ratio_value"],
        "basis_ratio_value": safe_float(getattr(budget, "basis_ratio_value", None)),
        "delta_reference_value": safe_float(getattr(budget, "delta_reference_value", None)),
        "delta_scale_factor": safe_float(getattr(budget, "delta_scale_factor", None)),
        "certified_reference_value": safe_float(getattr(budget, "certified_reference_value", None)),
        "absolute_scale_factor": safe_float(getattr(budget, "absolute_scale_factor", None)),
        "n_cycles": compat["n_cycles"],
        "u_precision": compat["u_precision"],
        "u_precision_ppm": compat["u_precision_ppm"],
        "u_blank": compat["u_blank"],
        "u_blank_ppm": compat["u_blank_ppm"],
        "u_type_a": compat["u_type_a"],
        "u_type_a_ppm": compat["u_type_a_ppm"],
        "u_type_b": compat["u_type_b"],
        "u_type_b_ppm": compat["u_type_b_ppm"],
        "u_combined": compat["u_combined"],
        "u_combined_ppm": compat["u_combined_ppm"],
        "u_expanded": compat["u_expanded"],
        "u_expanded_ppm": compat["u_expanded_ppm"],
        "coverage_factor": compat["coverage_factor"],
        "components": compat["components"],
        "components_ppm": compat["components_ppm"],
    }
    if not is_invalid_budget_scope(budget):
        return payload

    payload["scope_note"] = budget_scope_note(budget)
    for key in (
        "u_combined_abs",
        "u_combined_rel_permil",
        "expanded_abs",
        "expanded_rel_permil",
        "effective_dof",
        "coverage_factor_k",
        "mc_lower_95",
        "mc_upper_95",
        "mc_gum_agreement",
        "ratio_value",
        "basis_ratio_value",
        "delta_reference_value",
        "delta_scale_factor",
        "certified_reference_value",
        "absolute_scale_factor",
        "u_precision",
        "u_precision_ppm",
        "u_blank",
        "u_blank_ppm",
        "u_type_a",
        "u_type_a_ppm",
        "u_type_b",
        "u_type_b_ppm",
        "u_combined",
        "u_combined_ppm",
        "u_expanded",
        "u_expanded_ppm",
        "coverage_factor",
    ):
        payload[key] = None
    payload["components"] = {}
    payload["components_ppm"] = {}
    return payload


def _array_to_list(arr: np.ndarray) -> List[Any]:
    """Convert numpy array to list, replacing NaN/Inf with None."""
    source = np.asarray(arr)
    if np.issubdtype(source.dtype, np.bool_):
        return source.astype(bool).tolist()
    array = np.asarray(source, dtype=np.float64)
    result = array.astype(object)
    result[~np.isfinite(array)] = None
    return result.tolist()


def _sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize a dictionary for JSON serialization."""
    result: Dict[str, Any] = {}
    for key, value in d.items():
        # Skip private keys (internal state)
        if str(key).startswith("_"):
            continue
        result[str(key)] = _sanitize_value(value)
    return result


def _find_non_serialisable_path(obj: Any, *, _path: str = "") -> Optional[str]:
    """Walk a JSON-like structure and return the key path of the first problem.

    Used by the item-58 diagnostics wrapper in :func:`export_to_json` to
    produce a useful error message when ``json.dumps(allow_nan=False)`` raises.
    Returns ``None`` if no problematic value can be identified quickly.
    """
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                found = _find_non_serialisable_path(v, _path=f"{_path}.{k}" if _path else str(k))
                if found is not None:
                    return found
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                found = _find_non_serialisable_path(v, _path=f"{_path}[{i}]")
                if found is not None:
                    return found
        elif isinstance(obj, float):
            if not (obj == obj) or obj == float("inf") or obj == float("-inf"):  # NaN or Inf
                return _path or "<root>"
    except Exception:
        pass
    return None


def _sanitize_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable form."""
    if value is None:
        return None

    # Numpy arrays
    if isinstance(value, np.ndarray):
        return _array_to_list(value)

    # Numpy scalars
    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return safe_float(value)

    # Standard Python types
    if isinstance(value, (int, str, bool)):
        return value

    if isinstance(value, float):
        return safe_float(value)

    # Dataclasses (CycleData, UncertaintyBudget, etc.)
    if is_dataclass(value) and not isinstance(value, type):
        if isinstance(value, CycleData):
            return _serialize_cycle_stats(value)
        if isinstance(value, UncertaintyBudget):
            return _serialize_uncertainty(value)
        if isinstance(value, ProcessingConfig):
            payload = asdict(value)
            payload.pop("subtract_kr_blank", None)
            return _sanitize_dict(payload)
        # Generic dataclass
        return _sanitize_dict(asdict(value))

    # Dictionaries
    if isinstance(value, dict):
        return _sanitize_dict(value)

    # Lists/tuples
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]

    # Fallback: convert to string
    return str(value)

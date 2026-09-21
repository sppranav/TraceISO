"""Resolve ordinary A/C replay identity without sampling or changing live state."""
import copy
import json

import numpy as np

from config.scientific_identity import digest
from domain.ratio_selection import get_best_ratio_data


def resolve_current_mc_input_digest(record, sample, ratio_name, *, all_samples,
        element_config, uncertainty_config, processing_config, gum_budget,
        certified_value=None, ratio_values=None, ratio_mean=None, ratio_mask=None,
        cycle_ranges=None, custom_contributor_library=None, profile_defaults=None):
    """Return complete live identity using the record's original random context.

    An empty string means insufficient historical evidence. All extraction is
    performed on an owned graph; no RNG is constructed and no draw is run.
    Callers must supply the same effective inputs used to prepare a new MC run.
    Invalid live inputs raise ValueError rather than becoming matching evidence.
    """
    from domain.uncertainty.mc_result import MC_RESULT_SCHEMA_VERSION
    from domain.uncertainty.monte_carlo import (
        _extract_internal_perturbation_params, _extract_pb_tl_perturbation_params,
        _resolved_replay_evidence,
    )
    from domain.uncertainty.sr_chain_identity import engine_a_semantics_for_method

    if record.schema_version != MC_RESULT_SCHEMA_VERSION or not record.replay_snapshot_json:
        return ''
    try:
        saved = json.loads(record.replay_snapshot_json)
    except (ValueError, TypeError):
        return ''
    required = {'schema', 'method', 'resolved_parameters', 'scientific_configuration',
                'processing', 'uncertainty', 'observations', 'rng_initial_state',
                'ratio_values', 'ratio_mask', 'ratio_mean', 'analytical_contributor_specifications'}
    if not isinstance(saved, dict) or not required.issubset(saved) or saved['schema'] != 'traceiso.mc_replay_inputs.v2':
        return ''
    if not isinstance(saved['rng_initial_state'], dict) or 'bit_generator' not in saved['rng_initial_state']:
        return ''
    if digest(saved) != record.input_digest:
        return ''
    (sample, all_samples, element_config, uncertainty_config, processing_config,
     gum_budget, certified_value, ratio_values, ratio_mask, cycle_ranges,
     custom_contributor_library, profile_defaults) = copy.deepcopy((
        sample, all_samples, element_config, uncertainty_config, processing_config,
        gum_budget, certified_value, ratio_values, ratio_mask, cycle_ranges,
        custom_contributor_library, profile_defaults))
    if ratio_values is None:
        cd = get_best_ratio_data(sample, ratio_name)
        if cd is None: raise ValueError('Current MC ratio data are unavailable')
        ratio_values = cd.valid_values
    if ratio_mean is None:
        finite = ratio_values[np.isfinite(ratio_values)]
        ratio_mean = float(np.mean(finite)) if len(finite) else 0.0
    if len(ratio_values) < 2 or not np.isfinite(ratio_mean) or ratio_mean == 0:
        raise ValueError('Current MC ratio support is unavailable')
    engine = uncertainty_config.resolve_engine(element_config.symbol, processing_config=processing_config)
    common = dict(gum_budget=gum_budget, ratio_mask=ratio_mask, cycle_ranges=cycle_ranges,
                  custom_contributor_library=custom_contributor_library, profile_defaults=profile_defaults)
    if engine == 'internal_normalization':
        params = _extract_internal_perturbation_params(sample, ratio_name, ratio_values,
            ratio_mean, all_samples, element_config, uncertainty_config,
            processing_config=processing_config, certified_value=certified_value, **common)
        semantics = engine_a_semantics_for_method(params.reference_inputs.chain_method)
    elif engine == 'pb_tl_external_normalization':
        params = _extract_pb_tl_perturbation_params(sample, ratio_name, ratio_mean,
            all_samples, element_config, uncertainty_config, processing_config, **common)
        semantics = 'engine_c.chain_replay.v3.joint_tl_blank'
    else:
        raise ValueError('Current route is not an ordinary A/C replay')
    return digest(_resolved_replay_evidence(
        custom_base=params, draw_params=params, semantics_version=semantics,
        element_config=element_config, processing_config=processing_config,
        uncertainty_config=uncertainty_config, custom_contributor_library=custom_contributor_library,
        all_samples=all_samples, sample=sample, cycle_ranges=cycle_ranges,
        ratio_mask=ratio_mask, ratio_values=ratio_values, ratio_mean=ratio_mean,
        certified_value=certified_value, profile_defaults=profile_defaults,
        gum_budget=gum_budget, rng_initial_state=saved['rng_initial_state']))

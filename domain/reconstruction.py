"""Bounded restoration of recorded processing inputs; no UI or file writes."""
from copy import deepcopy
from dataclasses import fields

import numpy as np

from config.recorded_dependencies import RecordedDependencies
from config.settings import (ProcessingConfig, UncertaintyConfig, DriftConfig,
                             PbStandardCalibrationConfig, KappaFactor, KappaAssignment,
                             CustomUncertaintyContributor)
from domain.elements.base import ElementConfig, CertifiedValue, MonitorSpec, DataLayer
from domain.models import Sample, CycleData


class ReconstructionError(ValueError):
    """Machine-readable refusal; never implies a successful scientific replay."""
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)

    def to_dict(self):
        return {'code': self.code, 'message': str(self)}


def require_fields(value, expected, label):
    if not isinstance(value, dict):
        raise ReconstructionError('invalid_record', f'{label} must be an object')
    missing = set(expected) - value.keys()
    extra = value.keys() - set(expected)
    if missing:
        raise ReconstructionError('missing_input', f'{label} missing fields: {sorted(missing)}')
    if extra:
        raise ReconstructionError('unsupported_record', f'{label} unknown fields: {sorted(extra)}')


def restore_dataclass(cls, value):
    require_fields(value, [f.name for f in fields(cls)], cls.__name__)
    return cls(**value)


def validate_identity_header(identity):
    """Refuse incomplete or unsupported identity envelopes before reading input."""
    if not isinstance(identity, dict) or identity.get('processing_identity_version') != 'traceiso.processing_identity.v3':
        raise ReconstructionError('unsupported_version', 'Replay requires processing identity v3; older dependency evidence is unknown')
    required = {'identity_version', 'element_configuration', 'element_ratio_order',
                'managed_library', 'sr_method', 'custom_pdf_policy', 'processing_identity_version',
                'dependency_versions', 'software_identity', 'custom_contributors', 'input_observations',
                'processing', 'uncertainty', 'cycle_ranges', 'profile_defaults', 'profile_defaults_were_explicit', 'scope', 'observations'}
    require_fields(identity, required, 'processing identity')
    if identity['identity_version'] != 'traceiso.scientific_inputs.v2' or identity['scope'] != 'stored_processing':
        raise ReconstructionError('unsupported_version', 'Unsupported scientific identity or processing scope')


def restore_configuration(identity):
    """Restore explicit recorded dependencies without processing observations."""
    validate_identity_header(identity)
    try:
        from config.software_identity import validate_software_identity
        validate_software_identity(identity['software_identity'])
        if identity['custom_pdf_policy'] != 'declared_pdf_estimation_dof_separate.v1':
            raise ReconstructionError('unsupported_version', 'Unknown custom contributor policy')
        element = deepcopy(identity['element_configuration'])
        order = identity['element_ratio_order']
        if not isinstance(order, list) or len(order) != len(set(order)) or set(order) != set(element['default_ratios']):
            raise ReconstructionError('invalid_record', 'Element ratio order does not match recorded ratios')
        element['default_ratios'] = {key: tuple(element['default_ratios'][key]) for key in order}
        element['certified_values'] = {key: restore_dataclass(CertifiedValue, val) for key, val in element['certified_values'].items()}
        element['monitors'] = tuple(restore_dataclass(MonitorSpec, val) for val in element['monitors'])
        element['data_layers'] = tuple(DataLayer(val) for val in element['data_layers'])
        element = restore_dataclass(ElementConfig, element)
        expected_method = 'sr_natural_init_two_refinements_v1' if element.iterations > 1 else 'element_declared_single_pass'
        if identity['sr_method'] != expected_method:
            raise ReconstructionError('unsupported_version', 'Recorded correction method is unsupported')
        processing = deepcopy(identity['processing'])
        processing['drift'] = restore_dataclass(DriftConfig, processing['drift'])
        processing['pb_standard_calibration'] = restore_dataclass(PbStandardCalibrationConfig, processing['pb_standard_calibration'])
        processing = restore_dataclass(ProcessingConfig, processing)
        uncertainty = deepcopy(identity['uncertainty'])
        if uncertainty is None:
            raise ReconstructionError('missing_input', 'An explicit uncertainty configuration is required for replay')
        uncertainty['kappa_factors'] = [restore_dataclass(KappaFactor, item) for item in uncertainty['kappa_factors']]
        uncertainty['kappa_assignments'] = restore_dataclass(KappaAssignment, uncertainty['kappa_assignments'])
        uncertainty = restore_dataclass(UncertaintyConfig, uncertainty)
        customs = {}
        from config.validation import parse_custom_contributor_entries
        for symbol, definitions in identity['custom_contributors'].items():
            customs[symbol] = []
            for definition in definitions:
                definition = dict(definition)
                definition['degrees_of_freedom'] = float(definition['degrees_of_freedom'])
                customs[symbol].append(restore_dataclass(CustomUncertaintyContributor, definition))
            validated = parse_custom_contributor_entries(symbol, definitions)
            if validated != customs[symbol]:
                raise ReconstructionError('invalid_record', 'Custom contributor fields require normalization; record is not canonical')
        if not isinstance(identity['profile_defaults'], dict) or type(identity['profile_defaults_were_explicit']) is not bool:
            raise ReconstructionError('missing_input', 'Complete profile defaults and precedence are required')
        dependencies = RecordedDependencies(identity['managed_library'], customs, identity['profile_defaults'])
        return element, processing, uncertainty, dependencies
    except ReconstructionError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ReconstructionError('invalid_record', f'Recorded configuration cannot be restored: {exc}') from exc


def restore_observations(identity):
    """Validate typed arrays, masks, IDs and windows before HDF5 input loading."""
    validate_identity_header(identity)
    try:
        observations = identity['input_observations']
        if not isinstance(observations, list) or len(observations) > 10000:
            raise ReconstructionError('input_mismatch', 'Invalid recorded observation count')
        samples = []
        seen = set()
        cycle_layers = {f.name for f in fields(Sample) if 'CycleData' in str(f.type)}
        sample_fields = {f.name for f in fields(Sample)} - {'uncertainty', 'mc_results'}
        for saved in observations:
            require_fields(saved, sample_fields, 'observation')
            item = deepcopy(saved)
            if (any(not isinstance(item[key], str) for key in ('name', 'sample_type'))
                    or type(item['run_number']) is not int
                    or not isinstance(item['warnings'], list)
                    or any(not isinstance(warning, str) for warning in item['warnings'])):
                raise ReconstructionError('invalid_record', 'Invalid observation labels, run number or warnings')
            for spec in fields(Sample):
                if spec.name in item and 'Dict[' in str(spec.type) and not isinstance(item[spec.name], dict):
                    raise ReconstructionError('invalid_record', f'Observation {spec.name} must be an object')
            if not isinstance(item['observation_id'], str) or not item['observation_id'] or item['observation_id'] in seen:
                raise ReconstructionError('invalid_record', 'Observation IDs must be nonempty and unique')
            seen.add(item['observation_id'])
            if item['correction_records']:
                raise ReconstructionError('unsupported_record', 'Pre-existing correction records cannot be reconstructed by v1')
            for layer in cycle_layers:
                restored = {}
                for name, data in item[layer].items():
                    require_fields(data, {'values', 'mask'}, 'cycle data')
                    if not isinstance(data['values'], list) or any(
                        not (type(value) in (int, float) or
                             isinstance(value, str) and value in {'nan', 'inf', '-inf'})
                        for value in data['values']
                    ):
                        raise ReconstructionError('invalid_record', 'Cycle values require numbers or canonical nonfinite labels')
                    values = np.asarray(data['values'], dtype=float)
                    if not isinstance(data['mask'], list) or any(type(value) is not bool for value in data['mask']):
                        raise ReconstructionError('invalid_record', 'Cycle masks must contain Boolean values')
                    mask = np.asarray(data['mask'], dtype=bool)
                    if values.ndim != 1 or len(values) > 1000000 or mask.shape != values.shape:
                        raise ReconstructionError('invalid_record', 'Invalid cycle array or mask')
                    restored[name] = CycleData(values.copy(), mask.copy())
                item[layer] = restored
            samples.append(Sample(**item))
        if not isinstance(identity['cycle_ranges'], dict) or any(key not in seen for key in identity['cycle_ranges']):
            raise ReconstructionError('invalid_record', 'Cycle windows require recorded observation IDs')
        for window in identity['cycle_ranges'].values():
            if (not isinstance(window, (list, tuple)) or len(window) != 2
                    or any(type(value) is not int or value < 1 for value in window)
                    or window[0] > window[1]):
                raise ReconstructionError('invalid_record', 'Cycle windows must be ordered positive integer pairs')
        return samples
    except ReconstructionError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        raise ReconstructionError('invalid_record', f'Recorded observations cannot be restored: {exc}') from exc


def reconstruct(identity, loaded_samples, *, allow_dependency_mismatch=False):
    """Reprocess raw observations with recorded settings and managed libraries.

    Input correction-record objects are refused by this first contract. Drift
    and calibration requested in recorded settings are recomputed by production.
    Saved MC is retained by the package reader; this function never reruns MC.
    """
    from domain.provenance import dependency_versions
    from domain.processing_service import process_samples

    validate_identity_header(identity)
    recorded_versions = identity['dependency_versions']
    require_fields(recorded_versions, dependency_versions(), 'dependency versions')
    mismatch = recorded_versions != dependency_versions()
    if mismatch and not allow_dependency_mismatch:
        raise ReconstructionError('dependency_mismatch', 'Recorded numerical dependency versions differ; explicit opt-in is required')
    try:
        element, processing, uncertainty, dependencies = restore_configuration(identity)
        samples = restore_observations(identity)
        if len(samples) != len(loaded_samples):
            raise ReconstructionError('input_mismatch', 'Retained input and recorded observation counts differ')
        cycle_layers = {f.name for f in fields(Sample) if 'CycleData' in str(f.type)}
        for saved, raw in zip(samples, loaded_samples):
            for layer in cycle_layers:
                original = getattr(raw, layer)
                restored = getattr(saved, layer)
                if original.keys() != restored.keys() or any(
                    not np.array_equal(original[key].values, restored[key].values, equal_nan=True) for key in original
                ):
                    raise ReconstructionError('input_mismatch', f'Recorded {layer} differs from retained raw input')
        result = process_samples(samples, element, processing, uncertainty,
                                 profile_defaults=identity['profile_defaults'] if identity['profile_defaults_were_explicit'] else None,
                                 cycle_ranges=identity['cycle_ranges'], recorded_dependencies=dependencies)
        result.quality_metrics['reconstruction'] = {
            'schema': 'traceiso.reconstruction_run.v1', 'dependency_mismatch': mismatch,
            'original_software_identity': deepcopy(identity['software_identity']),
            'mc_replayed': False,
        }
        return result
    except ReconstructionError:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ReconstructionError('invalid_record', f'Recorded input cannot be reconstructed: {exc}') from exc

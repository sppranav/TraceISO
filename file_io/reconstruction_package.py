"""Versioned in-memory reconstruction ZIP; members are never extracted by name."""
from dataclasses import dataclass, fields
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED, BadZipFile

from config.scientific_identity import canonical_json
from domain.reconstruction import ReconstructionError, require_fields, restore_configuration, restore_observations

SCHEMA = 'traceiso.reconstruction_package.v1'
MAX_INPUT_BYTES = 100 * 1024 * 1024
MAX_RECORD_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class ReconstructionPackage:
    identity: dict
    samples: list
    saved_mc: dict
    input_sha256: str


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def build_reconstruction_package(result, input_bytes):
    """Retain caller-supplied input bytes and the original processing evidence.

    This serializes existing records only; no scientific calculation or MC runs.
    """
    identity = result.quality_metrics.get('processing_scientific_identity')
    if not identity or identity.get('processing_identity_version') != 'traceiso.processing_identity.v3':
        raise ReconstructionError('unsupported_version', 'A processing identity v3 is required; historical dependencies remain unknown')
    saved_mc = {sample.observation_id: {ratio: record.to_dict() for ratio, record in sample.mc_results.items()}
                for sample in result.samples}
    record = canonical_json({'schema': SCHEMA, 'input_sha256': _sha(input_bytes),
                             'identity': identity, 'saved_mc': saved_mc}).encode('utf-8')
    if len(input_bytes) > MAX_INPUT_BYTES or len(record) > MAX_RECORD_BYTES:
        raise ReconstructionError('size_limit', 'Reconstruction package exceeds its declared size limits')
    stream = BytesIO()
    with ZipFile(stream, 'w', compression=ZIP_DEFLATED) as archive:
        archive.writestr('input.h5', input_bytes)
        archive.writestr('record.json', record)
    return stream.getvalue()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReconstructionError('invalid_record', 'Duplicate JSON object keys are unsupported')
        result[key] = value
    return result


def _restore_saved_mc(payload):
    """Refuse lossy coercions at the package boundary; retain historical unknowns."""
    from domain.uncertainty.mc_result import (
        MCResultRecord, MCContributorRecord, MC_RESULT_SCHEMA_NAME,
        migrate_mc_result_payload,
    )

    def validate(value, cls, extra=()):
        if not isinstance(value, dict):
            raise ReconstructionError('invalid_record', 'Saved MC record must be an object')
        known = {item.name: item for item in fields(cls)}
        if set(value) - set(known) - set(extra):
            raise ReconstructionError('unsupported_record', 'Unknown saved MC fields cannot be preserved')
        for key, item in known.items():
            if key not in value:
                continue  # Historical absent fields retain the documented migration.
            data, kind = value[key], str(item.type)
            if data is None and 'Optional' in kind:
                continue
            valid = True
            if 'bool' in kind:
                valid = type(data) is bool
            elif 'int' in kind and 'Tuple' not in kind:
                valid = type(data) is int
                if valid and key in {'requested_draws', 'completed_draws', 'n_dropped', 'seed'}:
                    valid = data >= 0
            elif 'float' in kind:
                valid = type(data) in (int, float) or isinstance(data, str) and data in {'NaN', 'Infinity', '-Infinity'}
            elif kind == 'str':
                valid = isinstance(data, str)
            elif key == 'warnings':
                valid = isinstance(data, list) and all(isinstance(w, str) for w in data)
            elif key == 'contributors':
                valid = isinstance(data, list)
                if valid:
                    for contributor in data:
                        validate(contributor, MCContributorRecord)
            if not valid:
                raise ReconstructionError('invalid_record', f'Invalid saved MC field {key}')

    validate(payload, MCResultRecord, ('n_iter', 'mc_lower_95', 'mc_upper_95'))
    if payload.get('schema_name', MC_RESULT_SCHEMA_NAME) != MC_RESULT_SCHEMA_NAME:
        raise ReconstructionError('unsupported_record', 'Unknown saved MC schema name')
    if payload.get('draw_array_persisted', False):
        raise ReconstructionError('unsupported_record', 'Saved draw arrays are outside the package contract')
    migrated = migrate_mc_result_payload(payload)
    # Legacy aliases must pass the same type checks after migration.
    validate(migrated, MCResultRecord,
             ('n_iter', 'mc_lower_95', 'mc_upper_95', 'migrated_from_schema_version'))
    return MCResultRecord.from_dict(payload)


def load_reconstruction_package(data):
    """Validate exact member names, sizes, schema and hash before HDF5 loading."""
    from file_io.hdf5_reader import load_hdf5

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ReconstructionError('invalid_record', 'Reconstruction payload must be bytes')
    if len(data) > MAX_INPUT_BYTES + MAX_RECORD_BYTES + 1024 * 1024:
        raise ReconstructionError('size_limit', 'Compressed package exceeds the size limit')
    try:
        with ZipFile(BytesIO(data)) as archive:
            members = archive.infolist()
            if len(members) != 2 or {entry.filename for entry in members} != {'input.h5', 'record.json'}:
                raise ReconstructionError('unsupported_record', 'Only input.h5 and record.json members are supported')
            for entry in members:
                limit = MAX_INPUT_BYTES if entry.filename == 'input.h5' else MAX_RECORD_BYTES
                if entry.file_size > limit or entry.flag_bits & 1:
                    raise ReconstructionError('size_limit', 'Oversized or encrypted members are unsupported')
            record_bytes = archive.read('record.json')
            input_bytes = archive.read('input.h5')
        record = json.loads(record_bytes, object_pairs_hook=_unique_object,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON literal')))
        require_fields(record, {'schema', 'input_sha256', 'identity', 'saved_mc'}, 'package record')
        if record['schema'] != SCHEMA:
            raise ReconstructionError('unsupported_version', 'Unsupported reconstruction package version')
        if _sha(input_bytes) != record['input_sha256']:
            raise ReconstructionError('input_mismatch', 'Retained HDF5 hash does not match the record')
        identity = record['identity']
        _element, processing, _uncertainty, dependencies = restore_configuration(identity)
        observations = restore_observations(identity)
        saved_mc = {obs: {ratio: _restore_saved_mc(value) for ratio, value in records.items()}
                    for obs, records in record['saved_mc'].items()}
        for obs, records in saved_mc.items():
            for ratio, mc in records.items():
                if mc.ratio_name != ratio:
                    raise ReconstructionError('input_mismatch', 'Saved MC ratio identity does not match its key')
        ids = {item.observation_id for item in observations}
        if set(saved_mc) != ids:
            raise ReconstructionError('input_mismatch', 'Saved MC observations do not match recorded inputs')
        with tempfile.TemporaryDirectory(prefix='traceiso-reconstruction-') as directory:
            path = Path(directory) / 'input.h5'
            path.write_bytes(input_bytes)
            from config.recorded_dependencies import use_recorded_dependencies
            # Element detection constructs CRM-backed element configurations.
            # It must see the package's references before processing begins.
            with use_recorded_dependencies(dependencies):
                loaded = load_hdf5(path, data_preference=processing.data_preference)
        return ReconstructionPackage(identity, loaded.samples, saved_mc, record['input_sha256'])
    except ReconstructionError:
        raise
    except (BadZipFile, KeyError, TypeError, ValueError, AttributeError, OSError, RuntimeError, RecursionError) as exc:
        raise ReconstructionError('invalid_record', f'Cannot read reconstruction package: {exc}') from exc

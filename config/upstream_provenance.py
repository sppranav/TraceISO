"""Strict immutable JSON envelope for extractor lineage, independent of UI."""
import hashlib
import json

SCHEMA = "traceiso.extractor_provenance.v1"
def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
def seal(payload):
    if 'schema' in payload or 'sha256' in payload:
        raise ValueError('Envelope fields must not be supplied by the payload')
    record = {"schema": SCHEMA, **payload}
    record["sha256"] = hashlib.sha256(canonical(record).encode()).hexdigest()
    return canonical(record)
def validate(text):
    record = json.loads(text)
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("Unsupported extractor provenance schema")
    expected = record.pop("sha256", None)
    if hashlib.sha256(canonical(record).encode()).hexdigest() != expected:
        raise ValueError("Corrupt extractor provenance digest")
    sources = record.get("sources")
    if not isinstance(sources, list): raise ValueError("Extractor sources missing")
    from config.input_units import LEGACY_CONVENTION
    if record.get('bare_number_convention') != LEGACY_CONVENTION:
        raise ValueError('Unsupported bare-number convention')
    if not isinstance(record.get('template'), dict) or not isinstance(record.get('software'), dict):
        raise ValueError('Extractor template or software evidence missing')
    if not isinstance(record.get('warnings'), list):
        raise ValueError('Extractor warnings missing')
    ids = set()
    ordinals = set()
    for item in sources:
        if not isinstance(item, dict):
            raise ValueError('Invalid extractor source record')
        if item.get("source_id") in ids or not item.get("source_id"):
            raise ValueError("Extractor source identity missing or duplicated")
        ids.add(item["source_id"])
        ordinal = item.get('ordinal')
        if type(ordinal) is not int or ordinal < 1 or ordinal in ordinals:
            raise ValueError('Invalid or duplicate source ordinal')
        ordinals.add(ordinal)
        if not isinstance(item.get('filename'), str) or not isinstance(item.get('units'), dict):
            raise ValueError('Source filename or channel evidence missing')
        digest = item.get("sha256", "")
        if not isinstance(digest, str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError("Invalid source SHA-256")
    record["sha256"] = expected
    return canonical(record)

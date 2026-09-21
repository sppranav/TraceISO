"""Canonical snapshots of effective scientific inputs, never historical fill-ins."""
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from collections.abc import Mapping
from types import SimpleNamespace


def snapshot(value):
    if isinstance(value, SimpleNamespace):
        return snapshot(vars(value))
    if is_dataclass(value):
        return {f.name: snapshot(getattr(value, f.name)) for f in fields(value)
                if f.name not in {"mc_results", "uncertainty"} or not hasattr(value, "observation_id")}
    if isinstance(value, Enum):
        return snapshot(value.value)
    if isinstance(value, Mapping):
        return {str(k): snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((snapshot(v) for v in value), key=canonical_json)
    if hasattr(value, "tolist"):
        return snapshot(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported scientific identity input: {type(value).__name__}")


def canonical_json(value):
    return json.dumps(snapshot(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def scientific_configuration(element_config):
    from config.reference_materials import scientific_library_snapshot
    return {
        "identity_version": "traceiso.scientific_inputs.v2",
        "element_configuration": snapshot(element_config),
        # Canonical JSON sorts object keys. Preserve the declared ratio order
        # separately because ElementConfig.primary_ratio uses its first entry.
        "element_ratio_order": list(getattr(element_config, 'default_ratios', {})),
        # Includes auxiliary-element references and calibration materials. This
        # conservative superset deliberately invalidates on unrelated edits too.
        "managed_library": scientific_library_snapshot(),
        "sr_method": "sr_natural_init_two_refinements_v1" if getattr(element_config, "iterations", 1) > 1 else "element_declared_single_pass",
        "custom_pdf_policy": "declared_pdf_estimation_dof_separate.v1",
    }

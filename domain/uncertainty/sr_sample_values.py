"""Sample-specific numeric inputs for Sr Engine A optional contributors."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import numpy as np

from config.settings import DEFAULT_REPROD_DIG_SD, UncertaintyConfig
from domain.models import Sample


SR_SAMPLE_VALUES_KEY = "values"
SR_QC_BIAS_KEY = "u_bias_qc"
SR_DIGESTION_KEY = "u_reprod_dig"


def _positive_float_or_none(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def _sr_value_block(sample: Sample, contributor_name: str) -> Mapping[str, Any]:
    raw = (getattr(sample, "metadata", {}) or {}).get("uncertainty_contributors") or {}
    if not isinstance(raw, dict):
        return {}
    values = raw.get(SR_SAMPLE_VALUES_KEY) or {}
    if not isinstance(values, dict):
        return {}
    block = values.get(contributor_name) or {}
    return block if isinstance(block, dict) else {}


def resolve_sr_qc_bias_inputs(
    sample: Sample,
    uncertainty_config: UncertaintyConfig,
) -> Tuple[float, float]:
    """Return ``(numerator_abs, denominator_ratio)`` for Sr QC bias.

    Per-sample fields override the session defaults independently. This lets a
    user enter only the sample-specific numerator while inheriting the session
    certified ratio, or vice versa.
    """
    block = _sr_value_block(sample, SR_QC_BIAS_KEY)
    numerator = _positive_float_or_none(block.get("numerator_abs"))
    denominator = _positive_float_or_none(block.get("denominator_ratio"))
    if numerator is None:
        numerator = _positive_float_or_none(
            getattr(uncertainty_config, "sr_qc_bias_abs", 0.0)
        )
    if denominator is None:
        denominator = _positive_float_or_none(
            getattr(uncertainty_config, "sr_qc_cert_value", 0.0)
        )
    return float(numerator or 0.0), float(denominator or 0.0)


def resolve_sr_digestion_inputs(
    sample: Sample,
    uncertainty_config: UncertaintyConfig,
) -> Tuple[float, float]:
    """Return ``(sd_abs, denominator_ratio)`` for Sr digestion reproducibility."""
    block = _sr_value_block(sample, SR_DIGESTION_KEY)
    sd_abs = _positive_float_or_none(block.get("sd_abs"))
    denominator = _positive_float_or_none(block.get("denominator_ratio"))
    if sd_abs is None:
        sd_abs = _positive_float_or_none(
            getattr(uncertainty_config, "u_reprod_dig_sd", DEFAULT_REPROD_DIG_SD)
        )
    if denominator is None:
        denominator = _positive_float_or_none(
            getattr(uncertainty_config, "u_reprod_dig_ref_value", 0.0)
        )
    return float(sd_abs or 0.0), float(denominator or 0.0)


def build_processed_material_contributors(sample, uncertainty_config, ratio_mean, gate):
    """Transfer manual material uncertainties once to the reported Sr/Pb ratio.

    Retains the existing Sr configuration/override keys for session compatibility.
    The gate is supplied by the engine's sample/profile applicability resolver.
    """
    from domain.models import UncertaintyContributor
    from domain.uncertainty.engine_internal_sr import (
        compute_qc_bias_term, compute_digestion_reproducibility_term,
    )

    qc, qc_ref = resolve_sr_qc_bias_inputs(sample, uncertainty_config)
    sd, dig_ref = resolve_sr_digestion_inputs(sample, uncertainty_config)
    qc_abs, qc_rel, _ = compute_qc_bias_term(
        observed_bias_abs=qc, qc_cert_value=qc_ref, ratio_mean=abs(ratio_mean),
    )
    dig_abs, dig_rel, _ = compute_digestion_reproducibility_term(
        digestion_sd_abs=sd, digestion_ref_value=dig_ref, ratio_mean=abs(ratio_mean),
    )
    return [
        UncertaintyContributor(
            name=name, display_name=label, value_abs=value, value_rel_permil=rel,
            type_ab="B", degrees_of_freedom=float("inf"), percentage_contribution=0.0,
            description=description + " Reference: JCGM 100:2008, sections 4.3 and 5.1.",
            reference="JCGM 100:2008",
            **gate(name, value > 0.0),
        )
        for name, label, value, rel, description in (
            ("u_bias_qc", "Bias of processed QC material", qc_abs, qc_rel,
             "User-supplied QC standard uncertainty divided by the QC certified ratio, "
             "transferred fractionally to the final sample ratio; no bias correction is applied."),
            ("u_reprod_dig", "Between-digestion reproducibility (Type B)", dig_abs, dig_rel,
             "SD of independently processed digestion means divided by the digested material's "
             "reference ratio, transferred fractionally to the final sample ratio; no division by sqrt(n)."),
        )
    ]

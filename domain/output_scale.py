"""Deterministic output scaling between a replay layer and the reported ratio.

Engines A, B and C build several contributors by replaying a correction chain,
or by propagating blank sigmas through intensity-level sensitivities. Those
replays stop at a *measured-scale* layer: the Sr/Pb-Tl normalized ratio, or the
blank-corrected ratio. The pipeline may then apply further **deterministic
scalar** transforms before the ratio is reported — Sr session anchoring, drift
correction, and standard-sample bracketing — each of which multiplies every
cycle of the layer by one number recorded by its producer.

For a committed scalar ``a`` the sensitivity of the reported measurand to any
upstream quantity is ``d(aR)/dx = a * dR/dx``, so such a contributor must be
multiplied by the same ``a`` exactly once. This module resolves that product
from the producers' own records and reports which components entered it, so
double-scaling and missed scaling are both visible rather than implicit.

Uncertainties of the transforms themselves (the bracketing standards, the drift
model, the anchoring reference) are separate contributors and are not part of
this factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from domain.models import Sample
from domain.ratio_selection import select_best_ratio_layer

#: Sample metadata key: ``{ratio_name: factor}`` for the committed drift scalar.
DRIFT_OUTPUT_FACTORS_KEY = "_drift_output_factors"

#: Sample metadata key: the ratio Sr session anchoring actually rescaled.
SR_ANCHOR_RATIO_NAME_KEY = "_sr_anchor_ratio_name"

#: Replay layers a contributor can be expressed on.
INPUT_LAYER_PRE_ANCHOR = "iif_pre_anchor"
INPUT_LAYER_NORMALIZED = "iif"
INPUT_LAYER_BLANK_CORRECTED = "corrected"

#: Reported layers that lie above the normalization step.
_POST_NORMALIZATION_LAYERS = frozenset({"iif", "sr_standard", "drift", "ssb"})
_POST_DRIFT_LAYERS = frozenset({"drift", "ssb"})


@dataclass(frozen=True)
class OutputScale:
    """The deterministic multiplier from *input_layer* to *output_layer*."""

    factor: float
    input_layer: str
    output_layer: str
    components: Tuple[Tuple[str, float], ...] = ()

    @property
    def is_identity(self) -> bool:
        return not self.components

    def describe(self) -> str:
        """Human-readable scale statement for budget/UI provenance."""
        if self.is_identity:
            return f"reported on the {self.output_layer} layer; no output rescaling"
        parts = " x ".join(f"{name}={value:.9g}" for name, value in self.components)
        return (
            f"{self.input_layer} -> {self.output_layer}, "
            f"factor {self.factor:.9g} ({parts})"
        )


def _usable(value: object) -> Optional[float]:
    """Return *value* as a finite, non-zero float, else ``None``."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number == 0.0:
        return None
    return number


def _anchor_factor(sample: Sample, ratio_name: str) -> Optional[float]:
    metadata = sample.metadata or {}
    if metadata.get(SR_ANCHOR_RATIO_NAME_KEY) != ratio_name:
        return None
    return _usable(metadata.get("_sr_anchor_factor"))


def _drift_factor(sample: Sample, ratio_name: str) -> Optional[float]:
    recorded = (sample.metadata or {}).get(DRIFT_OUTPUT_FACTORS_KEY)
    if not isinstance(recorded, dict):
        return None
    return _usable(recorded.get(ratio_name))


def _ssb_factor(sample: Sample, ratio_name: str) -> Optional[float]:
    result = (getattr(sample, "ssb_results", None) or {}).get(ratio_name)
    if not isinstance(result, dict):
        return None
    return _usable(result.get("k_factor"))


def resolve_output_scale(
    sample: Sample,
    ratio_name: str,
    *,
    input_layer: str,
) -> OutputScale:
    """Deterministic multiplier carrying *input_layer* onto the reported layer.

    Only transforms the producers actually committed for this sample and ratio
    enter the product, and each enters once. A transform whose recorded factor
    is missing, non-finite or zero is not a deterministic scalar and is left
    out — the caller then reports the contributor on the layer it was computed
    on rather than on a guessed scale.
    """
    selected = select_best_ratio_layer(sample, ratio_name)
    output_layer = selected.key if selected is not None else "ratios"

    components: list[Tuple[str, float]] = []

    if input_layer == INPUT_LAYER_PRE_ANCHOR and output_layer in _POST_NORMALIZATION_LAYERS:
        anchor = _anchor_factor(sample, ratio_name)
        if anchor is not None:
            components.append(("sr_session_anchoring", anchor))

    if output_layer in _POST_DRIFT_LAYERS:
        drift = _drift_factor(sample, ratio_name)
        if drift is not None:
            components.append(("drift", drift))

    if output_layer == "ssb":
        ssb = _ssb_factor(sample, ratio_name)
        if ssb is not None:
            components.append(("ssb", ssb))

    factor = 1.0
    for _name, value in components:
        factor *= value

    return OutputScale(
        factor=float(factor),
        input_layer=input_layer,
        output_layer=output_layer,
        components=tuple(components),
    )


def scaled_contribution(
    value_abs: float,
    scale: OutputScale,
) -> float:
    """Carry an absolute uncertainty from the replay layer to the output layer."""
    if not np.isfinite(value_abs) or value_abs <= 0.0:
        return float(value_abs)
    return float(value_abs * abs(scale.factor))


def record_drift_output_factor(
    sample: Sample,
    ratio_name: str,
    factor: float,
) -> None:
    """Record the committed drift scalar so consumers need not re-derive it."""
    usable = _usable(factor)
    if usable is None:
        return
    factors: Dict[str, float] = sample.metadata.setdefault(
        DRIFT_OUTPUT_FACTORS_KEY, {},
    )
    factors[ratio_name] = usable

"""Which Sr correction method an uncertainty replay or MC record belongs to (A025).

Production selects the Sr method from the element configuration
(``iterations <= 1`` selects the single pass) and records it on each processed
sample as ``_sr_chain_method``. Every uncertainty replay must evaluate the same
method, and a stored Monte Carlo record may only be called current when it was
produced by the method that is active now.

Two rules follow. A replay uses the method the stored data were produced with;
if that disagrees with the active element configuration, the stored outputs are
stale and the replay is refused rather than silently re-modelled — reprocessing
is the way to change a method. And a freshness check without evidence of the
active method reports *unknown*, never *current*.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from domain.corrections.sr_chain import (
    SR_CHAIN_METHOD,
    SR_SINGLE_PASS_METHOD,
    SUPPORTED_SR_CHAIN_METHODS,
    require_sr_chain_method,
    sr_chain_method_for_iterations,
)

#: Engine A Monte Carlo semantics for each supported Sr method. The standard
#: identifier is unchanged from the release that introduced the two-refinement
#: chain, because its arithmetic is unchanged.
ENGINE_A_SEMANTICS_TWO_REFINEMENTS = (
    "engine_a.chain_replay.v4.monitor_blank.natural_init_two_refinements.finite_sample_precision"
)
ENGINE_A_SEMANTICS_SINGLE_PASS = (
    "engine_a.chain_replay.v4.monitor_blank.single_pass_measured_f.finite_sample_precision"
)
#: Records produced by the retired convergence-driven solver. Readable, never current.
ENGINE_A_SEMANTICS_RETIRED_BOUNDED_SOLVER = "engine_a.chain_replay.v2.finite_sample_precision"

ENGINE_A_SEMANTICS_BY_METHOD: Dict[str, str] = {
    SR_CHAIN_METHOD: ENGINE_A_SEMANTICS_TWO_REFINEMENTS,
    SR_SINGLE_PASS_METHOD: ENGINE_A_SEMANTICS_SINGLE_PASS,
}
SUPPORTED_ENGINE_A_SEMANTICS = frozenset(ENGINE_A_SEMANTICS_BY_METHOD.values())

#: Engine A freshness identity algorithm. ``v1`` requires complete matching
#: configuration and budget digests *and* a record semantics equal to the active
#: method's. Engine A records stamped with any other identity migrate as unknown.
ENGINE_A_FRESHNESS_IDENTITY_VERSION = "engine_a.freshness.v1"

ENGINE_A_ENGINE = "internal_normalization"


class SrChainMethodMismatchError(ValueError):
    """Stored Sr outputs were produced with a method the active configuration does not select."""


def configured_sr_chain_method(element_config: Any) -> str:
    """Method selected by an element configuration; the standard one when none is given."""
    iterations = getattr(element_config, "iterations", None) if element_config is not None else None
    if iterations is None:
        return SR_CHAIN_METHOD
    return sr_chain_method_for_iterations(iterations)


def recorded_sr_chain_method(sample: Any) -> Optional[str]:
    """Method recorded on a processed sample, or ``None`` for a legacy/unprocessed one."""
    metadata = getattr(sample, "metadata", None) or {}
    recorded = metadata.get("_sr_chain_method")
    return recorded if recorded else None


def replay_sr_chain_method(element_config: Any, sample: Any = None) -> str:
    """Method an uncertainty replay of ``sample`` must evaluate.

    A sample without a recorded method — a legacy session or a run in which the
    chain did not execute — uses the active element configuration. A recorded
    method must be supported and, when a configuration is given, agree with it.
    """
    configured = configured_sr_chain_method(element_config)
    recorded = recorded_sr_chain_method(sample) if sample is not None else None
    if recorded is None:
        return configured
    recorded = require_sr_chain_method(recorded)
    if element_config is not None and recorded != configured:
        raise SrChainMethodMismatchError(
            f"Sample '{getattr(sample, 'name', '')}' was processed with the Sr method "
            f"{recorded}, but the active element configuration selects {configured}. "
            f"Its stored outputs are stale for this configuration; reprocess the "
            f"session before evaluating its uncertainty."
        )
    return recorded


def engine_a_semantics_for_method(method: str) -> str:
    """Engine A MC semantics identifier for a supported Sr method."""
    return ENGINE_A_SEMANTICS_BY_METHOD[require_sr_chain_method(method)]


def active_engine_a_semantics(sample: Any = None, element_config: Any = None) -> str:
    """Semantics an Engine A MC record must carry to be current now, or ``""``.

    ``""`` means the active method cannot be established — no recorded method and
    no configuration, an unsupported recorded method, or a recorded method that
    disagrees with the configuration — and freshness must then stay unknown.
    """
    recorded = recorded_sr_chain_method(sample) if sample is not None else None
    configured = (
        configured_sr_chain_method(element_config) if element_config is not None else None
    )
    if recorded is not None:
        if recorded not in SUPPORTED_SR_CHAIN_METHODS:
            return ""
        if configured is not None and configured != recorded:
            return ""
        return ENGINE_A_SEMANTICS_BY_METHOD[recorded]
    if configured is not None:
        return ENGINE_A_SEMANTICS_BY_METHOD[configured]
    return ""

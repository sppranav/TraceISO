"""Shared engine-eligibility checks for stored and runtime uncertainty budgets.

The pipeline (stored budgets) and :mod:`domain.uncertainty.runtime` (runtime
budgets) route to the same three engines, so they must agree on *whether an
engine applies at all*. When those decisions were duplicated, the two routes
drifted: the pipeline refused to model a correction while runtime happily
returned a number for it (audit A019), and runtime guarded a missing
normalization layer that the pipeline did not (audit A012).

Every predicate here is pure, reads only the sample and the configuration, and
each refusal has exactly one reason string, so an unavailable budget carries the
same explanation whichever route produced it.
"""

from __future__ import annotations

from typing import Optional, Tuple

from config.settings import ProcessingConfig
from domain.elements.base import ElementConfig
from domain.layer_status import BLOCKING_STATUSES
from domain.models import Sample, UncertaintyBudget
from domain.pb_calibration_records import (
    ABSOLUTE_UNCERTAINTY_PENDING_REASON_CODE,
    DELTA_UNCERTAINTY_REASON,
    DELTA_UNCERTAINTY_REASON_CODE,
    PB_CALIBRATION_BUDGET_ENGINE,
    governing_calibration_record,
)
from domain.pb_correction_records import governing_hg_record
from domain.ratio_utils import normalize_ratio_name, normalize_ratio_token
from domain.uncertainty.blank import BlankSelection


#: Engine label for a correction TraceISO applies but does not model.
GENERIC_INTERNAL_UNAVAILABLE_ENGINE = "internal_normalization_unavailable"


def resolve_normalization_isotopes(
    element_config: Optional[ElementConfig],
    processing_config: Optional[ProcessingConfig],
) -> Optional[Tuple[str, str]]:
    """Return the active (numerator, denominator) normalization isotopes.

    Mirrors ``ProcessingPipeline._resolve_normalization_pair`` without touching
    the mass tables, which the eligibility question does not need.
    """
    if element_config is None:
        return None
    override = getattr(processing_config, "normalization_ratio_override", None)
    ratio_name = override or element_config.normalization_ratio
    if not ratio_name:
        return None
    normalized = normalize_ratio_name(ratio_name)
    if not normalized or normalized.count("/") != 1:
        return None
    numerator, denominator = normalized.split("/", 1)
    return normalize_ratio_token(numerator), normalize_ratio_token(denominator)


def sample_has_normalization_isotopes(
    sample: Sample,
    isotopes: Optional[Tuple[str, str]],
) -> bool:
    """Whether a sample carries the channels the normalization step consumes.

    Reads the same source layer the correction reads — blank-corrected
    intensities when present, raw otherwise — so a sample the pipeline skipped
    is recognised as skipped rather than re-derived from an output layer.
    """
    if isotopes is None:
        return True
    numerator, denominator = isotopes
    source = sample.corrected_intensities if sample.corrected_intensities else sample.intensities
    return numerator in source and denominator in source


def uses_generic_internal_normalization(
    element_config: Optional[ElementConfig],
    processing_config: Optional[ProcessingConfig],
) -> bool:
    """Whether this session applies Russell-law normalization with no engine for it.

    Sr has Engine A and Pb-Tl has Engine C. Any other element that is
    internally normalized takes ``_internal_normalization_simple``, whose
    uncertainty is not modelled by any of the three engines.
    """
    if element_config is None or processing_config is None:
        return False
    if element_config.symbol in {"Sr", "Pb"}:
        return False
    if not getattr(processing_config, "apply_mass_bias_correction", False):
        return False
    return resolve_normalization_isotopes(element_config, processing_config) is not None


def generic_internal_unavailable_budget(
    element_symbol: str,
    output_mode: str,
) -> UncertaintyBudget:
    """The single unavailable budget for an unmodelled internal normalization."""
    return UncertaintyBudget(
        engine=GENERIC_INTERNAL_UNAVAILABLE_ENGINE,
        output_mode=output_mode,
        budget_scope="unavailable",
        scope_note=(
            "Generic internal normalization was applied, but no validated "
            f"uncertainty engine exists for {element_symbol}."
        ),
    )


def missing_tl_normalized_layer_budget(
    sample: Sample,
    ratio_name: str,
    output_mode: str,
) -> Optional[UncertaintyBudget]:
    """Refuse an Engine C budget for a sample that was never Tl-normalized.

    Reaching Engine C means the session routed Pb-Tl external normalization. A
    sample with no normalized layer for this ratio was skipped by that
    correction, so its ratio is the uncorrected one and an Engine C budget would
    label an unnormalized measurand as normalized.
    """
    if sample.iif_corrected_ratios.get(ratio_name):
        return None
    available = sorted(sample.iif_corrected_ratios.keys())
    suffix = (
        f" Available Tl-normalized ratios: {', '.join(available)}."
        if available
        else ""
    )
    return UncertaintyBudget(
        engine="pb_tl_external_normalization",
        output_mode=output_mode,
        budget_scope="unavailable",
        scope_note=(
            f"Pb–Tl external normalization is active, but '{ratio_name}' has no "
            f"Tl-normalized corrected ratio layer for this sample.{suffix}"
        ),
    )


def hg_correction_unavailable_budget(
    sample: Sample,
    ratio_name: str,
    output_mode: str,
) -> Optional[UncertaintyBudget]:
    """Refuse a budget for a ratio whose requested Hg correction is unavailable.

    Only the ordinary-SSB Hg record governs the final layer. Without this
    refusal the engine would read the uncorrected blank-corrected ratio and
    report it under a corrected measurand.
    """
    record = governing_hg_record(sample, ratio_name)
    if record is None or record.status not in BLOCKING_STATUSES:
        return None
    return UncertaintyBudget(
        engine="ssb_delta",
        output_mode=output_mode,
        budget_scope="unavailable",
        scope_note=(
            f"Hg interference correction was requested for '{ratio_name}' but is "
            f"{record.status} for this observation ({record.reason_code}). {record.reason} "
            "No uncorrected ratio is reported in its place."
        ),
    )


def pb_calibration_budget_guard(
    sample: Sample,
    ratio_name: str,
    output_mode: str,
    *,
    extended_engine_c: bool = False,
) -> Optional[UncertaintyBudget]:
    """The refusal budget of a ratio whose final layer a Pb-standard calibration governs.

    Returns ``None`` only for an applied absolute-ratio result on a route that runs
    the extended Engine C (``extended_engine_c=True``, combined Pb plan C05).
    Otherwise the budget is ``unavailable`` and says why: the calibration itself is
    unavailable (no value is reported), its delta uncertainty is not calculated
    (owner-approved deferral), or the route does not run the extended Engine C.
    The Tl-only Engine C budget is never relabelled as the calibrated one, and no
    zero stands in for a missing uncertainty.
    """
    record = governing_calibration_record(sample, ratio_name)
    if record is None:
        return None
    if (
        extended_engine_c
        and record.status not in BLOCKING_STATUSES
        and str(output_mode or "").strip().lower() != "delta"
    ):
        return None
    final = (getattr(sample, "pb_standard_corrected_ratios", None) or {}).get(ratio_name)
    ratio_value = float(final.mean) if final is not None and final.n_valid else 0.0
    n_cycles = int(final.n_valid) if final is not None else 0
    if record.status in BLOCKING_STATUSES:
        note = (
            f"Pb-standard calibration was requested for '{ratio_name}' but is {record.status} for "
            f"this observation ({record.reason_code}). {record.reason} No Tl-normalized value is "
            "reported in its place."
        )
    elif str(output_mode or "").strip().lower() == "delta":
        note = f"Not calculated: {DELTA_UNCERTAINTY_REASON} ({DELTA_UNCERTAINTY_REASON_CODE})"
    else:
        note = (
            "Incomplete: the absolute-ratio uncertainty of a Pb-standard-calibrated result is modelled "
            "only by the extended Engine C on the Pb-Tl route, which this route does not run "
            f"({ABSOLUTE_UNCERTAINTY_PENDING_REASON_CODE}). The Tl-only Engine C budget is not "
            "relabelled as calibrated, and no uncertainty is reported."
        )
    return UncertaintyBudget(
        engine=PB_CALIBRATION_BUDGET_ENGINE,
        output_mode=output_mode,
        budget_scope="unavailable",
        scope_note=note,
        ratio_value=ratio_value,
        n_cycles=n_cycles,
    )


def missing_sr_normalization_input_budget(
    sample: Sample,
    ratio_name: str,
    element_config: ElementConfig,
    processing_config: ProcessingConfig,
    output_mode: str,
) -> Optional[UncertaintyBudget]:
    """Refuse an Engine A budget for a sample the Sr correction loop skipped.

    ``_sr_correction_loop`` needs both normalization-pair channels to compute
    the f-factor; without them it warns and moves on, leaving the sample's
    ratios uncorrected. Engine A's ``internal_normalization`` label would then
    describe a correction that never happened.

    The refusal requires positive evidence of the skip: the sample has source
    intensities, they lack the normalization pair, and no normalized layer was
    written for it. A caller that supplies a ratio layer directly, without
    intensities, is not claiming anything about a correction step and is left
    alone. Only ``87Sr/86Sr`` gets a normalized layer, so the check is
    per-sample rather than per-ratio; without the f-factor nothing on the
    sample was corrected.
    """
    if not (
        getattr(processing_config, "apply_interference_correction", False)
        or getattr(processing_config, "apply_mass_bias_correction", False)
    ):
        return None
    if sample.iif_corrected_ratios:
        return None
    source = sample.corrected_intensities if sample.corrected_intensities else sample.intensities
    if not source:
        return None
    isotopes = resolve_normalization_isotopes(element_config, processing_config)
    if sample_has_normalization_isotopes(sample, isotopes):
        return None
    numerator, denominator = isotopes  # type: ignore[misc]
    return UncertaintyBudget(
        engine="internal_normalization",
        output_mode=output_mode,
        budget_scope="unavailable",
        scope_note=(
            f"Internal normalization is active, but '{sample.name}' is missing the "
            f"normalization channels {numerator}/{denominator}, so '{ratio_name}' "
            "was never normalized for this sample."
        ),
    )


def unresolved_blank_reference_budget(
    sample: Sample,
    selection: "BlankSelection",
    *,
    engine: str,
    output_mode: str,
    basis_ratio_value: float,
    n_cycles: int,
    delta_reference_value: Optional[float] = None,
) -> UncertaintyBudget:
    """Refuse a budget whose required blank term cannot be identified.

    Every blank a session holds other than the one that was subtracted is a
    different measurement, so an unresolvable reference leaves no number to
    report — and leaving the term out instead would present an uncertainty
    computed without a contribution the configured budget requires.

    All three engines share this reason text for the same purpose the rest of
    this module exists: an unavailable budget must carry the same explanation
    whichever engine produced it. The measured and corrected ratios are
    untouched; only the uncertainty is withheld.
    """
    return UncertaintyBudget(
        engine=engine,
        output_mode=output_mode,
        ratio_value=0.0,
        basis_ratio_value=basis_ratio_value,
        delta_reference_value=delta_reference_value,
        n_cycles=n_cycles,
        budget_scope="unavailable",
        scope_note=(
            "Blank uncertainty is required for this budget, but the blank "
            f"reference recorded for '{sample.name}' could not be resolved "
            f"in this session ({selection.describe_unresolved()}). "
            "Reprocess the session, or restore the blank selection it was "
            "measured against, to report an uncertainty for this result."
        ),
    )


def required_contributor_unavailable_budget(
    *,
    engine: str,
    output_mode: str,
    contributor_name: str,
    reason: str,
    basis_ratio_value: float,
    n_cycles: int,
    ratio_value: float = 0.0,
    delta_reference_value: Optional[float] = None,
) -> UncertaintyBudget:
    """Refuse a complete budget when an enabled required term is unknown."""
    return UncertaintyBudget(
        engine=engine,
        output_mode=output_mode,
        ratio_value=ratio_value,
        basis_ratio_value=basis_ratio_value,
        delta_reference_value=delta_reference_value,
        n_cycles=n_cycles,
        budget_scope="unavailable",
        scope_note=(
            f"{contributor_name} is required for this budget but is unavailable: "
            f"{reason} The measured result is retained; no zero-uncertainty "
            "substitute was used."
        ),
    )

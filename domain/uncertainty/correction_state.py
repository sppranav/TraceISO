"""Calculation-time synchronization preserves selections and applied state."""
from dataclasses import replace
from config.settings import sync_uncertainty_with_processing


def synchronize_calculation_config(uncertainty, processing, *, ssb_applied):
    """Synchronize computed mode flags without treating this as a UI mode edit."""
    return sync_uncertainty_with_processing(
        uncertainty, replace(processing, enable_ssb=bool(ssb_applied)),
        preserve_contributor_selections=True,
    )


def runtime_correction_state(sample, samples, processing, element):
    """Read the producer's session route; retain conservative legacy guards."""
    recorded = sample.metadata.get("_applied_correction_state")
    if isinstance(recorded, dict):
        ssb_applied = bool(recorded["ssb"])
        pb_tl_applied = bool(recorded["pb_tl"])
    else:
        # Legacy outputs prove suppression/fallback only when actually present.
        has_iif = any(s.iif_corrected_ratios for s in samples)
        has_ssb = any(s.ssb_results for s in samples)
        ssb_applied = has_ssb if has_iif or has_ssb else processing.enable_ssb
        pb_tl_applied = has_iif or not has_ssb
    if element is not None and element.symbol == "Pb" and not pb_tl_applied:
        processing = replace(processing, apply_mass_bias_correction=False)
    return processing, ssb_applied

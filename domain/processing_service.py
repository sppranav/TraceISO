"""Shared production-processing service used by the TraceISO GUI."""

from __future__ import annotations

from typing import Iterable

from config.settings import ProcessingConfig, UncertaintyConfig
from domain.elements.base import ElementConfig
from domain.models import ProcessingResult, Sample
from domain import pipeline as pipeline_module


def process_samples(
    samples: Iterable[Sample], element: ElementConfig,
    processing_config: ProcessingConfig, uncertainty_config: UncertaintyConfig | None = None,
    *, profile_defaults=None, cycle_ranges=None, recorded_dependencies=None,
) -> ProcessingResult:
    """Single production entry point for already-loaded samples.

    ``cycle_ranges`` are the session's cycle windows; Sr- and Pb-standard
    calibration read them during processing (see ``ProcessingPipeline.process``).
    """
    if recorded_dependencies is not None:
        from config.recorded_dependencies import use_recorded_dependencies
        with use_recorded_dependencies(recorded_dependencies):
            return process_samples(samples, element, processing_config, uncertainty_config,
                                   profile_defaults=profile_defaults, cycle_ranges=cycle_ranges)
    sample_list = samples if isinstance(samples, list) else list(samples)
    from config.scientific_identity import snapshot, scientific_configuration
    from config.custom_uncertainty_contributors_loader import load_custom_contributors
    identity = scientific_configuration(element)
    identity['processing_identity_version'] = 'traceiso.processing_identity.v3'
    from domain.provenance import dependency_versions
    identity['dependency_versions'] = dependency_versions()
    from config.software_identity import software_identity
    identity['software_identity'] = software_identity()
    identity["custom_contributors"] = snapshot(load_custom_contributors())
    # Capture before the pipeline mutates/copies layers. Post-processing masks
    # alone cannot reconstruct the original processing inputs.
    identity['input_observations'] = snapshot(sample_list)
    from domain.uncertainty.contributors import PROFILE_DEFAULTS
    from config.recorded_dependencies import current_dependencies
    dependencies = current_dependencies()
    implicit_profiles = dependencies.profile_defaults if dependencies is not None else PROFILE_DEFAULTS
    identity.update({"processing": snapshot(processing_config),
                     "uncertainty": snapshot(uncertainty_config),
                     "cycle_ranges": snapshot(cycle_ranges or {}),
                     "profile_defaults": snapshot(profile_defaults if profile_defaults is not None else implicit_profiles),
                     "profile_defaults_were_explicit": profile_defaults is not None,
                     "scope": "stored_processing"})
    result = pipeline_module.ProcessingPipeline(element).process(
        sample_list, processing_config, uncertainty_config=uncertainty_config,
        profile_defaults=profile_defaults, cycle_ranges=cycle_ranges,
    )
    identity["observations"] = [
        {"observation_id": item.observation_id, "name": item.name,
         "run_number": item.run_number, "sample_type": item.sample_type,
         "metadata": snapshot(item.metadata), "used_blank_ids": snapshot(item.used_blank_ids),
         "masks": {layer: {name: snapshot(data.mask) for name, data in getattr(item, layer).items()}
                   for layer in ("ratios", "blank_corrected_ratios", "iif_corrected_ratios")}}
        for item in result.samples
    ]
    result.quality_metrics["processing_scientific_identity"] = identity
    return result

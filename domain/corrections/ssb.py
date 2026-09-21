"""Standard-Sample Bracketing (SSB) correction for TraceISO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from domain.models import Sample
from domain.pb_correction_records import hg_blocks_final_value
from domain.ratio_selection import (
    StandardProbe,
    find_nearest_usable_standard,
    format_skipped_standards,
    is_usable_bracketing_standard,
    select_best_pre_ssb_ratio_layer,
)


@dataclass
class SSBResult:
    """Per-sample SSB correction result for a single ratio."""

    corrected_cycles: np.ndarray
    prev_std_name: str
    next_std_name: str
    bracketing_avg: float
    k_factor: float
    prev_std_mean: float = 0.0
    next_std_mean: float = 0.0
    prev_std_se: float = 0.0
    next_std_se: float = 0.0
    prev_n: int = 0
    next_n: int = 0
    prev_std_run: Optional[float] = None
    next_std_run: Optional[float] = None


def apply_ssb(
    samples: List[Sample],
    ratio_name: str,
    certified_ratio: float,
    use_corrected: bool = True,
) -> List[Sample]:
    """Apply SSB correction to every non-blank, non-standard sample."""
    for idx, sample in enumerate(samples):
        if not sample.is_sample:
            if not sample.is_standard and not sample.is_blank:
                sample.warnings.append(
                    f"SSB skipped for {ratio_name}: sample type "
                    f"'{sample.sample_type}' is not corrected by SSB."
                )
            continue

        if use_corrected and hg_blocks_final_value(sample, ratio_name):
            # A requested Hg correction that is unavailable must not fall back
            # to the raw ratio below as the SSB source.
            sample.warnings.append(
                f"SSB skipped for {ratio_name}: the requested Hg interference correction "
                f"is unavailable for sample '{sample.name}'"
            )
            continue
        selected_source = select_best_pre_ssb_ratio_layer(sample, ratio_name) if use_corrected else None
        sample_cd = selected_source.data if selected_source is not None else sample.ratios.get(ratio_name)
        if sample_cd is None:
            sample.warnings.append(
                f"SSB skipped for {ratio_name}: sample '{sample.name}' has no source ratio data"
            )
            continue

        # Find bracketing standards
        prev_result = find_nearest_usable_standard(
            samples, idx, direction="before",
            probe=_bracketing_probe(ratio_name, use_corrected),
        )
        next_result = find_nearest_usable_standard(
            samples, idx, direction="after",
            probe=_bracketing_probe(ratio_name, use_corrected),
        )

        if prev_result.sample is None or next_result.sample is None:
            missing_sides = []
            if prev_result.sample is None:
                missing_sides.append("preceding standard (none usable — missing or all cycles excluded)")
            if next_result.sample is None:
                missing_sides.append("following standard (none usable — missing or all cycles excluded)")
            sample.warnings.append(
                f"SSB skipped for {ratio_name}: sample '{sample.name}' has no "
                + " and no ".join(missing_sides)
            )
            continue
        source_layer = selected_source.key if selected_source is not None else "ratios"
        if prev_result.layer_key != source_layer or next_result.layer_key != source_layer:
            sample.warnings.append(
                f"SSB skipped for {ratio_name}: sample and standards do not share one "
                f"source layer ({source_layer}, {prev_result.layer_key}, {next_result.layer_key})"
            )
            continue

        # Get cycle arrays
        prev_cd = prev_result.cycle_data
        next_cd = next_result.cycle_data
        prev_name, next_name = prev_result.name, next_result.name
        prev_run, next_run = prev_result.run_number, next_result.run_number
        all_skipped = prev_result.skipped + next_result.skipped

        # Emitted before the correction is attempted so bracket provenance
        # survives a downstream correction failure (the else branch below only
        # reports why the correction failed, not which standards were skipped).
        if all_skipped:
            sample.warnings.append(
                f"SSB bracket for {ratio_name} widened past unusable "
                f"standard(s): {format_skipped_standards(all_skipped)}"
            )

        result = _ssb_correct_cycles(
            sample_cycles=sample_cd.valid_values,
            prev_std_cycles=prev_cd.valid_values,
            next_std_cycles=next_cd.valid_values,
            certified_ratio=certified_ratio,
            prev_std_name=prev_name,
            next_std_name=next_name,
        )

        if result is not None:
            aligned_cycles = np.full(len(sample_cd.values), np.nan, dtype=np.float64)
            valid_indices = np.where(sample_cd.mask)[0]
            n = min(len(valid_indices), len(result.corrected_cycles))
            aligned_cycles[valid_indices[:n]] = result.corrected_cycles[:n]

            # Store as dict keyed by ratio name for downstream access.
            # prev_n / next_n / prev_std_run / next_std_run are stored
            # explicitly so that engine DoF resolution does not need to
            # parse the label or search samples by name (item 147).
            sample.ssb_results[ratio_name] = {
                "ssb_corrected_cycles": aligned_cycles,
                "ssb_mask": sample_cd.mask.copy(),
                "prev_std": result.prev_std_name,
                "next_std": result.next_std_name,
                "bracketing_avg": result.bracketing_avg,
                "k_factor": result.k_factor,
                "prev_std_mean": result.prev_std_mean,
                "next_std_mean": result.next_std_mean,
                "prev_std_se": result.prev_std_se,
                "next_std_se": result.next_std_se,
                "prev_n": len(prev_cd.valid_values),
                "next_n": len(next_cd.valid_values),
                "prev_std_run": prev_run,
                "next_std_run": next_run,
                # The identity of each selected bracketing observation. Names
                # and run numbers both repeat, so this is the only reference a
                # consumer can resolve back to the observation the correction
                # actually used (see TECHNICAL_DOCUMENTATION.md §4).
                "prev_std_obs": prev_result.sample.observation_id,
                "next_std_obs": next_result.sample.observation_id,
                "source_layer": source_layer,
                "skipped_standards": [s.to_payload() for s in all_skipped],
            }
        else:
            sample.warnings.append(
                f"SSB skipped for {ratio_name}: sample or standard cycles are empty, "
                "or the certified/bracketing ratio is not positive and finite"
            )

    return samples


def ssb_correct_single(
    sample_ratio: float,
    prev_std_ratio: float,
    next_std_ratio: float,
    certified_ratio: float,
) -> float:
    """SSB-correct a single scalar ratio value.

    Corrected = measured × (certified / bracketing_avg)
    """
    bracketing_avg = (prev_std_ratio + next_std_ratio) / 2.0
    if bracketing_avg <= 0:
        return np.nan
    return sample_ratio * (certified_ratio / bracketing_avg)


def _bracketing_probe(
    ratio_name: str, use_corrected: bool,
) -> Callable[[Sample], StandardProbe]:
    """Return a probe testing one candidate standard for SSB bracketing."""

    def probe(candidate: Sample) -> StandardProbe:
        if use_corrected and hg_blocks_final_value(candidate, ratio_name):
            return StandardProbe(False, reason="hg correction unavailable")
        selected = select_best_pre_ssb_ratio_layer(candidate, ratio_name) if use_corrected else None
        cd = selected.data if selected is not None else candidate.ratios.get(ratio_name)
        key = selected.key if selected is not None else "ratios"
        if cd is None:
            return StandardProbe(False, reason="missing ratio layer")
        if not is_usable_bracketing_standard(cd):
            return StandardProbe(False, reason="zero valid cycles")
        return StandardProbe(True, layer_key=key, cycle_data=cd)

    return probe


def _ssb_correct_cycles(
    sample_cycles: np.ndarray,
    prev_std_cycles: np.ndarray,
    next_std_cycles: np.ndarray,
    certified_ratio: float,
    prev_std_name: str,
    next_std_name: str,
) -> Optional[SSBResult]:
    """Core per-cycle SSB math."""
    if len(sample_cycles) == 0 or len(prev_std_cycles) == 0 or len(next_std_cycles) == 0:
        return None
    if not np.isfinite(certified_ratio) or certified_ratio <= 0:
        return None

    prev_mean = float(np.mean(prev_std_cycles))
    next_mean = float(np.mean(next_std_cycles))
    brack_avg = (prev_mean + next_mean) / 2.0

    if not np.isfinite(brack_avg) or brack_avg <= 0:
        return None

    k_factor = certified_ratio / brack_avg
    corrected = sample_cycles * k_factor

    n_prev = len(prev_std_cycles)
    n_next = len(next_std_cycles)
    prev_se = float(np.std(prev_std_cycles, ddof=1) / np.sqrt(n_prev)) if n_prev > 1 else 0.0
    next_se = float(np.std(next_std_cycles, ddof=1) / np.sqrt(n_next)) if n_next > 1 else 0.0

    return SSBResult(
        corrected_cycles=corrected,
        prev_std_name=prev_std_name,
        next_std_name=next_std_name,
        bracketing_avg=brack_avg,
        k_factor=k_factor,
        prev_std_mean=prev_mean,
        next_std_mean=next_mean,
        prev_std_se=prev_se,
        next_std_se=next_se,
        prev_n=n_prev,
        next_n=n_next,
    )


# Block-averaged SSB (block_average mode)

@dataclass
class StdBlock:
    """A group of consecutive standard samples forming a bracketing block."""

    indices: List[int]                # indices into the samples list
    names: List[str]                  # standard names in the block
    mean_ratio: float                 # arithmetic mean of individual standard means
    mean_position: float              # mean run_number (for LOO interpolation)
    individual_means: List[float]     # each standard's mean ratio
    sem: float                        # SEM = SD(individual_means) / sqrt(N)
    source_layer: str = "ratios"
    observation_ids: List[str] = field(default_factory=list)
    valid_counts: List[int] = field(default_factory=list)


def _block_members(block: StdBlock) -> List[Dict[str, Union[str, float, int]]]:
    """Every member of one bracket side with the weight its mean carries in B.

    B averages the two block means, each an equal-weight mean of its members, so
    a member of an n-member side enters B at 1/(2n). Names repeat, so the
    observation ID is the identity a replay resolves.
    """
    weight = 1.0 / (2.0 * len(block.individual_means))
    return [
        {
            "observation_id": observation_id,
            "label": name,
            "mean": float(mean),
            "n_valid": int(n_valid),
            "weight": weight,
        }
        for observation_id, name, mean, n_valid in zip(
            block.observation_ids, block.names, block.individual_means, block.valid_counts,
        )
    ]


def identify_std_blocks(
    samples: List[Sample],
    ratio_name: str,
    use_corrected: bool = True,
) -> List[StdBlock]:
    """Identify consecutive standard blocks in the sample sequence."""
    blocks: List[StdBlock] = []
    current_indices: List[int] = []
    current_names: List[str] = []
    current_means: List[float] = []
    current_positions: List[float] = []
    current_source_layers: List[str] = []
    current_ids: List[str] = []
    current_counts: List[int] = []

    def _flush() -> None:
        if not current_means:
            return
        block_mean = float(np.mean(current_means))
        block_pos = float(np.mean(current_positions))
        n = len(current_means)
        if n >= 2:
            sem = float(np.std(current_means, ddof=1) / np.sqrt(n))
        else:
            sem = 0.0
        blocks.append(StdBlock(
            indices=list(current_indices),
            names=list(current_names),
            mean_ratio=block_mean,
            mean_position=block_pos,
            individual_means=list(current_means),
            sem=sem,
            source_layer=(
                current_source_layers[0]
                if len(set(current_source_layers)) == 1
                else "mixed"
            ),
            observation_ids=list(current_ids),
            valid_counts=list(current_counts),
        ))

    for idx, sample in enumerate(samples):
        # A standard whose requested Hg correction is unavailable has no valid
        # data for this ratio: it ends the block rather than joining it raw.
        if sample.is_standard and not (use_corrected and hg_blocks_final_value(sample, ratio_name)):
            selected = select_best_pre_ssb_ratio_layer(sample, ratio_name) if use_corrected else None
            cd = selected.data if selected is not None else sample.ratios.get(ratio_name)
            if cd is not None and cd.n_valid >= 2:
                current_indices.append(idx)
                current_names.append(sample.name)
                current_means.append(cd.mean)
                current_positions.append(float(sample.run_number))
                current_source_layers.append(selected.key if selected is not None else "ratios")
                current_ids.append(sample.observation_id)
                current_counts.append(cd.n_valid)
                continue
        # Non-standard or no valid data → flush current block
        if current_indices:
            _flush()
            current_indices = []
            current_names = []
            current_means = []
            current_positions = []
            current_source_layers = []
            current_ids = []
            current_counts = []

    # Flush trailing block
    if current_indices:
        _flush()

    return blocks


def apply_ssb_block(
    samples: List[Sample],
    ratio_name: str,
    certified_ratio: float,
    use_corrected: bool = True,
) -> List[Sample]:
    """Apply block-averaged SSB correction."""
    blocks = identify_std_blocks(samples, ratio_name, use_corrected)
    if len(blocks) < 2:
        for sample in samples:
            if sample.is_sample:
                sample.warnings.append(
                    f"Block-average SSB skipped for {ratio_name}: "
                    f"fewer than 2 standard blocks found ({len(blocks)})."
                )
        return samples

    for idx, sample in enumerate(samples):
        if not sample.is_sample:
            if not sample.is_standard and not sample.is_blank:
                sample.warnings.append(
                    f"Block SSB skipped for {ratio_name}: sample type "
                    f"'{sample.sample_type}' is not corrected by SSB."
                )
            continue

        if use_corrected and hg_blocks_final_value(sample, ratio_name):
            sample.warnings.append(
                f"Block-average SSB skipped for {ratio_name}: the requested Hg interference "
                f"correction is unavailable for sample '{sample.name}'."
            )
            continue
        selected_source = select_best_pre_ssb_ratio_layer(sample, ratio_name) if use_corrected else None
        sample_cd = selected_source.data if selected_source is not None else sample.ratios.get(ratio_name)
        if sample_cd is None:
            sample.warnings.append(
                f"Block-average SSB skipped for {ratio_name}: sample '{sample.name}' "
                "has no source ratio data."
            )
            continue

        # Find nearest block before and after this sample index
        prev_block = _find_block_before(blocks, idx)
        next_block = _find_block_after(blocks, idx)

        if prev_block is None or next_block is None:
            sample.warnings.append(
                f"Block-average SSB skipped for {ratio_name}: missing bracketing standard block(s)."
            )
            continue
        source_layer = selected_source.key if selected_source is not None else "ratios"
        if (
            prev_block.source_layer != source_layer
            or next_block.source_layer != source_layer
        ):
            sample.warnings.append(
                f"Block-average SSB skipped for {ratio_name}: sample and standard blocks "
                f"do not share one source layer ({source_layer}, "
                f"{prev_block.source_layer}, {next_block.source_layer})."
            )
            continue

        sample_cycles = sample_cd.valid_values
        if len(sample_cycles) == 0:
            sample.warnings.append(
                f"Block-average SSB skipped for {ratio_name}: sample has no valid cycles."
            )
            continue

        brack_avg = (prev_block.mean_ratio + next_block.mean_ratio) / 2.0
        if (
            not np.isfinite(brack_avg)
            or not np.isfinite(certified_ratio)
            or brack_avg <= 0
            or certified_ratio <= 0
        ):
            sample.warnings.append(
                f"Block-average SSB skipped for {ratio_name}: certified or bracketing "
                "ratio is not positive."
            )
            continue

        k_factor = certified_ratio / brack_avg
        corrected = sample_cycles * k_factor

        # Align to full cycle array (same as alternating mode)
        aligned_cycles = np.full(len(sample_cd.values), np.nan, dtype=np.float64)
        valid_indices = np.where(sample_cd.mask)[0]
        n = min(len(valid_indices), len(corrected))
        aligned_cycles[valid_indices[:n]] = corrected[:n]

        prev_block_label = "+".join(prev_block.names)
        next_block_label = "+".join(next_block.names)

        sample.ssb_results[ratio_name] = {
            "ssb_corrected_cycles": aligned_cycles,
            "ssb_mask": sample_cd.mask.copy(),
            "prev_std": prev_block_label,
            "next_std": next_block_label,
            "bracketing_avg": brack_avg,
            "k_factor": k_factor,
            "prev_std_mean": prev_block.mean_ratio,
            "next_std_mean": next_block.mean_ratio,
            "prev_std_se": prev_block.sem,
            "next_std_se": next_block.sem,
            "ssb_mode": "block_average",
            "prev_block_sem": prev_block.sem,
            "next_block_sem": next_block.sem,
            # Explicit counts avoid engine-side split("+") label parsing (item 147).
            "prev_n": len(prev_block.individual_means),
            "next_n": len(next_block.individual_means),
            "source_layer": source_layer,
            # Every selected member and its weight in B, by observation identity
            # (Pb Hg plan R3): an uncertainty replay resolves these records
            # instead of searching the session for a block again.
            "prev_std_members": _block_members(prev_block),
            "next_std_members": _block_members(next_block),
        }

    return samples


def _find_block_before(blocks: List[StdBlock], sample_idx: int) -> Optional[StdBlock]:
    """Find the last block whose final index is before *sample_idx*."""
    result = None
    for block in blocks:
        if block.indices[-1] < sample_idx:
            result = block
        else:
            break
    return result


def _find_block_after(blocks: List[StdBlock], sample_idx: int) -> Optional[StdBlock]:
    """Find the first block whose first index is after *sample_idx*."""
    for block in blocks:
        if block.indices[0] > sample_idx:
            return block
    return None

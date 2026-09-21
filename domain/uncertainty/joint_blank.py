"""Observation-identity blank propagation for ordinary SSB (JCGM 100 section 5.2).

Distinct blank observations are modelled as independent; repeated identities
share one covariance block. SSB coefficients differentiate the recorded bracket.
"""
from dataclasses import dataclass
import numpy as np
from domain.ratio_selection import get_best_ratio_data
from domain.uncertainty.pb_hg_ssb_propagation import (
    _resolve_members, _window_mask, _blank_roles, _role_weight,
    _blank_covariance, HgBlankInput,
)
from domain.uncertainty.numerical_domain import validate_covariance_matrix
from domain.uncertainty.welch_satterthwaite import effective_dof

@dataclass(frozen=True)
class JointBlankResult:
    blank_inputs: tuple
    u_blank_rel: float
    u_blank_dof: float


def ordinary_ssb_blank(sample, ratio_name, all_samples, uncertainty_config,
                       processing_config=None, cycle_ranges=None, classic_delta=False,
                       runtime_sample_mask=None):
    """Differentiate ln(mean(N/D)) through target and recorded bracket members."""
    try:
        _, members, reason = _resolve_members(sample, ratio_name, all_samples,
            classic_delta=classic_delta, cycle_ranges=cycle_ranges)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Recorded bracket is incomplete or malformed.") from exc
    if reason:
        # A malformed bracket is relevant to blank propagation only when any
        # participating observation actually records a subtracted blank.
        if any(obs.used_blanks or obs.used_blank_ids for obs in all_samples if not obs.is_blank):
            raise ValueError(reason)
        return JointBlankResult((), 0.0, float("inf"))
    target = get_best_ratio_data(sample, ratio_name)
    if target is None:
        raise ValueError("Target ratio support is unavailable.")
    mask = np.asarray(target.mask, bool).copy() & _window_mask(sample, len(target.values), cycle_ranges)
    if runtime_sample_mask is not None:
        if mask.shape != np.asarray(runtime_sample_mask).shape:
            raise ValueError("Target support is not aligned.")
        mask &= runtime_sample_mask
    chains = [(sample, 1.0, mask)]
    if members:
        bracket = sum(weight * mean for _, _, weight, mean, _ in members)
        if not np.isfinite(bracket) or bracket == 0:
            raise ValueError("Invalid recorded bracket mean.")
        chains.extend((obs, -weight * mean / bracket, support) for obs, _, weight, mean, support in members)
    by_id = {}
    for obs in all_samples:
        if obs.observation_id in by_id:
            raise ValueError("Ambiguous observation identity in blank model.")
        by_id[obs.observation_id] = obs
    grouped = {}
    num, den = ratio_name.split("/")
    for obs, coefficient, support in chains:
        roles = _blank_roles(obs, processing_config)
        if not roles and obs.used_blanks and not obs.used_blank_ids:
            # Legacy label-only records resolve uniquely, then use observation IDs.
            for role, label in obs.used_blanks.items():
                matches = [b for b in all_samples if b.is_blank and b.name == label]
                if len(matches) != 1:
                    raise ValueError(f"Ambiguous or unresolved blank label {label!r}.")
                roles.append((role, matches[0].observation_id))
        if not roles:
            if obs.used_blanks or obs.used_blank_ids or any(obs.metadata.get("blank_channel_weights", {}).values()):
                raise ValueError(f"Blank identity unavailable for {obs.name}.")
            continue
        src = obs.blank_corrected_intensities or obs.corrected_intensities or obs.intensities
        if num not in src or den not in src:
            raise ValueError(f"Required ratio channels missing for {obs.name}.")
        n, d = np.asarray(src[num].values), np.asarray(src[den].values)
        if n.shape != support.shape or d.shape != support.shape or not support.any():
            raise ValueError(f"Unaligned or empty selected support for {obs.name}.")
        n, d = n[support], d[support]
        if not np.all(np.isfinite(n)) or not np.all(np.isfinite(d)) or np.any(d == 0):
            raise ValueError(f"Invalid required cycle for {obs.name}.")
        mean = np.mean(n / d)
        if not np.isfinite(mean) or mean == 0:
            raise ValueError(f"Invalid ratio mean for {obs.name}.")
        gradients = {num: -np.mean(1 / d) / mean, den: np.mean(n / d**2) / mean}
        for role, identity in roles:
            if identity not in by_id:
                raise ValueError(f"Unresolved blank observation {identity}.")
            group = grouped.setdefault(identity, {})
            for channel, gradient in gradients.items():
                weight = _role_weight(obs, channel, role, len(roles))
                if weight:
                    group[channel] = group.get(channel, 0.0) + coefficient * weight * gradient
    blocks = []
    for identity, gradient in grouped.items():
        channels = tuple(sorted(gradient))
        if not channels:
            continue
        stats = _blank_covariance(by_id[identity], channels, uncertainty_config, cycle_ranges)
        if stats is None:
            raise ValueError(f"Insufficient paired blank support for {by_id[identity].name}.")
        means, covariance, count, _ = stats
        validate_covariance_matrix(means, covariance)
        blocks.append(HgBlankInput(identity, by_id[identity].name, channels, means, covariance,
                                   np.array([gradient[c] for c in channels]), count))
    components = [(np.sqrt(max(b.variance_rel, 0)), b.n_pairs - 1) for b in blocks]
    return JointBlankResult(tuple(blocks), float(np.sqrt(sum(u*u for u, _ in components))), effective_dof(components))

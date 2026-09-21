"""Uncertainty of Hg-corrected ordinary Pb SSB ratios by first-order sensitivities.

Applies only to a 204Pb-bearing ratio whose governing ordinary-SSB Hg record is
``applied``. For every observation ``o`` in the reported quantity — the sample
and each recorded bracket member — the chain is

    I_c       = raw_c − Σ_role w_role,c · B̄_blank(role),c
    f         = ln(R_Tl · I_den / I_num) / ln(m_num / m_den)   (source "tl"), else 0
    Hg204     = I_202Hg · R_Hg · (m202 / m204)^f
    204Pb*    = I_204Pb − Hg204
    R̄_o       = mean over the frozen support of the per-cycle ratio

and the reported ratio is ``Y = R̄_s · C / B`` with ``B = Σ_j w_j M_j`` over the
recorded bracket (alternating sides at 1/2, every block member at 1/(2 n_side),
or no bracket when SSB was not applied). The relative sensitivity of an input
``x`` is ``∂lnY/∂x = ∂lnR̄_s/∂x − Σ_j a_j ∂lnR̄_j/∂x`` with ``a_j = w_j M_j / B``
from the recorded means, so committed drift scalars cancel and delta output
(``∂δ/∂x = (δ + 1000) ∂lnY/∂x``) shares the same relative terms.

What is propagated, each input exactly once:

* blank means of every blank observation subtracted from **any** chain, on the
  204Pb, 202Hg, other-isotope and (Tl chains) 203Tl/205Tl channels, with the
  within-blank channel covariance (JCGM 100:2008 §5.2) and one shared input per
  blank observation;
* the Hg reference ratio, one input shared by all chains;
* the Tl reference ratio through the Hg mass-bias factor of Tl chains.

What is not: per-cycle channel noise (already in the cycle SE of the corrected
layers), drift-model refits, invalid-cycle reclassification under perturbation,
and the Hg mass-bias transfer model itself (Tl→Hg exponential law, or ``f = 0``
without Tl), for which only the sensitivity is reported. Unknown reference
uncertainty stays ``None`` and is reported as omitted, never as zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from config.crm_schema import standard_uncertainty
from config.settings import ProcessingConfig, UncertaintyConfig
from domain.corrections.blank import BLANK_CHANNEL_WEIGHTS_KEY
from domain.filters.outlier import resolve_cycle_range, sample_cycle_key
from domain.layer_status import APPLIED
from domain.models import Sample, UncertaintyContributor
from domain.pb_correction_records import SOURCE_TL, governing_hg_record
from domain.pb_hg_correction import HG202, PB204, _ratio_isotopes
from domain.ratio_selection import (
    get_best_ratio_data,
    select_best_delta_ratio_layer,
    select_best_pre_ssb_ratio_layer,
)
from domain.uncertainty.blank import (
    MIN_BLANK_CYCLES_FOR_CORRELATION,
    _compute_empirical_correlation_matrix,
    _get_paired_blank_matrix,
    blank_input_sigmas,
    normalize_blank_uncertainty_input,
    resolve_blank_correction_mode,
)
from domain.uncertainty.welch_satterthwaite import effective_dof

#: Identity of this propagation model; part of the Engine B MC input identity.
PB_HG_SSB_PROPAGATION_METHOD = "pb_hg_ssb_linear_sensitivity.v1"

U_BLANK = "u_blank"
U_INTERF = "u_interf"
U_HG_TL_REFERENCE = "u_hg_tl_reference"
U_HG_MASS_BIAS_MODEL = "u_hg_mass_bias_model"

BRACKET_NONE = "none"
BRACKET_ALTERNATING = "alternating"
BRACKET_BLOCK = "block_average"
BRACKET_CLASSIC_DELTA = "classic_delta"

BLANK_EVALUATED = "evaluated"
BLANK_NOT_SUBTRACTED = "not_subtracted"
BLANK_INSUFFICIENT_DATA = "insufficient_blank_data"
BLANK_UNRESOLVED_REFERENCE = "unresolved_blank_reference"
BLANK_INVALID_COVARIANCE = "invalid_covariance"
BLANK_CHAIN_UNAVAILABLE = "chain_unavailable"

_REFERENCE = (
    "Reference: JCGM 100:2008 §5.1.3 (sensitivity coefficients) and §5.2.2 "
    "(correlated input quantities)."
)


@dataclass(frozen=True)
class ChainMember:
    """One observation of the reported quantity and its coefficient in ``lnY``."""

    observation_id: str
    label: str
    role: str
    weight: float
    recorded_mean: Optional[float]
    coefficient: float
    source: str


@dataclass(frozen=True, eq=False)
class HgBlankInput:
    """One blank observation as a shared input: covariance and ``∂lnY/∂B̄``."""

    blank_observation_id: str
    blank_label: str
    channels: Tuple[str, ...]
    mean_vector: np.ndarray
    covariance_matrix: np.ndarray
    gradient_rel: np.ndarray
    n_pairs: int

    @property
    def variance_rel(self) -> float:
        return float(self.gradient_rel @ self.covariance_matrix @ self.gradient_rel)


@dataclass(frozen=True, eq=False)
class HgSsbPropagation:
    """Linear Hg-correction propagation for one sample and ratio."""

    ratio_name: str
    sample_observation_id: str
    bracket_mode: str
    members: Tuple[ChainMember, ...]
    #: Non-empty when the chain cannot be evaluated; every term is then unknown.
    chain_reason: str
    blank_status: str
    blank_reason: str
    blank_inputs: Tuple[HgBlankInput, ...]
    u_blank_rel: Optional[float]
    u_blank_dof: float
    rho_r_hg: Optional[float]
    hg_reference: Mapping[str, Any]
    u_r_hg: Optional[float]
    interf_reason: str
    tl_used: bool
    rho_r_tl: Optional[float]
    tl_reference: Mapping[str, Any]
    u_r_tl: Optional[float]
    tl_reason: str
    rho_f_joint: Optional[float]
    rho_f_sample_only: Optional[float]
    method: str = PB_HG_SSB_PROPAGATION_METHOD

    @property
    def u_interf_rel(self) -> Optional[float]:
        if self.rho_r_hg is None or self.u_r_hg is None:
            return None
        return abs(self.rho_r_hg) * self.u_r_hg

    @property
    def u_tl_reference_rel(self) -> Optional[float]:
        if self.rho_r_tl is None or self.u_r_tl is None:
            return None
        return abs(self.rho_r_tl) * self.u_r_tl


def is_pb_hg_ssb_ratio(sample: Sample, ratio_name: str) -> bool:
    """Whether this ratio's reported value carries an applied ordinary-SSB Hg correction."""
    record = governing_hg_record(sample, ratio_name)
    return record is not None and record.status == APPLIED


def _window_mask(sample: Sample, n: int, cycle_ranges: Optional[Mapping[str, Tuple[int, int]]]) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    cycle_range = resolve_cycle_range(
        dict(cycle_ranges) if cycle_ranges else None,
        sample_name=sample.name,
        sample_key=sample_cycle_key(sample),
    )
    if cycle_range is not None:
        start, end = max(int(cycle_range[0]) - 1, 0), min(int(cycle_range[1]), n)
        mask[:] = False
        if end > start:
            mask[start:end] = True
    return mask


def _layer_mean(cycle_data, window: np.ndarray) -> Optional[float]:
    values = np.asarray(cycle_data.values, dtype=float)
    mask = np.asarray(cycle_data.mask, dtype=bool) & np.isfinite(values)
    if len(window) == len(mask):
        mask &= window
    return float(np.mean(values[mask])) if mask.any() else None


@dataclass(frozen=True)
class _ChainGradients:
    mean: float
    channels: Dict[str, float]
    r_hg: float
    r_tl: float
    f: float


def _chain_gradients(obs: Sample, record, ratio_isotopes: Tuple[str, str], mask: np.ndarray):
    """Closed-form ``∂lnR̄/∂`` for one observation, or a reason string."""
    num, den = ratio_isotopes
    other = den if num == PB204 else num
    src = obs.corrected_intensities if record.intensity_basis == "corrected_intensities" else obs.intensities
    tl_channels: Tuple[str, str] = ("", "")
    if record.source == SOURCE_TL:
        parts = str(record.tl_reference.get("ratio_name", "")).split("/")
        if len(parts) != 2:
            return "the Tl reference ratio of a Tl-assisted Hg correction is not recorded"
        tl_channels = (parts[0], parts[1])
    needed = [PB204, HG202, other] + ([tl_channels[0], tl_channels[1]] if record.source == SOURCE_TL else [])
    if any(channel not in src for channel in needed):
        return f"a channel of the Hg-corrected chain is absent for observation '{obs.name}'"
    arrays = {channel: np.asarray(src[channel].values, dtype=float) for channel in needed}
    if any(len(values) != len(mask) for values in arrays.values()):
        return f"the Hg-corrected chain of observation '{obs.name}' is not aligned with its support"
    support = mask.copy()
    for values in arrays.values():
        support &= np.isfinite(values)
    if not np.array_equal(support, mask):
        return f"required selected cycle is invalid for observation {obs.name!r}"
    if not support.any():
        return f"observation '{obs.name}' has no supported cycle for the Hg-corrected chain"
    ch = {channel: values[support] for channel, values in arrays.items()}

    masses = record.masses
    r_hg = float(record.hg_reference["value"])
    log_g = math.log(float(masses[HG202]) / float(masses["204Hg"]))
    if record.source == SOURCE_TL:
        tl_num, tl_den = tl_channels
        r_tl = float(record.tl_reference["value"])
        l_tl = math.log(float(masses[tl_num]) / float(masses[tl_den]))
        f = np.log(r_tl * ch[tl_den] / ch[tl_num]) / l_tl
    else:
        f = np.zeros(len(ch[PB204]))
    scale = np.exp(f * log_g)
    hg204 = ch[HG202] * r_hg * scale
    p_star = ch[PB204] - hg204
    if num == PB204:
        ratio = p_star / ch[other]
        d_other, d_pb, d_hg204 = -ratio / ch[other], 1.0 / ch[other], -1.0 / ch[other]
    else:
        ratio = ch[other] / p_star
        d_other, d_pb, d_hg204 = 1.0 / p_star, -ratio / p_star, ratio / p_star
    mean = float(np.mean(ratio))
    if not math.isfinite(mean) or mean == 0.0 or not np.all(np.isfinite(ratio)):
        return f"the Hg-corrected ratio of observation '{obs.name}' is not finite"

    d_f = d_hg204 * hg204 * log_g
    gradients = {
        other: float(np.mean(d_other)) / mean,
        PB204: float(np.mean(d_pb)) / mean,
        HG202: float(np.mean(d_hg204 * r_hg * scale)) / mean,
    }
    d_r_tl = 0.0
    if record.source == SOURCE_TL:
        gradients[tl_num] = float(np.mean(-d_f / (ch[tl_num] * l_tl))) / mean
        gradients[tl_den] = float(np.mean(d_f / (ch[tl_den] * l_tl))) / mean
        d_r_tl = float(np.mean(d_f)) / (r_tl * l_tl) / mean
    return _ChainGradients(
        mean=mean,
        channels=gradients,
        r_hg=float(np.mean(d_hg204 * ch[HG202] * scale)) / mean,
        r_tl=d_r_tl,
        f=float(np.mean(d_f)) / mean,
    )


def _resolve_members(
    sample: Sample,
    ratio_name: str,
    all_samples: Sequence[Sample],
    *,
    classic_delta: bool,
    cycle_ranges: Optional[Mapping[str, Tuple[int, int]]],
) -> Tuple[str, List[Tuple[Sample, str, float, float, np.ndarray]], str]:
    """``(bracket_mode, [(observation, role, weight, recorded_mean, mask)], reason)``."""
    by_id: Dict[str, List[Sample]] = {}
    for candidate in all_samples:
        by_id.setdefault(candidate.observation_id, []).append(candidate)

    def find(observation_id: object) -> Optional[Sample]:
        matches = by_id.get(str(observation_id or ""), [])
        return matches[0] if len(matches) == 1 else None

    def pre_ssb_mask(obs: Sample) -> Optional[np.ndarray]:
        selected = select_best_pre_ssb_ratio_layer(obs, ratio_name)
        return None if selected is None or selected.data is None else np.asarray(selected.data.mask, dtype=bool)

    if classic_delta:
        payload = (sample.delta_results or {}).get(ratio_name)
        if not isinstance(payload, dict):
            return BRACKET_CLASSIC_DELTA, [], "no delta bracket was recorded for this ratio"
        members = []
        for role, key in (("prev", "prev_std_obs"), ("next", "next_std_obs")):
            obs = find(payload.get(key))
            if obs is None:
                return BRACKET_CLASSIC_DELTA, [], f"the recorded {role} delta standard cannot be resolved by identity"
            selected = select_best_delta_ratio_layer(obs, ratio_name)
            if selected is None or selected.data is None:
                return BRACKET_CLASSIC_DELTA, [], f"the recorded {role} delta standard has no ratio layer"
            window = _window_mask(obs, len(selected.data.values), cycle_ranges)
            mean = _layer_mean(selected.data, window)
            if mean is None:
                return BRACKET_CLASSIC_DELTA, [], f"the recorded {role} delta standard has no valid cycle"
            members.append((obs, role, 0.5, mean, np.asarray(selected.data.mask, dtype=bool) & window))
        return BRACKET_CLASSIC_DELTA, members, ""

    payload = (sample.ssb_results or {}).get(ratio_name)
    if not isinstance(payload, dict) or not payload:
        return BRACKET_NONE, [], ""
    members = []
    if str(payload.get("ssb_mode") or "") == BRACKET_BLOCK:
        for side in ("prev", "next"):
            recorded = payload.get(f"{side}_std_members")
            if not isinstance(recorded, (list, tuple)) or not recorded:
                return BRACKET_BLOCK, [], (
                    "the block-average bracket does not record its member observations "
                    "(processed before member identities were persisted); reprocess the session"
                )
            for entry in recorded:
                obs = find(entry.get("observation_id")) if isinstance(entry, Mapping) else None
                mask = pre_ssb_mask(obs) if obs is not None else None
                if obs is None or mask is None:
                    return BRACKET_BLOCK, [], "a recorded block member cannot be resolved by identity"
                members.append((obs, side, float(entry["weight"]), float(entry["mean"]), mask))
        return BRACKET_BLOCK, members, ""
    for side in ("prev", "next"):
        obs = find(payload.get(f"{side}_std_obs"))
        mask = pre_ssb_mask(obs) if obs is not None else None
        if obs is None or mask is None:
            return BRACKET_ALTERNATING, [], f"the recorded {side} bracketing standard cannot be resolved by identity"
        members.append((obs, side, 0.5, float(payload[f"{side}_std_mean"]), mask))
    return BRACKET_ALTERNATING, members, ""


def _blank_roles(obs: Sample, processing_config: Optional[ProcessingConfig]) -> List[Tuple[str, str]]:
    mode = resolve_blank_correction_mode(obs, processing_config.blank_mode if processing_config else None)
    if mode == "none":
        return []
    roles = ("before", "after") if mode == "before_and_after" else ("before",)
    return [(role, str(obs.used_blank_ids.get(role) or "")) for role in roles if obs.used_blank_ids.get(role)]


def _role_weight(obs: Sample, channel: str, role: str, n_roles: int) -> float:
    recorded = (obs.metadata or {}).get(BLANK_CHANNEL_WEIGHTS_KEY)
    if not isinstance(recorded, Mapping):
        return 1.0 / n_roles
    per_role = recorded.get(channel)
    if not isinstance(per_role, Mapping):
        return 0.0
    try:
        value = float(per_role.get(role, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _is_psd(covariance: np.ndarray) -> bool:
    if covariance.size == 0:
        return True
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    tolerance = 100.0 * np.finfo(float).eps * covariance.shape[0] * scale
    return float(np.min(np.linalg.eigvalsh(covariance))) >= -tolerance


def _blank_covariance(
    blank: Sample,
    channels: Tuple[str, ...],
    uncertainty_config: UncertaintyConfig,
    cycle_ranges: Optional[Mapping[str, Tuple[int, int]]],
):
    paired = _get_paired_blank_matrix(blank, channels, cycle_ranges=dict(cycle_ranges) if cycle_ranges else None)
    if paired is None or paired.shape[1] < 2:
        return None
    n_pairs = int(paired.shape[1])
    sds = np.std(paired, axis=1, ddof=1)
    sigmas = blank_input_sigmas(
        sds, n_pairs, normalize_blank_uncertainty_input(getattr(uncertainty_config, "blank_uncertainty_input", "sd")),
    )
    dim = len(channels)
    method = str(getattr(uncertainty_config, "blank_correlation_method", "pearson_from_data"))
    if method == "uncorrelated" or dim == 1:
        corr = np.eye(dim)
    elif method == "fixed_value":
        corr = np.full((dim, dim), float(getattr(uncertainty_config, "blank_fixed_r", 0.0)))
        np.fill_diagonal(corr, 1.0)
    elif n_pairs < MIN_BLANK_CYCLES_FOR_CORRELATION:
        corr = np.eye(dim)
    else:
        corr = _compute_empirical_correlation_matrix(paired)
    return np.mean(paired, axis=1), np.outer(sigmas, sigmas) * corr, n_pairs, method


def compute_pb_hg_ssb_propagation(
    sample: Sample,
    ratio_name: str,
    *,
    all_samples: Sequence[Sample],
    uncertainty_config: UncertaintyConfig,
    processing_config: Optional[ProcessingConfig],
    cycle_ranges: Optional[Mapping[str, Tuple[int, int]]] = None,
    classic_delta: bool = False,
    runtime_sample_mask: Optional[np.ndarray] = None,
) -> Optional[HgSsbPropagation]:
    """Propagate the Hg-corrected chain, or ``None`` when no Hg correction governs the ratio.

    ``cycle_ranges`` windows the sample support and the blank cycles, as the
    runtime view does for every other Engine B blank sensitivity. Standards keep
    the support their bracket actually used; a classic delta bracket follows its
    own windowed resolution.
    """
    record = governing_hg_record(sample, ratio_name)
    if record is None or record.status != APPLIED:
        return None
    isotopes = _ratio_isotopes(ratio_name, {})
    samples = list(all_samples)

    def unknown(reason: str, bracket_mode: str = BRACKET_NONE) -> HgSsbPropagation:
        return HgSsbPropagation(
            ratio_name=ratio_name, sample_observation_id=sample.observation_id, bracket_mode=bracket_mode,
            members=(), chain_reason=reason, blank_status=BLANK_CHAIN_UNAVAILABLE, blank_reason=reason,
            blank_inputs=(), u_blank_rel=None, u_blank_dof=float("inf"), rho_r_hg=None,
            hg_reference=dict(record.hg_reference), u_r_hg=None, interf_reason=reason,
            tl_used=record.source == SOURCE_TL, rho_r_tl=None, tl_reference=dict(record.tl_reference),
            u_r_tl=None, tl_reason=reason, rho_f_joint=None, rho_f_sample_only=None,
        )

    if isotopes is None:
        return unknown("the ratio orientation cannot be resolved")
    best = get_best_ratio_data(sample, ratio_name)
    if best is None:
        return unknown("the reported ratio layer is absent")
    correction_mask = np.asarray(best.mask, dtype=bool)
    if runtime_sample_mask is None:
        sample_mask = correction_mask & _window_mask(sample, len(best.mask), cycle_ranges)
    else:
        sample_mask = np.asarray(runtime_sample_mask, dtype=bool)
        if len(sample_mask) != len(correction_mask):
            return unknown("the supplied runtime support is not cycle-aligned")
        if np.any(sample_mask & ~correction_mask):
            return unknown("the supplied runtime support reintroduces correction-invalid cycles")
        sample_mask = sample_mask.copy()
    if not np.any(sample_mask):
        return unknown("the supplied runtime support contains no accepted cycles")

    bracket_mode, resolved, reason = _resolve_members(
        sample, ratio_name, samples, classic_delta=classic_delta, cycle_ranges=cycle_ranges,
    )
    if reason:
        return unknown(reason, bracket_mode)
    bracket = sum(weight * mean for _obs, _role, weight, mean, _mask in resolved)
    if resolved and (not math.isfinite(bracket) or bracket <= 0.0):
        return unknown("the recorded bracket mean is not positive and finite", bracket_mode)

    chains: List[Tuple[Sample, Any, float, np.ndarray, str, float, Optional[float]]] = [
        (sample, record, 1.0, sample_mask, "sample", 0.0, None),
    ]
    for obs, role, weight, mean, mask in resolved:
        obs_record = governing_hg_record(obs, ratio_name)
        if obs_record is None or obs_record.status != APPLIED:
            return unknown(f"bracket member '{obs.name}' has no applied Hg correction for {ratio_name}", bracket_mode)
        chains.append((obs, obs_record, -weight * mean / bracket, mask, role, weight, mean))

    grads: List[_ChainGradients] = []
    for obs, obs_record, _coef, mask, _role, _w, _m in chains:
        result = _chain_gradients(obs, obs_record, isotopes, mask)
        if isinstance(result, str):
            return unknown(result, bracket_mode)
        grads.append(result)

    members = tuple(
        ChainMember(
            observation_id=obs.observation_id, label=obs.name, role=role, weight=float(weight),
            recorded_mean=None if mean is None else float(mean), coefficient=float(coef),
            source=str(obs_record.source),
        )
        for (obs, obs_record, coef, _mask, role, weight, mean) in chains
    )
    rho_r_hg = sum(c[2] * g.r_hg for c, g in zip(chains, grads))
    rho_r_tl = sum(c[2] * g.r_tl for c, g in zip(chains, grads))
    rho_f = sum(c[2] * g.f for c, g in zip(chains, grads))

    # --- Hg reference: one shared input --------------------------------------
    hg_reference = dict(record.hg_reference)
    u_r_hg: Optional[float] = None
    interf_reason = ""
    if any(dict(c[1].hg_reference) != hg_reference for c in chains[1:]):
        interf_reason = "the sample and its bracket members did not use one Hg reference record"
    else:
        u_r_hg = standard_uncertainty(
            hg_reference.get("uncertainty"), hg_reference.get("k"), hg_reference.get("uncertainty_semantics"),
        )
        if u_r_hg is None:
            semantics = str(hg_reference.get("uncertainty_semantics") or "")
            interf_reason = (
                f"the {hg_reference.get('ratio_name', '204Hg/202Hg')} reference uncertainty is unassigned "
                f"({hg_reference.get('record_id', '')}); it is omitted, not zero"
                if semantics == "unassigned" or hg_reference.get("uncertainty") is None
                else f"the Hg reference uncertainty ({semantics}) has no convertible standard uncertainty"
            )

    # --- Tl reference through the Hg mass-bias factor of Tl chains -----------
    tl_chains = [c for c in chains if c[1].source == SOURCE_TL]
    tl_reference = dict(tl_chains[0][1].tl_reference) if tl_chains else {}
    u_r_tl: Optional[float] = None
    tl_reason = ""
    if tl_chains:
        if any(dict(c[1].tl_reference) != tl_reference for c in tl_chains[1:]):
            tl_reason = "the Tl-assisted Hg corrections did not use one Tl reference"
        else:
            from domain.uncertainty.engine_external_pb_tl import (
                resolve_pb_tl_normalization_standard_uncertainty,
            )

            try:
                u_r_tl = float(resolve_pb_tl_normalization_standard_uncertainty(
                    uncertainty_config, str(tl_reference.get("ratio_name", "")),
                ))
            except ValueError as exc:
                tl_reason = f"{exc} It is omitted, not zero."

    # --- blank observations: shared inputs with channel covariance -----------
    blank_status, blank_reason = BLANK_NOT_SUBTRACTED, ""
    gradient_by_blank: Dict[str, Dict[str, float]] = {}
    for (obs, _rec, coef, _mask, _role, _w, _m), grad in zip(chains, grads):
        roles = _blank_roles(obs, processing_config)
        for role, blank_id in roles:
            per_channel = gradient_by_blank.setdefault(blank_id, {})
            for channel, value in grad.channels.items():
                weight = _role_weight(obs, channel, role, len(roles))
                if weight > 0.0:
                    per_channel[channel] = per_channel.get(channel, 0.0) + coef * (-weight) * value

    by_id = {s.observation_id: s for s in samples}
    blank_inputs: List[HgBlankInput] = []
    for blank_id in sorted(gradient_by_blank):
        gradient = gradient_by_blank[blank_id]
        if not gradient:
            continue
        blank = by_id.get(blank_id)
        if blank is None:
            blank_status = BLANK_UNRESOLVED_REFERENCE
            blank_reason = f"blank observation '{blank_id}' subtracted from a chain is not in this session"
            blank_inputs = []
            break
        channels = tuple(sorted(gradient))
        resolved_cov = _blank_covariance(blank, channels, uncertainty_config, cycle_ranges)
        if resolved_cov is None:
            blank_status = BLANK_INSUFFICIENT_DATA
            blank_reason = (
                f"blank '{blank.name}' has fewer than two paired cycles on {', '.join(channels)}"
            )
            blank_inputs = []
            break
        means, covariance, n_pairs, method = resolved_cov
        if not np.all(np.isfinite(covariance)) or not _is_psd(covariance):
            blank_status = BLANK_INVALID_COVARIANCE
            blank_reason = (
                f"the {len(channels)}-channel blank covariance of '{blank.name}' ({method}) is not "
                "positive semidefinite; no independent-channel substitute is used"
            )
            blank_inputs = []
            break
        blank_inputs.append(HgBlankInput(
            blank_observation_id=blank_id, blank_label=blank.name, channels=channels,
            mean_vector=np.asarray(means, dtype=float), covariance_matrix=np.asarray(covariance, dtype=float),
            gradient_rel=np.array([gradient[c] for c in channels], dtype=float), n_pairs=n_pairs,
        ))
        blank_status = BLANK_EVALUATED

    u_blank_rel: Optional[float] = None
    u_blank_dof = float("inf")
    if blank_status == BLANK_EVALUATED:
        variances = [max(b.variance_rel, 0.0) for b in blank_inputs]
        u_blank_rel = math.sqrt(sum(variances))
        u_blank_dof = effective_dof([
            (math.sqrt(v), float(max(b.n_pairs - 1, 1))) for v, b in zip(variances, blank_inputs)
        ])
    elif blank_status == BLANK_NOT_SUBTRACTED:
        u_blank_rel = 0.0

    return HgSsbPropagation(
        ratio_name=ratio_name, sample_observation_id=sample.observation_id, bracket_mode=bracket_mode,
        members=members, chain_reason="", blank_status=blank_status, blank_reason=blank_reason,
        blank_inputs=tuple(blank_inputs), u_blank_rel=u_blank_rel, u_blank_dof=u_blank_dof,
        rho_r_hg=float(rho_r_hg), hg_reference=hg_reference, u_r_hg=u_r_hg, interf_reason=interf_reason,
        tl_used=bool(tl_chains), rho_r_tl=float(rho_r_tl), tl_reference=tl_reference, u_r_tl=u_r_tl,
        tl_reason=tl_reason, rho_f_joint=float(rho_f), rho_f_sample_only=float(chains[0][2] * grads[0].f),
    )


def _sources_text(propagation: HgSsbPropagation) -> str:
    sample = next((m for m in propagation.members if m.role == "sample"), None)
    standards = sorted({m.source for m in propagation.members if m.role != "sample"})
    parts = [f"sample source {sample.source}" if sample else "sample source unknown"]
    if standards:
        parts.append(f"bracket member source(s) {', '.join(standards)}")
    return "; ".join(parts)


def hg_propagation_contributor_rows(
    propagation: HgSsbPropagation,
    *,
    ratio_mean: float,
    gate: Callable[..., Dict[str, Any]],
) -> List[UncertaintyContributor]:
    """``u_interf``, ``u_hg_tl_reference`` (Tl chains only) and ``u_hg_mass_bias_model`` rows.

    ``gate(name, data_available, missing_reason)`` is the engine's contributor
    gate, so user and profile exclusions keep their existing precedence.
    """
    rows: List[UncertaintyContributor] = []
    scale = abs(float(ratio_mean))

    u_interf = propagation.u_interf_rel
    interf_missing = propagation.chain_reason or propagation.interf_reason
    rows.append(UncertaintyContributor(
        name=U_INTERF,
        display_name="Hg reference ratio in the Hg correction (Type B)",
        value_abs=(u_interf or 0.0) * scale,
        value_rel_permil=1000.0 * (u_interf or 0.0),
        type_ab="B",
        degrees_of_freedom=float("inf"),
        percentage_contribution=0.0,
        description=(
            "204Hg/202Hg reference ratio propagated jointly through the Hg subtraction of the sample "
            "and every recorded bracket member: u_rel = |∂lnY/∂R_Hg|·u(R_Hg)"
            + (f", ∂lnY/∂R_Hg = {propagation.rho_r_hg:.6g}" if propagation.rho_r_hg is not None else "")
            + f". {_REFERENCE}"
        ),
        reference="JCGM 100:2008 §5.1.3; Pb SSB Hg interference plan R2",
        **gate(U_INTERF, u_interf is not None,
               f"u_interf omitted: {interf_missing}." if interf_missing else ""),
    ))

    if propagation.tl_used:
        u_tl = propagation.u_tl_reference_rel
        tl_missing = propagation.chain_reason or propagation.tl_reason
        rows.append(UncertaintyContributor(
            name=U_HG_TL_REFERENCE,
            display_name="Tl reference ratio via the Hg mass-bias factor (Type B)",
            value_abs=(u_tl or 0.0) * scale,
            value_rel_permil=1000.0 * (u_tl or 0.0),
            type_ab="B",
            degrees_of_freedom=float("inf"),
            percentage_contribution=0.0,
            description=(
                "Accepted Tl ratio entering the per-cycle Russell factor that scales the Hg subtraction of "
                "Tl-assisted chains; Pb is not externally normalized: u_rel = |∂lnY/∂R_Tl|·u(R_Tl)"
                + (f", ∂lnY/∂R_Tl = {propagation.rho_r_tl:.6g}" if propagation.rho_r_tl is not None else "")
                + f". {_REFERENCE}"
            ),
            reference="JCGM 100:2008 §5.1.3; Russell et al. (1978) exponential law",
            **gate(U_HG_TL_REFERENCE, u_tl is not None,
                   f"u_hg_tl_reference omitted: {tl_missing}" if tl_missing else ""),
        ))

    if propagation.rho_f_joint is not None:
        sensitivity = (
            f"Linear sensitivity 1000·∂lnY/∂f_Hg = {1000.0 * propagation.rho_f_joint:.6g} ‰ per unit f "
            f"through the sample and bracket members ({1000.0 * propagation.rho_f_sample_only:.6g} ‰ "
            "through the sample chain alone)."
        )
    else:
        sensitivity = f"The sensitivity could not be evaluated: {propagation.chain_reason}."
    reason = (
        "The Hg mass-bias model is not qualified: the exponential-law transfer of the Tl factor to Hg "
        "(Tl-assisted chains) and the zero Hg mass bias of natural-ratio subtraction (no Tl) have no "
        f"approved standard uncertainty ({_sources_text(propagation)}). {sensitivity} "
        "This term is omitted, not zero, so the combined uncertainty does not cover it."
    )
    rows.append(UncertaintyContributor(
        name=U_HG_MASS_BIAS_MODEL,
        display_name="Hg mass-bias model (not qualified)",
        value_abs=0.0,
        value_rel_permil=0.0,
        type_ab="B",
        degrees_of_freedom=float("inf"),
        percentage_contribution=0.0,
        description=f"{reason} Reference: JCGM 100:2008 §5.1; Pb SSB Hg interference plan R2 (qualification).",
        reference="JCGM 100:2008 §5.1; Pb SSB Hg interference plan R2",
        state="NO_APPROVED_MODEL",
        inactive_reason=reason,
    ))
    return rows


def hg_blank_row_values(propagation: HgSsbPropagation, ratio_mean: float) -> Tuple[float, float, float, str]:
    """``(value_abs, value_rel_permil, dof, missing_reason)`` of the corrected-model ``u_blank``."""
    u_rel = propagation.u_blank_rel
    if u_rel is None:
        return 0.0, 0.0, float("inf"), f"u_blank through the Hg-corrected model is unavailable: {propagation.blank_reason}."
    missing = ""
    if propagation.blank_status == BLANK_NOT_SUBTRACTED:
        missing = "not_applicable:no blank was subtracted from the sample or its bracket members"
    return u_rel * abs(float(ratio_mean)), 1000.0 * u_rel, propagation.u_blank_dof, missing

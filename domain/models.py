"""Core domain models for TraceISO."""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np


# Cycle-level data container

@dataclass
class CycleData:
    """Per-cycle measurement array with an associated validity mask."""

    values: np.ndarray
    mask: np.ndarray

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.mask is None:
            self.mask = np.ones(len(self.values), dtype=bool)
        else:
            self.mask = np.asarray(self.mask, dtype=bool)
        if len(self.values) != len(self.mask):
            raise ValueError(
                f"values length ({len(self.values)}) != mask length ({len(self.mask)})"
            )
        # A valid cycle must always contain a finite numeric value. Use a
        # non-mutating expression so caller-owned mask arrays are not modified.
        self.mask = self.mask & np.isfinite(self.values)


    @property
    def n_total(self) -> int:
        """Total number of cycles (including masked-out)."""
        return len(self.values)

    @property
    def n_valid(self) -> int:
        """Number of valid (unmasked) cycles."""
        return int(np.sum(self.mask))

    @property
    def valid_values(self) -> np.ndarray:
        """Return only the values where mask is True."""
        return self.values[self.mask]


    @property
    def mean(self) -> float:
        v = self.valid_values
        return float(np.mean(v)) if len(v) > 0 else np.nan

    @property
    def sd(self) -> float:
        v = self.valid_values
        return float(np.std(v, ddof=1)) if len(v) > 1 else np.nan

    @property
    def se(self) -> float:
        v = self.valid_values
        n = len(v)
        if n > 1:
            return float(np.std(v, ddof=1) / np.sqrt(n))
        return np.nan

    @property
    def rsd_percent(self) -> float:
        m = self.mean
        if np.isfinite(m) and m != 0:
            return abs(self.sd / m) * 100.0
        return np.nan

    def copy(self) -> "CycleData":
        """Return an independent copy with duplicated ``values`` and ``mask`` arrays."""
        return CycleData(values=self.values.copy(), mask=self.mask.copy())


@dataclass
class UncertaintyContributor:
    """A single named uncertainty contributor with full metadata."""

    name: str                          # e.g. "u_prec", "u_blank", "kappa_dry_ashing"
    display_name: str                  # e.g. "Sample measurement precision", "Blank contribution"
    value_abs: float                   # absolute standard uncertainty
    value_rel_permil: float            # relative standard uncertainty in permil
    type_ab: str                       # "A" or "B" (GUM classification)
    degrees_of_freedom: float          # n-1 for Type A, inf for Type B
    percentage_contribution: float     # variance-share % used for budget display/export
    description: str = ""              # human-readable explanation with reference
    reference: str = ""                 # citation/source for user-managed terms
    is_active: bool = True             # False if structurally absent
    state: str = ""                    # ContributorState enum value; empty = legacy path
    inactive_reason: str = ""          # free-text explanation for non-ACTIVE state
    # A029: the sampling distribution of a user-defined term. Built-in
    # contributor PDFs live on UncertaintyConfig and are covered by the
    # configuration identity; a custom term's PDF arrives with the definition
    # and is invisible everywhere else, yet it changes what Monte Carlo draws
    # at an unchanged magnitude. Empty means "not a distribution-bearing row".
    distribution: str = ""

    def __post_init__(self) -> None:
        # Backward-compat: if state is empty (legacy constructor), infer from is_active.
        # - If state is set (new path): derive is_active from state.
        # This preserves existing callers that pass only is_active=False
        # (MISSING_DATA semantics) without accidentally promoting them to
        # BY_SAMPLE_DESIGN.
        if not self.state:
            self.state = "ACTIVE" if self.is_active else "MISSING_DATA"
            return
        self.is_active = (self.state == "ACTIVE")


_COMPAT_COMPONENT_ALIASES = {
    "u_crm": "crm",
    "u_interf": "interf",
    "u_kappa_drift": "drift",
    "u_k1_sample_decomposition": "k1",
    "u_k2_matrix_separation": "k2",
    "u_k3_procedural_blank": "k3",
    "u_k4_bracketing_standard_heterogeneity": "k4",
    "u_k5_instrumental_drift": "k5",
    "u_k6_matrix_effects": "k6",
    "u_k7_residual_interferences": "k7",
    "u_k2": "k2",
    "u_k3": "k3",
    "u_k4": "k4",
}


@dataclass
class ReprodResult:
    """Result of u_std_repeatability computation with full metadata for UI display."""

    method: str                          # which strategy was used
    u_std_repeatability_abs: float       # absolute standard uncertainty
    u_std_repeatability_rel_permil: float  # relative in permil
    degrees_of_freedom: int

    # Per-standard data (for plotting)
    std_names: List[str]                 # names of all standards
    std_means: np.ndarray                # mean ratios of all standards
    std_positions: np.ndarray            # sequence positions (run_number)
    std_included: List[bool]             # which are included in calculation
    std_segments: List[int]              # segment assignment for each

    # Observation identities behind each displayed row: one entry per row, each
    # a list of the observation IDs that row aggregates. In alternating mode
    # that is a single standard; in block_average mode it is every standard in
    # the block, whose display name is their joined labels ("A+B"). This is the
    # only reference an include/exclude control can round-trip: a display name
    # is neither unique nor decomposable back into observations.
    std_identities: Optional[List[List[str]]] = None

    # Residuals (for LOO and drift methods)
    residuals: Optional[np.ndarray] = None
    predicted_values: Optional[np.ndarray] = None

    # Segment-level results
    segment_sds: Optional[Dict[int, float]] = None
    segment_dofs: Optional[Dict[int, int]] = None
    segment_n_stds: Optional[Dict[int, int]] = None

    # Break detection
    detected_breaks: Optional[List[int]] = None

    # kappa_drift (if enabled)
    kappa_drift_permil: float = 0.0
    drift_deltas: Optional[np.ndarray] = None

    # Block-averaged SSB (block_average mode)
    within_block_sem: Optional[float] = None             # pooled SEM within blocks
    within_block_sem_rel_permil: Optional[float] = None
    within_block_dof: Optional[int] = None               # sum(n_i - 1) across blocks
    block_means: Optional[np.ndarray] = None             # block mean ratios (for UI)
    block_positions: Optional[np.ndarray] = None         # block mean positions
    n_blocks: Optional[int] = None

    # Fallback info
    fallback_reason: Optional[str] = None
    requested_method: Optional[str] = None

    # Scientific availability/evidence.  A zero magnitude is not used to mean
    # that an unsupported estimator is exact.
    status: str = "available"
    unavailable_reason: Optional[str] = None
    estimator_identity: str = ""
    requested_estimator_identity: str = ""
    error_model: str = ""
    fitted_observations: Optional[int] = None
    design_rank: Optional[int] = None
    residual_dof: Optional[int] = None
    residual_sse: Optional[float] = None
    residual_scale: Optional[float] = None
    residual_linear_map: Optional[np.ndarray] = None
    skipped_segments: Tuple[int, ...] = ()


@dataclass
class UncertaintyBudget:
    """GUM-compliant uncertainty budget for a single ratio measurement.

    Canonical fields are the source of truth for new code:
    ``u_combined_abs``, ``u_combined_rel_permil``, ``expanded_abs``,
    ``expanded_rel_permil``, ``coverage_factor_k``, and ``contributors``.
    """

    contributors: List[UncertaintyContributor] = field(default_factory=list)
    u_combined_abs: float = 0.0
    u_combined_rel_permil: float = 0.0
    expanded_abs: float = 0.0
    expanded_rel_permil: float = 0.0
    effective_dof: float = float('inf')
    coverage_factor_k: float = 2.0
    coverage_method: str = "unknown"
    coverage_probability: Optional[float] = None
    coverage_factor_rule: str = "unknown"
    coverage_semantics_version: str = "unknown"
    dominant_contributor: str = ""
    engine: str = ""
    output_mode: str = ""
    mc_lower_95: Optional[float] = None
    mc_upper_95: Optional[float] = None
    mc_gum_agreement: Optional[float] = None
    reprod_result: Optional[ReprodResult] = None
    basis_ratio_value: Optional[float] = None
    delta_reference_value: Optional[float] = None
    delta_scale_factor: Optional[float] = None
    certified_reference_value: Optional[float] = None
    absolute_scale_factor: Optional[float] = None

    # Indicates which contributors are runtime-recomputed vs session-stored.
    # "full_runtime" = all contributors recomputed from current cycle window.
    # "hybrid" = some contributors are session-level (u_std, u_std_repeatability).
    # "stored" = pipeline-time budget, not recomputed.
    budget_scope: str = "stored"
    # Reason for None/missing budget (set by runtime layer)
    scope_note: str = ""

    # Result-level scientific coverage statements are deliberately separate
    # from numerical contributors: no zero-valued RSS term is invented merely
    # to make an unquantified limitation visible.
    coverage_limitations: List[Dict[str, str]] = field(default_factory=list)
    # Complete resolved replay-input identity used by durable MC freshness.
    # Empty means historical/unknown, never current.
    replay_input_digest: str = ""

    ratio_value: float = 0.0
    n_cycles: int = 0

    def reported_measurand_value(self) -> Optional[float]:
        """Return the value this budget's uncertainty belongs to.

        An absolute-ratio budget reports ``ratio_value``. A delta budget keeps a
        ratio in ``ratio_value`` but reports delta in permil,
        ``(basis / delta reference - 1) * 1000``, taking the engine's stored
        ``delta_scale_factor`` when present. This is the same conversion the
        Uncertainty tab displays. None when the scope is invalid or the inputs
        are not finite, so an undefined value is never written as a number.
        """
        from domain.uncertainty.scope import is_invalid_budget_scope

        if is_invalid_budget_scope(self):
            return None

        def _finite(value: object) -> Optional[float]:
            try:
                out = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            return out if np.isfinite(out) else None

        if (self.output_mode or "").strip().lower() != "delta":
            value = _finite(self.ratio_value)
            return value if value not in (None, 0.0) else None

        scale = _finite(self.delta_scale_factor)
        if scale is None or scale <= 0.0:
            ratio = _finite(self.basis_ratio_value) or _finite(self.ratio_value)
            reference = _finite(self.delta_reference_value)
            if ratio is None or reference in (None, 0.0):
                return None
            scale = ratio / reference
        return (scale - 1.0) * 1000.0

    def _find_contributor(self, name: str) -> Optional["UncertaintyContributor"]:
        """Find a contributor by name."""
        return next((c for c in self.contributors if c.name == name), None)

    def contributor_value_abs(self, name: str) -> float:
        """Return one contributor's absolute uncertainty, or zero if absent."""
        contributor = self._find_contributor(name)
        return float(contributor.value_abs) if contributor is not None else 0.0

    def contributor_value_ppm(self, name: str) -> float:
        """Return one contributor's relative uncertainty in ppm, or zero if absent."""
        contributor = self._find_contributor(name)
        if contributor is None:
            return 0.0
        return float(contributor.value_rel_permil) * 1000.0

    def type_ab_abs(self, type_ab: str) -> float:
        """Return RSS absolute uncertainty for active contributors of a Type A/B class."""
        values = [
            float(c.value_abs)
            for c in self.contributors
            if c.type_ab == type_ab and c.is_active
        ]
        return float(np.sqrt(sum(value * value for value in values))) if values else 0.0

    def type_ab_ppm(self, type_ab: str) -> float:
        """Return RSS relative uncertainty in ppm for active contributors of a Type A/B class."""
        values = [
            float(c.value_rel_permil) * 1000.0
            for c in self.contributors
            if c.type_ab == type_ab and c.is_active
        ]
        return float(np.sqrt(sum(value * value for value in values))) if values else 0.0

    def compat_components_abs(self) -> Dict[str, float]:
        """Return the legacy component map, preferring canonical contributors."""
        return {
            _COMPAT_COMPONENT_ALIASES.get(c.name, c.name): float(c.value_abs)
            for c in self.contributors
            if c.type_ab == "B" and c.is_active and c.value_abs > 0
        }

    def compat_components_ppm(self) -> Dict[str, float]:
        """Return the legacy component map in ppm, preferring canonical contributors."""
        return {
            _COMPAT_COMPONENT_ALIASES.get(c.name, c.name): float(c.value_rel_permil) * 1000.0
            for c in self.contributors
            # Keep parity with compat_components_abs: a positive absolute
            # contribution defines presence in the legacy component map.
            if c.type_ab == "B" and c.is_active and c.value_abs > 0
        }


def new_observation_id() -> str:
    """Mint a fresh session-scoped observation identifier."""
    return f"obs-{uuid4().hex}"


@dataclass
class Sample:
    """
    A single measurement sample with all data layers (one row in the HDF5 file).
    """

    name: str
    sample_type: str                       # "SMP", "STD", "BLK"
    run_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Raw intensities keyed by isotope name (e.g. "87Sr")
    intensities: Dict[str, CycleData] = field(default_factory=dict)

    # Current working intensities after the latest correction step
    corrected_intensities: Dict[str, CycleData] = field(default_factory=dict)

    # Snapshot of intensities immediately after blank correction
    blank_corrected_intensities: Dict[str, CycleData] = field(default_factory=dict)

    # Calculated ratios keyed by ratio name (e.g. "87Sr/86Sr")
    ratios: Dict[str, CycleData] = field(default_factory=dict)

    # Current corrected ratios (for Sr this becomes interference-corrected ratios)
    corrected_ratios: Dict[str, CycleData] = field(default_factory=dict)

    # Snapshot of ratios immediately after blank correction (before Sr interference/IIF)
    blank_corrected_ratios: Dict[str, CycleData] = field(default_factory=dict)

    ssb_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    delta_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    uncertainty: Dict[str, UncertaintyBudget] = field(default_factory=dict)

    # Durable, versioned Monte Carlo cross-check results, keyed by ratio name
    # exactly like ``uncertainty``. Each value is an immutable
    # ``domain.uncertainty.mc_result.MCResultRecord``. This is the
    # authoritative home for a completed MC result: the UI session cache is a
    # render convenience, not the owner. Typed as ``Any`` to keep this module
    # free of an import cycle with the uncertainty package.
    mc_results: Dict[str, Any] = field(default_factory=dict)

    # Blanks used for this sample (isotope_name → blank_sample_name)
    used_blanks: Dict[str, str] = field(default_factory=dict)

    iif_corrected_ratios: Dict[str, CycleData] = field(default_factory=dict)
    # Sr-standard calibration preserves the internally normalized input layer.
    sr_standard_corrected_ratios: Dict[str, CycleData] = field(default_factory=dict)

    drift_corrected_ratios: Dict[str, CycleData] = field(default_factory=dict)

    # Hg interference-corrected 204Pb and the 204Pb-bearing ratios formed from
    # it. On ordinary Pb SSB they feed drift and SSB when their Hg record is
    # applied; on Pb-Tl they are diagnostic intermediates saved before Tl
    # normalization and never govern the final layer.
    interference_corrected_intensities: Dict[str, CycleData] = field(default_factory=dict)
    interference_corrected_ratios: Dict[str, CycleData] = field(default_factory=dict)

    # Final Pb ratios after Pb-standard calibration of the Tl-only layer,
    # Y_i = X_i * K, for samples and independent QC whose calibration record is
    # applied. Never written for calibration standards; the Tl-only layer
    # (``iif_corrected_ratios``) is left as it is.
    pb_standard_corrected_ratios: Dict[str, CycleData] = field(default_factory=dict)

    # Per-cycle calibrated delta values in permil, 1000 * (Y_i / C - 1), from the
    # same calibration. Kept apart from the legacy ``delta_results`` so an
    # absolute and a delta result can never share an identity.
    pb_calibrated_delta_cycles: Dict[str, CycleData] = field(default_factory=dict)

    # Frozen correction records keyed by family, then ratio name, e.g.
    # ``{"hg": {"206Pb/204Pb": HgCorrectionRecord}}``. Typed as ``Any`` to keep
    # this module free of an import cycle with the record module.
    correction_records: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Processing warnings (e.g. missing brackets for SSB/delta)
    warnings: List[str] = field(default_factory=list)

    # Blanks used for this sample, by the identity the correction actually
    # selected (role -> observation_id). ``used_blanks`` records the *label* of
    # the same selection; two blanks may share a label, so only this map
    # identifies which observation was subtracted.
    used_blank_ids: Dict[str, str] = field(default_factory=dict)

    # Session-scoped identity for this observation. Opaque and stable: it is
    # minted once, survives ``copy()`` and reprocessing, and is deliberately
    # unrelated to ``name``, ``run_number`` and the measured values, all three
    # of which may legitimately repeat across distinct observations. It is not
    # a content fingerprint and carries no ordering.
    observation_id: str = ""

    def __post_init__(self) -> None:
        if not self.observation_id:
            self.observation_id = new_observation_id()

    @property
    def is_standard(self) -> bool:
        return self.sample_type.upper() in ("STD", "STANDARD")

    @property
    def is_blank(self) -> bool:
        return self.sample_type.upper() in ("BLK", "BLANK")

    @property
    def is_sample(self) -> bool:
        return self.sample_type.upper() in ("SMP", "SAMPLE")

    @property
    def n_cycles(self) -> int:
        """Number of cycles from the first available intensity."""
        for cd in self.intensities.values():
            return cd.n_total
        return 0

    def copy(self) -> "Sample":
        """Independent copy, including nested correction-record evidence."""
        return Sample(
            name=self.name,
            sample_type=self.sample_type,
            run_number=self.run_number,
            metadata=copy.deepcopy(self.metadata),
            intensities={k: v.copy() for k, v in self.intensities.items()},
            corrected_intensities={k: v.copy() for k, v in self.corrected_intensities.items()},
            blank_corrected_intensities={
                k: v.copy() for k, v in self.blank_corrected_intensities.items()
            },
            ratios={k: v.copy() for k, v in self.ratios.items()},
            corrected_ratios={k: v.copy() for k, v in self.corrected_ratios.items()},
            blank_corrected_ratios={k: v.copy() for k, v in self.blank_corrected_ratios.items()},
            ssb_results=copy.deepcopy(self.ssb_results),
            delta_results=copy.deepcopy(self.delta_results),
            uncertainty=copy.deepcopy(self.uncertainty),
            # MCResultRecord recursively normalizes mutable inputs to immutable
            # tuples/scalars, so preserving its identity remains safe.
            mc_results=dict(self.mc_results),
            used_blanks=copy.deepcopy(self.used_blanks),
            used_blank_ids=copy.deepcopy(self.used_blank_ids),
            iif_corrected_ratios={k: v.copy() for k, v in self.iif_corrected_ratios.items()},
            sr_standard_corrected_ratios={k: v.copy() for k, v in self.sr_standard_corrected_ratios.items()},
            drift_corrected_ratios={k: v.copy() for k, v in self.drift_corrected_ratios.items()},
            interference_corrected_intensities={
                k: v.copy() for k, v in self.interference_corrected_intensities.items()
            },
            interference_corrected_ratios={
                k: v.copy() for k, v in self.interference_corrected_ratios.items()
            },
            pb_standard_corrected_ratios={
                k: v.copy() for k, v in self.pb_standard_corrected_ratios.items()
            },
            pb_calibrated_delta_cycles={
                k: v.copy() for k, v in self.pb_calibrated_delta_cycles.items()
            },
            # Frozen records can still contain mutable nested mappings.  Copy
            # recursively so export/runtime copies cannot mutate source evidence.
            correction_records=copy.deepcopy(self.correction_records),
            warnings=list(self.warnings),
            observation_id=self.observation_id,
        )


@dataclass
class ProcessingResult:
    """Output of the processing pipeline."""

    samples: List[Sample]
    element_symbol: str = ""
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def standards(self) -> List[Sample]:
        return [s for s in self.samples if s.is_standard]

    @property
    def blanks(self) -> List[Sample]:
        return [s for s in self.samples if s.is_blank]

    @property
    def sample_measurements(self) -> List[Sample]:
        return [s for s in self.samples if s.is_sample]

    @property
    def count_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in self.samples:
            t = s.sample_type.upper()
            counts[t] = counts.get(t, 0) + 1
        return counts

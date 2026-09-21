"""Application configuration dataclasses."""

from __future__ import annotations

import math
from dataclasses import InitVar, dataclass, field, replace
from typing import List, Optional, Dict, Any

from config.constants import (
    COVERAGE_FACTOR_K,
    DEFAULT_OUTLIER_THRESHOLD_SD,
    DEFAULT_DISPLAY_PRECISION,
)
from config.filtering import (
    FILTER_METHOD_IQR,
    FILTER_METHOD_MAD,
    FILTER_METHOD_NONE,
    FILTER_METHOD_STANDARD_DEVIATION,
    is_removed_filter_method,
    normalize_filter_method_name,
)
from config.contributor_names import (
    LABEL_U_K4,
    canonical_contributor_mapping,
    canonical_contributor_name,
    parse_config_bool,
)


DEFAULT_DRIFT_OUTLIER_THRESHOLD = DEFAULT_OUTLIER_THRESHOLD_SD
DEFAULT_POST_SSB_OUTLIER_THRESHOLD = DEFAULT_OUTLIER_THRESHOLD_SD
DEFAULT_FIXED_COVERAGE_K = COVERAGE_FACTOR_K
# Legacy optional reference term convention. Kept for loading old sessions only;
# the SRM reference-value contributor was removed from the app.
DEFAULT_SRM_REF_UC_VALUE = 4.184e-7
# Intentional zero default: between-digestion reproducibility is user/lab supplied.
DEFAULT_REPROD_DIG_SD = 0.0
# Default SSB kappa-factors (permil).
DEFAULT_K1_SAMPLE_DECOMPOSITION_PERMIL = 0.0
DEFAULT_K2_MATRIX_SEPARATION_PERMIL = 0.2
DEFAULT_K3_PROCEDURAL_BLANK_PERMIL = 0.025
DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_PERMIL = 0.1
DEFAULT_K6_MATRIX_EFFECTS_PERMIL = 0.0
DEFAULT_K7_RESIDUAL_INTERFERENCES_PERMIL = 0.0
DEFAULT_KAPPA_DRIFT_DISTRIBUTION = "normal"
DEFAULT_K1_SAMPLE_DECOMPOSITION_DISTRIBUTION = "normal"
DEFAULT_K2_MATRIX_SEPARATION_DISTRIBUTION = "normal"
DEFAULT_K3_PROCEDURAL_BLANK_DISTRIBUTION = "normal"
DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_DISTRIBUTION = "normal"
DEFAULT_K6_MATRIX_EFFECTS_DISTRIBUTION = "normal"
DEFAULT_K7_RESIDUAL_INTERFERENCES_DISTRIBUTION = "normal"
# Legacy constant names retained only for old callers during migration.
DEFAULT_K2_COLUMN_SEPARATION_PERMIL = DEFAULT_K2_MATRIX_SEPARATION_PERMIL
DEFAULT_K4_HETEROGENEITY_PERMIL = DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_PERMIL
DEFAULT_MC_ITERATIONS_STANDARD = 100_000
DEFAULT_MC_ITERATIONS_HIGH_RIGOR = 500_000

SSB_DELTA_BUILTIN_CONTRIBUTORS = frozenset({
    "u_prec",
    "u_std",
    "u_std_repeatability",
    "u_blank",
    "u_crm",
    "u_k1_sample_decomposition",
    "u_k2_matrix_separation",
    "u_k3_procedural_blank",
    "u_k4_bracketing_standard_heterogeneity",
    "u_k5_instrumental_drift",
    "u_k6_matrix_effects",
    "u_k7_residual_interferences",
})
# SSB/delta defaults are intentionally minimal: only measurement precision
# (u_prec) and the bracketing standards term (u_std) are on by default. Every
# other contributor (std repeatability, blank, the k-factors, CRM, drift) is
# opt-in and must be enabled explicitly by the user.
SSB_DELTA_DEFAULT_ENABLED_CONTRIBUTORS = frozenset({
    "u_prec",
    "u_std",
})


def resolve_filter_parameter(
    method: str,
    threshold: float,
) -> float:
    """Return the active filter parameter for a filter method.

    The method parameter is accepted for compatibility (e.g. if different methods
    had different parameter resolution strategies in the past/future) but is
    currently unused, with the threshold returned directly.
    """
    return threshold


def _format_filter_parameter(method: str, value: float) -> str:
    """Format a filter parameter with method-appropriate units."""
    normalized = normalize_filter_method_name(method)
    if is_removed_filter_method(method):
        normalized = FILTER_METHOD_STANDARD_DEVIATION
    if normalized == FILTER_METHOD_STANDARD_DEVIATION:
        return f"{value:g} standard deviations"
    if normalized == FILTER_METHOD_MAD:
        return f"{value:g} x MAD"
    if normalized == FILTER_METHOD_IQR:
        return f"{value:g} x IQR"
    return f"{value:g}"


@dataclass
class DriftConfig:
    """Drift correction settings."""

    enabled: bool = False
    ratio_name: Optional[str] = None  # Ratio to correct (e.g. "87Sr/86Sr")
    method: str = "polynomial"  # "linear", "polynomial"
    degree: int = 2              # For polynomial (1=linear, 2=quadratic, etc.)
    x_axis: str = "run_number"   # "run_number", "index" (time not available in HDF5)
    apply_outlier_filter: bool = True  # outlier filter on standards before fitting
    outlier_threshold: float = DEFAULT_DRIFT_OUTLIER_THRESHOLD
    outlier_method: str = "sd"  # "sd" (classic global SD, used by the UI) or "mad" (robust, still supported by the domain layer)

    # Normalization reference
    norm_mode: str = "single"    # "average" (mean of all stds) or "single" (selected std)
    norm_standard: int = 0       # 0-based index of the reference standard (single mode)

    fit_info: Dict[str, Any] = field(default_factory=dict)


PB_CALIBRATION_MODE_LOCAL_SSB = "local_ssb"
PB_CALIBRATION_MODE_SESSION_MEAN_K = "session_mean_k"
PB_CALIBRATION_MODES = (PB_CALIBRATION_MODE_LOCAL_SSB, PB_CALIBRATION_MODE_SESSION_MEAN_K)
PB_CALIBRATION_ROLE_STANDARD = "calibration_standard"
PB_CALIBRATION_ROLE_QC = "independent_qc"
PB_CALIBRATION_ROLE_NOT_USED = "not_used"
PB_CALIBRATION_ROLES = (
    PB_CALIBRATION_ROLE_STANDARD, PB_CALIBRATION_ROLE_QC, PB_CALIBRATION_ROLE_NOT_USED,
)
PB_DELTA_PRECISION_STATISTICS = ("none", "sd", "se")


def _validate_scalar_fields(config):
    """Reject malformed scalar dataclass inputs before truth tests or arithmetic."""
    from dataclasses import fields
    for item in fields(config):
        value = getattr(config, item.name)
        if isinstance(item.default, bool) and not isinstance(value, bool):
            raise ValueError(f"{item.name} must be a boolean.")
        if (isinstance(item.default, (int, float)) and not isinstance(item.default, bool)) or (value is not None and str(item.type) in {"Optional[float]", "Optional[int]"}):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{item.name} must be a finite number.")
            if (isinstance(item.default, int) or str(item.type) == "Optional[int]") and int(value) != value:
                raise ValueError(f"{item.name} must be an integer.")


@dataclass
class PbStandardCalibrationConfig:
    """Pb-standard calibration applied after Pb-Tl external normalization.

    Off by default, in which case the Tl-only route is unchanged. Eligibility is
    an explicit role plus a matching material on a non-blank observation, keyed
    by ``observation_id``; the sample type, the name and any generic standard flag
    are never used to pick calibration standards. ``assignment_locators`` only
    help re-find an observation after a raw re-import and never assign a role.

    The minimum valid cycles per standard are the C01 Q-02 default adopted at C04
    entry (alternating 1, block 2, session 2) and stay owner-overridable here.
    """

    enabled: bool = False
    mode: str = PB_CALIBRATION_MODE_LOCAL_SSB
    #: Name of the accepted Pb reference material. It must be chosen explicitly;
    #: ``None`` leaves every calibrated ratio unavailable.
    reference_material: Optional[str] = None
    role_assignments: Dict[str, str] = field(default_factory=dict)
    material_assignments: Dict[str, str] = field(default_factory=dict)
    assignment_locators: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    enable_delta: bool = False
    #: Cycle scatter (``sd``) or precision of the mean (``se``) reported beside a
    #: calibrated delta. Neither is a combined measurement uncertainty.
    delta_precision_statistic: str = "none"
    min_valid_cycles_alternating: int = 1
    min_valid_cycles_block: int = 2
    min_valid_cycles_session: int = 2

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_scalar_fields(self)
        if self.mode not in PB_CALIBRATION_MODES:
            raise ValueError(
                f"Unsupported Pb-standard calibration mode {self.mode!r}; "
                f"expected one of {list(PB_CALIBRATION_MODES)}."
            )
        if self.delta_precision_statistic not in PB_DELTA_PRECISION_STATISTICS:
            raise ValueError(
                f"Unsupported calibrated-delta precision statistic "
                f"{self.delta_precision_statistic!r}; expected one of "
                f"{list(PB_DELTA_PRECISION_STATISTICS)}."
            )
        for name in ("min_valid_cycles_alternating", "min_valid_cycles_block", "min_valid_cycles_session"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be an integer of at least 1.")
            setattr(self, name, int(value))
        self.role_assignments = {str(k): str(v) for k, v in dict(self.role_assignments or {}).items()}
        unknown = sorted({v for v in self.role_assignments.values() if v not in PB_CALIBRATION_ROLES})
        if unknown:
            raise ValueError(f"Unknown Pb-standard calibration roles: {unknown}")
        self.material_assignments = {
            str(k): str(v) for k, v in dict(self.material_assignments or {}).items()
        }
        self.assignment_locators = {
            str(k): dict(v) for k, v in dict(self.assignment_locators or {}).items()
        }
        if self.reference_material is not None:
            self.reference_material = str(self.reference_material).strip() or None


@dataclass
class ProcessingConfig:
    """Settings controlling the data processing pipeline."""

    # Blank correction
    blank_mode: str = "before"  # "none", "before", "before_and_after"

    # Statistical filtering
    filter_method: str = FILTER_METHOD_STANDARD_DEVIATION
    filter_threshold: float = DEFAULT_OUTLIER_THRESHOLD_SD

    # SSB / Delta (Li, B, Mg)
    enable_ssb: bool = False
    ssb_mode: str = "alternating"  # "alternating" or "block_average"
    enable_delta: bool = False

    # Post-SSB outlier detection
    enable_post_ssb_outliers: bool = False
    post_ssb_outlier_threshold: float = DEFAULT_POST_SSB_OUTLIER_THRESHOLD

    # Sr-specific
    apply_interference_correction: bool = True
    apply_mass_bias_correction: bool = True
    # Legacy ignored field retained so old saved sessions/config constructors load.
    subtract_kr_blank: bool = False
    interference_monitors_enabled: Dict[str, bool] = field(default_factory=dict)
    sr_session_anchoring: bool = False
    sr_calibration_standard_ids: List[str] = field(default_factory=list)

    # Pb-specific
    apply_hg_interference_correction: bool = False

    # Reference Material
    reference_material: Optional[str] = None  # CRM name, None = use element default
    normalization_ratio_override: Optional[str] = None  # Session-only internal norm ratio override
    normalization_value_override: Optional[float] = None  # Session-only internal norm override

    # Uncertainty
    include_certified_uncertainty: bool = True

    # Data preference (for raw_corrected HDF5 files)
    data_preference: str = "auto"  # "auto", "raw", "corrected"

    # Global cycle settings
    global_cycle_range: bool = False

    # Drift correction
    drift: DriftConfig = field(default_factory=DriftConfig)

    # Pb-standard calibration after Pb-Tl external normalization (off by default)
    pb_standard_calibration: PbStandardCalibrationConfig = field(
        default_factory=PbStandardCalibrationConfig
    )

    def __post_init__(self) -> None:
        """Normalize stored filter names after legacy configuration loads."""
        if isinstance(self.pb_standard_calibration, dict):
            self.pb_standard_calibration = PbStandardCalibrationConfig(**self.pb_standard_calibration)
        if isinstance(self.drift, dict):
            self.drift = DriftConfig(**self.drift)
        if self.filter_method == "Sigma":
            self.filter_method = FILTER_METHOD_STANDARD_DEVIATION
        elif is_removed_filter_method(self.filter_method):
            self.filter_method = FILTER_METHOD_STANDARD_DEVIATION
        else:
            normalized = normalize_filter_method_name(self.filter_method)
            if normalized in {
                FILTER_METHOD_NONE,
                FILTER_METHOD_STANDARD_DEVIATION,
                FILTER_METHOD_MAD,
                FILTER_METHOD_IQR,
            }:
                self.filter_method = normalized
        self.validate()

    def validate(self) -> None:
        """Reject invalid processing settings before numerical work begins."""
        _validate_scalar_fields(self)
        for name, choices in {"blank_mode": {"none", "before", "before_and_after"}, "ssb_mode": {"alternating", "block_average"}, "data_preference": {"auto", "raw", "corrected"}}.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {sorted(choices)}.")
        if self.normalization_value_override is not None and (isinstance(self.normalization_value_override, bool) or not isinstance(self.normalization_value_override, (int, float)) or not math.isfinite(self.normalization_value_override) or self.normalization_value_override <= 0):
            raise ValueError("normalization_value_override must be finite and positive.")
        if not isinstance(self.interference_monitors_enabled, dict) or any(not isinstance(v, bool) for v in self.interference_monitors_enabled.values()):
            raise ValueError("interference_monitors_enabled must map channels to booleans.")
        valid_methods = {
            FILTER_METHOD_NONE,
            FILTER_METHOD_STANDARD_DEVIATION,
            FILTER_METHOD_MAD,
            FILTER_METHOD_IQR,
        }
        if self.filter_method not in valid_methods:
            raise ValueError(
                f"Unsupported cycle filter method {self.filter_method!r}; "
                f"expected one of {sorted(valid_methods)}."
            )
        for field_name in ("filter_threshold", "post_ssb_outlier_threshold"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be a finite value greater than zero.")
            setattr(self, field_name, value)
        if not isinstance(self.drift, DriftConfig):
            raise ValueError("drift must be a DriftConfig.")
        _validate_scalar_fields(self.drift)
        if self.drift.method == "spline":
            self.drift.method = "polynomial"  # explicit legacy UI migration
        for name, choices in {"method": {"linear", "polynomial"}, "x_axis": {"run_number", "index", "time_minutes"}, "outlier_method": {"sd", "mad"}, "norm_mode": {"single", "average"}}.items():
            if getattr(self.drift, name) not in choices:
                raise ValueError(f"drift.{name} must be one of {sorted(choices)}.")
        if self.drift.degree < 1 or self.drift.norm_standard < 0 or self.drift.outlier_threshold <= 0:
            raise ValueError("Invalid drift degree, standard index or outlier threshold.")
        if not isinstance(self.pb_standard_calibration, PbStandardCalibrationConfig):
            raise ValueError("pb_standard_calibration must be a PbStandardCalibrationConfig.")
        self.pb_standard_calibration.validate()

    def get_active_filter_threshold(
        self,
        method: Optional[str] = None,
        *,
        fallback_threshold: Optional[float] = None,
    ) -> float:
        """Return the active threshold/parameter for a filter method."""
        active_method = method or self.filter_method
        base_threshold = self.filter_threshold if fallback_threshold is None else fallback_threshold
        return resolve_filter_parameter(active_method, base_threshold)

    def format_filter_parameter(self, method: Optional[str] = None) -> str:
        """Return a display string for the active filter parameter."""
        active_method = method or self.filter_method
        active_value = self.get_active_filter_threshold(active_method)
        return _format_filter_parameter(active_method, active_value)

    def is_monitor_configured_enabled(self, interfering_isotope: str) -> bool:
        """Return the stored per-monitor preference, ignoring the master gate."""
        if not self.interference_monitors_enabled:
            return True
        return bool(self.interference_monitors_enabled.get(interfering_isotope, True))

    def is_monitor_enabled(self, interfering_isotope: str) -> bool:
        """Return whether an interferent correction is effectively enabled."""
        if not self.apply_interference_correction:
            return False
        return self.is_monitor_configured_enabled(interfering_isotope)


@dataclass
class DisplayConfig:
    """Settings controlling what is shown in the UI."""

    show_raw: bool = True
    show_corrected: bool = True
    show_drift_corrected: bool = True
    show_interference_corrected: bool = True
    show_iif_corrected: bool = True
    show_pb_standard_corrected: bool = True
    show_sr_standard_corrected: bool = True
    show_filtered_raw: bool = False
    show_filtered_corrected: bool = False
    show_threshold_lines: bool = True
    threshold_line_basis: str = "processing_mask"  # "processing_mask" or "iif_like"
    show_outliers: bool = True
    show_ratio_stats_box: bool = True
    ratio_stats_statistic: str = "2SD"  # "2SD" or "2SE"
    show_blank_info: bool = False
    precision: int = DEFAULT_DISPLAY_PRECISION


@dataclass
class ExportConfig:
    """Settings for the Export tab."""

    include_summary: bool = True
    include_individual_stats: bool = True
    include_corrected_ratios: bool = True
    include_uncertainty: bool = True
    include_raw_cycles: bool = False
    include_processing_log: bool = True
    include_plots: bool = True
    format: str = "excel"  # "excel", "json", "csv"


# Uncertainty framework configuration

DEFAULT_U_PREC_MODE = "se"
SUPPORTED_U_PREC_MODES = ("sd", "se")


def _canonical_u_prec_mode(value: object) -> str:
    """Canonicalize a stored precision-mode token, rejecting unknown ones.

    Surrounding whitespace and letter case are incidental to a serialized
    token, so ``"SD "`` still means SD. An unrecognized token is a different
    matter: the requested model cannot be established, and silently choosing
    SE would divide the reported precision uncertainty by sqrt(n). A missing
    or empty value is treated as absent and takes the default.
    """
    if value is None:
        return DEFAULT_U_PREC_MODE
    if not isinstance(value, str):
        raise ValueError(
            f"u_prec_mode must be one of {SUPPORTED_U_PREC_MODES}, got {value!r}."
        )
    token = value.strip().lower()
    if not token:
        return DEFAULT_U_PREC_MODE
    if token not in SUPPORTED_U_PREC_MODES:
        raise ValueError(
            f"u_prec_mode must be one of {SUPPORTED_U_PREC_MODES}, got {value!r}."
        )
    return token


@dataclass(frozen=True)
class CustomUncertaintyContributor:
    """Element-scoped lab-defined scalar uncertainty contributor.

    u_rel_permil is a standard relative uncertainty in permil. If the user
    enters an expanded value in the UI, convert it before storing this model.
    The declared normal/rectangular PDF is used by MC. degrees_of_freedom
    describes estimation uncertainty for GUM, not a Student-t shape parameter.
    """

    name: str
    display_name: str
    element_symbol: str
    u_rel_permil: float
    type_ab: str = "B"
    degrees_of_freedom: float = float("inf")
    distribution: str = "normal"
    description: str = ""
    reference: str = ""
    enabled: bool = True


@dataclass
class KappaFactor:
    """A single preparation uncertainty contributor."""

    name: str = ""
    display_name: str = ""
    u_rel_permil: float = 0.0
    distribution: str = "rectangular"  # "normal" or "rectangular"
    description: str = ""
    reference: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        """Normalize and validate the distribution string."""
        from config.validation import normalize_distribution
        self.distribution = normalize_distribution(
            self.distribution,
            field_name=f"KappaFactor {self.name or self.display_name!r}"
        )


@dataclass
class KappaAssignment:
    """Which kappa-factors apply to which samples."""

    sample_assignments: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class UncertaintyConfig:
    """All uncertainty-related configuration."""

    # Engine selection (auto-detected from element, but overridable)
    engine: str = "ssb_delta"  # "internal_normalization", "pb_tl_external_normalization", or "ssb_delta"

    # Blank
    blank_correlation_method: str = "pearson_from_data"  # "fixed_value", "uncorrelated"
    blank_fixed_r: float = 0.0
    blank_uncertainty_input: str = "sd"  # "sd" cycle scatter, "se" standard error of blank mean

    # IIF / Mass bias
    iif_method: str = "sd_of_standards"  # or "norm_constant"

    sr_iif_mode: str = "A"                    # Legacy: "B" enabled MC-only normalization propagation
    sr_norm_ratio_u_abs: float = 0.0          # absolute standard u of the selected normalization ratio
    # Legacy per-mil field. ``None`` means "no legacy value was supplied";
    # only a value actually present in a restored session is converted, so a
    # newly built config cannot inherit a magnitude the user never entered.
    sr_norm_ratio_u_permil: Optional[float] = None
    pb_tl_norm_ratio_u_abs: float = 0.0       # absolute standard u of the selected Tl normalization ratio
    pb_tl_norm_ratio_u_justification: str = ""
    pb_tl_norm_ratio_u_source: str = ""
    sr_blank_3var: bool = True                # include m/z 88 in Sr blank covariance
    sr_qc_bias_abs: float = 0.0               # optional user-supplied QC uncertainty numerator in absolute ratio units
    sr_qc_cert_value: float = 0.0             # optional certified ratio of the QC material (Δ_QC denominator)
    sr_reference_bias_ref_value: Optional[float] = None  # optional R_ref for Sr Δ_ref diagnostics/budget

    # Repeatability mode
    u_prec_mode: str = "se"                      # "se" or "sd"
    std_repeatability_mode: str = "sd"            # "sd" or "se"
    srm_ref_uc_value: float = DEFAULT_SRM_REF_UC_VALUE  # Legacy; no active contributor
    u_reprod_dig_sd: float = DEFAULT_REPROD_DIG_SD
    u_reprod_dig_ref_value: float = 0.0       # optional reference ratio for fractional digestion SD transfer

    # Preparation kappa-factors
    kappa_factors: List[KappaFactor] = field(default_factory=list)
    kappa_assignments: KappaAssignment = field(default_factory=KappaAssignment)

    # Output mode
    output_mode: str = "delta"  # or "absolute_ratio"

    # Coverage factor
    coverage_method: str = "fixed_k"  # or "welch_satterthwaite"
    coverage_k: float = DEFAULT_FIXED_COVERAGE_K
    summary_flag_elevated_multiple: float = 2.0
    summary_flag_high_multiple: float = 3.0

    # Monte Carlo
    mc_validation_mode: str = "standard"  # "standard", "high_rigor", "custom"
    mc_iterations: int = DEFAULT_MC_ITERATIONS_STANDARD

    # SSB/Delta state (mirrors ProcessingConfig; set by UI sync)
    enable_ssb: bool = False
    enable_delta: bool = False
    ssb_mode: str = "alternating"  # "alternating" or "block_average"

    # Standard repeatability (u_std_repeatability) strategy
    reprod_method: str = "auto"  # "auto", "sd_of_means", "loo_cross_validation", "drift_residuals", "robust_mad"
    segment_assignments: Dict[str, int] = field(default_factory=dict)  # std name -> segment (1-indexed)
    excluded_standards: List[str] = field(default_factory=list)
    include_kappa_drift: bool = False
    kappa_drift_distribution: str = DEFAULT_KAPPA_DRIFT_DISTRIBUTION
    k1_sample_decomposition_permil: float = DEFAULT_K1_SAMPLE_DECOMPOSITION_PERMIL
    k2_matrix_separation_permil: float = DEFAULT_K2_MATRIX_SEPARATION_PERMIL
    k3_procedural_blank_permil: float = DEFAULT_K3_PROCEDURAL_BLANK_PERMIL
    k4_bracketing_standard_heterogeneity_permil: float = (
        DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_PERMIL
    )
    k6_matrix_effects_permil: float = DEFAULT_K6_MATRIX_EFFECTS_PERMIL
    k7_residual_interferences_permil: float = DEFAULT_K7_RESIDUAL_INTERFERENCES_PERMIL
    k1_sample_decomposition_distribution: str = DEFAULT_K1_SAMPLE_DECOMPOSITION_DISTRIBUTION
    k2_matrix_separation_distribution: str = DEFAULT_K2_MATRIX_SEPARATION_DISTRIBUTION
    k3_procedural_blank_distribution: str = DEFAULT_K3_PROCEDURAL_BLANK_DISTRIBUTION
    k4_bracketing_standard_heterogeneity_distribution: str = (
        DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_DISTRIBUTION
    )
    k6_matrix_effects_distribution: str = DEFAULT_K6_MATRIX_EFFECTS_DISTRIBUTION
    k7_residual_interferences_distribution: str = DEFAULT_K7_RESIDUAL_INTERFERENCES_DISTRIBUTION
    k2_column_separation_permil: InitVar[Optional[float]] = None
    k4_heterogeneity_permil: InitVar[Optional[float]] = None

    # Runtime budget contributor switches (default = included)
    contributor_enabled: Dict[str, bool] = field(default_factory=dict)
    # UI control toggles that influence contributor selection but are not
    # contributors themselves.
    control_enabled: Dict[str, bool] = field(default_factory=dict)

    # Master enable
    enabled: bool = True

    def __post_init__(
        self,
        k2_column_separation_permil: Optional[float],
        k4_heterogeneity_permil: Optional[float],
    ) -> None:
        """Migrate legacy SSB kappa constructor names to canonical fields.

        Defensive `getattr(self, ...)` usage:
        When loading configurations from pickled states or older session files,
        attributes added in newer versions of the code (e.g., `u_prec_mode` or
        distribution names) might be missing from the deserialized instance dictionary.
        Using `getattr` prevents `AttributeError` and provides robust defaults.
        """
        if k2_column_separation_permil is not None:
            self.k2_matrix_separation_permil = float(k2_column_separation_permil)
        if k4_heterogeneity_permil is not None:
            self.k4_bracketing_standard_heterogeneity_permil = float(
                k4_heterogeneity_permil
            )
        self.u_prec_mode = _canonical_u_prec_mode(
            getattr(self, "u_prec_mode", DEFAULT_U_PREC_MODE)
        )
        for attr, default in (
            ("kappa_drift_distribution", DEFAULT_KAPPA_DRIFT_DISTRIBUTION),
            ("k1_sample_decomposition_distribution", DEFAULT_K1_SAMPLE_DECOMPOSITION_DISTRIBUTION),
            ("k2_matrix_separation_distribution", DEFAULT_K2_MATRIX_SEPARATION_DISTRIBUTION),
            ("k3_procedural_blank_distribution", DEFAULT_K3_PROCEDURAL_BLANK_DISTRIBUTION),
            (
                "k4_bracketing_standard_heterogeneity_distribution",
                DEFAULT_K4_BRACKETING_STANDARD_HETEROGENEITY_DISTRIBUTION,
            ),
            ("k6_matrix_effects_distribution", DEFAULT_K6_MATRIX_EFFECTS_DISTRIBUTION),
            ("k7_residual_interferences_distribution", DEFAULT_K7_RESIDUAL_INTERFERENCES_DISTRIBUTION),
        ):
            value = getattr(self, attr, default)
            if not isinstance(value, str):
                raise ValueError(f"{attr} distribution must be normal or rectangular.")
            value = value.strip().lower()
            if value == "gaussian":
                value = "normal"
            if value not in {"normal", "rectangular"}:
                raise ValueError(f"{attr} distribution must be normal or rectangular.")
            setattr(self, attr, value)
        self.validate()

    def validate(self) -> None:
        """Reject uncertainty settings that can produce invalid reported budgets."""
        _validate_scalar_fields(self)
        if self.coverage_method == "fixed":
            self.coverage_method = "fixed_k"  # explicit legacy enum migration
        # Zero retains its documented unset/disabled meaning. Negative physical
        # magnitudes must not silently select a consumer's fallback value.
        for name in (
            "sr_norm_ratio_u_abs", "sr_norm_ratio_u_permil", "pb_tl_norm_ratio_u_abs",
            "sr_qc_bias_abs", "sr_qc_cert_value", "u_reprod_dig_sd",
            "u_reprod_dig_ref_value", "k1_sample_decomposition_permil",
            "k2_matrix_separation_permil", "k3_procedural_blank_permil",
            "k4_bracketing_standard_heterogeneity_permil", "k6_matrix_effects_permil",
            "k7_residual_interferences_permil",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative.")
        for name in (
            "kappa_drift_distribution", "k1_sample_decomposition_distribution",
            "k2_matrix_separation_distribution", "k3_procedural_blank_distribution",
            "k4_bracketing_standard_heterogeneity_distribution",
            "k6_matrix_effects_distribution", "k7_residual_interferences_distribution",
        ):
            if getattr(self, name) not in {"normal", "rectangular"}:
                raise ValueError(f"{name} distribution must be normal or rectangular.")
        for name, choices in {"blank_correlation_method": {"uncorrelated", "fixed_value", "pearson_from_data"}, "blank_uncertainty_input": {"sd", "se"}, "output_mode": {"delta", "absolute_ratio"}, "coverage_method": {"welch_satterthwaite", "fixed_k"}, "engine": {"ssb_delta", "internal_normalization", "pb_tl_external_normalization"}, "iif_method": {"sd_of_standards", "norm_constant"}, "std_repeatability_mode": {"sd", "se"}, "ssb_mode": {"alternating", "block_average"}, "mc_validation_mode": {"standard", "high_rigor", "custom"}, "reprod_method": {"auto", "sd_of_means", "loo_cross_validation", "drift_residuals", "robust_mad"}}.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {sorted(choices)}.")
        if not -1 <= self.blank_fixed_r <= 1:
            raise ValueError("blank_fixed_r must lie in [-1, 1]; assembled covariance must also be PSD.")
        for field_name in (
            "coverage_k",
            "summary_flag_elevated_multiple",
            "summary_flag_high_multiple",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be a finite value greater than zero.")
            setattr(self, field_name, value)
        if self.summary_flag_high_multiple < self.summary_flag_elevated_multiple:
            raise ValueError(
                "summary_flag_high_multiple must be greater than or equal to "
                "summary_flag_elevated_multiple."
            )
        self.u_prec_mode = _canonical_u_prec_mode(
            getattr(self, "u_prec_mode", DEFAULT_U_PREC_MODE)
        )
        # The control namespace goes through the same strict parser as
        # contributor_enabled, so a serialized "false" disables the control
        # instead of being read as a truthy string.
        raw_controls = getattr(self, "control_enabled", None)
        if raw_controls:
            self.control_enabled = {
                str(name): parse_config_bool(
                    value, field_name=f"Control {str(name)!r} enabled"
                )
                for name, value in raw_controls.items()
            }

    def is_contributor_enabled(
        self,
        name: str,
        *,
        element_symbol: str = "",
        processing_config=None,
    ) -> bool:
        """Return whether the contributor is included in the budget."""
        name = canonical_contributor_name(name)
        raw_explicit = getattr(self, "contributor_enabled", {})
        explicit = canonical_contributor_mapping(
            raw_explicit
        )
        resolved_engine = (
            self.resolve_engine(element_symbol, processing_config=processing_config)
            if element_symbol
            else self.engine
        )

        # Reference bias is intentionally unavailable in the V1 Sr workflow.
        # Keep legacy saved configurations from silently re-enabling it after
        # the control was removed from the UI.
        if resolved_engine == "internal_normalization" and name == "u_bias_ref":
            return False

        # Engine A exposes one UI control ("enable_srm_repeatability") and one
        # mode selector ("std_repeatability_mode") for SD vs SE. Exactly one of
        # u_std_repeatability (SD) / u_std_repeatability_se (SE) is active when
        # the master control is enabled.
        if resolved_engine == "internal_normalization" and name in {
            "u_std_repeatability",
            "u_std_repeatability_se",
        }:
            mode = (self.std_repeatability_mode or "sd").lower()
            if mode not in {"sd", "se"}:
                mode = "sd"

            master_enabled = self.is_control_enabled("enable_srm_repeatability")
            if master_enabled is None:
                # Backward-compatible fallback for older configs that stored
                # SD/SE contributor toggles directly.
                legacy_sd = explicit.get("u_std_repeatability", None)
                legacy_se = explicit.get("u_std_repeatability_se", None)
                if legacy_sd is None and legacy_se is None:
                    master_enabled = True
                else:
                    master_enabled = bool((legacy_sd or False) or (legacy_se or False))

            if not bool(master_enabled):
                return False
            return (name == "u_std_repeatability" and mode == "sd") or (
                name == "u_std_repeatability_se" and mode == "se"
            )

        if name in explicit:
            return explicit[name]

        if (
            resolved_engine == "ssb_delta"
            and name == "u_k5_instrumental_drift"
            and "u_kappa_drift" in raw_explicit
        ):
            return parse_config_bool(
                raw_explicit["u_kappa_drift"],
                field_name="Contributor 'u_kappa_drift' enabled",
            )

        if is_russell_law_normalization_engine(resolved_engine) and name in {
            "u_kappa_drift",           # Engine A generic drift term
            "u_k5_instrumental_drift", # Engine B k5 drift term
            "u_bias_qc",               # optional user-supplied QC bias
            "u_crm",                   # conditional on output mode (engine handles)
            "u_ref_value",             # literature reference value; independent opt-in
            "u_reprod_dig",            # between-digestion SD — off by default
            "u_norm_ratio",            # accepted internal-normalization ratio u
        }:
            return False

        # SSB/Delta engine defaults follow the active correction layer. With no
        # SSB/delta correction selected, only measurement precision is active.
        # When either SSB or delta is selected, only measurement precision
        # (u_prec) and the bracketing standards term (u_std) are on by default;
        # every other contributor is opt-in (see
        # SSB_DELTA_DEFAULT_ENABLED_CONTRIBUTORS).
        if resolved_engine == "ssb_delta":
            if not (self.enable_ssb or self.enable_delta):
                return name == "u_prec"
            if name in SSB_DELTA_BUILTIN_CONTRIBUTORS:
                return name in SSB_DELTA_DEFAULT_ENABLED_CONTRIBUTORS

        return True

    def is_control_enabled(self, name: str) -> Optional[bool]:
        """Return whether UI control is enabled.

        This follows a tri-state design pattern:
        1. If explicitly configured in the control_enabled dict, return that bool value.
        2. Else if legacy contributor settings exist, return their equivalent state for
           backward compatibility.
        3. Otherwise, return None to indicate the control is unconfigured, allowing
           callers to resolve default states dynamically.

        Defensive `getattr` is used here to safely fetch fields that might be absent
        on legacy pickled configuration instances.
        """
        explicit = getattr(self, "control_enabled", {})
        if name in explicit:
            # Parse on read as well as at construction: a config restored by
            # unpickling bypasses __post_init__ and can still hold a
            # serialized token here.
            return parse_config_bool(
                explicit[name], field_name=f"Control {name!r} enabled"
            )

        legacy_contributors = canonical_contributor_mapping(
            getattr(self, "contributor_enabled", {})
        )

        if name == "enable_srm_repeatability":
            legacy = legacy_contributors.get("u_std_repeatability_enabled", None)
            if legacy is not None:
                return bool(legacy)

            legacy_sd = legacy_contributors.get("u_std_repeatability", None)
            legacy_se = legacy_contributors.get("u_std_repeatability_se", None)
            if legacy_sd is None and legacy_se is None:
                return True
            return bool((legacy_sd or False) or (legacy_se or False))

        if name == "enable_sr_norm_ratio_uncertainty":
            legacy = legacy_contributors.get("_sr_norm_ratio_u", None)
            if legacy is not None:
                return bool(legacy)
            explicit_norm = legacy_contributors.get("u_norm_ratio", None)
            if explicit_norm is not None:
                return bool(explicit_norm)
            return getattr(self, "sr_iif_mode", "A") == "B"

        return None

    def disabled_contributor_names(self) -> set[str]:
        """Return contributors explicitly disabled by the user."""
        legacy_control_names = {
            "u_std_repeatability_enabled",
            "_sr_norm_ratio_u",
        }
        explicit = canonical_contributor_mapping(
            {
                name: enabled
                for name, enabled in getattr(self, "contributor_enabled", {}).items()
                if name not in legacy_control_names
            }
        )
        return {
            name
            for name, enabled in explicit.items()
            if not enabled
        }

    def resolve_reprod_method(self, *, enable_ssb: bool = False, enable_delta: bool = False) -> str:
        """Resolve 'auto' reprod_method based on SSB/Delta correction state."""
        if self.reprod_method != "auto":
            return self.reprod_method
        if enable_ssb:
            return "loo_cross_validation"
        return "sd_of_means"

    def resolve_mc_iterations(
        self,
        *,
        override_mode: Optional[str] = None,
        override_custom_iterations: Optional[int] = None,
    ) -> int:
        """Return the effective Monte Carlo iteration count for the active mode."""
        mode = str(override_mode or getattr(self, "mc_validation_mode", "standard")).lower()
        if mode == "high_rigor":
            return DEFAULT_MC_ITERATIONS_HIGH_RIGOR
        if mode == "custom":
            custom_value = (
                self.mc_iterations
                if override_custom_iterations is None
                else override_custom_iterations
            )
            return max(1_000, int(custom_value))
        return DEFAULT_MC_ITERATIONS_STANDARD

    def resolve_engine(self, element_symbol: str, *, processing_config=None) -> str:
        """Return the uncertainty engine to use for the element.

        For Pb, the engine is selected based on whether mass-bias correction is
        active: Tl-normalised data -> pb_tl_external_normalization; SSB data -> ssb_delta.
        Pass *processing_config* (a :class:`ProcessingConfig`) to enable this
        dispatch; if omitted, ``self.engine`` is returned for Pb as well.
        """
        _SR_ENGINE = "internal_normalization"
        if element_symbol == "Sr" and self.engine == "ssb_delta":
            return _SR_ENGINE
        if element_symbol == "Pb" and processing_config is not None:
            if getattr(processing_config, "apply_mass_bias_correction", False):
                return "pb_tl_external_normalization"
            return "ssb_delta"
        return self.engine


_RUSSELL_LAW_NORMALIZATION_ENGINES = frozenset({
    "internal_normalization",
    "pb_tl_external_normalization",
})


def is_russell_law_normalization_engine(engine: str) -> bool:
    """True for engines that compute ratios through Russell-law normalization."""
    return engine in _RUSSELL_LAW_NORMALIZATION_ENGINES


def _clear_ssb_delta_contributor_overrides(
    contributor_enabled: Dict[str, bool],
) -> Dict[str, bool]:
    """Drop stored Engine B contributor states so mode defaults can re-apply."""
    canonical = canonical_contributor_mapping(contributor_enabled)
    return {
        name: enabled
        for name, enabled in canonical.items()
        if name not in SSB_DELTA_BUILTIN_CONTRIBUTORS
    }


# Lab-default kappa presets

_LAB_KAPPA_RAW = {
    "B": [
        dict(name="dry_ashing", display_name="Sample decomposition / dry ashing (\u03ba\u2081)",
             u_rel_permil=0.41, distribution="rectangular",
             description="B loss during ashing at \u2264600\u00b0C",
             reference="published uncertainty budget"),
        dict(name="matrix_separation", display_name="B-matrix separation (\u03ba\u2082)",
             u_rel_permil=0.25, distribution="normal",
             description="Mean \u0394\u00b9\u00b9B from B-matrix separation validation",
             reference="published uncertainty budget"),
        dict(name="procedure_blank", display_name="Procedure blank (\u03ba\u2083)",
             u_rel_permil=0.11, distribution="rectangular",
             description="Maximum blank shift at 4.4 ng B",
             reference="published uncertainty budget"),
        dict(name="crm_inhomogeneity", display_name=f"{LABEL_U_K4} (\u03ba\u2084)",
             u_rel_permil=0.20, distribution="rectangular",
             description="Potential heterogeneity of NIST SRM 951 working solutions",
             reference="published uncertainty budget"),
        dict(name="mass_disc_drift", display_name="Mass disc. drift (\u03ba\u2085)",
             u_rel_permil=0.20, distribution="normal",
             description="Instrumental mass discrimination fluctuation/drift",
             reference="published uncertainty budget"),
        dict(name="matrix_effects", display_name="Matrix effects (\u03ba\u2086)",
             u_rel_permil=0.20, distribution="normal",
             description="Plasma matrix load effects on mass discrimination",
             reference="published uncertainty budget"),
    ],
    "Sr": [
        dict(name="column_fractionation", display_name="Column fractionation (\u03ba\u2081)",
             u_rel_permil=0.15, distribution="rectangular",
             description="Potential Sr fractionation during ion exchange separation",
             reference="Lab-specific; typ. range 0.05\u20130.30 \u2030"),
        dict(name="digestion_loss", display_name="Digestion loss (\u03ba\u2082)",
             u_rel_permil=0.20, distribution="rectangular",
             description="Sr loss/fractionation during acid digestion",
             reference="Lab-specific estimate"),
    ],
    "Li": [
        dict(name="column_fractionation", display_name="Column fractionation (\u03ba\u2081)",
             u_rel_permil=0.30, distribution="rectangular",
             description="Li fractionation during cation exchange chromatography",
             reference="lab-specific estimate"),
    ],
    "Mg": [
        dict(name="column_fractionation", display_name="Column fractionation (\u03ba\u2081)",
             u_rel_permil=0.20, distribution="rectangular",
             description="Mg fractionation during ion exchange separation",
             reference="Lab-specific estimate"),
    ],
    "Cd": [],
    "Pb": [],
}


def get_kappa_defaults_for_element(element_symbol: str) -> List[KappaFactor]:
    """Return lab-default kappa presets for *element_symbol*."""
    raw_list = _LAB_KAPPA_RAW.get(element_symbol, [])
    return [KappaFactor(**d) for d in raw_list]


KAPPA_SUPPORTED_ELEMENTS = list(_LAB_KAPPA_RAW.keys())


def sync_uncertainty_with_processing(
    u_config: "UncertaintyConfig",
    processing_config: "ProcessingConfig",
    *,
    preserve_contributor_selections: bool = False,
) -> "UncertaintyConfig":
    """Return *u_config* with SSB/delta state and output_mode synced from *processing_config*.

    Called by runtime.compute_runtime_budget, the Uncertainty tab orchestrator,
    and config_controls so all three paths stay in sync without duplicating logic.
    Returns *u_config* unchanged when nothing needs syncing.
    """
    needs_sync = (
        u_config.ssb_mode != processing_config.ssb_mode
        or u_config.enable_ssb != processing_config.enable_ssb
        or u_config.enable_delta != processing_config.enable_delta
    )
    output_mode_invalid = (
        u_config.output_mode == "delta" and not processing_config.enable_delta
    ) or u_config.output_mode not in {"delta", "absolute_ratio"}

    if not (needs_sync or output_mode_invalid):
        return u_config

    contributor_enabled = getattr(u_config, "contributor_enabled", {})
    if (
        not preserve_contributor_selections
        and u_config.engine == "ssb_delta"
        and (
            u_config.enable_ssb != processing_config.enable_ssb
            or u_config.enable_delta != processing_config.enable_delta
        )
    ):
        contributor_enabled = _clear_ssb_delta_contributor_overrides(contributor_enabled)

    return replace(
        u_config,
        ssb_mode=processing_config.ssb_mode,
        enable_ssb=processing_config.enable_ssb,
        enable_delta=processing_config.enable_delta,
        contributor_enabled=contributor_enabled,
        output_mode=(
            "absolute_ratio" if output_mode_invalid else u_config.output_mode
        ),
    )

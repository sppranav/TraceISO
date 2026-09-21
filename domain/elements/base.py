"""Element configuration base class for TraceISO."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple


@dataclass
class CertifiedValue:
    """A certified ratio and uncertainty at the accompanying coverage factor."""

    value: float
    uncertainty: float = 0.0
    k: float = 1.0  # coverage factor accompanying uncertainty
    source: str = ""  # e.g. "NIST SRM 987"


class DataLayer(Enum):
    """Declared correction-chain capability for an element.

    Each value names a correction layer that the element's pipeline
    branch is *capable* of producing. Whether a given pipeline run
    actually populates that layer depends on ``ProcessingConfig`` flags
    (e.g. ``drift.enabled``, ``enable_ssb``).

    Used by Inspector/Sidebar display logic to decide which UI controls
    to show, alongside the ``has_interference`` ``@property``.
    """
    RAW = "raw"
    BLANK_CORRECTED = "blank_corrected"
    INTERFERENCE_CORRECTED = "interference_corrected"
    IIF_CORRECTED = "iif_corrected"
    SSB_CORRECTED = "ssb_corrected"
    DRIFT_CORRECTED = "drift_corrected"


@dataclass(frozen=True)
class MonitorSpec:
    """Declares one isobaric monitor used in Step-1 (f-family) interference correction."""

    corrected_isotope: str
    """Channel being cleaned, e.g. '87Sr', '204Pb'."""
    interfering_isotope: str
    """Isobaric species subtracted, e.g. '87Rb', '204Hg'.
    Distinct from corrected_isotope — same nominal mass, different element."""
    monitor_isotope: str
    """Measured monitor channel, e.g. '85Rb', '83Kr', '202Hg'."""
    natural_ratio_key: str
    """CRM-library key for the interfering/monitor natural ratio, e.g. '87Rb/85Rb'."""
    family: Literal["f", "k"] = "f"
    """Which primitive: 'f' = natural-abundance form, 'k' = IIF/K-factor form."""


@dataclass(frozen=True)
class ElementConfig:
    """Immutable configuration for one isotope system."""

    symbol: str
    isotopes: List[str]
    default_ratios: Dict[str, Tuple[str, str]]
    certified_values: Dict[str, CertifiedValue] = field(default_factory=dict)
    reference_material: str = ""
    correction_steps: List[str] = field(default_factory=lambda: ["blank", "uncertainty"])
    supports_ssb: bool = False
    supports_delta: bool = False
    data_layers: Tuple["DataLayer", ...] = field(default_factory=tuple)
    """Declarative list of correction layers this element can produce.
    Replaces the old ``has_interference: bool`` field — that name now resolves
    to the ``has_interference`` ``@property`` defined below."""
    iterations: int = 1
    mass_bias_law: Optional[str] = None
    normalization_ratio: Optional[str] = None
    normalization_value: Optional[float] = None
    correction_roles: Dict[str, str] = field(default_factory=dict)
    monitors: Tuple["MonitorSpec", ...] = field(default_factory=tuple)
    """Step-1 (f-family) interference monitors."""
    ratio_name_aliases: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def primary_ratio(self) -> Optional[str]:
        """Return the first (primary) ratio name, or None."""
        if self.default_ratios:
            return next(iter(self.default_ratios))
        return None

    @property
    def ratio_names(self) -> List[str]:
        """List of all ratio names."""
        return list(self.default_ratios.keys())

    @property
    def has_interference(self) -> bool:
        """Backward-compatible: True when INTERFERENCE_CORRECTED is declared."""
        return DataLayer.INTERFERENCE_CORRECTED in self.data_layers

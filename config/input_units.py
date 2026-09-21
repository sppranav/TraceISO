"""Versioned extractor unit contract; no detector/gain conversion is implied."""
from dataclasses import dataclass
import math
import re

UNIT_CONTRACT = "traceiso.channel_units.v1"
LEGACY_CONVENTION = "bare_intensity_V_ratio_dimensionless.v1"
_TOKEN = re.compile(r"^\s*(?:[A-Za-z]\s*)?(?P<number>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(?P<unit>\s+1|[^\d]*)$")
_VOLTAGE = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6, "μV": 1e-6, "nV": 1e-9}

@dataclass(frozen=True)
class ParsedToken:
    number: float
    source_unit: str
    canonical_unit: str
    scale: float
    status: str

    @property
    def value(self):
        return self.number * self.scale

def parse_token(raw, *, ratio=False, bare_convention=LEGACY_CONVENTION):
    """Normalize SI voltage prefixes; ratio channels accept dimensionless units only."""
    text = "" if raw is None else str(raw).strip()
    canonical = "1" if ratio else "V"
    if text.lower() in {"", "nan", "nat", "none", "null", "na", "n/a"}:
        return ParsedToken(float("nan"), "", canonical, 1., "missing")
    match = _TOKEN.fullmatch(text)
    if not match:
        raise ValueError("malformed scientific token")
    number = float(match['number'])
    unit = match['unit'].strip()
    if not math.isfinite(number):
        raise ValueError("non-finite scientific token")
    if not unit:
        if bare_convention != LEGACY_CONVENTION:
            raise ValueError("bare number requires the documented voltage/ratio convention")
        return ParsedToken(number, "", canonical, 1., "legacy_convention")
    scales = {"1": 1., "dimensionless": 1., "mol/mol": 1., "ratio": 1.} if ratio else _VOLTAGE
    if unit not in scales:
        raise ValueError("unsupported or incompatible unit; detector counts require an approved conversion model")
    result = ParsedToken(number, unit, canonical, scales[unit], "converted" if scales[unit] != 1 else "declared")
    if not math.isfinite(result.value):
        raise ValueError("unit conversion overflow")
    return result

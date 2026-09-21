"""Read the committed Sr calibration availability without changing processing state."""
from typing import Optional

from domain.models import Sample, UncertaintyBudget


def sr_calibration_record(sample: Sample, ratio_name: Optional[str] = None) -> dict:
    record = sample.metadata.get("sr_standard_calibration") or {}
    if ratio_name is not None and record.get("ratio") != ratio_name:
        return {}
    return record


def sr_calibration_unavailable(sample: Sample, ratio_name: str) -> bool:
    return sr_calibration_record(sample, ratio_name).get("status") == "unavailable"


def sr_calibration_message(sample: Sample, ratio_name: Optional[str] = None) -> str:
    record = sr_calibration_record(sample, ratio_name)
    if record.get("status") != "unavailable":
        return ""
    return (
        f"Sr calibration unavailable for {record['ratio']}: {record['reason']} "
        "Available internally normalized results can still be inspected; calibrated final values are not reported."
    )


def sr_calibration_budget_guard(
    sample: Sample, ratio_name: str, output_mode: str,
) -> Optional[UncertaintyBudget]:
    if not sr_calibration_unavailable(sample, ratio_name):
        return None
    return UncertaintyBudget(
        engine="internal_normalization", output_mode=output_mode,
        budget_scope="unavailable", scope_note=sr_calibration_message(sample, ratio_name),
    )

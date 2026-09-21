"""Pb-standard calibration controls on the Pb-Tl route.

Widget state and presentation only: the controls return a
``PbStandardCalibrationConfig``; eligibility, estimation and freshness are
decided in ``domain.pb_standard_calibration``. Roles are assigned to observation
identities explicitly from runs typed STD or STANDARD; names do not determine
which runs the calibration-standard picker offers.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import streamlit as st

from config.reference_materials import get_crm_records
from config.settings import (
    PB_CALIBRATION_MODE_LOCAL_SSB,
    PB_CALIBRATION_MODE_SESSION_MEAN_K,
    PB_CALIBRATION_ROLE_NOT_USED,
    PB_CALIBRATION_ROLE_QC,
    PB_CALIBRATION_ROLE_STANDARD,
    PbStandardCalibrationConfig,
)

_MODE_LABELS = {
    PB_CALIBRATION_MODE_LOCAL_SSB: "Local SSB (bracketing standards)",
    PB_CALIBRATION_MODE_SESSION_MEAN_K: "Session mean of individual K factors",
}
_BRACKET_LABELS = {
    "alternating": "Nearest standard on each side",
    "block_average": "Adjacent standard blocks",
}
_PRECISION_LABELS = {
    "none": "None",
    "sd": "Cycle scatter (SD)",
    "se": "Precision of the mean (SE)",
}


def _observation_label(sample) -> str:
    return f"#{int(sample.run_number)} · {sample.name} ({sample.sample_type})"


def render_pb_calibration_controls(
    current: PbStandardCalibrationConfig,
    *,
    current_ssb_mode: str,
    samples: Sequence,
    key_prefix: str,
    reference_material: str | None,
) -> Tuple[PbStandardCalibrationConfig, str]:
    """Render the controls; return the calibration settings and the bracket convention."""
    st.caption("**Pb-standard calibration after Tl normalization**")
    enabled = st.checkbox(
        "Calibrate against Pb standards",
        value=current.enabled,
        key=f"{key_prefix}_pb_calibration_enabled",
        help=(
            "Scale each Tl-normalized Pb ratio by K = C / S, from explicitly assigned "
            "standards of one reference material. Off keeps the Tl-only result."
        ),
    )
    if not enabled:
        return PbStandardCalibrationConfig(**{
            **current.__dict__, "enabled": False, "reference_material": reference_material,
        }), current_ssb_mode

    mode = st.radio(
        "Calibration estimator",
        options=list(_MODE_LABELS),
        index=list(_MODE_LABELS).index(current.mode),
        format_func=_MODE_LABELS.get,
        key=f"{key_prefix}_pb_calibration_mode",
        help=(
            "Local SSB: K = C / mean of the bracketing standard means. Session: K is the "
            "mean of the individual C / S factors of every eligible standard."
        ),
    )
    ssb_mode = current_ssb_mode if current_ssb_mode in _BRACKET_LABELS else "alternating"
    if mode == PB_CALIBRATION_MODE_LOCAL_SSB:
        ssb_mode = st.radio(
            "Bracket convention",
            options=list(_BRACKET_LABELS),
            index=list(_BRACKET_LABELS).index(ssb_mode),
            format_func=_BRACKET_LABELS.get,
            key=f"{key_prefix}_pb_calibration_bracket",
        )

    reference = reference_material
    if reference is not None:
        st.caption(f"Calibration reference: {reference} (selected under Reference material).")
    material_id = None
    if reference is not None:
        records = get_crm_records("Pb", reference)
        material_id = (records[0].material_id if records and records[0].material_id else reference)

    candidates = [s for s in (samples or []) if not s.is_blank]
    labels = {s.observation_id: _observation_label(s) for s in candidates}
    ids = list(labels)
    standard_ids = [s.observation_id for s in candidates if s.is_standard]
    standards_key = f"{key_prefix}_pb_calibration_standards"
    if standards_key in st.session_state:
        stored = st.session_state[standards_key]
        eligible = [o for o in stored if o in standard_ids]
        if eligible != stored:
            st.session_state[standards_key] = eligible
    standards = st.multiselect(
        "Calibration standards",
        options=standard_ids,
        default=[o for o in standard_ids if current.role_assignments.get(o) == PB_CALIBRATION_ROLE_STANDARD],
        format_func=labels.get,
        key=standards_key,
        help="Only runs typed STD or STANDARD are offered. Select the standards used to calculate K.",
    )
    qc_options = [o for o in ids if o not in standards]
    qc = st.multiselect(
        "Independent QC (corrected, never used for calibration)",
        options=qc_options,
        default=[o for o in qc_options if current.role_assignments.get(o) == PB_CALIBRATION_ROLE_QC],
        format_func=labels.get,
        key=f"{key_prefix}_pb_calibration_qc",
    )

    roles = {
        o: role for o, role in current.role_assignments.items()
        if o not in labels or role == PB_CALIBRATION_ROLE_NOT_USED
    }
    roles.update({o: PB_CALIBRATION_ROLE_STANDARD for o in standards})
    roles.update({o: PB_CALIBRATION_ROLE_QC for o in qc})
    materials = {o: m for o, m in current.material_assignments.items() if o not in labels}
    if material_id is not None:
        materials.update({o: material_id for o in standards})
    by_id = {s.observation_id: s for s in candidates}
    locators = {
        o: {"name": by_id[o].name, "run_number": int(by_id[o].run_number), "sample_type": by_id[o].sample_type}
        for o in roles if o in by_id
    }

    enable_delta = st.checkbox(
        "Report calibrated delta values",
        value=current.enable_delta,
        key=f"{key_prefix}_pb_calibration_delta",
        help="delta = 1000 (Y / C - 1) from the same calibration; no second bracket search.",
    )
    statistic = current.delta_precision_statistic
    if enable_delta:
        statistic = st.selectbox(
            "Delta precision shown",
            options=list(_PRECISION_LABELS),
            index=list(_PRECISION_LABELS).index(statistic),
            format_func=_PRECISION_LABELS.get,
            key=f"{key_prefix}_pb_calibration_delta_precision",
            help="Cycle scatter or precision of the mean. Neither is a combined measurement uncertainty.",
        )

    if reference is None:
        st.warning("Select the accepted Pb reference material; without it every calibrated ratio is unavailable.")
    if not standards:
        st.warning("No calibration standard is assigned; every calibrated ratio will be unavailable.")
    st.caption(
        "Separate drift correction is not applied on this route. Calibrated absolute ratios have "
        "their dedicated GUM budget and Monte Carlo cross-check when the required inputs are "
        "available; disclosed coverage limitations remain visible. Calibrated delta combined "
        "uncertainty and Monte Carlo are not calculated."
    )
    return PbStandardCalibrationConfig(
        enabled=True,
        mode=mode,
        reference_material=reference,
        role_assignments=roles,
        material_assignments=materials,
        assignment_locators=locators,
        enable_delta=enable_delta,
        delta_precision_statistic=statistic,
        min_valid_cycles_alternating=current.min_valid_cycles_alternating,
        min_valid_cycles_block=current.min_valid_cycles_block,
        min_valid_cycles_session=current.min_valid_cycles_session,
    ), ssb_mode

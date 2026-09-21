"""Explicit Sr calibration-standard selection after internal normalization."""

import streamlit as st


def render_sr_calibration_controls(current, samples, *, key_prefix):
    enabled = st.checkbox(
        "Calibrate against Sr standard",
        value=current.sr_session_anchoring,
        key=f"{key_prefix}_sr_anchoring",
        help=(
            "Rescales the internally normalised 87Sr/86Sr ratio by the assigned "
            "reference value divided by the mean of the selected internally "
            "normalised standards."
        ),
    )
    selected = list(current.sr_calibration_standard_ids)
    if enabled:
        candidates = [s for s in (samples or []) if s.is_standard and not s.metadata.get("excluded", False)]
        labels = {s.observation_id: f"#{s.run_number:g} · {s.name} ({s.sample_type})" for s in candidates}
        key = f"{key_prefix}_sr_calibration_standards"
        if key in st.session_state:
            eligible = [oid for oid in st.session_state[key] if oid in labels]
            if eligible != st.session_state[key]:
                st.session_state[key] = eligible
        selected = st.multiselect(
            "Calibration standards",
            options=list(labels),
            default=[oid for oid in selected if oid in labels],
            format_func=labels.get,
            key=key,
            help="Select the Sr standard runs used for calibration against the selected reference material.",
        )
        if not selected:
            st.warning("No calibration standard is selected. Calibrated results will be unavailable until you select standards and reprocess.")
    return enabled, selected


def render_sr_calibration_notice(samples) -> None:
    """Show distinct committed calibration refusals without hiding diagnostic data."""
    from domain.sr_standard_calibration import sr_calibration_message
    messages = dict.fromkeys(sr_calibration_message(sample) for sample in samples)
    for message in messages:
        if message:
            st.warning(message)

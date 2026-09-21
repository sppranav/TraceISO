"""Configure subview for the redesigned Uncertainty tab."""

from __future__ import annotations

import streamlit as st

from config.settings import UncertaintyConfig
from ui.tabs.uncertainty_sections import config_controls
from ui.tabs.uncertainty_sections import per_sample_contributors
from ui.tabs.uncertainty_sections import shared_ui
from ui.tabs.uncertainty_sections.ui_context import (
    UncertaintyUiContext,
    built_in_contributors_for_context,
    resolve_ui_context,
)

_CONFIGURE_PANES = ("Budget settings", "Profiles & applicability")
_CONFIGURE_PANE_KEY = "uncertainty_configure_pane"


def render_configure_view(
    *,
    samples: list,
    selected_ratio: str,
    ratio_list: list[str],
    context: UncertaintyUiContext,
    u_config: UncertaintyConfig,
    state,
) -> UncertaintyConfig:
    """Render budget setup controls and persist all config changes immediately.

    Split into two sub-panes (``Budget settings`` / ``Profiles &
    applicability``). This is presentation only: persistence and the staged
    per-sample applicability form are unchanged.
    """
    st.subheader("Budget Setup")

    pane = shared_ui.render_subview_nav(
        _CONFIGURE_PANES,
        key=_CONFIGURE_PANE_KEY,
        default="Budget settings",
    )

    if pane == "Budget settings":
        st.caption(
            "Changes to budget settings apply immediately. Review Results & Statistics "
            "again when uncertainty settings change, then continue to Export."
        )
        updated = config_controls._render_config_panel(
            u_config,
            ui_context=context,
            expanded=True,
            framed=False,
        )
        if updated != state.uncertainty_config:
            state.uncertainty_config = updated
            u_config = updated
        if context.ui_kind in {"sr_internal", "pb_tl"}:
            per_sample_contributors.render_sr_sample_value_matrix(
                samples,
                qc_bias_enabled=bool(
                    u_config.is_contributor_enabled(
                        "u_bias_qc",
                        element_symbol=context.element_symbol,
                    )
                ),
                reprod_dig_enabled=bool(
                    u_config.is_contributor_enabled(
                        "u_reprod_dig",
                        element_symbol=context.element_symbol,
                    )
                ),
                sr_qc_bias_abs=float(getattr(u_config, "sr_qc_bias_abs", 0.0)),
                sr_qc_cert_value=float(getattr(u_config, "sr_qc_cert_value", 0.0)),
                u_reprod_dig_sd=float(getattr(u_config, "u_reprod_dig_sd", 0.0)),
                u_reprod_dig_ref_value=float(
                    getattr(u_config, "u_reprod_dig_ref_value", 0.0)
                ),
                expanded=True,
            )
        return u_config

    # Profiles & applicability. Re-resolve contributor visibility so the matrix
    # reflects the persisted budget settings.
    context = resolve_ui_context(
        state,
        u_config,
        ratio_list=ratio_list,
        selected_ratio=selected_ratio,
        samples=samples,
    )
    st.caption(
        "Profile and per-sample applicability edits are staged. Use the Apply "
        "controls below to make them available to Budget review and Diagnostics."
    )

    custom_library = state.custom_contributor_library or {}
    custom_for_element = custom_library.get(context.element_symbol, [])
    per_sample_contributors.render_per_sample_contributor_matrix(
        samples,
        built_in_contributors=built_in_contributors_for_context(
            context,
            u_config,
            state,
        ),
        custom_contributors=custom_for_element,
        expanded=True,
        profile_manager_expanded=True,
        include_standards=context.ui_kind == "sr_internal",
    )
    if context.ui_kind in {"sr_internal", "pb_tl"}:
        per_sample_contributors.render_sr_sample_value_matrix(
            samples,
            qc_bias_enabled=bool(
                u_config.is_contributor_enabled(
                    "u_bias_qc",
                    element_symbol=context.element_symbol,
                )
            ),
            reprod_dig_enabled=bool(
                u_config.is_contributor_enabled(
                    "u_reprod_dig",
                    element_symbol=context.element_symbol,
                )
            ),
            sr_qc_bias_abs=float(getattr(u_config, "sr_qc_bias_abs", 0.0)),
            sr_qc_cert_value=float(getattr(u_config, "sr_qc_cert_value", 0.0)),
            u_reprod_dig_sd=float(getattr(u_config, "u_reprod_dig_sd", 0.0)),
            u_reprod_dig_ref_value=float(
                getattr(u_config, "u_reprod_dig_ref_value", 0.0)
            ),
            expanded=True,
        )

    return u_config

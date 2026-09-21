"""Theme Manager for TraceISO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

import plotly.graph_objects as go
import plotly.io as pio
from ui.navigation import SESSION_KEY_THEME_NAME
import streamlit as st

from ui.config_plotly import (
    PLOTLY_AXIS_TITLE_FONT_SIZE,
    PLOTLY_BASE_FONT_SIZE,
    PLOTLY_LEGEND_FONT_SIZE,
    PLOTLY_TITLE_FONT_SIZE,
)


# Shared interface type scale. Plotly screen and export figures use separate
# named profiles below; changing the interface scale does not resize exports.
UI_FONT_SIZE_BODY_PX = 14
UI_FONT_SIZE_CONTROL_PX = 14
UI_FONT_SIZE_SUPPORTING_PX = 13
UI_FONT_SIZE_SECTION_PX = 18


@dataclass
class ColorPalette:
    """Defines a consistent color palette."""

    primary: str
    secondary: str
    accent: str
    background: str
    surface: str
    text: str

    # Publication-figure surfaces and structural styling. These are separate
    # from application chrome so figures can migrate without recolouring the UI.
    figure_paper: str
    figure_surface: str
    figure_ink: str
    figure_grid: str
    figure_axis: str
    annotation_border: str

    # Scientific correction-layer colors. Do not reuse status colors for data.
    layer_raw: str
    layer_blank: str
    layer_drift: str
    layer_interference: str
    layer_final: str
    layer_excluded: str
    guide_bounds: str
    inactive_fill: str
    layer_raw_light: str
    layer_blank_light: str
    layer_drift_light: str
    layer_interference_light: str
    layer_final_light: str
    layer_excluded_light: str
    guide_bounds_light: str

    # Sample-type colors are a separate semantic dimension from corrections.
    sample_smp: str
    sample_std: str
    sample_blk: str
    sample_qc: str
    sample_smp_light: str
    sample_std_light: str
    sample_blk_light: str
    sample_qc_light: str

    annotation_bg: str = "rgba(255,255,255,0.9)"
    annotation_text_color: str = "#0B0F17"
    success: str = "#18BC9C"
    warning: str = "#F39C12"
    error: str = "#E74C3C"

    # Scientific sample-type colors
    samples: str = "#1565C0"  # Samples: blue
    standards: str = "#C62828"  # Standards: red
    blanks: str = "#616161"  # Blanks: grey
    excluded: str = "rgba(120, 120, 120, 0.35)"  # Excluded/faded

    # Transparent variants for shaded bands
    primary_light: str = "rgba(44, 62, 80, 0.2)"
    secondary_light: str = "rgba(231, 76, 60, 0.2)"
    success_light: str = "rgba(24, 188, 156, 0.2)"
    warning_light: str = "rgba(243, 156, 18, 0.2)"
    samples_light: str = "rgba(21, 101, 192, 0.18)"
    standards_light: str = "rgba(198, 40, 40, 0.18)"
    blanks_light: str = "rgba(97, 97, 97, 0.18)"

    # Plotly sequence
    sequence: List[str] = field(
        default_factory=lambda: [
            "#1565C0",
            "#00897B",
            "#EF6C00",
            "#5E35B1",
            "#2E7D32",
            "#6D4C41",
        ]
    )
    contributor_sequence: List[str] = field(
        default_factory=lambda: [
            "#1D4ED8",
            "#0F766E",
            "#B45309",
            "#6D28D9",
            "#15803D",
            "#9F1239",
            "#0891B2",
            "#7C3AED",
        ]
    )

    # Semantic flag chip colors
    flag_ok_fg: str = "#166534"
    flag_ok_bg: str = "rgba(22, 101, 52, 0.16)"
    flag_elev_fg: str = "#B45309"
    flag_elev_bg: str = "rgba(180, 83, 9, 0.18)"
    flag_high_fg: str = "#991B1B"
    flag_high_bg: str = "rgba(153, 27, 27, 0.16)"
    flag_interf_fg: str = "#6D28D9"
    flag_interf_bg: str = "rgba(109, 40, 217, 0.14)"
    flag_unav_fg: str = "#5A6172"
    flag_unav_bg: str = "#EEF0F3"


# Define standard themes
THEMES = {
    "light": ColorPalette(
        # Dense data-terminal palette: crisp ink on white, one restrained
        # blue accent, saturated categorical sequence for plots.
        primary="#0B0F17",  # Ink - metric values, plot titles
        secondary="#B91C1C",  # Danger red (sparingly)
        accent="#1D4ED8",  # Action/state blue
        background="#FFFFFF",
        surface="#F4F5F7",  # Subtle plot surface
        text="#0B0F17",
        figure_paper="#FFFFFF",
        figure_surface="#FFFFFF",
        figure_ink="#29313A",
        figure_grid="#E3E7EC",
        figure_axis="#29313A",
        annotation_border="#C8CED6",
        layer_raw="#7A828C",
        layer_blank="#0072B2",
        layer_drift="#009E73",
        layer_interference="#E69F00",
        layer_final="#6F3C97",
        layer_excluded="#C43C39",
        guide_bounds="#4B5563",
        inactive_fill="rgba(122,130,140,0.10)",
        layer_raw_light="rgba(122,130,140,0.16)",
        layer_blank_light="rgba(0,114,178,0.16)",
        layer_drift_light="rgba(0,158,115,0.16)",
        layer_interference_light="rgba(230,159,0,0.16)",
        layer_final_light="rgba(111,60,151,0.16)",
        layer_excluded_light="rgba(196,60,57,0.16)",
        guide_bounds_light="rgba(75,85,99,0.12)",
        sample_smp="#0072B2",
        sample_std="#D55E00",
        sample_blk="#7A828C",
        sample_qc="#6F3C97",
        sample_smp_light="rgba(0,114,178,0.16)",
        sample_std_light="rgba(213,94,0,0.16)",
        sample_blk_light="rgba(122,130,140,0.16)",
        sample_qc_light="rgba(111,60,151,0.16)",
        annotation_bg="rgba(255,255,255,0.9)",
        annotation_text_color="#0B0F17",
        success="#166534",
        warning="#B45309",
        error="#991B1B",
        samples="#1D4ED8",
        standards="#B91C1C",
        blanks="#6B7280",
        excluded="rgba(110, 116, 130, 0.35)",
        primary_light="rgba(11, 15, 23, 0.16)",
        secondary_light="rgba(185, 28, 28, 0.16)",
        success_light="rgba(22, 101, 52, 0.16)",
        warning_light="rgba(180, 83, 9, 0.18)",
        samples_light="rgba(29, 78, 216, 0.16)",
        standards_light="rgba(185, 28, 28, 0.16)",
        blanks_light="rgba(107, 114, 128, 0.16)",
        sequence=[
            "#1D4ED8",
            "#0F766E",
            "#B45309",
            "#6D28D9",
            "#15803D",
            "#9F1239",
            "#0891B2",
            "#7C3AED",
        ],
        contributor_sequence=[
            "#1D4ED8",
            "#9B59B6",
            "#0F766E",
            "#B45309",
            "#6D28D9",
            "#15803D",
            "#9F1239",
            "#0891B2",
        ],
    ),
    "dark": ColorPalette(
        primary="#F4F6FA",
        secondary="#F87171",
        accent="#7CA7FF",
        background="#0F1218",
        surface="#171B22",
        text="#F4F6FA",
        figure_paper="#0F1218",
        figure_surface="#171B22",
        figure_ink="#F4F6FA",
        figure_grid="rgba(244,246,250,0.12)",
        figure_axis="#F4F6FA",
        annotation_border="#4B5563",
        layer_raw="#A7B0BA",
        layer_blank="#56B4E9",
        layer_drift="#5EEAD4",
        layer_interference="#F2B86B",
        layer_final="#C4B5FD",
        layer_excluded="#F87171",
        guide_bounds="#CBD5E1",
        inactive_fill="rgba(167,176,186,0.12)",
        layer_raw_light="rgba(167,176,186,0.18)",
        layer_blank_light="rgba(86,180,233,0.18)",
        layer_drift_light="rgba(94,234,212,0.18)",
        layer_interference_light="rgba(242,184,107,0.18)",
        layer_final_light="rgba(196,181,253,0.18)",
        layer_excluded_light="rgba(248,113,113,0.18)",
        guide_bounds_light="rgba(203,213,225,0.16)",
        sample_smp="#56B4E9",
        sample_std="#FF9B73",
        sample_blk="#A7B0BA",
        sample_qc="#C4B5FD",
        sample_smp_light="rgba(86,180,233,0.18)",
        sample_std_light="rgba(255,155,115,0.18)",
        sample_blk_light="rgba(167,176,186,0.18)",
        sample_qc_light="rgba(196,181,253,0.18)",
        annotation_bg="rgba(23,27,34,0.9)",
        annotation_text_color="#F4F6FA",
        success="#7DD3A7",
        warning="#F2B86B",
        error="#F87171",
        samples="#8AB4FF",
        standards="#FCA5A5",
        blanks="#9CA3AF",
        excluded="rgba(210, 216, 228, 0.28)",
        primary_light="rgba(244, 246, 250, 0.14)",
        secondary_light="rgba(248, 113, 113, 0.18)",
        success_light="rgba(125, 211, 167, 0.18)",
        warning_light="rgba(242, 184, 107, 0.18)",
        samples_light="rgba(138, 180, 255, 0.18)",
        standards_light="rgba(252, 165, 165, 0.18)",
        blanks_light="rgba(156, 163, 175, 0.18)",
        sequence=[
            "#8AB4FF",
            "#5EEAD4",
            "#F2B86B",
            "#C4B5FD",
            "#7DD3A7",
            "#FDA4AF",
            "#67E8F9",
            "#A78BFA",
        ],
        contributor_sequence=[
            "#8AB4FF",
            "#C4B5FD",
            "#5EEAD4",
            "#F2B86B",
            "#7DD3A7",
            "#FDA4AF",
            "#67E8F9",
            "#A78BFA",
        ],
        flag_ok_fg="#7DD3A7",
        flag_ok_bg="rgba(125, 211, 167, 0.18)",
        flag_elev_fg="#F2B86B",
        flag_elev_bg="rgba(242, 184, 107, 0.18)",
        flag_high_fg="#F87171",
        flag_high_bg="rgba(248, 113, 113, 0.18)",
        flag_interf_fg="#C4B5FD",
        flag_interf_bg="rgba(196, 181, 253, 0.18)",
        flag_unav_fg="#A7B0C0",
        flag_unav_bg="rgba(167, 176, 192, 0.14)",
    ),
}


FigureProfile = Literal[
    "legacy",
    "timeseries",
    "overview",
    "screen_overview",
    "export_overview",
    "distribution",
    "diagnostic",
]

_FIGURE_PROFILE_GRIDS: dict[str, tuple[bool, bool]] = {
    "legacy": (True, True),
    "timeseries": (False, True),
    "overview": (False, False),
    "screen_overview": (False, False),
    "export_overview": (False, False),
    "distribution": (False, True),
    # Diagnostic figures are predominantly horizontal contribution/bias views,
    # so their quantitative grid is the x-axis. Callers with a vertical
    # quantitative axis should use ``timeseries`` instead.
    "diagnostic": (True, False),
}


class ThemeManager:
    """Manages application-wide theming and plot styling."""

    def __init__(self, theme_name: str = "light"):
        self.theme_name = theme_name
        self.palette = THEMES.get(theme_name, THEMES["light"])
        self._setup_plotly_template()

    def get_palette(self) -> ColorPalette:
        """Get the current color palette."""
        return self.palette

    def _setup_plotly_template(self) -> None:
        """Create a custom Plotly template based on the theme."""
        is_light = self.theme_name == "light"
        grid_color = "rgba(0,0,0,0.1)" if is_light else "rgba(255,255,255,0.1)"
        zero_color = "rgba(0,0,0,0.2)" if is_light else "rgba(255,255,255,0.2)"
        legend_bg = "rgba(255,255,255,0.5)" if is_light else "rgba(0,0,0,0.5)"
        border_color = "rgba(0,0,0,0.35)" if is_light else "rgba(255,255,255,0.35)"

        template = go.layout.Template()
        template.layout = go.Layout(
            font=dict(
                family='-apple-system, "Segoe UI", Roboto, sans-serif',
                color=self.palette.text,
                size=PLOTLY_BASE_FONT_SIZE,
            ),
            # ``text=""`` must be explicit: Plotly.js title merging treats a
            # missing "text" key as "leave the previous title untouched"
            # (not "no title"), which can leak a stale/undefined title onto
            # figures that never set one (e.g. a literal "undefined" string
            # rendered as the chart title on some redraw paths).
            title=dict(text="", font=dict(size=PLOTLY_TITLE_FONT_SIZE, color=self.palette.primary)),
            xaxis=dict(
                showgrid=True,
                gridcolor=grid_color,
                zeroline=True,
                zerolinecolor=zero_color,
                showline=True,
                linecolor=border_color,
                mirror=True,
                color=self.palette.text,
                tickcolor=self.palette.text,
                tickfont=dict(size=PLOTLY_BASE_FONT_SIZE),
                title=dict(font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE, color=self.palette.text)),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=grid_color,
                zeroline=True,
                zerolinecolor=zero_color,
                showline=True,
                linecolor=border_color,
                mirror=True,
                color=self.palette.text,
                tickcolor=self.palette.text,
                tickfont=dict(size=PLOTLY_BASE_FONT_SIZE),
                title=dict(font=dict(size=PLOTLY_AXIS_TITLE_FONT_SIZE, color=self.palette.text)),
            ),
            plot_bgcolor=self.palette.surface,
            paper_bgcolor=self.palette.background,
            colorway=self.palette.sequence,
            legend=dict(
                bgcolor=legend_bg,
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=dict(size=PLOTLY_LEGEND_FONT_SIZE),
            ),
        )

        # Register one product-named template; figures opt in per-session.
        pio.templates["traceiso"] = template
        # item 70: do NOT set pio.templates.default here — that is a
        # process-global setting shared across concurrent Streamlit sessions.
        # Each figure must call apply_to_figure() or pass template="traceiso"
        # explicitly (apply_to_figure does this via update_layout(template=...)).

    def apply_to_figure(
        self,
        fig: go.Figure,
        profile: FigureProfile = "legacy",
    ) -> go.Figure:
        """Apply the theme, optionally using a publication figure profile.

        ``legacy`` intentionally preserves the pre-publication behavior so
        figure families can migrate independently. New profiles apply the
        publication surface, typography, axes, annotations, and grid policy.
        """
        if profile not in _FIGURE_PROFILE_GRIDS:
            valid = ", ".join(_FIGURE_PROFILE_GRIDS)
            raise ValueError(f"Unknown figure profile {profile!r}; expected one of: {valid}.")

        fig.update_layout(
            template="traceiso",
            paper_bgcolor=self.palette.background,
            plot_bgcolor=self.palette.surface,
            font=dict(color=self.palette.text),
        )
        fig.update_xaxes(
            color=self.palette.text,
            linecolor=self.palette.text,
            tickcolor=self.palette.text,
            tickfont_color=self.palette.text,
            title_font_color=self.palette.text,
        )
        fig.update_yaxes(
            color=self.palette.text,
            linecolor=self.palette.text,
            tickcolor=self.palette.text,
            tickfont_color=self.palette.text,
            title_font_color=self.palette.text,
        )

        if profile == "legacy":
            return fig

        x_grid, y_grid = _FIGURE_PROFILE_GRIDS[profile]
        palette = self.palette
        font_family = "Arial, Helvetica, sans-serif"
        # Setting ``title_font`` (i.e. ``title=dict(font=...)``) without an
        # accompanying ``text`` makes this figure's own layout own the
        # "title" attribute; Plotly.js then no longer falls back to the
        # template's ``title.text`` default for it, and can render the
        # literal string "undefined" as the chart title. Always pass an
        # explicit ``text`` (preserving any title the caller already set)
        # alongside the font so the attribute is never left undefined.
        existing_title = fig.layout.title.text if fig.layout.title is not None else None
        fig.update_layout(
            paper_bgcolor=palette.figure_paper,
            plot_bgcolor=palette.figure_surface,
            font=dict(family=font_family, color=palette.figure_ink),
            title=dict(
                text=existing_title or "",
                font=dict(
                    family=font_family,
                    color=palette.figure_ink,
                ),
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=dict(
                    family=font_family,
                    color=palette.figure_ink,
                ),
            ),
        )
        fig.update_xaxes(
            showgrid=x_grid,
            gridcolor=palette.figure_grid,
            color=palette.figure_ink,
            linecolor=palette.figure_axis,
            tickcolor=palette.figure_axis,
            tickfont=dict(
                family=font_family,
                color=palette.figure_ink,
            ),
            title_font=dict(
                family=font_family,
                color=palette.figure_ink,
            ),
        )
        fig.update_yaxes(
            showgrid=y_grid,
            gridcolor=palette.figure_grid,
            color=palette.figure_ink,
            linecolor=palette.figure_axis,
            tickcolor=palette.figure_axis,
            tickfont=dict(
                family=font_family,
                color=palette.figure_ink,
            ),
            title_font=dict(
                family=font_family,
                color=palette.figure_ink,
            ),
        )
        fig.update_annotations(
            font_color=palette.figure_ink,
            bgcolor=palette.annotation_bg,
            bordercolor=palette.annotation_border,
        )
        if profile in {"screen_overview", "export_overview"}:
            sizes = (
                {
                    "tick": 14,
                    "axis": 16,
                    "title": 18,
                    "legend": 13,
                    "annotation": 13,
                }
                if profile == "screen_overview"
                else {
                    "tick": PLOTLY_BASE_FONT_SIZE,
                    "axis": PLOTLY_AXIS_TITLE_FONT_SIZE + 4,
                    "title": PLOTLY_TITLE_FONT_SIZE + 3,
                    "legend": PLOTLY_BASE_FONT_SIZE + 3,
                    "annotation": PLOTLY_LEGEND_FONT_SIZE + 3,
                }
            )
            fig.update_layout(
                font=dict(size=sizes["tick"]),
                title_font_size=sizes["title"],
                legend_font_size=sizes["legend"],
            )
            fig.update_xaxes(
                tickfont_size=sizes["tick"],
                title_font_size=sizes["axis"],
            )
            fig.update_yaxes(
                tickfont_size=sizes["tick"],
                title_font_size=sizes["axis"],
            )
            fig.update_annotations(font_size=sizes["annotation"])
        return fig

    def annotation_style(self) -> dict:
        """Return default styling for fig.add_annotation()."""
        return {
            "font": dict(color=self.palette.annotation_text_color),
            "bgcolor": self.palette.annotation_bg,
            "bordercolor": "rgba(0,0,0,0)",
        }

    def inject_css(self) -> None:
        """Inject theme CSS into the Streamlit page."""
        st.markdown(self._get_css(), unsafe_allow_html=True)
        # One-time sidebar localStorage cleanup: only on the very first
        # render of a session, clear any stale "collapsed" sidebar state so
        # initial_sidebar_state="expanded" is respected.  Uses
        # components.html() because st.markdown strips <script> tags.
        # Guarded by session_state so the iframe is NOT created on every
        # rerun (which would add overhead and prevent intentional collapse).
        if not st.session_state.get("_sidebar_ls_cleaned"):
            st.session_state["_sidebar_ls_cleaned"] = True
            import streamlit.components.v1 as components
            components.html(
                """
                <script>
                (function() {
                    try {
                        var storage = window.parent.localStorage;
                        var keys = Object.keys(storage);
                        for (var i = 0; i < keys.length; i++) {
                            if (keys[i].indexOf('sidebar') !== -1 || keys[i].indexOf('Sidebar') !== -1) {
                                storage.removeItem(keys[i]);
                            }
                        }
                    } catch(e) {}
                })();
                </script>
                """,
                height=0,
            )

    def _get_css(self) -> str:
        """Get CSS for compact, professional styling."""
        is_dark = self.theme_name == "dark"
        ink_2 = "#D5DBE7" if is_dark else "#2A2F3A"
        ink_3 = "#A7B0C0" if is_dark else "#5A6172"
        ink_4 = "#747F91" if is_dark else "#646B78"
        line = "#3A4351" if is_dark else "#DCDFE5"
        line_2 = "#556171" if is_dark else "#C4C8D0"
        accent_soft = self.palette.standards_light
        code_bg = "#171B22" if is_dark else "#F7F8FA"
        code_border = "#2E3744" if is_dark else "#EEF0F3"
        tooltip_text = "#F4F6FA" if is_dark else "#1A1A1A"
        tooltip_bg = "#171B22" if is_dark else "#FFFFFF"
        tooltip_border = "#3A4351" if is_dark else "#DCDFE5"
        placeholder_text = "#A7B0C0" if is_dark else "#6B6B6B"
        primary_button_text = "#0F1218" if is_dark else "#FFFFFF"
        contributor_sequence = list(
            self.palette.contributor_sequence or self.palette.sequence
        )
        while len(contributor_sequence) < 8:
            contributor_sequence.extend(self.palette.sequence)
        contributor_sequence = contributor_sequence[:8]
        return f"""
        <style>
            /*
             * Streamlit and BaseWeb inject high-specificity styles after app CSS.
             * These overrides intentionally use !important where needed; page-level
             * overrides should do the same when competing with Streamlit widgets.
             */
            :root {{
                --primary-color: {self.palette.primary};
                --secondary-color: {self.palette.secondary};
                --accent-color: {self.palette.accent};
                --background-color: {self.palette.background};
                --surface-color: {self.palette.surface};
                --text-color: {self.palette.text};

                --t-ink:    {self.palette.text};
                --t-ink-2:  {ink_2};
                --t-ink-3:  {ink_3};
                --t-ink-4:  {ink_4};
                --t-line:   {line};
                --t-line-2: {line_2};
                --t-bg:     {self.palette.background};
                --t-bg-2:   {self.palette.surface};
                --t-accent: {self.palette.accent};
                --t-accent-soft: {accent_soft};
                --t-mono:   "Cascadia Code","Consolas","SF Mono","DejaVu Sans Mono",monospace;
                --t-sans:   "Segoe UI",Roboto,-apple-system,Arial,sans-serif;
                --t-size-body: {UI_FONT_SIZE_BODY_PX}px;
                --t-size-control: {UI_FONT_SIZE_CONTROL_PX}px;
                --t-size-supporting: {UI_FONT_SIZE_SUPPORTING_PX}px;
                --t-size-section: {UI_FONT_SIZE_SECTION_PX}px;
                --smp: {self.palette.sample_smp};
                --std: {self.palette.sample_std};
                --blk: {self.palette.sample_blk};
                --qc:  {self.palette.sample_qc};
                --smp-soft: {self.palette.sample_smp_light};
                --std-soft: {self.palette.sample_std_light};
                --blk-soft: {self.palette.sample_blk_light};
                --qc-soft:  {self.palette.sample_qc_light};
                --code-bg: {code_bg};
                --code-border: {code_border};
                --tooltip-text: {tooltip_text};
                --tooltip-bg: {tooltip_bg};
                --tooltip-border: {tooltip_border};
                --placeholder-text: {placeholder_text};
                --primary-button-text: {primary_button_text};

                --flag-ok-fg: {self.palette.flag_ok_fg};
                --flag-ok-bg: {self.palette.flag_ok_bg};
                --flag-elev-fg: {self.palette.flag_elev_fg};
                --flag-elev-bg: {self.palette.flag_elev_bg};
                --flag-high-fg: {self.palette.flag_high_fg};
                --flag-high-bg: {self.palette.flag_high_bg};
                --flag-interf-fg: {self.palette.flag_interf_fg};
                --flag-interf-bg: {self.palette.flag_interf_bg};
                --flag-unav-fg: {self.palette.flag_unav_fg};
                --flag-unav-bg: {self.palette.flag_unav_bg};

                --contrib-1: {contributor_sequence[0]};
                --contrib-2: {contributor_sequence[1]};
                --contrib-3: {contributor_sequence[2]};
                --contrib-4: {contributor_sequence[3]};
                --contrib-5: {contributor_sequence[4]};
                --contrib-6: {contributor_sequence[5]};
                --contrib-7: {contributor_sequence[6]};
                --contrib-8: {contributor_sequence[7]};
            }}

            /* Captions and helper text */
            [data-testid="stCaptionContainer"],
            [data-testid="stCaptionContainer"] p,
            [data-testid="stCaptionContainer"] strong,
            [data-testid="stCaptionContainer"] span,
            [data-testid="stCaptionContainer"] a,
            .stCaption,
            .stCaption p,
            [data-testid="stMarkdownContainer"] small {{
                color: var(--t-ink-3) !important;
                font-size: var(--t-size-supporting) !important;
                line-height: 1.45 !important;
            }}

            [data-testid="stWidgetLabel"],
            [data-testid="stWidgetLabel"] p,
            [data-testid="stWidgetLabel"] label,
            label[data-testid="stWidgetLabel"] {{
                color: {self.palette.text} !important;
                margin-bottom: 0.1rem !important;
            }}
            [data-testid="stWidgetLabel"] p {{
                margin-bottom: 0 !important;
            }}
            input::placeholder,
            textarea::placeholder,
            [data-baseweb="input"] input::placeholder,
            [data-baseweb="textarea"] textarea::placeholder,
            [data-baseweb="select"] [class*="placeholder"] {{
                color: var(--placeholder-text) !important;
                opacity: 1 !important;
            }}

            [data-testid="stTooltipContent"],
            [data-testid="stTooltipContent"] p {{
                color: var(--tooltip-text) !important;
            }}
            [data-testid="stTooltipContent"] {{
                background: var(--tooltip-bg) !important;
                border: 1px solid var(--tooltip-border) !important;
            }}

            /* App chrome */
            /* Hide the header bar.  We use visibility:hidden rather than
               display:none or overflow:hidden because visibility is the one
               CSS property a descendant can override regardless of nesting
               depth (a child with visibility:visible IS visible even when
               every ancestor is visibility:hidden).  This lets the sidebar
               collapse-toggle remain clickable without knowing its exact
               DOM nesting path inside the header. */
            header[data-testid="stHeader"] {{
                visibility: hidden !important;
                height: 0 !important;
                min-height: 0 !important;
                padding: 0 !important;
                overflow: visible !important;
                background: transparent !important;
                border: none !important;
            }}
            header[data-testid="stHeader"] * {{
                visibility: hidden !important;
            }}
            /* Sidebar re-open button: force visible + fixed position so it
               escapes the hidden header.  Target every known data-testid
               variant across Streamlit versions, AND all their descendants.
               "stExpandSidebarButton" is the current (1.5x) testid; the
               others are kept for compatibility with older Streamlit builds. */
            [data-testid="stSidebarCollapsedControl"],
            [data-testid="stSidebarCollapsedControl"] *,
            [data-testid="collapsedControl"],
            [data-testid="collapsedControl"] *,
            [data-testid="stExpandSidebarButton"],
            [data-testid="stExpandSidebarButton"] *,
            [data-testid="stToolbar"] [data-testid="stExpandSidebarButton"],
            [data-testid="stToolbar"] [data-testid="stExpandSidebarButton"] * {{
                visibility: visible !important;
            }}
            [data-testid="stSidebarCollapsedControl"],
            [data-testid="collapsedControl"],
            [data-testid="stExpandSidebarButton"] {{
                position: fixed !important;
                top: 0.375rem !important;
                left: 0.375rem !important;
                z-index: 999999 !important;
                display: flex !important;
                opacity: 1 !important;
                pointer-events: auto !important;
                background: var(--t-bg, #FFFFFF) !important;
                border: 1px solid var(--t-line, #DCDFE5) !important;
                border-radius: 4px !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
                padding: 2px !important;
                min-width: 28px !important;
                min-height: 28px !important;
                align-items: center !important;
                justify-content: center !important;
            }}
            /* stToolbar hosts the sidebar re-expand button (nested several
               levels deep) alongside the deploy/menu controls we don't want.
               Must hide via visibility, not display:none -- display:none on
               an ancestor cannot be undone by any descendant rule, which
               previously made the re-expand button permanently unreachable
               once the sidebar was collapsed. */
            [data-testid="stToolbar"] {{
                visibility: hidden !important;
                height: 0 !important;
                min-height: 0 !important;
                overflow: visible !important;
            }}
            [data-testid="stToolbar"] * {{
                visibility: hidden !important;
            }}
            .stDeployButton {{ display: none; }}
            #MainMenu {{ visibility: hidden; }}
            footer {{ visibility: hidden; }}

            .stApp {{ background: var(--t-bg); color: var(--t-ink); }}

            /* Compact layout baseline */
            .block-container {{
                padding-top: 0.25rem !important;
                padding-left: 1.5rem !important;
                padding-right: 1.0rem !important;
                padding-bottom: 1.25rem !important;
                max-width: 1700px !important;
            }}
            [data-testid="stVerticalBlock"] {{ gap: 0.55rem !important; }}
            [data-testid="stHorizontalBlock"] {{ gap: 0.6rem !important; }}
            [data-testid="stVerticalBlockBorderWrapper"] {{ gap: 0.55rem !important; }}

            /* Collapse st.divider() - it ships ~2rem of vertical margin */
            [data-testid="stDivider"], .stApp hr {{
                margin: 0.45rem 0 !important;
            }}

            h1, h2, h3, h4 {{
                color: var(--t-ink);
                margin: 0.2rem 0 0.15rem 0 !important;
                line-height: 1.25 !important;
            }}
            h1 {{ font-size: calc(var(--t-size-section) + 6px) !important; }}
            h2 {{ font-size: calc(var(--t-size-section) + 1px) !important; }}
            h3 {{ font-size: var(--t-size-section) !important; }}
            .stApp p, .stApp li {{
                font-size: var(--t-size-body);
                line-height: 1.45;
                margin-bottom: 0.2rem;
            }}
            /* Expanders */
            [data-testid="stExpander"] {{
                border: 1px solid var(--t-line) !important;
                border-radius: 4px;
                background: var(--t-bg);
            }}
            [data-testid="stExpander"] summary {{
                font-family: var(--t-sans);
                font-size: var(--t-size-control) !important;
                line-height: 1.35 !important;
                text-transform: none;
                letter-spacing: normal !important;
                color: var(--t-ink-3);
            }}
            [data-testid="stExpander"] [data-testid="stVerticalBlock"] {{
                gap: 0.7rem !important;
            }}
            [data-testid="stExpander"] [data-testid="stCaptionContainer"] p {{
                color: var(--t-ink-3) !important;
                font-size: var(--t-size-supporting) !important;
                line-height: 1.45 !important;
                margin-bottom: 0.35rem !important;
            }}

            /* Sidebar */
            section[data-testid="stSidebar"] {{
                background: var(--t-bg);
                border-right: 1px solid var(--t-line);
                padding-top: 0.5rem !important;
            }}
            section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
                gap: 0.2rem !important;
            }}
            section[data-testid="stSidebar"] .element-container,
            section[data-testid="stSidebar"] [data-testid="stElementContainer"] {{
                margin-bottom: 0.15rem !important;
            }}
            section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
            section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
            section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] strong {{
                color: var(--t-ink-3) !important;
                font-size: var(--t-size-supporting) !important;
                line-height: 1.45 !important;
                margin-bottom: 0.08rem !important;
            }}
            section[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
            section[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
            section[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] *,
            section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * {{
                color: var(--t-ink-2) !important;
                font-size: var(--t-size-supporting) !important;
                line-height: 1.35 !important;
            }}

            .stPlotlyChart {{ margin: 0.2rem 0 !important; }}

            /* Checkboxes */
            [data-testid="stCheckbox"] {{
                margin-bottom: 0.45rem !important;
            }}
            [data-testid="stCheckbox"] label {{
                min-height: 28px !important;
                align-items: center !important;
            }}
            [data-testid="stCheckbox"] label p {{
                color: var(--t-ink) !important;
                font-size: var(--t-size-control) !important;
                line-height: 1.45 !important;
            }}

            /* Tabular data */
            .dataframe, [data-testid="stDataFrame"], [data-testid="stTable"] {{
                font-family: var(--t-sans) !important;
                font-size: var(--t-size-supporting) !important;
                font-feature-settings: "tnum";
            }}
            .dataframe thead th,
            [data-testid="stTable"] thead th,
            [data-testid="stDataFrame"] [role="columnheader"],
            [data-testid="stDataFrame"] [role="columnheader"] *,
            [data-testid="stDataFrame"] [data-testid*="columnHeader"],
            [data-testid="stDataFrame"] [data-testid*="columnHeader"] * {{
                color: var(--t-ink) !important;
                -webkit-text-fill-color: var(--t-ink) !important;
                font-weight: 600 !important;
            }}

            /* Code/stat cards */
            [data-testid="stCode"] pre,
            [data-testid="stCode"] code,
            .stCode pre,
            .stCode code,
            [data-testid="stCodeBlock"] pre,
            [data-testid="stCodeBlock"] code,
            .stCodeBlock pre,
            .stCodeBlock code {{
                font-family: var(--t-mono) !important;
                font-size: var(--t-size-section) !important;
                line-height: 1.65 !important;
                font-feature-settings: "tnum";
                color: var(--t-ink) !important;
            }}
            [data-testid="stCode"] pre *,
            [data-testid="stCode"] code *,
            .stCode pre *,
            .stCode code *,
            [data-testid="stCodeBlock"] pre *,
            [data-testid="stCodeBlock"] code *,
            .stCodeBlock pre *,
            .stCodeBlock code * {{
                color: var(--t-ink) !important;
            }}
            [data-testid="stCode"] pre,
            .stCode pre,
            [data-testid="stCodeBlock"] pre,
            .stCodeBlock pre {{
                padding: 14px 16px !important;
                border-radius: 6px !important;
                background: var(--code-bg) !important;
                border: 1px solid var(--code-border) !important;
            }}

            div[data-testid="stMetric"] {{
                background: var(--t-bg-2);
                padding: 0.3rem 0.55rem;
                border-radius: 4px;
                border: 0;
            }}
            div[data-testid="stMetric"] label {{
                font-family: var(--t-sans);
                font-size: var(--t-size-supporting) !important;
                text-transform: none;
                letter-spacing: normal;
                color: var(--t-ink-3) !important;
                margin-bottom: 0 !important;
            }}
            div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
                font-family: var(--t-mono);
                font-feature-settings: "tnum";
                font-weight: 600;
                font-size: calc(var(--t-size-section) + 4px) !important;
                line-height: 1.15 !important;
                color: var(--t-ink) !important;
            }}
            div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
                font-family: var(--t-mono);
                color: var(--t-ink-2) !important;
                padding-top: 0 !important;
            }}
            /* Collapse the phantom delta row metrics reserve when no delta. */
            div[data-testid="stMetric"] [data-testid="stMetricDelta"]:empty {{
                display: none !important;
            }}

            /* Buttons */
            .stButton > button,
            [data-testid="stButton"] > button,
            [data-testid="stButton"] button {{
                font-family: var(--t-sans);
                font-size: var(--t-size-control) !important;
                font-weight: 500;
                min-height: 38px !important;
                padding: 5px 14px !important;
                border: 1px solid var(--t-line-2) !important;
                background: var(--t-bg) !important;
                color: var(--t-ink) !important;
                border-radius: 4px;
            }}
            [data-testid="stButton"] button * {{
                color: inherit !important;
            }}
            .stButton > button:hover,
            [data-testid="stButton"] > button:hover,
            [data-testid="stButton"] button:hover {{
                border-color: var(--t-accent) !important;
                color: var(--t-accent);
            }}
            .stButton > button[kind="primary"],
            [data-testid="stButton"] > button[kind="primary"],
            [data-testid="stButton"] button[kind="primary"] {{
                background: var(--t-accent);
                color: var(--primary-button-text);
                border-color: var(--t-accent) !important;
            }}

            /* BaseWeb inputs and multiselect tags */
            [data-baseweb="input"] input,
            [data-baseweb="textarea"] textarea,
            [data-baseweb="select"] {{
                font-size: var(--t-size-control) !important;
                line-height: 1.35 !important;
            }}
            [data-baseweb="tag"] {{
                min-height: 28px !important;
                font-size: var(--t-size-supporting) !important;
                line-height: 1.25 !important;
            }}
            [data-baseweb="tag"] span {{
                font-size: var(--t-size-supporting) !important;
                line-height: 1.25 !important;
            }}
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 2px;
                border-bottom: 1px solid var(--t-line);
            }}
            .stTabs [data-baseweb="tab"] {{
                font-family: var(--t-sans) !important;
                font-size: var(--t-size-control) !important;
                text-transform: none;
                letter-spacing: normal;
                color: var(--t-ink-3) !important;
                padding: 6px 14px !important;
                border-bottom: 2px solid transparent !important;
            }}
            .stTabs [aria-selected="true"] {{
                color: var(--t-ink) !important;
                border-bottom-color: var(--t-accent) !important;
            }}

            /*
             * Main workspace navigation only. Streamlit 1.52/1.55 emit the
             * widget key as .st-key-main_section_nav_widget, a radiogroup, and
             * BaseWeb radio markers inside each label. Keep every selector
             * under that key so sidebar/content radios retain native styling.
             */
            .st-key-main_section_nav_widget {{
                position: relative;
                max-width: 100%;
                overflow-x: auto !important;
                overflow-y: hidden !important;
                overscroll-behavior-x: contain;
                scrollbar-width: thin;
                scrollbar-color: var(--t-line-2) transparent;
                scrollbar-gutter: stable;
                margin: 0 0 0.65rem 0;
                padding-top: 2px;
            }}
            .st-key-main_section_nav_widget::-webkit-scrollbar {{
                display: block;
                height: 6px;
            }}
            .st-key-main_section_nav_widget::-webkit-scrollbar-track {{
                background: transparent;
            }}
            .st-key-main_section_nav_widget::-webkit-scrollbar-thumb {{
                background: var(--t-line-2);
                border-radius: 999px;
            }}
            .st-key-main_section_nav_widget div[role="radiogroup"] {{
                display: inline-flex !important;
                flex-wrap: nowrap !important;
                align-items: flex-end !important;
                gap: 0 !important;
                width: max-content !important;
                min-width: max-content !important;
                border-bottom: 1px solid var(--t-line-2);
                padding: 2px 1px 0;
            }}
            .st-key-main_section_nav_widget div[role="radiogroup"] > label {{
                position: relative;
                z-index: 0;
                flex: 0 0 auto !important;
                margin: 0 0 -1px 0 !important;
                padding: 9px 14px 10px !important;
                border: 1px solid var(--t-line) !important;
                border-bottom-color: var(--t-line-2) !important;
                border-radius: 6px 6px 0 0 !important;
                background: var(--t-bg-2) !important;
                color: var(--t-ink-3) !important;
                cursor: pointer;
                white-space: nowrap !important;
                font-family: var(--t-sans) !important;
                font-size: var(--t-size-control) !important;
                font-weight: 500 !important;
                letter-spacing: normal !important;
                line-height: 1.2 !important;
                text-transform: none !important;
            }}
            .st-key-main_section_nav_widget div[role="radiogroup"] > label + label {{
                margin-left: -1px !important;
            }}
            .st-key-main_section_nav_widget div[role="radiogroup"] > label p,
            .st-key-main_section_nav_widget div[role="radiogroup"] > label span {{
                margin: 0 !important;
                color: var(--t-ink-3) !important;
                font-family: var(--t-sans) !important;
                font-size: var(--t-size-control) !important;
                font-weight: inherit !important;
                letter-spacing: normal !important;
                line-height: 1.2 !important;
                text-transform: none !important;
                white-space: nowrap !important;
            }}
            .st-key-main_section_nav_widget div[role="radiogroup"] > label > div:first-child,
            .st-key-main_section_nav_widget [data-baseweb="radio"] > div:first-child {{
                display: none !important;
            }}
            .st-key-main_section_nav_widget div[role="radiogroup"] > label:hover {{
                border-color: var(--t-line-2) !important;
                color: var(--t-ink-2) !important;
            }}
            .st-key-main_section_nav_widget div[role="radiogroup"] > label:has(input:checked) {{
                z-index: 1;
                border-color: var(--t-line-2) !important;
                border-bottom-color: var(--t-bg) !important;
                background: var(--t-bg) !important;
                color: var(--t-ink) !important;
                font-weight: 700 !important;
                box-shadow: inset 0 3px 0 var(--t-accent);
            }}
            .st-key-main_section_nav_widget div[role="radiogroup"] > label:hover p,
            .st-key-main_section_nav_widget div[role="radiogroup"] > label:hover span {{
                color: var(--t-ink-2) !important;
            }}
            .st-key-main_section_nav_widget div[role="radiogroup"] > label:has(input:checked) p,
            .st-key-main_section_nav_widget div[role="radiogroup"] > label:has(input:checked) span {{
                color: var(--t-ink) !important;
            }}
            .st-key-main_section_nav_widget div[role="radiogroup"] > label:focus-within,
            .st-key-main_section_nav_widget div[role="radiogroup"] > label:has(input:focus-visible) {{
                z-index: 2;
                outline: 2px solid var(--t-accent) !important;
                outline-offset: -3px;
            }}

            /*
             * Phase 2 workspace subnavigation. Only the Results,
             * Uncertainty, and Export format radio widget keys opt in.
             */
            :is(.st-key-results_subview_nav, .st-key-uncertainty_subview,
                .st-key-uncertainty_configure_pane, .st-key-export_format_radio) {{
                max-width: 100%;
                overflow-x: auto;
                overflow-y: hidden;
                scrollbar-width: thin;
                scrollbar-color: var(--t-line-2) transparent;
                margin: 0.1rem 0 0.85rem;
                padding-bottom: 3px;
            }}
            :is(.st-key-results_subview_nav, .st-key-uncertainty_subview,
                .st-key-uncertainty_configure_pane, .st-key-export_format_radio)
            :is([data-testid="stButtonGroup"], [data-baseweb="button-group"]) {{
                display: inline-flex !important;
                width: max-content !important;
                min-width: max-content !important;
                gap: 0 !important;
                padding: 2px;
                border: 1px solid var(--t-line);
                border-radius: 7px;
                background: var(--t-bg-2);
            }}
            :is(.st-key-results_subview_nav, .st-key-uncertainty_subview,
                .st-key-uncertainty_configure_pane, .st-key-export_format_radio)
            :is([data-testid="stButtonGroup"], [data-baseweb="button-group"]) button {{
                min-height: 32px !important;
                padding: 5px 11px !important;
                border: 0 !important;
                border-radius: 5px !important;
                background: transparent !important;
                color: var(--t-ink-3) !important;
                font-family: var(--t-sans) !important;
                font-size: var(--t-size-supporting) !important;
                font-weight: 500 !important;
                line-height: 1.2 !important;
                white-space: nowrap !important;
                box-shadow: none !important;
            }}
            :is(.st-key-results_subview_nav, .st-key-uncertainty_subview,
                .st-key-uncertainty_configure_pane, .st-key-export_format_radio)
            :is([data-testid="stButtonGroup"], [data-baseweb="button-group"])
            button[aria-pressed="true"],
            :is(.st-key-results_subview_nav, .st-key-uncertainty_subview,
                .st-key-uncertainty_configure_pane, .st-key-export_format_radio)
            :is([data-testid="stButtonGroup"], [data-baseweb="button-group"])
            button[aria-checked="true"],
            :is(.st-key-results_subview_nav, .st-key-uncertainty_subview,
                .st-key-uncertainty_configure_pane, .st-key-export_format_radio)
            :is([data-testid="stButtonGroup"], [data-baseweb="button-group"])
            button[data-selected="true"] {{
                background: var(--t-bg) !important;
                color: var(--t-ink) !important;
                font-weight: 650 !important;
                box-shadow: 0 0 0 1px var(--t-line-2) !important;
            }}
            :is(.st-key-results_subview_nav, .st-key-uncertainty_subview,
                .st-key-uncertainty_configure_pane, .st-key-export_format_radio)
            :is([data-testid="stButtonGroup"], [data-baseweb="button-group"])
            button:focus-visible {{
                outline: 2px solid var(--t-accent) !important;
                outline-offset: 1px;
            }}

            /* Streamlit fallback when segmented_control is unavailable. */
            :is(.st-key-results_subview_nav, .st-key-uncertainty_subview,
                .st-key-uncertainty_configure_pane, .st-key-export_format_radio)
            div[role="radiogroup"] {{
                display: inline-flex !important;
                flex-wrap: nowrap !important;
                width: max-content !important;
                min-width: max-content !important;
                gap: 0 !important;
                padding: 2px;
                border: 1px solid var(--t-line);
                border-radius: 7px;
                background: var(--t-bg-2);
            }}
            :is(.st-key-results_subview_nav, .st-key-uncertainty_subview,
                .st-key-uncertainty_configure_pane, .st-key-export_format_radio)
            div[role="radiogroup"] > label {{
                margin: 0 !important;
                padding: 7px 11px !important;
                border-radius: 5px !important;
                color: var(--t-ink-3) !important;
                font-size: var(--t-size-supporting) !important;
                line-height: 1.2 !important;
                white-space: nowrap !important;
            }}
            :is(.st-key-results_subview_nav, .st-key-uncertainty_subview,
                .st-key-uncertainty_configure_pane, .st-key-export_format_radio)
            div[role="radiogroup"] > label > div:first-child {{
                display: none !important;
            }}
            :is(.st-key-results_subview_nav, .st-key-uncertainty_subview,
                .st-key-uncertainty_configure_pane, .st-key-export_format_radio)
            div[role="radiogroup"] > label:has(input:checked) {{
                background: var(--t-bg) !important;
                color: var(--t-ink) !important;
                font-weight: 650 !important;
                box-shadow: 0 0 0 1px var(--t-line-2);
            }}
            :is(.st-key-results_subview_nav, .st-key-uncertainty_subview,
                .st-key-uncertainty_configure_pane, .st-key-export_format_radio)
            div[role="radiogroup"] > label:focus-within {{
                outline: 2px solid var(--t-accent) !important;
                outline-offset: 1px;
            }}

            /* Results controls use ordinary Streamlit radios rather than the
             * tab-like navigation treatment.  BaseWeb puts the visible text
             * below an intermediate div with its own light-theme colour, so
             * `inherit` is insufficient after an in-app switch to dark mode.
             * Keep this scoped to the three Results widget keys. */
            :is(.st-key-sticky_results_sample_axis,
                .st-key-overview_error_mode,
                .st-key-overview_delta_error_mode)
            div[role="radiogroup"] > label,
            :is(.st-key-sticky_results_sample_axis,
                .st-key-overview_error_mode,
                .st-key-overview_delta_error_mode)
            div[role="radiogroup"] > label p,
            :is(.st-key-sticky_results_sample_axis,
                .st-key-overview_error_mode,
                .st-key-overview_delta_error_mode)
            div[role="radiogroup"] > label span {{
                color: var(--t-ink) !important;
            }}

            /* Shared presentation-only workspace primitives. */
            .ws-panel-heading {{ margin: 0 0 0.7rem; }}
            .ws-panel-eyebrow {{
                color: var(--t-ink-4);
                font-family: var(--t-sans);
                font-size: var(--t-size-supporting);
                letter-spacing: normal;
                text-transform: none;
            }}
            .ws-panel-title {{
                color: var(--t-ink);
                font-size: var(--t-size-section);
                font-weight: 650;
                line-height: 1.3;
            }}
            .ws-panel-subtitle,
            .ws-next-step-reason {{
                color: var(--t-ink-3);
                font-size: var(--t-size-supporting);
                line-height: 1.45;
            }}
            .ws-status-chip {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 3px 8px;
                border: 1px solid var(--t-line-2);
                border-radius: 999px;
                background: var(--t-bg-2);
                color: var(--t-ink-2);
                font-family: var(--t-sans);
                font-size: var(--t-size-supporting);
            }}
            .ws-status-dot {{
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background: var(--t-ink-4);
            }}
            .ws-status-chip[data-tone="positive"] {{
                background: var(--flag-ok-bg);
                color: var(--flag-ok-fg);
            }}
            .ws-status-chip[data-tone="positive"] .ws-status-dot {{
                background: var(--flag-ok-fg);
            }}
            .ws-status-chip[data-tone="warning"] {{
                background: var(--flag-elev-bg);
                color: var(--flag-elev-fg);
            }}
            .ws-status-chip[data-tone="warning"] .ws-status-dot {{
                background: var(--flag-elev-fg);
            }}
            .ws-status-chip[data-tone="critical"] {{
                background: var(--flag-high-bg);
                color: var(--flag-high-fg);
            }}
            .ws-status-chip[data-tone="critical"] .ws-status-dot {{
                background: var(--flag-high-fg);
            }}
            .ws-next-step-title {{
                color: var(--t-ink);
                font-size: var(--t-size-body);
                font-weight: 650;
                line-height: 1.3;
                margin-bottom: 2px;
            }}
            .ws-next-step {{ margin-bottom: 0.55rem; }}
            .ws-metadata {{
                display: flex;
                flex-wrap: wrap;
                gap: 6px 16px;
                margin: 0.1rem 0 0.7rem;
                color: var(--t-ink-2);
                font-family: var(--t-mono);
                font-size: var(--t-size-supporting);
            }}
            .ws-metadata-item {{ display: inline-flex; gap: 5px; }}
            .ws-metadata-label {{
                color: var(--t-ink-3);
                font-family: var(--t-sans);
                letter-spacing: normal;
                text-transform: none;
            }}
            .ws-metadata-value {{ color: var(--t-ink-2); font-family: var(--t-mono); }}

            /* Phase 3 Session Configuration workflow. */
            .ws-workflow-progress {{
                display: grid;
                grid-template-columns: auto 1fr auto 1fr auto;
                align-items: center;
                gap: 10px;
                margin: 0.1rem 0 0.75rem;
            }}
            .ws-workflow-step {{
                display: inline-flex;
                align-items: center;
                gap: 7px;
                color: var(--t-ink-4);
                font-size: var(--t-size-supporting);
                font-weight: 600;
                white-space: nowrap;
            }}
            .ws-workflow-number {{
                display: inline-grid;
                place-items: center;
                width: 24px;
                height: 24px;
                border: 1px solid var(--t-line-2);
                border-radius: 50%;
                background: var(--t-bg);
                color: var(--t-ink-4);
                font-family: var(--t-mono);
                font-size: var(--t-size-supporting);
                font-weight: 700;
            }}
            .ws-workflow-step[data-state="done"],
            .ws-workflow-step[data-state="current"] {{
                color: var(--t-ink);
            }}
            .ws-workflow-step[data-state="done"] .ws-workflow-number,
            .ws-workflow-step[data-state="current"] .ws-workflow-number {{
                border-color: var(--t-accent);
                background: var(--t-accent);
                color: var(--t-bg);
            }}
            .ws-workflow-line {{
                height: 2px;
                background: var(--t-line);
            }}
            .ws-workflow-line[data-state="done"] {{ background: var(--t-accent); }}
            .st-key-session_classify [data-testid="stVerticalBlockBorderWrapper"],
            .st-key-session_configure [data-testid="stVerticalBlockBorderWrapper"],
            .st-key-session_run [data-testid="stVerticalBlockBorderWrapper"] {{
                border-color: var(--t-line-2);
            }}
            .st-key-session_classify [data-testid="stForm"] {{
                border: 0 !important;
                padding: 0 !important;
            }}
            .st-key-session_configure [data-testid="stExpander"] {{
                border-width: 1px 0 0 !important;
                border-radius: 0 !important;
            }}
            /* Phase 4: Configure scrolls with the page, never inside its
               own card; its settings are full-width groups. */
            .ws-group-heading {{
                margin: 0.85rem 0 0.15rem;
                padding-top: 0.55rem;
                border-top: 1px solid var(--t-line);
                color: var(--t-ink);
                font-family: var(--t-sans);
                font-size: var(--t-size-body, 14px);
                font-weight: 600;
            }}
            .st-key-session_run button[kind="primary"] {{
                min-height: 42px !important;
                max-width: 32rem;
                font-weight: 700 !important;
            }}

            /* Phase 4: one aligned Inspector statistics table. */
            .ws-stats-table-wrap {{ overflow-x: auto; margin: 0 0 0.5rem; }}
            .ws-stats-table {{
                width: 100%;
                border-collapse: collapse;
                font-family: var(--t-sans);
                font-size: {PLOTLY_BASE_FONT_SIZE + 3}px;
                color: var(--t-ink);
                background: transparent;
            }}
            .ws-stats-table th,
            .ws-stats-table td {{
                padding: 5px 12px;
                font-family: var(--t-sans) !important;
                font-size: {PLOTLY_BASE_FONT_SIZE + 3}px !important;
                border: 0;
                border-bottom: 1px solid var(--t-line);
                background: transparent;
                text-align: left;
                vertical-align: baseline;
            }}
            .ws-stats-table thead th {{
                color: var(--t-ink-2);
                font-weight: 600;
                border-bottom-color: var(--t-line-2);
            }}
            .ws-stats-table tbody th {{
                color: var(--t-ink-3);
                font-weight: 500;
                white-space: nowrap;
            }}
            .ws-stats-table td {{
                font-variant-numeric: tabular-nums;
                white-space: nowrap;
            }}
            .ws-stats-table caption {{
                caption-side: bottom;
                padding-top: 6px;
                text-align: left;
                color: var(--t-ink-3);
                font-size: var(--t-size-supporting);
            }}

            /* Phase 4: deliberate stacking. Container queries measure the
               workspace, so an open sidebar counts as reduced width. Only the
               keyed row is stacked; columns nested inside keep their layout. */
            @container traceiso-workspace (max-width: 900px) {{
                .st-key-session_layout > [data-testid="stHorizontalBlock"],
                .st-key-session_layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"],
                .st-key-inspector_selectors > [data-testid="stHorizontalBlock"],
                .st-key-inspector_selectors > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"],
                .st-key-inspector_controls > [data-testid="stHorizontalBlock"],
                .st-key-inspector_controls > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"],
                .st-key-export_options_layout > [data-testid="stHorizontalBlock"],
                .st-key-export_options_layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {{
                    flex-wrap: wrap !important;
                }}
                .st-key-session_layout > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
                .st-key-session_layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
                .st-key-inspector_selectors > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
                .st-key-inspector_selectors > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
                .st-key-inspector_controls > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
                .st-key-inspector_controls > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
                .st-key-export_options_layout > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
                .st-key-export_options_layout > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
                    flex: 1 1 100% !important;
                    width: 100% !important;
                    min-width: 100% !important;
                }}
            }}
            /* Plots need more room than forms: stack them earlier. */
            @container traceiso-workspace (max-width: 1000px) {{
                .st-key-inspector_plots > [data-testid="stHorizontalBlock"],
                .st-key-inspector_plots > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] {{
                    flex-wrap: wrap !important;
                }}
                .st-key-inspector_plots > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
                .st-key-inspector_plots > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
                    flex: 1 1 100% !important;
                    width: 100% !important;
                    min-width: 100% !important;
                }}
            }}

            .block-container {{
                container-type: inline-size;
                container-name: traceiso-workspace;
            }}
            .main-nav-overflow-cue {{ display: none; }}
            @container traceiso-workspace (max-width: 860px) {{
                .main-nav-overflow-cue {{
                    display: block;
                    margin: -0.45rem 0 0.65rem;
                    color: var(--t-ink-3);
                    font-size: var(--t-size-supporting);
                    text-align: right;
                }}
            }}
            @media (max-width: 800px) {{
                .block-container {{
                    padding-left: 0.8rem !important;
                    padding-right: 0.8rem !important;
                }}
                .ws-workflow-progress {{
                    grid-template-columns: auto;
                    gap: 6px;
                }}
                .ws-workflow-line {{ display: none; }}
            }}

            /* Markdown utility classes */
            .pl-kicker {{
                font-family: var(--t-sans); font-size: var(--t-size-supporting);
                letter-spacing: normal;
                color: var(--t-ink-4); margin-bottom: 1px;
            }}
            .pl-title {{
                font-size: var(--t-size-section); font-weight: 600;
                color: var(--t-ink); margin: 0 0 4px 0;
            }}
            .pl-sub {{ color: var(--t-ink-3); font-size: var(--t-size-supporting); margin: 0 0 12px 0; }}
            .pl-strip {{
                display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
                padding: 8px 0 9px 0; border-bottom: 1px solid var(--t-line);
                font-family: var(--t-sans); font-size: var(--t-size-supporting);
                font-feature-settings: "tnum";
                color: var(--t-ink-2); margin-bottom: 12px;
            }}
            .pl-strip b {{ color: var(--t-ink); font-weight: 600; }}
            .pl-strip .sep {{ color: var(--t-ink-4); user-select: none; }}
            .pl-status {{
                display: inline-flex; align-items: center; gap: 6px;
                font-family: var(--t-sans); font-size: var(--t-size-supporting);
                padding: 3px 9px; border-radius: 999px;
                border: 1px solid var(--t-line-2);
                background: var(--t-bg); color: var(--t-ink-2);
            }}
            .pl-status .dot {{ width: 7px; height: 7px; border-radius: 50%; }}
            .pl-status[data-state="fresh"] .dot {{ background: {self.palette.success}; }}
            .pl-status[data-state="stale"] .dot {{ background: {self.palette.warning}; }}
            .pl-status[data-state="empty"] .dot {{ background: var(--t-ink-4); }}
            .pl-stat {{
                border: 1px solid var(--t-line); border-radius: 4px;
                background: var(--t-bg); padding: 10px 14px;
            }}
            .pl-stat .l {{
                font-family: var(--t-sans); font-size: var(--t-size-supporting);
                letter-spacing: normal;
                color: var(--t-ink-3);
            }}
            .pl-stat .v {{
                font-family: var(--t-mono); font-feature-settings: "tnum";
                font-size: calc(var(--t-size-section) + 4px); font-weight: 600;
                color: var(--t-ink); margin-top: 2px;
            }}
            .pl-stat .s {{ font-family: var(--t-sans); font-size: var(--t-size-supporting); color: var(--t-ink-3); }}
            .pl-hr {{ border: none; border-top: 1px solid var(--t-line); margin: 12px 0; }}

            /* Sample-type and flag badges */
            .badge-std, .badge-smp, .badge-blk, .badge-exc,
            .pl-flag-ok, .pl-flag-elev, .pl-flag-high,
            .pl-flag-interf, .pl-flag-unav {{
                display: inline-block;
                font-family: var(--t-mono);
                font-size: var(--t-size-supporting);
                font-weight: 600;
                padding: 2px 7px;
                border-radius: 3px;
                text-transform: uppercase;
                letter-spacing: 0.04em;
            }}
            .badge-std {{ background: var(--std-soft); color: var(--t-ink); border: 1px solid var(--std); }}
            .badge-smp {{ background: var(--smp-soft); color: var(--t-ink); border: 1px solid var(--smp); }}
            .badge-blk {{ background: var(--blk-soft); color: var(--t-ink); border: 1px solid var(--blk); }}
            .badge-exc {{ background: var(--flag-elev-bg); color: var(--flag-elev-fg); }}
            .pl-flag-ok {{ background: var(--flag-ok-bg); color: var(--flag-ok-fg); }}
            .pl-flag-elev {{ background: var(--flag-elev-bg); color: var(--flag-elev-fg); }}
            .pl-flag-high {{ background: var(--flag-high-bg); color: var(--flag-high-fg); }}
            .pl-flag-interf {{ background: var(--flag-interf-bg); color: var(--flag-interf-fg); }}
            .pl-flag-unav {{ background: var(--flag-unav-bg); color: var(--flag-unav-fg); }}
        </style>
        """

    @staticmethod
    def normalize_sample_type(sample_type: str) -> str:
        """Collapse sample labels to canonical display codes."""
        stype = (sample_type or "").upper()
        if stype in ("STD", "STANDARD"):
            return "STD"
        if stype in ("BLK", "BLANK"):
            return "BLK"
        if stype == "QC":
            return "QC"
        return "SMP"

    def sample_color(self, sample_type: str, *, excluded: bool = False) -> str:
        """Get the central display colour for a sample type."""
        if excluded:
            return self.palette.layer_excluded

        stype = self.normalize_sample_type(sample_type)
        if stype == "STD":
            return self.palette.sample_std
        if stype == "BLK":
            return self.palette.sample_blk
        if stype == "QC":
            return self.palette.sample_qc
        return self.palette.sample_smp

    def sample_symbol(self, sample_type: str) -> str:
        """Return the canonical marker symbol for a sample type."""
        stype = self.normalize_sample_type(sample_type)
        if stype == "STD":
            return "diamond"
        if stype == "BLK":
            return "square"
        if stype == "QC":
            return "triangle-up"
        return "circle"


_theme_manager: ThemeManager | None = None


def get_theme(theme_name: str | None = None) -> ThemeManager:
    """Get the per-session theme manager (item 70).

    Stores the ``ThemeManager`` instance in ``st.session_state`` so that each
    concurrent Streamlit session keeps its own object and is not affected by
    another session changing ``_theme_manager``.  Falls back to a
    process-level singleton when Streamlit session state is not available
    (e.g. in unit tests or tools).
    """
    if theme_name is None:
        try:
            theme_name = str(st.session_state.get(SESSION_KEY_THEME_NAME, "light"))
        except Exception:
            theme_name = "light"

    _SESSION_KEY = "_traceiso_theme_manager"

    # --- per-session path (Streamlit running) ---
    try:
        current = st.session_state.get(_SESSION_KEY)
        if current is None or current.theme_name != theme_name:
            current = ThemeManager(theme_name)
            st.session_state[_SESSION_KEY] = current
        return current  # type: ignore[return-value]
    except Exception:
        pass

    # --- fallback: process-level singleton (tests / standalone tools) ---
    global _theme_manager
    if _theme_manager is None or _theme_manager.theme_name != theme_name:
        _theme_manager = ThemeManager(theme_name)
    return _theme_manager

"""Publication controls and display-only reference overlays for grouped plots."""

from html import escape

import numpy as np
import streamlit as st


def _number(settings, field, label, prefix, default, **kwargs):
    settings[field] = st.number_input(
        label, value=settings.get(field, default), key=f"{prefix}_{field}", **kwargs,
    )
    return settings[field]


def render_group_publication_controls(groups, ratio, *, delta_enabled, run_order, has_blanks):
    """Store ratio/delta axes and references separately; retain absent widgets' values."""
    from ui.components.sample_plot_groups import GROUPS_KEY

    settings = st.session_state[GROUPS_KEY].setdefault("publication", {})
    with st.expander("Publication sizing, axes and references", expanded=False):
        st.caption("These settings apply only to the grouped figures. Physical dimensions and type sizes apply to the enlarged export view.")
        cols = st.columns(3)
        with cols[0]:
            _number(settings, "width_mm", "Figure width (mm)", "group_pub", 90.0, min_value=50.0, max_value=300.0, step=5.0)
            _number(settings, "height_mm", "Figure height (mm)", "group_pub", 70.0, min_value=40.0, max_value=300.0, step=5.0)
        with cols[1]:
            _number(settings, "font_pt", "Font size (pt)", "group_pub", 9.0, min_value=6.0, max_value=24.0, step=0.5)
            _number(settings, "marker_pt", "Marker size (pt)", "group_pub", 5.0, min_value=2.0, max_value=16.0, step=0.5)
        with cols[2]:
            _number(settings, "error_pt", "Error-bar thickness (pt)", "group_pub", 0.75, min_value=0.25, max_value=3.0, step=0.25)
            _number(settings, "cap_pt", "Error-bar cap width (pt)", "group_pub", 3.0, min_value=0.0, max_value=10.0, step=0.5)
        settings["format"] = st.selectbox("Export format", ["svg", "png"],
            index=["svg", "png"].index(settings.get("format", "svg")), key="group_pub_format")
        _number(settings, "dpi", "PNG resolution (dpi)", "group_pub", 300, min_value=150, max_value=1200, step=150)

        modes = ["ratio", "delta"] if delta_enabled else ["ratio"]
        for mode in modes:
            mode_settings = settings.setdefault("views", {}).setdefault(f"{ratio}:{mode}", {})
            prefix = f"group_pub_{ratio}_{mode}"
            st.markdown(f"**{ratio} — {mode} axes**")
            axes = [("yaxis", "Y (‰)" if mode == "delta" else "Y (ratio)")]
            if run_order:
                axes.insert(0, ("xaxis", "X (run order)"))
            else:
                st.caption("The sample-name axis is categorical; numeric X limits and steps apply only in Run order mode.")
            if has_blanks and mode == "ratio":
                axes.append(("yaxis2", "Y (blank ratio)"))
            for axis, label in axes:
                opts = mode_settings.setdefault(axis, {})
                key = f"{prefix}_{axis}"
                opts["manual"] = st.checkbox(f"Set {label} limits", value=opts.get("manual", False), key=f"{key}_manual")
                if opts["manual"]:
                    lo, hi = st.columns(2)
                    with lo:
                        _number(opts, "min", f"{label} minimum", key, None, format="%.8g")
                    with hi:
                        _number(opts, "max", f"{label} maximum", key, None, format="%.8g")
                    if not _valid_range(opts):
                        st.warning("Enter finite limits with minimum below maximum; automatic limits are used until then.")
                opts["custom_ticks"] = st.checkbox(f"Set {label} ticks", value=opts.get("custom_ticks", False), key=f"{key}_ticks")
                if opts["custom_ticks"]:
                    a, b = st.columns(2)
                    with a:
                        _number(opts, "step", f"{label} tick step", key, None, min_value=0.0, format="%.8g")
                    with b:
                        _number(opts, "decimals", f"{label} decimal places", key, 6 if axis != "xaxis" else 0, min_value=0, max_value=12)
                    if not _positive(opts.get("step")):
                        st.warning("Enter a positive tick step; automatic tick spacing is used until then.")

            st.markdown(f"**{ratio} — {mode} group references**")
            st.caption("Enter display reference values in the plotted units. The optional ± band is an absolute half-width; state its meaning and source. These entries do not change calibration.")
            for group in groups:
                ref = group.setdefault("references", {}).setdefault(f"{ratio}:{mode}", {})
                key = f"{prefix}_{group['id']}_ref"
                ref["enabled"] = st.checkbox(f"Reference: {group.get('legend_label') or group['name']}",
                    value=ref.get("enabled", False), key=f"{key}_enabled")
                if ref["enabled"]:
                    a, b = st.columns(2)
                    with a:
                        _number(ref, "value", "Reference value", key, None, format="%.9g")
                    with b:
                        _number(ref, "band", "Reference ± half-width (optional)", key, None, min_value=0.0, format="%.9g")
                    ref["description"] = st.text_input("Reference source / band definition", value=ref.get("description", ""),
                        key=f"{key}_description", placeholder="Source; e.g. expanded uncertainty, k = 2")
    return settings


def _positive(value):
    return value is not None and np.isfinite(value) and value > 0


def _valid_range(opts):
    lo, hi = opts.get("min"), opts.get("max")
    return lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi) and lo < hi


def apply_group_axes(fig, settings, ratio, mode):
    for axis, opts in settings.get("views", {}).get(f"{ratio}:{mode}", {}).items():
        if axis == "xaxis" and fig.layout.xaxis.type == "category":
            continue
        if axis == "yaxis2" and not fig.layout.yaxis2.visible:
            continue
        changes = {}
        if opts.get("manual") and _valid_range(opts):
            changes.update(range=[opts["min"], opts["max"]], autorange=False)
        if opts.get("custom_ticks"):
            changes["tickformat"] = f".{opts.get('decimals', 6)}f"
            if _positive(opts.get("step")):
                changes.update(dtick=opts["step"], tickmode="linear", tick0=opts.get("min") if opts.get("manual") and _valid_range(opts) else 0)
        fig.update_layout(**{axis: changes})


def add_group_references(fig, groups, ratio, mode):
    """Draw each group's reference only across its visible span, on its own axis."""
    import plotly.graph_objects as go

    categories = list(fig.layout.xaxis.categoryarray or [])
    for rank, group in enumerate(groups):
        ref = group.get("references", {}).get(f"{ratio}:{mode}", {})
        value = ref.get("value")
        if not ref.get("enabled") or value is None or not np.isfinite(value):
            continue
        axis_points = {}
        for trace in fig.data:
            if trace.legendgroup == f"appearance_{group['id']}" and trace.mode == "markers":
                points = axis_points.setdefault(trace.yaxis or "y", [])
                points.extend(categories.index(x) if categories else float(x) for x in trace.x)
        label = group.get("legend_label") or group["name"]
        for i, (axis, points) in enumerate(axis_points.items()):
            lo, hi = min(points), max(points)
            if lo == hi:
                lo, hi = lo - 0.35, hi + 0.35
            band = ref.get("band")
            legendgroup = f"reference_{group['id']}"
            # Shape coordinates on a categorical axis use category indices.
            # Numeric scatter coordinates would instead introduce new categories.
            if categories:
                fig.add_shape(type="line", x0=lo, x1=hi, y0=value, y1=value,
                    xref="x", yref=axis, line=dict(color=group["colour"], dash="dash", width=1))
                if _positive(band):
                    fig.add_shape(type="rect", x0=lo, x1=hi, y0=value-band, y1=value+band,
                        xref="x", yref=axis, fillcolor=group["colour"], opacity=0.12,
                        line_width=0, layer="below")
            if _positive(band):
                if not categories:
                    fig.add_trace(go.Scatter(x=[lo, hi, hi, lo, lo],
                        y=[value-band, value-band, value+band, value+band, value-band],
                        fill="toself", fillcolor=group["colour"], opacity=0.12,
                        line=dict(width=0), mode="lines", hoverinfo="skip", showlegend=False,
                        yaxis=axis, legendgroup=legendgroup))
            definition = escape(ref.get("description", ""))
            fig.add_trace(go.Scatter(x=[None] if categories else [lo, hi],
                y=[None] if categories else [value, value], mode="lines",
                line=dict(color=group["colour"], dash="dash", width=1), yaxis=axis,
                name=f"{label} reference" + (f" ± {band:g}" if _positive(band) else ""),
                legendgroup=legendgroup, legendrank=100 + rank, showlegend=i == 0,
                hovertemplate=f"{escape(label)} reference: {value:g}<br>{definition}<extra></extra>"))


def apply_publication_sizing(fig, settings):
    """Use CSS pixels at 96 dpi; point dimensions are converted from 72 pt/in."""
    if settings.get("format") == "svg":
        # WebGL point layers rasterize inside SVG; publication exports use
        # ordinary scatter traces to retain vector markers and error bars.
        import plotly.graph_objects as go

        traces = []
        for trace in fig.data:
            if trace.type == "scattergl":
                payload = trace.to_plotly_json()
                payload["type"] = "scatter"
                trace = go.Scatter(payload)
            traces.append(trace)
        fig.data = ()
        fig.add_traces(traces)
    px = 96 / 72
    font = settings["font_pt"] * px
    fig.update_layout(width=round(settings["width_mm"] * 96 / 25.4),
        height=round(settings["height_mm"] * 96 / 25.4), font_size=font,
        title_font_size=font, legend_font_size=font,
        margin=dict(l=font*4, r=font*4, t=font*5, b=font*4))
    fig.update_xaxes(tickfont_size=font, title_font_size=font, automargin=True)
    fig.update_yaxes(tickfont_size=font, title_font_size=font, automargin=True)
    fig.update_annotations(font_size=font)
    for trace in fig.data:
        if trace.type in {"scatter", "scattergl"} and trace.mode == "markers":
            trace.marker.size = settings["marker_pt"] * px
            trace.error_y.thickness = settings["error_pt"] * px
            trace.error_y.width = settings["cap_pt"] * px

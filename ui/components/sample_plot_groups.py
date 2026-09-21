"""Appearance groups scoped to the Selected Samples Plot."""

from uuid import uuid4

import numpy as np
import streamlit as st

from ui.components.stats_charts import _sample_display_labels


GROUPS_KEY = "selected_samples_plot_groups"
MARKERS = {
    "Triangle": "triangle-up", "Square": "square", "Circle": "circle",
    "Diamond": "diamond", "Down triangle": "triangle-down",
    "Cross": "cross", "X": "x", "Star": "star",
}
COLOURS = ("#0072B2", "#D62728", "#009E73", "#CC79A7", "#E69F00")


def render_sample_plot_groups(samples, *, session_id=None):
    """Keep committed appearance records independent of widget lifetime."""
    stored = st.session_state.get(GROUPS_KEY)
    if stored is None or stored["session_id"] != session_id:
        for key in list(st.session_state):
            if isinstance(key, str) and key.startswith("group_pub_"):
                st.session_state.pop(key, None)
        stored = {"session_id": session_id, "groups": []}
        st.session_state[GROUPS_KEY] = stored
    groups = stored["groups"]
    labels = _sample_display_labels(samples)
    sample_ids = list(labels)
    with st.expander("Sample groups and appearance", expanded=False):
        st.caption(
            "Groups apply only to the Grouped Samples Plot (ratio and delta), "
            "including its enlarged export view. Each measurement belongs to at most "
            "one group. Colours also apply to error bars; calculations are unchanged. "
            "Groups are retained while this session is open."
        )
        if st.button("Add group", key="summary_group_add_button"):
            groups.append({
                "id": uuid4().hex, "name": f"Group {len(groups) + 1}",
                "colour": COLOURS[len(groups) % len(COLOURS)],
                "marker": list(MARKERS.values())[len(groups) % len(MARKERS)],
                "members": [],
            })
            st.rerun()
        for group in groups:
            prefix = f"summary_group_{group['id']}"
            st.markdown(f"**{group['name']}**")
            label_col, earlier_col, later_col = st.columns([4, 1, 1])
            position = groups.index(group)
            with label_col:
                group["legend_label"] = st.text_input(
                    "Legend label (blank uses group name)", value=group.get("legend_label", ""),
                    key=f"{prefix}_legend_label",
                ).strip()
            with earlier_col:
                if st.button("Move up", disabled=position == 0, key=f"{prefix}_up_button"):
                    groups[position-1], groups[position] = groups[position], groups[position-1]
                    st.rerun()
            with later_col:
                if st.button("Move down", disabled=position == len(groups)-1, key=f"{prefix}_down_button"):
                    groups[position+1], groups[position] = groups[position], groups[position+1]
                    st.rerun()
            name_col, marker_col, colour_col, remove_col = st.columns([3, 2, 2, 1])
            with remove_col:
                if st.button("Remove", key=f"{prefix}_remove_button"):
                    groups.remove(group)
                    st.rerun()
            with name_col:
                name = st.text_input("Group name", value=group["name"], key=f"{prefix}_name").strip()
                if name and not any(g is not group and g["name"].casefold() == name.casefold() for g in groups):
                    group["name"] = name
                else:
                    st.warning("Enter a non-empty, unique group name.")
            with marker_col:
                marker = st.selectbox("Marker", list(MARKERS),
                    index=list(MARKERS.values()).index(group["marker"]), key=f"{prefix}_marker")
                group["marker"] = MARKERS[marker]
            with colour_col:
                group["colour"] = st.color_picker("Colour", group["colour"], key=f"{prefix}_colour")
            assigned_elsewhere = {member for g in groups if g is not group for member in g["members"]}
            available = [sid for sid in sample_ids if sid not in assigned_elsewhere]
            members_key = f"{prefix}_members"
            # Prune stale widget selections without removing filtered measurements
            # from the durable record (the caller supplies the whole session).
            current = st.session_state.get(members_key, group["members"])
            st.session_state[members_key] = [sid for sid in current if sid in available]
            search = st.text_input("Find samples by name", key=f"{prefix}_search")
            matches = [s.observation_id for s in samples
                       if search.strip() and search.casefold().strip() in s.name.casefold()
                       and s.observation_id in available]
            if st.button(f"Add matching samples ({len(matches)})", disabled=not matches,
                         key=f"{prefix}_matches_button"):
                st.session_state[members_key] = list(dict.fromkeys(st.session_state[members_key] + matches))
            group["members"] = st.multiselect(
                "Samples in group", available, format_func=labels.get, key=members_key,
                help="Search above to add matching runs, then remove individual measurements here. "
                     "To move a measurement, first remove it from its current group.",
            )
    return groups


def apply_sample_plot_groups(fig, groups):
    """Split only observation marker traces, preserving coordinates and errors."""
    membership = {sid: group for group in groups for sid in group["members"]}
    if not membership:
        return fig
    traces = []
    legend_seen = set()
    for trace in fig.data:
        if trace.type not in {"scatter", "scattergl"} or trace.mode != "markers":
            traces.append(trace)
            continue
        if trace.ids is None or len(trace.ids) == 0:
            traces.append(trace)
            continue
        partitions = {}
        for i, sid in enumerate(trace.ids):
            group = membership.get(sid)
            partitions.setdefault(group["id"] if group else None, []).append(i)
        for group_id, indices in partitions.items():
            payload = trace.to_plotly_json()
            for key in ("x", "y", "ids", "customdata", "text", "hovertext"):
                if key in payload and not isinstance(payload[key], str):
                    payload[key] = np.asarray(payload[key])[indices].tolist()
            for axis in ("error_x", "error_y"):
                for key in ("array", "arrayminus"):
                    if key in payload.get(axis, {}):
                        payload[axis][key] = np.asarray(payload[axis][key])[indices]
            if group_id is not None:
                group = membership[trace.ids[indices[0]]]
                payload["name"] = group.get("legend_label") or group["name"]
                payload["legendrank"] = next(i for i, g in enumerate(groups) if g["id"] == group_id)
                payload["legendgroup"] = f"appearance_{group_id}"
                payload["showlegend"] = group_id not in legend_seen
                legend_seen.add(group_id)
                payload["marker"].update(color=group["colour"], symbol=group["marker"])
                payload["marker"]["line"]["color"] = group["colour"]
                if "error_y" in payload:
                    payload["error_y"]["color"] = group["colour"]
            traces.append(type(trace)(payload))
    fig.data = ()
    fig.add_traces(traces)
    return fig

"""Sample classification editor for TraceISO."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from domain.models import Sample
from ui.navigation import SESSION_KEY_LAST_EDITED_SAMPLE_NAME, SESSION_KEY_SAMPLE_SEARCH_FILTER
from ui.state import get_state


def _build_editor_dataframe(samples: List[Sample]) -> pd.DataFrame:
    """Build editor DataFrame from the current source-of-truth samples."""
    rows = []
    for idx, sample in enumerate(samples):
        rows.append({
            "_row_key": idx,
            "Include": not bool(sample.metadata.get("excluded", False)),
            "Name": sample.name,
            "Type": sample.sample_type.upper(),
            "Run Number": sample.run_number,
            "Cycles": sample.n_cycles,
        })
    return pd.DataFrame(rows)


def _normalize_editor_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize editor data types/values for stable comparisons."""
    normalized = df.copy()
    if "Type" in normalized.columns:
        normalized["Type"] = (
            normalized["Type"]
            .astype(str)
            .str.strip()
            .str.upper()
        )
        normalized["Type"] = normalized["Type"].where(
            normalized["Type"].isin({"STD", "SMP", "BLK"}),
            "SMP",
        )
    if "Include" in normalized.columns:
        normalized["Include"] = normalized["Include"].astype(bool)
    if "_row_key" in normalized.columns:
        normalized["_row_key"] = normalized["_row_key"].astype(int)
    return normalized


def _merge_widget_edits(edited_df: pd.DataFrame, widget_state: object) -> pd.DataFrame:
    """Merge direct widget edited_rows into returned dataframe."""
    effective_df = edited_df.copy()
    if not isinstance(widget_state, dict):
        return effective_df

    edited_rows = widget_state.get("edited_rows", {})
    if not isinstance(edited_rows, dict):
        return effective_df

    for row_key, row_changes in edited_rows.items():
        try:
            row_idx = int(row_key)
        except (TypeError, ValueError):
            continue
        if row_idx < 0 or row_idx >= len(effective_df):
            continue
        if not isinstance(row_changes, dict):
            continue

        for col_name, col_value in row_changes.items():
            if col_name in effective_df.columns:
                effective_df.at[row_idx, col_name] = col_value

    return effective_df


def clear_sample_editor_state(key: str = "sample_editor") -> None:
    """Clear cached widget/session state for the sample editor."""
    static_keys = (
        key,
        f"{key}_source_signature",
        f"{key}_working_df",
        f"{key}_df",
        f"{key}_names",
        f"{key}_df_cache",
        f"{key}_names_cache",
        f"{key}_identity_signature",
    )
    filter_prefix = f"{key}__f"
    dynamic_keys = [k for k in st.session_state if k.startswith(filter_prefix)]
    for cache_key in (*static_keys, *dynamic_keys):
        st.session_state.pop(cache_key, None)


def render_sample_editor(
    samples: List[Sample],
    key: str = "sample_editor",
    filter_term: str = "",
    height: int | None = None,
) -> Tuple[List[Sample], bool]:
    """Render an editable sample classification table.

    When *filter_term* is non-empty the table shows only rows whose name
    contains the term (case-insensitive).  Edits to visible rows are merged
    back into the full working_df via _row_key so pending changes to hidden
    rows are not lost.
    """
    if not samples:
        st.info("No samples loaded.")
        return samples, False

    source_df = _normalize_editor_dataframe(_build_editor_dataframe(samples))

    # Identity signature: stable fields only — NOT Type/Include, to avoid
    # resetting the widget on every user edit.
    identity_signature = tuple(
        (
            int(row["_row_key"]),
            str(row["Name"]),
            int(row["Run Number"]),
            int(row["Cycles"]),
        )
        for _, row in source_df.iterrows()
    )

    signature_key = f"{key}_identity_signature"
    working_key = f"{key}_working_df"

    if st.session_state.get(signature_key) != identity_signature:
        st.session_state[signature_key] = identity_signature
        st.session_state[working_key] = source_df.copy()
        st.session_state.pop(key, None)

    working_df = st.session_state.get(working_key)
    if not isinstance(working_df, pd.DataFrame):
        working_df = source_df.copy()
    else:
        working_df = _normalize_editor_dataframe(working_df)

    if (
        list(working_df.columns) != list(source_df.columns)
        or len(working_df) != len(source_df)
        or not working_df["_row_key"].equals(source_df["_row_key"])
    ):
        working_df = source_df.copy()

    # --- Filter for display -----------------------------------------------
    term_lower = filter_term.strip().lower()
    if term_lower:
        row_mask = working_df["Name"].str.lower().str.contains(term_lower, regex=False)
        display_df = working_df[row_mask].reset_index(drop=True)
        # Key encodes the exact set of visible rows so the widget is reset
        # when the filter changes the row count/composition.
        row_key_hash = abs(hash(tuple(display_df["_row_key"].tolist()))) % 1_000_000
        editor_key = f"{key}__f{row_key_hash}"
        st.caption(
            f"Showing {len(display_df)} of {len(working_df)} samples "
            f"matching \"{filter_term.strip()}\"."
        )
        if display_df.empty:
            st.info(f"No samples match \"{filter_term.strip()}\".")
            st.session_state[working_key] = working_df
            # Return samples unchanged (pending = current working state)
            pending: Dict[int, Tuple[str, bool]] = {
                int(r["_row_key"]): (str(r["Type"]).upper(), not bool(r["Include"]))
                for _, r in working_df.iterrows()
            }
            return _apply_pending(samples, pending)
    else:
        display_df = working_df.copy()
        editor_key = key
    # ----------------------------------------------------------------------

    column_config = {
        "Include": st.column_config.CheckboxColumn(
            "Include",
            help="Uncheck to exclude this measurement from processing.",
            default=True,
            width="small",
        ),
        "Name": st.column_config.TextColumn(
            "Sample",
            disabled=True,
            width="medium",
        ),
        "Type": st.column_config.SelectboxColumn(
            "Type",
            options=["STD", "SMP", "BLK"],
            required=True,
            width="small",
        ),
        "Run Number": st.column_config.NumberColumn(
            "Run",
            disabled=True,
            width="small",
        ),
        "Cycles": st.column_config.NumberColumn(
            "Cycles",
            disabled=True,
            width="small",
        ),
    }

    editor_kwargs = {}
    if height is not None:
        editor_kwargs["height"] = height

    edited_df = st.data_editor(
        display_df,
        column_config=column_config,
        column_order=["Include", "Name", "Type", "Run Number", "Cycles"],
        hide_index=True,
        width="stretch",
        key=editor_key,
        num_rows="fixed",
        **editor_kwargs,
    )

    effective_display = _merge_widget_edits(edited_df, st.session_state.get(editor_key))
    effective_display = _normalize_editor_dataframe(effective_display)

    # Merge filtered-view edits back into the full working_df via _row_key
    if term_lower:
        updated_working = working_df.copy()
        for _, row in effective_display.iterrows():
            rk = int(row["_row_key"])
            m = updated_working["_row_key"] == rk
            updated_working.loc[m, "Type"] = str(row["Type"]).upper()
            updated_working.loc[m, "Include"] = bool(row["Include"])
        effective_df = _normalize_editor_dataframe(updated_working)
    else:
        effective_df = effective_display

    if "_row_key" not in effective_df.columns:
        effective_df["_row_key"] = range(len(effective_df))
    st.session_state[working_key] = effective_df.copy()

    pending = {
        int(row["_row_key"]): (str(row["Type"]).upper(), not bool(row["Include"]))
        for _, row in effective_df.iterrows()
    }
    return _apply_pending(samples, pending)


def _apply_pending(
    samples: List[Sample],
    pending: Dict[int, Tuple[str, bool]],
) -> Tuple[List[Sample], bool]:
    """Apply a pending {row_key: (type, excluded)} map to the samples list."""
    changed = False
    last_edited_sample_name = None
    updated_samples: List[Sample] = []

    for idx, sample in enumerate(samples):
        new_type, new_excluded = pending.get(
            idx,
            (sample.sample_type.upper(), bool(sample.metadata.get("excluded", False))),
        )

        old_excluded = bool(sample.metadata.get("excluded", False))
        type_changed = new_type != sample.sample_type.upper()
        exclude_changed = new_excluded != old_excluded

        if type_changed or exclude_changed:
            updated = sample.copy()
            updated.sample_type = new_type
            updated.metadata = dict(sample.metadata)
            updated.metadata["excluded"] = new_excluded
            updated_samples.append(updated)
            changed = True
            last_edited_sample_name = sample.name
        else:
            updated_samples.append(sample)

    if last_edited_sample_name:
        st.session_state[SESSION_KEY_LAST_EDITED_SAMPLE_NAME] = last_edited_sample_name

    return updated_samples, changed


def render_batch_type_buttons() -> str | None:
    """Render batch type assignment buttons."""
    st.caption("Batch assign type to all samples:")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("All STD", width="stretch", key="batch_std"):
            return "STD"
    with col2:
        if st.button("All SMP", width="stretch", key="batch_smp"):
            return "SMP"
    with col3:
        if st.button("All BLK", width="stretch", key="batch_blk"):
            return "BLK"
    with col4:
        if st.button("Reset", width="stretch", key="batch_reset"):
            return "RESET"

    return None


def apply_batch_type(samples: List[Sample], new_type: str) -> List[Sample]:
    """Apply a type to all samples."""
    if new_type == "RESET":
        state = get_state()
        if state.original_samples:
            original_map = {
                (s.name, s.run_number): s
                for s in state.original_samples
            }
            reset_samples: List[Sample] = []
            for sample in samples:
                original = original_map.get((sample.name, sample.run_number))
                if original is None:
                    reset_samples.append(sample)
                    continue

                restored = sample.copy()
                restored.sample_type = original.sample_type
                restored.metadata = dict(sample.metadata)
                restored.metadata["excluded"] = bool(original.metadata.get("excluded", False))
                reset_samples.append(restored)
            return reset_samples
        return samples

    updated = []
    for sample in samples:
        new_sample = sample.copy()
        new_sample.sample_type = new_type
        updated.append(new_sample)

    return updated


def render_search_assign(
    samples: List[Sample],
) -> Tuple[str | None, str | None]:
    """Render a search bar with targeted batch-assign buttons."""
    st.caption("Search & assign type to matching samples:")

    col_search, col_count = st.columns([3, 1])
    with col_search:
        search_term = st.text_input(
            "Filter by name",
            placeholder="e.g. NIST, Blank, SRM",
            key=SESSION_KEY_SAMPLE_SEARCH_FILTER,
            label_visibility="collapsed",
        )

    if search_term:
        term_lower = search_term.strip().lower()
        matches = [
            s for s in samples
            if term_lower in s.name.lower()
        ]
        with col_count:
            st.markdown(
                f"<span style='color:#4ec9b0; line-height:2.4'>" 
                f"{len(matches)}/{len(samples)} match</span>",
                unsafe_allow_html=True,
            )

        if matches:
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button(
                    f"Set {len(matches)} → STD",
                    key="search_assign_std",
                    width="stretch",
                ):
                    return search_term, "STD"
            with c2:
                if st.button(
                    f"Set {len(matches)} → SMP",
                    key="search_assign_smp",
                    width="stretch",
                ):
                    return search_term, "SMP"
            with c3:
                if st.button(
                    f"Set {len(matches)} → BLK",
                    key="search_assign_blk",
                    width="stretch",
                ):
                    return search_term, "BLK"

    return None, None


def apply_filtered_type(
    samples: List[Sample],
    search_term: str,
    new_type: str,
) -> List[Sample]:
    """Apply a type only to samples whose name matches *search_term*."""
    term_lower = search_term.strip().lower()
    updated = []
    for sample in samples:
        if term_lower in sample.name.lower():
            new_sample = sample.copy()
            new_sample.sample_type = new_type
            updated.append(new_sample)
        else:
            updated.append(sample)
    return updated

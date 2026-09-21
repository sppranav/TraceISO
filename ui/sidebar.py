"""Sidebar for TraceISO."""

import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

from ui.navigation import (
    MAIN_SECTION_KEY,
    MAIN_SECTION_WIDGET_KEY,
    SESSION_KEY_THEME_NAME,
    is_sticky_widget_key,
)
from ui.state import get_state
from ui.theme import get_theme
from ui.utils import format_isotope_label, get_active_internal_normalization_ratio
from config.settings import DisplayConfig, ProcessingConfig, ExportConfig, UncertaintyConfig


# Redundant colour, symbol, and text encoding for sample navigation. Streamlit
# colour directives are used for buttons; the exact sample palette is carried
# by badges, tables, and Plotly traces through the shared theme tokens.
_TYPE_MARKER = {
    "STD": ":orange[◆] STD",
    "SMP": ":blue[●] SMP",
    "BLK": ":gray[■] BLK",
    "QC": ":violet[▲] QC",
}

_SELECTBOX_TYPE_MARKER = {
    "STD": "◆ STD",
    "SMP": "● SMP",
    "BLK": "■ BLK",
    "QC": "▲ QC",
}


def _type_marker(theme, sample_type: str) -> str:
    """Return redundant colour, shape, and code for a sample type."""
    return _TYPE_MARKER.get(theme.normalize_sample_type(sample_type), ":gray[●] SMP")


def render_sidebar() -> None:
    """Render the sidebar content."""
    with st.sidebar:
        _render_file_upload()
        _render_file_info()
        _render_sample_list()
        _render_advanced()
        _render_plot_appearance()


def _render_file_upload() -> None:
    """Render the file upload section."""
    st.header("TraceISO")
    state = get_state()

    uploaded_file = st.file_uploader(
        "Upload HDF5 file",
        type=["h5", "hdf5"],
        key="file_uploader",
        help="Upload an MC-ICP-MS data file in HDF5 format",
    )

    if uploaded_file is not None:
        _handle_file_upload(uploaded_file)
    elif state.has_data:
        # User dismissed the file via the native X — clear application state
        # to keep the UI consistent with the empty uploader widget.
        st.session_state.pop("_suppress_reload", None)  # item 61: clear stale token
        st.session_state.pop("_upload_identity_memo_v1", None)
        state.clear()
        st.rerun()
    else:
        # Discard any legacy reload token after the upload is dismissed.
        st.session_state.pop("_suppress_reload", None)
        st.session_state.pop("_upload_identity_memo_v1", None)

    if state.has_data:
        if st.button(
            "Reset to Original",
            key="btn_reset_to_original",
            width="stretch",
            help="Restore all cycle masks, sample types, and manual exclusions "
            "to the state they were in when the file was first loaded.",
            disabled=not bool(getattr(state, "original_samples", None)),
        ):
            st.session_state["_sidebar_pending_destructive_action"] = "reset"
            st.rerun()

        pending_action = st.session_state.get("_sidebar_pending_destructive_action")
        if pending_action == "reset":
            confirm_label, description = _sidebar_confirmation_spec(pending_action)
            st.caption(description)
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                confirmed = st.button(
                    confirm_label,
                    key=f"confirm_sidebar_{pending_action}",
                    width="stretch",
                )
            with cancel_col:
                cancelled = st.button(
                    "Cancel",
                    key=f"cancel_sidebar_{pending_action}",
                    width="stretch",
                )
            if cancelled:
                _cancel_pending_sidebar_action()
                st.rerun()
            if confirmed:
                st.session_state.pop("_sidebar_pending_destructive_action", None)
                _apply_confirmed_sidebar_action(state, pending_action)
                st.rerun()


def _sidebar_confirmation_spec(action: str) -> tuple[str, str]:
    """Describe the reset controller for its confirmation prompt."""
    if action == "reset":
        return (
            "Confirm Reset",
            "Restore cycle masks, sample types and manual exclusions from the "
            "post-load snapshot. The processed result is discarded and must be "
            "created again.",
        )
    raise ValueError(f"Unsupported sidebar action: {action!r}")


def _cancel_pending_sidebar_action() -> None:
    """Cancel confirmation without touching scientific or edit state."""
    st.session_state.pop("_sidebar_pending_destructive_action", None)


def _apply_confirmed_sidebar_action(state, action: str) -> None:
    """Dispatch a confirmed request to the pre-existing action controller."""
    if action == "reset":
        _reset_to_original_samples(state)
    else:
        raise ValueError(f"Unsupported sidebar action: {action!r}")


# Project root (parent of the ``ui`` package) — used as the cwd when
# spawning the standalone Qt tools via ``-m tools.<module>``.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Standalone Qt desktop tools launchable from the running app, mirroring the
# *_Launcher.bat files. (module, button label, help text). These open a native
# window on the machine running the Streamlit server, so they are a
# LOCAL-DESKTOP convenience only — they do nothing useful on a remote deploy.
_DESKTOP_TOOLS = (
    (
        "tools.crm_manager",
        "CRM Library Manager",
        "Open the desktop editor for certified reference values "
        "(config/crm_library.json).",
    ),
    (
        "tools.neptune_data_extractor.main",
        "Neptune Data Extractor",
        "Open the desktop tool for extracting Neptune data into TraceISO HDF5.",
    ),
    (
        "tools.global_uncertainty_manager",
        "Global Uncertainty Manager",
        "Open the desktop editor for global uncertainty input values.",
    ),
)


def _windowless_python() -> str:
    """Resolve a console-less Python (``pythonw.exe``) that has the Qt GUI deps.

    The interpreter running Streamlit may not have PyQt5 installed (it is only
    required by the standalone tools). Prefer the project ``venv`` used by the
    ``*_Launcher.bat`` scripts and the desktop launcher — it is provisioned with
    PyQt5 — and fall back to the running interpreter only if no venv is present.
    """
    candidates = (
        _PROJECT_ROOT / "venv" / "Scripts" / "pythonw.exe",
        Path(sys.executable).with_name("pythonw.exe"),
        Path(sys.executable),
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _launch_desktop_tool(module: str, label: str) -> None:
    """Spawn a standalone Qt tool via ``pythonw -m <module>`` (detached).

    Mirrors the spawn pattern in ``tools/traceiso_launcher_gui.py`` but hardens
    it for the Streamlit context:

    * The child is given clean standard handles. When TraceISO is itself
      launched headless/detached (e.g. via the desktop launcher, a console-less
      ``pythonw`` process), Streamlit can inherit a closed ``stdin``; a Qt child
      that inherits it dies silently at startup. ``stdin=DEVNULL`` avoids that.
    * On Windows the child is detached into its own process group with no
      inherited console so it cleanly outlives the Streamlit rerun.
    * stdout/stderr are captured to a log file (not discarded), and the process
      is polled briefly so an immediate startup failure is surfaced in the UI
      instead of vanishing.

    The Qt window opens on the host running the Streamlit server, so this is a
    local-desktop convenience only.
    """
    log_path = Path(tempfile.gettempdir()) / f"traceiso_launch_{module.replace('.', '_')}.log"

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        log_file = open(log_path, "w", encoding="utf-8")
    except OSError:
        log_file = None  # diagnostics unavailable; still attempt launch

    try:
        proc = subprocess.Popen(
            [_windowless_python(), "-m", module],
            cwd=str(_PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_file or subprocess.DEVNULL,
            stderr=subprocess.STDOUT if log_file else subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception as exc:  # noqa: BLE001 - surface any spawn failure to the user
        if log_file is not None:
            log_file.close()
        st.error(f"Could not launch {label}: {exc}")
        return

    # Qt import / window-construction errors surface within the first second.
    time.sleep(1.0)
    returncode = proc.poll()
    if log_file is not None:
        log_file.close()

    if returncode is None:
        st.success(f"Launching {label}…")
        return

    # Process exited (and thus released the log file) — safe to read it back.
    detail = ""
    try:
        detail = log_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        pass
    tail = "\n".join(detail.splitlines()[-12:]) if detail else "(no output captured)"
    st.error(
        f"{label} exited immediately (code {returncode}).\n\n```\n{tail}\n```\n\n"
        f"Full log: {log_path}"
    )


def _reload_crm_library() -> None:
    """Re-read crm_library.json and refresh the active element's defaults."""
    from config.reference_materials import reload_library
    from domain.elements.registry import get_element

    state = get_state()
    reload_library()

    target_symbol = (
        state.element_config.symbol
        if state.element_config is not None
        else state.element_symbol
    )
    if target_symbol:
        try:
            state.element_config = get_element(target_symbol)
        except ValueError:
            state.element_config = None

    state.invalidate_processed_result()
    st.success("CRM library reloaded.")
    st.rerun()


def _reload_global_uncertainty_values() -> None:
    """Re-read global_uncertainty_values.json and clear dependent caches."""
    from ui.runtime_budget_cache import clear_runtime_budget_cache

    state = get_state()
    state.refresh_global_uncertainty_values()
    clear_runtime_budget_cache()
    state.invalidate_processed_result()
    st.session_state.pop("_global_uncertainty_values_warning_shown", None)
    st.success("Uncertainty values reloaded.")
    st.rerun()


def _render_advanced() -> None:
    """Render desktop-tool launchers (with CRM reload) inside an Advanced expander."""
    state = get_state()
    with st.expander("Advanced", expanded=False):
        dev_mode = st.toggle(
            "Developer mode",
            value=state.dev_mode,
            key="toggle_dev_mode",
            help="Show diagnostic debug panels — e.g. 'Budget Debug Values' and "
            "'MC Debug Values' in the Uncertainty tab, full error tracebacks, "
            "performance timings and a downloadable diagnostic report.",
        )
        if dev_mode != state.dev_mode:
            state.dev_mode = dev_mode
        st.divider()
        st.caption("Desktop tools (local use only)")
        for module, label, help_text in _DESKTOP_TOOLS:
            if st.button(
                label,
                key=f"btn_launch_{module}",
                width="stretch",
                help=help_text,
            ):
                _launch_desktop_tool(module, label)

            # Keep "Reload CRM Library" directly below the CRM manager so the
            # edit-then-reload workflow reads top-to-bottom.
            if module == "tools.crm_manager":
                if st.button(
                    "Reload CRM Library",
                    key="btn_reload_crm",
                    width="stretch",
                    help="Re-read config/crm_library.json after editing it in the "
                    "CRM Library Manager above; refreshes element defaults.",
                ):
                    _reload_crm_library()
            if module == "tools.global_uncertainty_manager":
                if st.button(
                    "Reload uncertainty values",
                    key="btn_reload_global_uncertainty",
                    width="stretch",
                    help="Re-read config/global_uncertainty_values.json after editing it in "
                    "the Global Uncertainty Manager above; clears runtime budget caches.",
                ):
                    _reload_global_uncertainty_values()


def _handle_file_upload(uploaded_file) -> None:
    """Process an uploaded file."""
    state = get_state()
    previous_session = dict(st.session_state)
    previous_attributes = dict(vars(state))

    file_size = uploaded_file.size
    # Streamlit's file_id is stable for one genuine upload and changes when the
    # user supplies replacement content, including same-name/same-size files.
    upload_id = str(getattr(uploaded_file, "file_id", "") or id(uploaded_file))
    memo = st.session_state.get("_upload_identity_memo_v1")
    if not isinstance(memo, dict) or memo.get("upload_id") != upload_id:
        file_bytes = uploaded_file.getvalue()
        memo = {
            "upload_id": upload_id,
            "name": uploaded_file.name,
            "size": file_size,
            "hash": hashlib.sha256(file_bytes).hexdigest(),
        }
        st.session_state["_upload_identity_memo_v1"] = memo
    else:
        file_bytes = None
    file_hash = memo["hash"]
    if (
        state.loaded_file == uploaded_file.name
        and state.file_size == file_size
        and state.file_hash == file_hash
        and state.loaded_data_preference == state.processing_config.data_preference
    ):
        return

    if file_bytes is None:
        file_bytes = uploaded_file.getvalue()

    # Load the file
    import tempfile
    from pathlib import Path

    from file_io.hdf5_reader import load_hdf5

    tmp_path = None
    try:
        with st.status("Loading file...", expanded=False) as status:
            status.update(label="Writing temporary file…")
            # Write to temp file (h5py needs a file path).
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                tmp.write(file_bytes)
                tmp_path = Path(tmp.name)

            status.update(label="Parsing HDF5 data…")
            # A065: the preference in force when the read happens is the one the
            # samples in memory were actually taken from. Everything committed
            # below must name this layer, not whatever the config holds after an
            # element reset.
            data_preference_used = state.processing_config.data_preference
            result = load_hdf5(
                tmp_path,
                data_preference=data_preference_used,
            )

            status.update(label="Detecting element…")
            switching_element = (
                state.element_config is not None
                and result.detected_element is not None
                and state.element_config.symbol != result.detected_element.symbol
            )
            loaded_samples = result.samples

            status.update(label="Preparing default ratios…")
            # Pre-populate default ratios into raw samples so the UI immediately
            # shows available defaults before the pipeline is executed.
            if result.detected_element and result.detected_element.default_ratios:
                from domain.corrections.blank import calculate_ratios
                loaded_samples = calculate_ratios(
                    loaded_samples,
                    result.detected_element.default_ratios,
                    use_corrected=False,
                )

            for sample in loaded_samples:
                sample.metadata.setdefault("_source_file_name", uploaded_file.name)

            import copy
            prepared_processing = ProcessingConfig(data_preference=data_preference_used) if switching_element else copy.deepcopy(state.processing_config)
            prepared_processing.global_cycle_range = False
            original_samples = [s.copy() for s in loaded_samples]
            # Commit loaded session state only after all preparation above succeeds.
            state.processing_config = prepared_processing
            if switching_element:
                # data_preference names which arrays were taken out of the file,
                # not an element-specific setting, so it is carried across the
                # reset. Constructing a bare ProcessingConfig() here recorded the
                # default preference against data read under a different one, and
                # the unchanged-file guard then kept serving it mislabelled.
                state.processing_config = ProcessingConfig(
                    data_preference=data_preference_used,
                )
                state.display_config = DisplayConfig()
                state.export_config = ExportConfig()
                state.uncertainty_config = UncertaintyConfig()
                if "selected_ratios" in st.session_state:
                    del st.session_state["selected_ratios"]

            state.samples = loaded_samples
            state.element_config = result.detected_element
            state.detected_isotopes = result.detected_isotopes
            state.file_structure = result.file_structure
            state.loaded_file = uploaded_file.name
            state.warnings = result.warnings
            from ui.invalidation import on_new_file_loaded
            on_new_file_loaded(state)
            state.selected_sample_idx = 0
            _reset_cycle_range_state(state)
            _purge_sticky_widget_state()

            # Snapshot original samples for "Reset" functionality
            state.original_samples = original_samples
            state.file_size = file_size
            state.file_hash = file_hash
            state.loaded_data_preference = data_preference_used

            status.update(label=f"Successfully loaded {len(result.samples)} samples", state="complete")

        st.success(f"Loaded {len(result.samples)} samples")

        if result.warnings:
            for warn in result.warnings:
                st.warning(warn)
    except Exception as e:
        vars(state).clear()
        vars(state).update(previous_attributes)
        for key in list(st.session_state):
            if key not in previous_session:
                del st.session_state[key]
        for key, value in previous_session.items():
            st.session_state[key] = value
        st.error(f"Failed to load file: {e}")
        if state.dev_mode:
            from ui.diagnostics import record_error
            record_error("ui/sidebar.py", e)
            st.exception(e)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _render_file_info() -> None:
    """Render file information panel."""
    state = get_state()

    if not state.has_data:
        return

    st.divider()
    with st.expander("Loaded file details", expanded=False):
        full_name = state.loaded_file or ""
        display_name = full_name[:18] + "\u2026" + full_name[-14:] if len(full_name) > 35 else full_name
        import html as _html
        escaped_full = _html.escape(full_name, quote=True)
        escaped_display = _html.escape(display_name)
        st.markdown(
            f"<small><strong>File:</strong> <span title='{escaped_full}' "
            f"style='cursor:help;'>{escaped_display}</span></small>",
            unsafe_allow_html=True,
        )
        st.caption(f"**Element:** {state.element_symbol}" if state.element_config else "**Element:** Not detected")
        if state.detected_isotopes:
            iso_str = ", ".join(format_isotope_label(iso) for iso in state.detected_isotopes)
            st.caption(f"**Isotopes:** {iso_str}")
        counts = state.sample_count_by_type
        css_class = {"STD": "badge-std", "SMP": "badge-smp", "BLK": "badge-blk"}
        badges_html = " ".join(
            f'<span class="{css_class.get(k, "badge-smp")}">{_html.escape(k)}: {v}</span>'
            for k, v in sorted(counts.items())
        )
        st.markdown(
            f"<small><strong>Samples:</strong></small>&nbsp;{badges_html}",
            unsafe_allow_html=True,
        )

def _render_sample_list() -> None:
    """Render the sample list for navigation."""
    state = get_state()

    if not state.has_data:
        return

    st.divider()
    st.subheader("Inspect samples")
    st.caption(
        "Search and type filters affect this list only. Selecting a sample opens "
        "Sample Inspector."
    )

    theme = get_theme()
    # Get samples (from result if processed, else raw)
    samples = state.result.samples if state.has_result else state.samples

    # Per-type counts (normalised so BLANK/STANDARD collapse to BLK/STD)
    type_counts: dict = {}
    for sample in samples:
        code = theme.normalize_sample_type(sample.sample_type)
        type_counts[code] = type_counts.get(code, 0) + 1

    # Name search
    search = (
        st.text_input(
            "Filter inspect-sample list",
            key="sample_search",
            placeholder="🔍 Filter samples…",
            label_visibility="collapsed",
        )
        .strip()
        .lower()
    )

    # Type filters (with per-type counts)
    col1, col2, col3 = st.columns(3)
    with col1:
        show_std = st.checkbox(f"STD ({type_counts.get('STD', 0)})", value=True, key="filter_std")
    with col2:
        show_smp = st.checkbox(f"SMP ({type_counts.get('SMP', 0)})", value=True, key="filter_smp")
    with col3:
        show_blk = st.checkbox(f"BLK ({type_counts.get('BLK', 0)})", value=True, key="filter_blk")

    filter_by_type = {"STD": show_std, "SMP": show_smp, "BLK": show_blk}

    filtered_samples = []
    for i, sample in enumerate(samples):
        code = theme.normalize_sample_type(sample.sample_type)
        # Only the three governed types are listed (matches prior behaviour;
        # QC/other types have no filter toggle and stay hidden).
        if not filter_by_type.get(code, False):
            continue
        if search and search not in sample.name.lower():
            continue
        filtered_samples.append((i, sample))

    # Sample list — one full-width button per sample, with redundant colour,
    # shape, and type code inlined into the button label.
    if filtered_samples:
        by_id = {
            sample.observation_id: (orig_idx, sample)
            for orig_idx, sample in filtered_samples
        }
        option_ids = list(by_id)
        selected_id = next(
            (
                sample.observation_id
                for orig_idx, sample in filtered_samples
                if orig_idx == state.selected_sample_idx
            ),
            None,
        )
        if st.session_state.get("sample_selectbox_picker") not in option_ids:
            st.session_state.pop("sample_selectbox_picker", None)

        def format_sample_option(observation_id: str) -> str:
            orig_idx, sample = by_id[observation_id]
            marker = _SELECTBOX_TYPE_MARKER.get(
                theme.normalize_sample_type(sample.sample_type), "● SMP"
            )
            return f"{marker} · {sample.name} · run {sample.run_number} · #{orig_idx + 1}"

        def select_sidebar_sample() -> None:
            chosen = st.session_state.get("sample_selectbox_picker")
            if chosen in by_id:
                _handle_sidebar_sample_selection(state, by_id[chosen][0], rerun=False)

        st.selectbox(
            "Select a sample to open Sample Inspector",
            options=option_ids,
            index=option_ids.index(selected_id) if selected_id in option_ids else 0,
            format_func=format_sample_option,
            key="sample_selectbox_picker",
            on_change=select_sidebar_sample,
        )
        chosen = st.session_state.get("sample_selectbox_picker")
        if st.button(
            "Open selected sample",
            key="open_selected_sample_button",
            width="stretch",
            disabled=chosen not in by_id,
        ) and chosen in by_id:
            _handle_sidebar_sample_selection(state, by_id[chosen][0])
    elif search:
        st.caption("No samples match the search.")
    else:
        st.caption("No samples match the filter.")


def render_plot_display_options() -> None:
    """Render display options."""
    state = get_state()

    st.divider()
    st.subheader("Correction Layer Visibility")

    config = state.display_config
    element_cfg = state.element_config
    proc_cfg = state.processing_config

    col1, col2 = st.columns(2)
    with col1:
        show_raw = st.checkbox("Raw", value=config.show_raw, key="display_raw")
    with col2:
        show_corr = st.checkbox(
            "Blank corrected",
            value=config.show_corrected,
            key="display_corr",
            help="Show blank-corrected data. If no blank correction is applied, shows the same as Raw.",
        )

    show_drift = config.show_drift_corrected
    show_interference = config.show_interference_corrected
    show_iif = config.show_iif_corrected
    show_pb_standard = getattr(config, "show_pb_standard_corrected", True)
    show_sr_standard = config.show_sr_standard_corrected
    threshold_line_basis = config.threshold_line_basis
    show_stats_box = config.show_ratio_stats_box
    stats_statistic = getattr(config, "ratio_stats_statistic", "2SD")

    has_drift_layer = bool(proc_cfg.drift.enabled)
    has_interference_layer = bool(
        element_cfg
        and (
            (element_cfg.has_interference and proc_cfg.apply_interference_correction)
            or (element_cfg.symbol == "Pb" and proc_cfg.apply_hg_interference_correction)
        )
    )
    has_iif_like_layer = False
    iif_like_label = "IIF-corrected"
    if element_cfg:
        if element_cfg.has_interference:
            has_iif_like_layer = bool(proc_cfg.apply_mass_bias_correction)
            iif_like_label = "IIF-corrected"
        elif element_cfg.supports_ssb:
            has_iif_like_layer = bool(proc_cfg.enable_ssb)
            iif_like_label = "SSB-corrected"

    # Pb-standard calibration: the Tl-only and final layers get independent
    # toggles, and the calibration suppresses the separate drift layer.
    pb_calibration_route = bool(
        element_cfg
        and element_cfg.symbol == "Pb"
        and proc_cfg.apply_mass_bias_correction
        and proc_cfg.pb_standard_calibration.enabled
    )
    if pb_calibration_route:
        has_drift_layer = False
        has_iif_like_layer = True
        iif_like_label = "Tl-normalized"

    sr_calibration_route = bool(
        element_cfg and element_cfg.symbol == "Sr"
        and proc_cfg.apply_mass_bias_correction and proc_cfg.sr_session_anchoring
    )
    if sr_calibration_route:
        iif_like_label = "Internally normalized"
    optional_layers = []
    if has_drift_layer:
        optional_layers.append(("Drift-corrected", "display_drift_corr", "show_drift_corrected"))
    if has_interference_layer:
        optional_layers.append(("Interference-corrected", "display_interf_corr", "show_interference_corrected"))
    if has_iif_like_layer:
        optional_layers.append((iif_like_label, "display_iif_corr", "show_iif_corrected"))
    if pb_calibration_route:
        optional_layers.append(("Tl + Pb-standard-corrected", "display_pb_standard_corr", "show_pb_standard_corrected"))

    if sr_calibration_route:
        optional_layers.append(("Internal normalization + Sr-standard-corrected", "display_sr_standard_corr", "show_sr_standard_corrected"))

    if optional_layers:
        for col, (label, key, attr) in zip(st.columns(len(optional_layers)), optional_layers):
            with col:
                value = st.checkbox(label, value=getattr(config, attr, True), key=key)
            if attr == "show_drift_corrected":
                show_drift = value
            elif attr == "show_interference_corrected":
                show_interference = value
            elif attr == "show_sr_standard_corrected":
                show_sr_standard = value
            elif attr == "show_pb_standard_corrected":
                show_pb_standard = value
            else:
                show_iif = value

    show_thresh = st.checkbox(
        "Threshold lines",
        value=config.show_threshold_lines,
        key="display_thresh",
    )
    show_outliers = st.checkbox(
        "Outliers",
        value=getattr(config, "show_outliers", True),
        key="display_outliers",
        help="Show committed cycles excluded by the active processing mask.",
    )
    show_stats_box = st.checkbox(
        "Stats box",
        value=config.show_ratio_stats_box,
        key="display_ratio_stats_box",
    )
    if show_stats_box:
        statistic_labels = {"2 SD": "2SD", "2 SE": "2SE"}
        current_statistic = stats_statistic if stats_statistic in statistic_labels.values() else "2SD"
        selected_statistic_label = st.radio(
            "Stats box statistic",
            options=list(statistic_labels),
            index=list(statistic_labels.values()).index(current_statistic),
            horizontal=True,
            key="display_ratio_stats_statistic",
        )
        stats_statistic = statistic_labels[selected_statistic_label]
    if show_thresh:
        threshold_options = [("Outlier", "processing_mask")]
        if has_iif_like_layer:
            threshold_options.append((f"{iif_like_label.replace('-', ' ')} ratio", "iif_like"))

        if threshold_line_basis not in {value for _, value in threshold_options}:
            threshold_line_basis = "processing_mask"

        if len(threshold_options) > 1:
            option_labels = [label for label, _ in threshold_options]
            label_to_value = {label: value for label, value in threshold_options}
            selected_label = next(
                label for label, value in threshold_options
                if value == threshold_line_basis
            )
            chosen_label = st.selectbox(
                "Threshold basis",
                options=option_labels,
                index=option_labels.index(selected_label),
                key="display_threshold_basis",
                help=(
                    "Choose whether threshold overlays represent the canonical "
                    "outlier-processing layer or the downstream corrected "
                    "IIF/SSB statistics."
                ),
            )
            threshold_line_basis = label_to_value[chosen_label]
        else:
            threshold_line_basis = "processing_mask"

    if element_cfg and element_cfg.has_interference:
        interference_status = "ON" if proc_cfg.apply_interference_correction else "OFF"
        mass_bias_status = "ON" if proc_cfg.apply_mass_bias_correction else "OFF"
        st.caption(
            "Sr Corrections: "
            f"Interference {interference_status} | "
            f"Instrumental isotope fractionation (IIF) {mass_bias_status}"
        )
    elif element_cfg:
        ratio_name = get_active_internal_normalization_ratio(proc_cfg, element_cfg)
        if ratio_name:
            normalization_label = (
                "External normalization"
                if element_cfg.symbol == "Pb"
                else "Internal normalization"
            )
            st.caption(f"{normalization_label}: {ratio_name}")

    if (
        show_raw != config.show_raw
        or show_corr != config.show_corrected
        or show_drift != config.show_drift_corrected
        or show_interference != config.show_interference_corrected
        or show_iif != config.show_iif_corrected
        or show_pb_standard != getattr(config, "show_pb_standard_corrected", True)
        or show_sr_standard != config.show_sr_standard_corrected
        or show_thresh != config.show_threshold_lines
        or show_outliers != getattr(config, "show_outliers", True)
        or threshold_line_basis != config.threshold_line_basis
        or show_stats_box != config.show_ratio_stats_box
        or stats_statistic != getattr(config, "ratio_stats_statistic", "2SD")
    ):
        state.display_config = DisplayConfig(
            show_raw=show_raw,
            show_corrected=show_corr,
            show_drift_corrected=show_drift,
            show_interference_corrected=show_interference,
            show_iif_corrected=show_iif,
            show_pb_standard_corrected=show_pb_standard,
            show_sr_standard_corrected=show_sr_standard,
            show_threshold_lines=show_thresh,
            threshold_line_basis=threshold_line_basis,
            show_outliers=show_outliers,
            show_ratio_stats_box=show_stats_box,
            ratio_stats_statistic=stats_statistic,
            show_filtered_raw=config.show_filtered_raw,
            show_filtered_corrected=config.show_filtered_corrected,
            show_blank_info=config.show_blank_info,
            precision=config.precision,
        )


def _render_plot_appearance() -> None:
    """Render plot appearance controls (Retro Style)."""
    state = get_state()

    with st.sidebar.expander("Visualisation Settings", expanded=False):
        config = dict(state.plot_config)

        # Clamp saved settings from the former 300–1000 px range.
        if "plot_height_slider" in st.session_state:
            st.session_state["plot_height_slider"] = max(
                300, min(550, int(st.session_state["plot_height_slider"]))
            )
        config["height"] = st.slider(
            "Height (px)",
            min_value=300,
            max_value=550,
            value=max(300, min(550, int(config.get("height", 550)))),
            step=50,
            key="plot_height_slider",
            help="Sample Inspector chart height. Delta charts are capped at 500 px.",
        )

        # styling
        config["marker_size"] = st.slider(
            "Marker Size",
            min_value=3,
            max_value=12,
            value=config.get("marker_size", 9),
            key="plot_marker_size_slider",
            help="Sample Inspector point size. Excluded and selected points retain their size emphasis.",
        )

        config["download_format"] = st.radio(
            "Figure download format",
            options=["png", "svg"],
            index=0 if config.get("download_format", "png") == "png" else 1,
            format_func=lambda value: "PNG (3×)" if value == "png" else "SVG (vector)",
            horizontal=True,
            key="plot_download_format_radio",
            help="SVG is vector output and ignores the PNG scale multiplier; fonts use their logical size.",
        )
        state.plot_config = config

        st.markdown("---")
        
        # Theme switcher
        current_theme = st.session_state.get(SESSION_KEY_THEME_NAME, "light")
        theme_options = ["light", "dark"]
        selected_theme = st.selectbox(
            "Application Theme",
            options=theme_options,
            index=theme_options.index(current_theme) if current_theme in theme_options else 0,
            format_func=lambda x: f"☀ {x.capitalize()}" if x == "light" else f"🌙 {x.capitalize()}",
            key="theme_selector_widget",
        )
        if selected_theme != current_theme:
            st.session_state[SESSION_KEY_THEME_NAME] = selected_theme
            st.rerun()


def _handle_sidebar_sample_selection(state, sample_idx: int, *, rerun: bool = True) -> None:
    """Select the sample and jump to the Sample Inspector tab.

    Safe because the sidebar renders before the section radio in
    ``app_shell.run()``: the rerun happens before the route widget is
    instantiated, so setting its session-state key is not a
    mutate-after-instantiation anti-pattern.
    Callbacks pass ``rerun=False`` because Streamlit reruns after the callback.
    """
    state.select_sample(sample_idx)
    st.session_state[MAIN_SECTION_KEY] = "Sample Inspector"
    st.session_state[MAIN_SECTION_WIDGET_KEY] = "Sample Inspector"
    if rerun:
        st.rerun()


def _purge_sticky_widget_state() -> None:
    """Drop kept-alive Results widget state on a genuine new-file load.

    The keep-alive sweep would otherwise resurrect stale selections (e.g.
    a ``sticky_results_type_filter`` value containing a sample type absent from
    the new dataset, which raises a Streamlit options error).
    """
    for key in list(st.session_state.keys()):
        if is_sticky_widget_key(key):
            st.session_state.pop(key, None)


def _reset_cycle_range_state(state) -> None:
    """Clear stale cycle-range widgets when a new file/session is loaded."""
    state.global_cycle_range_enabled = False
    state.global_cycle_range = None
    if getattr(state, "processing_config", None) is not None:
        state.processing_config.global_cycle_range = False

    keys_to_remove = [
        key
        for key in list(st.session_state.keys())
        if key.startswith("cycle_range_")
        or key.startswith("inspector_")
        or key.startswith("excluded_")
        or key in {
            "global_cycle_range_toggle",
            "global_cycle_range_slider",
            "_traceiso_cycle_ranges",  # item 69: structural dict key
        }
    ]
    for key in keys_to_remove:
        st.session_state.pop(key, None)


def _reset_to_original_samples(state) -> None:
    """item 66: Restore all cycle masks, sample types, and exclusions to the
    post-load snapshot stored in state.original_samples."""
    originals = getattr(state, "original_samples", None)
    if not originals:
        return
    # Re-snapshot from originals (deep copies to prevent shared mutation)
    restored = [s.copy() for s in originals]
    state.samples = restored
    # Invalidate any processed result and runtime budget cache so stale data
    # is not displayed after the reset.
    state.invalidate_processed_result()
    from ui.runtime_budget_cache import clear_runtime_budget_cache
    clear_runtime_budget_cache()
    # Clear per-sample inspector widget state that encoded the old masks/ranges
    _reset_cycle_range_state(state)

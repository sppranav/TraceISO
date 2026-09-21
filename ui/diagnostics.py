"""Session-local developer diagnostics; never changes scientific inputs."""

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
import json
import math
import platform
from time import perf_counter
import traceback

import streamlit as st

from config.constants import APP_VERSION

KEY = "_developer_diagnostics_v1"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _store():
    return st.session_state.setdefault(KEY, {"started_utc": _now(), "performance": {},
                                             "errors": [], "debug_reports": {}})


def record_error(section, exc):
    if not st.session_state.get("dev_mode", False):
        return
    errors = _store()["errors"]
    errors.append({"captured_utc": _now(), "section": section,
                   "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))})
    del errors[:-20]


@contextmanager
def measure(operation):
    """Measure elapsed wall time, including failed calls, only while enabled."""
    if not st.session_state.get("dev_mode", False):
        yield
        return
    started = perf_counter()
    failed = False
    try:
        yield
    except Exception as exc:
        failed = True
        record_error(operation, exc)
        raise
    finally:
        elapsed = perf_counter() - started
        row = _store()["performance"].setdefault(operation, {
            "calls": 0, "failures": 0, "total_seconds": 0.0,
            "last_seconds": 0.0, "max_seconds": 0.0})
        row["calls"] += 1
        row["failures"] += int(failed)
        row["total_seconds"] += elapsed
        row["last_seconds"] = elapsed
        row["max_seconds"] = max(row["max_seconds"], elapsed)


def timed(operation):
    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with measure(operation):
                return func(*args, **kwargs)
        return wrapped
    return decorate


def cache_event(cache, hit):
    if st.session_state.get("dev_mode", False):
        counts = _store().setdefault("cache", {}).setdefault(cache, {"hits": 0, "misses": 0})
        counts["hits" if hit else "misses"] += 1


def capture_debug_report(kind, report):
    if st.session_state.get("dev_mode", False):
        _store()["debug_reports"][kind] = {"captured_utc": _now(), "text": report}


def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_diagnostic_report(state):
    """Export explicitly selected context, not arbitrary session state or raw arrays."""
    from ui.utils import get_cycle_ranges

    dependencies = {}
    for package in ("streamlit", "numpy", "scipy", "pandas", "h5py", "plotly"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "unavailable"
    payload = {
        "schema": "traceiso.developer_diagnostics.v1",
        "generated_utc": _now(),
        "software": {"traceiso": APP_VERSION, "python": platform.python_version(),
                     "os": platform.system(), "os_release": platform.release(),
                     "dependencies": dependencies},
        "session": {"loaded_file": state.loaded_file, "input_sha256": state.file_hash,
                    "element": state.element_symbol, "has_result": state.has_result},
        "current_settings": {"processing": state.processing_config,
                             "uncertainty": state.uncertainty_config,
                             "cycle_ranges": get_cycle_ranges(state)},
        "diagnostics": _store(),
        "scope": "Timings cover operations while Developer mode is on. Nested timings overlap; "
                 "do not sum them. Debug reports are the latest viewed snapshots, not necessarily "
                 "the current settings. Raw measurements and MC draw arrays are excluded.",
    }
    return json.dumps(_json_value(payload), indent=2, ensure_ascii=False, allow_nan=False)


def render_diagnostics(state):
    """Render after the workspace so downloads include the current render's diagnostics."""
    if not state.dev_mode:
        return
    with st.sidebar.expander("Developer diagnostics", expanded=False):
        st.caption("Recorded while Developer mode is on. Timings overlap and should not be added.")
        if st.button("Reset diagnostic history", key="reset_developer_diagnostics"):
            st.session_state.pop(KEY, None)
        data = _store()
        rows = [{"Operation": name, **row} for name, row in data["performance"].items()]
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
        else:
            st.caption("No timings yet. Execute data reduction or open a results/uncertainty view.")
        for name, counts in data.get("cache", {}).items():
            st.caption(f"{name}: {counts['hits']} hits / {counts['misses']} misses")
        st.caption("Report includes settings, file/sample names in debug snapshots, and error paths.")
        st.download_button("Download diagnostic report", build_diagnostic_report(state),
                           file_name="traceiso_diagnostics.json", mime="application/json",
                           key="download_developer_diagnostics", on_click="ignore")

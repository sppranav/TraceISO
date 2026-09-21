"""Qt runtime path setup for source and frozen executions."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _prepend_env_path(name: str, path: Path) -> None:
    value = str(path)
    current = [part for part in os.environ.get(name, "").split(os.pathsep) if part]
    if value not in current:
        os.environ[name] = os.pathsep.join([value, *current]) if current else value


def _resolve_qt_root() -> Path | None:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(getattr(sys, "_MEIPASS"))
        for relative in (Path("PyQt5") / "Qt5", Path("PyQt5") / "Qt"):
            candidate = base / relative
            if candidate.is_dir():
                return candidate
        return None

    spec = importlib.util.find_spec("PyQt5")
    if spec is None or not spec.submodule_search_locations:
        return None

    package_root = Path(next(iter(spec.submodule_search_locations)))
    for name in ("Qt5", "Qt"):
        candidate = package_root / name
        if candidate.is_dir():
            return candidate
    return None


def configure_qt_runtime() -> Path | None:
    """Configure Qt plugin and DLL paths before importing PyQt modules."""
    qt_root = _resolve_qt_root()
    if qt_root is None:
        return None

    plugins_dir = qt_root / "plugins"
    platforms_dir = plugins_dir / "platforms"
    bin_dir = qt_root / "bin"
    qml_dir = qt_root / "qml"

    if plugins_dir.is_dir():
        os.environ["QT_PLUGIN_PATH"] = str(plugins_dir)
    if platforms_dir.is_dir():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms_dir)
    if qml_dir.is_dir():
        os.environ["QML2_IMPORT_PATH"] = str(qml_dir)

    for path in (bin_dir, qt_root, plugins_dir, platforms_dir):
        if path.is_dir():
            _prepend_env_path("PATH", path)

    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for path in (bin_dir, qt_root):
            if not path.is_dir():
                continue
            try:
                os.add_dll_directory(str(path))
            except OSError:
                pass

    return qt_root


# --- Platform font families -------------------------------------------------
#
# The tools were styled on Windows against "Segoe UI"/"Consolas". Neither font
# exists on macOS or Linux, where Qt then falls back to an arbitrary default.
# These helpers name the closest native equivalent per platform. The QSS rules
# that consume them keep every alternative listed as well, so a family missing
# at runtime simply advances to the next candidate.


def ui_font_family() -> str:
    """Native UI font family for the current platform."""
    if sys.platform == "darwin":
        return "Helvetica Neue"
    if sys.platform == "win32":
        return "Segoe UI"
    return "DejaVu Sans"


def mono_font_family() -> str:
    """Native monospace font family for the current platform."""
    if sys.platform == "darwin":
        return "Menlo"
    if sys.platform == "win32":
        return "Consolas"
    return "DejaVu Sans Mono"

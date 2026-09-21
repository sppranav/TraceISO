"""GUI setup and launcher for TraceISO."""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import List

from tools.qt_runtime import configure_qt_runtime

configure_qt_runtime()

from PyQt5 import QtCore, QtGui, QtWidgets


APP_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_FILE = APP_ROOT / "requirements.txt"
VENV_DIR = APP_ROOT / "venv"
SNAPSHOT_FILE = VENV_DIR / ".requirements.snapshot"

_STYLE_SHEET = """
QWidget {
    background-color: #f8f6f1;
    color: #1a1715;
    font-family: "Segoe UI", "Helvetica Neue", "DejaVu Sans", sans-serif;
    font-size: 13px;
}

QGroupBox {
    border: 1px solid #e6dfd2;
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 14px;
    font-weight: 600;
    color: #4a443e;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    color: #7a7269;
}

QPlainTextEdit {
    background-color: #ffffff;
    border: 1px solid #e6dfd2;
    border-radius: 4px;
    padding: 6px;
    font-family: "Consolas", "Menlo", "DejaVu Sans Mono", monospace;
    font-size: 12px;
}

QPushButton {
    background-color: #ffffff;
    color: #1a1715;
    border: 1px solid #e6dfd2;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #f1ede4;
    border-color: #d4cab8;
}
QPushButton:pressed {
    background-color: #e6dfd2;
}
QPushButton:disabled {
    background-color: #f8f6f1;
    color: #a89f93;
    border-color: #e6dfd2;
}
QPushButton[class="primary"] {
    background-color: #1a1715;
    color: #ffffff;
    border-color: #1a1715;
}
QPushButton[class="primary"]:hover {
    background-color: #4a443e;
}
QPushButton[class="primary"]:disabled {
    background-color: #a89f93;
    color: #f8f6f1;
    border-color: #a89f93;
}
"""


def _venv_python() -> Path:
    candidate = VENV_DIR / "Scripts" / "python.exe"
    if candidate.exists():
        return candidate
    return Path(sys.executable)


def _requirements_changed() -> bool:
    if not REQUIREMENTS_FILE.exists():
        return True
    if not SNAPSHOT_FILE.exists():
        return True
    requirements = REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines()
    snapshot = SNAPSHOT_FILE.read_text(encoding="utf-8").splitlines()
    return requirements != snapshot


def _runtime_health_error() -> str | None:
    """Return None if core runtime imports work, else a short error string."""
    python_exe = _venv_python()
    if not python_exe.exists():
        return "Virtual environment python executable was not found."

    health_script = (
        "import importlib; "
        "mods=['numpy','pandas','scipy','h5py','pyarrow','streamlit']; "
        "[importlib.import_module(m) for m in mods]; "
        "import numpy._core._multiarray_umath as _m; "
        "print('ok')"
    )
    proc = subprocess.run(
        [str(python_exe), "-c", health_script],
        cwd=str(APP_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode == 0:
        return None

    output = (proc.stdout or "").strip()
    if len(output) > 1200:
        output = output[:1200] + "\n... (truncated)"
    return output or "Unknown runtime import error."


def _find_free_port(start: int = 8501, end: int = 8510) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


class SetupWorker(QtCore.QThread):
    log_line = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(bool, str)

    def __init__(self, *, preview_only: bool = False, repair_runtime: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.preview_only = preview_only
        self.repair_runtime = repair_runtime
        self.python_exe = _venv_python()

    def _run_cmd(self, cmd: List[str]) -> int:
        self.log_line.emit(f"> {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(APP_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            self.log_line.emit(line.rstrip())
        return proc.wait()

    def run(self) -> None:
        if not REQUIREMENTS_FILE.exists():
            self.done.emit(False, "requirements.txt was not found.")
            return

        try:
            if self.preview_only:
                code = self._run_cmd(
                    [str(self.python_exe), "-m", "pip", "install", "--dry-run", "-r", str(REQUIREMENTS_FILE)]
                )
                if code != 0:
                    self.done.emit(False, "Dry-run preview failed.")
                    return
                self.done.emit(True, "Dry-run preview completed.")
                return

            code = self._run_cmd([str(self.python_exe), "-m", "pip", "install", "--upgrade", "pip"])
            if code != 0:
                self.done.emit(False, "Failed to upgrade pip.")
                return

            if self.repair_runtime:
                self.log_line.emit("[Setup] Runtime health check failed. Repairing scientific stack...")
                code = self._run_cmd(
                    [
                        str(self.python_exe),
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "--force-reinstall",
                        "--no-cache-dir",
                        "--only-binary=:all:",
                        "numpy",
                        "scipy",
                        "pandas<3",
                        "h5py",
                        "pyarrow",
                    ]
                )
                if code != 0:
                    self.done.emit(False, "Runtime repair install failed.")
                    return

            code = self._run_cmd([str(self.python_exe), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])
            if code != 0:
                self.done.emit(False, "Dependency installation failed.")
                return

            code = self._run_cmd([str(self.python_exe), "-m", "pip", "check"])
            if code != 0:
                self.done.emit(False, "Dependency compatibility check failed.")
                return

            runtime_error = _runtime_health_error()
            if runtime_error is not None:
                self.done.emit(False, f"Runtime validation failed after install:\n{runtime_error}")
                return

            SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REQUIREMENTS_FILE, SNAPSHOT_FILE)
            self.done.emit(True, "Dependencies installed/updated successfully.")
        except Exception as exc:  # pragma: no cover - defensive UI path
            self.done.emit(False, f"Unexpected error: {exc}")


class RuntimeHealthWorker(QtCore.QThread):
    done = QtCore.pyqtSignal(object)

    def run(self) -> None:
        self.done.emit(_runtime_health_error())


class LauncherWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.worker: SetupWorker | None = None
        self.runtime_worker: RuntimeHealthWorker | None = None
        self.python_exe = _venv_python()
        self._autolaunch_attempted = False
        self._last_url: str | None = None
        self._runtime_error: str | None = None
        self._can_launch = False
        self._build_ui()
        self._refresh_state()
        QtCore.QTimer.singleShot(0, self._start_runtime_health_check)

    def _build_ui(self) -> None:
        self.setStyleSheet(_STYLE_SHEET)
        self.setWindowTitle("TraceISO Setup + Launcher")
        self.resize(860, 620)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("TraceISO Setup + Launcher")
        title.setStyleSheet("font-size: 22px; font-weight: 600; color: #1a1715; background: transparent;")
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "First run installs dependencies. Later runs launch directly."
        )
        subtitle.setStyleSheet("color: #7a7269; background: transparent;")
        layout.addWidget(subtitle)

        self.status_label = QtWidgets.QLabel("")
        layout.addWidget(self.status_label)

        req_group = QtWidgets.QGroupBox("Packages from requirements.txt")
        req_layout = QtWidgets.QVBoxLayout(req_group)
        self.requirements_view = QtWidgets.QPlainTextEdit()
        self.requirements_view.setReadOnly(True)
        req_layout.addWidget(self.requirements_view)
        layout.addWidget(req_group, 2)

        log_group = QtWidgets.QGroupBox("Setup Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_group, 3)

        btn_row = QtWidgets.QHBoxLayout()
        self.preview_btn = QtWidgets.QPushButton("Preview Install Plan")
        self.install_btn = QtWidgets.QPushButton("Install / Update")
        self.install_btn.setProperty("class", "primary")
        self.launch_btn = QtWidgets.QPushButton("Launch TraceISO")
        self.launch_btn.setProperty("class", "primary")
        self.open_browser_btn = QtWidgets.QPushButton("Open in Browser")
        self.open_browser_btn.setEnabled(False)
        self.global_uc_btn = QtWidgets.QPushButton("Global UC Values")
        self.close_btn = QtWidgets.QPushButton("Close")
        btn_row.addWidget(self.preview_btn)
        btn_row.addWidget(self.install_btn)
        btn_row.addWidget(self.launch_btn)
        btn_row.addWidget(self.open_browser_btn)
        btn_row.addWidget(self.global_uc_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

        self.preview_btn.clicked.connect(self._preview_install_plan)
        self.install_btn.clicked.connect(self._install_or_update)
        self.launch_btn.clicked.connect(self._launch_traceiso)
        self.open_browser_btn.clicked.connect(self._open_last_url)
        self.global_uc_btn.clicked.connect(self._launch_global_uncertainty_manager)
        self.close_btn.clicked.connect(self.close)

    def _refresh_state(self) -> None:
        if REQUIREMENTS_FILE.exists():
            self.requirements_view.setPlainText(REQUIREMENTS_FILE.read_text(encoding="utf-8", errors="replace"))
        else:
            self.requirements_view.setPlainText("requirements.txt not found.")

        requirements_changed = _requirements_changed()
        self._can_launch = False
        if requirements_changed:
            self.status_label.setText("Status: setup required (dependencies not synced).")
            self.status_label.setStyleSheet("color: #c2410c; font-weight: 600; font-size: 14px; background: transparent;")
            self.launch_btn.setEnabled(False)
        else:
            self.status_label.setText("Status: checking runtime...")
            self.status_label.setStyleSheet("color: #7a7269; font-weight: 600; font-size: 14px; background: transparent;")
            self.launch_btn.setEnabled(False)
        
        # Force re-evaluation of QSS for buttons since their properties or states changed
        self.install_btn.style().unpolish(self.install_btn)
        self.install_btn.style().polish(self.install_btn)
        self.launch_btn.style().unpolish(self.launch_btn)
        self.launch_btn.style().polish(self.launch_btn)

    def _start_runtime_health_check(self) -> None:
        if _requirements_changed():
            return
        if self.runtime_worker is not None and self.runtime_worker.isRunning():
            return
        self.runtime_worker = RuntimeHealthWorker(self)
        self.runtime_worker.done.connect(self._on_runtime_health_done)
        self.runtime_worker.start()

    def _on_runtime_health_done(self, error: object) -> None:
        self._runtime_error = error if isinstance(error, str) else None
        self._can_launch = self._runtime_error is None and not _requirements_changed()
        if self._runtime_error:
            self.status_label.setText("Status: runtime repair required (package import check failed).")
            self.status_label.setStyleSheet("color: #c2410c; font-weight: 600; font-size: 14px; background: transparent;")
        else:
            self.status_label.setText("Status: ready. Dependencies are up to date.")
            self.status_label.setStyleSheet("color: #15803d; font-weight: 600; font-size: 14px; background: transparent;")
        self.launch_btn.setEnabled(self._can_launch)
        QtCore.QTimer.singleShot(0, self._auto_launch_if_ready)

    def _auto_launch_if_ready(self) -> None:
        if self._autolaunch_attempted:
            return
        self._autolaunch_attempted = True
        if self._can_launch:
            self._launch_traceiso(auto=True)

    def _set_busy(self, busy: bool) -> None:
        self.preview_btn.setEnabled(not busy)
        self.install_btn.setEnabled(not busy)
        self.launch_btn.setEnabled(not busy and self._can_launch)
        self.close_btn.setEnabled(not busy)

    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _open_url(self, url: str) -> bool:
        if QtGui.QDesktopServices.openUrl(QtCore.QUrl(url)):
            return True
        if webbrowser.open(url, new=2):
            return True
        try:
            subprocess.Popen(["cmd", "/c", "start", "", url], cwd=str(APP_ROOT))
            return True
        except Exception:
            return False

    def _open_last_url(self) -> None:
        if not self._last_url:
            return
        if not self._open_url(self._last_url):
            QtWidgets.QMessageBox.warning(self, "Open Browser Failed", f"Could not open:\n{self._last_url}")

    def _launch_global_uncertainty_manager(self) -> None:
        try:
            self._start_owned_launch(
                [str(self.python_exe), "-m", "tools.global_uncertainty_manager"],
                title="Global Uncertainty Manager",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Launch Failed",
                f"Could not launch Global Uncertainty Values Manager:\n{exc}",
            )

    def _start_worker(self, *, preview_only: bool, repair_runtime: bool = False) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        self.worker = SetupWorker(preview_only=preview_only, repair_runtime=repair_runtime, parent=self)
        self.worker.log_line.connect(self._append_log)
        self.worker.done.connect(self._on_worker_done)
        self._set_busy(True)
        self.worker.start()

    def _preview_install_plan(self) -> None:
        self._append_log("")
        self._append_log("[Setup] Running pip dry-run preview...")
        self._start_worker(preview_only=True, repair_runtime=False)

    def _install_or_update(self) -> None:
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Install Dependencies",
            "Proceed with dependency installation/update now?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return

        self._append_log("")
        self._append_log("[Setup] Installing/updating dependencies...")
        self._start_worker(preview_only=False, repair_runtime=self._runtime_error is not None)

    def _on_worker_done(self, ok: bool, message: str) -> None:
        self._append_log(f"[Setup] {message}")
        self._set_busy(False)
        self._refresh_state()
        if ok:
            self._start_runtime_health_check()
        if not ok:
            QtWidgets.QMessageBox.warning(self, "TraceISO Setup", message)
        else:
            QtWidgets.QMessageBox.information(self, "TraceISO Setup", message)

    def _launch_traceiso(self, auto: bool = False) -> None:
        if not self._can_launch:
            QtWidgets.QMessageBox.warning(
                self,
                "TraceISO Setup Required",
                "Dependencies/runtime are not ready. Install/update first.",
            )
            return

        port = _find_free_port()
        import uuid
        token = 'traceiso-' + uuid.uuid4().hex
        cmd = [str(self.python_exe), "-m", "streamlit", "run", "TraceISO.py", "--server.port", str(port),
               "--server.address", "127.0.0.1", "--server.headless", "true", "--server.baseUrlPath", token]
        url = f"http://127.0.0.1:{port}/{token}"

        try:
            self._start_owned_launch(cmd, title="TraceISO", url=url)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Launch Failed", f"Could not launch TraceISO:\n{exc}")
            return

        return

    def _start_owned_launch(self, command, *, title, url=None):
        from tools.desktop_startup import OwnedLaunch
        launches = getattr(self, '_owned_launches', [])
        self._owned_launches = launches
        if any(name == title and not child.closed and child.process.poll() is None for name, child in launches):
            self._append_log(f'{title} is already starting or running.')
            return
        child = OwnedLaunch(command, cwd=str(APP_ROOT), health_url=url+'/_stcore/health' if url else None)
        launches.append((title, child))
        self._append_log(f'{title} runtime log: {child.log_path}')
        timer = QtCore.QTimer(self)
        child.timer = timer
        is_ready = False
        def poll():
            nonlocal is_ready
            if child.process.poll() is not None:
                code = child.process.returncode
                timer.stop(); child.close()
                if code or not is_ready:
                    QtWidgets.QMessageBox.warning(self, 'Launch Failed', f'{title} exited ({code}).\nLog: {child.log_path}')
                return
            if not is_ready and child.ready():
                is_ready = True
                self._append_log(f'{title} ready.')
                if url:
                    self._last_url = url
                    self.open_browser_btn.setEnabled(True)
                    self._open_url(url)
            if not is_ready and time.monotonic()-child.started > 60:
                timer.stop(); child.close()
                QtWidgets.QMessageBox.warning(self, 'Launch Failed', f'{title} readiness timed out; child stopped.\nLog: {child.log_path}')
        timer.timeout.connect(poll)
        timer.start(250)

    def closeEvent(self, event):
        for title, child in getattr(self, '_owned_launches', []):
            child.timer.stop()
            child.close()
        event.accept()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

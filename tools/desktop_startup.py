"""Console-free desktop supervisor with bounded logs and startup observation."""

from __future__ import annotations

import importlib
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import uuid

TOOLS = {
    "crm": ("CRM Library Manager", "tools.crm_manager.__main__"),
    "extractor": ("Neptune Data Extractor", "tools.neptune_data_extractor.main"),
    "uncertainty": ("Global Uncertainty Manager", "tools.global_uncertainty_manager.__main__"),
}


class _ProcessTree:
    """Own descendants, including interpreter redirectors, without PID searches.

    Windows children wait on a pipe until assigned to a kill-on-close Job Object.
    This prevents a fast child from spawning outside the job before assignment.
    POSIX children start in a private session/process group.
    """

    def __init__(self, command, **kwargs):
        self.job = None
        self.closed = False
        if os.name != "nt":
            self.process = subprocess.Popen(command, start_new_session=True, **kwargs)
            return
        import ctypes
        from ctypes import wintypes as w

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", w.DWORD), ("min_ws", ctypes.c_size_t),
                        ("max_ws", ctypes.c_size_t), ("active", w.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", w.DWORD),
                        ("scheduling", w.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", ctypes.c_uint64 * 6),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        kernel.SetInformationJobObject.restype = w.BOOL
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.AssignProcessToJobObject.restype = w.BOOL
        kernel.CloseHandle.argtypes = [w.HANDLE]
        kernel.CloseHandle.restype = w.BOOL
        self._kernel = kernel
        self.job = kernel.CreateJobObjectW(None, None)
        if not self.job:
            raise ctypes.WinError(ctypes.get_last_error())
        self.process = None
        try:
            limits = ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            # -c avoids importing the application before the ownership barrier.
            bootstrap = ("import subprocess,sys; "
                         "token=sys.stdin.buffer.read(1); "
                         "sys.exit(subprocess.call(sys.argv[1:],creationflags=subprocess.CREATE_NO_WINDOW,"
                         "stdin=subprocess.DEVNULL,stdout=sys.stdout,stderr=sys.stderr) "
                         "if token==b'G' else 125)")
            self.process = subprocess.Popen(
                [sys.executable, "-c", bootstrap, *command], stdin=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW, **kwargs)
            if not kernel.AssignProcessToJobObject(self.job, int(self.process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            self.process.stdin.write(b"G")
            self.process.stdin.close()
        except BaseException:
            # Assignment failure never releases the target command.
            if self.process is not None:
                if self.process.stdin and not self.process.stdin.closed:
                    self.process.stdin.close()
                if self.process.poll() is None:
                    self.process.terminate()
                self.process.wait(timeout=5)
                if self.process.stdout:
                    self.process.stdout.close()
            kernel.CloseHandle(self.job)
            self.job = None
            raise

    def close(self):
        if self.closed:
            return
        if os.name == "nt":
            if self.job:
                self._kernel.CloseHandle(self.job)
                self.job = None
        else:
            import signal
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            # A parent may exit before a descendant that ignores SIGTERM.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process.wait(timeout=5)
        self.closed = True


def mark_window_ready():
    """Called by the first event-loop turn after the main window is shown."""
    target = os.environ.get("TRACEISO_STARTUP_READY")
    if target:
        Path(target).write_text("ready", encoding="ascii")


def show_failure(title, message, log_path):
    text = (f"{message}\n\nStartup log: {log_path}\n\n"
            "For missing dependencies, run 00_Setup_Update_Dependencies.bat.\n"
            "For invalid library data, inspect the log before replacing files.")
    try:
        import tkinter as tk
        root = tk.Tk()
        root.title(f"{title} — startup problem")
        tk.Label(root, text=text, justify="left", wraplength=640, padx=20, pady=20).pack()
        def open_log():
            try:
                if os.name == "nt":
                    os.startfile(str(log_path))
                else:
                    subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(log_path)])
            except OSError:
                pass
        tk.Button(root, text="Open Log", command=open_log).pack(side="left", padx=20, pady=12)
        tk.Button(root, text="Close", command=root.destroy).pack(side="right", padx=20, pady=12)
        root.mainloop()
    except Exception:
        if os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, text, title, 0x10)
        elif sys.stderr:
            print(text, file=sys.stderr)


def supervise(tool, arguments=()):
    title, module = TOOLS[tool]
    root = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "TraceISO" / "logs"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        root = Path(tempfile.mkdtemp(prefix="traceiso-startup-"))
    token = uuid.uuid4().hex
    log_path = root / f"{tool}-{time.strftime('%Y%m%d-%H%M%S')}-{token[:8]}.log"
    # Retain ten runs for this tool; rotation also bounds each run's output.
    for old in sorted(root.glob(f"{tool}-*.log"), key=lambda p: p.stat().st_mtime)[:-9]:
        try:
            old.unlink()
            old.with_name(old.name + ".1").unlink(missing_ok=True)
        except OSError:
            pass
    logger = logging.getLogger(f"desktop-startup-{token}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=1, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    ready = root / f"ready-{token}"
    env = dict(os.environ, TRACEISO_STARTUP_READY=str(ready), PYTHONUNBUFFERED="1")
    command = [sys.executable, "-m", "tools.desktop_startup", "--child", tool, *arguments]
    logger.info("Starting %s with %s; Qt platform=%s", title, sys.executable, env.get("QT_QPA_PLATFORM", "system default"))
    process = None
    ownership = None
    try:
        ownership = _ProcessTree(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        process = ownership.process
        def capture():
            while True:
                chunk = process.stdout.read1(4096)
                if not chunk:
                    return
                logger.info("%s", chunk.decode("utf-8", errors="replace").rstrip())
        reader = threading.Thread(target=capture, daemon=True)
        reader.start()
        deadline = time.monotonic() + 60
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if not ready.exists():
            if process.poll() is None:
                # Avoid leaving a second, hidden startup attempt running.
                ownership.close()
                reason = "No main-window readiness signal within 60 seconds. The startup attempt was stopped."
            else:
                reason = f"Application exited before opening its main window (exit code {process.returncode})."
            reader.join(timeout=2)
            logger.error(reason)
            show_failure(title, reason, log_path)
            return 1
        logger.info("Main window ready")
        code = process.wait()
        reader.join(timeout=2)
        if code:
            show_failure(title, f"Application exited unexpectedly (exit code {code}).", log_path)
        return code
    except Exception as exc:
        logger.exception("Startup failed")
        show_failure(title, str(exc), log_path)
        return 1
    finally:
        if ownership is not None:
            ownership.close()
        ready.unlink(missing_ok=True)
        handler.close()
        logger.removeHandler(handler)


class OwnedLaunch:
    """One child, a private readiness capability, and bounded output retained on exit."""
    def __init__(self, command, *, cwd, health_url=None):
        self.directory = Path(tempfile.mkdtemp(prefix='traceiso-launch-'))
        self.ready_path = self.directory/'ready'
        self.log_path = self.directory/'runtime.log'
        self.health_url = health_url
        self.started = time.monotonic()
        self.logger = logging.getLogger('owned-launch-'+self.directory.name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.handler = RotatingFileHandler(self.log_path, maxBytes=1_000_000, backupCount=1, encoding='utf-8')
        self.logger.addHandler(self.handler)
        try:
            self._ownership = _ProcessTree(command, cwd=cwd,
                env=dict(os.environ, TRACEISO_STARTUP_READY=str(self.ready_path), PYTHONUNBUFFERED='1'),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.process = self._ownership.process
        except BaseException:
            self.handler.close(); self.logger.removeHandler(self.handler)
            raise
        def capture():
            for chunk in iter(lambda: self.process.stdout.read1(4096), b''):
                self.logger.info('%s', chunk.decode('utf-8', errors='replace'))
        self.reader = threading.Thread(target=capture, daemon=True)
        self.reader.start()
        self.closed = False

    def ready(self):
        if self.process.poll() is not None: return False
        if self.health_url:
            from urllib.request import urlopen
            try:
                with urlopen(self.health_url, timeout=.3) as reply:
                    return reply.status == 200 and reply.read(16).strip()==b'ok'
            except OSError:
                return False
        try:
            return self.ready_path.is_file() and self.ready_path.read_text()=='ready'
        except OSError:
            return False

    def close(self):
        if self.closed: return
        self._ownership.close()
        self.reader.join(timeout=2)
        self.process.stdout.close()
        self.handler.close(); self.logger.removeHandler(self.handler)
        self.ready_path.unlink(missing_ok=True)
        self.closed = True


def main():
    args = sys.argv[1:]
    if args and args[0] == "--child":
        tool = args[1]
        sys.argv = [TOOLS[tool][1], *args[2:]]
        importlib.import_module(TOOLS[tool][1]).main()
        return 0
    if not args or args[0] not in TOOLS:
        return 2
    return supervise(args[0], args[1:])


if __name__ == "__main__":
    raise SystemExit(main())

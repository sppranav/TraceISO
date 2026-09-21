"""Launch the Global Uncertainty Values Manager."""

from __future__ import annotations

import sys
from pathlib import Path

from tools.qt_runtime import configure_qt_runtime

configure_qt_runtime()

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from tools.global_uncertainty_manager.app import GlobalUncertaintyManagerWindow


def main() -> None:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("TraceISO — Global Uncertainty Manager")
    app.setOrganizationName("TraceISO")
    from tools.desktop_theme import apply
    apply(app)

    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    window = GlobalUncertaintyManagerWindow(json_path=json_path)
    window.show()
    from PyQt5.QtCore import QTimer
    from tools.desktop_startup import mark_window_ready
    QTimer.singleShot(0, mark_window_ready)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

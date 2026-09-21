"""Launch the CRM & Reference Data Manager: ``python -m tools.crm_manager``."""

import sys
from pathlib import Path

from tools.qt_runtime import configure_qt_runtime

configure_qt_runtime()

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication

from tools.crm_manager.app import CRMManagerWindow


def main() -> None:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("TraceISO — CRM Library Manager")
    app.setOrganizationName("TraceISO")
    from tools.desktop_theme import apply
    apply(app)

    # Accept an optional path argument
    json_path = None
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])

    window = CRMManagerWindow(json_path=json_path)
    window.show()
    from tools.desktop_startup import mark_window_ready
    QTimer.singleShot(0, mark_window_ready)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.qt_runtime import configure_qt_runtime

configure_qt_runtime()

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QTimer

from tools import desktop_theme as theme
from tools.neptune_data_extractor.ui.main_window import NeptuneDataExtractorWindow

def main():
    # Enable High DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("TraceISO — Neptune Data Extractor")
    app.setOrganizationName("TraceISO")
    theme.apply(app)

    window = NeptuneDataExtractorWindow()
    window.show()
    from tools.desktop_startup import mark_window_ready
    QTimer.singleShot(0, mark_window_ready)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

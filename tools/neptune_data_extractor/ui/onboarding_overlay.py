"""Onboarding overlay shown when no file is loaded."""

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt


class OnboardingOverlay(QWidget):
    """Translucent card shown over the empty data grid."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("""
            OnboardingOverlay {
                background: rgba(255, 255, 255, 235);
                border: 1px solid #e0e4e9;
                border-radius: 14px;
            }
            QLabel { background: transparent; }
        """)

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setContentsMargins(40, 30, 40, 30)

        title = QLabel(
            "<b style='font-size:16px; color:#0e6b8a;'>Getting Started</b>"
        )
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        steps = QLabel(
            "<ol style='font-size:13px; line-height:1.8;'>"
            "<li>Load a sample file or <b>drag &amp; drop</b> one here</li>"
            "<li>Right-click cells to <b>map metadata</b> and columns</li>"
            "<li>Use the <b>Preview</b> panel to verify extraction</li>"
            "<li>Load your full folder and click <b>Convert</b>!</li>"
            "</ol>"
        )
        steps.setWordWrap(True)
        steps.setAlignment(Qt.AlignLeft)
        lay.addWidget(steps)

        tip = QLabel(
            "<i style='color:#888; font-size:11px;'>"
            "Tip: Save your mapping as a Template to reuse it later.</i>"
        )
        tip.setAlignment(Qt.AlignCenter)
        lay.addWidget(tip)

        self.setFixedSize(420, 260)

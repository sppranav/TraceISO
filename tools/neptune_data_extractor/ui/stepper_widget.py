"""Horizontal workflow stepper widget for the Neptune Data Extractor."""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from tools.neptune_data_extractor.ui import theme


class StepperWidget(QWidget):
    """Horizontal step indicator with click support."""

    step_clicked = pyqtSignal(int)

    STEPS = [
        "Load File",
        "Set Header",
        "Map Columns",
        "Map Metadata",
        "Preview",
        "Convert",
    ]

    PENDING = 0
    ACTIVE = 1
    DONE = 2

    # fill, text, ring
    _COLORS = {
        PENDING: (QColor("#e9edf1"), QColor(theme.INK_FAINT), QColor("#dfe3e8")),
        ACTIVE: (QColor(theme.PAPER), QColor(theme.ACCENT), QColor(theme.ACCENT)),
        DONE: (QColor(theme.OK), QColor(theme.PAPER), QColor(theme.OK)),
    }
    _LINE_DONE = QColor(theme.OK)
    _LINE_PENDING = QColor(theme.BORDER)
    _LABEL = QColor(theme.INK_DIM)
    _LABEL_ACTIVE = QColor(theme.ACCENT)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._states = [self.PENDING] * len(self.STEPS)
        self.setFixedHeight(58)
        self.setMouseTracking(True)
        self._hover = -1

    def update_from_mapping(
        self, *, has_file, has_header, has_columns, has_metadata, has_preview
    ):
        """Bulk-update step states from current mapping progress."""
        flags = [has_file, has_header, has_columns, has_metadata, has_preview, False]
        first_incomplete = None
        for i, ok in enumerate(flags):
            if ok:
                self._states[i] = self.DONE
            elif first_incomplete is None:
                first_incomplete = i
                self._states[i] = self.ACTIVE
            else:
                self._states[i] = self.PENDING
        self.update()

    def mark_converted(self):
        self._states[-1] = self.DONE
        self.update()

    def mark_conversion_incomplete(self):
        """Show that the latest conversion did not complete successfully."""
        self._states[-1] = self.ACTIVE
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        step_count = len(self.STEPS)
        step_width = self.width() / step_count
        radius = 13
        center_y = 20

        for index, label in enumerate(self.STEPS):
            state = self._states[index]
            fill, foreground, ring = self._COLORS[state]
            if index == self._hover and state != self.DONE:
                fill = fill.lighter(104)

            center_x = int(index * step_width + step_width / 2)

            if index < step_count - 1:
                next_x = int((index + 1) * step_width + step_width / 2)
                line_color = (
                    self._LINE_DONE if state == self.DONE else self._LINE_PENDING
                )
                painter.setPen(QPen(line_color, 2))
                painter.drawLine(
                    center_x + radius + 3,
                    center_y,
                    next_x - radius - 3,
                    center_y,
                )

            painter.setBrush(fill)
            painter.setPen(QPen(ring, 2 if state == self.ACTIVE else 1))
            painter.drawEllipse(
                center_x - radius,
                center_y - radius,
                2 * radius,
                2 * radius,
            )

            painter.setPen(foreground)
            painter.setFont(QFont(theme.ui_family(), 9, QFont.Bold))
            text = "\u2713" if state == self.DONE else str(index + 1)
            painter.drawText(
                center_x - radius,
                center_y - radius,
                2 * radius,
                2 * radius,
                Qt.AlignCenter,
                text,
            )

            painter.setPen(
                self._LABEL_ACTIVE if state == self.ACTIVE else self._LABEL
            )
            label_font = QFont(theme.ui_family(), 8)
            label_font.setBold(state == self.ACTIVE)
            painter.setFont(label_font)
            painter.drawText(
                int(index * step_width),
                38,
                int(step_width),
                16,
                Qt.AlignCenter,
                label,
            )

        painter.end()

    def mousePressEvent(self, event):
        index = self._hit(event.pos())
        if index >= 0:
            self.step_clicked.emit(index)

    def mouseMoveEvent(self, event):
        index = self._hit(event.pos())
        if index != self._hover:
            self._hover = index
            self.update()

    def leaveEvent(self, _event):
        self._hover = -1
        self.update()

    def _hit(self, position):
        step_width = self.width() / len(self.STEPS)
        index = int(position.x() / step_width)
        return index if 0 <= index < len(self.STEPS) else -1

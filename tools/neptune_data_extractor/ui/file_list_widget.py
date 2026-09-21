"""Enhanced file list with count badge, validation icons, and context menu."""

import os
import subprocess
import sys

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
    QMenu, QGroupBox, QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap, QPainter, QIcon, QFont, QPen

from tools.neptune_data_extractor.ui import theme


def _reveal_label() -> str:
    """Menu wording for revealing a file, per platform file manager."""
    if sys.platform == "win32":
        return "Open in Explorer"
    if sys.platform == "darwin":
        return "Reveal in Finder"
    return "Open Containing Folder"


class EnhancedFileList(QWidget):
    """File list with count badge, per-file validation status, and right-click."""

    file_selected = pyqtSignal(str)
    files_changed = pyqtSignal()

    UNKNOWN = 0
    VALID = 1
    INVALID = 2
    ACTIVE = 3          # currently loaded/viewed file

    _ACTIVE_BG = QColor(theme.ACCENT_SOFT)
    _ACTIVE_FG = QColor(theme.ACCENT)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._status = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.group = QGroupBox("Files (0)")
        gl = QVBoxLayout(self.group)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._ctx_menu)
        self.list_widget.itemClicked.connect(
            lambda item: self.file_selected.emit(item.data(Qt.UserRole))
        )
        gl.addWidget(self.list_widget)
        lay.addWidget(self.group)

    # ---- public API ----
    def add_file(self, filepath):
        for i in range(self.list_widget.count()):
            if self.list_widget.item(i).data(Qt.UserRole) == filepath:
                return
        item = QListWidgetItem()
        bn = os.path.basename(filepath)
        sz = self._fmt_size(os.path.getsize(filepath)) if os.path.exists(filepath) else "?"
        item.setText(f"{bn}  ({sz})")
        item.setData(Qt.UserRole, filepath)
        item.setToolTip(filepath)
        item.setIcon(self._icon(self.UNKNOWN))
        self.list_widget.addItem(item)
        self._refresh_title()

    def add_files(self, paths):
        self.list_widget.clear()
        self._status.clear()
        for p in paths:
            self.add_file(p)
        self._refresh_title()
        self.files_changed.emit()

    def set_status(self, filepath, status):
        self._status[filepath] = status
        for i in range(self.list_widget.count()):
            it = self.list_widget.item(i)
            if it.data(Qt.UserRole) == filepath:
                it.setIcon(self._icon(status))
                break

    def clear_status(self):
        self._status.clear()
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setIcon(self._icon(self.UNKNOWN))

    def mark_active(self, filepath):
        """Highlight the currently loaded file; reset all others."""
        bold_font = QFont()
        bold_font.setBold(True)
        normal_font = QFont()
        for i in range(self.list_widget.count()):
            it = self.list_widget.item(i)
            is_active = it.data(Qt.UserRole) == filepath
            it.setFont(bold_font if is_active else normal_font)
            it.setForeground(
                self._ACTIVE_FG if is_active else QColor(theme.INK)
            )
            it.setBackground(
                self._ACTIVE_BG if is_active else QColor(theme.PAPER)
            )

    def all_paths(self):
        return [self.list_widget.item(i).data(Qt.UserRole)
                for i in range(self.list_widget.count())]

    def count(self):
        return self.list_widget.count()

    def clear(self):
        self.list_widget.clear()
        self._status.clear()
        self._refresh_title()
        self.files_changed.emit()

    # ---- internals ----
    def _ctx_menu(self, pos):
        menu = QMenu(self)
        sel = self.list_widget.selectedItems()
        if sel:
            a = menu.addAction(f"Remove {len(sel)} file(s)")
            a.triggered.connect(self._remove_selected)
            menu.addSeparator()
            if len(sel) == 1:
                path = sel[0].data(Qt.UserRole)
                menu.addAction(_reveal_label()).triggered.connect(
                    lambda: self._open_in_explorer(path)
                )
        menu.addSeparator()
        menu.addAction("Clear All").triggered.connect(self.clear)
        menu.exec_(self.list_widget.mapToGlobal(pos))

    def _remove_selected(self):
        for it in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(it))
        self._refresh_title()
        self.files_changed.emit()

    def _refresh_title(self):
        self.group.setTitle(f"Files ({self.list_widget.count()})")

    @staticmethod
    def _open_in_explorer(filepath):
        """Reveal a file in the platform file manager.

        Wrapped in try/except on purpose: an unhandled exception raised inside a
        Qt slot aborts the whole application, so a missing file manager must not
        take the extractor down with unsaved work in it.
        """
        target = os.path.abspath(filepath)
        folder = os.path.dirname(target)
        if sys.platform == "win32":
            command = ["explorer", "/select,", target]
        elif sys.platform == "darwin":
            command = ["open", "-R", target]
        else:
            command = ["xdg-open", folder]
        try:
            subprocess.Popen(command)
        except OSError:
            # No file manager available (or the helper binary is missing).
            QMessageBox.information(
                None,
                "Show In Folder",
                "Could not open a file manager on this system.\n\nFile:\n" + target,
            )

    @staticmethod
    def _fmt_size(b):
        for u in ("B", "KB", "MB"):
            if b < 1024:
                return f"{b:.0f} {u}"
            b /= 1024
        return f"{b:.1f} GB"

    @staticmethod
    def _icon(status):
        pm = QPixmap(12, 12)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        if status == 0:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(theme.INK_FAINT), 1.5))
            p.drawEllipse(2, 2, 8, 8)
        else:
            fills = {1: QColor(theme.OK), 2: QColor(theme.BAD)}
            p.setBrush(fills.get(status, QColor(theme.INK_FAINT)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(1, 1, 10, 10)
        p.end()
        return QIcon(pm)

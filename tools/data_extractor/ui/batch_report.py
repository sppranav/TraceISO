"""Persistent batch outcomes, separate from the editable source-file list."""

import csv
import io
from pathlib import Path

from PyQt5.QtCore import QUrl, pyqtSignal, Qt
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTextEdit, QFileDialog, QMessageBox, QSizePolicy,
)


class BatchReport(QDialog):
    retry_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Conversion results")
        self.resize(900, 560)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        # Unbroken output paths must not impose an off-screen minimum width.
        # QLabel still wraps within the available width; the report retains the
        # exact path for copying/exporting.
        self.summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Input file", "Status", "Diagnostic"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details)
        actions = QHBoxLayout()
        for title, handler in (("Copy Report", self.copy_report), ("Export Report...", self.export_report),
                               ("Open Output Folder", self.open_folder)):
            button = QPushButton(title)
            button.clicked.connect(handler)
            actions.addWidget(button)
        self.retry = QPushButton("Retry Failed Files...")
        self.retry.clicked.connect(self.retry_requested)
        actions.addWidget(self.retry)
        layout.addLayout(actions)
        self.output = ""
        self.rows = []

    def set_outcome(self, outcome, paths, verification):
        self.output = outcome.output_path
        errors_by_path = dict(zip(outcome.failed, outcome.errors))
        self.rows = [(str(path), "Failed" if path in outcome.failed else
                      "Converted" if path in outcome.processed else "Not attempted", errors_by_path.get(path, "")) for path in paths]
        self.table.setRowCount(len(self.rows))
        for row, values in enumerate(self.rows):
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.resizeColumnToContents(0)
        self.table.setColumnWidth(0, min(400, self.table.columnWidth(0)))
        self.summary.setText(f"{outcome.completion_status or 'Batch finished'} — {outcome.sample_count} samples\nOutput: {self.output}\n{verification}")
        self.details.setPlainText("\n".join(outcome.errors) or "No conversion errors.")
        self.retry.setEnabled(bool(outcome.failed))

    def report_text(self):
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["Input file", "Status", "Diagnostic"])
        writer.writerows(self.rows)
        return self.summary.text() + "\n\n" + stream.getvalue() + "\n" + self.details.toPlainText()

    def copy_report(self):
        QApplication.clipboard().setText(self.report_text())

    def export_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export conversion report", "conversion_report.txt", "Text (*.txt)")
        if path:
            try:
                Path(path).write_text(self.report_text(), encoding="utf-8")
            except OSError as exc:
                QMessageBox.warning(self, "Report export failed", str(exc))

    def open_folder(self):
        if self.output:
            if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.output).resolve().parent))):
                QMessageBox.warning(self, "Open folder", "Could not open the output folder.")

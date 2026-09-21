"""Mapping history and user-facing filename presets over existing extraction."""

from copy import deepcopy
from pathlib import Path
import re

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAction, QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QWidget
from tools.neptune_data_extractor.mapper_logic import extract_metadata_from_df


class MappingWorkflow:
    def _init_mapping_workflow(self):
        self._history = []
        self._history_index = -1
        self._restoring_mapping = False
        self._template_name = "Custom mapping"
        self._template_baseline = None
        row = QWidget()
        layout = QHBoxLayout(row)
        self._template_choice = QComboBox()
        self._template_choice.addItem("Custom mapping", None)
        templates = Path(__file__).resolve().parents[1] / "templates"
        for path in sorted(templates.glob("*.json")):
            self._template_choice.addItem(path.stem.replace("_", " "), str(path))
        self._template_choice.setToolTip("Choose an instrument template after loading your first file, or keep Custom mapping.")
        self._template_choice.activated.connect(self._choose_template)
        layout.addWidget(self._template_choice)
        self._mapping_actions = []
        for title, handler in (("Set Header", self._selected_header),
                               ("Set Data End", self._selected_end),
                               ("Map Selected Cells", self._selected_map)):
            button = QPushButton(title)
            button.clicked.connect(handler)
            button.setToolTip("Select a cell or range in the source table first. Data End marks the first excluded footer row.")
            layout.addWidget(button)
            self._mapping_actions.append(button)
        for title, shortcut, offset in (("Undo", "Ctrl+Z", -1), ("Redo", "Ctrl+Shift+Z", 1)):
            button = QPushButton(title)
            button.clicked.connect(lambda checked=False, step=offset: self._history_move(step))
            action = QAction(title, self)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda checked=False, step=offset: self._history_move(step))
            self.addAction(action)
            layout.addWidget(button)
        self.centralWidget().layout().insertWidget(2, row)
        self._template_status = QLabel("Custom mapping")
        self._status_bar.addWidget(self._template_status)
        self.table.itemSelectionChanged.connect(self._mapping_selection_changed)
        self._mapping_selection_changed()
        self._history_boundary()

    def _init_name_presets(self, form):
        self._name_preset = QComboBox()
        self._name_preset.addItems(["Advanced: custom regex", "Full filename", "Before first underscore", "Between underscores"])
        self._name_segment = QSpinBox()
        self._name_segment.setRange(2, 30)
        self._name_segment.setValue(2)
        self._name_segment.setToolTip("Underscore-separated segment number, counting from 1.")
        form.addRow("Sample naming:", self._name_preset)
        form.addRow("Segment:", self._name_segment)
        self._name_segment.hide()
        form.labelForField(self._name_segment).hide()
        self._name_preset.currentIndexChanged.connect(self._apply_name_preset)
        self._name_segment.valueChanged.connect(self._apply_name_preset)
        self._names_preview = QLabel()
        self._names_preview.setWordWrap(True)
        self._names_preview.setTextFormat(Qt.PlainText)
        form.addRow("Filename preview:", self._names_preview)

    def _apply_name_preset(self):
        index = self._name_preset.currentIndex()
        form = self.controls_grp.layout()
        self.combo_name_parse.setVisible(index == 0)
        form.labelForField(self.combo_name_parse).setVisible(index == 0)
        self._name_segment.setVisible(index == 3)
        form.labelForField(self._name_segment).setVisible(index == 3)
        patterns = {1: r"^(.*)$", 2: r"^([^_]+)",
                    3: r"^(?:[^_]*_){" + str(self._name_segment.value() - 1) + r"}([^_]+)(?:_|$)"}
        if index in patterns:
            self.combo_name_parse.setCurrentText(patterns[index])
        self._update_preview()

    def _filename_preview(self):
        pattern = self.combo_name_parse.currentText()
        rows, seen = [], set()
        for path in self.file_list_widget.all_paths():
            try:
                # Filename-only preview. Mapped Sample Name cells can override it.
                name, _ = extract_metadata_from_df(pd.DataFrame(), path, {"name_pattern": pattern})
                flags = []
                if pattern and not re.search(pattern, Path(path).stem):
                    flags.append("no match; filename fallback")
                if not name:
                    flags.append("empty name")
                if name in seen:
                    flags.append("duplicate name")
                seen.add(name)
                rows.append(f"{Path(path).name} → {name}" + (" [" + "; ".join(flags) + "]" if flags else ""))
            except (ValueError, re.error) as exc:
                rows.append(f"{Path(path).name}: {exc}")
        return rows

    def _mapping_snapshot(self):
        return deepcopy((self.mapping, self.header_row_idx, self.footer_row_idx,
                         self.combo_system.currentText(), self.input_instrument.text(), self.combo_name_parse.currentText()))

    def _history_boundary(self):
        if not hasattr(self, "_history"):
            return
        self._history = [self._mapping_snapshot()]
        self._history_index = 0

    def _record_mapping(self):
        if not hasattr(self, "_history") or self._restoring_mapping:
            return
        snapshot = self._mapping_snapshot()
        if not self._history or snapshot != self._history[self._history_index]:
            self._history = self._history[:self._history_index + 1] + [snapshot]
            self._history = self._history[-50:]
            self._history_index = len(self._history) - 1
        modified = self._template_baseline is not None and snapshot != self._template_baseline
        label = Path(self._template_name).name + (" (modified)" if modified else "")
        self._template_status.setText(
            self._template_status.fontMetrics().elidedText(label, Qt.ElideMiddle, 260))
        self._template_status.setToolTip(self._template_name)
        self._names_preview.setText("\n".join(self._filename_preview()[:5]) + "\nFilename rules only; mapped Sample Name cells override these names.")

    def _history_move(self, step):
        target = self._history_index + step
        if not 0 <= target < len(self._history):
            return
        self._restoring_mapping = True
        try:
            mapping, header, footer, system, instrument, pattern = deepcopy(self._history[target])
            self.mapping, self.header_row_idx, self.footer_row_idx = mapping, header, footer
            self.combo_system.setCurrentText(system)
            self.input_instrument.setText(instrument)
            self.combo_name_parse.setCurrentText(pattern)
            self._name_preset.setCurrentIndex(0)
            self._history_index = target
            self._update_ui_labels()
            if self.current_df is not None:
                self._refresh_highlights()
        finally:
            self._restoring_mapping = False
        self._record_mapping()

    def _mapping_selection_changed(self):
        for button in self._mapping_actions:
            button.setEnabled(bool(self.table.selectedItems()))

    def _selected_header(self):
        cells = self.table.selectedItems()
        if cells:
            self.set_header_row(min(c.row() for c in cells))

    def _selected_end(self):
        cells = self.table.selectedItems()
        if cells:
            self.set_footer_row(min(c.row() for c in cells))

    def _selected_map(self):
        cells = self.table.selectedItems()
        if cells:
            self.show_context_menu(self.table.visualItemRect(cells[0]).center())

    def _choose_template(self):
        path = self._template_choice.currentData()
        if path:
            self._load_template_from_path(path)
        else:
            self._template_name = "Custom mapping"
            self._template_baseline = None
            self._record_mapping()

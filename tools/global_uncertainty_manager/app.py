"""PyQt manager for TraceISO global uncertainty values."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QColor
from tools.desktop_theme import style_window, style_toolbar, BAD_SOFT

from config.global_uncertainty_values_loader import (
    GLOBAL_UNCERTAINTY_VALUES_PATH,
    K5_KEY,
    KAPPA_SPECS,
    SR_ENGINE_A_DEFAULT_SPECS,
    ElementGlobalUncertaintyValues,
    GlobalUncertaintyValues,
    KappaDefault,
    SrEngineADefault,
    global_uncertainty_values_revision,
    load_global_uncertainty_values,
    save_global_uncertainty_values,
)
from config.uncertainty_profiles_loader import ProfileConflictError
from config.contributor_names import LABEL_U_K4
from config.settings import CustomUncertaintyContributor
from tools.global_uncertainty_manager_validation import (
    is_valid_dof_cell,
    is_valid_permil_cell,
    parse_dof_cell,
    parse_permil_cell,
)


_KAPPA_ORDER = [
    "k1_sample_decomposition",
    "k2_matrix_separation",
    "k3_procedural_blank",
    "k4_bracketing_standard_heterogeneity",
    K5_KEY,
    "k6_matrix_effects",
    "k7_residual_interferences",
]
_KAPPA_LABELS = {
    "k1_sample_decomposition": "Sample decomposition",
    "k2_matrix_separation": "Matrix separation",
    "k3_procedural_blank": "Procedural blank",
    "k4_bracketing_standard_heterogeneity": LABEL_U_K4,
    K5_KEY: "Instrumental drift",
    "k6_matrix_effects": "Matrix effects",
    "k7_residual_interferences": "Residual interferences",
}
_DISTRIBUTIONS = ["normal", "rectangular"]
_ELEMENTS = ["B", "Li", "Mg", "Cd", "Pb", "Sr"]
_SR_ENGINE_A_ORDER = [
    "u_bias_qc",
    "u_reprod_dig",
    "enable_sr_norm_ratio_uncertainty",
]
_SR_ENGINE_A_LABELS = {
    "u_bias_qc": "Bias in processed control sample",
    "u_reprod_dig": "Between-digestion reproducibility",
    "enable_sr_norm_ratio_uncertainty": "Normalization ratio uncertainty",
}


def _item(text: object = "", *, editable: bool = True, checkable: bool = False) -> QTableWidgetItem:
    item = QTableWidgetItem(str(text))
    flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
    if editable:
        flags |= Qt.ItemIsEditable
    if checkable:
        flags |= Qt.ItemIsUserCheckable
        item.setCheckState(Qt.Checked if bool(text) else Qt.Unchecked)
    item.setFlags(flags)
    return item


class FiniteDoubleDelegate(QStyledItemDelegate):
    """Numeric table delegate that prevents NaN/Inf text entry."""

    def createEditor(self, parent, option, index):  # noqa: N802 - Qt API
        editor = QDoubleSpinBox(parent)
        editor.setDecimals(8)
        editor.setRange(-1.0e12, 1.0e12)
        editor.setSingleStep(0.01)
        return editor

    def setEditorData(self, editor, index):  # noqa: N802 - Qt API
        try:
            editor.setValue(float(index.data() or 0.0))
        except (TypeError, ValueError):
            editor.setValue(0.0)

    def setModelData(self, editor, model, index):  # noqa: N802 - Qt API
        model.setData(index, f"{editor.value():.8g}")


class GlobalUncertaintyManagerWindow(QMainWindow):
    """Main window for editing ``global_uncertainty_values.json``."""

    def __init__(self, json_path: Path | None = None) -> None:
        super().__init__()
        self._json_path = json_path or GLOBAL_UNCERTAINTY_VALUES_PATH
        self._values = self._load_values()
        # The revision this window is editing against; see _save.
        self._loaded_revision = global_uncertainty_values_revision(self._json_path)
        self._current_element = ""
        self._loading = False

        style_window(self, "Global Uncertainty Manager")
        self.resize(980, 720)
        self._build_toolbar()
        self._build_central()
        self._populate_element_combo()

    def _load_values(self) -> GlobalUncertaintyValues:
        try:
            return load_global_uncertainty_values(self._json_path)
        except FileNotFoundError:
            return GlobalUncertaintyValues()

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Global uncertainty values")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._save_action = QAction("Save", self)
        self._save_action.triggered.connect(self._save)
        toolbar.addAction(self._save_action)

        reload_action = QAction("Reload", self)
        reload_action.triggered.connect(self._reload)
        toolbar.addAction(reload_action)

        validate_action = QAction("Validate", self)
        validate_action.triggered.connect(self._validate)
        toolbar.addAction(validate_action)
        style_toolbar(toolbar, primary=self._save_action)

    def _build_central(self) -> None:
        root = QWidget(self)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(12)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Element"))
        self._element_combo = QComboBox()
        self._element_combo.currentTextChanged.connect(self._on_element_changed)
        top_row.addWidget(self._element_combo)
        add_element = QPushButton("Add Element")
        add_element.clicked.connect(self._add_element)
        top_row.addWidget(add_element)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        self._engine_defaults_label = QLabel("SSB kappa defaults")
        self._engine_defaults_label.setProperty("heading", True)
        layout.addWidget(self._engine_defaults_label)
        self._kappa_table = QTableWidget(0, 7)
        self._kappa_table.verticalHeader().setDefaultSectionSize(38)
        self._kappa_table.setAlternatingRowColors(True)
        self._kappa_table.setHorizontalHeaderLabels(
            ["Kappa", "Contributor", "u_rel_permil", "Enabled", "Distribution", "Source", "Effective value in TraceISO"]
        )
        self._kappa_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        for column, width in enumerate((200, 190, 100, 85, 125, 190, 210)):
            self._kappa_table.setColumnWidth(column, width)
        self._kappa_table.setItemDelegateForColumn(2, FiniteDoubleDelegate(self._kappa_table))
        self._kappa_table.itemChanged.connect(self._refresh_validation_state)
        layout.addWidget(self._kappa_table, 2)

        custom_row = QHBoxLayout()
        custom_heading = QLabel("Custom uncertainty contributors")
        custom_heading.setProperty("heading", True)
        custom_row.addWidget(custom_heading)
        custom_row.addStretch(1)
        add_custom = QPushButton("Add")
        add_custom.clicked.connect(self._add_custom_row)
        delete_custom = QPushButton("Delete Selected")
        delete_custom.setProperty("class", "danger")
        delete_custom.clicked.connect(self._delete_custom_rows)
        custom_row.addWidget(add_custom)
        custom_row.addWidget(delete_custom)
        layout.addLayout(custom_row)

        self._custom_table = QTableWidget(0, 9)
        self._custom_table.verticalHeader().setDefaultSectionSize(38)
        self._custom_table.setAlternatingRowColors(True)
        self._custom_table.setHorizontalHeaderLabels(
            [
                "Name",
                "Display Name",
                "Type",
                "u_rel_permil",
                "DoF",
                "Distribution",
                "Enabled",
                "Source",
                "Description",
            ]
        )
        self._custom_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        for column, width in enumerate((130, 150, 80, 100, 80, 120, 85, 160, 220)):
            self._custom_table.setColumnWidth(column, width)
        self._custom_table.setItemDelegateForColumn(3, FiniteDoubleDelegate(self._custom_table))
        self._custom_table.itemChanged.connect(self._refresh_validation_state)
        layout.addWidget(self._custom_table, 3)

        self.setCentralWidget(root)

    def _populate_element_combo(self) -> None:
        symbols = sorted(set(_ELEMENTS) | set(self._values.elements))
        self._loading = True
        self._element_combo.clear()
        self._element_combo.addItems(symbols)
        self._loading = False
        if symbols:
            self._element_combo.setCurrentText(symbols[0])
            self._load_element(symbols[0])

    def _element_values(self, element_symbol: str) -> ElementGlobalUncertaintyValues:
        if element_symbol not in self._values.elements:
            return ElementGlobalUncertaintyValues(element_symbol=element_symbol)
        return self._values.elements[element_symbol]

    def _load_element(self, element_symbol: str) -> None:
        self._loading = True
        self._current_element = element_symbol
        values = self._element_values(element_symbol)
        self._kappa_table.setRowCount(0)
        if element_symbol == "Sr":
            self._engine_defaults_label.setText("Sr Engine A defaults")
            self._kappa_table.setHorizontalHeaderLabels(
                ["Key", "Contributor / control", "Value", "Enabled", "Units", "Source", "Effective value in TraceISO"]
            )
            for key in _SR_ENGINE_A_ORDER:
                self._append_sr_engine_a_row(key, values.sr_engine_a_defaults.get(key))
        else:
            self._engine_defaults_label.setText("SSB kappa defaults")
            self._kappa_table.setHorizontalHeaderLabels(
                ["Kappa", "Contributor", "u_rel_permil", "Enabled", "Distribution", "Source", "Effective value in TraceISO"]
            )
            for key in _KAPPA_ORDER:
                self._append_kappa_row(key, values.ssb_kappa_defaults.get(key))
        self._custom_table.setRowCount(0)
        for contributor in values.custom_contributors:
            self._append_custom_row(contributor)
        self._loading = False
        self._refresh_validation_state()

    def _append_kappa_row(self, key: str, value: KappaDefault | None) -> None:
        row = self._kappa_table.rowCount()
        self._kappa_table.insertRow(row)
        self._kappa_table.setItem(row, 0, _item(key, editable=False))
        self._kappa_table.setItem(row, 1, _item(_KAPPA_LABELS[key], editable=False))
        if key == K5_KEY:
            value = value or KappaDefault(
                enabled=False,
                calculation="standard_sequence",
                distribution="normal",
            )
            self._kappa_table.setItem(row, 2, _item("", editable=False))
            self._kappa_table.setItem(row, 3, _item(value.enabled, editable=False, checkable=True))
            combo = QComboBox()
            combo.addItems(_DISTRIBUTIONS)
            combo.setCurrentText(value.distribution)
            combo.currentTextChanged.connect(
                lambda _value: self._refresh_validation_state()
            )
            self._kappa_table.setCellWidget(row, 4, combo)
            self._kappa_table.setItem(row, 5, _item("", editable=False))
            self._kappa_table.setItem(
                row,
                6,
                _item(
                    "standard sequence: mean(|Δ|)/2" if value.enabled else "disabled",
                    editable=False,
                ),
            )
            return

        spec = KAPPA_SPECS[key]
        value = value or KappaDefault(
            u_rel_permil=spec.default_value,
            enabled=False,
            distribution=spec.default_distribution,
            source="n/a",
        )
        self._kappa_table.setItem(row, 2, _item(value.u_rel_permil))
        self._kappa_table.setItem(row, 3, _item(value.enabled, editable=False, checkable=True))
        combo = QComboBox()
        combo.addItems(_DISTRIBUTIONS)
        combo.setCurrentText(value.distribution)
        self._kappa_table.setCellWidget(row, 4, combo)
        self._kappa_table.setItem(row, 5, _item(value.source))
        effective = f"{value.u_rel_permil:.8g} permil" if value.enabled else "disabled"
        self._kappa_table.setItem(row, 6, _item(effective, editable=False))

    def _append_sr_engine_a_row(self, key: str, value: SrEngineADefault | None) -> None:
        row = self._kappa_table.rowCount()
        self._kappa_table.insertRow(row)
        spec = SR_ENGINE_A_DEFAULT_SPECS[key]
        value = value or SrEngineADefault(
            value=spec.default_value,
            enabled=False,
            units=spec.units,
            source=spec.source,
        )
        self._kappa_table.setItem(row, 0, _item(key, editable=False))
        self._kappa_table.setItem(row, 1, _item(_SR_ENGINE_A_LABELS[key], editable=False))
        self._kappa_table.setItem(row, 2, _item(value.value))
        self._kappa_table.setItem(row, 3, _item(value.enabled, editable=False, checkable=True))
        self._kappa_table.setItem(row, 4, _item(spec.units, editable=False))
        self._kappa_table.setItem(row, 5, _item(value.source or spec.source))
        effective = f"{value.value:.8g} {spec.units}".strip() if value.enabled else "disabled"
        self._kappa_table.setItem(row, 6, _item(effective, editable=False))

    def _append_custom_row(self, contributor: CustomUncertaintyContributor | None = None) -> None:
        row = self._custom_table.rowCount()
        self._custom_table.insertRow(row)
        contributor = contributor or CustomUncertaintyContributor(
            name="u_custom_new",
            display_name="New contributor",
            element_symbol=self._current_element,
            u_rel_permil=0.0,
            type_ab="B",
            degrees_of_freedom=float("inf"),
            distribution="normal",
            reference="",
            description="",
            enabled=True,
        )
        self._custom_table.setItem(row, 0, _item(contributor.name))
        self._custom_table.setItem(row, 1, _item(contributor.display_name))
        type_combo = QComboBox()
        type_combo.addItems(["A", "B"])
        type_combo.setCurrentText(contributor.type_ab)
        self._custom_table.setCellWidget(row, 2, type_combo)
        self._custom_table.setItem(row, 3, _item(contributor.u_rel_permil))
        dof = "inf" if contributor.degrees_of_freedom == float("inf") else contributor.degrees_of_freedom
        self._custom_table.setItem(row, 4, _item(dof))
        dist_combo = QComboBox()
        dist_combo.addItems(_DISTRIBUTIONS)
        dist_combo.setCurrentText(contributor.distribution)
        self._custom_table.setCellWidget(row, 5, dist_combo)
        self._custom_table.setItem(row, 6, _item(contributor.enabled, editable=False, checkable=True))
        self._custom_table.setItem(row, 7, _item(contributor.reference))
        self._custom_table.setItem(row, 8, _item(contributor.description))

    def _on_element_changed(self, element_symbol: str) -> None:
        """Move to another element only if the current one can be captured.

        ``_load_element`` replaces both tables. Running it after a failed
        capture discarded every pending edit on the element being left,
        including the valid ones, and said nothing (audit A082). A refused
        capture now keeps the selection and the tables exactly as they are and
        shows why.
        """
        if self._loading or not element_symbol:
            return
        previous = self._current_element
        if previous and previous != element_symbol:
            if not self._capture_current_element(show_errors=True):
                self._loading = True
                try:
                    self._element_combo.setCurrentText(previous)
                finally:
                    self._loading = False
                return
        self._load_element(element_symbol)

    def _add_element(self) -> None:
        symbol, ok = QInputDialog.getText(self, "Add Element", "Element symbol:")
        if not ok:
            return
        symbol = symbol.strip()
        if not symbol:
            return
        symbol = symbol[:1].upper() + symbol[1:].lower()
        if self._element_combo.findText(symbol) < 0:
            self._element_combo.addItem(symbol)
        self._element_combo.setCurrentText(symbol)

    def _add_custom_row(self) -> None:
        self._append_custom_row()
        self._refresh_validation_state()

    def _delete_custom_rows(self) -> None:
        rows = sorted({index.row() for index in self._custom_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self._custom_table.removeRow(row)
        self._refresh_validation_state()

    def _cell_text(self, table: QTableWidget, row: int, col: int) -> str:
        item = table.item(row, col)
        return item.text().strip() if item is not None else ""

    def _cell_checked(self, table: QTableWidget, row: int, col: int) -> bool:
        item = table.item(row, col)
        return bool(item is not None and item.checkState() == Qt.Checked)

    def _combo_text(self, table: QTableWidget, row: int, col: int) -> str:
        widget = table.cellWidget(row, col)
        return widget.currentText() if isinstance(widget, QComboBox) else self._cell_text(table, row, col)

    @staticmethod
    def _parse_permil(raw: str, field_name: str) -> float:
        """Parse a permil-value cell using the loader's non-negative finite contract.

        Reuses ``parse_permil_cell`` so the Save gate can never accept a value
        the loader would reject on the next load (e.g. negative or non-finite).
        """
        try:
            return parse_permil_cell(raw, field_name)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    def _mark_cell(self, table: QTableWidget, row: int, col: int, invalid: bool) -> None:
        item = table.item(row, col)
        if item is not None:
            if invalid:
                item.setBackground(QColor(BAD_SOFT))
            else:
                item.setBackground(table.palette().base())
                item.setForeground(table.palette().text())

    def _finite_cell(self, table: QTableWidget, row: int, col: int) -> bool:
        raw = self._cell_text(table, row, col)
        return is_valid_permil_cell(raw)

    def _custom_reference_required(self, row: int) -> bool:
        """Require a citation only for contributors with non-zero uncertainty."""
        try:
            return self._parse_permil(self._cell_text(self._custom_table, row, 3), "u_rel") > 0.0
        except ValueError:
            return True

    def _effective_text(self, row: int) -> str:
        """Return what TraceISO will actually use for one defaults row.

        Resolved from the row's own cells with the same parser the capture
        uses, so the column cannot drift from what a save would write. It was
        previously written once when the element was loaded and never again,
        so it kept reporting the loaded magnitude after an edit and a
        successful save (audit A070).
        """
        key = self._cell_text(self._kappa_table, row, 0)
        enabled = self._cell_checked(self._kappa_table, row, 3)
        if self._current_element == "Sr":
            spec = SR_ENGINE_A_DEFAULT_SPECS.get(key)
            if spec is None:
                return ""
            if not enabled:
                return "disabled"
            try:
                value = self._parse_permil(
                    self._cell_text(self._kappa_table, row, 2), "value",
                )
            except ValueError:
                return "invalid value"
            return f"{value:.8g} {spec.units}".strip()
        if key == K5_KEY:
            return "standard sequence: mean(|d|)/2" if enabled else "disabled"
        if not enabled:
            return "disabled"
        try:
            value = self._parse_permil(
                self._cell_text(self._kappa_table, row, 2), "u_rel",
            )
        except ValueError:
            return "invalid value"
        return f"{value:.8g} permil"

    def _refresh_effective_values(self) -> None:
        """Rewrite the effective-value column from the current cells."""
        for row in range(self._kappa_table.rowCount()):
            item = self._kappa_table.item(row, 6)
            if item is None:
                continue
            item.setText(self._effective_text(row))

    def _refresh_validation_state(self) -> None:
        """Highlight invalid cells and disable Save until the table is valid."""
        if getattr(self, "_loading", False):
            return
        if getattr(self, "_refreshing", False):
            return
        self._refreshing = True
        try:
            self._refresh_validation_state_inner()
        finally:
            self._refreshing = False

    def _refresh_validation_state_inner(self) -> None:
        ok = True

        for row in range(self._kappa_table.rowCount()):
            numeric_item = self._kappa_table.item(row, 2)
            editable = bool(numeric_item and numeric_item.flags() & Qt.ItemIsEditable)
            invalid = editable and not self._finite_cell(self._kappa_table, row, 2)
            self._mark_cell(self._kappa_table, row, 2, invalid)
            ok = ok and not invalid

        for row in range(self._custom_table.rowCount()):
            name = self._cell_text(self._custom_table, row, 0)
            if not name:
                continue
            invalid_u = not self._finite_cell(self._custom_table, row, 3)
            self._mark_cell(self._custom_table, row, 3, invalid_u)
            ok = ok and not invalid_u

            dof_raw = self._cell_text(self._custom_table, row, 4) or "inf"
            row_type_ab = self._combo_text(self._custom_table, row, 2)
            invalid_dof = not is_valid_dof_cell(dof_raw, type_ab=row_type_ab)
            self._mark_cell(self._custom_table, row, 4, invalid_dof)
            ok = ok and not invalid_dof

            reference_missing = self._custom_reference_required(row) and not bool(
                self._cell_text(self._custom_table, row, 7)
            )
            self._mark_cell(self._custom_table, row, 7, reference_missing)
            ok = ok and not reference_missing

        self._refresh_effective_values()

        if hasattr(self, "_save_action"):
            self._save_action.setEnabled(ok)

    def _capture_current_element(self, *, show_errors: bool = True) -> bool:
        element_symbol = self._current_element or self._element_combo.currentText()
        if not element_symbol:
            return True
        try:
            kappa_defaults: Dict[str, KappaDefault] = {}
            sr_engine_a_defaults: Dict[str, SrEngineADefault] = {}
            if element_symbol == "Sr":
                for row in range(self._kappa_table.rowCount()):
                    key = self._cell_text(self._kappa_table, row, 0)
                    spec = SR_ENGINE_A_DEFAULT_SPECS[key]
                    sr_engine_a_defaults[key] = SrEngineADefault(
                        value=self._parse_permil(
                            self._cell_text(self._kappa_table, row, 2),
                            f"Row {row + 1} value",
                        ),
                        enabled=self._cell_checked(self._kappa_table, row, 3),
                        units=spec.units,
                        source=self._cell_text(self._kappa_table, row, 5) or spec.source,
                    )
            else:
                for row in range(self._kappa_table.rowCount()):
                    key = self._cell_text(self._kappa_table, row, 0)
                    enabled = self._cell_checked(self._kappa_table, row, 3)
                    if key == K5_KEY:
                        kappa_defaults[key] = KappaDefault(
                            enabled=enabled,
                            calculation="standard_sequence",
                            distribution=self._combo_text(self._kappa_table, row, 4),
                        )
                        continue
                    kappa_defaults[key] = KappaDefault(
                        u_rel_permil=self._parse_permil(
                            self._cell_text(self._kappa_table, row, 2),
                            f"Row {row + 1} u_rel (‰)",
                        ),
                        enabled=enabled,
                        distribution=self._combo_text(self._kappa_table, row, 4),
                        source=self._cell_text(self._kappa_table, row, 5),
                    )

            custom_contributors: List[CustomUncertaintyContributor] = []
            for row in range(self._custom_table.rowCount()):
                name = self._cell_text(self._custom_table, row, 0)
                if not name:
                    continue
                row_type_ab = self._combo_text(self._custom_table, row, 2)
                dof_raw = self._cell_text(self._custom_table, row, 4) or "inf"
                dof = parse_dof_cell(
                    dof_raw,
                    type_ab=row_type_ab,
                    field_name=f"Custom contributor '{name}' degrees_of_freedom",
                )
                reference = self._cell_text(self._custom_table, row, 7)
                u_rel_permil = self._parse_permil(
                    self._cell_text(self._custom_table, row, 3),
                    f"Custom contributor '{name}' u_rel (‰)",
                )
                if u_rel_permil > 0.0 and not reference:
                    raise ValueError(
                        f"Custom contributor '{name}' requires a Source/reference citation."
                    )
                custom_contributors.append(
                    CustomUncertaintyContributor(
                        name=name,
                        display_name=self._cell_text(self._custom_table, row, 1) or name,
                        element_symbol=element_symbol,
                        u_rel_permil=u_rel_permil,
                        type_ab=row_type_ab,
                        degrees_of_freedom=dof,
                        distribution=self._combo_text(self._custom_table, row, 5),
                        enabled=self._cell_checked(self._custom_table, row, 6),
                        reference=reference,
                        description=self._cell_text(self._custom_table, row, 8),
                    )
                )

            elements = dict(self._values.elements)
            elements[element_symbol] = ElementGlobalUncertaintyValues(
                element_symbol=element_symbol,
                ssb_kappa_defaults=kappa_defaults,
                sr_engine_a_defaults=sr_engine_a_defaults,
                custom_contributors=custom_contributors,
            )
            self._values = GlobalUncertaintyValues(
                version=self._values.version,
                elements=elements,
            )
            return True
        except Exception as exc:
            if show_errors:
                QMessageBox.warning(self, "Invalid Values", str(exc))
            return False

    def _save(self) -> None:
        if not self._capture_current_element():
            return
        try:
            self._loaded_revision = save_global_uncertainty_values(
                self._values,
                self._json_path,
                expected_revision=self._loaded_revision,
            )
        except ProfileConflictError as conflict:
            # Another manager wrote the file after this one read it. The
            # captured values stay in memory, so the draft is not lost.
            QMessageBox.warning(self, "Save Conflict", str(conflict))
            return
        except Exception as exc:
            QMessageBox.warning(self, "Save Failed", str(exc))
            return
        self._refresh_effective_values()
        QMessageBox.information(
            self,
            "Saved",
            f"Saved:\n{self._json_path}\n\n"
            "Click Reload uncertainty values in TraceISO to apply changes.",
        )

    def _reload(self) -> None:
        try:
            self._values = load_global_uncertainty_values(self._json_path)
        except Exception as exc:
            QMessageBox.warning(self, "Reload Failed", str(exc))
            return
        self._loaded_revision = global_uncertainty_values_revision(self._json_path)
        self._populate_element_combo()

    def _validate(self) -> None:
        if not self._capture_current_element():
            return
        try:
            from config.global_uncertainty_values_loader import validate_global_uncertainty_values
            validate_global_uncertainty_values(self._values)
        except Exception as exc:
            QMessageBox.warning(self, "Validation Failed", str(exc))
            return
        QMessageBox.information(self, "Validation", "Global uncertainty values are valid.")

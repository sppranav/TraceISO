"""Detail editor widget - right side of the CRM Manager."""

from __future__ import annotations

import math

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config.crm_schema import UNCERTAINTY_SEMANTICS, VALUE_KINDS, is_unassigned, standard_uncertainty
from tools.crm_manager.formatting import (
    UNASSIGNED_DISPLAY,
    format_coverage_factor,
    format_display_uncertainty,
    format_display_value,
    format_uncertainty,
    format_value,
    parse_optional_number,
)
from tools.crm_manager.models.crm_data import (
    CRMLibrary,
    CertifiedRatio,
    InternalNormalization,
    NaturalRatio,
    ReferenceData,
    ReferenceMaterial,
    collect_derivation_inputs,
    derive_ratio_from_certified,
    preview_certified_ratios,
)


def _readonly_item(text: object = "") -> QTableWidgetItem:
    item = QTableWidgetItem(str(text))
    item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
    return item


#: Roles carrying the editor's numeric model. ``_ROLE_NUMBER`` holds the float
#: the cell was populated from and ``_ROLE_RENDERED`` the text written for it,
#: so an untouched cell saves the original float rather than a re-parsed
#: rendering of it (audit A062).
_ROLE_NUMBER = Qt.UserRole + 1
_ROLE_RENDERED = Qt.UserRole + 2


def _numeric_item(value, text: str) -> QTableWidgetItem:
    """Build an editable cell that remembers the number it was built from."""
    item = QTableWidgetItem(text)
    item.setData(_ROLE_NUMBER, None if value is None else float(value))
    item.setData(_ROLE_RENDERED, text)
    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return item


def _cell_number(item, parsed):
    """Return the stored float when the cell has not been edited.

    The rendered text is lossless, so *parsed* normally equals the stored
    number already; preferring the stored one keeps a no-edit save
    bit-identical even if a display formatter is later changed.
    """
    if item is None:
        return parsed
    if item.text() != item.data(_ROLE_RENDERED):
        return parsed
    stored = item.data(_ROLE_NUMBER)
    if stored is None:
        return parsed
    try:
        return float(stored)
    except (TypeError, ValueError):
        return parsed


class DetailEditor(QScrollArea):
    """Editable detail view for a CRM or element."""

    # Emitted when user saves a CRM or element.
    crm_saved = pyqtSignal(str, str)      # (element, crm_name)
    element_saved = pyqtSignal(str)       # (element,)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._library: CRMLibrary | None = None
        self._current_element: str = ""
        self._current_crm: str = ""
        self._mode: str = "none"  # "none" | "crm" | "element"

        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)

        self._inner = QWidget()
        self.setWidget(self._inner)
        self._layout = QVBoxLayout(self._inner)
        self._layout.setContentsMargins(16, 12, 16, 12)
        self._layout.setSpacing(12)

        header_row = QHBoxLayout()
        self._lbl_name = QLabel("Select a CRM or element")
        self._lbl_name.setProperty("class", "header")
        self._lbl_badge = QLabel("")
        self._lbl_badge.setProperty("class", "badge")
        self._lbl_badge.setVisible(False)
        header_row.addWidget(self._lbl_name)
        header_row.addWidget(self._lbl_badge)
        header_row.addStretch()
        self._layout.addLayout(header_row)

        # CRM metadata
        self._meta_group = QGroupBox("Metadata")
        meta_form = QFormLayout()
        meta_form.setLabelAlignment(Qt.AlignRight)
        self._edit_name = QLineEdit()
        self._edit_source = QLineEdit()
        self._edit_description = QLineEdit()
        # Record identity and value concept are shown, not edited: a stable
        # record id is what exported provenance refers to, and
        # two records for one material are told apart by these three fields
        # rather than by their display names (review item A-3).
        self._lbl_record_id = QLabel("")
        self._lbl_record_id.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._lbl_value_kind = QLabel("")
        self._lbl_display_label = QLabel("")
        self._lbl_display_label.setWordWrap(True)
        meta_form.addRow("Name:", self._edit_name)
        meta_form.addRow("Record ID:", self._lbl_record_id)
        meta_form.addRow("Value kind:", self._lbl_value_kind)
        meta_form.addRow("Display label:", self._lbl_display_label)
        meta_form.addRow("Source:", self._edit_source)
        meta_form.addRow("Description:", self._edit_description)
        self._meta_group.setLayout(meta_form)
        self._layout.addWidget(self._meta_group)

        # CRM ratios
        self._ratio_group = QGroupBox("Certified Ratios")
        ratio_vbox = QVBoxLayout()
        self._tbl_ratios = QTableWidget(0, 4)
        self._tbl_ratios.setHorizontalHeaderLabels(
            ["Ratio", "Value", "Uncertainty", "k"],
        )
        hh = self._tbl_ratios.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._tbl_ratios.setMinimumHeight(160)
        self._tbl_ratios.verticalHeader().setVisible(False)
        ratio_vbox.addWidget(self._tbl_ratios)

        ratio_btns = QHBoxLayout()
        self._btn_add_ratio = QPushButton("+ Add Ratio")
        self._btn_add_ratio.setProperty("class", "secondary")
        self._btn_derive_ratio = QPushButton("Derive Ratio")
        self._btn_derive_ratio.setProperty("class", "secondary")
        self._btn_del_ratio = QPushButton("- Remove Ratio")
        self._btn_del_ratio.setProperty("class", "danger")
        ratio_btns.addWidget(self._btn_add_ratio)
        ratio_btns.addWidget(self._btn_derive_ratio)
        ratio_btns.addWidget(self._btn_del_ratio)
        ratio_btns.addStretch()
        ratio_vbox.addLayout(ratio_btns)
        self._ratio_group.setLayout(ratio_vbox)
        self._layout.addWidget(self._ratio_group)

        self._derived_group = QGroupBox("Derived Ratios Preview")
        derived_vbox = QVBoxLayout()
        self._derived_hint = QLabel(
            "Read-only preview from TraceISO CRM derivation logic. Rows marked "
            "'derived' are not stored directly in this CRM entry."
        )
        self._derived_hint.setWordWrap(True)
        derived_vbox.addWidget(self._derived_hint)
        # Inline, non-modal, and persistent while it applies. The refresh runs
        # on every cell change, so an unusable input has to be a state on the
        # panel rather than a dialog (audit A064).
        self._derived_status = QLabel("")
        self._derived_status.setWordWrap(True)
        self._derived_status.setVisible(False)
        derived_vbox.addWidget(self._derived_status)
        self._tbl_derived = QTableWidget(0, 7)
        self._tbl_derived.setHorizontalHeaderLabels(
            ["Ratio", "Value", "Uncertainty", "k", "Semantics", "Type", "Standard u"],
        )
        dh = self._tbl_derived.horizontalHeader()
        dh.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 6):
            dh.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self._tbl_derived.setMinimumHeight(130)
        self._tbl_derived.verticalHeader().setVisible(False)
        derived_vbox.addWidget(self._tbl_derived)
        self._derived_group.setLayout(derived_vbox)
        self._layout.addWidget(self._derived_group)

        # CRM masses
        self._mass_group = QGroupBox("Atomic Masses")
        mass_vbox = QVBoxLayout()
        self._tbl_masses = QTableWidget(0, 2)
        self._tbl_masses.setHorizontalHeaderLabels(["Isotope", "Mass (u)"])
        mh = self._tbl_masses.horizontalHeader()
        mh.setSectionResizeMode(0, QHeaderView.Stretch)
        mh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._tbl_masses.setMinimumHeight(120)
        self._tbl_masses.verticalHeader().setVisible(False)
        mass_vbox.addWidget(self._tbl_masses)

        mass_btns = QHBoxLayout()
        self._btn_add_mass = QPushButton("+ Add Isotope")
        self._btn_add_mass.setProperty("class", "secondary")
        self._btn_del_mass = QPushButton("- Remove Isotope")
        self._btn_del_mass.setProperty("class", "danger")
        mass_btns.addWidget(self._btn_add_mass)
        mass_btns.addWidget(self._btn_del_mass)
        mass_btns.addStretch()
        mass_vbox.addLayout(mass_btns)
        self._mass_group.setLayout(mass_vbox)
        self._layout.addWidget(self._mass_group)

        # Element-level internal normalization
        self._norm_group = QGroupBox("Internal Normalization (Element Level)")
        norm_form = QFormLayout()
        norm_form.setLabelAlignment(Qt.AlignRight)
        self._edit_norm_ratio = QLineEdit()
        self._edit_norm_value = QLineEdit()
        self._edit_norm_ratio.setPlaceholderText("e.g. 86Sr/88Sr")
        self._edit_norm_value.setPlaceholderText("e.g. 0.1194")
        norm_form.addRow("Ratio:", self._edit_norm_ratio)
        norm_form.addRow("Value:", self._edit_norm_value)
        self._norm_hint = QLabel(
            "Leave both fields blank to clear internal normalization for this element."
        )
        self._norm_hint.setWordWrap(True)
        norm_form.addRow("", self._norm_hint)
        self._norm_group.setLayout(norm_form)
        self._layout.addWidget(self._norm_group)

        # Element-level reference data
        self._refdata_group = QGroupBox("Reference Data (Element Level)")
        refdata_vbox = QVBoxLayout()

        self._refdata_masses_label = QLabel("Element masses")
        refdata_vbox.addWidget(self._refdata_masses_label)

        self._tbl_ref_masses = QTableWidget(0, 2)
        self._tbl_ref_masses.setHorizontalHeaderLabels(["Isotope", "Mass (u)"])
        rmh = self._tbl_ref_masses.horizontalHeader()
        rmh.setSectionResizeMode(0, QHeaderView.Stretch)
        rmh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._tbl_ref_masses.setMinimumHeight(120)
        self._tbl_ref_masses.verticalHeader().setVisible(False)
        refdata_vbox.addWidget(self._tbl_ref_masses)

        ref_mass_btns = QHBoxLayout()
        self._btn_add_ref_mass = QPushButton("+ Add Element Mass")
        self._btn_add_ref_mass.setProperty("class", "secondary")
        self._btn_del_ref_mass = QPushButton("- Remove Element Mass")
        self._btn_del_ref_mass.setProperty("class", "danger")
        ref_mass_btns.addWidget(self._btn_add_ref_mass)
        ref_mass_btns.addWidget(self._btn_del_ref_mass)
        ref_mass_btns.addStretch()
        refdata_vbox.addLayout(ref_mass_btns)

        # PUB-07: these rows are not all "natural" ratios. Some are assigned
        # correction ratios (the 204Hg/202Hg correction ratio, for one), and
        # calling them natural hides the difference in role.
        self._refdata_ratios_label = QLabel("Natural / assigned correction ratios")
        refdata_vbox.addWidget(self._refdata_ratios_label)

        # PUB-08 rejects the public tree's "uncertainty 0 means unassigned"
        # hint. Blank means unassigned; 0 means exact. The two must stay
        # visibly different.
        self._refdata_ratios_hint = QLabel(
            "Leave Uncertainty and k blank when no uncertainty is assigned. "
            "Do not enter 0: 0 states the value is exact, which is a different "
            "claim from unknown."
        )
        self._refdata_ratios_hint.setWordWrap(True)
        refdata_vbox.addWidget(self._refdata_ratios_hint)

        self._tbl_ref_ratios = QTableWidget(0, 7)
        self._tbl_ref_ratios.setHorizontalHeaderLabels(
            ["Ratio", "Value", "Uncertainty", "k", "Value kind", "Semantics", "Source"],
        )
        rrh = self._tbl_ref_ratios.horizontalHeader()
        for _column in range(6):
            rrh.setSectionResizeMode(_column, QHeaderView.ResizeToContents)
        rrh.setSectionResizeMode(6, QHeaderView.Stretch)
        self._tbl_ref_ratios.setMinimumHeight(150)
        self._tbl_ref_ratios.verticalHeader().setVisible(False)
        refdata_vbox.addWidget(self._tbl_ref_ratios)

        ref_ratio_btns = QHBoxLayout()
        self._btn_add_ref_ratio = QPushButton("+ Add Natural Ratio")
        self._btn_add_ref_ratio.setProperty("class", "secondary")
        self._btn_del_ref_ratio = QPushButton("- Remove Natural Ratio")
        self._btn_del_ref_ratio.setProperty("class", "danger")
        ref_ratio_btns.addWidget(self._btn_add_ref_ratio)
        ref_ratio_btns.addWidget(self._btn_del_ref_ratio)
        ref_ratio_btns.addStretch()
        refdata_vbox.addLayout(ref_ratio_btns)

        self._refdata_group.setLayout(refdata_vbox)
        self._layout.addWidget(self._refdata_group)

        self._layout.addStretch()
        action_row = QHBoxLayout()
        self._btn_save = QPushButton("Apply to Library")
        self._btn_discard = QPushButton("Discard")
        self._btn_discard.setProperty("class", "secondary")
        action_row.addStretch()
        action_row.addWidget(self._btn_discard)
        action_row.addWidget(self._btn_save)
        self.actions = QWidget()
        self.actions.setLayout(action_row)
        self._layout.addWidget(self.actions)

        # Connections
        self._btn_add_ratio.clicked.connect(self._add_ratio_row)
        self._btn_derive_ratio.clicked.connect(self._derive_ratio_row)
        self._btn_del_ratio.clicked.connect(self._del_ratio_row)
        self._btn_add_mass.clicked.connect(self._add_mass_row)
        self._btn_del_mass.clicked.connect(self._del_mass_row)
        self._btn_add_ref_mass.clicked.connect(self._add_ref_mass_row)
        self._btn_del_ref_mass.clicked.connect(self._del_ref_mass_row)
        self._btn_add_ref_ratio.clicked.connect(self._add_ref_ratio_row)
        self._btn_del_ref_ratio.clicked.connect(self._del_ref_ratio_row)
        self._btn_save.clicked.connect(self._save)
        self._btn_discard.clicked.connect(self._discard)

        for group in (self._derived_group, self._mass_group):
            group.setCheckable(True)
            group.setChecked(True)
            group.toggled.connect(lambda checked, g=group: [
                child.setVisible(checked) for child in g.findChildren(QWidget, options=Qt.FindDirectChildrenOnly)
            ])
        guidance = "Absolute uncertainty in ratio units. Certificate k is separate from report k. Blank means unassigned; zero means exact. For expanded uncertainty, standard u = U / k."
        self._tbl_ratios.horizontalHeaderItem(2).setToolTip(guidance)
        self._tbl_ratios.horizontalHeaderItem(3).setToolTip(guidance)
        hint = QLabel(guidance)
        hint.setWordWrap(True)
        ratio_vbox.insertWidget(0, hint)
        self._set_mode("none")

    # Public API

    def _form_snapshot(self):
        fields = tuple(w.text() for w in self._inner.findChildren(QLineEdit))
        tables = tuple(
            tuple(tuple(t.item(r, c).text() if t.item(r, c) else ""
                        for c in range(t.columnCount())) for r in range(t.rowCount()))
            for t in (self._tbl_ratios, self._tbl_masses, self._tbl_ref_masses, self._tbl_ref_ratios)
        )
        return fields, tables

    @property
    def has_draft(self):
        return self._mode != "none" and self._form_snapshot() != getattr(self, "_clean_snapshot", None)

    def accept_draft(self):
        self._clean_snapshot = self._form_snapshot()

    def commit_active_cell(self):
        focus = QApplication.focusWidget()
        if focus and any(table.isAncestorOf(focus) for table in (
            self._tbl_ratios, self._tbl_masses, self._tbl_ref_masses, self._tbl_ref_ratios)):
            self._btn_save.setFocus()

    def apply_draft(self):
        self.commit_active_cell()
        if not self.has_draft:
            return True
        self._save()
        return not self.has_draft

    def set_library(self, library: CRMLibrary) -> None:
        self._library = library

    def show_crm(self, element: str, crm_name: str) -> None:
        """Populate editor with CRM data."""
        if self._library is None:
            return
        crm = self._library.get_crm(element, crm_name)
        if crm is None:
            self.clear_editor()
            return

        self._mode = "crm"
        self._current_element = element
        self._current_crm = crm_name

        self._lbl_name.setText(crm_name)
        self._lbl_badge.setText(element)
        self._lbl_badge.setVisible(True)

        self._edit_name.setText(crm.name)
        self._edit_source.setText(crm.source)
        self._edit_description.setText(crm.description)
        self._lbl_record_id.setText(crm.record_id or crm.name)
        self._lbl_value_kind.setText(crm.value_kind)
        self._lbl_display_label.setText(crm.display_label or crm.name)

        self._tbl_ratios.setRowCount(0)
        for rname, rdata in crm.ratios.items():
            self._insert_ratio_row(rname, rdata.value, rdata.uncertainty, rdata.k)
        self._refresh_derived_ratios()

        self._tbl_masses.setRowCount(0)
        for isotope, mass in crm.masses.items():
            self._insert_mass_row(isotope, mass)

        self._set_mode("crm")

    def show_element(self, element: str) -> None:
        """Populate editor with element-level settings."""
        if self._library is None:
            return
        elem = self._library.get_element(element)
        if elem is None:
            self.clear_editor()
            return

        self._mode = "element"
        self._current_element = element
        self._current_crm = ""

        self._lbl_name.setText(f"{element} - Element Settings")
        self._lbl_badge.setText(element)
        self._lbl_badge.setVisible(True)

        norm = elem.internal_normalization
        if norm is not None:
            self._edit_norm_ratio.setText(norm.ratio_name)
            self._edit_norm_value.setText(format_value(norm.value))
        else:
            self._edit_norm_ratio.clear()
            self._edit_norm_value.clear()

        self._tbl_ref_masses.setRowCount(0)
        self._tbl_ref_ratios.setRowCount(0)
        if elem.reference_data is not None:
            for isotope, mass in elem.reference_data.masses.items():
                self._insert_ref_mass_row(isotope, mass)
            for ratio_name, ratio in elem.reference_data.natural_ratios.items():
                self._insert_ref_ratio_row(
                    ratio_name,
                    ratio.value,
                    ratio.uncertainty,
                    ratio.k,
                    ratio.source,
                    ratio.value_kind,
                    ratio.uncertainty_semantics,
                )

        self._set_mode("element")

    def clear_editor(self) -> None:
        """Reset to empty state."""
        self._mode = "none"
        self._current_element = ""
        self._current_crm = ""

        self._lbl_name.setText("Select a CRM or element")
        self._lbl_badge.setVisible(False)

        self._edit_name.clear()
        self._edit_source.clear()
        self._edit_description.clear()
        self._lbl_record_id.clear()
        self._lbl_value_kind.clear()
        self._lbl_display_label.clear()
        self._tbl_ratios.setRowCount(0)
        self._tbl_masses.setRowCount(0)
        self._tbl_derived.setRowCount(0)
        self._edit_norm_ratio.clear()
        self._edit_norm_value.clear()
        self._tbl_ref_masses.setRowCount(0)
        self._tbl_ref_ratios.setRowCount(0)

        self._set_mode("none")

    # Table helpers

    def _insert_ratio_row(self, name: str, value: float, unc, k) -> None:
        row = self._tbl_ratios.rowCount()
        self._tbl_ratios.insertRow(row)

        val_item = _numeric_item(value, format_value(value))
        unc_item = _numeric_item(unc, format_uncertainty(unc))
        k_item = _numeric_item(k, format_coverage_factor(k))

        self._tbl_ratios.setItem(row, 0, QTableWidgetItem(name))
        self._tbl_ratios.setItem(row, 1, val_item)
        self._tbl_ratios.setItem(row, 2, unc_item)
        self._tbl_ratios.setItem(row, 3, k_item)
        if getattr(self, "_mode", "none") == "crm":
            self._refresh_derived_ratios()

    PREVIEW_UNAVAILABLE_TEXT = (
        "Preview unavailable while inputs are incomplete or invalid."
    )

    def preview_status_text(self) -> str:
        """The inline preview status, or ``""`` when the preview is current."""
        status = getattr(self, "_derived_status", None)
        return "" if status is None else status.text()

    def preview_is_available(self) -> bool:
        """Is the derived table showing consequences of the current input?"""
        return self.preview_status_text() == ""

    def _set_preview_status(self, message: str) -> None:
        status = getattr(self, "_derived_status", None)
        if status is None:
            return
        status.setText(message)
        status.setVisible(bool(message))

    def _refresh_derived_ratios(self) -> None:
        """Refresh the read-only derived-ratio preview for the active CRM.

        The preview states the consequences of the record **currently in the
        editor**, so an edited or imported value is what it reasons about.
        Reading the process's default managed library instead meant the panel
        could answer for a record the user was no longer looking at.

        When the table cannot be validated — a half-typed ratio name, a cell
        that is not yet a number — there is no current record to reason about.
        The panel says so and shows nothing. It used to fall back to the stored
        record, which put saved numbers on screen under a heading that claims
        they are the consequences of the edit in progress (audit A064).
        """
        if not hasattr(self, "_tbl_derived"):
            return
        self._tbl_derived.setRowCount(0)
        self._set_preview_status("")
        if self._mode != "crm" or not self._current_element or not self._current_crm:
            return

        current = self._collect_ratio_map_from_table(announce=False)
        if current is None:
            self._set_preview_status(self.PREVIEW_UNAVAILABLE_TEXT)
            return

        try:
            preview = preview_certified_ratios(current)
        except Exception as exc:  # noqa: BLE001 - preview must not block editing
            self._set_preview_status(f"Preview unavailable: {exc}")
            return

        for ratio_name, entry in sorted(preview.items()):
            row = self._tbl_derived.rowCount()
            self._tbl_derived.insertRow(row)
            self._tbl_derived.setItem(row, 0, _readonly_item(ratio_name))
            self._tbl_derived.setItem(
                row, 1, _readonly_item(format_display_value(entry.value))
            )
            self._tbl_derived.setItem(
                row, 2, _readonly_item(format_display_uncertainty(entry.uncertainty))
            )
            self._tbl_derived.setItem(
                row, 3, _readonly_item(format_coverage_factor(entry.k))
            )
            self._tbl_derived.setItem(
                row, 4, _readonly_item(entry.uncertainty_semantics)
            )
            self._tbl_derived.setItem(row, 5, _readonly_item(entry.origin))
            u = standard_uncertainty(entry.uncertainty, entry.k, entry.uncertainty_semantics)
            self._tbl_derived.setItem(row, 6, _readonly_item(
                format_display_uncertainty(u) if u is not None else "Unavailable for these semantics"))

    def _insert_mass_row(self, isotope: str, mass: float) -> None:
        row = self._tbl_masses.rowCount()
        self._tbl_masses.insertRow(row)

        mass_item = _numeric_item(mass, format_value(mass))

        self._tbl_masses.setItem(row, 0, QTableWidgetItem(isotope))
        self._tbl_masses.setItem(row, 1, mass_item)

    def _add_ratio_row(self) -> None:
        # A brand-new row has no uncertainty yet - blank, not 0.
        self._insert_ratio_row("NewIsotope/DenIsotope", 0.0, None, None)
        row = self._tbl_ratios.rowCount() - 1
        self._tbl_ratios.scrollToItem(self._tbl_ratios.item(row, 0))
        self._tbl_ratios.editItem(self._tbl_ratios.item(row, 0))

    def _del_ratio_row(self) -> None:
        row = self._tbl_ratios.currentRow()
        if row >= 0:
            self._tbl_ratios.removeRow(row)
            self._refresh_derived_ratios()

    def _derive_ratio_row(self) -> None:
        """Derive a ratio from currently available certified ratios."""
        current = self._collect_ratio_map_from_table()
        if current is None:
            return
        if not current:
            QMessageBox.information(
                self,
                "Derive Ratio",
                "Add at least one certified ratio before deriving.",
            )
            return

        # A row the editor legitimately accepts - an unassigned uncertainty, or
        # a source-stated limit - has no standard uncertainty to propagate.
        # Report which rows those are instead of raising on them or letting
        # them stand in as exact values.
        inputs = collect_derivation_inputs(current)
        if inputs.unusable:
            excluded = "\n".join(
                f"  {name}: {reason}" for name, reason in inputs.unusable
            )
            if not inputs.usable:
                QMessageBox.warning(
                    self,
                    "Derive Ratio",
                    "No certified ratio in this record can be used to derive "
                    f"another:\n\n{excluded}",
                )
                return
            QMessageBox.information(
                self,
                "Derive Ratio",
                "These rows are excluded from the derivation because they "
                f"carry no standard uncertainty:\n\n{excluded}",
            )

        text, ok = QInputDialog.getText(
            self,
            "Derive Ratio",
            "Target ratio (e.g. 112Cd/116Cd):",
        )
        if not ok:
            return

        target = text.strip().replace("\\", "/")
        if target.count("/") != 1:
            QMessageBox.warning(
                self,
                "Derive Ratio",
                "Ratio must be in the form Numerator/Denominator.",
            )
            return

        if target in current:
            QMessageBox.information(
                self,
                "Derive Ratio",
                f"{target} already exists in this CRM.",
            )
            return

        derived = derive_ratio_from_certified(target, current)
        if derived is None:
            QMessageBox.warning(
                self,
                "Derive Ratio",
                f"Could not derive {target} from available ratios.",
            )
            return

        # The solver propagates standard uncertainties, so what comes back is
        # one: k = 1, stated rather than inferred from the inputs' coverage
        # factors. Guessing an expanded k here understated a mixed-k
        # derivation by the ratio of the guess to the true factor.
        value, unc = derived
        k_default = 1.0

        self._insert_ratio_row(target, value, unc, k_default)
        self._refresh_derived_ratios()
        row = self._tbl_ratios.rowCount() - 1
        self._tbl_ratios.setCurrentCell(row, 0)
        QMessageBox.information(
            self,
            "Derive Ratio",
            f"Derived {target} = {format_value(value)} +- "
            f"{format_display_uncertainty(unc)} "
            f"(standard uncertainty, k={k_default:.0f})",
        )

    def _collect_ratio_map_from_table(
        self, *, announce: bool = True,
    ) -> dict[str, CertifiedRatio] | None:
        """Read ratio rows with validation for derive/save actions.

        ``announce=False`` performs the same validation silently, for the
        passive preview refresh: a half-typed row should grey the preview out,
        not interrupt with a modal.
        """
        def reject(message: str) -> None:
            if announce:
                QMessageBox.warning(self, "Validation", message)
            return None

        ratios: dict[str, CertifiedRatio] = {}
        for row in range(self._tbl_ratios.rowCount()):
            rname = (self._tbl_ratios.item(row, 0) or QTableWidgetItem()).text().strip()
            if not rname:
                continue
            rname = rname.replace("\\", "/")
            if rname.count("/") != 1:
                return reject(f"Invalid ratio name in row {row + 1}: {rname}")
            val_item = self._tbl_ratios.item(row, 1)
            unc_item = self._tbl_ratios.item(row, 2)
            k_item = self._tbl_ratios.item(row, 3)
            try:
                val = _cell_number(
                    val_item, float((val_item or QTableWidgetItem()).text()),
                )
                unc = _cell_number(
                    unc_item,
                    parse_optional_number((unc_item or QTableWidgetItem()).text()),
                )
                k = _cell_number(
                    k_item,
                    parse_optional_number((k_item or QTableWidgetItem()).text()),
                )
            except ValueError:
                return reject(
                    f"Invalid number in ratio row {row + 1}. Leave Uncertainty "
                    "and k blank for an unassigned uncertainty."
                )
            if not math.isfinite(val):
                return reject(
                    f"Non-finite value (NaN or Inf) in ratio row {row + 1}."
                )
            if val == 0:
                return reject(f"Ratio value cannot be zero in row {row + 1}.")
            previous = self._stored_certified_ratio(rname)
            semantics = (
                "unassigned" if unc is None
                else self._retained_semantics(previous, k)
            )
            problem = self._uncertainty_cell_problem(
                unc, k, row + 1, semantics=semantics,
            )
            if problem:
                return reject(problem)
            if rname in ratios:
                return reject(
                    f"Duplicate ratio name in row {row + 1}: {rname}"
                )
            ratios[rname] = CertifiedRatio(
                value=val,
                uncertainty=unc,
                k=k,
                uncertainty_semantics=semantics,
                coverage_status=("" if unc is None else getattr(previous, "coverage_status", "")),
                note=("" if unc is None else getattr(previous, "note", "")),
            )
        return ratios

    def _stored_certified_ratio(self, ratio_name: str):
        """Return the record currently stored for *ratio_name*, if any.

        Editing a value must not silently drop the semantics declaration and
        coverage status the row already carried.
        """
        if self._library is None or not self._current_element or not self._current_crm:
            return None
        crm = self._library.get_crm(self._current_element, self._current_crm)
        if crm is None:
            return None
        return crm.ratios.get(ratio_name)

    @staticmethod
    def _retained_semantics(previous, k) -> str:
        """Keep an existing semantics declaration unless it no longer fits."""
        existing = getattr(previous, "uncertainty_semantics", "")
        if existing in UNCERTAINTY_SEMANTICS and existing != "unassigned":
            return existing
        try:
            return "standard_uncertainty" if float(k) == 1.0 else "expanded_uncertainty"
        except (TypeError, ValueError):
            return "expanded_uncertainty"

    @staticmethod
    def _uncertainty_cell_problem(
        unc, k, row_number: int, *, semantics: str = "",
    ) -> str:
        """Return a message when an uncertainty/k pair is not a valid row.

        Blank/blank is an unassigned uncertainty and is allowed. Anything
        half-blank is rejected, so a row can never end up with a coverage
        factor covering nothing, or an uncertainty with no stated coverage.
        """
        if unc is None and k is None:
            return ""
        if unc is None:
            return (
                f"Row {row_number}: a coverage factor was given but no "
                "uncertainty. Clear k as well to record the uncertainty as "
                "unassigned."
            )
        if k is None and semantics == "source_stated_limit":
            return ""
        if k is None:
            return (
                f"Row {row_number}: an uncertainty was given but no coverage "
                "factor k."
            )
        if not math.isfinite(unc) or unc < 0:
            return f"Row {row_number}: uncertainty must be a finite value >= 0."
        if not math.isfinite(k) or k <= 0:
            return f"Row {row_number}: coverage factor k must be positive."
        return ""

    def _add_mass_row(self) -> None:
        self._insert_mass_row("XXn", 0.0)
        row = self._tbl_masses.rowCount() - 1
        self._tbl_masses.scrollToItem(self._tbl_masses.item(row, 0))
        self._tbl_masses.editItem(self._tbl_masses.item(row, 0))

    def _del_mass_row(self) -> None:
        row = self._tbl_masses.currentRow()
        if row >= 0:
            self._tbl_masses.removeRow(row)

    def _insert_ref_mass_row(self, isotope: str, mass: float) -> None:
        row = self._tbl_ref_masses.rowCount()
        self._tbl_ref_masses.insertRow(row)

        mass_item = _numeric_item(mass, format_value(mass))

        self._tbl_ref_masses.setItem(row, 0, QTableWidgetItem(isotope))
        self._tbl_ref_masses.setItem(row, 1, mass_item)

    def _insert_ref_ratio_row(
        self,
        ratio_name: str,
        value: float,
        uncertainty,
        k,
        source: str,
        value_kind: str = "unspecified",
        semantics: str = "standard_uncertainty",
    ) -> None:
        row = self._tbl_ref_ratios.rowCount()
        self._tbl_ref_ratios.insertRow(row)

        val_item = _numeric_item(value, format_value(value))
        unc_item = _numeric_item(uncertainty, format_uncertainty(uncertainty))
        k_item = _numeric_item(k, format_coverage_factor(k))

        self._tbl_ref_ratios.setItem(row, 0, QTableWidgetItem(ratio_name))
        self._tbl_ref_ratios.setItem(row, 1, val_item)
        self._tbl_ref_ratios.setItem(row, 2, unc_item)
        self._tbl_ref_ratios.setItem(row, 3, k_item)
        self._tbl_ref_ratios.setItem(row, 4, QTableWidgetItem(value_kind))
        self._tbl_ref_ratios.setItem(row, 5, QTableWidgetItem(semantics))
        self._tbl_ref_ratios.setItem(row, 6, QTableWidgetItem(source))

    def _stored_natural_ratio(self, ratio_name: str):
        """Return the natural-ratio record currently stored, if any."""
        if self._library is None or not self._current_element:
            return None
        elem = self._library.get_element(self._current_element)
        if elem is None or elem.reference_data is None:
            return None
        return elem.reference_data.natural_ratios.get(ratio_name)

    def _add_ref_mass_row(self) -> None:
        self._insert_ref_mass_row("86Sr", 0.0)
        row = self._tbl_ref_masses.rowCount() - 1
        self._tbl_ref_masses.scrollToItem(self._tbl_ref_masses.item(row, 0))
        self._tbl_ref_masses.editItem(self._tbl_ref_masses.item(row, 0))

    def _del_ref_mass_row(self) -> None:
        row = self._tbl_ref_masses.currentRow()
        if row >= 0:
            self._tbl_ref_masses.removeRow(row)

    def _add_ref_ratio_row(self) -> None:
        self._insert_ref_ratio_row(
            "87Rb/85Rb", 0.0, None, None, "", "unspecified", "unassigned",
        )
        row = self._tbl_ref_ratios.rowCount() - 1
        self._tbl_ref_ratios.scrollToItem(self._tbl_ref_ratios.item(row, 0))
        self._tbl_ref_ratios.editItem(self._tbl_ref_ratios.item(row, 0))

    def _del_ref_ratio_row(self) -> None:
        row = self._tbl_ref_ratios.currentRow()
        if row >= 0:
            self._tbl_ref_ratios.removeRow(row)

    def _collect_reference_data_from_tables(self) -> ReferenceData | None:
        """Read element-level reference data rows with validation."""
        masses: dict[str, float] = {}
        for row in range(self._tbl_ref_masses.rowCount()):
            isotope = (self._tbl_ref_masses.item(row, 0) or QTableWidgetItem()).text().strip()
            if not isotope:
                continue
            ref_mass_item = self._tbl_ref_masses.item(row, 1)
            try:
                mass = _cell_number(
                    ref_mass_item, float((ref_mass_item or QTableWidgetItem()).text()),
                )
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Validation",
                    f"Invalid number in reference-data mass row {row + 1}.",
                )
                return None
            if (not math.isfinite(mass)) or mass <= 0:
                QMessageBox.warning(
                    self,
                    "Validation",
                    f"Mass in reference-data row {row + 1} must be a finite value > 0.",
                )
                return None
            if isotope in masses:
                QMessageBox.warning(
                    self, "Validation", f"Duplicate mass name in row {row + 1}: {isotope}",
                )
                return None
            masses[isotope] = mass

        natural_ratios: dict[str, NaturalRatio] = {}
        for row in range(self._tbl_ref_ratios.rowCount()):
            ratio_name = (self._tbl_ref_ratios.item(row, 0) or QTableWidgetItem()).text().strip()
            if not ratio_name:
                continue
            ratio_name = ratio_name.replace("\\", "/")
            if ratio_name.count("/") != 1:
                QMessageBox.warning(
                    self,
                    "Validation",
                    f"Invalid natural ratio name in row {row + 1}: {ratio_name}",
                )
                return None
            value_item = self._tbl_ref_ratios.item(row, 1)
            unc_item = self._tbl_ref_ratios.item(row, 2)
            k_item = self._tbl_ref_ratios.item(row, 3)
            try:
                value = _cell_number(
                    value_item, float((value_item or QTableWidgetItem()).text()),
                )
                uncertainty = _cell_number(
                    unc_item,
                    parse_optional_number((unc_item or QTableWidgetItem()).text()),
                )
                k = _cell_number(
                    k_item,
                    parse_optional_number((k_item or QTableWidgetItem()).text()),
                )
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Validation",
                    f"Invalid number in natural-ratio row {row + 1}. Leave "
                    "Uncertainty and k blank for an unassigned uncertainty.",
                )
                return None
            if (not math.isfinite(value)) or value <= 0:
                QMessageBox.warning(
                    self,
                    "Validation",
                    f"Natural ratio value in row {row + 1} must be a finite value > 0.",
                )
                return None
            value_kind = (
                self._tbl_ref_ratios.item(row, 4) or QTableWidgetItem()
            ).text().strip() or "unspecified"
            if value_kind not in VALUE_KINDS:
                QMessageBox.warning(
                    self,
                    "Validation",
                    f"Row {row + 1}: unknown value kind {value_kind!r}. Expected "
                    f"one of {sorted(VALUE_KINDS)}.",
                )
                return None
            semantics = (
                self._tbl_ref_ratios.item(row, 5) or QTableWidgetItem()
            ).text().strip()
            if uncertainty is None:
                semantics = "unassigned"
            elif semantics == "unassigned":
                QMessageBox.warning(
                    self,
                    "Validation",
                    f"Row {row + 1}: an unassigned row must leave Uncertainty "
                    "and k blank. Zero would claim the value is exact.",
                )
                return None
            if semantics not in UNCERTAINTY_SEMANTICS:
                QMessageBox.warning(
                    self,
                    "Validation",
                    f"Row {row + 1}: unknown uncertainty semantics "
                    f"{semantics!r}. Expected one of "
                    f"{sorted(UNCERTAINTY_SEMANTICS)}.",
                )
                return None
            problem = self._uncertainty_cell_problem(
                uncertainty, k, row + 1, semantics=semantics,
            )
            if problem:
                QMessageBox.warning(self, "Validation", problem)
                return None
            source = (self._tbl_ref_ratios.item(row, 6) or QTableWidgetItem()).text().strip()
            if ratio_name in natural_ratios:
                QMessageBox.warning(
                    self,
                    "Validation",
                    f"Duplicate natural ratio name in row {row + 1}: {ratio_name}",
                )
                return None
            previous = self._stored_natural_ratio(ratio_name)
            natural_ratios[ratio_name] = NaturalRatio(
                value=value,
                uncertainty=uncertainty,
                k=k,
                source=source,
                record_id=getattr(previous, "record_id", "") or ratio_name,
                display_label=getattr(previous, "display_label", "") or ratio_name,
                value_kind=value_kind,
                uncertainty_semantics=semantics,
                coverage_status=getattr(previous, "coverage_status", ""),
                source_doi=getattr(previous, "source_doi", None),
                source_url=getattr(previous, "source_url", None),
                note=getattr(previous, "note", ""),
            )

        return ReferenceData(masses=masses, natural_ratios=natural_ratios)

    # Save / Discard

    def _save(self) -> None:
        if self._mode == "crm":
            self._save_crm()
            return
        if self._mode == "element":
            self._save_element_settings()

    def _save_crm(self) -> None:
        if not self._library or not self._current_element:
            return

        new_name = self._edit_name.text().strip()
        if not new_name:
            QMessageBox.warning(self, "Validation", "Name cannot be empty.")
            return

        ratios = self._collect_ratio_map_from_table()
        if ratios is None:
            return

        masses = {}
        for row in range(self._tbl_masses.rowCount()):
            isotope = (self._tbl_masses.item(row, 0) or QTableWidgetItem()).text().strip()
            if not isotope:
                continue
            mass_item = self._tbl_masses.item(row, 1)
            try:
                mass = _cell_number(
                    mass_item, float((mass_item or QTableWidgetItem()).text()),
                )
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Validation",
                    f"Invalid number in mass row {row + 1}.",
                )
                return
            if not (math.isfinite(mass) and mass > 0):
                QMessageBox.warning(
                    self,
                    "Validation",
                    f"Mass must be a finite positive number in row {row + 1}.",
                )
                return
            if isotope in masses:
                QMessageBox.warning(
                    self, "Validation", f"Duplicate mass name in row {row + 1}: {isotope}",
                )
                return
            masses[isotope] = mass

        existing_record = self._library.get_crm(
            self._current_element, self._current_crm,
        )
        crm = ReferenceMaterial(
            name=new_name,
            element=self._current_element,
            source=self._edit_source.text().strip(),
            description=self._edit_description.text().strip(),
            ratios=ratios,
            masses=masses,
            # Identity and value semantics survive an edit. Losing them would
            # make two records for one material indistinguishable again.
            record_id=getattr(existing_record, "record_id", "") or new_name,
            material_id=getattr(existing_record, "material_id", "") or new_name,
            display_label=getattr(existing_record, "display_label", "") or new_name,
            value_kind=getattr(existing_record, "value_kind", "unspecified"),
            source_doi=getattr(existing_record, "source_doi", None),
            source_url=getattr(existing_record, "source_url", None),
            normalization=getattr(existing_record, "normalization", None),
        )

        if new_name != self._current_crm:
            # Guard against overwriting a *different* existing CRM with the new name
            existing = self._library.get_crm(self._current_element, new_name)
            if existing is not None:
                QMessageBox.warning(
                    self,
                    "Validation",
                    f"A CRM named \"{new_name}\" already exists for "
                    f"{self._current_element}. Choose a different name.",
                )
                return
            self._library.remove_crm(self._current_element, self._current_crm)

        # This is an edit of the record on screen, so replacing it is the point.
        self._library.add_crm(self._current_element, crm, replace=True)
        self._current_crm = new_name
        self.accept_draft()
        self.crm_saved.emit(self._current_element, new_name)

    def _save_element_settings(self) -> None:
        if not self._library or not self._current_element:
            return

        ratio_name = self._edit_norm_ratio.text().strip().replace("\\", "/")
        value_text = self._edit_norm_value.text().strip()
        reference_data = self._collect_reference_data_from_tables()
        if reference_data is None:
            return

        elem = self._library.add_element(self._current_element)

        if not ratio_name and not value_text:
            elem.internal_normalization = None
            elem.reference_data = (
                reference_data
                if reference_data.masses or reference_data.natural_ratios
                else None
            )
            self.accept_draft()
            self.element_saved.emit(self._current_element)
            return

        if ratio_name.count("/") != 1:
            QMessageBox.warning(
                self,
                "Validation",
                "Internal normalization ratio must be in A/B format.",
            )
            return
        if not value_text:
            QMessageBox.warning(
                self,
                "Validation",
                "Internal normalization value is required.",
            )
            return
        try:
            value = float(value_text)
        except ValueError:
            QMessageBox.warning(
                self,
                "Validation",
                "Internal normalization value must be numeric.",
            )
            return
        if not math.isfinite(value) or value <= 0:
            # "nan" parses and is not <= 0, so a bare positivity test let it
            # through; the accessor then reads the saved normalization as
            # absent (audit A089).
            QMessageBox.warning(
                self,
                "Validation",
                "Internal normalization value must be a finite value > 0.",
            )
            return

        # Patch the two visible fields into the stored definition. The
        # convention is schema-3 metadata with no editor control, so rebuilding
        # the record from the form alone silently replaced Sr's
        # "internal_normalization_exponential_law" with "unspecified"
        # (audit A085).
        stored_norm = elem.internal_normalization
        elem.internal_normalization = InternalNormalization(
            ratio_name=ratio_name,
            value=value,
            convention=getattr(stored_norm, "convention", "") or "unspecified",
        )
        elem.reference_data = (
            reference_data
            if reference_data.masses or reference_data.natural_ratios
            else None
        )
        self.accept_draft()
        self.element_saved.emit(self._current_element)

    def _discard(self) -> None:
        if self._mode == "crm" and self._current_element and self._current_crm:
            self.show_crm(self._current_element, self._current_crm)
        elif self._mode == "element" and self._current_element:
            self.show_element(self._current_element)

    # Mode helper

    def _set_mode(self, mode: str) -> None:
        # A stale "unavailable" carried onto a healthy record would be its own
        # misstatement, so the status resets on every record change, import and
        # navigation. show_crm recomputes it immediately afterwards.
        self._set_preview_status("")
        self._mode = mode
        is_crm = mode == "crm"
        is_element = mode == "element"
        is_active = mode in ("crm", "element")

        self._meta_group.setVisible(is_crm)
        self._ratio_group.setVisible(is_crm)
        self._derived_group.setVisible(is_crm)
        self._mass_group.setVisible(is_crm)
        self._norm_group.setVisible(is_element)
        self._refdata_group.setVisible(is_element)

        for w in (
            self._edit_name,
            self._edit_source,
            self._edit_description,
            self._tbl_ratios,
            self._tbl_derived,
            self._tbl_masses,
            self._btn_add_ratio,
            self._btn_derive_ratio,
            self._btn_del_ratio,
            self._btn_add_mass,
            self._btn_del_mass,
        ):
            w.setEnabled(is_crm)

        for w in (
            self._edit_norm_ratio,
            self._edit_norm_value,
            self._tbl_ref_masses,
            self._tbl_ref_ratios,
            self._btn_add_ref_mass,
            self._btn_del_ref_mass,
            self._btn_add_ref_ratio,
            self._btn_del_ref_ratio,
        ):
            w.setEnabled(is_element)

        self._btn_save.setEnabled(is_active)
        self._btn_discard.setEnabled(is_active)
        self.accept_draft()

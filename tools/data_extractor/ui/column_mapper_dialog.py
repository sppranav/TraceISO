"""Dual-list column mapper dialog for isotope / ratio assignment."""

from typing import Optional, Tuple

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QDialogButtonBox, QLineEdit, QGroupBox,
    QFormLayout, QMessageBox,
)
from PyQt5.QtCore import Qt

from tools.data_extractor.mapper_logic import (
    DISAMBIGUATED_COLUMN_PATTERN,
    ISOTOPE_NAME_PATTERN,
    POSITIONAL_COLUMN_PATTERN,
)

UNNAMED = "(type a name)"


def _default_target_name(source_col: str) -> str:
    """Return a clean target name from a source column label.

    A headerless column ('col 23') has no name to offer: it starts empty and
    the user types one.
    """
    text = str(source_col).strip()
    if POSITIONAL_COLUMN_PATTERN.fullmatch(text):
        return ""
    match = DISAMBIGUATED_COLUMN_PATTERN.fullmatch(text)
    return match.group("name") if match else text


def _item_text(source: str, target: str) -> str:
    return f"{source}  ->  {target or UNNAMED}"


class ColumnMapperDialog(QDialog):
    """Dual-list dialog: Available Columns  <-->  Mapped Columns."""

    def __init__(self, available_cols, already_mapped, mode="isotope", parent=None,
                 *, preselect=None, require_isotope_names=False):
        """
        Parameters
        ----------
        available_cols : list[str]
            All column names from the header row.
        already_mapped : dict[str, str]
            Existing mapping {source_col: target_name}.
        mode : str
            "isotope" or "ratio" - only affects the title.
        preselect : list[str], optional
            Columns chosen in the grid; they start on the mapped side, so the
            user only names them instead of searching the full column list.
        require_isotope_names : bool
            Every target must be a canonical isotope name such as ``87Sr``.
            A position such as ``col 23`` says nothing about the isotope, so
            it is never accepted as one.
        """
        super().__init__(parent)
        self.setWindowTitle(f"Map Columns -> {mode.title()}s")
        self.resize(620, 420)
        self._result_mapping = dict(already_mapped)
        self.require_isotope_names = require_isotope_names

        root = QVBoxLayout(self)

        body = QHBoxLayout()

        # ---- Available ----
        left_box = QGroupBox("Available Columns")
        ll = QVBoxLayout(left_box)
        self.avail_list = QListWidget()
        self.avail_list.setSelectionMode(QListWidget.ExtendedSelection)
        for col in available_cols:
            if col not in already_mapped:
                self.avail_list.addItem(col)
        ll.addWidget(self.avail_list)
        body.addWidget(left_box)

        # ---- Buttons ----
        btn_col = QVBoxLayout()
        btn_col.addStretch()
        self.btn_add = QPushButton("Add >")
        self.btn_add.clicked.connect(self._move_right)
        btn_col.addWidget(self.btn_add)
        self.btn_remove = QPushButton("< Remove")
        self.btn_remove.clicked.connect(self._move_left)
        btn_col.addWidget(self.btn_remove)
        btn_col.addStretch()
        body.addLayout(btn_col)

        # ---- Mapped ----
        right_box = QGroupBox(f"Mapped {mode.title()}s")
        rl = QVBoxLayout(right_box)
        self.mapped_list = QListWidget()
        self.mapped_list.setSelectionMode(QListWidget.ExtendedSelection)
        for src, tgt in already_mapped.items():
            if src == tgt:
                tgt = _default_target_name(src)
            it = QListWidgetItem(_item_text(src, tgt))
            it.setData(Qt.UserRole, (src, tgt))
            self.mapped_list.addItem(it)
        rl.addWidget(self.mapped_list)

        # rename area: also where an unnamed (headerless) column gets its name
        rename_lay = QFormLayout()
        self.rename_input = QLineEdit()
        self.rename_input.setPlaceholderText(
            "Isotope name, e.g. 87Sr" if require_isotope_names else "Target name")
        self.rename_input.returnPressed.connect(self._apply_rename)
        rename_lay.addRow("Name:", self.rename_input)
        self.btn_rename = QPushButton("Apply Name")
        self.btn_rename.clicked.connect(self._apply_rename)
        rename_lay.addRow("", self.btn_rename)
        rl.addLayout(rename_lay)

        body.addWidget(right_box)
        root.addLayout(body)

        # ---- Dialog buttons ----
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        # Enter in the name box applies the name; it must not close the dialog.
        for button in (*btns.buttons(), self.btn_add, self.btn_remove, self.btn_rename):
            button.setAutoDefault(False)
            button.setDefault(False)
        root.addWidget(btns)

        self.mapped_list.currentItemChanged.connect(self._on_select_mapped)

        if preselect:
            wanted = set(preselect)
            for i in range(self.avail_list.count()):
                item = self.avail_list.item(i)
                item.setSelected(item.text() in wanted)
            self._move_right(validate=False)
            if self.mapped_list.count():
                self._select_next_unnamed(fallback=self.mapped_list.count() - 1)
                self.rename_input.setFocus()
                self.rename_input.selectAll()

    # ---- actions ----
    def _move_right(self, validate=True):
        for it in self.avail_list.selectedItems():
            col = it.text()
            target = _default_target_name(col)
            new_it = QListWidgetItem(_item_text(col, target))
            new_it.setData(Qt.UserRole, (col, target))
            self.mapped_list.addItem(new_it)
            self.avail_list.takeItem(self.avail_list.row(it))
        # Disambiguated duplicate headers (e.g. "88Sr [col 1]" / "88Sr [col 2]")
        # both default to the same clean target — surface that immediately
        # rather than waiting for Ok, without undoing the move.
        if validate:
            self._validate_mapping()

    def _move_left(self):
        for it in self.mapped_list.selectedItems():
            src, _tgt = it.data(Qt.UserRole)
            self.avail_list.addItem(src)
            self.mapped_list.takeItem(self.mapped_list.row(it))

    def _on_select_mapped(self, current, _prev):
        if current:
            _src, tgt = current.data(Qt.UserRole)
            self.rename_input.setText(tgt)

    def _apply_rename(self):
        cur = self.mapped_list.currentItem()
        if not cur:
            return
        src, old_tgt = cur.data(Qt.UserRole)
        new_name = self.rename_input.text().strip()
        if not new_name:
            QMessageBox.warning(
                self, "Invalid target name",
                "Target name cannot be empty.",
            )
            return

        # Judge only the name being entered: other rows may still be waiting
        # for their own names (several selected columns arrive unnamed), and
        # those are checked when the dialog is accepted.
        if self.require_isotope_names and not ISOTOPE_NAME_PATTERN.fullmatch(new_name):
            QMessageBox.warning(
                self, "Invalid target name",
                f"'{new_name}' is not an isotope name; use the form 87Sr.",
            )
            return
        for i in range(self.mapped_list.count()):
            other = self.mapped_list.item(i)
            if other is not cur and str(other.data(Qt.UserRole)[1]).strip() == new_name:
                QMessageBox.warning(
                    self, "Duplicate target name",
                    f"Target name '{new_name}' is already assigned to "
                    f"'{other.data(Qt.UserRole)[0]}'.",
                )
                return
        cur.setData(Qt.UserRole, (src, new_name))
        cur.setText(_item_text(src, new_name))
        # Move on to the next column still waiting for a name.
        if self._select_next_unnamed():
            self.rename_input.setFocus()

    def _select_next_unnamed(self, fallback=None) -> bool:
        for i in range(self.mapped_list.count()):
            if not str(self.mapped_list.item(i).data(Qt.UserRole)[1]).strip():
                self.mapped_list.setCurrentRow(i)
                return True
        if fallback is not None:
            self.mapped_list.setCurrentRow(fallback)
        return False

    # ---- validation ----
    def _find_collision(
        self,
    ) -> Optional[Tuple[QListWidgetItem, QListWidgetItem, str]]:
        """Return (existing_item, colliding_item, target) for the first
        empty or duplicate target, or ``None`` when the mapping is clean."""
        seen: dict = {}
        for i in range(self.mapped_list.count()):
            item = self.mapped_list.item(i)
            _src, tgt = item.data(Qt.UserRole)
            tgt = str(tgt).strip()
            if not tgt:
                return item, item, tgt
            if self.require_isotope_names and not ISOTOPE_NAME_PATTERN.fullmatch(tgt):
                return item, item, tgt
            if tgt in seen:
                return seen[tgt], item, tgt
            seen[tgt] = item
        return None

    def _validate_mapping(self, *, interactive: bool = True) -> bool:
        """Ensure every mapped target is non-empty and unique.

        Used from ``_move_right``, ``_apply_rename``, and ``accept()`` so an
        automatically-defaulted collision (two disambiguated source columns
        defaulting to the same target) is caught the same way as a manual
        rename collision. On failure, highlights the pre-existing owner of
        the colliding target and shows a precise warning naming both source
        columns; the caller is responsible for leaving the dialog open.
        """
        collision = self._find_collision()
        if collision is None:
            return True
        existing_item, colliding_item, target = collision
        if interactive:
            self.mapped_list.setCurrentItem(existing_item)
            existing_src, _ = existing_item.data(Qt.UserRole)
            colliding_src, _ = colliding_item.data(Qt.UserRole)
            if existing_item is colliding_item:
                QMessageBox.warning(
                    self, "Invalid target name",
                    f"Source column '{existing_src}' needs "
                    + ("an isotope name such as 87Sr" if self.require_isotope_names
                       else "a target name")
                    + f" (now '{target}'). Rename it before continuing.",
                )
            else:
                QMessageBox.warning(
                    self, "Duplicate target name",
                    f"Target name '{target}' is assigned to both source "
                    f"columns '{existing_src}' and '{colliding_src}'. Choose "
                    "a unique, scientifically meaningful target for each.",
                )
        return False

    def accept(self):
        if not self._validate_mapping():
            return
        super().accept()

    # ---- result ----
    def get_mapping(self):
        """Returns dict {source_col: target_name}.

        ``accept()`` already validates the mapping before the dialog can
        close, so this raise is a defensive final backstop for callers that
        bypass ``exec_()`` acceptance (e.g. programmatic use) — every
        interactive main-window call site catches it.
        """
        collision = self._find_collision()
        if collision is not None:
            existing_item, colliding_item, target = collision
            existing_src, _ = existing_item.data(Qt.UserRole)
            colliding_src, _ = colliding_item.data(Qt.UserRole)
            if existing_item is colliding_item:
                raise ValueError(
                    f"Source column '{existing_src}' has an invalid target name {target!r}."
                )
            raise ValueError(
                f"Duplicate target name '{target}' assigned to both "
                f"'{existing_src}' and '{colliding_src}'."
            )

        result = {}
        for i in range(self.mapped_list.count()):
            src, tgt = self.mapped_list.item(i).data(Qt.UserRole)
            result[src] = str(tgt).strip()
        return result

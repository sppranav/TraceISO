"""
CRM & Reference Data Manager -- main application window.

Toolbar, splitter, tree panel, detail editor, status bar,
plus all file I/O and dialog logic.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QIcon, QKeySequence, QFont
from PyQt5.QtWidgets import (
    QMainWindow, QSplitter, QStatusBar, QToolBar, QAction,
    QMessageBox, QFileDialog, QInputDialog, QWidget, QVBoxLayout,
    QLabel, QDialog, QTextEdit, QDialogButtonBox, QStyle, QLineEdit, QListWidget, QListWidgetItem, QApplication, QShortcut,
)

from tools.qt_runtime import mono_font_family
from tools.desktop_theme import style_window, style_toolbar

from tools.crm_manager.models.crm_data import (
    CRMLibrary, CRMLibraryConflictError, CRMPayloadError, ReferenceMaterial,
    CertifiedRatio, crm_library_revision, load_library, save_library,
    validate_library, _library_to_dict,
)
from tools.crm_manager.constants import DEFAULT_CRM_LIBRARY_PATH
from tools.crm_manager.widgets.tree_panel import CRMTreePanel
from tools.crm_manager.widgets.detail_editor import DetailEditor
from tools.crm_manager.widgets.workflow import new_crm, replacement_summary, confirm_replacement


_DEFAULT_JSON = DEFAULT_CRM_LIBRARY_PATH


class CRMManagerWindow(QMainWindow):
    """Main window for the CRM & Reference Data Manager."""

    def __init__(self, json_path=None):
        super().__init__()
        self._json_path = json_path or _DEFAULT_JSON
        self._library = load_library(self._json_path)
        # The revision this window is editing against. A save that no longer
        # matches what is on disk is a conflict, not an overwrite.
        self._loaded_revision = crm_library_revision(self._json_path)
        self._unsaved = False

        style_window(self, "CRM Library Manager")
        self.setMinimumSize(960, 640)
        self.resize(1100, 720)

        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

        self._refresh_tree()
        self._update_status()

    # UI construction

    def _build_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        tb.setMovable(False)
        self.addToolBar(tb)

        style = self.style()

        self._act_add_element = QAction("Add Element", self)
        self._act_add_element.setIcon(style.standardIcon(QStyle.SP_DirIcon))
        self._act_add_element.setToolTip("Add a new element group")
        self._act_add_element.triggered.connect(self._on_add_element)
        tb.addAction(self._act_add_element)

        self._act_add_crm = QAction("Add CRM", self)
        self._act_add_crm.setIcon(style.standardIcon(QStyle.SP_FileIcon))
        self._act_add_crm.setToolTip("Add a new reference material")
        self._act_add_crm.triggered.connect(self._on_add_crm)
        tb.addAction(self._act_add_crm)

        tb.addSeparator()

        self._act_del = QAction("Delete", self)
        self._act_del.setIcon(style.standardIcon(QStyle.SP_TrashIcon))
        self._act_del.setToolTip("Delete selected element or CRM (Del)")
        self._act_del.triggered.connect(self._on_delete)
        tb.addAction(self._act_del)

        tb.addSeparator()

        self._act_import = QAction("Replace Library from JSON...", self)
        self._act_import.setIcon(style.standardIcon(QStyle.SP_DialogOpenButton))
        self._act_import.setToolTip("Import CRM library from JSON")
        self._act_import.triggered.connect(self._on_import)
        tb.addAction(self._act_import)

        self._act_export = QAction("Export", self)
        self._act_export.setIcon(style.standardIcon(QStyle.SP_DialogSaveButton))
        self._act_export.setToolTip("Export CRM library to JSON")
        self._act_export.triggered.connect(self._on_export)
        tb.addAction(self._act_export)

        tb.addSeparator()

        self._act_save = QAction("Save Library to Disk", self)
        self._act_save.setIcon(style.standardIcon(QStyle.SP_DriveFDIcon))
        self._act_save.setToolTip("Save changes to file (Ctrl+S)")
        self._act_save.setShortcut(QKeySequence.Save)
        self._act_save.triggered.connect(self._on_save_file)
        tb.addAction(self._act_save)

        tb.addSeparator()

        self._act_validate = QAction("Validate", self)
        self._act_validate.setIcon(style.standardIcon(QStyle.SP_MessageBoxInformation))
        self._act_validate.setToolTip("Check library for inconsistencies")
        self._act_validate.triggered.connect(self._on_validate)
        tb.addAction(self._act_validate)
        style_toolbar(tb, primary=self._act_save, destructive=self._act_del)

    def _build_central(self):
        splitter = QSplitter(Qt.Horizontal)

        # Left panel -- tree
        self._tree = CRMTreePanel()
        self._tree.setMinimumWidth(260)
        browser = QWidget()
        browser_layout = QVBoxLayout(browser)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search CRM or element...")
        self._search.textEdited.connect(self._filter_tree)
        browser_layout.addWidget(self._search)
        browser_layout.addWidget(self._tree)
        splitter.addWidget(browser)
        delete_key = QShortcut(QKeySequence(Qt.Key_Delete), self._tree)
        delete_key.setContext(Qt.WidgetShortcut)
        delete_key.activated.connect(self._on_delete)

        # Right panel -- detail editor
        self._editor = DetailEditor()
        self._editor.set_library(self._library)
        editor_panel = QWidget()
        editor_layout = QVBoxLayout(editor_panel)
        editor_layout.addWidget(self._editor)
        editor_layout.addWidget(self._editor.actions)
        self._draft_label = QLabel()
        editor_layout.addWidget(self._draft_label)
        self._draft_timer = QTimer(self)
        self._draft_timer.timeout.connect(lambda: self._draft_label.setText(
            "Pending form edits" if self._editor.has_draft else
            ("Library changes not saved to disk" if self._unsaved else "Saved to disk")))
        self._draft_timer.start(250)
        splitter.addWidget(editor_panel)

        splitter.setSizes([300, 700])
        self.setCentralWidget(splitter)

        # Connections
        self._tree.crm_selected.connect(self._on_crm_selected)
        self._tree.element_selected.connect(self._on_element_selected)
        self._tree.selection_cleared.connect(self._on_selection_cleared)
        self._editor.crm_saved.connect(self._on_crm_saved)
        self._editor.element_saved.connect(self._on_element_saved)

    def _build_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

    # Refresh helpers

    def _refresh_tree(self, select=None):
        self._tree.populate(self._library)
        if select:
            self._tree.select_crm(*select)

    def _update_status(self):
        n_elem = len(self._library.elements)
        n_crm = self._library.total_crms()
        modified = " | Modified" if self._unsaved else ""
        self._statusbar.showMessage(
            "{} elements, {} CRMs loaded | {}{}".format(
                n_elem, n_crm, str(self._json_path), modified
            )
        )

    def _mark_dirty(self):
        self._unsaved = True
        self._update_status()

    def _changed_crm_count(self) -> int:
        """Return count of CRM entries whose serialized payload changed."""
        try:
            old = _library_to_dict(load_library(self._json_path))
        except Exception:  # noqa: BLE001 - missing/unreadable existing file
            old = {"elements": {}}
        new = _library_to_dict(self._library)
        old_entries = {}
        new_entries = {}
        for symbol, elem in old.get("elements", {}).items():
            for name, payload in elem.get("reference_materials", {}).items():
                old_entries[(symbol, name)] = payload
        for symbol, elem in new.get("elements", {}).items():
            for name, payload in elem.get("reference_materials", {}).items():
                new_entries[(symbol, name)] = payload
        keys = set(old_entries) | set(new_entries)
        return sum(1 for key in keys if old_entries.get(key) != new_entries.get(key))

    # Toolbar actions

    def _on_add_element(self):
        if not self._resolve_draft():
            return
        text, ok = QInputDialog.getText(
            self, "Add Element", "Element symbol (e.g. Fe):",
        )
        if ok and text.strip():
            sym = text.strip().capitalize()
            if sym in self._library.elements:
                QMessageBox.information(
                    self, "Exists", "{} already exists.".format(sym),
                )
                return
            self._library.add_element(sym)
            self._mark_dirty()
            self._refresh_tree()

    def _on_add_crm(self):
        if not self._resolve_draft():
            return
        symbols = self._library.element_symbols()
        if not symbols:
            QMessageBox.information(
                self, "No Elements",
                "Add an element first.",
            )
            return

        result = new_crm(self, symbols, self._editor._current_element)
        if result is None:
            return
        sym, name, source = result
        if self._library.get_crm(sym, name) is not None:
            # Adding an existing name used to overwrite the certificate with a
            # blank record, and the only later signal was a "no ratios"
            # warning that does not block Save.
            QMessageBox.warning(
                self,
                "CRM Exists",
                f"{sym} already has a reference material named '{name}'.\n\n"
                "Opening it instead. Use a different name to add a new record.",
            )
            self._refresh_tree(select=(sym, name))
            return

        crm = ReferenceMaterial(name=name, element=sym, source=source)
        self._library.add_crm(sym, crm)
        self._mark_dirty()
        self._refresh_issues()
        self._refresh_tree(select=(sym, name))

    def _on_delete(self):
        items = self._tree.selectedItems()
        if not items:
            return

        item = items[0]
        from tools.crm_manager.widgets.tree_panel import (
            _ROLE_TYPE, _ROLE_ELEMENT, _ROLE_CRM,
        )

        node_type = item.data(0, _ROLE_TYPE)
        element = item.data(0, _ROLE_ELEMENT)

        if node_type == "ratio" or not self._resolve_draft():
            return

        if node_type == "element":
            msg = "Delete element '{}' and ALL its CRMs?".format(element)
            reply = QMessageBox.question(
                self, "Delete Element", msg,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._library.remove_element(element)
                self._mark_dirty()
                self._editor.clear_editor()
                self._refresh_tree()

        elif node_type == "crm":
            crm_name = item.data(0, _ROLE_CRM)
            msg = "Delete '{}' from {}?".format(crm_name, element)
            reply = QMessageBox.question(
                self, "Delete CRM", msg,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._library.remove_crm(element, crm_name)
                self._mark_dirty()
                self._editor.clear_editor()
                self._refresh_tree()

    def _on_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Replace Library from JSON", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            candidate = load_library(Path(path))
            issues = validate_library(candidate)
            errors = [i.message for i in issues if i.level == "error"]
            if errors:
                QMessageBox.warning(self, "Invalid replacement", "\n".join(errors))
                return
            summary, details = replacement_summary(_library_to_dict(self._library), _library_to_dict(candidate))
            if not confirm_replacement(self, path, self._json_path, summary, details,
                                       self._unsaved or self._editor.has_draft):
                return
            self._library = candidate
            self._editor.set_library(candidate)
            self._editor.clear_editor()
            self._mark_dirty()
            self._refresh_tree()
        except Exception as exc:
            QMessageBox.critical(self, "Import Error", str(exc))

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CRM Library", "crm_library.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        target = Path(path)
        try:
            same_file = target.resolve() == self._json_path.resolve()
        except OSError:  # pragma: no cover - unresolvable path
            same_file = False
        if same_file:
            # Export does not carry this window's revision token, so writing
            # the live library through it would bypass the stale-snapshot
            # check that Save performs.
            QMessageBox.warning(
                self,
                "Export",
                "That is the library this window is editing. Use Save to "
                "write it, so a change made by another session is not "
                "overwritten.",
            )
            return
        try:
            save_library(self._library, target)
            QMessageBox.information(
                self, "Exported",
                "Saved to {}".format(target.name),
            )
        except CRMPayloadError as invalid:
            QMessageBox.critical(self, "Export Error", str(invalid))
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_save_file(self):
        if not self._editor.apply_draft():
            return
        try:
            # No "Save anyway?" override. It was inert for the errors the
            # payload gate already refused, and for the rest it wrote records
            # the application reads as absent or reinterprets (audit A089).
            # The model gate in save_library refuses either way; asking first
            # only lets the message name what to correct.
            changed = self._changed_crm_count()
            write_warnings: list = []
            self._loaded_revision = save_library(
                self._library,
                self._json_path,
                expected_revision=self._loaded_revision,
                warnings=write_warnings,
            )
            # The primary committed, so the edit is saved and the window is
            # clean whatever happened to the recovery copy. Reporting a
            # degraded backup as a failed save would be the opposite lie to the
            # silence it replaces (audit A084).
            self._unsaved = False
            self._update_status()
            summary = (
                f"Saved {self._json_path.name}; "
                f"{changed} CRM entr{'y' if changed == 1 else 'ies'} changed. "
                "Click Reload CRM Library in TraceISO to apply changes."
            )
            body = (
                f"Saved {self._json_path.name}.\n\n"
                f"{changed} CRM entr{'y' if changed == 1 else 'ies'} changed.\n"
                "Click Reload CRM Library in TraceISO to apply changes."
            )
            if write_warnings:
                detail = "\n".join(w.message for w in write_warnings)
                self._statusbar.showMessage(
                    summary + " Recovery backup not updated."
                )
                QMessageBox.warning(
                    self,
                    "Saved, backup not updated",
                    f"{body}\n\n{detail}",
                )
            else:
                self._statusbar.showMessage(summary)
        except CRMPayloadError as invalid:
            # Nothing was written. The draft stays in memory and stays dirty.
            QMessageBox.critical(self, "Save Error", str(invalid))
        except CRMLibraryConflictError as conflict:
            # The draft stays in memory and stays dirty: nothing was written,
            # and the user can reload and reapply rather than lose the edit.
            QMessageBox.warning(self, "Save Conflict", str(conflict))
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _on_validate(self):
        if not self._resolve_draft():
            return
        if not hasattr(self, "_issues_dialog"):
            self._issues_dialog = QDialog(self)
            self._issues_dialog.setWindowTitle("Library validation — select an issue")
            self._issues_dialog.resize(700, 350)
            layout = QVBoxLayout(self._issues_dialog)
            self._issues_list = QListWidget()
            self._issues_list.itemClicked.connect(self._visit_issue)
            layout.addWidget(self._issues_list)
        self._refresh_issues()
        self._issues_dialog.show()

    def _refresh_issues(self):
        if not hasattr(self, "_issues_list"):
            return
        self._issues_list.clear()
        for issue in validate_library(self._library):
            item = QListWidgetItem(f"{issue.level.upper()} [{issue.element}] {issue.crm}: {issue.message}")
            item.setData(Qt.UserRole, issue)
            self._issues_list.addItem(item)
        if not self._issues_list.count():
            self._issues_list.addItem("No issues found.")

    def _visit_issue(self, item):
        issue = item.data(Qt.UserRole)
        if issue is None or not self._resolve_draft():
            return
        self._search.clear()
        self._tree.filter_text("")
        if self._library.get_crm(issue.element, issue.crm):
            self._tree.select_crm(issue.element, issue.crm)
        else:
            self._on_element_selected(issue.element)
        for table in (self._editor._tbl_ratios, self._editor._tbl_ref_ratios):
            for row in range(table.rowCount()):
                cell = table.item(row, 0)
                if cell and cell.text() and cell.text() in issue.message:
                    table.setCurrentCell(row, 0)
                    table.scrollToItem(cell)
                    self._editor.ensureWidgetVisible(table)
                    table.setFocus()
                    return
        self._editor.setFocus()

    def _restore_selection(self):
        self._tree.blockSignals(True)
        self._tree.select_record(self._editor._current_element, self._editor._current_crm if self._editor._mode == "crm" else None)
        self._tree.blockSignals(False)

    def _resolve_draft(self):
        self._editor.commit_active_cell()
        if not self._editor.has_draft:
            return True
        answer = QMessageBox.question(self, "Pending form edits",
            "Apply this form to the library before continuing? Save applies to memory only; use Save Library to Disk to persist.",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel)
        if answer == QMessageBox.Save:
            return self._editor.apply_draft()
        if answer == QMessageBox.Discard:
            self._editor._discard()
            return True
        return False

    def _filter_tree(self, text):
        if not self._resolve_draft():
            self._search.setText(getattr(self, "_filter", ""))
            return
        self._filter = text
        self._tree.filter_text(text)

    def _on_crm_selected(self, element, crm_name):
        if not self._resolve_draft():
            self._restore_selection()
            return
        self._editor.show_crm(element, crm_name)

    def _on_element_selected(self, element):
        if not self._resolve_draft():
            self._restore_selection()
            return
        self._editor.show_element(element)

    def _on_selection_cleared(self):
        if self._resolve_draft():
            self._editor.clear_editor()
        else:
            self._restore_selection()

    # Editor save handler

    def _on_crm_saved(self, element, crm_name):
        self._mark_dirty()
        self._refresh_issues()
        self._refresh_tree(select=(element, crm_name))

    def _on_element_saved(self, element):
        self._mark_dirty()
        self._refresh_issues()
        self._refresh_tree()
        self._editor.show_element(element)

    # Close guard

    def closeEvent(self, event):
        if self._unsaved or self._editor.has_draft:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self._on_save_file()
                if self._unsaved or self._editor.has_draft:
                    # Save failed — keep window open
                    event.ignore()
                    return
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

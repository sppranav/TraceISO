"""Small dialogs shared by the CRM authoring workflow."""

import difflib
import json

from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
    QMessageBox,
)


def new_crm(parent, symbols, current):
    dialog = QDialog(parent)
    dialog.setWindowTitle("Add CRM")
    layout = QFormLayout(dialog)
    element = QComboBox()
    element.addItems(symbols)
    element.setCurrentText(current or symbols[0])
    name, source = QLineEdit(), QLineEdit()
    name.setPlaceholderText("e.g. NIST SRM 987")
    layout.addRow("Element", element)
    layout.addRow("Name", name)
    layout.addRow("Source", source)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.button(QDialogButtonBox.Ok).setEnabled(False)
    name.textChanged.connect(lambda text: buttons.button(QDialogButtonBox.Ok).setEnabled(bool(text.strip())))
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addRow(buttons)
    if dialog.exec_() != QDialog.Accepted:
        return None
    return element.currentText(), name.text().strip(), source.text().strip()


def replacement_summary(old, new):
    """Include element settings as well as CRM additions/removals/edits."""
    def entries(payload):
        result = {}
        for symbol, element in payload.get("elements", {}).items():
            result[f"{symbol}: element settings"] = {
                k: v for k, v in element.items() if k != "reference_materials"
            }
            for name, record in element.get("reference_materials", {}).items():
                result[f"{symbol}: {name}"] = record
        return result
    before, after = entries(old), entries(new)
    added, removed = after.keys() - before.keys(), before.keys() - after.keys()
    changed = {key for key in before.keys() & after.keys() if before[key] != after[key]}
    summary = f"{len(added)} added, {len(removed)} removed, {len(changed)} changed records/settings."
    details = "\n".join(difflib.unified_diff(
        json.dumps(old, indent=2, sort_keys=True).splitlines(),
        json.dumps(new, indent=2, sort_keys=True).splitlines(),
        fromfile="Current library", tofile="Replacement", lineterm="",
    ))
    return summary, details


def confirm_replacement(parent, source, destination, summary, details, pending):
    box = QMessageBox(parent)
    box.setWindowTitle("Replace Library from JSON")
    box.setIcon(QMessageBox.Warning)
    box.setText(f"Source: {source}\nDestination: {destination}\n\n{summary}")
    box.setInformativeText(
        ("Pending form and library changes will be discarded.\n" if pending else "")
        + "The replacement stays in memory until Save Library to Disk."
    )
    box.setDetailedText(details or "No differences.")
    box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
    box.setDefaultButton(QMessageBox.Cancel)
    return box.exec_() == QMessageBox.Yes

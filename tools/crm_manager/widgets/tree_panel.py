"""Tree panel widget — left side of the CRM & Reference Data Manager."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem, QStyle

from tools.crm_manager.formatting import format_display_uncertainty
from tools.crm_manager.models.crm_data import CRMLibrary


_ROLE_TYPE = Qt.UserRole      # "element", "crm", "ratio"
_ROLE_ELEMENT = Qt.UserRole + 1
_ROLE_CRM = Qt.UserRole + 2
_ROLE_RATIO = Qt.UserRole + 3


class CRMTreePanel(QTreeWidget):
    """Hierarchical CRM browser."""

    # Emitted when a CRM node is selected: (element_symbol, crm_name)
    crm_selected = pyqtSignal(str, str)
    # Emitted when an element node is selected: (element_symbol,)
    element_selected = pyqtSignal(str)
    # Emitted when nothing meaningful is selected
    selection_cleared = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["CRM & Reference Data"])
        self.setColumnCount(1)
        self.setIndentation(20)
        self.setAnimated(True)
        self.setRootIsDecorated(True)
        self.setExpandsOnDoubleClick(True)
        self.itemSelectionChanged.connect(self._on_selection_changed)

    # Public API

    def populate(self, library: CRMLibrary) -> None:
        """Rebuild the tree from a CRMLibrary."""
        expanded = {self._key(i) for i in self._items() if i.isExpanded()}
        selected = self._key(self.currentItem()) if self.currentItem() else None
        scroll = self.verticalScrollBar().value()
        first = self.topLevelItemCount() == 0
        self.blockSignals(True)
        self.clear()

        style = self.style()

        for sym in library.element_symbols():
            elem = library.get_element(sym)
            if elem is None:
                continue

            # Element node
            elem_item = QTreeWidgetItem(self, [sym])
            elem_item.setIcon(0, style.standardIcon(QStyle.SP_DirIcon))
            elem_item.setData(0, _ROLE_TYPE, "element")
            elem_item.setData(0, _ROLE_ELEMENT, sym)
            elem_item.setFlags(
                elem_item.flags() | Qt.ItemIsSelectable
            )

            for crm_name in elem.crm_names():
                crm = elem.reference_materials[crm_name]
                # CRM node
                crm_item = QTreeWidgetItem(elem_item, [crm_name])
                crm_item.setIcon(0, style.standardIcon(QStyle.SP_FileIcon))
                crm_item.setData(0, _ROLE_TYPE, "crm")
                crm_item.setData(0, _ROLE_ELEMENT, sym)
                crm_item.setData(0, _ROLE_CRM, crm_name)
                crm_item.setFlags(
                    crm_item.flags() | Qt.ItemIsSelectable
                )

                for rname, rdata in crm.ratios.items():
                    # Ratio leaf — show value preview
                    label = (
                        f"{rname}: {rdata.value:.6g} ± "
                        f"{format_display_uncertainty(rdata.uncertainty)}"
                    )
                    ratio_item = QTreeWidgetItem(crm_item, [label])
                    ratio_item.setIcon(0, style.standardIcon(QStyle.SP_FileLinkIcon))
                    ratio_item.setData(0, _ROLE_TYPE, "ratio")
                    ratio_item.setData(0, _ROLE_ELEMENT, sym)
                    ratio_item.setData(0, _ROLE_CRM, crm_name)
                    ratio_item.setData(0, _ROLE_RATIO, rname)
                    ratio_item.setFlags(
                        ratio_item.flags() | Qt.ItemIsSelectable
                    )

        for item in self._items():
            item.setExpanded(self._key(item) in expanded or (first and item.parent() is None))
            if self._key(item) == selected:
                self.setCurrentItem(item)
        self.filter_text(getattr(self, "_filter", ""))
        self.verticalScrollBar().setValue(scroll)
        self.blockSignals(False)

    def _items(self):
        def walk(parent):
            for n in range(parent.childCount()):
                child = parent.child(n)
                yield child
                yield from walk(child)
        return walk(self.invisibleRootItem())

    @staticmethod
    def _key(item):
        return tuple(item.data(0, role) for role in (_ROLE_TYPE, _ROLE_ELEMENT, _ROLE_CRM, _ROLE_RATIO))

    def select_record(self, element, crm=None):
        for item in self._items():
            kind = "crm" if crm else "element"
            if item.data(0, _ROLE_TYPE) == kind and item.data(0, _ROLE_ELEMENT) == element and (not crm or item.data(0, _ROLE_CRM) == crm):
                self.setCurrentItem(item)
                return

    def filter_text(self, text):
        self._filter = text
        needle = text.casefold().strip()
        for n in range(self.topLevelItemCount()):
            element = self.topLevelItem(n)
            element_match = needle in element.text(0).casefold()
            any_match = False
            for m in range(element.childCount()):
                crm = element.child(m)
                match = element_match or needle in crm.text(0).casefold()
                crm.setHidden(not match)
                any_match |= match
            element.setHidden(not (element_match or any_match))
            if needle and any_match:
                element.setExpanded(True)

    def select_crm(self, element: str, crm_name: str) -> None:
        """Programmatically select a CRM node."""
        root = self.invisibleRootItem()
        for i in range(root.childCount()):
            elem_item = root.child(i)
            if elem_item.data(0, _ROLE_ELEMENT) == element:
                for j in range(elem_item.childCount()):
                    crm_item = elem_item.child(j)
                    if crm_item.data(0, _ROLE_CRM) == crm_name:
                        self.setCurrentItem(crm_item)
                        return

    # Internals

    def _on_selection_changed(self) -> None:
        items = self.selectedItems()
        if not items:
            self.selection_cleared.emit()
            return

        item = items[0]
        node_type = item.data(0, _ROLE_TYPE)

        if node_type == "element":
            self.element_selected.emit(item.data(0, _ROLE_ELEMENT))

        elif node_type in ("crm", "ratio"):
            element = item.data(0, _ROLE_ELEMENT)
            crm_name = item.data(0, _ROLE_CRM)
            self.crm_selected.emit(element, crm_name)

        else:
            self.selection_cleared.emit()

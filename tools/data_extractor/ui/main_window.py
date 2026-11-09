
import os
import re
import json
import logging
import traceback
from html import escape
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QMenu,
    QSplitter, QGroupBox, QFormLayout, QLineEdit, QComboBox, QMessageBox,
    QProgressBar, QInputDialog, QApplication, QStatusBar, QAction,
    QTextEdit, QToolBar, QScrollArea, QFrame, QSizePolicy, QLayout, QCheckBox,
)
from PyQt5.QtCore import Qt, QPoint, QSettings
from PyQt5.QtGui import QColor, QFont, QDropEvent, QDragEnterEvent, QKeySequence

from tools.data_extractor.mapper_logic import (
    NAN_FRACTION_THRESHOLD,
    base_column_name,
    column_nan_fractions,
    derive_positional_layout,
    disambiguate_header_values,
    extract_metadata_from_df,
    extract_sample_data,
    freeze_template_data,
    load_file_as_df,
    normalize_template_schema,
    suggest_baseline_cols,
)
from tools.data_extractor.batch_worker import BatchWorker, ValidationWorker
from tools.data_extractor.ui.mapping_workflow import MappingWorkflow
from tools.data_extractor.ui.batch_report import BatchReport
from tools.desktop_theme import style_window, action_role
from tools.data_extractor.ui import theme
from tools.data_extractor.ui.stepper_widget import StepperWidget
from tools.data_extractor.ui.file_list_widget import EnhancedFileList
from tools.data_extractor.ui.onboarding_overlay import OnboardingOverlay
from tools.data_extractor.ui.column_mapper_dialog import ColumnMapperDialog

_TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
_SETTINGS_KEY_RECENT = "neptunedataextractor/recent_templates"
_LEGACY_SETTINGS_KEY_RECENT = "universalmapper/recent_templates"
MAX_RECENT = 5

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Template keys the window does not edit but must carry through preview,
# save and batch unchanged. The positional keys describe headerless exports
# (classic Nu Plasma). footer_marker is carried in both modes; a template
# without one gets the window's default "***".
SOURCE_TEMPLATE_KEYS = (
    "run_number_pattern", "intensity_baseline_corrected", "preserve_sample_name",
    "layout", "data_start_marker", "data_row_width", "baseline_cols", "baseline_enabled",
    "metadata_patterns",
    "footer_marker",
)


class DataExtractorWindow(QMainWindow, MappingWorkflow):
    def __init__(self):
        super().__init__()
        style_window(self, "Data Extractor")
        self.resize(1280, 820)
        self.setAcceptDrops(True)

        # State
        self.current_df = None
        self.current_filepath = None
        self.mapping = {
            "time_col": None,
            "cycle_col": None,
            "metadata_cells": {},
            "isotope_cols": {},
            "ratio_cols": {}
        }
        self.header_row_idx = 0
        self.footer_row_idx = None
        self._batch_worker = None
        self._validation_worker = None
        self._active_run_token = 0
        self._run_token_counter = 0
        self._validation_cancel_requested = False
        self._last_outcome = None
        self._workers = []
        # Settings key kept from the tool's earlier name, so saved window state and
        # recent folders carry over.
        self._settings = QSettings("TraceISO", "NeptuneDataExtractor")
        self._migrate_legacy_settings()

        self._init_ui()
        self._init_shortcuts()
        self._init_mapping_workflow()
        self._update_conversion_scope()
        self.batch_report = BatchReport(self)
        self.batch_report.retry_requested.connect(self._retry_failed)

    def _migrate_legacy_settings(self):
        """Copy recent templates from the former UniversalMapper namespace."""
        if self._settings.contains(_SETTINGS_KEY_RECENT):
            return
        legacy = QSettings("TraceISO", "UniversalMapper")
        recents = legacy.value(_LEGACY_SETTINGS_KEY_RECENT, []) or []
        if recents:
            self._settings.setValue(_SETTINGS_KEY_RECENT, recents)

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 8, 12, 4)
        main_layout.setSpacing(8)

        # Stepper
        self.stepper = StepperWidget()
        main_layout.addWidget(self.stepper)

        # Toolbar
        toolbar_row = QHBoxLayout()
        toolbar_row.setSpacing(4)

        btn_load = QPushButton("Load File...")
        btn_load.setToolTip("Open a single data file (Ctrl+O)")
        btn_load.clicked.connect(self.load_file_dialog)
        toolbar_row.addWidget(btn_load)

        btn_load_folder = QPushButton("Load Folder (Batch)")
        btn_load_folder.setToolTip("Load all files from a folder (Ctrl+Shift+O)")
        btn_load_folder.clicked.connect(self.load_folder)
        toolbar_row.addWidget(btn_load_folder)

        self.combo_ext = QComboBox()
        self.combo_ext.addItems([".exp", ".csv", ".xlsx", ".xls", ".txt", "All Formats"])
        self.combo_ext.setFixedWidth(110)
        toolbar_row.addWidget(self.combo_ext)

        toolbar_row.addSpacing(12)

        self.btn_template_menu = QPushButton("Templates ▾")
        self.btn_template_menu.setToolTip("Load / Save / Built-in templates (Ctrl+L / Ctrl+S)")
        self.btn_template_menu.clicked.connect(self._show_template_menu)
        toolbar_row.addWidget(self.btn_template_menu)

        toolbar_row.addStretch()

        self.btn_clear_map = QPushButton("Clear Mappings")
        self.btn_clear_map.clicked.connect(self._clear_all_mappings)
        toolbar_row.addWidget(self.btn_clear_map)

        self.btn_convert = QPushButton("Convert...")
        self.btn_convert.setToolTip("Run batch HDF5 conversion (Ctrl+Return)")
        self.btn_convert.clicked.connect(self.run_batch_conversion)
        self.btn_convert.setObjectName("btnConvert")
        action_role(self.btn_convert, "primary")
        toolbar_row.addWidget(self.btn_convert)
        self.btn_selected = QPushButton("Convert Selected...")
        self.btn_selected.clicked.connect(lambda: self.run_batch_conversion(selected=True))
        toolbar_row.addWidget(self.btn_selected)
        report_button = QPushButton("Results")
        report_button.clicked.connect(lambda: self.batch_report.show())
        toolbar_row.addWidget(report_button)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self._cancel_batch)
        self.btn_cancel.setObjectName("btnCancel")
        action_role(self.btn_cancel, "danger")
        toolbar_row.addWidget(self.btn_cancel)

        main_layout.addLayout(toolbar_row)

        # Body: left (file list + grid) | right (controls + preview)
        body_splitter = QSplitter(Qt.Horizontal)

        # LEFT panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self.file_list_widget = EnhancedFileList()
        self.file_list_widget.file_selected.connect(self.load_file)
        self.file_list_widget.setMaximumHeight(155)
        left_layout.addWidget(self.file_list_widget)

        # Grid + onboarding overlay
        self._grid_container = QWidget()
        gc_lay = QVBoxLayout(self._grid_container)
        gc_lay.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        gc_lay.addWidget(self.table)

        self.onboarding = OnboardingOverlay(self._grid_container)
        self.onboarding.raise_()
        self.onboarding.show()

        left_layout.addWidget(self._grid_container, 1)

        self.file_list_widget.list_widget.model().rowsInserted.connect(self._update_conversion_scope)
        self.file_list_widget.list_widget.model().rowsRemoved.connect(self._update_conversion_scope)
        self.file_list_widget.list_widget.itemSelectionChanged.connect(self._update_conversion_scope)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)

        self.lbl_progress_file = QLabel("")
        self.lbl_progress_file.setStyleSheet(
            f"color:{theme.INK_DIM}; font-size:12px;"
        )
        left_layout.addWidget(self.lbl_progress_file)

        body_splitter.addWidget(left_widget)

        # RIGHT panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(6)
        right_layout.setSizeConstraint(QLayout.SetMinimumSize)

        self.controls_grp = QGroupBox("Mapping Controls")
        self.controls_grp.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
        form_layout = QFormLayout()
        form_layout.setHorizontalSpacing(12)
        form_layout.setVerticalSpacing(7)
        form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form_layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form_layout.setFormAlignment(Qt.AlignTop)
        form_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        form_layout.addRow("File:", self.lbl_file)

        self.lbl_header_row = QLabel("0")
        form_layout.addRow("Header Row:", self.lbl_header_row)

        self.lbl_footer_row = QLabel("(End of File)")
        form_layout.addRow("Footer Row:", self.lbl_footer_row)

        self.lbl_time = QLabel("(Not Set — optional)")
        self.lbl_time.setToolTip(
            "A column in the DATA TABLE with elapsed time per cycle (seconds/ms).\n"
            "Not the same as 'Analysis Time' in the header metadata.\n"
            "Neptune .exp files typically do not have a time column — leave as Not Set."
        )
        form_layout.addRow("Time Column:", self.lbl_time)

        self.lbl_cycle = QLabel("(Not Set)")
        self.lbl_cycle.setToolTip(
            "A column in the DATA TABLE containing cycle numbers (1, 2, 3...).\n"
            "Right-click the Cycle column in the grid and choose 'Map as Cycle Column'."
        )
        form_layout.addRow("Cycle Column:", self.lbl_cycle)

        self.lbl_isotopes = QLabel("0 Selected")
        form_layout.addRow("Isotopes:", self.lbl_isotopes)

        # Subtraction can be switched off without losing the assignments.
        self.chk_subtract_baselines = QCheckBox("Subtract baselines")
        self.chk_subtract_baselines.setToolTip(
            "On: every mapped isotope minus its assigned baseline column, cycle by cycle.\n"
            "Off: values are exported as read; baseline assignments are kept in the template.\n"
            "The output records whether subtraction was applied."
        )
        self.chk_subtract_baselines.toggled.connect(self._set_baseline_enabled)
        form_layout.addRow("Baselines:", self.chk_subtract_baselines)

        self.lbl_ratios = QLabel("0 Selected")
        form_layout.addRow("Ratios:", self.lbl_ratios)

        # Fix #9: quick-access column mapping buttons
        self.btn_map_iso = QPushButton("Map Isotope Cols…")
        self.btn_map_iso.setToolTip(
            "Choose & rename isotope columns from the header row"
        )
        self.btn_map_iso.setMinimumHeight(32)
        self.btn_map_iso.clicked.connect(self._quick_map_isotopes)
        form_layout.addRow("", self.btn_map_iso)

        self.btn_map_ratio = QPushButton("Map Ratio Cols…")
        self.btn_map_ratio.setToolTip(
            "Choose & rename ratio columns from the header row"
        )
        self.btn_map_ratio.setMinimumHeight(32)
        self.btn_map_ratio.clicked.connect(self._quick_map_ratios)
        form_layout.addRow("", self.btn_map_ratio)

        self.combo_system = QComboBox()
        self.combo_system.setEditable(True)
        self.combo_system.setMinimumHeight(32)
        self.combo_system.addItems([
            "Sr",
            "Li",
            "B",
            "Mg",
            "Pb",
            "Ni",
            "Zn",
            "Nd",
            "Hf",
            "U",
        ])
        form_layout.addRow("Isotope System:", self.combo_system)

        self.input_instrument = QLineEdit()
        self.input_instrument.setMinimumHeight(32)
        self.input_instrument.setPlaceholderText("e.g. Neptune, Nu Plasma")
        form_layout.addRow("Instrument:", self.input_instrument)

        self.include_audit_metadata = QCheckBox("Include full audit records")
        self.include_audit_metadata.setToolTip(
            "Off: keep sample details and channel units in a smaller file. "
            "On: also save source hashes, per-cell conversion evidence and global provenance. "
            "This output option applies to the next conversion and resets to off on restart."
        )
        form_layout.addRow("Output metadata:", self.include_audit_metadata)

        self.combo_name_parse = QComboBox()
        self.combo_name_parse.setEditable(True)
        self.combo_name_parse.setMinimumHeight(32)
        self.combo_name_parse.setMinimumWidth(160)
        self.combo_name_parse.setToolTip(
            "Regex to extract sample name from filename.\n"
            "The first capture group () is used as the sample name.\n"
            "Example: .*?(\\d{2,}_.+)$ extracts '001_Blank' from 'c:/path/001_Blank.exp'"
        )
        self.combo_name_parse.addItems([
            r".*?(\d{2,}_.+)$",
            r".*",
            r"^.*?_(.*)$",
            r"^([^_]+).*",
            r".*?_([^_]+)_.*"
        ])
        self.combo_name_parse.currentTextChanged.connect(self._update_preview)
        form_layout.addRow("Advanced pattern:", self.combo_name_parse)
        self._init_name_presets(form_layout)

        self.controls_grp.setLayout(form_layout)
        right_layout.addWidget(self.controls_grp)

        meta_grp = QGroupBox("Metadata Map & Guide")
        self.meta_layout = QVBoxLayout()
        self.lbl_checklist = QLabel("<b>Extraction Workflow Guide</b><br><i>(Load file and map cells)</i>")
        self.lbl_checklist.setWordWrap(True)
        self.meta_layout.addWidget(self.lbl_checklist)
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{theme.GRID_LINE};")
        self.meta_layout.addWidget(sep)
        self.lbl_meta_count = QLabel("No cells mapped")
        self.meta_layout.addWidget(self.lbl_meta_count)
        meta_grp.setLayout(self.meta_layout)
        right_layout.addWidget(meta_grp)

        preview_grp = QGroupBox("Live Extraction Preview  [F5]")
        pv_layout = QVBoxLayout(preview_grp)
        self.lbl_preview_status = QLabel("Ready for preview.")
        self.lbl_preview_status.setStyleSheet(
            f"font-weight:600; color:{theme.INK_DIM};"
        )
        pv_layout.addWidget(self.lbl_preview_status)
        self.text_preview = QTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.setObjectName("extractionConsole")
        self.text_preview.setMaximumHeight(200)
        pv_layout.addWidget(self.text_preview)
        right_layout.addWidget(preview_grp)

        right_layout.addStretch()

        # On short or high-DPI displays, never compress the form controls below
        # their usable height. The right panel scrolls vertically instead.
        self.right_scroll = QScrollArea()
        self.right_scroll.setObjectName("mapperRightScroll")
        self.right_scroll.setWidgetResizable(True)
        self.right_scroll.setFrameShape(QFrame.NoFrame)
        self.right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.right_scroll.setMinimumWidth(320)
        self.right_scroll.setWidget(right_widget)
        body_splitter.addWidget(self.right_scroll)

        body_splitter.setStretchFactor(0, 3)
        body_splitter.setStretchFactor(1, 1)
        body_splitter.setSizes([860, 380])

        main_layout.addWidget(body_splitter, 1)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel("")
        self._status_bar.addPermanentWidget(self._status_label)

        self._update_ui_labels()

    def _init_shortcuts(self):
        """Register keyboard shortcuts."""
        def _a(key, fn):
            act = QAction(self)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(fn)
            self.addAction(act)
        _a("Ctrl+O",       self.load_file_dialog)
        _a("Ctrl+Shift+O", self.load_folder)
        _a("Ctrl+S",       self.save_template)
        _a("Ctrl+L",       self.load_template)
        _a("Ctrl+Return",  self.run_batch_conversion)
        _a("F5",           self._update_preview)

    def _show_template_menu(self):
        """Drop-down with Load/Save/Built-ins/Recent."""
        menu = QMenu(self)
        menu.addAction("Load Template...  (Ctrl+L)").triggered.connect(self.load_template)
        menu.addAction("Save Template...  (Ctrl+S)").triggered.connect(self.save_template)
        menu.addSeparator()
        builtin_menu = menu.addMenu("Built-in Templates")
        for tpl_path in sorted(_TEMPLATES_DIR.glob("*.json")):
            name = tpl_path.stem.replace("_", " ").title()
            builtin_menu.addAction(name).triggered.connect(
                lambda _checked, p=tpl_path: self._load_template_from_path(str(p))
            )
        recents = self._settings.value(_SETTINGS_KEY_RECENT, []) or []
        valid_recents = [r for r in recents if Path(r).exists()]
        if valid_recents:
            menu.addSeparator()
            rm = menu.addMenu("Recent Templates")
            for rp in valid_recents:
                rm.addAction(Path(rp).name).triggered.connect(
                    lambda _checked, p=rp: self._load_template_from_path(p)
                )
        menu.exec_(self.btn_template_menu.mapToGlobal(
            self.btn_template_menu.rect().bottomLeft()
        ))

    def _update_status_bar(self):
        n_files = self.file_list_widget.count()
        n_iso   = len(self.mapping.get("isotope_cols", {}))
        n_ratio = len(self.mapping.get("ratio_cols", {}))
        n_meta  = len(self.mapping.get("metadata_cells", {}))
        time_s  = self.mapping.get("time_col") or ""
        sys_s   = self.combo_system.currentText() if hasattr(self, "combo_system") else "—"
        parts = [
            f"Files: {n_files}",
            f"Isotopes: {n_iso}",
            f"Ratios: {n_ratio}",
            f"Metadata: {n_meta}",
            f"Time: {'OK' if time_s else 'not set'}",
            f"System: {sys_s}",
        ]
        self._status_label.setText("  |  ".join(parts))

    def _normalize_mapping_schema(self) -> None:
        """Normalize legacy list-style template mappings to dict form."""
        normalized = normalize_template_schema(
            {
                "isotope_cols": self.mapping.get("isotope_cols", {}),
                "ratio_cols": self.mapping.get("ratio_cols", {}),
            }
        )
        self.mapping["isotope_cols"] = normalized["isotope_cols"]
        self.mapping["ratio_cols"] = normalized["ratio_cols"]

    def _clear_all_mappings(self):
        self.mapping = {
            "time_col": "",
            "cycle_col": "",
            "isotope_cols": {},
            "ratio_cols": {},
            "metadata_cells": {}
        }
        self.input_instrument.clear()
        self._update_ui_labels()
        if self.current_df is not None:
            self._refresh_highlights()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.load_file(files[0])

    def load_file_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Data Files (*.csv *.txt *.exp *.xls *.xlsx)")
        if fname:
            self.load_file(fname)

    def load_file(self, filepath):
        self._history_boundary()
        self.file_list_widget.add_file(filepath)
        self.file_list_widget.mark_active(filepath)   # Fix #7: highlight active file
        self.current_filepath = filepath
        self.lbl_file.setText(os.path.basename(filepath))
        # Hide onboarding overlay once a file is loaded
        self.onboarding.hide()
        try:
            self.current_df = load_file_as_df(filepath)
            self._render_grid()
            self._update_ui_labels()
            self._history_boundary()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")


    def _render_grid(self):
        if self.current_df is None:
            return

        df = self.current_df.fillna("")
        rows, cols = df.shape

        # --- Fix #3: strip trailing columns that are entirely empty ---
        # Check every previewed row: instrument exports often have a long, narrow
        # metadata preamble (Nu Plasma II: 77 lines) above a much wider table,
        # and judging by the first 50 rows hid every signal column.
        last_col = cols - 1
        while last_col > 0:
            col_vals = [str(df.iloc[r, last_col]).strip() for r in range(min(rows, 500))]
            if all(v == "" or v.lower() == "nan" for v in col_vals):
                last_col -= 1
            else:
                break
        cols = last_col + 1

        preview_rows = min(rows, 500)

        self.table.setRowCount(preview_rows)
        self.table.setColumnCount(cols)
        self.table.setHorizontalHeaderLabels(
            [self.get_col_name(i) if self._is_positional() else str(i) for i in range(cols)]
        )

        # Row numbers count from 0, like the column numbers, the [col N] labels,
        # the context-menu actions and the template's row indices. Qt's default
        # labels count from 1, which put every row one above its menu number.
        self.table.setVerticalHeaderLabels([str(r) for r in range(preview_rows)])
        self.table.verticalHeader().setVisible(True)
        self.table.verticalHeader().setDefaultSectionSize(22)

        for r in range(preview_rows):
            for c in range(cols):
                raw = df.iloc[r, c]
                # Fix #2: smart number formatting
                # - empty stays empty
                # - whole numbers show as integers (no decimals)
                # - other floats show with 5 sig decimal places
                try:
                    f = float(raw)
                    if raw == "":
                        val = ""
                    elif f == int(f) and abs(f) < 1e10:
                        val = str(int(f))   # whole number → no decimals
                    else:
                        val = f"{f:.5f}"    # float → fixed 5dp
                    # Fix #4: numbers right-aligned
                    align = Qt.AlignRight | Qt.AlignVCenter
                except (ValueError, TypeError):
                    val = str(raw)
                    # Fix #4: text left-aligned
                    align = Qt.AlignLeft | Qt.AlignVCenter

                item = QTableWidgetItem(val)
                item.setTextAlignment(align)
                self.table.setItem(r, c, item)

        self._refresh_highlights()

        # Fix #5 + #10: auto-fit AFTER highlights applied so widths include styled content
        self.table.resizeColumnsToContents()
        for c in range(cols):
            if self.table.columnWidth(c) > 200:
                self.table.setColumnWidth(c, 200)


    def show_context_menu(self, position: QPoint):
        self._normalize_mapping_schema()
        item = self.table.itemAt(position)
        if not item:
            return
            
        row, col = item.row(), item.column()
        
        # Handle Selection
        selected_items = self.table.selectedItems()
        selected_cols = set()
        min_row = row
        max_row = row
        
        if selected_items:
            # Check if the right-clicked cell itself is in the selection
            # If the user right clicks outside of their selection, Qt doesn't necessarily clear it,
            # so we should focus locally on the clicked cell instead of the stale selection block.
            clicked_in_selection = any(i.row() == row and i.column() == col for i in selected_items)
            
            if clicked_in_selection:
                min_row = min(item.row() for item in selected_items)
                max_row = max(item.row() for item in selected_items)
                for item in selected_items:
                    selected_cols.add(item.column())
            else:
                selected_cols.add(col)
        else:
            selected_cols.add(col)
        
        menu = QMenu()
        
        cols_list = sorted(list(selected_cols))
        dynamic_column_mapping = self._selection_is_header_row(min_row, max_row)
        
        # Metadata Actions (Single Cell focused)
        meta_menu = menu.addMenu("Map Metadata Cell")
        action_meta_name = meta_menu.addAction("Sample Name")
        action_meta_run = meta_menu.addAction("Run Number")
        action_meta_date = meta_menu.addAction("Date")
        action_meta_time = meta_menu.addAction("Time (Metadata)")
        action_meta_type = meta_menu.addAction("Sample Type")
        action_meta_b1 = meta_menu.addAction("Blank 1")
        action_meta_b2 = meta_menu.addAction("Blank 2")
        meta_menu.addSeparator()
        action_meta_custom = meta_menu.addAction("Custom Field...")
        menu.addSeparator()
        
        # Data actions. A one-row header selection maps dynamically to each
        # file's footer marker; a multi-row data selection creates a fixed
        # row range shared by the batch.
        column_word = "Column" if len(cols_list) == 1 else "Columns"
        iso_label = f"Map {len(cols_list)} Selected {column_word} to Isotopes"
        ratio_label = f"Map {len(cols_list)} Selected {column_word} to Ratios"
        action_map_iso = menu.addAction(iso_label)
        action_map_ratio = menu.addAction(ratio_label)
        # With subtraction on, the selected column can be made the baseline of
        # an isotope already mapped, straight from here (one column per isotope).
        baseline_actions = {}
        if self._baseline_enabled():
            baseline_menu = menu.addMenu(
                f"Map {len(cols_list)} Selected {column_word} as Baseline of")
            self._fill_baseline_menu(baseline_menu, cols_list, baseline_actions)
        action_set_data_start = menu.addAction(f"Row {min_row}: Set as Data Start Row (Header {max(0, min_row-1)})")
        action_set_footer = menu.addAction(f"Row {max_row+1}: Set as Footer Start Row (End {max_row})")
        if self._is_positional():
            for row_action in (action_set_data_start, action_set_footer):
                row_action.setEnabled(False)
                row_action.setText(row_action.text() + " — " + self._row_action_note())
            action_leave_positional = menu.addAction("Leave Headerless Layout (clears positional mappings)")
        else:
            action_leave_positional = None
        action_positional = menu.addAction(f"Row {min_row}: No Header – Data Starts Here (positional)")
        action_positional.setToolTip(
            "For exports without a column-header row (classic Nu Plasma): columns are\n"
            "named 'col N', the data is anchored on the text line above this row and\n"
            "every data row must have the same width."
        )
        
        data_menu = menu.addMenu("More Column Actions")
        action_map_time_col = data_menu.addAction("Map as Time Column")
        action_map_time_col.setToolTip(
            "Sets this data column as the per-cycle elapsed time (seconds/ms).\n"
            "Different from 'Analysis Time' metadata — this is a column in the data table."
        )
        action_map_cycle_col = data_menu.addAction("Map as Cycle Column")
        action_map_cycle_col.setToolTip(
            "Sets this data column as the cycle number column (1, 2, 3...).\n"
            "Required for extracting cycle-resolved data."
        )
        data_menu.addSeparator()
        mapped_targets = sorted(set(self.mapping["isotope_cols"].values()))
        action_map_baseline = data_menu.addAction("Map as Baseline of…")
        action_map_baseline.setEnabled(len(cols_list) == 1 and bool(mapped_targets))
        action_map_baseline.setToolTip(
            "This column holds the zero/baseline reading of one mapped isotope;\n"
            "it is subtracted from that isotope on every cycle."
        )
        action_suggest_baseline = data_menu.addAction("Suggest Baseline Pairs…")
        action_suggest_baseline.setEnabled(bool(mapped_targets))
        data_menu.addSeparator()
        action_dynamic_footer = data_menu.addAction("Use Dynamic Footer (***)")
        action_dynamic_footer.setEnabled(not self._is_positional())
        action_dynamic_footer.setToolTip(
            "Stop each file at its own first *** row. Use this when blanks, "
            "standards, and samples have different cycle counts."
        )
        
        menu.addSeparator()
        action_clear = menu.addAction("Clear Mappings in Selection")
        
        action = menu.exec_(self.table.viewport().mapToGlobal(position))
        
        
        # Metadata
        if action == action_meta_name: self._set_meta(row, col, "Sample Name")
        elif action == action_meta_run: self._set_meta(row, col, "Run Number")
        elif action == action_meta_date: self._set_meta(row, col, "Date")
        elif action == action_meta_time: self._set_meta(row, col, "Time")
        elif action == action_meta_type: self._set_meta(row, col, "Sample Type")
        elif action == action_meta_b1: self._set_meta(row, col, "Blank 1")
        elif action == action_meta_b2: self._set_meta(row, col, "Blank 2")
        elif action == action_meta_custom:
            field, ok = QInputDialog.getText(self, "Custom Metadata", "Enter Field Name:")
            if ok and field: self._set_meta(row, col, field)

        # Data
        elif action == action_set_data_start:
            self.set_header_row(max(0, min_row - 1))

        elif action_leave_positional is not None and action == action_leave_positional:
            self._leave_positional()
            self.set_header_row(0)
            self._render_grid()
            self._update_ui_labels()

        elif action == action_positional:
            self._set_positional_start(min_row)

        elif action in baseline_actions:
            self._assign_baseline_from_selection(cols_list[0], min_row, baseline_actions[action])

        elif action == action_map_baseline:
            self._map_baseline(cols_list[0], mapped_targets)

        elif action == action_suggest_baseline:
            self._suggest_baselines()
            
        elif action == action_set_footer:
            self.set_footer_row(max_row + 1)

        elif action == action_map_iso:
            self._apply_selection_bounds_for_column_mapping(
                min_row, max_row, dynamic=dynamic_column_mapping
            )
            self._open_isotope_dialog(cols_list, min_row)

        elif action == action_map_ratio:
            self._apply_selection_bounds_for_column_mapping(
                min_row, max_row, dynamic=dynamic_column_mapping
            )
            self._open_ratio_dialog(cols_list, min_row)
            
        elif action == action_map_time_col:
            self.mapping["time_col"] = self.get_col_name(cols_list[0])
            self._update_ui_labels()
            
        elif action == action_map_cycle_col:
            self.mapping["cycle_col"] = self.get_col_name(cols_list[0])
            self._update_ui_labels()

        elif action == action_dynamic_footer:
            self.set_footer_row(None)

        elif action == action_clear:
            for c in cols_list:
                col_name = self.get_col_name(c)
                
                if col_name in self.mapping["isotope_cols"]:
                    del self.mapping["isotope_cols"][col_name]
                else:
                    keys_to_delete = [
                        k for k, v in self.mapping["isotope_cols"].items()
                        if k == col_name or v == col_name
                    ]
                    for k in keys_to_delete:
                        del self.mapping["isotope_cols"][k]
                        
                # Clear Ratios
                if isinstance(self.mapping.get("ratio_cols"), dict):
                    if col_name in self.mapping["ratio_cols"]:
                        del self.mapping["ratio_cols"][col_name]
                        
                # Clear Time/Cycle
                if self.mapping.get("time_col") == col_name: self.mapping["time_col"] = ""
                if self.mapping.get("cycle_col") == col_name: self.mapping["cycle_col"] = ""

                # Clear baselines held in this column
                baselines = self.mapping.get("baseline_cols") or {}
                for target in [t for t, src in baselines.items() if src == col_name]:
                    del baselines[target]

            self._drop_orphan_baselines()
            # Clear Metadata in the selected block
            for r in range(min_row, max_row + 1):
                for c in cols_list:
                    self.mapping["metadata_cells"].pop((r, c), None)

            self._update_ui_labels()
            self._refresh_highlights()

    # --- Baselines and headerless (positional) layouts -----------------------

    def _set_isotope_cols(self, new_cols):
        """Replace the isotope mapping; each baseline stays with its signal column.

        ``baseline_cols`` is keyed by isotope name, but a baseline belongs to a
        signal *column*. Renaming that column's isotope (or swapping two names)
        must carry the baseline along, or it would be subtracted from the
        wrong signal. Baselines whose signal column is no longer mapped go.
        """
        new_cols = dict(new_cols)
        old_cols = dict(self.mapping.get("isotope_cols", {}))
        baselines = self.mapping.get("baseline_cols") or {}
        if baselines:
            source_of = {target: source for source, target in old_cols.items()}
            rekeyed = {}
            for target, baseline_source in baselines.items():
                new_target = new_cols.get(source_of.get(target))
                if new_target:
                    rekeyed[new_target] = baseline_source
            if rekeyed:
                self.mapping["baseline_cols"] = rekeyed
            else:
                self.mapping.pop("baseline_cols", None)
        self.mapping["isotope_cols"] = new_cols

    def _row_action_note(self):
        return ("not used by a headerless layout: each file's data runs from its "
                "start-marker line to its footer line")

    def _baseline_enabled(self):
        """Mirror mapper_logic.baseline_subtraction_enabled for the live mapping."""
        if "baseline_enabled" in self.mapping:
            return bool(self.mapping["baseline_enabled"])
        return bool(self.mapping.get("baseline_cols"))

    def _set_baseline_enabled(self, checked):
        # Written explicitly either way, so a saved template says what it means.
        self.mapping["baseline_enabled"] = bool(checked)
        self._update_ui_labels()
        self._refresh_highlights()

    def _drop_orphan_baselines(self):
        """Remove baselines whose isotope is no longer mapped."""
        baselines = self.mapping.get("baseline_cols")
        if not baselines:
            self.mapping.pop("baseline_cols", None)
            return
        targets = set(self.mapping["isotope_cols"].values())
        for target in [t for t in baselines if t not in targets]:
            del baselines[target]
        if not baselines:
            self.mapping.pop("baseline_cols", None)

    def _fill_baseline_menu(self, submenu, cols_list, actions):
        """One entry per mapped isotope, showing the baseline it has now.

        *actions* receives {QAction: isotope}. Entries are disabled (with the
        reason as the only entry) when the choice cannot be made from here.
        """
        targets = sorted(set(self.mapping["isotope_cols"].values()))
        if not targets:
            submenu.addAction("(map the signal columns to isotopes first)").setEnabled(False)
            return
        if len(cols_list) != 1:
            submenu.addAction("(select cells in one column: one baseline per isotope)").setEnabled(False)
            return
        source = self.get_col_name(cols_list[0])
        if source and source in self.mapping["isotope_cols"]:
            submenu.addAction(f"('{source}' is mapped as an isotope signal)").setEnabled(False)
            return
        current = self.mapping.get("baseline_cols") or {}
        for target in targets:
            existing = current.get(target)
            if existing is None:
                text = target
            elif source and existing == source:
                text = f"{target}   (this column)"
            else:
                text = f"{target}   (replaces {existing})"
            action = submenu.addAction(text)
            if source and existing == source:
                action.setCheckable(True)
                action.setChecked(True)
            actions[action] = target

    def _assign_baseline_from_selection(self, col_idx, top_row, target):
        # An unnamed column under no real header gets its layout first, exactly
        # as when the same cells are mapped to isotopes.
        if not self._name_selected_columns([col_idx], top_row):
            return
        source = self.get_col_name(col_idx)
        if not source:
            QMessageBox.warning(self, "Map Baseline",
                                "This column has no name in the current layout; set the header "
                                "row or the headerless layout first.")
            return
        self._assign_baseline(target, source)

    def _map_baseline(self, col_idx, targets):
        target, ok = QInputDialog.getItem(
            self, "Map Baseline",
            f"Column '{self.get_col_name(col_idx)}' is the baseline of:", targets, 0, False)
        if not ok or not target:
            return
        self._assign_baseline(target, self.get_col_name(col_idx))

    def _assign_baseline(self, target, source):
        if source in self.mapping["isotope_cols"]:
            QMessageBox.warning(self, "Map Baseline",
                                "That column is mapped as an isotope signal; clear it first.")
            return
        if not self.mapping.get("baseline_cols"):
            # The first assignment switches subtraction on, as the box says.
            self.mapping["baseline_enabled"] = True
        self.mapping.setdefault("baseline_cols", {})[target] = source
        self._update_ui_labels()
        self._refresh_highlights()

    def _suggest_baselines(self):
        """Propose baseline pairs, show them, and apply only on confirmation."""
        if self.current_df is None:
            return
        columns = [self.get_col_name(c) for c in range(len(self.current_df.columns))]
        existing = dict(self.mapping.get("baseline_cols") or {})
        pairs, notes = suggest_baseline_cols(columns, self.mapping["isotope_cols"], existing)
        new = {t: s for t, s in pairs.items() if t not in existing}
        if not new:
            QMessageBox.information(self, "Suggest Baseline Pairs",
                                    "No new pairs found.\n\n" + "\n".join(notes))
            return
        text = "\n".join(f"{t}  ←  {s}" for t, s in sorted(new.items()))
        if notes:
            text += "\n\nNot paired:\n" + "\n".join(notes)
        answer = QMessageBox.question(self, "Suggest Baseline Pairs",
                                      "Subtract these baselines?\n\n" + text)
        if answer == QMessageBox.Yes:
            self.mapping["baseline_cols"] = pairs
            self._update_ui_labels()
            self._refresh_highlights()

    def _set_positional_start(self, first_data_row):
        """Switch to a headerless layout whose data starts at *first_data_row*."""
        if self.current_df is None:
            return
        try:
            layout = derive_positional_layout(self.current_df, first_data_row)
        except ValueError as exc:
            QMessageBox.warning(self, "No Header – Data Starts Here", str(exc))
            return
        # Header-named mappings mean nothing without a header; drop them openly.
        named = [k for k in self.mapping["isotope_cols"] if not k.startswith("col ")]
        dropped = len(named) + len([v for v in (self.mapping.get("baseline_cols") or {}).values()
                                    if not v.startswith("col ")])
        for key in named:
            del self.mapping["isotope_cols"][key]
        for key in ("time_col", "cycle_col"):
            if self.mapping.get(key) and not str(self.mapping[key]).startswith("col "):
                self.mapping[key] = ""
        baselines = self.mapping.get("baseline_cols") or {}
        for target in [t for t, v in baselines.items() if not v.startswith("col ")]:
            del baselines[target]
        self._drop_orphan_baselines()
        self.mapping.update(layout)
        self.footer_row_idx = None
        self.set_header_row(0)
        self._render_grid()
        self._update_ui_labels()
        if dropped:
            QMessageBox.information(
                self, "No Header – Data Starts Here",
                f"{dropped} header-named column mapping(s) were cleared; map the "
                "columns again by position.")

    def _leave_positional(self):
        if not self._is_positional():
            return
        for key in ("layout", "data_start_marker", "data_row_width", "footer_marker"):
            self.mapping.pop(key, None)
        for key in [k for k in self.mapping["isotope_cols"] if k.startswith("col ")]:
            del self.mapping["isotope_cols"][key]
        baselines = self.mapping.get("baseline_cols") or {}
        for target in [t for t, v in baselines.items() if v.startswith("col ")]:
            del baselines[target]
        self._drop_orphan_baselines()
        for key in ("time_col", "cycle_col"):
            if str(self.mapping.get(key) or "").startswith("col "):
                self.mapping[key] = ""

    def _is_positional(self):
        return self.mapping.get("layout") == "positional"

    @staticmethod
    def _is_blank(value):
        return str(value).strip().lower() in ("", "nan", "none")

    def _first_data_row_of(self, row):
        """Return the first row of the numeric block containing *row*, or None.

        Data rows start with a number in the first column (a cycle number or a
        zero reading); the block begins below the first row that does not.
        """
        if self.current_df is None or not 0 <= row < len(self.current_df):
            return None
        first = self.current_df.iloc[:, 0].tolist()

        def numeric(value):
            try:
                float(str(value).strip())
                return not self._is_blank(value)
            except ValueError:
                return False

        if not numeric(first[row]):
            return None
        while row > 0 and numeric(first[row - 1]):
            row -= 1
        return row

    def _header_cell_blank(self, col_idx):
        """True when the current header row has no name for this column."""
        row = self.header_row_idx
        if self.current_df is None or not 0 <= row < len(self.current_df):
            return True
        return self._is_blank(self.current_df.iat[row, col_idx])

    def _name_selected_columns(self, cols_list, top_row):
        """Set the layout the selected data cells imply, without asking.

        Only when a selected column has no name in the current header row and
        that header row is not the row directly above the selected data block:
        - the row above the block is a header (it names several columns):
          make it the header row;
        - it is not (classic Nu: one text line, 'Sample Name is ...'): switch to
          the headerless layout anchored on that line.
        Otherwise a blank header cell is just an unnamed column ('col N'),
        named in the dialog. Returns False when the dialog should not open.
        """
        if self.current_df is None or self._is_positional():
            return True
        if not any(self._header_cell_blank(c) for c in cols_list):
            return True
        first = self._first_data_row_of(top_row if top_row is not None else -1)
        if first is None or first == 0 or first - 1 == self.header_row_idx:
            return True
        above = self.current_df.iloc[first - 1].tolist()
        if sum(not self._is_blank(v) for v in above) >= 2:
            self.set_header_row(first - 1)
            self._update_ui_labels()
            return True
        try:
            derive_positional_layout(self.current_df, first)
        except ValueError as exc:
            QMessageBox.warning(self, "Columns without a header",
                                "There is no header above the selected cells, and the "
                                "file cannot be read without one:\n\n" + str(exc))
            return False
        self._set_positional_start(first)
        return self._is_positional()

    def _positional_rows(self):
        """Return (data-start marker row, footer row) of the loaded file.

        Mirrors mapper_logic._positional_data_block for display only; either
        may be None when the file does not match the template, in which case
        preview/validation report the real error.
        """
        if self.current_df is None or self.current_df.empty:
            return None, None
        first = self.current_df.iloc[:, 0].tolist()
        marker_row = None
        try:
            pattern = re.compile(str(self.mapping.get("data_start_marker") or ""))
            hits = [i for i, v in enumerate(first) if isinstance(v, str) and pattern.search(v)]
            marker_row = hits[0] if len(hits) == 1 else None
        except re.error:
            pass
        footer_row = None
        footer = self.mapping.get("footer_marker")
        if marker_row is not None and footer:
            footer_row = next((i for i in range(marker_row + 1, len(first))
                               if str(first[i]).strip().startswith(footer)), None)
        return marker_row, footer_row

    def get_col_name(self, col_idx):
        # Positional templates have no header row: columns are named by index.
        if self._is_positional():
            return f"col {col_idx}"
        # If header row set, use value from that row. Duplicate labels are
        # decorated with their visible grid column index so mappings can target
        # a specific repeated cup label.
        if self.current_df is not None and 0 <= self.header_row_idx < len(self.current_df):
            header_values = self.current_df.iloc[self.header_row_idx].tolist()
            labels = disambiguate_header_values(header_values)
            if 0 <= col_idx < len(labels):
                # A blank header cell is an unnamed column ('col N') only under a
                # real header row. Before the header is set, row 0 is usually a
                # preamble line ('Version 2', a report title): its blanks are not
                # columns of anything, so they stay unnamed and hidden.
                if self._is_blank(header_values[col_idx]) and \
                        sum(not self._is_blank(v) for v in header_values) < 2:
                    return ""
                return labels[col_idx]
        return str(col_idx)

    def _header_row_is_real(self):
        """True when the current header row names at least two columns."""
        if self.current_df is None or not 0 <= self.header_row_idx < len(self.current_df):
            return False
        values = self.current_df.iloc[self.header_row_idx].tolist()
        return sum(not self._is_blank(v) for v in values) >= 2

    def set_header_row(self, row_idx):
        self.header_row_idx = row_idx
        # Update table headers visually
        cols = self.table.columnCount()
        labels = []
        for c in range(cols):
            labels.append(self.get_col_name(c))
        self.table.setHorizontalHeaderLabels(labels)
        self.lbl_header_row.setText(str(row_idx))
        self._refresh_highlights()

    def set_footer_row(self, row_idx):
        self.footer_row_idx = row_idx
        self.lbl_footer_row.setText(
            str(row_idx) if row_idx is not None else "(End of File / ***)"
        )
        self._refresh_highlights()

    def _selection_is_header_row(self, min_row, max_row):
        """Return whether a one-row selection represents column headers."""
        if min_row != max_row:
            return False
        if min_row == self.header_row_idx:
            return True
        if self.current_df is None or not 0 <= min_row < len(self.current_df):
            return False

        row_tokens = {
            str(value).strip().lower()
            for value in self.current_df.iloc[min_row].tolist()
        }
        # Neptune/Nu measurement tables normally identify their header row
        # with Cycle and/or Time. This also supports numeric mass headers such
        # as 6.01 and 7.01, which cannot be recognized from the selected cell
        # text alone.
        return "cycle" in row_tokens or "time" in row_tokens

    def _apply_selection_bounds_for_column_mapping(
        self, min_row, max_row, *, dynamic=None
    ):
        """Apply automatic dynamic-header or fixed-data-block row bounds."""
        # A selection identifies columns. The exported rows come from the header
        # row and footer (or, headerless, the data-start marker and footer), and
        # are changed only by the explicit Data Start / Footer row actions; a
        # selection of a few data cells must never shorten the exported block.
        if self._is_positional():
            return
        if dynamic is None:
            dynamic = self._selection_is_header_row(min_row, max_row)
        if dynamic:
            # Selecting the header row itself names the header; that is layout,
            # not a row range.
            self.set_header_row(min_row)
            self.set_footer_row(None)

    def _available_column_names(self):
        """Return the columns that can be mapped.

        Named header columns, plus unnamed ones ('col N') that hold data below
        the header; the user names those in the mapping dialog.
        """
        names = []
        df = self.current_df
        for column in range(self.table.columnCount()):
            name = self.get_col_name(column).strip()
            if not name or name.lower() == "nan":
                continue
            if (not self._is_positional() and df is not None
                    and column < len(df.columns) and self._header_cell_blank(column)):
                below = df.iloc[self.header_row_idx + 1:, column].tolist()
                if all(self._is_blank(v) for v in below):
                    continue
            names.append(name)
        return names

    # (Keep set_header_row as is, but ensure footer_row_idx is init in __init__)

    def _update_ui_labels(self):
        self._record_mapping()
        self._normalize_mapping_schema()
        # Header/Footer
        self.lbl_header_row.setText(str(self.header_row_idx))
        
        fr = getattr(self, "footer_row_idx", None)
        self.lbl_footer_row.setText(str(fr) if fr is not None else "(End of File)")
        if self._is_positional():
            marker_row, footer_row = self._positional_rows()
            self.lbl_header_row.setText(
                "none (positional)" if marker_row is None
                else f"none (positional) - data after row {marker_row}")
            self.lbl_footer_row.setText(
                f"{footer_row} ({self.mapping.get('footer_marker')})" if footer_row is not None
                else f"not found: {self.mapping.get('footer_marker')!r}")
        
        # Time Column: clarify 'Not Set' vs truly absent instruments
        time_col = self.mapping.get("time_col") or ""
        self.lbl_time.setText(time_col if time_col else "(Not Set — optional)")
        self.lbl_cycle.setText(self.mapping["cycle_col"] or "(Not Set)")
        
        # Isotopes
        iso_count = len(self.mapping["isotope_cols"])
        base_count = len([t for t in (self.mapping.get("baseline_cols") or {})
                          if t in self.mapping["isotope_cols"].values()])
        enabled = self._baseline_enabled()
        suffix = ""
        if base_count:
            suffix = f" · {base_count} with baseline" + ("" if enabled else " (subtraction off)")
        self.lbl_isotopes.setText(f"{iso_count} Selected" + suffix)
        self.chk_subtract_baselines.blockSignals(True)
        self.chk_subtract_baselines.setChecked(enabled)
        # Usable whenever there is something to switch: assignments exist, or
        # subtraction is on without any (so it can always be switched off).
        # Always clickable. Switched on before any baseline is assigned, it says
        # what is still needed; extraction refuses to run in that state rather
        # than export unsubtracted values under a 'subtracted' switch.
        self.chk_subtract_baselines.setEnabled(True)
        waiting = enabled and not self.mapping.get("baseline_cols")
        self.chk_subtract_baselines.setText(
            "Subtract baselines (now assign a baseline column to each isotope)"
            if waiting else "Subtract baselines")
        self.chk_subtract_baselines.setToolTip(
            "On: every mapped isotope minus its assigned baseline column, cycle by cycle.\n"
            "Off: values are exported as read; baseline assignments are kept in the template.\n"
            "The output records whether subtraction was applied.\n\n"
            "Assign a baseline: select cells of a zero/baseline column, right-click ->\n"
            "Map 1 Selected Column as Baseline of -> the isotope (shown while this is on),\n"
            "or More Column Actions -> Map as Baseline of... / Suggest Baseline Pairs....\n"
            "The first assignment switches subtraction on.")
        if hasattr(self, "_mapping_actions"):
            self._mapping_selection_changed()
        self.chk_subtract_baselines.blockSignals(False)
        
        # Ratios
        ratio_count = len(self.mapping.get("ratio_cols", {}))
        self.lbl_ratios.setText(f"{ratio_count} Selected")
        
        # Metadata
        # Keep checklist at top
        while self.meta_layout.count() > 2: # Keep the checklist and line
             child = self.meta_layout.takeAt(2)
             if child.widget():
                 child.widget().deleteLater()
                 
        meta_str_keys = {f"{r},{c}": v for (r, c), v in self.mapping["metadata_cells"].items()}
        template_preview = {
            **self._source_template_options(),
            "metadata_cells": meta_str_keys,
            "name_pattern": self.combo_name_parse.currentText()
        }
        
        cl_html = "<b>Extraction Workflow Guide</b><br>"

        def _rich(value):
            return escape(str(value), quote=False)

        def _status(label, color):
            return f"<span style='color:{color}; font-weight:bold;'>{label}</span>"

        c_yes = _status("OK", "#008000")
        c_no = _status("Missing", "#aa0000")
        c_opt = _status("Optional", "#666")

        # Check mapping state directly for missing items
        mapped_fields = list(self.mapping.get("metadata_cells", {}).values())
        has_run = bool(self.mapping.get("run_number_pattern")) or "Run Number" in mapped_fields or "run" in [f.lower() for f in mapped_fields]
        has_type = "Sample Type" in mapped_fields or "Type" in mapped_fields
        has_date = "Date" in mapped_fields or "date" in [f.lower() for f in mapped_fields]

        if self.current_df is not None and self.current_filepath:
            try:
                sample_name, ext_meta = extract_metadata_from_df(self.current_df, self.current_filepath, template_preview)
                
                cl_html += f"{c_yes} <b>Sample Name:</b> <span style='color:#0055aa;'>{_rich(sample_name)}</span><br>"
                if "Run number" in ext_meta:
                     cl_html += f"{c_yes} <b>Run Number:</b> <span style='color:#0055aa;'>{_rich(ext_meta['Run number'])}</span><br>"
                else:
                     cl_html += f"{c_no} <b>Run Number:</b> <span style='color:gray;'>(Map cell)</span><br>"
                
                if "Type" in ext_meta:
                     cl_html += f"{c_yes} <b>Type:</b> <span style='color:#0055aa;'>{_rich(ext_meta['Type'])}</span><br>"
                else:
                     cl_html += f"{c_no} <b>Type:</b> <span style='color:gray;'>(Map cell)</span><br>"

                if "Date" in ext_meta:
                     cl_html += f"{c_yes} <b>Date:</b> <span style='color:#0055aa;'>{_rich(ext_meta['Date'])}</span><br>"
                else:
                     cl_html += f"{c_no} <b>Date:</b> <span style='color:gray;'>(Map cell)</span><br>"

                if "Time" in ext_meta:
                     cl_html += f"{c_yes} <b>Analysis Time:</b> <span style='color:#0055aa;'>{_rich(ext_meta['Time'])}</span><br>"
                else:
                     cl_html += f"{c_opt} <b>Analysis Time:</b> <span style='color:gray;'>(optional - map cell)</span><br>"

                # Cycle column
                cyc = self.mapping.get("cycle_col", "")
                if cyc:
                    cl_html += f"{c_yes} <b>Cycle Col:</b> <span style='color:#0055aa;'>{_rich(cyc)}</span><br>"
                else:
                    cl_html += f"{c_no} <b>Cycle Col:</b> <span style='color:gray;'>(right-click data column)</span><br>"

                # Show Isotopes Selected
                iso_cols = self.mapping.get("isotope_cols", {})
                if isinstance(iso_cols, dict):
                    iso_list = list(iso_cols.values())
                else:
                    iso_list = iso_cols

                if iso_list:
                    iso_text = ", ".join(_rich(iso) for iso in iso_list)
                    cl_html += f"{c_yes} <b>Isotopes ({len(iso_list)}):</b> <span style='color:#0055aa;'>{iso_text}</span><br>"
                else:
                    cl_html += f"{c_no} <b>Isotopes:</b> <span style='color:gray;'>(Map to Isotopes)</span><br>"
                    
            except Exception as e:
                cl_html += f"<i><span style='color:red;'>Error predicting metadata. {_rich(e)}</span></i>"
        else:
            # Generic empty state
            cl_html += f"{c_yes} <b>Sample Name:</b> <span style='color:gray;'>(Parsed from filename)</span><br>"
            
            if has_run:
                cl_html += f"{c_yes} <b>Run Number:</b> <span style='color:gray;'>(Mapped)</span><br>"
            else:
                cl_html += f"{c_no} <b>Run Number:</b> <span style='color:gray;'>(Map cell)</span><br>"
                
            if has_type:
                cl_html += f"{c_yes} <b>Type:</b> <span style='color:gray;'>(Mapped)</span><br>"
            else:
                cl_html += f"{c_no} <b>Type:</b> <span style='color:gray;'>(Map cell)</span><br>"

            if has_date:
                cl_html += f"{c_yes} <b>Date:</b> <span style='color:gray;'>(Mapped)</span><br>"
            else:
                cl_html += f"{c_no} <b>Date:</b> <span style='color:gray;'>(Map cell)</span><br>"

            has_time_meta = "Time" in mapped_fields or "time" in [f.lower() for f in mapped_fields]
            if has_time_meta:
                cl_html += f"{c_yes} <b>Analysis Time:</b> <span style='color:gray;'>(Mapped)</span><br>"
            else:
                cl_html += f"{c_opt} <b>Analysis Time:</b> <span style='color:gray;'>(optional)</span><br>"

            cyc = self.mapping.get("cycle_col", "")
            if cyc:
                cl_html += f"{c_yes} <b>Cycle Col:</b> <span style='color:gray;'>{_rich(cyc)}</span><br>"
            else:
                cl_html += f"{c_no} <b>Cycle Col:</b> <span style='color:gray;'>(right-click data column)</span><br>"

            # Show Isotopes Selected
            iso_cols = self.mapping.get("isotope_cols", {})
            if isinstance(iso_cols, dict):
                iso_list = list(iso_cols.values())
            else:
                iso_list = iso_cols
                
            if iso_list:
                iso_text = ", ".join(_rich(iso) for iso in iso_list)
                cl_html += f"{c_yes} <b>Isotopes ({len(iso_list)}):</b> <span style='color:#0055aa;'>{iso_text}</span><br>"
            else:
                cl_html += f"{c_no} <b>Isotopes:</b> <span style='color:gray;'>(Map to Isotopes)</span><br>"
            
        self.lbl_checklist.setTextFormat(Qt.RichText)
        self.lbl_checklist.setText(cl_html)
                
        if not self.mapping["metadata_cells"]:
            self.lbl_meta_count = QLabel("No cells mapped")
            self.meta_layout.addWidget(self.lbl_meta_count)
        else:
            for (r, c), field in self.mapping["metadata_cells"].items():
                lbl = QLabel(f"Row {r}, Col {c} -> {field}")
                if self.current_df is not None and r < len(self.current_df) and c < len(self.current_df.columns):
                    val = str(self.current_df.iloc[r, c]).strip()
                    lbl.setToolTip(str(val))
                self.meta_layout.addWidget(lbl)
        
        self._refresh_highlights()
        self._update_preview()
        self._update_status_bar()
        self._update_stepper()

    def _update_preview(self):
        """Preview the current file using the production extraction path."""
        self._record_mapping()
        if not self.current_filepath:
            self.lbl_preview_status.setText("No file loaded.")
            self.text_preview.clear()
            return
            
        # Build Template
        meta_str_keys = {f"{r},{c}": v for (r, c), v in self.mapping["metadata_cells"].items()}
        template_data = {
            **self._source_template_options(),
            "header_row_idx": self.header_row_idx,
            "footer_row_idx": getattr(self, "footer_row_idx", None),
            "time_col": self.mapping["time_col"],
            "cycle_col": self.mapping["cycle_col"],
            "isotope_cols": self.mapping["isotope_cols"],
            "ratio_cols": self.mapping.get("ratio_cols", {}),
            "metadata_cells": meta_str_keys,
            "isotope_system": self.combo_system.currentText(),
            "instrument": self.input_instrument.text().strip(),
            "name_pattern": self.combo_name_parse.currentText(),
            "footer_marker": self.mapping.get("footer_marker", "***")
        }
        
        try:
            sample = extract_sample_data(self.current_filepath, template_data)
            self.lbl_preview_status.setText(
                f"Current file: {sample.sample_name} | {self.combo_system.currentText()} | "
                f"{len(sample.data)} cycles\nChannels: {', '.join(map(str, sample.data.columns)) or 'none'}\n"
                f"Missing fields / warnings: {len(sample.warnings)} (see details). Batch validation runs at conversion."
            )
            self.lbl_preview_status.setWordWrap(True)
            self.lbl_preview_status.setStyleSheet(
                f"color:{theme.OK}; font-weight:600;"
            )
            
            if sample.data.empty or not self.mapping.get("isotope_cols"):
                self.lbl_preview_status.setText("Mapping incomplete: no isotope cycles extracted. Map isotope columns before converting a TraceISO session.")
                self.lbl_preview_status.setStyleSheet(f"color:{theme.WARNING}; font-weight:600;")

            # Formatting the preview output
            lines = [
                f"=== Final TraceISO Domain Representation ===",
                f"Sample Group Name : {sample.sample_name}",
                f"",
                f"--- Metadata JSON ---"
            ]
            for k, v in sample.metadata.items():
                lines.append(f"  {k}: {v}")
                
            lines.append(f"")
            lines.append(f"--- Extracted Data Shape ---")
            lines.append(f"  {sample.data.shape[0]} Rows x {sample.data.shape[1]} Columns")
            
            lines.append(f"")
            lines.append(f"--- Extracted Columns (Preview top 3 rows) ---")
            if not sample.data.empty:
                # Get a string representation of the head of the dataframe
                df_str = sample.data.head(3).to_string()
                lines.append(df_str)
            else:
                lines.append("  (No Data Extracted)")

            # Column health: a mostly-NaN column means a mis-mapped source
            # column or wrong header row — flag it before the user converts.
            fractions = column_nan_fractions(sample)
            if fractions:
                lines.append(f"")
                lines.append(f"--- Column Health (NaN %) ---")
                high_nan = False
                for col, frac in fractions.items():
                    flag = "   <-- check mapping" if frac > NAN_FRACTION_THRESHOLD else ""
                    lines.append(f"  {col}: {frac * 100:.0f}% NaN{flag}")
                    high_nan = high_nan or frac > NAN_FRACTION_THRESHOLD
                if high_nan:
                    self.lbl_preview_status.setText(
                        "Extracted, but some columns are mostly empty — check mapping"
                    )
                    self.lbl_preview_status.setStyleSheet(
                        f"color:{theme.WARNING}; font-weight:600;"
                    )

            if len(sample.time_data) > 0:
                lines.append(f"")
                lines.append(f"--- Time Data (First 3 values in seconds) ---")
                lines.append(f"  {sample.time_data[:3]}")

            if sample.warnings:
                lines.append(f"")
                lines.append(f"--- Extraction Warnings ---")
                for message in sample.warnings:
                    lines.append(f"  {message}")
                self.lbl_preview_status.setText(
                    "Extracted with warnings — see below"
                )
                self.lbl_preview_status.setStyleSheet(
                    f"color:{theme.WARNING}; font-weight:600;"
                )

            self.text_preview.setText("\n".join(lines))
            
        except Exception as e:
            self.lbl_preview_status.setText("Extraction Failed format/mapping errors.")
            self.lbl_preview_status.setStyleSheet(
                f"color:{theme.BAD}; font-weight:600;"
            )
            self.text_preview.setText(traceback.format_exc())

    def _set_meta(self, r, c, field_name):
        self.mapping["metadata_cells"][(r, c)] = field_name
        self._update_ui_labels()

    def _open_isotope_dialog(self, cols_list, top_row=None):
        """Open ColumnMapperDialog with the selected grid columns already added.

        The user names each one (an isotope name is required); the rest of the
        file's columns stay available on the left for anything not selected.
        """
        if not self._name_selected_columns(cols_list, top_row):
            return
        available = self._available_column_names()
        already = dict(self.mapping.get("isotope_cols", {}))
        selected = [n for n in (self.get_col_name(c) for c in cols_list)
                    if n in available and n not in already]
        dlg = ColumnMapperDialog(available, already, mode="isotope", parent=self,
                                 preselect=selected, require_isotope_names=True)
        if dlg.exec_():
            try:
                self._set_isotope_cols(dlg.get_mapping())
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid column mapping", str(exc))
                return
            self._update_ui_labels()
            self._refresh_highlights()


    def _open_ratio_dialog(self, cols_list, top_row=None):
        """Open ColumnMapperDialog with the selected grid columns already added."""
        if not self._name_selected_columns(cols_list, top_row):
            return
        available = self._available_column_names()
        already = dict(self.mapping.get("ratio_cols", {}))
        selected = [n for n in (self.get_col_name(c) for c in cols_list)
                    if n in available and n not in already]
        dlg = ColumnMapperDialog(available, already, mode="ratio", parent=self,
                                 preselect=selected)
        if dlg.exec_():
            try:
                self.mapping["ratio_cols"] = dlg.get_mapping()
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid column mapping", str(exc))
                return
            self._update_ui_labels()


    def _effective_footer_start_row(self) -> int:
        """Return explicit footer row or first marker row in the preview table."""
        rows = self.table.rowCount()
        if self._is_positional():
            _, footer_row = self._positional_rows()
            return rows if footer_row is None else min(footer_row, rows)
        footer_start_row = getattr(self, "footer_row_idx", None)
        if footer_start_row is not None:
            return min(max(int(footer_start_row), 0), rows)

        if self.current_df is None or self.current_df.empty:
            return rows

        marker = self.mapping.get("footer_marker", "***")
        if not marker:
            return rows   # no footer marker: data runs to the end of the file
        start = min(max(int(self.header_row_idx) + 1, 0), len(self.current_df))
        stop = min(rows, len(self.current_df))
        for r in range(start, stop):
            first_col = str(self.current_df.iloc[r, 0]).strip()
            if first_col.startswith(marker):
                return r
        return rows


    def _refresh_highlights(self):
        self._normalize_mapping_schema()
        bg_header = QColor(theme.HL["header"][0])
        fg_header = QColor(theme.HL["header"][1])
        bg_meta = QColor(theme.HL["meta"][0])
        fg_meta = QColor(theme.HL["meta"][1])
        bg_time = QColor(theme.HL["time"][0])
        fg_time = QColor(theme.HL["time"][1])
        bg_iso = QColor(theme.HL["iso"][0])
        fg_iso = QColor(theme.HL["iso"][1])
        bg_ratio = QColor(theme.HL["ratio"][0])
        fg_ratio = QColor(theme.HL["ratio"][1])
        bg_footer = QColor(theme.HL["footer"][0])
        fg_footer = QColor(theme.HL["footer"][1])
        
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        
        # Explicit footer row, or first footer marker row when using dynamic
        # footer detection. This mirrors mapper_logic.extract_sample_data().
        footer_start_row = self._effective_footer_start_row()

        # Positional templates: the data-start marker line plays the header's
        # role and metadata comes from pattern-matched text lines. Baseline
        # columns are shown in their own colour next to their isotope.
        header_row = self.header_row_idx
        zero_of = {src: target for target, src in (self.mapping.get("baseline_cols") or {}).items()}
        pattern_rows = {}
        if self._is_positional():
            marker_row, _ = self._positional_rows()
            header_row = rows if marker_row is None else marker_row
            first = self.current_df.iloc[:, 0].tolist() if self.current_df is not None else []
            for field, pattern in (self.mapping.get("metadata_patterns") or {}).items():
                try:
                    compiled = re.compile(str(pattern))
                except re.error:
                    continue
                row = next((i for i, v in enumerate(first) if isinstance(v, str) and compiled.search(v)), None)
                if row is not None:
                    pattern_rows.setdefault(row, []).append(field)
        bg_zero = QColor(theme.HL["zero"][0])
        fg_zero = QColor(theme.HL["zero"][1])
        # Column names are the same on every row: work them out once, not per cell
        # (per cell, a 500 x 42 grid took seconds to repaint).
        col_names = [self.get_col_name(c) for c in range(cols)]
        baseline_on = self._baseline_enabled()

        for r in range(rows):
            is_footer = r >= footer_start_row
            
            for c in range(cols):
                item = self.table.item(r, c)
                if item:
                    item.setBackground(
                        QColor(theme.ROW_ODD) if r % 2 else QColor(theme.PAPER)
                    )
                    item.setForeground(QColor(theme.INK))
                    item.setFont(self.table.font())
                    item.setToolTip("")
                    
                    if is_footer:
                        item.setBackground(bg_footer)
                        item.setForeground(fg_footer)
                        continue # Skip other highlights for footer
                    
                    # Header Row
                    if r == header_row:
                        item.setBackground(bg_header)
                        item.setForeground(fg_header)
                        item.setFont(QFont(theme.ui_family(), 8, QFont.Bold))
                    
                    # Metadata
                    if (r, c) in self.mapping["metadata_cells"]:
                        item.setBackground(bg_meta)
                        item.setForeground(fg_meta)
                        item.setToolTip(f"Mapped to: {self.mapping['metadata_cells'][(r,c)]}")
                    if c == 0 and r in pattern_rows:
                        item.setBackground(bg_meta)
                        item.setForeground(fg_meta)
                        item.setToolTip("Metadata pattern: " + ", ".join(pattern_rows[r]))

                    # Column Logic (only applies BELOW header row)
                    if r > header_row:
                        col_name = col_names[c]
                        base_col = base_column_name(col_name)
                        zero_target = zero_of.get(col_name, zero_of.get(base_col))
                        if zero_target:
                            item.setBackground(bg_zero)
                            item.setForeground(fg_zero)
                            item.setToolTip(
                                f"Baseline of {zero_target}: subtracted from it on every cycle"
                                if baseline_on
                                else f"Baseline of {zero_target}: assigned, subtraction is off")
                        if (
                            col_name == self.mapping["time_col"]
                            or base_col == self.mapping["time_col"]
                            or col_name == self.mapping["cycle_col"]
                            or base_col == self.mapping["cycle_col"]
                        ):
                            item.setBackground(bg_time)
                            item.setForeground(fg_time)
                            item.setToolTip("Mapped as Time/Cycle column")

                        target_name = self.mapping["isotope_cols"].get(
                            col_name,
                            self.mapping["isotope_cols"].get(base_col),
                        )
                        if target_name:
                            item.setBackground(bg_iso)
                            item.setForeground(fg_iso)
                            item.setToolTip(f"Mapped to Isotope: {target_name}")

                        ratio_cols = self.mapping.get("ratio_cols", {})
                        ratio_target = None
                        if isinstance(ratio_cols, dict):
                            ratio_target = ratio_cols.get(
                                col_name,
                                ratio_cols.get(base_col),
                            )
                        if ratio_target:
                            item.setBackground(bg_ratio)
                            item.setForeground(fg_ratio)
                            item.setToolTip(f"Mapped to Ratio: {ratio_target}")

    def _clear_mapping(self, action, cols_list, min_row, max_row): # Assuming this is the method where the new code belongs
        self._normalize_mapping_schema()
        if action == "clear_selected_columns": # Placeholder for action_clear
            for c in cols_list:
                col_name = self.get_col_name(c)
                
                if col_name in self.mapping["isotope_cols"]:
                    del self.mapping["isotope_cols"][col_name]
                else:
                    keys_to_delete = [k for k, v in self.mapping["isotope_cols"].items() if k == col_name or v == col_name]
                    for k in keys_to_delete:
                        del self.mapping["isotope_cols"][k]
                        
                # Clear Ratios
                if isinstance(self.mapping.get("ratio_cols"), dict):
                    if col_name in self.mapping["ratio_cols"]:
                        del self.mapping["ratio_cols"][col_name]
                        
                # Clear Time/Cycle
                if self.mapping.get("time_col") == col_name: self.mapping["time_col"] = ""
                if self.mapping.get("cycle_col") == col_name: self.mapping["cycle_col"] = ""
                
            # Clear Metadata in the selected block
            for r in range(min_row, max_row + 1):
                for c in cols_list:
                    self.mapping["metadata_cells"].pop((r, c), None)
                    
            self._update_ui_labels()

    def save_template(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Template", "", "JSON Files (*.json)")
        if fname:
            self._normalize_mapping_schema()
            meta_str_keys = {f"{r},{c}": v for (r, c), v in self.mapping["metadata_cells"].items()}
            data = {
                **self._source_template_options(),
                "header_row_idx": self.header_row_idx,
                "footer_row_idx": self.footer_row_idx,
                "time_col": self.mapping["time_col"],
                "cycle_col": self.mapping["cycle_col"],
                "isotope_cols": self.mapping["isotope_cols"],
                "ratio_cols": self.mapping.get("ratio_cols", {}),
                "metadata_cells": meta_str_keys,
                "isotope_system": self.combo_system.currentText(),
                "instrument": self.input_instrument.text().strip(),
                "name_pattern": self.combo_name_parse.currentText(),
                "footer_marker": self.mapping.get("footer_marker", "***"),
            }
            try:
                with open(fname, "w") as f:
                    json.dump(data, f, indent=4)
                self._save_recent_template(fname)
                self._template_name = str(fname)
                self._template_baseline = self._mapping_snapshot()
                self._record_mapping()
                QMessageBox.information(self, "Success", "Template saved!")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def load_template(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Template", "", "JSON Files (*.json)")
        if fname:
            self._load_template_from_path(fname)

    def load_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Folder Containing Data")
        if dir_path:
            ext_filter = self.combo_ext.currentText()
            if ext_filter == "All Formats":
                valid_exts = [".csv", ".txt", ".exp", ".xls", ".xlsx"]
            else:
                valid_exts = [ext_filter]
                
            import glob
            files = []
            for ext in valid_exts:
                files.extend(list(Path(dir_path).glob(f"*{ext}")))
                files.extend(list(Path(dir_path).glob(f"*{ext.upper()}")))

            # Filter out hidden/temporary files (e.g., ~$Sequence.xlsx)
            files = [f for f in files if not f.name.startswith(('~', '.'))]
            files = sorted(list(set(files)), key=lambda p: p.name)
            if not files:
                QMessageBox.warning(self, "No Files Found",
                    f"No matching files ({valid_exts}) found in folder.")
                return
            self.file_list_widget.add_files([str(f) for f in files])
            self.onboarding.hide()
            if files:
                self.load_file(str(files[0]))
            self._update_status_bar()


    def _confirm_unsupported_isotope_system_if_needed(self) -> bool:
        """Warn before converting to an isotope system TraceISO cannot process.

        Ni/Zn (and other systems without a registered ElementConfig) can still
        be extracted/exported to HDF5, but TraceISO's correction/uncertainty
        pipeline cannot load them as a processable session. Returns False if
        the user cancels.
        """
        from domain.elements.registry import list_elements

        system = self.combo_system.currentText().strip()
        if not system or system in list_elements():
            return True
        reply = QMessageBox.question(
            self, "Export-only isotope system",
            f"Isotope system '{system}' has no TraceISO correction/uncertainty "
            "pipeline registered. The HDF5 file will be written (export-only) "
            "but TraceISO cannot process it as a session.\n\n"
            "Continue anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def run_batch_conversion(self, checked=False, *, selected=False, retry=False):
        # Refuse to start a second conversion/validation while one is still
        # active — this is the primary defense against overlapping batches;
        # the Convert/Cancel button states are a UX mirror of this, not the
        # source of truth.
        if self._batch_worker is not None and self._batch_worker.isRunning():
            return
        if self._validation_worker is not None and self._validation_worker.isRunning():
            return
        if not retry and self.file_list_widget.count() == 0:
            QMessageBox.warning(self, "No Files", "Please load files first.")
            return
        if not retry and not self.mapping["isotope_cols"] and not self.mapping["metadata_cells"]:
            QMessageBox.warning(self, "No Mapping",
                "Please define a mapping (Isotopes or Metadata) first.")
            return
        if not retry and not self._confirm_unsupported_isotope_system_if_needed():
            return
        file_paths = (list(self._last_outcome.failed) if retry else
                      [i.data(Qt.UserRole) for i in self.file_list_widget.list_widget.selectedItems()] if selected else
                      self.file_list_widget.all_paths())
        if not file_paths:
            QMessageBox.information(self, "No files selected", "Select one or more input files.")
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, f"Save HDF5 — {'retry ' if retry else ''}{len(file_paths)} files", "", "HDF5 Files (*.h5)")
        if not save_path:
            return
        if Path(save_path).exists() or Path(save_path).is_symlink():
            QMessageBox.warning(self, "Choose a new output", "Conversion requires a new file; existing outputs are preserved.")
            return
        if retry and Path(save_path).resolve() == Path(self._last_outcome.output_path).resolve():
            QMessageBox.warning(self, "Choose a new output", "Retry writes a separate output; choose a different destination.")
            return
        template_data = self._last_template if retry else self._build_template_data()
        self._validation_cancel_requested = False
        self.btn_cancel.setVisible(True)
        self.btn_cancel.setEnabled(True)
        self.btn_selected.setEnabled(False)

        # Phase 1: validate on background thread
        self.btn_convert.setEnabled(False)
        self.btn_convert.setText("Validating...")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.lbl_progress_file.setText("Running pre-flight validation...")
        self._pending_save_path = save_path
        self._pending_file_paths = file_paths
        self._pending_template = template_data

        self._validation_worker = ValidationWorker(file_paths, template_data, self)
        self._validation_worker.progress_update.connect(
            lambda pct, fn: (self.progress.setValue(pct),
                             self.lbl_progress_file.setText(f"Validating: {fn}")))
        self._validation_worker.finished.connect(self._on_validation_done)
        self._workers.append(self._validation_worker)
        self._validation_worker.start()

    def _on_validation_done(self, errors):
        if self._validation_cancel_requested:
            self._validation_cancel_requested = False
            self._reset_batch_ui()
            self.lbl_progress_file.setText("Validation cancelled.")
            return

        self._validation_messages = list(errors)
        if errors:
            self.batch_report.summary.setText(f"Pre-flight: {len(errors)} issues. Review the complete details below.")
            self.batch_report.details.setPlainText("\n".join(errors))
            self.batch_report.retry.setEnabled(False)
            self.batch_report.show()
            msg = (f"Pre-flight found {len(errors)} issues. Full details are in Results."
                   "\n\nProceed anyway? Failing files will be skipped.")
            reply = QMessageBox.question(self, "Validation Warnings", msg,
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                self._reset_batch_ui()
                return

        # Phase 2: convert. A fresh run token is minted here — the only point
        # a BatchWorker is created — so any signal delivered from a previously
        # started (and by now superseded) worker can be identified and
        # ignored by the slots below.
        self._run_token_counter += 1
        self._active_run_token = self._run_token_counter

        self.btn_convert.setText("Converting...")
        self.btn_convert.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_cancel.setVisible(True)
        self.progress.setValue(0)
        for p in self._pending_file_paths:
            self.file_list_widget.set_status(p, self.file_list_widget.UNKNOWN)

        self._batch_worker = BatchWorker(
            self._pending_file_paths, self._pending_template,
            self._pending_save_path, self._active_run_token, self)
        self._batch_worker.progress_update.connect(self._on_batch_progress)
        self._batch_worker.finished.connect(self._on_batch_done)
        self._batch_worker.fatal_error.connect(self._on_batch_error)
        self._workers.append(self._batch_worker)
        self._batch_worker.start()

    def _on_batch_progress(self, pct, filename, run_token):
        if run_token != self._active_run_token:
            return
        self.progress.setValue(pct)
        self.lbl_progress_file.setText(f"Processing: {filename}")

    def _on_batch_done(self, outcome):
        if outcome.run_token != self._active_run_token:
            # A stale/superseded run finished after a newer run took over
            # (or was itself cancelled and reset) — its result must not
            # touch the UI or be treated as the current output.
            return
        self._batch_worker = None
        self._reset_batch_ui()
        if outcome.cancelled:
            self.stepper.mark_conversion_incomplete()
        else:
            self.stepper.mark_converted()
        verification = getattr(outcome, "reader_verification", None)
        roundtrip_element_unknown = bool(verification and verification.get("element") == "unknown")
        verification_failed = bool(verification and not verification.get("ok"))
        if verification_failed:
            roundtrip_report = "\n\nReader verification failed: " + verification["error"]
            self.stepper.mark_conversion_incomplete()
        elif verification:
            roundtrip_report = (f"\n\nReader verification: {verification['samples']} samples; element {verification['element']}\n"
                                + "\n".join(map(str, verification["warnings"])))
        else:
            roundtrip_report = "\n\nReader verification was not performed."
        self._last_outcome = outcome
        self._last_template = getattr(self, "_pending_template", self._build_template_data())
        self.batch_report.set_outcome(outcome, getattr(self, "_pending_file_paths", self.file_list_widget.all_paths()), roundtrip_report)
        if getattr(self, "_validation_messages", None):
            self.batch_report.details.append("\nPre-flight diagnostics:\n" + "\n".join(self._validation_messages))
        self.batch_report.show()

        # Mark file statuses from the structured outcome (matched by path, not
        # by parsing error strings). Unprocessed files are left at their
        # existing UNKNOWN status rather than being marked VALID.
        failed_paths = set(outcome.failed)
        processed_paths = set(outcome.processed)
        for p in self.file_list_widget.all_paths():
            if p in failed_paths:
                self.file_list_widget.set_status(p, self.file_list_widget.INVALID)
            elif p in processed_paths:
                self.file_list_widget.set_status(p, self.file_list_widget.VALID)

        if outcome.cancelled:
            msg = (
                f"Batch cancelled by user — {len(outcome.processed)} file(s) "
                f"converted, {len(outcome.unprocessed)} file(s) not attempted."
            )
            if outcome.errors:
                msg += (
                    f"\n{len(outcome.errors)} file(s) failed before cancellation:\n"
                    + "\n".join(outcome.errors[:5])
                    + (f"\n...and {len(outcome.errors) - 5} more."
                       if len(outcome.errors) > 5 else "")
                )
            msg += (
                f"\n\nPartial output retained: {outcome.output_path}\n"
                f"Samples written before stopping: {outcome.sample_count}\n"
                "It is named and marked partial, and TraceISO warns when it is "
                "loaded. The output name you chose was not used."
            )
            QMessageBox.warning(self, "Cancelled", msg)
        elif outcome.errors:
            QMessageBox.warning(self, "Done with Errors",
                f"Batch complete — {len(outcome.errors)} file(s) failed.\n"
                f"Saved to {outcome.output_path}{roundtrip_report}")
        elif verification_failed:
            QMessageBox.warning(self, "Output written; verification failed", f"Saved to {outcome.output_path}{roundtrip_report}")
        elif roundtrip_element_unknown:
            QMessageBox.warning(
                self, "Export-only (not TraceISO-ready)",
                f"Batch complete, but no processable element was detected.\n"
                f"Saved to {outcome.output_path}{roundtrip_report}")
        else:
            QMessageBox.information(self, "Success",
                f"Batch complete!\nSaved to {outcome.output_path}{roundtrip_report}")

    def _on_batch_error(self, msg, run_token=0):
        if run_token != self._active_run_token:
            return
        self._batch_worker = None
        self._reset_batch_ui()
        self.stepper.mark_conversion_incomplete()
        for p in self.file_list_widget.all_paths():
            self.file_list_widget.set_status(p, self.file_list_widget.UNKNOWN)
        QMessageBox.critical(
            self, "Fatal Error",
            f"Batch failed:\n{msg}\n\n"
            "The output name you chose was not used. Nothing here should be "
            "treated as a complete session.")

    def _cancel_batch(self):
        if self._batch_worker is not None and self._batch_worker.isRunning():
            self._batch_worker.cancel()
            self.btn_convert.setText("Cancelling...")
            self.btn_convert.setEnabled(False)
            self.btn_cancel.setEnabled(False)
            self.lbl_progress_file.setText("Cancelling...")
            # UI is reset by _on_batch_done/_on_batch_error once this same
            # worker (matched by run token) actually stops — not here, and
            # without blocking this thread on the worker's completion.
            return
        if self._validation_worker is not None and self._validation_worker.isRunning():
            self._validation_cancel_requested = True
            self._validation_worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.lbl_progress_file.setText("Cancelling...")
            return
        self._reset_batch_ui()

    def _reset_batch_ui(self):
        self.btn_convert.setText(f"Convert All {self.file_list_widget.count()} Files...")
        self.btn_selected.setEnabled(True)
        self.btn_convert.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        self.btn_cancel.setVisible(False)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.lbl_progress_file.setText("")

    def _update_conversion_scope(self, *args):
        busy = any(w.isRunning() for w in self._workers)
        if not busy:
            self.btn_convert.setText(f"Convert All {self.file_list_widget.count()} Files...")
        self.btn_selected.setEnabled(not busy and bool(self.file_list_widget.list_widget.selectedItems()))

    def _retry_failed(self):
        if self._last_outcome and self._last_outcome.failed:
            self.run_batch_conversion(retry=True)

    def closeEvent(self, event):
        running = [w for w in self._workers if w.isRunning()]
        if running:
            for worker in running:
                worker.cancel()
            self.lbl_progress_file.setText("Stopping safely. Close again when the current operation finishes.")
            event.ignore()
            return
        event.accept()

    def _source_template_options(self):
        """Retain source-state and filename rules through preview/save/batch."""
        return {key: self.mapping[key] for key in
                SOURCE_TEMPLATE_KEYS
                if key in self.mapping}

    def _build_template_data(self):
        """Snapshot the current mapping for a batch.

        Returns a read-only copy, not the live ``self.mapping`` dictionaries:
        a batch must run under the mapping it was started with, and the
        mapping controls stay enabled while it runs.
        """
        meta_str_keys = {f"{r},{c}": v for (r, c), v in self.mapping["metadata_cells"].items()}
        return freeze_template_data({
            **self._source_template_options(),
            "include_audit_metadata": self.include_audit_metadata.isChecked(),
            "header_row_idx": self.header_row_idx,
            "footer_row_idx": self.footer_row_idx,
            "time_col": self.mapping["time_col"],
            "cycle_col": self.mapping["cycle_col"],
            "isotope_cols": self.mapping["isotope_cols"],
            "ratio_cols": self.mapping.get("ratio_cols", {}),
            "metadata_cells": meta_str_keys,
            "isotope_system": self.combo_system.currentText(),
            "instrument": self.input_instrument.text().strip(),
            "name_pattern": self.combo_name_parse.currentText(),
            "footer_marker": self.mapping.get("footer_marker", "***"),
        })

    def _load_template_from_path(self, path):
        """Load a template JSON from a file path (used by built-ins and recents)."""
        try:
            with open(path, "r") as f:
                data = normalize_template_schema(json.load(f))
            # A positional template has no header row; the grid still needs one.
            self.header_row_idx = data.get("header_row_idx") or 0
            self.set_footer_row(data.get("footer_row_idx", None))
            self.mapping["time_col"]  = data.get("time_col")
            self.mapping["cycle_col"] = data.get("cycle_col")
            self.mapping["isotope_cols"] = data.get("isotope_cols", {})
            self.mapping["ratio_cols"]   = data.get("ratio_cols", {})
            for key in SOURCE_TEMPLATE_KEYS:
                self.mapping.pop(key, None)
                if key in data:
                    self.mapping[key] = data[key]
            if "isotope_system" in data:
                self.combo_system.setCurrentText(data["isotope_system"])
            if "instrument" in data:
                self.input_instrument.setText(data["instrument"])
            if "name_pattern" in data:
                self.combo_name_parse.setCurrentText(data["name_pattern"])
            raw_meta = data.get("metadata_cells", {})
            self.mapping["metadata_cells"] = {
                tuple(map(int, k.split(","))): v for k, v in raw_meta.items()
            }
            self.set_header_row(self.header_row_idx)
            self._update_ui_labels()
            self._save_recent_template(path)
            self._name_preset.setCurrentIndex(0)
            self._template_name = str(path)
            self._template_baseline = self._mapping_snapshot()
            self._history_boundary()
            self._record_mapping()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _save_recent_template(self, path):
        recents = list(self._settings.value(_SETTINGS_KEY_RECENT, []) or [])
        path = str(path)
        if path in recents:
            recents.remove(path)
        recents.insert(0, path)
        self._settings.setValue(_SETTINGS_KEY_RECENT, recents[:MAX_RECENT])

    def _update_stepper(self):
        has_file    = self.current_df is not None
        has_header  = self.header_row_idx > 0 or has_file
        has_columns = bool(self.mapping.get("isotope_cols"))
        has_metadata = bool(self.mapping.get("metadata_cells"))
        has_preview = has_file and has_columns
        self.stepper.update_from_mapping(
            has_file=has_file, has_header=has_header,
            has_columns=has_columns, has_metadata=has_metadata,
            has_preview=has_preview)

    def _quick_map_isotopes(self):
        """Fix #9: Open isotope column mapper directly from the right panel."""
        if self.current_df is None:
            QMessageBox.information(self, "No File", "Load a file first.")
            return
        # Use the grid selection when there is one; otherwise offer every column.
        cells = self.table.selectedItems()
        selected_cols = sorted({i.column() for i in cells})
        if cells and not self._name_selected_columns(selected_cols, min(i.row() for i in cells)):
            return
        available = self._available_column_names()
        already   = dict(self.mapping.get("isotope_cols", {}))
        selected = [n for n in (self.get_col_name(c) for c in selected_cols)
                    if n in available and n not in already]
        dlg = ColumnMapperDialog(available, already, mode="isotope", parent=self,
                                 preselect=selected, require_isotope_names=True)
        if dlg.exec_():
            try:
                self._set_isotope_cols(dlg.get_mapping())
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid column mapping", str(exc))
                return
            self._update_ui_labels()
            self._refresh_highlights()

    def _quick_map_ratios(self):
        """Fix #9: Open ratio column mapper directly from the right panel."""
        if self.current_df is None:
            QMessageBox.information(self, "No File", "Load a file first.")
            return
        available = self._available_column_names()
        already   = dict(self.mapping.get("ratio_cols", {}))
        dlg = ColumnMapperDialog(available, already, mode="ratio", parent=self)
        if dlg.exec_():
            try:
                self.mapping["ratio_cols"] = dlg.get_mapping()
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid column mapping", str(exc))
                return
            self._update_ui_labels()


# Backward-compatible class name for callers written before the product rename.
UniversalMapperWindow = DataExtractorWindow

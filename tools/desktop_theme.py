"""Shared TraceISO desktop palette, controls, typography and window branding."""


from PyQt5.QtCore import Qt, QRectF, QSize
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPixmap
from PyQt5.QtWidgets import QApplication, QToolButton

from tools.qt_runtime import mono_font_family, ui_font_family

INK = "#1c2430"
INK_DIM = "#586573"
INK_FAINT = "#9aa4b0"
PAPER = "#ffffff"
APP_BG = "#f1f4f6"
PANEL_HEAD = "#f5f7f9"
BORDER = "#e0e4e9"
BORDER_STRONG = "#cdd4dc"
GRID_LINE = "#eef1f4"
ROW_ODD = "#fafbfc"
ACCENT = "#0e6b8a"
ACCENT_DK = "#0b566f"
ACCENT_SOFT = "#e7f2f6"
OK = "#2fa05a"
WARNING = "#9a6a10"
BAD = "#b9382b"
BAD_SOFT = "#fff0ed"


CONSOLE_BG = "#0f1620"
CONSOLE_FG = "#cfe0dc"
SEL_BG = "#d7eaf2"
SEL_FG = "#0b3b4c"

_UI_FAMILY = ui_font_family()
_MONO_FAMILY = mono_font_family()


def ui_family() -> str:
    return _UI_FAMILY


def mono_family() -> str:
    return _MONO_FAMILY


def _stylesheet() -> str:
    ui = _UI_FAMILY
    mono = _MONO_FAMILY
    return f"""
    QMainWindow, QDialog, QMainWindow > QWidget {{
        background: {APP_BG};
        color: {INK};
        font-family: "{ui}", "Segoe UI", "Helvetica Neue", sans-serif;
        font-size: 13px;
    }}
    QWidget {{
        color: {INK};
        font-family: "{ui}", "Segoe UI", "Helvetica Neue", sans-serif;
        font-size: 13px;
    }}
    QLabel {{ background: transparent; color: {INK}; }}
    QToolTip {{
        background: {INK}; color: #eef2f4; border: none;
        padding: 5px 8px; font-size: 12px;
    }}

    QGroupBox {{
        background: {PAPER};
        border: 1px solid {BORDER};
        border-radius: 6px;
        margin-top: 16px;
        padding: 10px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px; top: 3px; padding: 0 4px;
        color: {INK_DIM}; font-size: 12px; font-weight: 600;
        background: {PAPER};
    }}

    QPushButton, QToolButton {{
        background: {PAPER};
        border: 1px solid {BORDER_STRONG};
        border-radius: 6px;
        padding: 6px 12px;
        min-height: 20px;
        color: #2c3640;
        font-weight: 500;
    }}
    QPushButton:hover, QToolButton:hover {{ background: #f3f6f8; border-color: #c4ccd5; }}
    QPushButton:pressed, QToolButton:pressed {{ background: #eaeff3; }}
    QPushButton:disabled, QToolButton:disabled {{ color: {INK_FAINT}; border-color: #e4e8ed; }}
    *[actionRole="primary"] {{
        background: {ACCENT}; border: 1px solid {ACCENT};
        color: white; font-weight: 600; padding: 7px 16px;
    }}
    *[actionRole="primary"]:hover {{
        background: {ACCENT_DK}; border-color: {ACCENT_DK};
    }}
    *[actionRole="primary"]:disabled {{
        background: #aab9c0; border-color: #aab9c0; color: #eef2f4;
    }}
    *[actionRole="danger"] {{
        background: {BAD_SOFT}; border: 1px solid {BAD};
        color: {BAD}; font-weight: 600; padding: 6px 12px;
    }}
    *[actionRole="danger"]:hover {{ background: #fbe0da; border-color: {BAD}; }}

    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
        background: {PAPER}; border: 1px solid {BORDER_STRONG};
        border-radius: 6px; padding: 5px 9px; color: {INK};
        min-height: 20px;
        selection-background-color: {SEL_BG}; selection-color: {SEL_FG};
    }}
    QComboBox:hover, QLineEdit:hover {{ border-color: #c4ccd5; }}
    QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QPushButton:focus, QToolButton:focus, QTableView:focus, QTreeView:focus,
    QListWidget:focus, QTextEdit:focus {{ border: 2px solid {ACCENT}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{
        background: {PAPER}; border: 1px solid {BORDER};
        selection-background-color: {ACCENT_SOFT}; selection-color: {ACCENT};
        outline: 0;
    }}

    QListWidget, QTreeWidget {{
        background: {PAPER}; border: 1px solid {BORDER}; border-radius: 6px;
        alternate-background-color: {ROW_ODD};
        selection-background-color: {SEL_BG}; selection-color: {SEL_FG};
        outline: 0;
    }}
    QListWidget::item, QTreeWidget::item {{ padding: 4px 6px; border-radius: 4px; }}
    QListWidget::item:hover, QTreeWidget::item:hover {{ background: {ACCENT_SOFT}; }}

    QTableWidget, QTableView {{
        background: {PAPER}; border: 1px solid {BORDER}; border-radius: 6px;
        gridline-color: {GRID_LINE}; alternate-background-color: {ROW_ODD};
        selection-background-color: {SEL_BG}; selection-color: {SEL_FG};
        font-family: "{mono}", "Consolas", "Menlo", monospace; font-size: 12px;
    }}
    QHeaderView::section {{
        background: {PANEL_HEAD}; color: {INK_DIM};
        border: none; border-right: 1px solid {GRID_LINE};
        border-bottom: 1px solid {BORDER};
        padding: 6px 8px; font-family: "{ui}", "Segoe UI", "Helvetica Neue";
        font-size: 12px; font-weight: 600;
    }}
    QTableCornerButton::section {{ background: {PANEL_HEAD}; border: none; }}

    QTextEdit {{
        background: {PAPER}; color: {INK}; border: 1px solid {BORDER_STRONG};
        border-radius: 6px; padding: 8px;
    }}
    QTextEdit#extractionConsole {{
        background: {CONSOLE_BG}; color: {CONSOLE_FG};
        border: 1px solid #0a0f17; border-radius: 8px; padding: 8px;
        font-family: "{mono}", "Consolas", "Menlo", monospace; font-size: 12px;
        selection-background-color: {ACCENT}; selection-color: white;
    }}

    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{
        background: #cfd6dd; border-radius: 5px; min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{ background: #b6bfc8; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{
        background: #cfd6dd; border-radius: 5px; min-width: 24px;
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{
        width: 0; height: 0; background: none;
    }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

    QProgressBar {{
        background: {GRID_LINE}; border: 1px solid {BORDER}; border-radius: 6px;
        height: 14px; text-align: center; color: {INK_DIM}; font-size: 12px;
    }}
    QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}

    QStatusBar {{
        background: #f4f6f8; border-top: 1px solid #e4e8ed; color: {INK_DIM};
        font-family: "{mono}", "Consolas", "Menlo"; font-size: 12px;
    }}
    QStatusBar::item {{ border: none; }}

    QMenu {{
        background: {PAPER}; border: 1px solid {BORDER}; padding: 4px;
    }}
    QMenu::item {{
        padding: 6px 22px 6px 14px; border-radius: 5px; color: #2c3640;
    }}
    QMenu::item:selected {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
    QMenu::separator {{ height: 1px; background: {GRID_LINE}; margin: 4px 8px; }}

    QSplitter::handle {{ background: transparent; }}
    QSplitter::handle:horizontal {{ width: 8px; }}

    QScrollArea {{ background: transparent; border: none; }}
    QScrollArea > QWidget > QWidget {{ background: {APP_BG}; }}
    QToolBar {{
        background: {PAPER}; border: none; border-bottom: 1px solid {BORDER};
        spacing: 8px; padding: 8px;
    }}
    QToolBar QToolButton {{ padding: 6px 10px; }}
    QLabel[class="header"] {{ color: {INK}; font-size: 18px; font-weight: 600; }}
    QLabel[class="badge"] {{
        background: {ACCENT_SOFT}; color: {ACCENT}; border-radius: 4px;
        padding: 3px 8px; font-size: 12px; font-weight: 600;
    }}
    QLabel[heading="true"] {{ color: {INK}; font-size: 14px; font-weight: 600; }}
    QPushButton[class="danger"] {{
        background: {BAD_SOFT}; color: {BAD}; border: 1px solid {BAD};
    }}
    QPushButton[class="danger"]:hover {{ background: #fbe0da; }}
    *[actionRole="danger"]:disabled {{ background: {ROW_ODD}; color: {INK_FAINT}; border-color: {BORDER}; }}
    QCheckBox, QRadioButton {{ spacing: 6px; background: transparent; }}
    QCheckBox:focus, QRadioButton:focus {{ color: {ACCENT}; }}
    QTabWidget::pane {{ border: 1px solid {BORDER}; background: {PAPER}; }}
    QTabBar::tab {{ background: {PANEL_HEAD}; color: {INK_DIM}; padding: 8px 14px; }}
    QTabBar::tab:selected {{ background: {PAPER}; color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
    """


def apply(app) -> None:
    """Apply the theme to a QApplication."""
    app.setStyle("Fusion")
    app.setFont(QFont(_UI_FAMILY, 10))
    app.setWindowIcon(brand_icon())
    app.setProperty("traceisoDesktopTheme", True)

    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(APP_BG))
    palette.setColor(QPalette.Base, QColor(PAPER))
    palette.setColor(QPalette.AlternateBase, QColor(ROW_ODD))
    palette.setColor(QPalette.Text, QColor(INK))
    palette.setColor(QPalette.WindowText, QColor(INK))
    palette.setColor(QPalette.ButtonText, QColor(INK))
    palette.setColor(QPalette.Button, QColor(PAPER))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(INK_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(INK_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(INK_FAINT))
    palette.setColor(QPalette.Highlight, QColor(SEL_BG))
    palette.setColor(QPalette.HighlightedText, QColor(SEL_FG))
    palette.setColor(QPalette.ToolTipBase, QColor(INK))
    palette.setColor(QPalette.ToolTipText, QColor("#eef2f4"))
    app.setPalette(palette)
    app.setStyleSheet(_stylesheet())


def brand_icon():
    """Resolution-independent rendering of a shared TraceISO T mark."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(ACCENT))
        painter.drawRoundedRect(QRectF(0, 0, size, size), size * .18, size * .18)
        painter.setBrush(QColor(PAPER))
        painter.drawRect(QRectF(size * .22, size * .23, size * .56, size * .13))
        painter.drawRect(QRectF(size * .435, size * .30, size * .13, size * .47))
        painter.end()
        icon.addPixmap(pixmap)
    return icon


def style_window(window, title):
    app = QApplication.instance()
    if app is not None and not app.property("traceisoDesktopTheme"):
        apply(app)
    window.setWindowTitle(f"TraceISO — {title}")
    window.setWindowIcon(brand_icon())


def action_role(widget, role):
    """Assign primary/secondary/danger treatment without tool-specific selectors."""
    widget.setProperty("actionRole", role)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def style_toolbar(toolbar, primary=None, destructive=None):
    toolbar.setIconSize(QSize(18, 18))
    toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    for action in toolbar.actions():
        widget = toolbar.widgetForAction(action)
        if isinstance(widget, QToolButton):
            action_role(widget, "primary" if action is primary else "danger" if action is destructive else "secondary")

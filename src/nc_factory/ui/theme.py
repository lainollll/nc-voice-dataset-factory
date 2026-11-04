"""NeuralCompanion's charcoal/cyan scheme, kept local to the dataset app."""

from pathlib import Path


def configure_fonts() -> None:
    """Qt's offscreen Windows plugin needs explicit fonts; native Windows also benefits."""
    import os
    from PySide6.QtGui import QFontDatabase

    for filename in ("segoeui.ttf", "seguisb.ttf"):
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / filename
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))


# Palette taken from NeuralCompanion's stylesheet and the supplied reference.
# Resolve copied assets from this package, never from the working directory or NC.
_ASSETS = Path(__file__).resolve().parent / "assets"

DARK_STYLE = """
QWidget {
 background:#131a23; color:#e5e9f0;
 font-family:'Segoe UI'; font-size:13px; font-weight:400;
}
QMainWindow { background:#11161d; }
QWidget#workspacePage {
 background:#131a23; border:1px solid #283342; border-radius:12px;
}
QLabel { background:transparent; }
QLabel#title { font-size:22px; font-weight:600; color:#f2f5f9; }
QLabel#subtitle { color:#b7c1ce; }
QWidget#speechControlRow, QWidget#trainingExportRow,
QWidget#expandableSection, QWidget#expandableBody { background:transparent; }

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
 background:#131a23; border:1px solid #283342; border-radius:10px; padding:7px;
 selection-background-color:#233245; selection-color:#f2f5f9;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color:#66d9ef; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled { color:#7f8791; background:#1a2028; border-color:#27303b; }
QComboBox QAbstractItemView {
 background:#18202a; color:#e5e9f0; border:1px solid #4c6682;
 selection-background-color:#233245; selection-color:#f2f5f9;
}

QPushButton {
 background:#18202a; color:#f2f5f9; border:1px solid #4c6682;
 border-radius:10px; padding:8px 12px; font-weight:600;
}
QPushButton:hover { background:#233245; border-color:#66d9ef; }
QPushButton:pressed { background:#202b38; border-color:#35bfd7; }
QPushButton:focus { border-color:#66d9ef; }
QPushButton:disabled { color:#7f8791; background:#1a2028; border-color:#27303b; }
QPushButton#primary { background:#35bfd7; border-color:#66d9ef; color:#08212b; }
QPushButton#primary:hover { background:#66d9ef; color:#08212b; }
QPushButton#primary:pressed { background:#249eb5; color:#08212b; }
QPushButton#primary:disabled { background:#1a2028; border-color:#27303b; color:#7f8791; }

QListWidget {
 background:#11161d; border:1px solid #283342; border-radius:10px;
 padding:8px; min-width:180px;
}
QListWidget#workspaceNavigation { background:#131a23; }
QListWidget::item { padding:10px; border:2px solid transparent; border-radius:8px; }
QListWidget::item:hover { background:#18202a; }
QListWidget::item:selected { background:#233245; color:#f2f5f9; border-color:#66d9ef; }
QGroupBox {
 background:#18202a; border:1px solid #283342; border-radius:12px;
 margin-top:14px; padding:12px; font-weight:600;
}
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 6px; color:#f2f5f9; }

QToolButton#sectionToggle {
 background:transparent; border:1px solid transparent; border-radius:6px;
 padding:5px 0; text-align:left; color:#b7c1ce; font-weight:600;
}
QToolButton#sectionToggle:hover, QToolButton#sectionToggle:checked { color:#66d9ef; }
QToolButton#sectionToggle:focus { border-color:#66d9ef; }
QCheckBox { background:transparent; color:#e5e9f0; spacing:9px; min-height:24px; }
QCheckBox:disabled { color:#7f8791; }
QCheckBox::indicator { width:20px; height:20px; image:url("@CHECKBOX_OFF@"); }
QCheckBox::indicator:checked { image:url("@CHECKBOX_ON@"); }

QSlider { background:transparent; min-height:26px; }
QSlider::groove:horizontal {
 height:6px; background:#11161d; border:1px solid #283342; border-radius:4px;
}
QSlider::sub-page:horizontal { background:#35bfd7; border-radius:4px; }
QSlider::handle:horizontal {
 background:#66d9ef; border:1px solid #4c6682; width:14px;
 margin:-5px 0; border-radius:7px;
}
QSlider::handle:horizontal:hover { background:#e5e9f0; border-color:#66d9ef; }
QSlider::sub-page:horizontal:disabled { background:#35445a; }
QSlider::handle:horizontal:disabled { background:#5b6675; border-color:#7f8791; }

QTableWidget {
 background:#11161d; alternate-background-color:#18202a; color:#e5e9f0;
 gridline-color:#283342; border:1px solid #283342; border-radius:10px;
 selection-background-color:#233245; selection-color:#f2f5f9;
}
QHeaderView::section { background:#18202a; color:#b7c1ce; padding:8px; border:0; }
QProgressBar {
 background:#11161d; border:1px solid #283342; border-radius:8px;
 text-align:center; min-height:20px; color:#e5e9f0;
}
QProgressBar::chunk { background:#35bfd7; border-radius:7px; }
QScrollArea { background:#11161d; border:0; }
QScrollArea > QWidget { background:#11161d; }
QScrollBar:vertical { background:#131a23; width:16px; border-radius:8px; margin:2px; }
QScrollBar::handle:vertical { background:#3a516c; border-radius:6px; min-height:36px; }
QScrollBar::handle:vertical:hover { background:#4c6682; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:transparent; }
QScrollBar:horizontal { background:#131a23; height:16px; border-radius:8px; margin:2px; }
QScrollBar::handle:horizontal { background:#3a516c; border-radius:6px; min-width:36px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width:0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background:transparent; }
QDockWidget::title { background:#131a23; color:#f2f5f9; }
QStatusBar { background:#11161d; color:#8ea3b8; border-top:1px solid #273342; }
QToolTip { background:#18202a; color:#f2f5f9; border:1px solid #4c6682; padding:4px; }
""".replace("@CHECKBOX_OFF@", (_ASSETS / "checkbox_round_inactive.svg").as_posix()).replace(
    "@CHECKBOX_ON@", (_ASSETS / "checkbox_round_active.svg").as_posix()
)

def configure_fonts() -> None:
    """Qt's offscreen Windows plugin needs explicit fonts; native Windows also benefits."""
    import os
    from pathlib import Path
    from PySide6.QtGui import QFontDatabase

    for filename in ("segoeui.ttf", "seguisb.ttf"):
        font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / filename
        if font.is_file():
            QFontDatabase.addApplicationFont(str(font))


DARK_STYLE = """
QWidget { background:#10151e; color:#dfe7f3; font-family:'Segoe UI'; font-size:13px; }
QMainWindow { background:#10151e; }
QLabel#title { font-size:26px; font-weight:600; color:#f4f7fc; }
QLabel#subtitle { color:#93a4bd; font-size:13px; }
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
 background:#1a2433; border:1px solid #34435a; border-radius:5px; padding:7px;
 selection-background-color:#277c90; }
QPushButton { background:#253348; border:1px solid #3c506a; border-radius:5px; padding:8px 14px; }
QPushButton:hover { background:#34516d; } QPushButton:disabled { color:#617087; background:#17202d; }
QPushButton#primary { background:#157d8a; border-color:#20a7b7; font-weight:600; }
QListWidget { background:#151d2a; border:none; padding:10px; min-width:180px; }
QListWidget::item { padding:12px; border-radius:5px; } QListWidget::item:selected { background:#244558; color:#63dfdf; }
QGroupBox { border:1px solid #2e3b50; border-radius:7px; margin-top:14px; padding:14px; }
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 5px; color:#9cadc5; }
QTableWidget { background:#141d2a; alternate-background-color:#1a2433; gridline-color:#29374b; border:1px solid #2e3b50; }
QHeaderView::section { background:#243147; color:#b7c6db; padding:8px; border:none; }
QProgressBar { border:1px solid #34435a; border-radius:5px; text-align:center; min-height:20px; }
QProgressBar::chunk { background:#168e9c; } QScrollArea { border:none; }
QStatusBar { color:#95a8c2; } QToolTip { background:#26364c; color:white; }
"""

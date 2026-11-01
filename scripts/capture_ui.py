"""Capture all application pages for local visual verification, without inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from nc_factory.ui.window import MainWindow

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
app = QApplication([])
window = MainWindow(args.project)
window.resize(1450, 950)
window.show()
state = {"page": 0}


def capture() -> None:
    if window.runner.busy or window.aux_runner.busy or window.repo is None:
        QTimer.singleShot(100, capture)
        return
    page = state["page"]
    if page == window.pages.count():
        window.close()
        app.quit()
        return
    window.navigation.setCurrentRow(page)
    if window.navigation.currentItem().text() == "Dataset Browser" and window.browser_table.rowCount():
        window.browser_table.selectRow(0)

    def save() -> None:
        name = window.navigation.item(page).text().lower().replace(" ", "-")
        window.grab().save(str(args.output / f"{page + 1}-{name}.png"))
        state["page"] += 1
        QTimer.singleShot(100, capture)

    QTimer.singleShot(250, save)


QTimer.singleShot(100, capture)
raise SystemExit(app.exec())

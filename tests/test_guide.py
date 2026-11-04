from __future__ import annotations

import threading

import pytest
from PySide6.QtCore import Qt


def test_first_launch_guide_walks_to_existing_pages_without_starting_work(qtbot):
    from nc_factory.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert window.navigation.currentItem().text() == "Getting Started"
    guide = window.guide
    assert guide.steps.count() == 5
    assert "WAV" in guide.instructions.toPlainText()
    guide.mode.setCurrentIndex(1)
    assert guide.steps.count() == 8
    assert not guide.back_button.isEnabled()
    qtbot.mouseClick(guide.next_button, Qt.MouseButton.LeftButton)
    assert guide.steps.currentRow() == 1
    assert "model" in guide.instructions.toPlainText().lower()
    qtbot.mouseClick(guide.action_button, Qt.MouseButton.LeftButton)
    assert window.navigation.currentItem().text() == "Voice Manager"
    assert not window.runner.busy and not window.aux_runner.busy
    assert window.repo is None and window.pipeline is None
    window.open_page("Getting Started")
    assert guide.steps.currentRow() == 1
    guide.steps.setCurrentRow(7)
    assert not guide.next_button.isEnabled()
    qtbot.mouseClick(guide.action_button, Qt.MouseButton.LeftButton)
    assert window.navigation.currentItem().text() == "Export"


def test_guide_small_test_preset_is_explicit_and_preserves_voice_settings(qtbot):
    from nc_factory.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.target_count.setValue(500)
    window.target_seconds.setValue(3600)
    window.text_count.setValue(500)
    window.synthesis_spins["temperature"].setValue(0.63)
    window.guide.mode.setCurrentIndex(1)
    window.guide.steps.setCurrentRow(5)
    assert window.target_count.value() == 500  # Merely reading a step changes nothing.
    qtbot.mouseClick(window.guide.action_button, Qt.MouseButton.LeftButton)
    assert window.target_count.value() == window.text_count.value() == 10
    assert window.target_seconds.value() == 0
    assert window.synthesis_spins["temperature"].value() == 0.63
    assert window.navigation.currentItem().text() == "Batch Generator"
    assert not window.runner.busy and window.pipeline is None


def test_guide_does_not_change_test_targets_during_active_work(qtbot):
    from nc_factory.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    release = threading.Event()
    window.runner.start(lambda: release.wait(3))
    try:
        window.guide.mode.setCurrentIndex(1)
        window.guide.steps.setCurrentRow(5)
        qtbot.mouseClick(window.guide.action_button, Qt.MouseButton.LeftButton)
        assert window.target_count.value() == 1000
    finally:
        release.set()
        qtbot.waitUntil(lambda: not window.runner.busy, timeout=5000)


@pytest.mark.parametrize("with_project", [False, True])
def test_guide_stays_visible_beside_pages(qtbot, tmp_path, monkeypatch, with_project):
    from nc_factory.ui.window import MainWindow

    # Exercise project startup without opening storage or scheduling diagnostics.
    monkeypatch.setattr(MainWindow, "open_project", lambda self, path: None)
    window = MainWindow(tmp_path if with_project else None)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitUntil(window.isVisible)
    guide = window.guide
    guide.mode.setCurrentIndex(1)
    guide.steps.setCurrentRow(2)
    for title in ("Voice Manager", "Text Generator", "Batch Generator", "Export"):
        window.open_page(title)
        assert guide.isVisible()
        assert guide.steps.currentRow() == 2
        assert guide.mapTo(window, guide.rect().topLeft()).x() >= (window.centralWidget().geometry().right())
    assert not window.runner.busy and not window.aux_runner.busy


def test_detached_guide_keeps_navigation_and_returns_on_close(qtbot):
    from nc_factory.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    dock = window.guide_dock
    guide = window.guide
    guide.mode.setCurrentIndex(1)
    guide.steps.setCurrentRow(2)
    qtbot.mouseClick(dock.detach_button, Qt.MouseButton.LeftButton)
    assert dock.isFloating() and guide.isVisible()
    qtbot.mouseClick(guide.action_button, Qt.MouseButton.LeftButton)
    assert window.navigation.currentItem().text() == "Voice Manager"
    window.open_page("Text Generator")
    assert guide.isVisible() and guide.steps.currentRow() == 2
    instructions = guide.instructions.toPlainText()
    qtbot.mouseClick(dock.detach_button, Qt.MouseButton.LeftButton)
    assert not dock.isFloating() and guide.isVisible()
    qtbot.mouseClick(dock.detach_button, Qt.MouseButton.LeftButton)
    dock.close()
    assert not dock.isFloating() and guide.isVisible()
    assert guide.instructions.toPlainText() == instructions
    assert window.guide is guide
    qtbot.mouseClick(dock.detach_button, Qt.MouseButton.LeftButton)
    window.close()
    assert not dock.isVisible()

from __future__ import annotations

import threading


def test_duplicate_start_does_not_replace_active_pipeline(qtbot, tmp_path, config):
    from nc_factory.ui.window import MainWindow
    from nc_factory.pipeline import Pipeline
    from unittest.mock import Mock

    window = MainWindow()
    qtbot.addWidget(window)
    from nc_factory.storage import Repository

    window.repo = Repository(tmp_path / "project")
    window.apply_config(config)
    pipeline = Mock(spec=Pipeline)
    window.pipeline = pipeline
    release = threading.Event()
    window.run_task(lambda: release.wait(3))
    window.start_job()
    assert window.pipeline is pipeline
    window.stop_job()
    pipeline.request_stop.assert_called_once()
    release.set()
    qtbot.waitUntil(lambda: not window.runner.busy, timeout=5000)

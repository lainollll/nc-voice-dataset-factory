from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest


def test_runner_waits_for_native_cleanup_without_blocking_the_gui(qtbot, monkeypatch):
    from PySide6.QtCore import QTimer
    from nc_factory.ui.workers import TaskRunner

    release_task = threading.Event()
    native_cleanup_done = threading.Event()
    runner = TaskRunner()
    results, finished, waits, heartbeats = [], [], [], []
    runner.result.connect(results.append)
    runner.finished.connect(lambda: finished.append(True))
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: heartbeats.append(True))
    timer.start()
    runner.start(lambda: release_task.wait(3))
    thread = runner._thread
    real_wait = thread.wait

    def join_if_cleanup_done(timeout):
        waits.append(timeout)
        # Model native TLS cleanup that can continue after QThread.finished.
        return native_cleanup_done.is_set() and real_wait(timeout)

    monkeypatch.setattr(thread, "wait", join_if_cleanup_done)
    try:
        release_task.set()
        qtbot.waitUntil(lambda: bool(results))
        qtbot.waitUntil(lambda: bool(waits))
        heartbeat_count = len(heartbeats)
        qtbot.waitUntil(lambda: len(heartbeats) > heartbeat_count)
        assert runner.busy and not finished, "Do not release wrappers before native cleanup joins"
        assert waits and set(waits) == {0}, "Joining must never block the GUI"
    finally:
        native_cleanup_done.set()
        qtbot.waitUntil(lambda: not runner.busy, timeout=5000)
        timer.stop()
    assert results == [True] and finished == [True]


def test_close_waits_for_auxiliary_worker(qtbot):
    from nc_factory.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    release = threading.Event()
    window.run_aux(lambda: release.wait(3))
    assert window.aux_runner.busy
    window.close()
    assert window.isVisible()
    release.set()
    qtbot.waitUntil(lambda: not window.aux_runner.busy, timeout=5000)
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=5000)


def test_worker_error_is_visible_and_runner_reusable(qtbot):
    from nc_factory.ui.workers import TaskRunner

    runner = TaskRunner()
    errors, results = [], []
    runner.error.connect(errors.append)
    runner.result.connect(results.append)

    def fail():
        raise RuntimeError("a meaningful operation failure")

    with qtbot.waitSignal(runner.finished, timeout=5000):
        runner.start(fail)
        with pytest.raises(RuntimeError):
            runner.start(lambda: 2)
    assert errors and "meaningful operation failure" in errors[0]
    with qtbot.waitSignal(runner.finished, timeout=5000):
        runner.start(lambda: 7)
    assert results == [7]


def test_config_roundtrip_and_disk_estimator(tmp_path, config):
    from nc_factory.diagnostics import estimate_storage
    from nc_factory.domain import ProjectConfig
    from nc_factory.storage import Repository

    cfg = replace(config, custom_topics=("skärgård", "fåglar"), max_retries=3, target_count=1000)
    repo = Repository(tmp_path / "project")
    repo.save_config(cfg)
    assert repo.load_config() == cfg
    assert ProjectConfig.from_dict(cfg.to_dict()) == cfg
    estimate = estimate_storage(cfg)
    assert estimate["accepted_pcm_bytes"] >= 1000 * 3 * 24000 * 2
    assert estimate["estimated_total_bytes"] > estimate["accepted_pcm_bytes"]


def test_no_implicit_asr_tokenizer_download(tmp_path):
    from nc_factory.domain import QualitySettings
    from nc_factory.quality import FasterWhisperASR

    root = tmp_path / "incomplete-model"
    root.mkdir()
    with pytest.raises(ValueError, match="Incomplete|tokenizer"):
        FasterWhisperASR(QualitySettings(asr_model_path=str(root)))


def test_cli_ten_recording_verified_mock_smoke(tmp_path):
    from nc_factory.cli import run_smoke
    from nc_factory.export import validate_export

    output = tmp_path / "smoke"
    result = run_smoke(output, language="sv", count=10)
    assert result["accepted"] == 10 and result["verification"] == "MOCK ONLY"
    assert validate_export(Path(result["export"]))["valid"]
    assert (output / "project" / "factory.sqlite3").is_file()


def test_gui_configuration_keeps_synthesis_settings(tmp_path, config, qtbot):
    from nc_factory.storage import Repository
    from nc_factory.ui.window import MainWindow

    cfg = replace(
        config, synthesis=replace(config.synthesis, temperature=0.62, seed=231, variation=0.03), max_retries=4
    )
    window = MainWindow()
    qtbot.addWidget(window)
    window.repo = Repository(tmp_path / "project")
    window.apply_config(cfg)
    roundtrip = window.configuration()
    assert roundtrip.synthesis.temperature == 0.62
    assert roundtrip.synthesis.seed == 231
    assert roundtrip.synthesis.variation == 0.03
    assert roundtrip.max_retries == 4

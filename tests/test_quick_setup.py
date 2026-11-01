from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest


def model_files(root: Path, kind: str) -> Path:
    from nc_factory.models import PATTERNS

    root.mkdir(parents=True)
    for pattern in PATTERNS[kind]:
        (root / pattern.replace("*", "json")).write_bytes(b"model-fixture")
    if kind == "asr":
        (root / "config.json").write_text('{"lang_ids":[1,2]}', encoding="utf-8")
    return root


@pytest.fixture
def local_models(tmp_path):
    root = tmp_path / "app"
    model_files(root / "models" / "english", "english")
    model_files(root / "models" / "asr", "asr")
    return root


@pytest.fixture
def gpu(monkeypatch):
    info = {
        "cuda_available": True,
        "torch_mem_get_info": {"free_bytes": 12 * 1024**3},
        "nvidia_smi": [{"memory_free_mb": 12 * 1024}],
        "torch_device_name": "Test GPU",
        "torch_cuda_allocated_bytes": 0,
        "torch_cuda_reserved_bytes": 0,
    }
    info["nvidia_smi"][0].update(name="Test GPU", memory_total_mb=16 * 1024, utilization_percent=0)
    monkeypatch.setattr("nc_factory.diagnostics.gpu_status", lambda: info)
    return info


def test_discovery_reuses_saved_paths_and_skips_stale_or_incomplete_models(tmp_path):
    from nc_factory.quick_setup import discover_models

    tts = model_files(tmp_path / "external-tts", "english")
    asr = model_files(tmp_path / "external-asr", "asr")
    app = tmp_path / "app"
    project = app / "projects" / "prior"
    project.mkdir(parents=True)
    (project / "project_config.json").write_text(
        json.dumps({"synthesis": {"model_dir": str(tts)}, "quality": {"asr_model_path": str(asr)}}),
        encoding="utf-8",
    )
    model_files(app / "models" / "broken", "english")
    (app / "models" / "broken" / "s3gen.safetensors").unlink()
    result = discover_models(app, "en", cache_root=tmp_path / "empty-cache")
    assert result.engine == "english"
    assert result.tts_path == tts and result.asr_path == asr
    assert not result.missing
    (tts / "t3_cfg.safetensors").unlink()
    assert discover_models(app, "en", cache_root=tmp_path / "empty-cache").missing == ("english",)


def test_swedish_never_selects_english_and_honors_local_cache(local_models, tmp_path):
    from nc_factory.quick_setup import discover_models

    cache = tmp_path / "hub"
    result = discover_models(local_models, "sv", cache_root=cache)
    assert result.missing == ("multilingual-v3",)
    mtl = model_files(cache / "models--ResembleAI--chatterbox" / "snapshots" / "local", "multilingual-v3")
    result = discover_models(local_models, "sv", cache_root=cache)
    assert result.engine == "multilingual-v3" and result.tts_path == mtl
    (mtl / "Cangjie5_TC.json").write_bytes(b"")
    assert discover_models(local_models, "sv", cache_root=cache).missing == ("multilingual-v3",)


@pytest.mark.parametrize("minutes,free_gb,workers", [(0, 12, 2), (30, 6, 1)])
def test_prepare_automates_settings_and_creates_an_isolated_project(
    reference, local_models, tmp_path, gpu, minutes, free_gb, workers
):
    from nc_factory.quick_setup import QuickStartRequest, prepare_dataset

    gpu["nvidia_smi"][0]["memory_free_mb"] = free_gb * 1024
    request = QuickStartRequest(reference, "en", 20 if not minutes else 0, minutes, True)
    prepared = prepare_dataset(request, local_models, cache_root=tmp_path / "empty-cache")
    config = prepared.config
    assert config.target_count == (20 if not minutes else 0)
    assert config.target_seconds == minutes * 60
    assert config.max_attempts >= (100 if not minutes else 3000)
    assert config.synthesis.engine == "english" and config.synthesis.device == "cuda"
    from nc_factory.parallel import gpu_preflight

    assert config.auto_generation_workers and config.generation_workers == 8
    assert gpu_preflight(config)["workers"] == workers and config.overlap_quality and config.require_cuda
    assert config.quality.verify_asr and config.quality.asr_device == "cpu"
    assert config.quality.asr_compute_type == "int8" and not config.quality.speaker_enabled
    assert prepared.repo.load_config() == config
    assert prepared.items and all(item.language == "en" for item in prepared.items)
    config.validate()
    second = prepare_dataset(request, local_models, cache_root=tmp_path / "empty-cache")
    assert second.repo.root != prepared.repo.root
    assert prepared.repo.list_jobs() == []


def test_prepare_validates_permission_audio_and_gpu_before_creating_project(reference, local_models, gpu):
    from nc_factory.quick_setup import QuickStartRequest, prepare_dataset

    request = QuickStartRequest(reference, "en", 10, 0, False)
    with pytest.raises(ValueError, match="permission"):
        prepare_dataset(request, local_models)
    broken = reference.with_name("broken.wav")
    broken.write_bytes(b"not audio")
    with pytest.raises(Exception):
        prepare_dataset(replace(request, reference=broken, permission_confirmed=True), local_models)
    gpu["cuda_available"] = False
    with pytest.raises(RuntimeError, match="GPU|CUDA"):
        prepare_dataset(replace(request, permission_confirmed=True), local_models)
    assert not (local_models / "projects").exists()


def test_cancelled_setup_leaves_existing_project_untouched(reference, local_models, gpu):
    from nc_factory.quick_setup import QuickStartRequest, prepare_dataset

    project = local_models / "projects" / "keep"
    project.mkdir(parents=True)
    config = project / "project_config.json"
    config.write_text('{"name":"keep"}', encoding="utf-8")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        prepare_dataset(QuickStartRequest(reference, "en", 10, 0, True), local_models, cancel=cancel)
    assert config.read_text(encoding="utf-8") == '{"name":"keep"}'
    assert list((local_models / "projects").iterdir()) == [project]


def test_simple_form_generates_through_existing_pipeline_in_background(
    qtbot, reference, local_models, gpu, monkeypatch
):
    from PySide6.QtCore import Qt
    from nc_factory.domain import QualityResult
    from nc_factory.pipeline import Pipeline
    from nc_factory.tts import MockTTS
    from nc_factory.ui import quick_start
    from nc_factory.ui.window import MainWindow

    monkeypatch.setattr(quick_start, "APPLICATION_ROOT", local_models)
    gpu["nvidia_smi"][0]["memory_free_mb"] = 6 * 1024
    preparation_threads = []
    prepare = quick_start.prepare_dataset

    def prepare_tracked(*args, **kwargs):
        preparation_threads.append(threading.get_ident())
        return prepare(*args, **kwargs)

    class Quality:
        def check(self, *args):
            return QualityResult(True, {}, [])

    monkeypatch.setattr(quick_start, "prepare_dataset", prepare_tracked)
    monkeypatch.setattr(
        "nc_factory.ui.batch_page.Pipeline",
        lambda repo, on_progress: Pipeline(repo, tts=MockTTS(), quality=Quality(), on_progress=on_progress),
    )
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    page = window.quick_start
    page.reference.setText(str(reference))
    page.permission.setChecked(True)
    page.size.setValue(3)
    qtbot.mouseClick(page.generate_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: "COMPLETED" in page.status.text(), timeout=10000)
    qtbot.waitUntil(lambda: not window.runner.busy and not window.aux_runner.busy, timeout=5000)
    assert window.repo.statistics()["accepted"] == 3
    assert len(window.repo.list_jobs()) == 1
    assert preparation_threads and preparation_threads[0] != threading.get_ident()
    assert window.navigation.currentItem().text() == "Getting Started"
    assert page.export_button.isEnabled()
    assert window.target_count.value() == 3
    assert window.verify_asr.isChecked()
    assert window.guide.isVisible()


def test_simple_form_errors_are_visible_and_busy_job_is_preserved(
    qtbot, reference, local_models, monkeypatch
):
    from PySide6.QtCore import Qt
    from nc_factory.ui import quick_start
    from nc_factory.ui.window import MainWindow

    monkeypatch.setattr(quick_start, "APPLICATION_ROOT", local_models)
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    page = window.quick_start
    page.reference.setText(str(reference))
    qtbot.mouseClick(page.generate_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: "permission" in page.status.text().lower())
    qtbot.waitUntil(lambda: not window.runner.busy)
    assert window.repo is None
    release = threading.Event()
    window.runner.start(lambda: release.wait(5))
    try:
        page.permission.setChecked(True)
        qtbot.mouseClick(page.generate_button, Qt.MouseButton.LeftButton)
        assert "active" in page.status.text().lower()
        assert window.repo is None and window.pipeline is None
    finally:
        release.set()
        qtbot.waitUntil(lambda: not window.runner.busy)


def test_stop_during_simple_preparation_never_starts_a_job(qtbot, reference, local_models, gpu, monkeypatch):
    from PySide6.QtCore import Qt
    from nc_factory.ui import quick_start
    from nc_factory.ui.window import MainWindow

    monkeypatch.setattr(quick_start, "APPLICATION_ROOT", local_models)
    release, started = threading.Event(), threading.Event()
    prepare = quick_start.prepare_dataset

    def delayed(*args, **kwargs):
        started.set()
        release.wait(5)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(quick_start, "prepare_dataset", delayed)
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    page = window.quick_start
    page.reference.setText(str(reference))
    page.permission.setChecked(True)
    qtbot.mouseClick(page.generate_button, Qt.MouseButton.LeftButton)
    try:
        qtbot.waitUntil(started.is_set)
        qtbot.mouseClick(page.stop_button, Qt.MouseButton.LeftButton)
    finally:
        release.set()
        qtbot.waitUntil(lambda: not window.runner.busy)
    assert window.repo is None and window.pipeline is None
    assert page.generate_button.isEnabled()
    assert "cancel" in page.status.text().lower()
    assert not (local_models / "projects").exists()


def test_stop_while_project_diagnostics_are_pending_prevents_generation(
    qtbot, reference, local_models, gpu, monkeypatch
):
    from PySide6.QtCore import Qt
    from nc_factory.ui import quick_start
    from nc_factory.ui.window import MainWindow

    monkeypatch.setattr(quick_start, "APPLICATION_ROOT", local_models)
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    started, release = threading.Event(), threading.Event()

    def diagnostics():
        started.set()
        release.wait(5)

    monkeypatch.setattr(window, "refresh_diagnostics", lambda: window.run_aux(diagnostics))
    page = window.quick_start
    page.reference.setText(str(reference))
    page.permission.setChecked(True)
    qtbot.mouseClick(page.generate_button, Qt.MouseButton.LeftButton)
    try:
        qtbot.waitUntil(started.is_set)
        qtbot.mouseClick(page.stop_button, Qt.MouseButton.LeftButton)
    finally:
        release.set()
        qtbot.waitUntil(lambda: not window.runner.busy and not window.aux_runner.busy)
    assert window.repo.list_jobs() == [] and window.pipeline is None
    assert page.generate_button.isEnabled()
    assert "cancel" in page.status.text().lower()


def test_missing_models_require_inspection_then_explicit_download(qtbot, tmp_path, monkeypatch):
    from PySide6.QtCore import Qt
    from nc_factory.ui import quick_start
    from nc_factory.ui.window import MainWindow
    from nc_factory.quick_setup import discover_models

    root = tmp_path / "app"
    root.mkdir()
    monkeypatch.setattr(quick_start, "APPLICATION_ROOT", root)
    monkeypatch.setattr(
        quick_start,
        "discover_models",
        lambda *args, **kwargs: discover_models(*args, **kwargs, cache_root=tmp_path / "empty-cache"),
    )
    worker_threads = []

    def inspect(kind):
        worker_threads.append(threading.get_ident())
        return {
            "kind": kind,
            "repository": "fixture/" + kind,
            "bytes": 1024**3,
            "license": "test-license",
            "source": "https://example.test/model",
            "revision": "pinned",
        }

    def download(plan, destination):
        worker_threads.append(threading.get_ident())
        model_files(destination, plan["kind"])

    monkeypatch.setattr(quick_start, "model_download_plan", inspect)
    monkeypatch.setattr(quick_start, "download_model", download)
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    page = window.quick_start
    qtbot.mouseClick(page.models_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not window.runner.busy)
    assert "Download models" in page.models_button.text()
    assert "2.00 GiB" in page.models_button.text() and "test-license" in page.status.text()
    assert not (root / "models").exists()
    qtbot.mouseClick(page.models_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: not window.runner.busy)
    assert not discover_models(root, "en", cache_root=tmp_path / "empty-cache").missing
    assert "installed" in page.status.text()
    assert all(ident != threading.get_ident() for ident in worker_threads)


def test_simple_form_duration_request_and_new_wav_reset_permission(qtbot, tmp_path):
    from PySide6.QtCore import Qt
    from nc_factory.ui.quick_start import QuickStartPage

    page = QuickStartPage()
    qtbot.addWidget(page)
    page.show()
    page.reference.setText(str(tmp_path / "one.wav"))
    page.permission.setChecked(True)
    page.reference.setText(str(tmp_path / "two.wav"))
    assert not page.permission.isChecked()
    page.permission.setChecked(True)
    page.size_unit.setCurrentIndex(1)
    page.size.setValue(30)
    with qtbot.waitSignal(page.generate_requested) as signal:
        qtbot.mouseClick(page.generate_button, Qt.MouseButton.LeftButton)
    request = signal.args[0]
    assert request.target_count == 0 and request.target_minutes == 30
    assert request.reference.name == "two.wav" and request.permission_confirmed


def test_swedish_rejects_english_only_asr_even_in_a_custom_folder(local_models, tmp_path):
    from nc_factory.quick_setup import discover_models

    model_files(local_models / "models" / "multilingual", "multilingual-v3")
    asr = local_models / "models" / "asr"
    (asr / "config.json").write_text('{"lang_ids":[],"is_multilingual":false}', encoding="utf-8")
    (asr / "tokenizer.json").write_text('{"model":{"vocab":{"hello":1}}}', encoding="utf-8")
    result = discover_models(local_models, "sv", cache_root=tmp_path / "empty-cache")
    assert result.missing == ("asr",)
    assert discover_models(local_models, "en", cache_root=tmp_path / "empty-cache").asr_path == asr


def test_simple_form_fits_compact_window_without_horizontal_scrolling(qtbot):
    from nc_factory.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1000, 700)
    window.show()
    qtbot.waitUntil(window.isVisible)
    assert window.pages.currentWidget().horizontalScrollBar().maximum() == 0


def test_simple_resume_restores_progress_for_the_same_job(qtbot, reference, local_models, gpu, monkeypatch):
    from PySide6.QtCore import Qt
    from nc_factory.domain import QualityResult
    from nc_factory.pipeline import Pipeline
    from nc_factory.tts import MockTTS
    from nc_factory.ui import quick_start
    from nc_factory.ui.window import MainWindow

    monkeypatch.setattr(quick_start, "APPLICATION_ROOT", local_models)
    gpu["nvidia_smi"][0]["memory_free_mb"] = 6 * 1024
    started, release = threading.Event(), threading.Event()

    class Quality:
        def check(self, *args):
            started.set()
            release.wait(5)
            return QualityResult(True, {}, [])

    monkeypatch.setattr(
        "nc_factory.ui.batch_page.Pipeline",
        lambda repo, on_progress: Pipeline(repo, tts=MockTTS(), quality=Quality(), on_progress=on_progress),
    )
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    page = window.quick_start
    page.reference.setText(str(reference))
    page.permission.setChecked(True)
    page.size.setValue(8)
    qtbot.mouseClick(page.generate_button, Qt.MouseButton.LeftButton)
    try:
        qtbot.waitUntil(started.is_set, timeout=5000)
        qtbot.mouseClick(page.stop_button, Qt.MouseButton.LeftButton)
    finally:
        release.set()
        qtbot.waitUntil(lambda: not window.runner.busy and not window.aux_runner.busy, timeout=5000)
    assert window.repo.statistics()["accepted"] < 8
    qtbot.mouseClick(page.resume_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: "COMPLETED" in page.status.text(), timeout=10000)
    qtbot.waitUntil(lambda: not window.runner.busy and not window.aux_runner.busy)
    assert page.progress.value() == 8 and page.export_button.isEnabled()
    assert len(window.repo.list_jobs()) == 1

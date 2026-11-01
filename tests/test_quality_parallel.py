from __future__ import annotations

import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest


@pytest.fixture
def asr_folder(tmp_path):
    root = tmp_path / "asr"
    root.mkdir()
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (root / name).write_bytes(b"fixture")
    return root


def test_cpu_quality_auto_respects_cpu_ram_and_unsupported_modes(config, asr_folder, monkeypatch):
    from nc_factory.parallel import quality_worker_plan

    cfg = replace(
        config,
        auto_generation_workers=True,
        quality=replace(config.quality, verify_asr=True, asr_model_path=str(asr_folder)),
    )
    monkeypatch.setattr("nc_factory.parallel.os.cpu_count", lambda: 24)
    monkeypatch.setattr("nc_factory.diagnostics.available_memory_bytes", lambda: 16 * 1024**3)
    assert quality_worker_plan(cfg, 4)["quality_workers"] == 3
    monkeypatch.setattr("nc_factory.parallel.os.cpu_count", lambda: 32)
    assert quality_worker_plan(cfg, 4)["quality_workers"] == 4
    monkeypatch.setattr("nc_factory.parallel.os.cpu_count", lambda: 128)
    assert quality_worker_plan(cfg, 8)["quality_workers"] == 4
    monkeypatch.setattr("nc_factory.parallel.os.cpu_count", lambda: 4)
    assert quality_worker_plan(cfg, 4)["quality_workers"] == 1
    monkeypatch.setattr("nc_factory.parallel.os.cpu_count", lambda: 24)
    assert quality_worker_plan(replace(cfg, auto_generation_workers=False), 4)["quality_workers"] == 1
    assert (
        quality_worker_plan(replace(cfg, quality=replace(cfg.quality, asr_device="cuda")), 4)[
            "quality_workers"
        ]
        == 1
    )
    assert (
        quality_worker_plan(replace(cfg, quality=replace(cfg.quality, speaker_enabled=True)), 4)[
            "quality_workers"
        ]
        == 1
    )
    monkeypatch.setattr("nc_factory.diagnostics.available_memory_bytes", lambda: 2 * 1024**3)
    assert quality_worker_plan(cfg, 4)["quality_workers"] == 1
    monkeypatch.setattr("nc_factory.diagnostics.available_memory_bytes", lambda: None)
    assert quality_worker_plan(cfg, 4)["quality_workers"] == 1


def test_shared_asr_loads_once_but_transcriptions_really_overlap(config, asr_folder, monkeypatch):
    from nc_factory.quality import FasterWhisperASR

    created, calls = [], []
    barrier = threading.Barrier(2)

    class Model:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)
            time.sleep(0.02)

        def transcribe(self, *args, **kwargs):
            calls.append(threading.get_ident())
            barrier.wait(timeout=3)
            return [
                types.SimpleNamespace(
                    text="A correctly recognized sentence.", avg_logprob=-0.1, no_speech_prob=0.0
                )
            ], types.SimpleNamespace(language="en", language_probability=1.0)

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = Model
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    backend = FasterWhisperASR(replace(config.quality, asr_model_path=str(asr_folder)), num_workers=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: backend.transcribe(asr_folder / f"{i}.wav", "en"), range(2)))
    assert len(created) == 1 and created[0]["num_workers"] == 2 and created[0]["cpu_threads"] == 4
    assert len(set(calls)) == 2
    assert all(result.text == "A correctly recognized sentence." for result in results)


def test_scheduler_runs_independent_quality_checkers_concurrently_and_keeps_exact_target(
    tmp_path, config, asr_folder, monkeypatch
):
    from nc_factory import parallel
    from nc_factory.domain import QualityResult, TextItem
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    active = peak = 0
    instances, shared_ids = [], set()
    lock = threading.Lock()

    class Checker:
        def __init__(self, cfg, checker=None, asr=None):
            assert asr is not None
            instances.append(self)
            shared_ids.add(id(asr))

        def check(self, work):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.2)
                return QualityResult(True, {}, [])
            finally:
                with lock:
                    active -= 1

    monkeypatch.setattr("nc_factory.parallel.os.cpu_count", lambda: 24)
    monkeypatch.setattr("nc_factory.diagnostics.available_memory_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(parallel, "QualityWorker", Checker)
    monkeypatch.setattr(
        parallel,
        "ProcessPoolExecutor",
        lambda max_workers, **kwargs: ThreadPoolExecutor(max_workers=max_workers),
    )
    monkeypatch.setattr(
        parallel,
        "generate",
        lambda text, voice, settings, seed: MockTTS().synthesize(text, voice, settings, seed),
    )
    repo = Repository(tmp_path / "project")
    cfg = replace(
        config,
        auto_generation_workers=True,
        generation_workers=4,
        overlap_quality=True,
        target_count=8,
        quality=replace(config.quality, verify_asr=True, asr_model_path=str(asr_folder)),
    )
    items = [
        TextItem(f"Read a different sentence about item {i}.", f"Read a different sentence about item {i}.")
        for i in range(12)
    ]
    job = repo.create_job(cfg, items)
    events = []
    result = Pipeline(repo, on_progress=events.append).run(job)
    assert peak == 3 and len(instances) == 3 and len(shared_ids) == 1
    assert result["status"] == "COMPLETED" and result["accepted"] == result["attempts"] == 8
    assert result["runtime"]["quality_workers"] == 3
    assert any(event.get("runtime", {}).get("quality_active", 0) > 1 for event in events)
    assert not any(row["state"] in {"GENERATING", "GENERATED", "VALIDATING"} for row in repo.iter_samples())


@pytest.mark.parametrize("pause", [False, True])
def test_parallel_quality_drains_before_stop_or_pause(tmp_path, config, asr_folder, monkeypatch, pause):
    from nc_factory import parallel
    from nc_factory.domain import QualityResult, TextItem
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    started, release = threading.Event(), threading.Event()
    active, peak = 0, 0
    lock = threading.Lock()

    class Checker:
        def __init__(self, *args, **kwargs):
            pass

        def check(self, work):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 3:
                    started.set()
            try:
                assert release.wait(5)
                return QualityResult(True, {}, [])
            finally:
                with lock:
                    active -= 1

    monkeypatch.setattr("nc_factory.parallel.os.cpu_count", lambda: 24)
    monkeypatch.setattr("nc_factory.diagnostics.available_memory_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(parallel, "QualityWorker", Checker)
    monkeypatch.setattr(
        parallel,
        "ProcessPoolExecutor",
        lambda max_workers, **kwargs: ThreadPoolExecutor(max_workers=max_workers),
    )
    monkeypatch.setattr(
        parallel,
        "generate",
        lambda text, voice, settings, seed: MockTTS().synthesize(text, voice, settings, seed),
    )
    repo = Repository(tmp_path / "drain")
    cfg = replace(
        config,
        auto_generation_workers=True,
        generation_workers=4,
        overlap_quality=True,
        target_count=10,
        quality=replace(config.quality, verify_asr=True, asr_model_path=str(asr_folder)),
    )
    items = [
        TextItem(
            f"A distinct sentence for draining object {i}.", f"A distinct sentence for draining object {i}."
        )
        for i in range(10)
    ]
    job = repo.create_job(cfg, items)
    events, results, errors = [], [], []
    pipeline = Pipeline(repo, on_progress=events.append)

    def run():
        try:
            results.append(pipeline.run(job))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert started.wait(5)
        pipeline.request_pause() if pause else pipeline.request_stop()
        release.set()
        if pause:
            deadline = time.monotonic() + 5
            while repo.get_job(job)["status"] != "PAUSED" and time.monotonic() < deadline:
                time.sleep(0.01)
            assert repo.get_job(job)["status"] == "PAUSED" and active == 0
            assert not any(
                row["state"] in {"GENERATING", "GENERATED", "VALIDATING"} for row in repo.iter_samples()
            )
            snapshots = [e["runtime"] for e in events if "runtime" in e]
            assert snapshots[-1]["quality_active"] == snapshots[-1]["generation_active"] == 0
            pipeline.request_resume()
    finally:
        release.set()
        if pause:
            pipeline.request_resume()
        thread.join(10)
        if thread.is_alive():
            pipeline.request_stop()
            thread.join(5)
    assert not thread.is_alive() and not errors and peak == 3
    assert results[0]["status"] == ("COMPLETED" if pause else "STOPPED")
    if pause:
        assert results[0]["accepted"] == 10
    else:
        assert results[0]["accepted"] <= 7


def test_runtime_ui_reports_active_and_queued_work(qtbot):
    from nc_factory.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    runtime = {
        "device": "cuda:0",
        "workers": 4,
        "automatic_workers": True,
        "worker_limit": 8,
        "asr_device": "cpu",
        "quality_workers": 3,
        "generation_active": 2,
        "quality_active": 3,
        "quality_waiting": 1,
    }
    window._progress({"runtime": runtime, "status": "RUNNING", "accepted": 4})
    assert "2/4" in window.runtime_label.text() and "3/3" in window.runtime_label.text()
    assert "Queued checks: 1" in window.runtime_label.text()

from __future__ import annotations

import threading
import time
from dataclasses import replace

import pytest


def items(count):
    from nc_factory.domain import TextItem

    return [
        TextItem(
            f"This is a distinct sentence about object {n}.", f"This is a distinct sentence about object {n}."
        )
        for n in range(count)
    ]


def test_parallel_settings_validate_and_preserve_old_configuration(config):
    from nc_factory.domain import ProjectConfig

    old = config.to_dict()
    for key in ("generation_workers", "overlap_quality", "require_cuda", "gpu_memory_per_worker_gb"):
        old.pop(key, None)
    loaded = ProjectConfig.from_dict(old)
    assert loaded.generation_workers == 1 and not loaded.overlap_quality
    replace(config, generation_workers=8).validate()
    for workers in (0, 9, True, 1.5):
        with pytest.raises(ValueError, match="worker"):
            replace(config, generation_workers=workers).validate()
    native = replace(config, synthesis=replace(config.synthesis, engine="english", device="cpu"))
    with pytest.raises(ValueError, match="CUDA|GPU"):
        replace(native, require_cuda=True).validate()


def test_spawned_generators_are_isolated_and_exact_target_is_not_overshot(tmp_path, config):
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository

    repo = Repository(tmp_path / "parallel")
    cfg = replace(config, target_count=8, generation_workers=2, overlap_quality=True)
    job = repo.create_job(cfg, items(12))
    result = Pipeline(repo).run(job)
    accepted = list(repo.iter_samples(state="ACCEPTED"))
    assert result["accepted"] == result["attempts"] == len(accepted) == 8
    assert result["status"] == "COMPLETED"
    assert len({s["parameters"]["worker_pid"] for s in accepted}) == 2
    assert len({s["path"] for s in accepted}) == 8
    assert not any(s["state"] in {"GENERATING", "GENERATED", "VALIDATING"} for s in repo.iter_samples())


def test_gpu_work_overlaps_quality_and_stop_drains_all_owned_workers(tmp_path, config):
    from nc_factory.domain import QualityResult
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    quality_started, second_synthesis, release = threading.Event(), threading.Event(), threading.Event()
    owner = threading.get_ident()
    quality_threads = []

    class Synth(MockTTS):
        def synthesize(self, *args):
            if quality_started.is_set():
                second_synthesis.set()
            time.sleep(0.02)
            return super().synthesize(*args)

    class Quality:
        def prepare(self, voice):
            quality_threads.append(threading.get_ident())

        def check(self, *args):
            quality_threads.append(threading.get_ident())
            quality_started.set()
            assert release.wait(5)
            return QualityResult(True, {}, [])

    repo = Repository(tmp_path / "overlap")
    cfg = replace(config, target_count=5, overlap_quality=True)
    job = repo.create_job(cfg, items(8))
    pipeline = Pipeline(repo, tts=Synth(), quality=Quality())
    results, errors = [], []

    def run():
        try:
            results.append(pipeline.run(job))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert quality_started.wait(5)
        assert second_synthesis.wait(5), "TTS should continue while previous audio is checked"
        pipeline.request_stop()
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and not errors
    assert results[0]["status"] == "STOPPED"
    assert 1 <= results[0]["accepted"] <= 2
    assert len(set(quality_threads)) == 1 and quality_threads[0] != owner
    assert not any(s["state"] in {"GENERATING", "GENERATED", "VALIDATING"} for s in repo.iter_samples())


def test_parallel_oom_releases_lock_and_budget_survives_resume(tmp_path, config):
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    class OOM(MockTTS):
        def synthesize(self, *args):
            raise RuntimeError("CUDA out of memory")

    repo = Repository(tmp_path / "oom")
    cfg = replace(config, target_count=2, max_attempts=1, overlap_quality=True)
    job = repo.create_job(cfg, items(2))
    result = Pipeline(repo, tts=OOM()).run(job)
    assert result["status"] == "NEEDS_ATTENTION" and result["accepted"] == 0
    assert result["attempts"] == 1
    with Repository(repo.root).writer_lock():
        pass
    assert Pipeline(repo, tts=MockTTS()).run(job)["attempts"] == 1


def test_parallel_controls_round_trip_and_preflight_rejects_insufficient_vram(
    tmp_path, config, qtbot, monkeypatch
):
    from nc_factory.storage import Repository
    from nc_factory.ui.window import MainWindow
    from nc_factory.parallel import gpu_preflight
    from nc_factory import diagnostics

    cfg = replace(
        config, generation_workers=8, overlap_quality=True, require_cuda=True, gpu_memory_per_worker_gb=4.5
    )
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.generation_workers.value() == 8 and window.auto_generation_workers.isChecked()
    window.repo = Repository(tmp_path / "ui")
    window.apply_config(cfg)
    restored = window.configuration()
    assert restored.generation_workers == 8 and restored.overlap_quality and restored.require_cuda
    assert restored.gpu_memory_per_worker_gb == 4.5
    native = replace(cfg, synthesis=replace(cfg.synthesis, engine="english", device="cuda"))
    gpu_info = {
        "cuda_available": True,
        "torch_device_name": "Test GPU",
        "torch_mem_get_info": {"free_bytes": 40 * 1024**3},
        "nvidia_smi": [{"memory_free_mb": 32 * 1024}],
    }
    monkeypatch.setattr(diagnostics, "gpu_status", lambda: gpu_info)
    with pytest.raises(RuntimeError, match="memory preflight"):
        gpu_preflight(native)
    gpu_info["nvidia_smi"][0]["memory_free_mb"] = 40 * 1024
    runtime = gpu_preflight(native)
    assert runtime["workers"] == 8 and runtime["estimated_required_gb"] == 37


def test_parallel_retry_replenishes_without_exceeding_budget_or_target(tmp_path, config):
    from nc_factory.domain import QualityResult
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    class RejectFirst:
        calls = 0

        def check(self, *args):
            self.calls += 1
            return QualityResult(self.calls > 1, {}, [] if self.calls > 1 else ["retry-me"])

    repo = Repository(tmp_path / "retry")
    cfg = replace(config, overlap_quality=True, target_count=3, max_attempts=4, max_retries=1)
    job = repo.create_job(cfg, items(3))
    result = Pipeline(repo, tts=MockTTS(), quality=RejectFirst()).run(job)
    assert result["status"] == "COMPLETED" and result["accepted"] == 3 and result["attempts"] == 4
    all_attempts = [attempt for sample in repo.iter_samples() for attempt in repo.attempts(sample["id"])]
    assert len({attempt["seed"] for attempt in all_attempts}) == 4


def test_parallel_pause_drains_then_resume_completes(tmp_path, config):
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    repo = Repository(tmp_path / "pause")
    cfg = replace(config, overlap_quality=True, target_count=4)
    job = repo.create_job(cfg, items(4))
    pipeline = Pipeline(repo, tts=MockTTS())

    def progress(event):
        if event.get("audio_path") and event["accepted"] == 1:
            pipeline.request_pause()

    pipeline.on_progress = progress
    errors = []

    def run():
        try:
            pipeline.run(job)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while repo.get_job(job)["status"] != "PAUSED" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert repo.get_job(job)["status"] == "PAUSED"
        assert not any(s["state"] in {"GENERATING", "GENERATED", "VALIDATING"} for s in repo.iter_samples())
        pipeline.request_resume()
    finally:
        pipeline.request_resume()
        thread.join(10)
        if thread.is_alive():
            pipeline.request_stop()
            thread.join(5)
    assert not thread.is_alive() and not errors
    assert repo.statistics(job)["accepted"] == 4 and repo.get_job(job)["status"] == "COMPLETED"


def test_overlapping_quality_never_accepts_duplicate_audio(tmp_path, config):
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    class SameAudio(MockTTS):
        def synthesize(self, text, voice, settings, seed):
            return super().synthesize("The same audio for every item.", voice, settings, 0)

    repo = Repository(tmp_path / "duplicates")
    cfg = replace(config, target_count=2, max_attempts=2, max_retries=0, overlap_quality=True)
    job = repo.create_job(cfg, items(2))
    result = Pipeline(repo, tts=SameAudio()).run(job)
    assert result["accepted"] == 1 and result["status"] == "NEEDS_ATTENTION"
    assert "duplicate_audio" in next(repo.iter_samples(state="REJECTED"))["reasons"]


def test_quality_gpu_oom_stops_submissions_and_retains_review_results(tmp_path, config):
    from nc_factory.domain import QualityResult
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    class QualityOOM:
        def check(self, *args):
            return QualityResult(
                False, {"asr_error": "CUDA out of memory"}, ["asr_error_review_required"], needs_review=True
            )

    repo = Repository(tmp_path / "asr-oom")
    job = repo.create_job(replace(config, overlap_quality=True, target_count=5), items(5))
    result = Pipeline(repo, tts=MockTTS(), quality=QualityOOM()).run(job)
    assert result["status"] == "NEEDS_ATTENTION" and "out of memory" in result["error"]
    assert result["attempts"] <= 2 and result["accepted"] == 0
    assert all(
        sample["state"] not in {"GENERATING", "GENERATED", "VALIDATING"} for sample in repo.iter_samples()
    )

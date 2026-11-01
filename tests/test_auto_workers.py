from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest


@pytest.fixture
def native_config(config):
    return replace(config, synthesis=replace(config.synthesis, engine="english", device="cuda"))


def test_old_jobs_keep_manual_workers_and_auto_requires_native_cuda(native_config):
    from nc_factory.domain import ProjectConfig

    saved = native_config.to_dict()
    saved.pop("auto_generation_workers", None)
    assert not ProjectConfig.from_dict(saved).auto_generation_workers
    with pytest.raises(ValueError, match="Auto|automatic|GPU"):
        replace(native_config, auto_generation_workers="yes").validate(check_reference=False)
    with pytest.raises(ValueError, match="GPU|CUDA"):
        replace(
            native_config,
            synthesis=replace(native_config.synthesis, device="cpu"),
            generation_workers=1,
            auto_generation_workers=True,
        ).validate(check_reference=False)


@pytest.mark.parametrize(
    "free_gb,limit,expected", [(6, 8, 1), (12, 8, 2), (24, 8, 5), (80, 8, 8), (24, 3, 3)]
)
def test_auto_workers_fit_available_vram_and_obey_limit(native_config, monkeypatch, free_gb, limit, expected):
    from nc_factory.parallel import gpu_preflight

    monkeypatch.setattr(
        "nc_factory.diagnostics.gpu_status",
        lambda: {
            "cuda_available": True,
            "torch_device_name": "Test GPU",
            "torch_mem_get_info": {"free_bytes": 80 * 1024**3},
            "nvidia_smi": [{"memory_free_mb": free_gb * 1024}],
        },
    )
    config = replace(native_config, auto_generation_workers=True, generation_workers=limit)
    result = gpu_preflight(config)
    assert result["workers"] == expected and result["worker_limit"] == limit
    assert result["automatic_workers"] and result["estimated_required_gb"] <= free_gb
    assert config.generation_workers == limit


def test_auto_never_disables_memory_reserve_or_silently_falls_back(native_config, monkeypatch):
    from nc_factory.parallel import gpu_preflight

    info = {
        "cuda_available": True,
        "torch_device_name": "Test GPU",
        "torch_mem_get_info": {"free_bytes": 4 * 1024**3},
        "nvidia_smi": [],
    }
    monkeypatch.setattr("nc_factory.diagnostics.gpu_status", lambda: info)
    config = replace(native_config, auto_generation_workers=True, generation_workers=8)
    with pytest.raises(RuntimeError, match="memory preflight"):
        gpu_preflight(config)
    info["torch_mem_get_info"]["free_bytes"] = 12 * 1024**3
    assert gpu_preflight(replace(config, minimum_gpu_free_gb=5))["workers"] == 1
    with pytest.raises(RuntimeError, match="memory preflight"):
        gpu_preflight(replace(config, auto_generation_workers=False))
    info["cuda_available"] = False
    with pytest.raises(RuntimeError, match="CUDA"):
        gpu_preflight(config)


def test_auto_leaves_extra_headroom_for_gpu_transcript_checking(native_config, monkeypatch):
    from nc_factory.parallel import gpu_preflight

    monkeypatch.setattr(
        "nc_factory.diagnostics.gpu_status",
        lambda: {
            "cuda_available": True,
            "torch_device_name": "Test GPU",
            "torch_mem_get_info": {"free_bytes": 12 * 1024**3},
            "nvidia_smi": [],
        },
    )
    config = replace(
        native_config,
        generation_workers=8,
        auto_generation_workers=True,
        quality=replace(native_config.quality, verify_asr=True, asr_device="cuda"),
    )
    assert gpu_preflight(config)["workers"] == 1


def test_scheduler_uses_resolved_workers_instead_of_saved_maximum(tmp_path, config, monkeypatch):
    from nc_factory import parallel
    from nc_factory.domain import QualityResult, TextItem
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    active = peak = 0
    lock = threading.Lock()

    def generate(text, voice, settings, seed):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.1)
            return MockTTS().synthesize(text, voice, settings, seed)
        finally:
            with lock:
                active -= 1

    class Quality:
        def check(self, *args):
            return QualityResult(True, {}, [])

    monkeypatch.setattr(
        parallel,
        "gpu_preflight",
        lambda cfg: {
            "device": "cpu",
            "workers": 2,
            "worker_limit": 8,
            "automatic_workers": True,
        },
    )
    monkeypatch.setattr(
        parallel,
        "ProcessPoolExecutor",
        lambda max_workers, **kwargs: ThreadPoolExecutor(max_workers=max_workers),
    )
    monkeypatch.setattr(parallel, "generate", generate)
    repo = Repository(tmp_path / "project")
    cfg = replace(
        config, target_count=6, generation_workers=8, auto_generation_workers=True, overlap_quality=True
    )
    texts = [
        TextItem(
            f"Read this distinct sentence about object {n}.", f"Read this distinct sentence about object {n}."
        )
        for n in range(6)
    ]
    job = repo.create_job(cfg, texts)
    result = Pipeline(repo, quality=Quality()).run(job)
    assert peak == 2
    assert result["status"] == "COMPLETED" and result["accepted"] == result["attempts"] == 6
    assert result["runtime"]["workers"] == 2
    saved = json.loads(repo.get_job(job)["config_json"])
    assert saved["generation_workers"] == 8 and saved["auto_generation_workers"]


def test_auto_rechecks_memory_on_resume_and_keeps_saved_limit(tmp_path, native_config, monkeypatch):
    from nc_factory import parallel
    from nc_factory.domain import QualityResult, TextItem
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    info = {
        "cuda_available": True,
        "torch_device_name": "Test GPU",
        "torch_mem_get_info": {"free_bytes": 6 * 1024**3},
        "nvidia_smi": [],
    }
    monkeypatch.setattr("nc_factory.diagnostics.gpu_status", lambda: info)
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

    class Quality:
        def check(self, *args):
            return QualityResult(True, {}, [])

    repo = Repository(tmp_path / "resume")
    cfg = replace(
        native_config,
        target_count=6,
        generation_workers=8,
        auto_generation_workers=True,
        overlap_quality=True,
    )
    texts = [
        TextItem(
            f"Another distinct training sentence about object {n}.",
            f"Another distinct training sentence about object {n}.",
        )
        for n in range(6)
    ]
    job = repo.create_job(cfg, texts)
    pipeline = Pipeline(repo, quality=Quality())
    pipeline.on_progress = lambda event: pipeline.request_stop() if event.get("accepted") == 1 else None
    first = pipeline.run(job)
    assert first["status"] == "STOPPED" and first["runtime"]["workers"] == 1
    assert first["accepted"] < 6
    info["torch_mem_get_info"]["free_bytes"] = 12 * 1024**3
    resumed = Pipeline(repo, quality=Quality()).run(job)
    assert resumed["status"] == "COMPLETED" and resumed["accepted"] == 6
    assert resumed["runtime"]["workers"] == 2
    saved = json.loads(repo.get_job(job)["config_json"])
    assert saved["generation_workers"] == 8 and saved["auto_generation_workers"]


def test_auto_controls_round_trip_and_manual_settings_are_preserved(qtbot, tmp_path, config):
    from nc_factory.storage import Repository
    from nc_factory.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.repo = Repository(tmp_path / "project")
    assert window.auto_generation_workers.isChecked() and window.generation_workers.value() == 8
    window.apply_config(replace(config, generation_workers=3))
    assert not window.auto_generation_workers.isChecked() and window.generation_workers.value() == 3
    window.auto_generation_workers.setChecked(True)
    saved = window.configuration()
    assert saved.auto_generation_workers and saved.generation_workers == 3
    window.apply_config(saved)
    assert window.auto_generation_workers.isChecked()

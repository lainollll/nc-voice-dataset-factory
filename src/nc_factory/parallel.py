"""Bounded generation/QC scheduler; the calling thread is the sole project writer."""

from __future__ import annotations

import multiprocessing
import math
import os
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import TYPE_CHECKING
from dataclasses import replace

from .domain import ProjectConfig
from .generation_worker import CPU_THREADS_PER_GENERATOR, generate, initialize
from .storage import utcnow

if TYPE_CHECKING:
    from .pipeline import Pipeline


def gpu_preflight(config: ProjectConfig) -> dict:
    mode = {"worker_limit": config.generation_workers, "automatic_workers": False}
    if config.synthesis.engine in {"mock", "documented-service"}:
        return {"device": config.synthesis.device, "workers": config.generation_workers, **mode}
    if not config.synthesis.device.startswith("cuda"):
        if config.auto_generation_workers:
            raise ValueError("Automatic GPU workers require the CUDA synthesis device")
        return {"device": "cpu", "workers": 1, **mode}
    if config.synthesis.device not in {"cuda", "cuda:0"}:
        raise ValueError("Parallel GPU mode currently uses cuda:0; select cuda or cuda:0")
    from .diagnostics import gpu_status

    info = gpu_status()
    if not info["cuda_available"]:
        raise RuntimeError("CUDA is required but unavailable; CPU fallback is disabled")
    free = (info.get("torch_mem_get_info") or {}).get("free_bytes", 0) / 1024**3
    physical = info.get("nvidia_smi") or []
    if physical and physical[0].get("memory_free_mb") is not None:
        free = min(free, physical[0]["memory_free_mb"] / 1024)
    reserve = max(1, config.minimum_gpu_free_gb)
    workers = config.generation_workers
    if config.auto_generation_workers:
        if config.quality.verify_asr and config.quality.asr_device.startswith("cuda"):
            reserve += 4.0
        if not math.isfinite(free):
            raise RuntimeError("GPU memory preflight could not measure available VRAM")
        workers = min(workers, max(0, math.floor((free - reserve) / config.gpu_memory_per_worker_gb)))
        mode["automatic_workers"] = True
    required = max(1, workers) * config.gpu_memory_per_worker_gb + reserve
    if free < required:
        advice = "Free GPU memory or choose fewer workers."
        if config.auto_generation_workers:
            advice = "Free GPU memory before trying again."
            if config.quality.verify_asr and config.quality.asr_device.startswith("cuda"):
                advice += " Moving transcript checking to CPU also frees headroom."
        raise RuntimeError(
            f"GPU worker memory preflight: {free:.2f} GiB free; {required:.2f} GiB estimated for "
            f"{max(1, workers)} worker(s) including reserve. {advice}"
        )
    return {
        "device": "cuda:0",
        "gpu_name": info["torch_device_name"],
        "workers": workers,
        "free_gb": free,
        "estimated_required_gb": required,
        "reserved_gb": reserve,
        **mode,
    }


def quality_worker_plan(config: ProjectConfig, generation_workers: int) -> dict:
    """Budget CPU transcript concurrency while keeping manual/GPU/speaker paths serial."""
    plan = {"quality_workers": 1}
    settings = config.quality
    if (
        not config.auto_generation_workers
        or not settings.verify_asr
        or settings.asr_device != "cpu"
        or settings.speaker_enabled
    ):
        return plan
    from .diagnostics import available_memory_bytes
    from .quality import ASR_CPU_THREADS

    available = available_memory_bytes()
    if available is None:
        return plan
    try:
        weights = (Path(settings.asr_model_path) / "model.bin").stat().st_size
    except OSError:
        return plan
    # One shared model budget plus half a GiB workspace per extra CT2 worker.
    base = 2 * weights + 512 * 1024**2
    memory_limit = max(1, 1 + (available - 2 * 1024**3 - base) // (512 * 1024**2))
    cores = os.cpu_count() or 1
    cpu_limit = max(1, (cores - 2 - generation_workers * CPU_THREADS_PER_GENERATOR) // ASR_CPU_THREADS)
    plan.update(
        quality_workers=min(4, generation_workers + 1, cpu_limit, memory_limit),
        quality_cpu_threads=ASR_CPU_THREADS,
        available_ram_gb=available / 1024**3,
    )
    return plan


class QualityWorker:
    def __init__(self, config: ProjectConfig, checker=None, asr=None) -> None:
        self.config, self.checker, self.prepared = config, checker, False
        self.asr = asr

    def check(self, work):
        started = time.perf_counter()
        if self.checker is None:
            from .quality import QualityController

            self.checker = QualityController(self.config.quality, asr=self.asr)
        if not self.prepared:
            if hasattr(self.checker, "prepare"):
                self.checker.prepare(self.config.voice)
            self.prepared = True
        result = self.checker.check(
            Path(work.fields["path"]),
            work.sample["training_text"],
            work.sample["language"],
            self.config.voice,
        )
        return replace(
            result,
            metrics={
                **result.metrics,
                "quality_seconds": time.perf_counter() - started,
                "quality_worker_thread": threading.get_ident(),
            },
        )


def run_parallel(pipeline: Pipeline, job_id: str, config: ProjectConfig, attempt_budget: int) -> dict:
    repo = pipeline.repo
    runtime = gpu_preflight(config)
    workers = runtime["workers"]
    if pipeline.tts is not None and workers != 1:
        raise ValueError(
            "Injected TTS backends support one generation worker; use native spawned workers for parallel synthesis"
        )
    runtime.update(
        quality_worker_plan(config, workers) if pipeline.quality is None else {"quality_workers": 1}
    )
    quality_workers = runtime["quality_workers"]
    runtime.update(
        asr_device=config.quality.asr_device,
        overlap_quality=config.overlap_quality,
        generation_active=0,
        quality_active=0,
        quality_waiting=0,
    )
    repo.record_event("parallel_runtime", {"job_id": job_id, **runtime})
    pipeline._emit(job_id, config, status="RUNNING", runtime=runtime.copy())
    # No database/file writes occur in either executor. Process workers own TTS/RNG;
    # each QC thread owns a checker; CPU ASR shares weights via native workers.
    # Only CPU NumPy samples cross the process boundary.
    if pipeline.tts is None:
        generation = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,
            initargs=(config.synthesis,),
        )

        def submit(work):
            return generation.submit(generate, work.spoken_text, config.voice, work.settings, work.seed)
    else:
        generation = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-tts")

        def submit(work):
            return generation.submit(
                pipeline.tts.synthesize, work.spoken_text, config.voice, work.settings, work.seed
            )

    quality = ThreadPoolExecutor(max_workers=quality_workers, thread_name_prefix="audio-qc")
    local_checkers = threading.local()
    shared_asr = None
    shared_asr_lock = threading.Lock()

    def check(work):
        nonlocal shared_asr
        if not hasattr(local_checkers, "checker"):
            if quality_workers > 1:
                from .quality import FasterWhisperASR

                with shared_asr_lock:
                    if shared_asr is None:
                        shared_asr = FasterWhisperASR(config.quality, num_workers=quality_workers)
            local_checkers.checker = QualityWorker(config, pipeline.quality, asr=shared_asr)
        return local_checkers.checker.check(work)

    generating, checking, ready = {}, {}, deque()
    fatal, exhausted = "", False
    last_activity, last_snapshot = 0.0, None
    try:
        while True:
            pipeline._external_control(job_id)
            # Finish and journal completions on this thread, never in callbacks.
            for future, work in list(generating.items()):
                if not future.done():
                    continue
                del generating[future]
                try:
                    result = future.result()
                    pipeline._save_generated(work, result, config)
                    ready.append(work)
                except Exception as exc:
                    outcome = pipeline._fail_work(job_id, config, work, exc)
                    if outcome == "oom" or isinstance(exc, BrokenProcessPool):
                        fatal = f"Generation worker stopped: {type(exc).__name__}: {exc}"
            for future, work in list(checking.items()):
                if not future.done():
                    continue
                del checking[future]
                try:
                    checked = future.result()
                    pipeline._finish_checked(job_id, config, work, checked)
                    errors = " ".join(
                        str(checked.metrics.get(key, "")) for key in ("asr_error", "speaker_error")
                    )
                    if "out of memory" in errors.lower():
                        fatal = "Quality GPU out of memory; free VRAM before Resume, or start a new job with CPU/int8 ASR"
                except Exception as exc:
                    pipeline._fail_work(job_id, config, work, exc)
                    fatal = f"Quality worker failed: {type(exc).__name__}: {exc}"
            while ready and len(checking) < quality_workers and (config.overlap_quality or not generating):
                work = ready.popleft()
                checking[quality.submit(check, work)] = work
            outstanding = len(generating) + len(checking) + len(ready)
            if not outstanding:
                runtime.update(generation_active=0, quality_active=0, quality_waiting=0)
                if last_snapshot not in {None, (0, 0, 0)}:
                    pipeline._emit(job_id, config, status="RUNNING", runtime=runtime.copy())
                    last_snapshot = (0, 0, 0)
                if fatal:
                    repo.update_job(job_id, status="NEEDS_ATTENTION", error=fatal)
                    break
                if not pipeline._checkpoint(job_id, config):
                    break
                stats = repo.statistics(job_id)
                if pipeline.targets_met(stats, config):
                    repo.update_job(job_id, status="COMPLETED", finished_at=utcnow())
                    break
                exhausted = exhausted and repo.next_pending(job_id) is None
                if stats["attempts"] >= attempt_budget or exhausted:
                    repo.update_job(
                        job_id,
                        status="NEEDS_ATTENTION",
                        error="attempt_budget" if stats["attempts"] >= attempt_budget else "needs_text",
                    )
                    break
            draining = (
                fatal
                or pipeline.stop_event.is_set()
                or pipeline.pause_event.is_set()
                or (not config.overlap_quality and (checking or ready))
            )
            while not draining and len(generating) < workers and outstanding < workers + quality_workers:
                stats = repo.statistics(job_id)
                if pipeline.targets_met(stats, config) or stats["attempts"] >= attempt_budget:
                    break
                # Count-only jobs reserve an acceptance slot for every outstanding
                # attempt; retries release slots. Duration targets can overshoot by
                # bounded in-flight clips because their durations are unknown.
                if config.target_count and stats["accepted"] < config.target_count:
                    if stats["accepted"] + outstanding >= config.target_count:
                        break
                sample = repo.next_pending(job_id)
                if sample is None and pipeline.text_source:
                    repo.add_texts(
                        job_id, pipeline.text_source(min(32, max(1, config.target_count - stats["accepted"])))
                    )
                    sample = repo.next_pending(job_id)
                if sample is None:
                    exhausted = True
                    break
                exhausted = False
                pipeline._check_disk(config)
                work = pipeline._begin_work(job_id, config, sample, stats["attempts"])
                try:
                    generating[submit(work)] = work
                except Exception as exc:
                    pipeline._fail_work(job_id, config, work, exc)
                    fatal = f"Unable to submit generation worker: {exc}"
                    break
                outstanding += 1
                draining = pipeline.stop_event.is_set() or pipeline.pause_event.is_set()
            snapshot = (len(generating), len(checking), len(ready))
            now = time.monotonic()
            if snapshot != last_snapshot and now - last_activity >= 0.5:
                runtime.update(
                    generation_active=snapshot[0], quality_active=snapshot[1], quality_waiting=snapshot[2]
                )
                pipeline._emit(job_id, config, status="RUNNING", runtime=runtime.copy())
                last_activity, last_snapshot = now, snapshot
            if generating or checking:
                wait([*generating, *checking], timeout=0.05, return_when=FIRST_COMPLETED)
            else:
                # No active work: re-evaluate target/budget/text/control immediately.
                continue
    finally:
        generation.shutdown(wait=True, cancel_futures=False)
        quality.shutdown(wait=True, cancel_futures=False)
    final = repo.get_job(job_id)
    runtime.update(generation_active=0, quality_active=0, quality_waiting=0)
    pipeline._emit(job_id, config, status=final["status"], error=final["error"], runtime=runtime.copy())
    return {**repo.statistics(job_id), "status": final["status"], "error": final["error"], "runtime": runtime}

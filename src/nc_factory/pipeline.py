from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from dataclasses import asdict, replace, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .audio import audio_metrics, fingerprint, process_audio, write_wav_atomic
from .domain import AudioResult, ProjectConfig, QualityResult, SynthesisSettings, TextItem
from .storage import Repository, utcnow
from .tts import TTSBackend, create_tts

log = logging.getLogger(__name__)


@dataclass
class AttemptWork:
    sample: dict[str, Any]
    attempt_id: str
    settings: SynthesisSettings
    seed: int
    spoken_text: str
    fields: dict[str, Any] = field(default_factory=dict)
    raw_metrics: dict[str, Any] = field(default_factory=dict)
    audio_fingerprint: str = ""
    synthesis_seconds: float = 0.0


class Pipeline:
    def __init__(
        self,
        repo: Repository,
        tts: TTSBackend | None = None,
        quality: Any = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        text_source: Callable[[int], list[TextItem]] | None = None,
    ) -> None:
        self.repo, self.tts, self.quality = repo, tts, quality
        self.on_progress, self.text_source = on_progress, text_source
        self.stop_event, self.pause_event = threading.Event(), threading.Event()
        self._started = 0.0
        self._baseline: dict[str, Any] = {}

    def request_pause(self) -> None:
        self.pause_event.set()

    def request_resume(self) -> None:
        self.pause_event.clear()

    def request_stop(self) -> None:
        self.stop_event.set()

    def _emit(self, job_id: str, config: ProjectConfig, **extra: Any) -> None:
        stats = self.repo.statistics(job_id)
        elapsed = max(0.001, time.monotonic() - self._started)
        accepted_delta = stats["accepted"] - self._baseline.get("accepted", 0)
        duration_delta = stats["accepted_seconds"] - self._baseline.get("accepted_seconds", 0)
        rate = accepted_delta / elapsed
        eta = None
        if accepted_delta >= 3 and rate > 0:
            eta = max(0, config.target_count - stats["accepted"]) / rate
            if config.target_seconds and duration_delta > 0:
                eta = max(
                    eta, (config.target_seconds - stats["accepted_seconds"]) / (duration_delta / elapsed)
                )
        if self.on_progress:
            try:
                self.on_progress(
                    {
                        **stats,
                        "job_id": job_id,
                        "rate": rate,
                        "eta_seconds": eta,
                        "elapsed_seconds": elapsed,
                        **extra,
                    }
                )
            except Exception:
                log.exception("Progress callback failed")

    def _checkpoint(self, job_id: str, config: ProjectConfig) -> bool:
        self._external_control(job_id)
        if self.pause_event.is_set():
            self.repo.update_job(job_id, status="PAUSED", command="pause")
            self._emit(job_id, config, status="PAUSED")
            while self.pause_event.is_set() and not self.stop_event.wait(0.1):
                self._external_control(job_id)
            if not self.stop_event.is_set():
                self.repo.update_job(job_id, status="RUNNING", command="run")
        if self.stop_event.is_set():
            self.repo.update_job(job_id, status="STOPPED", command="stop")
            self._emit(job_id, config, status="STOPPED")
            return False
        return True

    def _external_control(self, job_id: str) -> None:
        from .control import take_job_control

        command = take_job_control(self.repo.root, job_id)
        if command == "pause":
            self.pause_event.set()
        elif command == "resume":
            self.pause_event.clear()
        elif command == "stop":
            self.stop_event.set()
        if command:
            self.repo.record_event("external_job_control", {"job_id": job_id, "command": command})

    @staticmethod
    def targets_met(stats: dict[str, Any], config: ProjectConfig) -> bool:
        return (not config.target_count or stats["accepted"] >= config.target_count) and (
            not config.target_seconds or stats["accepted_seconds"] >= config.target_seconds
        )

    def run(self, job_id: str) -> dict[str, Any]:
        owned_backend = self.tts is None
        with self.repo.writer_lock():
            self.repo.recover()
            job = self.repo.get_job(job_id)
            config = ProjectConfig.from_dict(json.loads(job["config_json"]))
            attempt_budget = max(config.max_attempts, job.get("attempt_budget_override", 0))
            self._started = time.monotonic()
            self._baseline = self.repo.statistics(job_id)
            try:
                config.validate()
                self._check_disk(config)
                if (
                    config.synthesis.device.startswith("cuda")
                    and config.synthesis.engine not in {"mock", "documented-service"}
                    and (config.minimum_gpu_free_gb or config.require_cuda)
                ):
                    from .diagnostics import gpu_status

                    info = gpu_status()
                    free = (info.get("torch_mem_get_info") or {}).get("free_bytes", 0) / 1024**3
                    if not info["cuda_available"] or free < config.minimum_gpu_free_gb:
                        raise RuntimeError(
                            f"GPU preflight: {free:.2f} GiB available; minimum {config.minimum_gpu_free_gb:.2f} GiB"
                        )
                self.repo.update_job(job_id, status="RUNNING", command="run", error="", started_at=utcnow())
                if config.auto_generation_workers or config.generation_workers > 1 or config.overlap_quality:
                    from .parallel import run_parallel

                    return run_parallel(self, job_id, config, attempt_budget)
                backend = self.tts or create_tts(config.synthesis)
                from .quality import QualityController

                checker = self.quality or QualityController(config.quality)
                checker.prepare(config.voice) if hasattr(checker, "prepare") else None
                log.info(
                    "Job %s started (engine=%s, language=%s)",
                    job_id,
                    config.synthesis.engine,
                    config.language,
                )
                while self._checkpoint(job_id, config):
                    stats = self.repo.statistics(job_id)
                    if self.targets_met(stats, config):
                        self.repo.update_job(job_id, status="COMPLETED", finished_at=utcnow())
                        break
                    if stats["attempts"] >= attempt_budget:
                        self.repo.update_job(job_id, status="NEEDS_ATTENTION", error="attempt_budget")
                        break
                    sample = self.repo.next_pending(job_id)
                    if sample is None and self.text_source:
                        self.repo.add_texts(
                            job_id, self.text_source(min(32, max(1, config.target_count - stats["accepted"])))
                        )
                        sample = self.repo.next_pending(job_id)
                    if sample is None:
                        self.repo.update_job(job_id, status="NEEDS_ATTENTION", error="needs_text")
                        break
                    self._check_disk(config)
                    outcome = self._attempt(job_id, config, sample, backend, checker, stats["attempts"])
                    if outcome == "oom":
                        self.repo.update_job(
                            job_id,
                            status="NEEDS_ATTENTION",
                            error="CUDA out of memory; free VRAM before resuming",
                        )
                        break
                final = self.repo.get_job(job_id)
                log.info("Job %s ended: %s (%s)", job_id, final["status"], final["error"])
                self._emit(job_id, config, status=final["status"], error=final["error"])
                return {**self.repo.statistics(job_id), "status": final["status"], "error": final["error"]}
            except Exception as exc:
                self.repo.update_job(job_id, status="FAILED", error=f"{type(exc).__name__}: {exc}")
                self._emit(job_id, config, status="FAILED", error=str(exc))
                raise
            finally:
                if owned_backend and "backend" in locals():
                    backend.close()

    def _check_disk(self, config: ProjectConfig) -> None:
        free = shutil.disk_usage(self.repo.root).free / 1024**3
        if free < config.minimum_free_gb:
            raise OSError(f"Low disk space: {free:.2f} GiB free; minimum {config.minimum_free_gb:.2f} GiB")

    def _begin_work(
        self, job_id: str, config: ProjectConfig, sample: dict[str, Any], total_attempts: int
    ) -> AttemptWork:
        seed = (config.synthesis.seed + total_attempts) % (2**31)
        settings = config.synthesis
        if settings.variation:
            delta = np.random.default_rng(seed).uniform(-settings.variation, settings.variation)
            settings = replace(settings, temperature=float(np.clip(settings.temperature + delta, 0.01, 2)))
        spoken_text = sample.get("generation_text") or sample["spoken_text"]
        attempt_id = self.repo.begin_attempt(
            sample["id"], seed, {**asdict(settings), "spoken_text": spoken_text}
        )
        self._emit(job_id, config, status="RUNNING", current_text=spoken_text, sample_id=sample["id"])
        return AttemptWork(sample, attempt_id, settings, seed, spoken_text)

    def _save_generated(self, work: AttemptWork, result: AudioResult, config: ProjectConfig) -> None:
        sample, attempt_id = work.sample, work.attempt_id
        work.synthesis_seconds = result.elapsed_seconds
        work.raw_metrics = audio_metrics(result.samples, result.sample_rate)
        original_path = None
        if config.keep_original:
            original_path = self.repo.root / "original" / f"{sample['id']}_{attempt_id}.wav"
            write_wav_atomic(original_path, result.samples, result.sample_rate, subtype="FLOAT")
            work.fields["original_path"] = str(original_path)
        processed = process_audio(
            result.samples,
            result.sample_rate,
            target_rate=config.audio_sample_rate,
            trim=config.quality.trim_silence,
            margin_ms=config.quality.trim_margin_ms,
        )
        path = self.repo.root / "audio" / f"{sample['id']}_{attempt_id}.wav"
        digest = write_wav_atomic(path, processed, config.audio_sample_rate, subtype=config.audio_subtype)
        work.audio_fingerprint = fingerprint(processed, config.audio_sample_rate)
        work.fields = dict(
            path=str(path),
            original_path=str(original_path) if original_path else None,
            sha256=digest,
            duration=len(processed) / config.audio_sample_rate,
            model_id=result.model_id,
            parameters={**result.parameters, "seed": work.seed, "spoken_text": work.spoken_text},
        )
        self.repo.finish_attempt(attempt_id, state="GENERATED", **work.fields)
        self.repo.update_sample(
            sample["id"], state="GENERATED", audio_fingerprint=work.audio_fingerprint, **work.fields
        )
        self.repo.update_sample(sample["id"], state="VALIDATING")

    def _finish_checked(
        self, job_id: str, config: ProjectConfig, work: AttemptWork, checked: QualityResult
    ) -> str:
        sample = work.sample
        reasons = list(checked.reasons)
        if self.repo.find_audio_duplicate(work.audio_fingerprint, sample["id"]):
            reasons.append("duplicate_audio")
        if work.raw_metrics["clipping_ratio"] > config.quality.max_clipping_ratio:
            reasons.append("source_clipping")
        metrics = {
            **checked.metrics,
            "source_audio": work.raw_metrics,
            "synthesis_seconds": work.synthesis_seconds,
            "needs_review": checked.needs_review,
        }
        passed = checked.passed and not reasons
        state = "ACCEPTED" if passed else "REJECTED"
        self.repo.update_sample(
            sample["id"],
            state=state,
            metrics=metrics,
            reasons=reasons,
            asr_text=checked.asr_text,
            wer=checked.wer,
            cer=checked.cer,
            speaker_similarity=checked.speaker_similarity,
            validated_revision=sample["training_revision"] if passed else -1,
        )
        self.repo.finish_attempt(
            work.attempt_id, state=state, metrics=metrics, reasons=reasons, **work.fields
        )
        if not passed and not checked.needs_review and sample["attempt_count"] < config.max_retries:
            self.repo.update_sample(sample["id"], state="PENDING")
        self._emit(
            job_id, config, status="RUNNING", audio_path=work.fields["path"], current_text=work.spoken_text
        )
        return state.lower()

    def _fail_work(self, job_id: str, config: ProjectConfig, work: AttemptWork, exc: BaseException) -> str:
        error = f"{type(exc).__name__}: {exc}"
        self.repo.finish_attempt(work.attempt_id, state="FAILED", error=error, reasons=[error], **work.fields)
        self.repo.update_sample(work.sample["id"], state="FAILED", reasons=[error])
        self.repo.record_event(
            "attempt_failed", {"sample_id": work.sample["id"], "attempt_id": work.attempt_id, "error": error}
        )
        oom = "out of memory" in str(exc).lower() or isinstance(exc, MemoryError)
        if not oom and work.sample["attempt_count"] < config.max_retries:
            self.repo.update_sample(work.sample["id"], state="PENDING")
        self._emit(job_id, config, error=error, status="RUNNING")
        return "oom" if oom else "failed"

    def _attempt(
        self,
        job_id: str,
        config: ProjectConfig,
        sample: dict[str, Any],
        backend: TTSBackend,
        checker: Any,
        total_attempts: int,
    ) -> str:
        work = self._begin_work(job_id, config, sample, total_attempts)
        try:
            result = backend.synthesize(work.spoken_text, config.voice, work.settings, work.seed)
            self._save_generated(work, result, config)
            checked = checker.check(
                Path(work.fields["path"]), sample["training_text"], sample["language"], config.voice
            )
            return self._finish_checked(job_id, config, work, checked)
        except Exception as exc:
            return self._fail_work(job_id, config, work, exc)

from __future__ import annotations

import time
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .audio import audio_metrics, read_audio
from .domain import QualityResult, QualitySettings, VoiceProfile
from .transcript import COMPARISON_NORMALIZATION_VERSION, transcript_metrics

ASR_CPU_THREADS = 4


@dataclass(frozen=True)
class ASRResult:
    text: str
    model_id: str
    language: str = "en"
    language_probability: float = 1.0
    avg_logprob: float = 0.0
    no_speech_probability: float = 0.0
    elapsed_seconds: float = 0.0


class ASRBackend(Protocol):
    def transcribe(self, path: Path, language: str) -> ASRResult: ...


class MockASR:
    """Explicit test transcripts; does not recognize audio."""

    def __init__(
        self, transcripts: Mapping[str, str | ASRResult] | Callable[[Path], str | ASRResult]
    ) -> None:
        self.transcripts = transcripts

    def transcribe(self, path: Path, language: str) -> ASRResult:
        value = self.transcripts(path) if callable(self.transcripts) else self.transcripts[str(path)]
        return (
            value if isinstance(value, ASRResult) else ASRResult(value, "MOCK-ASR-NOT-RECOGNITION", language)
        )


class FasterWhisperASR:
    def __init__(self, settings: QualitySettings, *, num_workers: int = 1) -> None:
        if not 1 <= num_workers <= 4:
            raise ValueError("Choose 1–4 ASR workers")
        self.settings = settings
        self._model: Any = None
        self._num_workers = num_workers
        self._load_lock = threading.Lock()
        if not settings.asr_model_path or not Path(settings.asr_model_path).is_dir():
            raise ValueError(
                "Choose a complete local faster-whisper model folder; automatic downloads are disabled"
            )
        required = ("model.bin", "config.json", "tokenizer.json")
        missing = [name for name in required if not (Path(settings.asr_model_path) / name).is_file()]
        if missing:
            raise ValueError(
                f"Incomplete local ASR model: missing {', '.join(missing)}; tokenizer fallback downloads are disabled"
            )
        import hashlib
        from .audio import sha256_file

        digest = hashlib.sha256()
        for name in required:
            digest.update(name.encode())
            digest.update(sha256_file(Path(settings.asr_model_path) / name).encode())
        self._model_hash = digest.hexdigest()

    def _load_model(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError("Install faster-whisper using scripts/setup.ps1 -Models") from exc
            self._model = WhisperModel(
                self.settings.asr_model_path,
                device=self.settings.asr_device,
                compute_type=self.settings.asr_compute_type,
                local_files_only=True,
                cpu_threads=ASR_CPU_THREADS,
                num_workers=self._num_workers,
            )

    def transcribe(self, path: Path, language: str) -> ASRResult:
        start = time.perf_counter()
        self._load_model()
        segments, info = self._model.transcribe(
            str(path),
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments).strip()
        import importlib.metadata

        identity = f"faster-whisper/{importlib.metadata.version('faster-whisper')}/sha256:{self._model_hash}"
        return ASRResult(
            text,
            identity,
            info.language,
            float(info.language_probability),
            float(np.mean([s.avg_logprob for s in segments])) if segments else -10.0,
            max((s.no_speech_prob for s in segments), default=1.0),
            time.perf_counter() - start,
        )


def repeated_audio(samples: np.ndarray, sr: int) -> float:
    """Conservative repeated-phrase proxy over 0.8-second energy/spectral sketches."""
    window = max(1, int(sr * 0.8))
    if len(samples) < window * 4:
        return 0.0
    chunks = samples[: len(samples) // window * window].reshape(-1, window)
    # A literal repeated block is suspicious. Spectral resemblance alone is not speaker-safe.
    scores = []
    for i in range(len(chunks)):
        for j in range(i + 2, len(chunks)):
            a, b = chunks[i].astype(np.float64), chunks[j].astype(np.float64)
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denominator > 0.01:
                scores.append(float(np.dot(a, b) / denominator))
    return max(scores, default=0.0)


class QualityController:
    def __init__(self, settings: QualitySettings, asr: ASRBackend | None = None, speaker: Any = None) -> None:
        self.settings, self.asr, self.speaker = settings, asr, speaker

    def prepare(self, voice: VoiceProfile) -> None:
        self.settings.validate()
        if self.settings.verify_asr and self.asr is None:
            self.asr = FasterWhisperASR(self.settings)
        if self.settings.speaker_enabled and self.speaker is None:
            from .speaker import SpeakerVerifier

            self.speaker = SpeakerVerifier(
                self.settings.speaker_model_path, self.settings.speaker_calibration_path, voice
            )

    def check(self, path: Path, reference_text: str, language: str, voice: VoiceProfile) -> QualityResult:
        try:
            data, sr = read_audio(path)
            metrics = audio_metrics(data, sr)
        except (OSError, ValueError, RuntimeError) as exc:
            return QualityResult(False, {"audio_error": str(exc)}, ["invalid_audio"])
        reasons = []
        s = self.settings
        for failed, reason in (
            (not s.min_duration <= metrics["duration"] <= s.max_duration, "duration"),
            (metrics["clipping_ratio"] > s.max_clipping_ratio, "clipping"),
            (metrics["silence_ratio"] > s.max_silence_ratio, "excessive_silence"),
            (metrics["rms"] < s.min_rms, "low_signal"),
            (s.check_abrupt_edges and bool(metrics["abrupt_edges"]), "abrupt_cutoff"),
        ):
            if failed:
                reasons.append(reason)
        metrics["asr_check"] = "disabled" if not s.verify_asr else "not_run"
        metrics["speaker_check"] = "disabled" if not s.speaker_enabled else "not_run"
        metrics["thresholds"] = asdict(s)
        if s.check_repetition:
            repetition = repeated_audio(data, sr)
            metrics["repeated_audio_correlation"] = repetition
            if repetition > 0.9995:
                reasons.append("repeated_audio")
        asr_text, wer, cer, similarity, needs_review = "", None, None, None, False
        if s.verify_asr:
            try:
                if self.asr is None:
                    self.prepare(voice)
                asr = self.asr.transcribe(Path(path), language)
                asr_text = asr.text
                comparison = transcript_metrics(reference_text, asr_text, language)
                wer, cer = comparison["wer"], comparison["cer"]
                metrics.update(comparison)
                metrics["asr"] = asdict(asr)
                metrics["comparison_normalization"] = COMPARISON_NORMALIZATION_VERSION
                metrics["asr_check"] = "passed" if wer <= s.max_wer and cer <= s.max_cer else "failed"
                if wer > s.max_wer:
                    reasons.append("transcript_wer")
                if cer > s.max_cer:
                    reasons.append("transcript_cer")
                if comparison["repeated_phrase"]:
                    reasons.append("repeated_speech")
                if (
                    asr.language != language
                    or asr.language_probability < 0.5
                    or asr.avg_logprob < -1.2
                    or asr.no_speech_probability > 0.6
                ):
                    needs_review = True
                    reasons.append("asr_unreliable_review_required")
            except Exception as exc:
                needs_review = True
                reasons.append("asr_error_review_required")
                metrics["asr_error"] = str(exc)
        if s.speaker_enabled:
            try:
                if self.speaker is None:
                    self.prepare(voice)
                similarity, threshold = self.speaker.compare(Path(path))
                metrics["speaker_threshold"] = threshold
                metrics["speaker_model"] = self.speaker.model_id
                metrics["speaker_check"] = "passed" if similarity >= threshold else "failed"
                if similarity < threshold:
                    reasons.append("speaker_mismatch")
            except Exception as exc:
                needs_review = True
                reasons.append("speaker_error_review_required")
                metrics["speaker_error"] = str(exc)
        return QualityResult(not reasons, metrics, reasons, asr_text, wer, cer, similarity, needs_review)


def revalidate_sample(repo, sample_id: str, checker: QualityController, voice: VoiceProfile) -> QualityResult:
    """Recheck edited training text against immutable audio, preserving synthesis provenance."""
    from .audio import sha256_file

    with repo.writer_lock():
        sample = repo.get_sample(sample_id)
        if sample["state"] in {"GENERATING", "GENERATED", "VALIDATING", "DELETED"}:
            raise ValueError("Cannot revalidate an active/deleted sample")
        path = repo._safe_audio(sample["path"])
        if sha256_file(path) != sample["sha256"]:
            raise ValueError("Audio hash mismatch; repair the recording first")
        checker.prepare(voice)
        result = checker.check(path, sample["training_text"], sample["language"], voice)
        repo.update_sample(
            sample_id,
            state="ACCEPTED" if result.passed else "REJECTED",
            metrics=result.metrics,
            reasons=result.reasons,
            asr_text=result.asr_text,
            wer=result.wer,
            cer=result.cer,
            speaker_similarity=result.speaker_similarity,
            validated_revision=sample["training_revision"] if result.passed else -1,
        )
        repo.record_event(
            "sample_revalidated",
            {
                "sample_id": sample_id,
                "passed": result.passed,
                "training_revision": sample["training_revision"],
            },
        )
        return result

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np


class SampleState(StrEnum):
    PENDING = "PENDING"
    GENERATING = "GENERATING"
    GENERATED = "GENERATED"
    VALIDATING = "VALIDATING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    DELETED = "DELETED"


@dataclass(frozen=True)
class VoiceProfile:
    id: str
    name: str
    reference_path: str
    reference_sha256: str
    language: str = "en"
    permission_confirmed: bool = False
    source: str = "reference-wav"
    model_version: str = ""
    synthesis_settings: dict[str, Any] = field(default_factory=dict)
    service_voice_id: str = ""

    def validate(self) -> None:
        from .audio import read_audio, sha256_file

        if not self.permission_confirmed:
            raise ValueError("Confirm that you own or have permission to use this voice.")
        path = Path(self.reference_path)
        if not path.is_file():
            raise FileNotFoundError(f"Voice reference is missing: {path}")
        if sha256_file(path) != self.reference_sha256:
            raise ValueError("Voice reference hash changed. Import and confirm the reference again.")
        read_audio(path)
        if self.language not in {"en", "sv"}:
            raise ValueError("Choose English (en) or Swedish (sv).")


@dataclass(frozen=True)
class SynthesisSettings:
    engine: str = "multilingual-v3"
    device: str = "cuda"
    model_dir: str = ""
    temperature: float = 0.8
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    repetition_penalty: float = 1.2
    top_p: float = 1.0
    min_p: float = 0.05
    seed: int = 0
    variation: float = 0.0
    service_config: str = ""

    def validate(self) -> None:
        if self.engine not in {"english", "multilingual-v2", "multilingual-v3", "mock", "documented-service"}:
            raise ValueError(f"Unsupported TTS engine: {self.engine}")
        for key in (
            "temperature",
            "exaggeration",
            "cfg_weight",
            "repetition_penalty",
            "top_p",
            "min_p",
            "variation",
        ):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        if not 0.01 <= self.temperature <= 2 or not 0 <= self.exaggeration <= 2:
            raise ValueError("Temperature/exaggeration outside supported range")
        if not 0 <= self.cfg_weight <= 2 or not 1 <= self.repetition_penalty <= 3:
            raise ValueError("CFG weight/repetition penalty outside supported range")
        if not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1 or not 0 <= self.variation <= 0.1:
            raise ValueError("Invalid probability or variation (maximum 0.1)")


@dataclass(frozen=True)
class QualitySettings:
    verify_asr: bool = True
    asr_model_path: str = ""
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    max_wer: float = 0.10
    max_cer: float = 0.10
    min_duration: float = 3.0
    max_duration: float = 15.0
    max_silence_ratio: float = 0.5
    min_rms: float = 0.003
    max_clipping_ratio: float = 0.001
    trim_silence: bool = False
    trim_margin_ms: int = 120
    speaker_enabled: bool = False
    speaker_model_path: str = ""
    speaker_calibration_path: str = ""
    check_abrupt_edges: bool = True
    check_repetition: bool = True

    def validate(self) -> None:
        values = (
            self.min_duration,
            self.max_duration,
            self.max_wer,
            self.max_cer,
            self.max_silence_ratio,
            self.min_rms,
            self.max_clipping_ratio,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Quality thresholds must be finite")
        if not 0 < self.min_duration < self.max_duration <= 120:
            raise ValueError("Clip duration must be positive, ordered, and at most 120 seconds")
        if not all(
            0 <= v <= 1 for v in (self.max_wer, self.max_cer, self.max_silence_ratio, self.max_clipping_ratio)
        ):
            raise ValueError("Quality ratios must be in [0, 1]")
        if self.min_rms < 0 or not 0 <= self.trim_margin_ms <= 2000:
            raise ValueError("Invalid RMS or trimming margin")


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    voice: VoiceProfile
    language: str = "en"
    synthesis: SynthesisSettings = field(default_factory=SynthesisSettings)
    quality: QualitySettings = field(default_factory=QualitySettings)
    target_count: int = 1000
    target_seconds: float = 0.0
    max_retries: int = 2
    max_attempts: int = 5000
    keep_original: bool = True
    min_words: int = 6
    max_words: int = 36
    category_weights: dict[str, float] = field(default_factory=dict)
    minimum_free_gb: float = 2.0
    minimum_gpu_free_gb: float = 0.0
    schema_version: int = 1
    text_method: str = "builtin"
    custom_topics: tuple[str, ...] = ()
    llm_endpoint: str = "http://127.0.0.1:1234/v1"
    llm_model: str = ""
    allow_remote_llm: bool = False
    audio_sample_rate: int = 24000
    audio_subtype: str = "PCM_16"
    generation_workers: int = 1
    overlap_quality: bool = False
    require_cuda: bool = False
    gpu_memory_per_worker_gb: float = 4.0
    auto_generation_workers: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProjectConfig:
        data = dict(value)
        data["voice"] = VoiceProfile(**data["voice"])
        data["synthesis"] = SynthesisSettings(**data.get("synthesis", {}))
        data["quality"] = QualitySettings(**data.get("quality", {}))
        data["custom_topics"] = tuple(data.get("custom_topics", ()))
        return cls(**data)

    def validate(self, check_reference: bool = True) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported project configuration version")
        if not self.name.strip() or self.language not in {"en", "sv"}:
            raise ValueError("Project name and supported language are required")
        if self.text_method not in {"builtin", "local-llm", "imported"}:
            raise ValueError("Unsupported text source")
        if self.audio_sample_rate not in {16000, 22050, 24000, 44100, 48000} or self.audio_subtype not in {
            "PCM_16",
            "PCM_24",
            "FLOAT",
        }:
            raise ValueError("Unsupported working WAV format")
        self.synthesis.validate()
        self.quality.validate()
        if not isinstance(self.auto_generation_workers, bool):
            raise ValueError("Automatic generation workers must be enabled or disabled")
        if (
            not isinstance(self.generation_workers, int)
            or isinstance(self.generation_workers, bool)
            or not 1 <= self.generation_workers <= 8
        ):
            raise ValueError("Choose 1–8 generation workers")
        native = self.synthesis.engine not in {"mock", "documented-service"}
        if (
            native
            and (self.require_cuda or self.auto_generation_workers or self.generation_workers > 1)
            and not self.synthesis.device.startswith("cuda")
        ):
            raise ValueError("GPU generation requires the CUDA synthesis device; CPU fallback is disabled")
        if self.generation_workers > 1 and self.synthesis.engine == "documented-service":
            raise ValueError(
                "Multiple workers require native GPU TTS; a service's GPU ownership cannot be verified"
            )
        if not math.isfinite(self.gpu_memory_per_worker_gb) or self.gpu_memory_per_worker_gb <= 0:
            raise ValueError("GPU memory budget per worker must be positive")
        if self.synthesis.engine == "english" and self.language != "en":
            raise ValueError("Swedish requires a multilingual engine")
        if self.voice.language != self.language:
            raise ValueError("Voice language must match the job language")
        if (
            not 0 <= self.target_count <= 1_000_000
            or not math.isfinite(self.target_seconds)
            or self.target_seconds < 0
        ):
            raise ValueError("Invalid accepted target")
        if self.target_count == 0 and self.target_seconds == 0:
            raise ValueError("Set an accepted-count or accepted-duration target")
        if not 0 <= self.max_retries <= 20 or not 1 <= self.max_attempts <= 10_000_000:
            raise ValueError("Invalid retry/attempt budget")
        if not 1 <= self.min_words <= self.max_words <= 200:
            raise ValueError("Invalid sentence word limits")
        if not math.isfinite(self.minimum_free_gb) or self.minimum_free_gb < 0:
            raise ValueError("Invalid free-disk minimum")
        if not math.isfinite(self.minimum_gpu_free_gb) or self.minimum_gpu_free_gb < 0:
            raise ValueError("Invalid free-GPU-memory minimum")
        if check_reference:
            self.voice.validate()


@dataclass(frozen=True)
class TextItem:
    original_text: str
    spoken_text: str
    category: str = "custom"
    language: str = "en"
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioResult:
    samples: np.ndarray
    sample_rate: int
    model_id: str
    parameters: dict[str, Any]
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    metrics: dict[str, Any]
    reasons: list[str]
    asr_text: str = ""
    wer: float | None = None
    cer: float | None = None
    speaker_similarity: float | None = None
    needs_review: bool = False

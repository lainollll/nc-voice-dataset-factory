"""Local model discovery and simple dataset preparation, called from a worker."""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .audio import sha256_file
from .domain import ProjectConfig, QualitySettings, SynthesisSettings, TextItem, VoiceProfile
from .models import PATTERNS
from .storage import Repository

APPLICATION_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class QuickStartRequest:
    reference: Path
    language: str
    target_count: int
    target_minutes: float
    permission_confirmed: bool


@dataclass(frozen=True)
class LocalModels:
    engine: str
    tts_path: Path | None
    asr_path: Path | None

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(
            kind for kind, path in ((self.engine, self.tts_path), ("asr", self.asr_path)) if path is None
        )


@dataclass(frozen=True)
class PreparedDataset:
    repo: Repository
    config: ProjectConfig
    items: list[TextItem]


def _complete_model(path: Path, kind: str) -> bool:
    # Built-in voice conditioning is optional when a reference WAV is supplied.
    patterns = [pattern for pattern in PATTERNS[kind] if pattern != "conds.pt"]
    if kind == "asr":
        patterns = ["model.bin", "config.json", "tokenizer.json"]
    try:
        return all(
            any(p.is_file() and p.stat().st_size > 0 for p in path.glob(pattern)) for pattern in patterns
        )
    except OSError:
        return False


def _compatible_asr(path: Path, language: str) -> bool:
    if not _complete_model(path, "asr"):
        return False
    try:
        config_path = path / "config.json"
        if config_path.stat().st_size > 1_000_000:
            return False
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            return False
        if language == "en":
            return True
        if config.get("is_multilingual") is False or any(part.lower().endswith(".en") for part in path.parts):
            return False
        # CTranslate2's multilingual Whisper config includes its language token IDs.
        language_ids = config.get("lang_ids", [])
        if isinstance(language_ids, list) and len(language_ids) > 1:
            return True
        tokenizer_path = path / "tokenizer.json"
        if tokenizer_path.stat().st_size > 10_000_000:
            return False
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        return "<|sv|>" in tokenizer.get("model", {}).get("vocab", {}) or any(
            token.get("content") == "<|sv|>" for token in tokenizer.get("added_tokens", [])
        )
    except (OSError, ValueError, AttributeError, TypeError):
        return False


def discover_models(
    app_root: Path,
    language: str,
    *,
    preferred: tuple[Path, ...] = (),
    cache_root: Path | None = None,
) -> LocalModels:
    """Read known local locations only. Never load models or access the network."""
    if language not in {"en", "sv"}:
        raise ValueError("Choose English or Swedish")
    candidates = list(preferred)
    for config_path in sorted((app_root / "projects").glob("*/project_config.json")):
        try:
            if config_path.stat().st_size > 1_000_000:
                continue
            data = json.loads(config_path.read_text(encoding="utf-8"))
            for section, key in (("synthesis", "model_dir"), ("quality", "asr_model_path")):
                value = data.get(section, {}).get(key)
                if isinstance(value, str) and value.strip():
                    path = Path(value)
                    candidates.append(path if path.is_absolute() else config_path.parent / path)
        except (OSError, ValueError, AttributeError, TypeError):
            continue  # Old or damaged projects must not prevent a new setup.
    candidates.extend(sorted(path for path in (app_root / "models").glob("*") if path.is_dir()))
    if cache_root is None:
        hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        cache_root = Path(os.environ.get("HF_HUB_CACHE", str(hf_home / "hub")))
    candidates.extend(sorted((cache_root / "models--ResembleAI--chatterbox" / "snapshots").glob("*")))
    # Prefer the already validated medium recognizer when present.
    for name in ("models--Systran--faster-whisper-medium", "models--Systran--faster-whisper-*"):
        candidates.extend(sorted(cache_root.glob(f"{name}/snapshots/*")))
    candidates = list(dict.fromkeys(candidates))
    engines = (
        ("english", "multilingual-v3", "multilingual-v2")
        if language == "en"
        else ("multilingual-v3", "multilingual-v2")
    )
    asr = next((path for path in candidates if _compatible_asr(path, language)), None)
    for engine in engines:
        tts = next((path for path in candidates if _complete_model(path, engine)), None)
        if tts is not None:
            return LocalModels(engine, tts, asr)
    return LocalModels(engines[0], None, asr)


def prepare_dataset(
    request: QuickStartRequest,
    app_root: Path = APPLICATION_ROOT,
    *,
    preferred: tuple[Path, ...] = (),
    cache_root: Path | None = None,
    cancel: threading.Event | None = None,
) -> PreparedDataset:
    """Validate and save a new project; synthesis is left to the existing UI pipeline."""
    from .diagnostics import gpu_status
    from .text import generate_texts

    def check_cancelled() -> None:
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Dataset setup cancelled")

    check_cancelled()
    if not request.permission_confirmed:
        raise ValueError("Confirm that you own or have permission to use this voice.")
    if request.reference.suffix.lower() != ".wav":
        raise ValueError("Choose a reference WAV recording")
    if (
        not 0 <= request.target_count <= 1_000_000
        or not math.isfinite(request.target_minutes)
        or not 0 <= request.target_minutes <= 100_000
        or (request.target_count > 0) == (request.target_minutes > 0)
    ):
        raise ValueError("Choose a positive recording count or a duration in minutes")
    reference = request.reference.resolve()
    voice = VoiceProfile(
        "voice-" + uuid.uuid5(uuid.NAMESPACE_URL, str(reference) + ":" + request.language).hex,
        reference.stem,
        str(reference),
        sha256_file(reference),
        request.language,
        request.permission_confirmed,
    )
    voice.validate()
    models = discover_models(app_root, request.language, preferred=preferred, cache_root=cache_root)
    if models.missing:
        raise ValueError(
            "Missing local models: "
            + ", ".join(models.missing)
            + ". Use Inspect missing models here to install the matching files automatically."
        )
    check_cancelled()
    gpu = gpu_status()
    if not gpu.get("cuda_available"):
        raise RuntimeError("GPU generation is unavailable. Check the NVIDIA driver and model dependencies.")
    free = (gpu.get("torch_mem_get_info") or {}).get("free_bytes", 0) / 1024**3
    physical = gpu.get("nvidia_smi") or []
    if physical and physical[0].get("memory_free_mb") is not None:
        free = min(free, physical[0]["memory_free_mb"] / 1024)
    if free < 5:
        raise RuntimeError(
            "Not enough free GPU memory. Free at least 5 GiB, then try Generate dataset again."
        )
    synthesis = SynthesisSettings(
        engine=models.engine, device="cuda", model_dir=str(models.tts_path), seed=100
    )
    voice = replace(voice, model_version=models.engine, synthesis_settings=asdict(synthesis))
    # Budget duration targets for clips at the minimum accepted length, plus retries.
    expected_count = request.target_count or math.ceil(request.target_minutes * 60 / 3)
    config = ProjectConfig(
        name=reference.stem,
        voice=voice,
        language=request.language,
        synthesis=synthesis,
        quality=QualitySettings(asr_model_path=str(models.asr_path)),
        target_count=request.target_count,
        target_seconds=request.target_minutes * 60,
        max_attempts=min(10_000_000, max(50, expected_count * 5)),
        generation_workers=8,
        auto_generation_workers=True,
        overlap_quality=True,
        require_cuda=True,
    )
    config.validate()
    items = generate_texts(min(32, request.target_count or 32), request.language, seed=synthesis.seed)
    check_cancelled()
    # A unique directory avoids overwriting or extending any existing voice/job.
    project_root = app_root / "projects" / f"dataset-{uuid.uuid4().hex}"
    project_root.mkdir(parents=True, exist_ok=False)
    repo = Repository(project_root)
    from .logging_config import configure_logging

    configure_logging(repo.root)
    repo.save_voice(voice)
    repo.save_config(config)
    return PreparedDataset(repo, config, items)

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol
from unittest.mock import patch

import numpy as np

from .audio import mono, sha256_file
from .domain import AudioResult, SynthesisSettings, VoiceProfile

GENERATE_PARAMS = ("temperature", "exaggeration", "cfg_weight", "repetition_penalty", "top_p", "min_p")


class TTSBackend(Protocol):
    def synthesize(
        self, text: str, voice: VoiceProfile, settings: SynthesisSettings, seed: int
    ) -> AudioResult: ...
    def close(self) -> None: ...


class MockTTS:
    """Deterministic test tones; never represents cloned speech."""

    def synthesize(
        self, text: str, voice: VoiceProfile, settings: SynthesisSettings, seed: int
    ) -> AudioResult:
        voice.validate()
        if not text.strip():
            raise ValueError("Cannot synthesize empty text")
        start = time.perf_counter()
        key = hashlib.sha256(f"{text}:{seed}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
        t = np.arange(84000, dtype=np.float64) / 24000
        f0 = 110 + int.from_bytes(key[8:10], "little") % 180
        signal = 0.16 * np.sin(2 * np.pi * f0 * t) * (0.65 + 0.35 * np.sin(2 * np.pi * 3.1 * t))
        signal += rng.normal(0, 0.001, len(t))
        fade = np.linspace(0, 1, 1200)
        signal[:1200] *= fade
        signal[-1200:] *= fade[::-1]
        return AudioResult(
            signal.astype(np.float32),
            24000,
            "MOCK-tone-v1-NOT-SPEECH",
            {**asdict(settings), "seed": seed, "mock": True},
            time.perf_counter() - start,
        )

    def close(self) -> None:
        pass


class NativeChatterbox:
    """Instance-owned offline Chatterbox model. Call only from a background worker."""

    def __init__(self, settings: SynthesisSettings) -> None:
        settings.validate()
        if not settings.model_dir or not Path(settings.model_dir).is_dir():
            raise FileNotFoundError(
                "Choose an existing local Chatterbox model directory; implicit downloads are disabled"
            )
        self.settings = settings
        self._model: Any = None
        self._torch: Any = None
        self._identity = ""

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            self._torch = importlib.import_module("torch")
            module_name = "chatterbox.tts" if self.settings.engine == "english" else "chatterbox.mtl_tts"
            cls_name = "ChatterboxTTS" if self.settings.engine == "english" else "ChatterboxMultilingualTTS"
            cls = getattr(importlib.import_module(module_name), cls_name)
        except ImportError as exc:
            raise RuntimeError("Native TTS dependencies are missing. Run scripts/setup.ps1 -Models.") from exc
        if self.settings.device.startswith("cuda") and not self._torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and PyTorch CUDA installation.")
        kwargs: dict[str, Any] = {"device": self.settings.device}
        if self.settings.engine.startswith("multilingual"):
            if "t3_model" not in inspect.signature(cls.from_local).parameters:
                raise RuntimeError(
                    "Installed Chatterbox lacks explicit multilingual checkpoint selection; install the pinned revision"
                )
            kwargs["t3_model"] = self.settings.engine.rsplit("-", 1)[-1]
        root = Path(self.settings.model_dir)
        if self.settings.engine.startswith("multilingual"):
            # Upstream MTLTokenizer requests Cangjie via hf_hub_download even in from_local.
            # Limit that one constructor dependency to the explicitly selected local file.
            tokenizer = importlib.import_module("chatterbox.models.tokenizers.tokenizer")

            def local_mapping(*args: Any, **fetch: Any) -> str:
                if (
                    fetch.get("repo_id") != "ResembleAI/chatterbox"
                    or fetch.get("filename") != "Cangjie5_TC.json"
                ):
                    raise RuntimeError("Unexpected model fetch blocked by offline adapter")
                path = root / "Cangjie5_TC.json"
                if not path.is_file():
                    raise FileNotFoundError("Local multilingual model is missing Cangjie5_TC.json")
                return str(path)

            if not (root / "Cangjie5_TC.json").is_file():
                raise FileNotFoundError(
                    "Local multilingual model is missing Cangjie5_TC.json; use explicit model download"
                )
            with patch.object(tokenizer, "hf_hub_download", local_mapping):
                self._model = cls.from_local(root, **kwargs)
        else:
            self._model = cls.from_local(root, **kwargs)
        if self.settings.device.startswith("cuda"):
            for name in ("t3", "s3gen", "ve"):
                module = getattr(self._model, name)
                if any(parameter.device.type != "cuda" for parameter in module.parameters()):
                    self._model = None
                    raise RuntimeError("GPU model placement failed; CPU fallback is disabled")
        supported = inspect.signature(self._model.generate).parameters
        required = {*GENERATE_PARAMS, "audio_prompt_path"}
        if self.settings.engine != "english":
            required.add("language_id")
        if not required.issubset(supported):
            self._model = None
            raise RuntimeError(
                f"Incompatible Chatterbox generate signature; missing {sorted(required - set(supported))}"
            )
        weights = sorted(p for p in root.iterdir() if p.suffix in {".pt", ".safetensors", ".json"})
        digest = hashlib.sha256()
        for path in weights:
            digest.update(path.name.encode())
            digest.update(sha256_file(path).encode())
        try:
            version = importlib.metadata.version("chatterbox-tts")
        except importlib.metadata.PackageNotFoundError:
            version = "source"
        self._identity = f"chatterbox/{version}/{self.settings.engine}/sha256:{digest.hexdigest()}"

    def synthesize(
        self, text: str, voice: VoiceProfile, settings: SynthesisSettings, seed: int
    ) -> AudioResult:
        voice.validate()
        settings.validate()
        if not text.strip():
            raise ValueError("Cannot synthesize empty text")
        if (settings.engine, settings.device, settings.model_dir) != (
            self.settings.engine,
            self.settings.device,
            self.settings.model_dir,
        ):
            raise ValueError("A loaded backend cannot change model/device. Start a new job.")
        if settings.engine == "english" and voice.language != "en":
            raise ValueError("English Chatterbox cannot generate Swedish")
        start = time.perf_counter()
        try:
            self._load()
            random.seed(seed)
            np.random.seed(seed % (2**32))
            self._torch.manual_seed(seed)
            if self._torch.cuda.is_available():
                self._torch.cuda.manual_seed_all(seed)
            parameters = {key: getattr(settings, key) for key in GENERATE_PARAMS}
            if settings.engine != "english":
                parameters["language_id"] = voice.language
            inference_started = time.time()
            with self._torch.inference_mode():
                result = self._model.generate(text, audio_prompt_path=voice.reference_path, **parameters)
            data = result.detach().cpu().numpy().squeeze()
            return AudioResult(
                mono(data),
                int(self._model.sr),
                self._identity,
                {
                    **parameters,
                    "seed": seed,
                    "inference_started_at": inference_started,
                    "inference_finished_at": time.time(),
                    "inference_device": str(next(self._model.t3.parameters()).device)
                    if settings.device.startswith("cuda")
                    else settings.device,
                    "gpu_name": self._torch.cuda.get_device_name(settings.device)
                    if settings.device.startswith("cuda")
                    else None,
                },
                time.perf_counter() - start,
            )
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                if self._torch is not None and self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA out of memory. Job paused safely; free GPU memory or change configuration."
                ) from exc
            raise

    def close(self) -> None:
        self._model = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def create_tts(settings: SynthesisSettings) -> TTSBackend:
    settings.validate()
    if settings.engine == "documented-service":
        from .service import DocumentedServiceTTS

        return DocumentedServiceTTS(settings)
    return MockTTS() if settings.engine == "mock" else NativeChatterbox(settings)


def capability_report() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in ("chatterbox-tts", "torch", "faster-whisper", "PySide6", "numpy", "soundfile"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "versions": versions,
        "engines": ["english", "multilingual-v2", "multilingual-v3", "mock"],
        "service": {
            "available": False,
            "reason": "No documented connected Chatterbox service was configured",
        },
    }

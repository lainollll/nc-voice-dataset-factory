"""Import-safe spawn worker. No Qt, project storage or CUDA tensor IPC."""

from __future__ import annotations

import atexit
import os
import time
from dataclasses import replace

from .domain import AudioResult, SynthesisSettings, VoiceProfile
from .tts import TTSBackend, create_tts

_backend: TTSBackend | None = None
CPU_THREADS_PER_GENERATOR = 2


def initialize(settings: SynthesisSettings) -> None:
    global _backend
    if settings.engine not in {"mock", "documented-service"}:
        import torch

        torch.set_num_threads(CPU_THREADS_PER_GENERATOR)
    _backend = create_tts(settings)
    atexit.register(_backend.close)


def generate(text: str, voice: VoiceProfile, settings: SynthesisSettings, seed: int) -> AudioResult:
    if _backend is None:
        raise RuntimeError("Generation worker was not initialized")
    started = time.time()
    result = _backend.synthesize(text, voice, settings, seed)
    return replace(
        result,
        parameters={
            **result.parameters,
            "worker_pid": os.getpid(),
            "worker_started_at": started,
            "worker_finished_at": time.time(),
        },
    )

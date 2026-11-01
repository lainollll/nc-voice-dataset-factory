from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    path = tmp_path / "reference.wav"
    t = np.arange(48000) / 24000
    sf.write(path, (0.1 * np.sin(2 * np.pi * 170 * t)).astype("float32"), 24000)
    return path


@pytest.fixture
def config(reference: Path):
    from nc_factory.audio import sha256_file
    from nc_factory.domain import ProjectConfig, QualitySettings, SynthesisSettings, VoiceProfile

    return ProjectConfig(
        name="Test project",
        voice=VoiceProfile(
            "test-voice", "Test voice", str(reference), sha256_file(reference), permission_confirmed=True
        ),
        synthesis=SynthesisSettings(engine="mock", device="cpu"),
        quality=QualitySettings(verify_asr=False, min_duration=0.2, max_duration=15),
        target_count=10,
        max_attempts=50,
        minimum_free_gb=0,
    )

"""Create an explicit real-backend configuration from existing local model files."""

from __future__ import annotations

import argparse
from pathlib import Path

from nc_factory.audio import sha256_file
from nc_factory.domain import ProjectConfig, QualitySettings, SynthesisSettings, VoiceProfile
from nc_factory.storage import atomic_json

parser = argparse.ArgumentParser()
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--tts-model", type=Path, required=True)
parser.add_argument("--asr-model", type=Path, required=True)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--target", type=int, default=10)
parser.add_argument("--language", choices=["en", "sv"], default="en")
parser.add_argument("--engine", choices=["english", "multilingual-v2", "multilingual-v3"], default="english")
parser.add_argument("--permission-confirmed", action="store_true", required=True)
args = parser.parse_args()
digest = sha256_file(args.reference)
voice = VoiceProfile(
    "voice-" + digest[:16],
    args.reference.stem,
    str(args.reference.resolve()),
    digest,
    args.language,
    args.permission_confirmed,
)
config = ProjectConfig(
    "English real-backend validation",
    voice,
    args.language,
    synthesis=SynthesisSettings(
        engine=args.engine, device="cuda", model_dir=str(args.tts_model.resolve()), seed=100
    ),
    quality=QualitySettings(asr_model_path=str(args.asr_model.resolve())),
    target_count=args.target,
    max_attempts=max(args.target * 5, 50),
    minimum_free_gb=2,
)
config.validate()
atomic_json(args.config.resolve(), config.to_dict())
print(args.config.resolve())

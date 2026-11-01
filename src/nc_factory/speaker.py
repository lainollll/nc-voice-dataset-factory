from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .audio import process_audio, read_audio, sha256_file
from .domain import VoiceProfile
from .storage import utcnow


def calibrate_scores(
    positive: list[float],
    negative: list[float],
    model_id: str,
    reference_hash: str,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(positive) < 2 or len(negative) < 2:
        raise ValueError("Calibration needs multiple independent same-speaker and other-speaker recordings")
    if not all(math.isfinite(v) and -1 <= v <= 1 for v in positive + negative):
        raise ValueError("Invalid cosine scores")
    lower_positive, upper_negative = min(positive), max(negative)
    if lower_positive <= upper_negative:
        raise ValueError("Calibration classes overlap; no separating threshold can be justified")
    return {
        "version": 1,
        "model_id": model_id,
        "reference_sha256": reference_hash,
        "threshold": (lower_positive + upper_negative) / 2,
        "positive_scores": positive,
        "negative_scores": negative,
        "provenance": provenance or {},
        "created_at": utcnow(),
        "method": "empirical separated calibration classes; held-out verification recommended",
    }


def load_calibration(path: Path, model_id: str, reference_hash: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        data.get("version") != 1
        or data.get("model_id") != model_id
        or data.get("reference_sha256") != reference_hash
    ):
        raise ValueError("Calibration does not match the selected model/reference")
    recomputed = calibrate_scores(data["positive_scores"], data["negative_scores"], model_id, reference_hash)
    if abs(recomputed["threshold"] - data["threshold"]) > 1e-9:
        raise ValueError("Calibration threshold was modified; recalibrate with labeled examples")
    return data


class SpeakerEmbedding:
    """Independent SpeechBrain ECAPA encoder from a complete, local model folder."""

    def __init__(self, model_path: str) -> None:
        root = Path(model_path)
        if not model_path or not (root / "hyperparams.yaml").is_file():
            raise ValueError("Choose a complete local SpeechBrain speaker model folder")
        import hashlib

        digest = hashlib.sha256()
        for path in sorted(p for p in root.iterdir() if p.is_file() and p.suffix in {".ckpt", ".yaml"}):
            digest.update(path.name.encode())
            digest.update(sha256_file(path).encode())
        self.model_id = "speechbrain-ecapa/sha256:" + digest.hexdigest()
        self.root, self._model = root, None

    def embed(self, path: Path) -> np.ndarray:
        import torch

        if self._model is None:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
                from hyperpyyaml import load_hyperpyyaml
            except ImportError as exc:
                raise RuntimeError("Install the pinned optional SpeechBrain 1.0.3 speaker runtime") from exc
            # Deliberately bypass from_hparams/collect_files: both can fetch remote files.
            # Bind every declared loadable to a checked local file, then use the pinned
            # Pretrainer loading hooks and EncoderClassifier constructor directly.
            with (self.root / "hyperparams.yaml").open(encoding="utf-8") as handle:
                hparams = load_hyperpyyaml(handle, {"pretrained_path": str(self.root)})
            pretrainer = hparams["pretrainer"]
            for name in pretrainer.loadables:
                declared = Path(pretrainer.paths[name])
                path = (self.root / declared.name).resolve()
                if not path.is_relative_to(self.root.resolve()) or not path.is_file():
                    raise ValueError(f"Incomplete local speaker model: {declared.name}")
                pretrainer.paths[name] = str(path)
                if name not in pretrainer.is_local:
                    pretrainer.is_local.append(name)
            pretrainer.load_collected()
            self._model = EncoderClassifier(
                modules=hparams["modules"], hparams=hparams, run_opts={"device": "cpu"}
            )
        data, sr = read_audio(path)
        data = process_audio(data, sr, target_rate=16000)
        with torch.inference_mode():
            embedding = self._model.encode_batch(torch.from_numpy(data).unsqueeze(0)).squeeze().cpu().numpy()
        length = float(np.linalg.norm(embedding))
        if length == 0 or not np.isfinite(embedding).all():
            raise ValueError("Invalid speaker embedding")
        return embedding / length


class SpeakerVerifier:
    def __init__(self, model_path: str, calibration_path: str, voice: VoiceProfile) -> None:
        self.embedding = SpeakerEmbedding(model_path)
        self.model_id = self.embedding.model_id
        self.calibration = load_calibration(Path(calibration_path), self.model_id, voice.reference_sha256)
        self.voice = voice
        self._reference: np.ndarray | None = None

    def compare(self, path: Path) -> tuple[float, float]:
        self.voice.validate()
        if self._reference is None:
            self._reference = self.embedding.embed(Path(self.voice.reference_path))
        return float(np.dot(self._reference, self.embedding.embed(path))), self.calibration["threshold"]


def calibrate_reference(
    model_path: str, voice: VoiceProfile, positives: list[Path], negatives: list[Path]
) -> dict[str, Any]:
    voice.validate()
    model = SpeakerEmbedding(model_path)
    reference = model.embed(Path(voice.reference_path))
    hashes = [[sha256_file(p) for p in group] for group in (positives, negatives)]
    if len(set(hashes[0] + hashes[1] + [voice.reference_sha256])) != len(positives) + len(negatives) + 1:
        raise ValueError("Calibration recordings must be distinct from each other and the reference")
    scores = [[float(np.dot(reference, model.embed(p))) for p in group] for group in (positives, negatives)]
    return calibrate_scores(
        scores[0],
        scores[1],
        model.model_id,
        voice.reference_sha256,
        {"positive_sha256": hashes[0], "negative_sha256": hashes[1]},
    )

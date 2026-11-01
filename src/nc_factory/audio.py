from __future__ import annotations

import hashlib
import os
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mono(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 2 and 1 <= data.shape[1] <= 8:
        data = data.mean(axis=1)
    if data.ndim != 1 or not data.size:
        raise ValueError("Empty audio or unsupported shape; expected samples or samples × channels")
    if not np.isfinite(data).all():
        raise ValueError("Audio contains non-finite NaN or infinite samples")
    return np.ascontiguousarray(data)


def read_audio(path: Path | str) -> tuple[np.ndarray, int]:
    info = sf.info(str(path))
    if info.frames == 0 or info.samplerate <= 0 or info.duration > 120:
        raise ValueError("Empty, invalid or excessively long audio (maximum 120 seconds)")
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return mono(data), int(rate)


def process_audio(
    samples: np.ndarray, sr: int, target_rate: int = 24000, trim: bool = False, margin_ms: int = 120
) -> np.ndarray:
    from scipy.signal import resample_poly

    data = mono(samples)
    if sr <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if sr != target_rate:
        divisor = gcd(sr, target_rate)
        data = resample_poly(data, target_rate // divisor, sr // divisor).astype(np.float32)
    if trim:
        active = np.flatnonzero(np.abs(data) > max(0.001, float(np.max(np.abs(data))) * 0.01))
        if active.size:
            margin = max(0, margin_ms) * target_rate // 1000
            data = data[max(0, int(active[0]) - margin) : min(len(data), int(active[-1]) + margin + 1)]
    return mono(data)


def write_wav_atomic(path: Path | str, samples: np.ndarray, sr: int, subtype: str = "PCM_16") -> str:
    data = mono(samples)
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".nc-audio-", suffix=".partial", dir=destination.parent)
    temporary = Path(name)
    try:
        os.close(fd)
        if subtype not in {"PCM_16", "PCM_24", "FLOAT"}:
            raise ValueError("Supported WAV subtypes: PCM_16, PCM_24, FLOAT")
        sf.write(str(temporary), data, sr, format="WAV", subtype=subtype)
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)  # Atomic publication; existing destination is never replaced.
        except FileExistsError:
            raise
        except OSError as exc:
            raise OSError("Atomic WAV publication requires NTFS or another hard-link capable volume") from exc
        return sha256_file(destination)
    finally:
        temporary.unlink(missing_ok=True)


def audio_metrics(samples: np.ndarray, sr: int) -> dict[str, float]:
    data = mono(samples)
    if sr <= 0:
        raise ValueError("Invalid sample rate")
    frame = max(1, sr // 50)
    padded = np.pad(data, (0, (-len(data)) % frame))
    rms_frames = np.sqrt(np.mean(padded.reshape(-1, frame).astype(np.float64) ** 2, axis=1))
    edge = max(abs(float(data[0])), abs(float(data[-1])))
    return {
        "duration": len(data) / sr,
        "rms": float(np.sqrt(np.mean(data.astype(np.float64) ** 2))),
        "peak": float(np.max(np.abs(data))),
        "silence_ratio": float(np.mean(rms_frames < 0.001)),
        "clipping_ratio": float(np.mean(np.abs(data) >= 0.999)),
        "abrupt_edges": float(edge > 0.05),
        "edge_amplitude": edge,
    }


def fingerprint(samples: np.ndarray, sr: int) -> str:
    data = process_audio(samples, sr)
    quantized = np.round(np.clip(data, -1, 1) * 32767).astype("<i2")
    return hashlib.sha256(quantized.tobytes()).hexdigest()

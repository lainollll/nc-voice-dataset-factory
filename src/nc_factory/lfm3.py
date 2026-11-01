"""Standalone loader for the inspected Liquid LFM3 TTS notebook.

Copied into exports; no dependency on the desktop app. Audio remains WAV bytes
with decode=False, and the official processor performs codec encoding.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any


def _jsonl_records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_records(records: list[dict[str, Any]], root: str):
    for record in records:
        data = (Path(root) / record["filename"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError(f"Audio hash mismatch: {record['filename']}")
        yield {
            "id": record["id"],
            "audio": {"bytes": data, "path": record["filename"]},
            "normalized_text": record["normalized_text"],
            "text": record["text"],
            "language": record["language"],
        }


def load_lfm3_dataset(root: str | Path, cache_dir: str | Path | None = None):
    from datasets import Audio, Dataset, DatasetDict, Features, Value
    import soundfile as sf

    root = Path(root).resolve()
    accepted, manifest_ids, incomplete = {}, set(), []
    fields = ("id", "filename", "normalized_text", "text", "language", "sha256", "split")
    for record in _jsonl_records(root / "manifest.jsonl"):
        if record["id"] in manifest_ids:
            raise ValueError("Duplicate manifest IDs")
        manifest_ids.add(record["id"])
        if record["status"] == "ACCEPTED":
            missing = set(fields) - record.keys()
            if missing:
                incomplete.append((record["id"], sorted(missing)))
            else:
                accepted[record["id"]] = {key: record[key] for key in fields}
    if incomplete:
        raise ValueError(f"Incomplete accepted manifest records: {incomplete[:5]}")
    seen = set()
    features = Features(
        {
            "id": Value("string"),
            "audio": Audio(sampling_rate=24000, decode=False),
            "normalized_text": Value("string"),
            "text": Value("string"),
            "language": Value("string"),
        }
    )
    prepared = {}
    for split in ("train", "validation"):
        rows = []
        for record in _jsonl_records(root / f"{split}.jsonl"):
            identity = record["id"]
            if identity in seen or identity not in accepted or record["split"] != split:
                raise ValueError("Duplicate or invalid training split record")
            expected = accepted[identity]
            if any(
                record[k] != expected[k]
                for k in ("filename", "normalized_text", "sha256", "split", "language")
            ):
                raise ValueError("Split transcript/audio mapping differs from manifest")
            path = (root / record["filename"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Audio path escapes dataset")
            if _file_hash(path) != record["sha256"]:
                raise ValueError(f"Audio hash mismatch: {path.name}")
            info = sf.info(path)
            if (
                info.samplerate != 24000
                or info.channels != 1
                or info.subtype != "PCM_16"
                or not 0 < info.duration <= 120
            ):
                raise ValueError(f"Invalid training audio format: {path.name}")
            seen.add(identity)
            rows.append({key: record[key] for key in fields})
        prepared[split] = rows
    if seen != set(accepted):
        raise ValueError("Missing accepted records in train/validation files")
    # Validate current files before any cache reuse. Arrow generation holds only one
    # WAV in memory, rather than materializing the entire speech corpus as byte lists.
    cache = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "nc-voice-factory-hf"
    splits = {}
    for split, records in prepared.items():
        splits[split] = (
            Dataset.from_generator(
                _audio_records,
                features=features,
                cache_dir=str(cache),
                keep_in_memory=False,
                writer_batch_size=16,
                gen_kwargs={"records": records, "root": str(root)},
            )
            if records
            else Dataset.from_dict({key: [] for key in features}, features=features)
        )
    return DatasetDict(splits)


def encode_for_lfm3(dataset, processor):
    """Use in the separate training environment with the notebook's processor."""
    import librosa
    import soundfile as sf

    def encode(example):
        audio, rate = sf.read(io.BytesIO(example["audio"]["bytes"]), dtype="float32")
        if audio.ndim != 1:
            raise ValueError("LFM3 codec requires mono audio")
        if rate != 24000:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=24000)
        example["codes"] = processor.encode_audio(audio).tolist()
        return example

    return dataset.map(encode, remove_columns=["audio"], desc="Encoding speech for LFM3")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect an NC dataset using the official LFM3 input schema")
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    print(load_lfm3_dataset(args.root))

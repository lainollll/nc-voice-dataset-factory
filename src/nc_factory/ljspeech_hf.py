"""Portable Parquet export matching the inspected MikhailT/lj-speech schema."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import soundfile as sf

from .audio import process_audio, read_audio, sha256_file

REFERENCE_URL = "https://huggingface.co/datasets/MikhailT/lj-speech"
REFERENCE_REVISION = "f04bb20ddce4b1dbcfb95b10e5573121d871bf5f"
FOLDER = "ljspeech_hf"
SAMPLE_RATE = 22050


def _features():
    try:
        from datasets import Audio, Features, Value
    except ImportError as exc:
        raise RuntimeError("Install export dependencies with scripts/setup.ps1 -Exports") from exc
    return Features(
        {
            "file": Value("string"),
            "spoken_text": Value("string"),
            "normalized_text": Value("string"),
            "audio": Audio(sampling_rate=SAMPLE_RATE),
        }
    )


def write_ljspeech_hf(
    root: Path, records: list[dict[str, Any]], max_shard_bytes: int = 256 * 1024**2
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if max_shard_bytes <= 0:
        raise ValueError("Parquet shard byte budget must be positive")
    accepted = [record for record in records if record["status"] == "ACCEPTED"]
    if not accepted:
        raise ValueError("No accepted records for LJ Speech HF export")
    if len({Path(record["filename"]).name for record in accepted}) != len(accepted):
        raise ValueError("Parquet source filenames must have unique basenames")
    destination = root / FOLDER
    if destination.exists():
        raise FileExistsError(destination)
    data_dir = destination / "data"
    data_dir.mkdir(parents=True)
    schema = _features().arrow_schema
    shard_paths, shard_counts, languages = [], [], set()
    writer, batch, shard_bytes, shard_count = None, [], 0, 0

    def flush():
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            batch.clear()

    mapping_path = destination / "audio_manifest.jsonl"
    try:
        with mapping_path.open("w", encoding="utf-8", newline="\n") as mapping:
            for record in accepted:
                source = (root / record["filename"]).resolve()
                if not source.is_relative_to(root.resolve()) or sha256_file(source) != record["sha256"]:
                    raise ValueError("Source WAV path/hash mismatch during Parquet export")
                samples, rate = read_audio(source)
                converted = process_audio(samples, rate, target_rate=SAMPLE_RATE)
                buffer = io.BytesIO()
                sf.write(buffer, converted, SAMPLE_RATE, format="WAV", subtype="PCM_16")
                wav = buffer.getvalue()
                if writer is not None and shard_bytes + len(wav) > max_shard_bytes:
                    flush()
                    writer.close()
                    writer = None
                    shard_counts.append(shard_count)
                if writer is None:
                    path = data_dir / f".full-{len(shard_paths):05d}.parquet"
                    shard_paths.append(path)
                    writer = pq.ParquetWriter(path, schema, compression="snappy")
                    shard_bytes, shard_count = 0, 0
                filename = source.name
                spoken = record["synthesis_text"]
                normalized = record["normalized_text"]
                if not spoken.strip() or not normalized.strip():
                    raise ValueError("Empty transcript in Parquet export")
                batch.append(
                    {
                        "file": filename,
                        "spoken_text": spoken,
                        "normalized_text": normalized,
                        "audio": {"bytes": wav, "path": filename},
                    }
                )
                shard_bytes += len(wav)
                shard_count += 1
                if len(batch) == 16:
                    flush()
                languages.add(record["language"])
                mapping.write(
                    json.dumps(
                        {
                            "file": filename,
                            "source_id": record["id"],
                            "source_filename": record["filename"],
                            "source_sha256": record["sha256"],
                            "spoken_text": spoken,
                            "normalized_text": normalized,
                            "sha256": hashlib.sha256(wav).hexdigest(),
                            "duration": len(converted) / SAMPLE_RATE,
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
            flush()
            writer.close()
            writer = None
            shard_counts.append(shard_count)
    finally:
        if writer is not None:
            writer.close()
    shards = []
    for index, path in enumerate(shard_paths):
        final = data_dir / f"full-{index:05d}-of-{len(shard_paths):05d}.parquet"
        path.rename(final)
        shards.append(
            {
                "file": final.relative_to(destination).as_posix(),
                "rows": shard_counts[index],
                "sha256": sha256_file(final),
            }
        )
    info = {
        "schema_version": 1,
        "reference_url": REFERENCE_URL,
        "reference_revision": REFERENCE_REVISION,
        "split": "full",
        "sample_rate": SAMPLE_RATE,
        "accepted_count": len(accepted),
        "shard_byte_budget": max_shard_bytes,
        "shards": shards,
        "mapping_sha256": sha256_file(mapping_path),
    }
    (destination / "export_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    card = """---
configs:
- config_name: default
  data_files:
  - split: full
    path: data/full-*.parquet
dataset_info:
  features:
  - name: file
    dtype: string
  - name: spoken_text
    dtype: string
  - name: normalized_text
    dtype: string
  - name: audio
    dtype:
      audio:
        sampling_rate: 22050
  splits:
  - name: full
    num_examples: COUNT
pretty_name: NC Voice Dataset
language:
LANGUAGES
---
# NC Voice Dataset

Accepted recordings generated with NC Voice Dataset Factory. This is an independent
dataset using the four-column layout of MikhailT/lj-speech, not a copy of its corpus.
Audio is embedded as 22,050 Hz mono PCM16 WAV bytes in the Parquet files.
`spoken_text` preserves the actual synthesis text; `normalized_text` is the current
validated training transcript. Per-file hashes and source mapping are recorded in
`audio_manifest.jsonl`; the parent export retains full generation/QC provenance.
No reference dataset license is assigned to these separately generated recordings.
"""
    card = card.replace("COUNT", str(len(accepted))).replace(
        "LANGUAGES", "\n".join("- " + lang for lang in sorted(languages))
    )
    (destination / "README.md").write_text(card, encoding="utf-8")


def validate_ljspeech_hf(root: Path, records: list[dict[str, Any]]) -> None:
    import pyarrow.parquet as pq
    import yaml

    folder = root / FOLDER
    info = json.loads((folder / "export_info.json").read_text(encoding="utf-8"))
    mapping_path = folder / "audio_manifest.jsonl"
    if sha256_file(mapping_path) != info["mapping_sha256"]:
        raise ValueError("Parquet audio mapping hash mismatch")
    expected = {Path(record["filename"]).name: record for record in records if record["status"] == "ACCEPTED"}
    mapping = {}
    with mapping_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["file"] in mapping:
                raise ValueError("Duplicate Parquet audio mapping")
            mapping[row["file"]] = row
    if set(mapping) != set(expected) or info["accepted_count"] != len(expected):
        raise ValueError("Parquet accepted record count/mapping mismatch")
    card = (folder / "README.md").read_text(encoding="utf-8")
    if not card.startswith("---\n") or len(card.split("---", 2)) != 3:
        raise ValueError("Parquet dataset card requires YAML metadata")
    try:
        metadata = yaml.safe_load(card.split("---", 2)[1])
    except yaml.YAMLError as exc:
        raise ValueError("Parquet dataset card YAML is invalid") from exc
    expected_features = [
        {"name": name, "dtype": "string"} for name in ("file", "spoken_text", "normalized_text")
    ]
    expected_features.append({"name": "audio", "dtype": {"audio": {"sampling_rate": SAMPLE_RATE}}})
    if (
        metadata["configs"]
        != [{"config_name": "default", "data_files": [{"split": "full", "path": "data/full-*.parquet"}]}]
        or metadata["dataset_info"]["features"] != expected_features
        or metadata["dataset_info"]["splits"] != [{"name": "full", "num_examples": len(expected)}]
    ):
        raise ValueError("Parquet dataset card full split/features/count mismatch")
    if info["split"] != "full" or info["sample_rate"] != SAMPLE_RATE:
        raise ValueError("Parquet split/sample-rate summary mismatch")
    schema, seen, listed = _features().arrow_schema, set(), set()
    for shard in info["shards"]:
        path = (folder / shard["file"]).resolve()
        if not path.is_relative_to((folder / "data").resolve()):
            raise ValueError("Parquet shard path escapes data folder")
        if path.name in listed or sha256_file(path) != shard["sha256"]:
            raise ValueError("Parquet shard duplicate/hash mismatch")
        listed.add(path.name)
        parquet = pq.ParquetFile(path)
        if (
            not parquet.schema_arrow.equals(schema, check_metadata=True)
            or parquet.metadata.num_rows != shard["rows"]
        ):
            raise ValueError("Parquet schema/row count mismatch")
        for batch in parquet.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                name = row["file"]
                if name in seen or name not in expected:
                    raise ValueError("Parquet duplicate/unexpected filename")
                seen.add(name)
                original, audit = expected[name], mapping[name]
                if (
                    row["spoken_text"] != original["synthesis_text"]
                    or row["normalized_text"] != original["normalized_text"]
                ):
                    raise ValueError("Parquet spoken_text/normalized_text mapping mismatch")
                if (
                    audit["source_id"] != original["id"]
                    or audit["source_filename"] != original["filename"]
                    or audit["source_sha256"] != original["sha256"]
                ):
                    raise ValueError("Parquet source audio mapping mismatch")
                wav = row["audio"]["bytes"]
                if row["audio"]["path"] != name or hashlib.sha256(wav).hexdigest() != audit["sha256"]:
                    raise ValueError("Parquet embedded audio path/hash mismatch")
                header = sf.info(io.BytesIO(wav))
                if (header.samplerate, header.channels, header.subtype) != (SAMPLE_RATE, 1, "PCM_16"):
                    raise ValueError("Parquet audio must be 22050 Hz mono PCM16")
                if (
                    abs(header.duration - audit["duration"]) > 1 / SAMPLE_RATE
                    or abs(header.duration - original["duration"]) > 1 / SAMPLE_RATE
                ):
                    raise ValueError("Parquet audio duration mismatch")
    if seen != set(expected) or listed != {p.name for p in (folder / "data").glob("*.parquet")}:
        raise ValueError("Parquet missing/extra audio records or shards")

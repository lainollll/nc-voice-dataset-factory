from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import soundfile as sf

from .audio import fingerprint, process_audio, read_audio, sha256_file, write_wav_atomic
from .splits import grouped_split
from .storage import Repository, atomic_json, canonical_text, utcnow

COOKBOOK_REVISION = "3d0cbcb7dc40b690e3e4996c5b78e4d3020f8232"
CHATTERBOX_REVISION = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"


@dataclass(frozen=True)
class ExportOptions:
    formats: tuple[str, ...] = ("ljspeech_parquet",)
    validation_fraction: float = 0.1
    seed: int = 0
    include_rejected: bool = False


@dataclass(frozen=True)
class ExportReport:
    destination: str
    accepted_count: int
    accepted_seconds: float
    warnings: tuple[str, ...] = ()


def dependency_versions() -> dict[str, str | None]:
    result = {}
    for package in (
        "nc-voice-dataset-factory",
        "PySide6",
        "numpy",
        "scipy",
        "soundfile",
        "torch",
        "torchaudio",
        "chatterbox-tts",
        "faster-whisper",
        "datasets",
        "speechbrain",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _relative_file(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Export audio path escapes the dataset")
    return path


def validate_export(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    issues, warnings, seen_files, seen_audio = [], [], set(), set()
    manifest = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    accepted = [record for record in manifest if record["status"] == "ACCEPTED"]
    total_duration = 0.0
    for record in manifest:
        try:
            name = record["filename"]
            if name in seen_files:
                raise ValueError(f"duplicate filename {name}")
            seen_files.add(name)
            path = _relative_file(root, name)
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise ValueError(f"missing audio or hash mismatch: {name}")
            data, rate = read_audio(path)
            header = sf.info(path)
            if header.channels != 1 or header.subtype != "PCM_16" or rate != 24000:
                raise ValueError(f"invalid training audio format: {name}")
            duration = len(data) / rate
            if abs(duration - record["duration"]) > 1 / rate:
                raise ValueError(f"duration mismatch: {name}")
            if record["status"] == "ACCEPTED":
                value = fingerprint(data, rate)
                if value in seen_audio:
                    raise ValueError(f"duplicate audio: {name}")
                seen_audio.add(value)
                total_duration += duration
                if not record["normalized_text"].strip():
                    raise ValueError(f"empty transcript: {name}")
        except (OSError, ValueError, RuntimeError) as exc:
            issues.append(str(exc))
    with (root / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    by_name = {r["filename"]: r for r in accepted}
    if len(metadata) != len(accepted) or len({r["filename"] for r in metadata}) != len(metadata):
        issues.append("metadata record count/duplicate mismatch")
    for row in metadata:
        ref = by_name.get(row["filename"])
        if ref is None or any(
            row[key] != ref[key] for key in ("normalized_text", "text", "language", "split")
        ):
            issues.append(f"metadata text/audio mapping mismatch: {row['filename']}")
    if "ljspeech" in info["formats"]:
        try:
            lj_rows = [
                line.split("|")
                for line in (root / "metadata_ljspeech.csv").read_text(encoding="utf-8").splitlines()
            ]
            expected_lj = {Path(r["filename"]).stem: (r["text"], r["normalized_text"]) for r in accepted}
            actual_lj = {row[0]: tuple(row[1:]) for row in lj_rows if len(row) == 3}
            if len(lj_rows) != len(expected_lj) or len(actual_lj) != len(lj_rows) or actual_lj != expected_lj:
                issues.append("LJSpeech metadata delimiter/count/text mapping mismatch")
        except OSError as exc:
            issues.append(f"LJSpeech metadata missing/unreadable: {exc}")
    expected = (
        grouped_split(
            [{**r, "training_text": r["normalized_text"]} for r in accepted],
            info["validation_fraction"],
            info["split_seed"],
        )
        if accepted
        else {}
    )
    if any(r["split"] != expected[r["id"]] for r in accepted):
        issues.append("split membership mismatch or near-duplicate leakage")
    split_ids = set()
    for split in ("train", "validation"):
        lines = [
            json.loads(line)
            for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected_rows = {r["id"]: r for r in accepted if r["split"] == split}
        if {r["id"] for r in lines} != set(expected_rows) or len(lines) != len(expected_rows):
            issues.append(f"{split} JSONL record mismatch")
        for row in lines:
            ref = expected_rows.get(row["id"])
            if (
                row["id"] in split_ids
                or ref is None
                or any(
                    row.get(k) != ref.get(k) for k in ("filename", "normalized_text", "language", "sha256")
                )
            ):
                issues.append(f"invalid {split} mapping for {row['id']}")
            split_ids.add(row["id"])
    if len(accepted) != info["accepted_count"] or abs(total_duration - info["accepted_seconds"]) > 0.001:
        issues.append("dataset summary totals mismatch")
    if not any(r["split"] == "validation" for r in accepted) and info["validation_fraction"]:
        warnings.append(
            "No independent transcript group is available for validation; all recordings remain in train"
        )
    if "ljspeech_parquet" in info["formats"]:
        try:
            from .ljspeech_hf import validate_ljspeech_hf

            validate_ljspeech_hf(root, manifest)
        except (OSError, ValueError, RuntimeError, KeyError, ImportError, TypeError) as exc:
            issues.append(f"Parquet verification failed: {exc}")
    return {
        "valid": not issues,
        "issues": issues,
        "warnings": warnings,
        "accepted_count": len(accepted),
        "accepted_seconds": total_duration,
    }


def export_dataset(
    repo: Repository,
    destination: Path,
    options: ExportOptions | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> ExportReport:
    options = options or ExportOptions()
    if not options.formats or set(options.formats) - {
        "csv",
        "ljspeech",
        "lfm3",
        "huggingface",
        "ljspeech_parquet",
    }:
        raise ValueError("Choose supported export formats")
    if not math.isfinite(options.validation_fraction) or not 0 <= options.validation_fraction <= 0.5:
        raise ValueError("Validation fraction must be in [0, 0.5]")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Choose a new export folder; existing exports are never overwritten")
    if destination == repo.root or repo.root.is_relative_to(destination):
        raise ValueError("Export destination cannot contain the working project")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".nc-export-", dir=destination.parent))
    warnings = []
    try:
        with repo.writer_lock():
            records = sorted(repo.iter_samples(state="ACCEPTED"), key=lambda r: r["id"])
            if not records:
                raise ValueError("There are no accepted recordings to export")
            splits = grouped_split(records, options.validation_fraction, options.seed)
            if options.validation_fraction and "validation" not in splits.values():
                warnings.append("No independent transcript group is available for validation")
            for name in ("audio", "rejected", "logs"):
                (staging / name).mkdir()
            versions = dependency_versions()
            manifests, job_configs, keys, hashes = [], {}, set(), set()
            for index, sample in enumerate(records, 1):
                if sample["integrity_hold"] or sample["training_revision"] != sample["validated_revision"]:
                    raise ValueError(f"Sample requires integrity repair/revalidation: {sample['id']}")
                source = repo._safe_audio(sample["path"])
                if sha256_file(source) != sample["sha256"]:
                    raise ValueError(f"Audio hash mismatch: {source.name}")
                data, rate = read_audio(source)
                header = sf.info(source)
                transcript_key = (sample["language"], canonical_text(sample["training_text"]))
                audio_key = fingerprint(data, rate)
                if transcript_key in keys or audio_key in hashes:
                    raise ValueError(
                        "Duplicate transcript or audio in accepted dataset; review before export"
                    )
                keys.add(transcript_key)
                hashes.add(audio_key)
                export_text = (
                    sample["training_text"] if sample["training_revision"] else sample["original_text"]
                )
                if "ljspeech" in options.formats and any(
                    c in export_text + sample["training_text"] for c in "|\r\n"
                ):
                    raise ValueError(
                        f"LJSpeech cannot represent pipe/newline delimiters in sample {sample['id']}"
                    )
                filename = f"audio/{index:06d}.wav"
                if rate == 24000 and header.channels == 1 and header.subtype == "PCM_16":
                    shutil.copyfile(source, staging / filename)
                else:
                    data = process_audio(data, rate, target_rate=24000)
                    write_wav_atomic(staging / filename, data, 24000)
                    rate = 24000
                exported_hash = sha256_file(staging / filename)
                if sample["job_id"] not in job_configs:
                    job_configs[sample["job_id"]] = json.loads(repo.get_job(sample["job_id"])["config_json"])
                config = job_configs[sample["job_id"]]
                row = dict(
                    id=sample["id"],
                    filename=filename,
                    audio=filename,
                    text=export_text,
                    source_original_text=sample["original_text"],
                    normalized_text=sample["training_text"],
                    language=sample["language"],
                    category=sample["category"],
                    split=splits[sample["id"]],
                    status="ACCEPTED",
                    duration=len(data) / rate,
                    sha256=exported_hash,
                    source_audio_sha256=sample["sha256"],
                    synthesis_text=sample["parameters"].get("spoken_text", sample["spoken_text"]),
                    provenance=sample["provenance"],
                    model_id=sample["model_id"],
                    voice=config["voice"],
                    generation_parameters=sample["parameters"],
                    quality_metrics=sample["metrics"],
                    asr_text=sample["asr_text"],
                    wer=sample["wer"],
                    cer=sample["cer"],
                    speaker_similarity=sample["speaker_similarity"],
                    dependency_versions=versions,
                    training_revision=sample["training_revision"],
                    job_id=sample["job_id"],
                    attempt_id=sample["latest_attempt_id"],
                    review_history=repo.events(sample["id"]),
                )
                manifests.append(row)
                if on_progress:
                    on_progress({"exported": index, "total": len(records)})
            with (staging / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
                fields = ["filename", "text", "normalized_text", "language", "split"]
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(manifests)
            if "ljspeech" in options.formats:
                with (staging / "metadata_ljspeech.csv").open("w", encoding="utf-8", newline="\n") as handle:
                    for row in manifests:
                        handle.write(f"{Path(row['filename']).stem}|{row['text']}|{row['normalized_text']}\n")
            for split in ("train", "validation"):
                _jsonl(staging / f"{split}.jsonl", [r for r in manifests if r["split"] == split])
            rejected = []
            if options.include_rejected:
                for sample in repo.iter_samples(state="REJECTED"):
                    if not sample["path"]:
                        continue
                    source = repo._safe_audio(sample["path"])
                    if sha256_file(source) != sample["sha256"]:
                        raise ValueError("Rejected audio hash mismatch")
                    filename = f"rejected/{sample['id']}.wav"
                    data, rate = read_audio(source)
                    data = process_audio(data, rate, target_rate=24000)
                    write_wav_atomic(staging / filename, data, 24000)
                    rejected.append(
                        dict(
                            id=sample["id"],
                            filename=filename,
                            status="REJECTED",
                            split=None,
                            normalized_text=sample["training_text"],
                            duration=len(data) / 24000,
                            sha256=sha256_file(staging / filename),
                            source_audio_sha256=sample["sha256"],
                            quality_metrics=sample["metrics"],
                            reasons=sample["reasons"],
                        )
                    )
            _jsonl(staging / "manifest.jsonl", manifests + rejected)
            duration = sum(r["duration"] for r in manifests)
            atomic_json(
                staging / "dataset_info.json",
                dict(
                    schema_version=1,
                    name="NC Voice Dataset",
                    created_at=utcnow(),
                    accepted_count=len(records),
                    accepted_seconds=duration,
                    rejected_in_snapshot=len(rejected),
                    split_seed=options.seed,
                    validation_fraction=options.validation_fraction,
                    split_algorithm="nc-grouped-v1-0.90",
                    dependency_versions=versions,
                    formats=options.formats,
                    sample_rate=24000,
                    channels=1,
                    subtype="PCM_16",
                    cookbook_revision=COOKBOOK_REVISION,
                    chatterbox_revision=CHATTERBOX_REVISION,
                    warnings=warnings,
                    contains_mock_audio=any("MOCK" in r["model_id"] for r in manifests),
                ),
            )
            atomic_json(staging / "project_config.json", {"schema_version": 1, "jobs": job_configs})
            if "lfm3" in options.formats:
                shutil.copyfile(Path(__file__).with_name("lfm3.py"), staging / "load_lfm3_dataset.py")
            if "ljspeech_parquet" in options.formats:
                from .ljspeech_hf import write_ljspeech_hf

                write_ljspeech_hf(staging, manifests)
            validation = validate_export(staging)
            atomic_json(staging / "logs" / "integrity_report.json", validation)
            if not validation["valid"]:
                raise ValueError("Export integrity validation failed: " + "; ".join(validation["issues"]))
            if "huggingface" in options.formats:
                from .lfm3 import load_lfm3_dataset
                import gc

                with tempfile.TemporaryDirectory(prefix=".nc-hf-cache-", dir=destination.parent) as cache:
                    dataset = load_lfm3_dataset(staging, cache_dir=cache)
                    dataset.save_to_disk(str(staging / "huggingface"))
                    del dataset
                    gc.collect()
            if destination.exists():
                raise FileExistsError(destination)
            staging.rename(destination)
            repo.record_event(
                "dataset_exported", {"destination": str(destination), "accepted_count": len(records)}
            )
        return ExportReport(str(destination), len(records), duration, tuple(warnings))
    finally:
        if staging.exists():
            shutil.rmtree(staging)

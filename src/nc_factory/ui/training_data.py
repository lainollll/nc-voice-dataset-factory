"""Read-only training source summaries, called on a worker before enabling Start."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from ..export import validate_export
from ..storage import Repository


@dataclass(frozen=True)
class TrainingSourceSummary:
    ready: bool
    message: str
    accepted: int = 0
    seconds: float = 0.0
    voice: str = ""
    language: str = ""


def inspect_training_source(repo: Repository | None, export: Path | None) -> TrainingSourceSummary:
    try:
        voices: dict[str, str] = {}
        languages: set[str] = set()
        if export is not None:
            if not export.is_dir():
                return TrainingSourceSummary(False, "Choose an existing NC dataset export folder.")
            report = validate_export(export)
            if not report["valid"]:
                return TrainingSourceSummary(False, "Dataset needs attention: " + "; ".join(report["issues"][:3]))
            count, seconds = report["accepted_count"], report["accepted_seconds"]
            for line in (export / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record["status"] == "ACCEPTED":
                    voices[record["voice"]["id"]] = record["voice"]["name"]
                    languages.add(record["language"])
            source = f"Saved dataset: {export}"
        elif repo is not None:
            stats = repo.statistics()
            count, seconds = stats["accepted"], stats["accepted_seconds"]
            for job in repo.list_jobs():
                if repo.statistics(job["id"])["accepted"]:
                    config = json.loads(job["config_json"])
                    voices[config["voice"]["id"]] = config["voice"]["name"]
                    languages.add(config["language"])
            source = f"Current project: {repo.root}"
        else:
            return TrainingSourceSummary(False, "Open a project with accepted recordings, or choose a saved dataset.")
        if not count:
            return TrainingSourceSummary(False, source + "\nNo accepted recordings yet. Generate and review a dataset first.")
        if len(voices) != 1 or len(languages) != 1 or not languages <= {"en", "sv"}:
            return TrainingSourceSummary(False, source + "\nChoose a dataset with one voice and one supported language.")
        return TrainingSourceSummary(True, source, count, seconds, next(iter(voices.values())), next(iter(languages)))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return TrainingSourceSummary(False, f"Unable to use this dataset: {exc}")

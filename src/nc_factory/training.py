"""Prepare an isolated LFM3 run; no training packages are imported in the app."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .export import ExportOptions, export_dataset, validate_export
from .storage import Repository, atomic_json
from .training_notebook import COOKBOOK_REVISION, NOTEBOOK_NAME, adapt_notebook

NOTEBOOK_SHA256 = "9d2d5601d6a73a1e304b7456200cf9751ed5ed49dddce44a4d840e8d19661565"
RAW_NOTEBOOK_URL = f"https://raw.githubusercontent.com/Liquid4All/cookbook/{COOKBOOK_REVISION}/finetuning/notebooks/{NOTEBOOK_NAME}"


@dataclass(frozen=True)
class TrainingRequest:
    dataset_root: Path | None = None
    project_root: Path | None = None
    voice_name: str = ""
    max_steps: int = 200


@dataclass(frozen=True)
class PreparedTraining:
    run_root: Path
    notebook: Path
    runner: Path


def fetch_notebook() -> dict[str, Any]:
    response = requests.get(RAW_NOTEBOOK_URL, timeout=(10, 45))
    response.raise_for_status()
    content = response.content
    if len(content) > 2_000_000 or hashlib.sha256(content).hexdigest() != NOTEBOOK_SHA256:
        raise ValueError("The official notebook checksum differs from the inspected version")
    return json.loads(content)


def prepare_training(
    request: TrainingRequest, training_root: Path, cancel: threading.Event | None = None
) -> PreparedTraining:
    def check_cancelled() -> None:
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Training preparation cancelled")

    check_cancelled()
    if not 1 <= request.max_steps <= 1_000_000:
        raise ValueError("Choose 1–1,000,000 training steps")
    if (request.dataset_root is None) == (request.project_root is None):
        raise ValueError("Choose a dataset export or open a project with accepted recordings")
    if request.project_root is not None and not (request.project_root / "factory.sqlite3").is_file():
        raise ValueError("Open an existing project with accepted recordings")
    if request.dataset_root is not None:
        report = validate_export(request.dataset_root)
        if not report["valid"] or not report["accepted_count"]:
            raise ValueError("Dataset export is invalid: " + "; ".join(report["issues"]))
    check_cancelled()
    run_root = training_root / "runs" / uuid.uuid4().hex
    run_root.mkdir(parents=True, exist_ok=False)
    dataset_root = request.dataset_root
    if dataset_root is None:
        dataset_root = run_root / "dataset"
        export_dataset(Repository(request.project_root), dataset_root, ExportOptions(formats=("csv", "lfm3")))
    check_cancelled()
    voices, languages = set(), set()
    voice_name = request.voice_name.strip()
    with (dataset_root / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["status"] != "ACCEPTED":
                continue
            voices.add(row["voice"]["id"])
            languages.add(row["language"])
            voice_name = voice_name or row["voice"]["name"]
    if len(voices) != 1 or len(languages) != 1 or not languages <= {"en", "sv"}:
        raise ValueError("Fine-tune one voice and language at a time; choose a separate dataset export")
    notebook = adapt_notebook(fetch_notebook())
    check_cancelled()
    notebook_path = run_root / "finetuning" / "notebooks" / NOTEBOOK_NAME
    atomic_json(notebook_path, notebook)
    atomic_json(
        run_root / "training_request.json",
        {
            "dataset_root": "dataset" if request.dataset_root is None else str(dataset_root.resolve()),
            "voice_name": voice_name,
            "language": next(iter(languages)),
            "max_steps": request.max_steps,
            "generation_prefix": sys.prefix,
            "source_revision": COOKBOOK_REVISION,
            "source_sha256": NOTEBOOK_SHA256,
        },
    )
    shutil.copyfile(Path(__file__).with_name("lfm3.py"), run_root / "load_lfm3_dataset.py")
    runner_path = run_root / "run_training.py"
    shutil.copyfile(Path(__file__).with_name("training_runner.py"), runner_path)
    return PreparedTraining(run_root, notebook_path, runner_path)

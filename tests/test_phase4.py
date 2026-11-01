from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest


def test_grouped_split_no_near_duplicate_leakage_and_permutation_stable():
    from nc_factory.splits import grouped_split

    records = [
        dict(id="a", language="en", training_text="Please bring the red book to the quiet room today."),
        dict(id="b", language="en", training_text="Please bring the blue book to the quiet room today."),
        dict(
            id="c", language="en", training_text="Ocean currents shape the weather along the distant coast."
        ),
        dict(
            id="d",
            language="sv",
            training_text="Åsa läser en spännande berättelse för barnen på biblioteket.",
        ),
    ]
    a = grouped_split(records, 0.25, 42)
    assert a["a"] == a["b"]
    assert set(a.values()) == {"train", "validation"}
    assert a == grouped_split(list(reversed(records)), 0.25, 42)
    # Length ratio 9/11 is less than .9; SequenceMatcher similarity is nevertheless .9.
    short = "one two three four five six seven eight nine"
    long = short + " ten eleven"
    grouped = grouped_split(
        [
            dict(id="short", language="en", training_text=short),
            dict(id="long", language="en", training_text=long),
        ],
        0.5,
        0,
    )
    assert grouped["short"] == grouped["long"]


@pytest.fixture
def completed(tmp_path, config):
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS
    from nc_factory.text import generate_texts

    repo = Repository(tmp_path / "project")
    cfg = replace(config, target_count=10)
    repo.save_config(cfg)
    job = repo.create_job(cfg, generate_texts(10, seed=12))
    Pipeline(repo, tts=MockTTS()).run(job)
    return repo


def test_complete_export_exact_manifest_and_rejected_exclusion(tmp_path, completed):
    from nc_factory.export import ExportOptions, export_dataset, validate_export
    from nc_factory.audio import sha256_file

    sample = next(completed.iter_samples())
    completed.review(sample["id"], "reject", "Intentionally held out by reviewer")
    destination = tmp_path / "dataset"
    report = export_dataset(completed, destination, ExportOptions(formats=("csv", "ljspeech", "lfm3")))
    assert report.accepted_count == 9
    assert validate_export(destination)["valid"]
    for name in (
        "metadata.csv",
        "metadata_ljspeech.csv",
        "train.jsonl",
        "validation.jsonl",
        "manifest.jsonl",
        "dataset_info.json",
        "project_config.json",
    ):
        assert (destination / name).is_file()
    records = [
        json.loads(line) for line in (destination / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 9 and all(r["status"] == "ACCEPTED" for r in records)
    for record in records:
        assert sha256_file(destination / record["filename"]) == record["sha256"]
        assert record["voice"]["reference_sha256"]
        assert record["generation_parameters"]["seed"] >= 0
        assert "MOCK" in record["model_id"]
    with (destination / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert sum(r["split"] == "validation" for r in rows) >= 1
    assert set(r["filename"] for r in rows) == set(r["filename"] for r in records)


def test_export_integrity_failure_never_publishes(tmp_path, completed):
    from nc_factory.export import export_dataset

    sample = next(completed.iter_samples())
    Path(sample["path"]).write_bytes(b"corrupt")
    destination = tmp_path / "bad-export"
    with pytest.raises(ValueError, match="hash|invalid|audio"):
        export_dataset(completed, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".nc-export-*"))


def test_existing_destination_and_bad_ljspeech_delimiter(tmp_path, completed):
    from nc_factory.export import ExportOptions, export_dataset

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "marker.txt"
    marker.write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_dataset(completed, existing)
    assert marker.read_text(encoding="utf-8") == "keep me"
    sample = next(completed.iter_samples())
    completed.edit_training_text(sample["id"], "Pipe | delimiter is not valid in LJSpeech.")
    completed.review(sample["id"], "accept", "Delimiter validation regression fixture")
    with pytest.raises(ValueError, match="pipe|delimiter|LJSpeech"):
        export_dataset(completed, tmp_path / "bad-pipe", ExportOptions(formats=("ljspeech",)))


def test_optional_hf_loader_bytes_match_notebook(tmp_path, completed):
    pytest.importorskip("datasets")
    from nc_factory.export import ExportOptions, export_dataset
    from nc_factory.lfm3 import load_lfm3_dataset

    destination = tmp_path / "hf"
    export_dataset(completed, destination, ExportOptions(formats=("csv", "ljspeech", "lfm3", "huggingface")))
    dataset = load_lfm3_dataset(destination)
    assert set(dataset) == {"train", "validation"}
    row = dataset["train"][0]
    assert isinstance(row["audio"]["bytes"], bytes) and row["normalized_text"]
    import io
    import soundfile as sf

    audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]))
    assert rate == 24000 and audio.ndim == 1

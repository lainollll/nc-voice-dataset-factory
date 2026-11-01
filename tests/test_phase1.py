from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf


def test_voice_reference_integrity(config, reference):
    config.voice.validate()
    reference.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed|hash"):
        config.voice.validate()


def test_permission_and_missing_reference(config):
    with pytest.raises(ValueError, match="permission"):
        replace(config.voice, permission_confirmed=False).validate()
    with pytest.raises((ValueError, FileNotFoundError)):
        replace(config.voice, reference_path="Z:/missing/reference.wav").validate()


def test_true_resampling_and_no_overwrite(tmp_path):
    from nc_factory.audio import process_audio, read_audio, write_wav_atomic

    samples = (0.15 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)).astype("float32")
    converted = process_audio(samples, 48000)
    assert len(converted) == 24000
    path = tmp_path / "output.wav"
    digest = write_wav_atomic(path, converted, 24000)
    decoded, rate = read_audio(path)
    assert rate == 24000 and decoded.shape == (24000,)
    assert sf.info(path).subtype == "PCM_16"
    with pytest.raises(FileExistsError):
        write_wav_atomic(path, np.zeros(24000, dtype="float32"), 24000)
    from nc_factory.audio import sha256_file

    assert sha256_file(path) == digest


def test_nonfinite_audio_rejected(tmp_path):
    from nc_factory.audio import process_audio, write_wav_atomic

    bad = np.array([0.0, float("nan"), 0.2], dtype="float32")
    for operation in (
        lambda: process_audio(bad, 24000),
        lambda: write_wav_atomic(tmp_path / "bad.wav", bad, 24000),
    ):
        with pytest.raises(ValueError, match="finite|NaN"):
            operation()
    assert not (tmp_path / "bad.wav").exists()


def test_ten_recording_csv_pipeline(tmp_path, config):
    from nc_factory.domain import TextItem
    from nc_factory.export import ExportOptions, export_dataset
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    repo = Repository(tmp_path / "project")
    repo.save_config(config)
    items = [
        TextItem(
            f"This is a distinct test sentence about topic {i}.",
            f"This is a distinct test sentence about topic {i}.",
        )
        for i in range(10)
    ]
    job = repo.create_job(config, items)
    Pipeline(repo, tts=MockTTS()).run(job)
    assert repo.get_job(job)["status"] == "COMPLETED"
    assert repo.statistics(job)["accepted"] == 10
    report = export_dataset(repo, tmp_path / "export", ExportOptions(formats=("csv",)))
    assert report.accepted_count == 10
    with (tmp_path / "export" / "metadata.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 10
    for row in rows:
        assert (tmp_path / "export" / row["filename"]).is_file()
        assert row["normalized_text"].startswith("This is a distinct")
    assert all("MOCK" in sample["model_id"] for sample in repo.iter_samples())


def test_native_missing_model_is_actionable(config):
    from nc_factory.tts import NativeChatterbox

    with pytest.raises((ValueError, FileNotFoundError), match="model|offline|directory"):
        NativeChatterbox(replace(config.synthesis, engine="multilingual-v3", model_dir=""))

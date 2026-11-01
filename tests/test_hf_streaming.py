from __future__ import annotations

from dataclasses import replace

import pytest


def test_hf_export_builds_audio_through_disk_backed_generator(tmp_path, config, monkeypatch):
    from datasets import Dataset
    from nc_factory.domain import TextItem
    from nc_factory.export import ExportOptions, export_dataset
    from nc_factory.lfm3 import load_lfm3_dataset
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    repo = Repository(tmp_path / "project")
    job = repo.create_job(
        replace(config, target_count=1),
        [TextItem("Please bring the blue book.", "Please bring the blue book.")],
    )
    Pipeline(repo, tts=MockTTS()).run(job)
    root = tmp_path / "dataset"
    export_dataset(repo, root, ExportOptions(formats=("csv", "lfm3")))
    calls = []
    original = Dataset.from_generator

    def recording_generator(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(Dataset, "from_generator", recording_generator)
    loaded = load_lfm3_dataset(root, cache_dir=tmp_path / "arrow-cache")
    assert calls and len(loaded["train"]) == 1 and len(loaded["validation"]) == 0
    assert loaded["train"][0]["audio"]["bytes"]
    # A cached Arrow dataset must not bypass validation of the current WAV files.
    (root / "audio" / "000001.wav").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_lfm3_dataset(root, cache_dir=tmp_path / "arrow-cache")

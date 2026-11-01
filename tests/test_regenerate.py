from __future__ import annotations

from dataclasses import replace
from pathlib import Path


def test_regenerate_edited_text_preserves_original_audio_and_synthesis_history(tmp_path, config):
    from nc_factory.audio import sha256_file
    from nc_factory.domain import TextItem
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    spoken = []

    class RecordingTTS(MockTTS):
        def synthesize(self, text, *args):
            spoken.append(text)
            return super().synthesize(text, *args)

    repo = Repository(tmp_path / "project")
    job = repo.create_job(
        replace(config, target_count=1),
        [TextItem("Please bring the red book.", "Please bring the red book.")],
    )
    Pipeline(repo, tts=RecordingTTS()).run(job)
    first = next(repo.iter_samples())
    repo.edit_training_text(first["id"], "Please bring the blue book.")
    repo.review(first["id"], "regenerate", "Generate a new recording for the edited text")
    Pipeline(repo, tts=RecordingTTS()).run(job)
    latest = repo.get_sample(first["id"])
    assert spoken == ["Please bring the red book.", "Please bring the blue book."]
    assert latest["spoken_text"] == "Please bring the red book."
    assert latest["parameters"]["spoken_text"] == "Please bring the blue book."
    assert latest["path"] != first["path"]
    assert sha256_file(Path(first["path"])) == first["sha256"]
    assert len(repo.attempts(first["id"])) == 2
    import csv
    import json
    from nc_factory.export import export_dataset

    output = tmp_path / "export"
    export_dataset(repo, output)
    with (output / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["text"] == row["normalized_text"] == "Please bring the blue book."
    manifest = json.loads((output / "manifest.jsonl").read_text(encoding="utf-8"))
    assert manifest["source_original_text"] == "Please bring the red book."


def test_explicit_attempt_budget_extension_is_audited(tmp_path, config):
    from nc_factory.storage import Repository

    repo = Repository(tmp_path / "project")
    job = repo.create_job(replace(config, max_attempts=1), [])
    repo.extend_attempt_budget(job, 10, "Reviewed failures and added source material")
    assert repo.get_job(job)["attempt_budget_override"] == 11
    assert repo.events()[-1]["kind"] == "attempt_budget_extended"

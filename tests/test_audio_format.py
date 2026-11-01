from __future__ import annotations

from dataclasses import replace

import soundfile as sf


def test_working_format_converted_to_training_default(tmp_path, config):
    from nc_factory.domain import TextItem
    from nc_factory.export import export_dataset, validate_export
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    cfg = replace(config, target_count=1, audio_sample_rate=48000, audio_subtype="FLOAT")
    repo = Repository(tmp_path / "project")
    job = repo.create_job(
        cfg, [TextItem("Please bring the blue book tomorrow.", "Please bring the blue book tomorrow.")]
    )
    Pipeline(repo, tts=MockTTS()).run(job)
    source = next(repo.iter_samples(state="ACCEPTED"))
    assert sf.info(source["path"]).samplerate == 48000
    assert sf.info(source["path"]).subtype == "FLOAT"
    root = tmp_path / "dataset"
    export_dataset(repo, root)
    output = sf.info(root / "audio" / "000001.wav")
    assert output.samplerate == 24000 and output.subtype == "PCM_16" and output.channels == 1
    assert validate_export(root)["valid"]

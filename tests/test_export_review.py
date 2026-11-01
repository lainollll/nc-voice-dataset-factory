from __future__ import annotations


def test_validator_checks_ljspeech_mapping(tmp_path, config):
    from dataclasses import replace
    from nc_factory.domain import TextItem
    from nc_factory.export import ExportOptions, export_dataset, validate_export
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
    export_dataset(repo, root, ExportOptions(formats=("ljspeech",)))
    (root / "metadata_ljspeech.csv").write_text(
        "000001|Wrong source text|Wrong training text\n", encoding="utf-8"
    )
    result = validate_export(root)
    assert not result["valid"] and any("LJSpeech" in issue for issue in result["issues"])

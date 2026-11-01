from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace

import soundfile as sf


def completed_project(tmp_path, config):
    from nc_factory.domain import TextItem
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    repo = Repository(tmp_path / "project")
    cfg = replace(config, target_count=3)
    job = repo.create_job(
        cfg,
        [
            TextItem(
                f"Åsa read sentence {n} at the quiet café.",
                f"Åsa read sentence number {n} at the quiet café.",
            )
            for n in range(3)
        ],
    )
    Pipeline(repo, tts=MockTTS()).run(job)
    return repo


def test_lj_hf_export_has_exact_columns_full_split_embedded_22050_wavs(tmp_path, config):
    from datasets import Audio, load_dataset
    from nc_factory.export import ExportOptions, export_dataset, validate_export

    repo = completed_project(tmp_path, config)
    rejected = next(repo.iter_samples())
    repo.review(rejected["id"], "reject", "Excluded from compatible training format")
    destination = tmp_path / "export"
    export_dataset(repo, destination, ExportOptions(formats=("csv", "lfm3", "ljspeech_parquet")))
    root = destination / "ljspeech_hf"
    dataset = load_dataset(str(root), cache_dir=str(tmp_path / "hf-cache"))
    assert set(dataset) == {"full"} and len(dataset["full"]) == 2
    assert dataset["full"].column_names == ["file", "spoken_text", "normalized_text", "audio"]
    assert dataset["full"].features["audio"].sampling_rate == 22050
    rows = list(dataset["full"].cast_column("audio", Audio(decode=False)))
    manifest = [
        json.loads(line) for line in (destination / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    mapping = {r["filename"].split("/")[-1]: r for r in manifest if r["status"] == "ACCEPTED"}
    for row in rows:
        record = mapping[row["file"]]
        assert row["spoken_text"] == record["synthesis_text"]
        assert row["normalized_text"] == record["normalized_text"]
        original = sf.info(destination / record["filename"])
        with sf.SoundFile(io.BytesIO(row["audio"]["bytes"])) as audio:
            assert audio.samplerate == 22050 and audio.channels == 1 and audio.subtype == "PCM_16"
            assert abs(audio.frames / audio.samplerate - original.duration) < 1 / 22050
        assert row["audio"]["path"] == row["file"]
    assert validate_export(destination)["valid"]
    assert all(sf.info(destination / r["filename"]).samplerate == 24000 for r in mapping.values())


def test_lj_parquet_shards_and_validation_detect_corruption(tmp_path, config):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from nc_factory.export import ExportOptions, export_dataset, validate_export
    from nc_factory.ljspeech_hf import write_ljspeech_hf

    repo = completed_project(tmp_path, config)
    destination = tmp_path / "export"
    export_dataset(repo, destination, ExportOptions(formats=("csv",)))
    records = [
        json.loads(line) for line in (destination / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    write_ljspeech_hf(destination, records, max_shard_bytes=1)
    shards = sorted((destination / "ljspeech_hf/data").glob("*.parquet"))
    assert len(shards) == 3
    info = json.loads((destination / "dataset_info.json").read_text(encoding="utf-8"))
    info["formats"].append("ljspeech_parquet")
    (destination / "dataset_info.json").write_text(json.dumps(info), encoding="utf-8")
    assert validate_export(destination)["valid"]
    table = pq.read_table(shards[0])
    index = table.column_names.index("spoken_text")
    changed = table.set_column(index, table.field(index), pa.array(["Wrong spoken mapping."]))
    pq.write_table(changed, shards[0])
    # Force semantic validation too, rather than relying only on the shard hash.
    audit = json.loads((destination / "ljspeech_hf/export_info.json").read_text(encoding="utf-8"))
    audit["shards"][0]["sha256"] = hashlib.sha256(shards[0].read_bytes()).hexdigest()
    (destination / "ljspeech_hf/export_info.json").write_text(json.dumps(audit), encoding="utf-8")
    check = validate_export(destination)
    assert not check["valid"] and any("Parquet" in issue or "parquet" in issue for issue in check["issues"])


def test_lj_hf_option_is_available_in_export_ui(qtbot):
    from nc_factory.export import ExportOptions
    from nc_factory.ui.window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    assert "ljspeech_parquet" in window.export_formats
    assert "22,050" in window.export_formats["ljspeech_parquet"].text()
    assert ExportOptions().formats == ("ljspeech_parquet",)
    assert (
        tuple(key for key, check in window.export_formats.items() if check.isChecked())
        == ExportOptions().formats
    )


def test_default_export_produces_full_ljspeech_hf_dataset(tmp_path, config):
    from nc_factory.export import export_dataset, validate_export

    repo = completed_project(tmp_path, config)
    destination = tmp_path / "default-export"
    export_dataset(repo, destination)
    assert (destination / "ljspeech_hf/README.md").is_file()
    assert list((destination / "ljspeech_hf/data").glob("full-*.parquet"))
    info = json.loads((destination / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["formats"] == ["ljspeech_parquet"]
    assert validate_export(destination)["valid"]


def test_lj_hf_dataset_card_is_validated(tmp_path, config):
    from nc_factory.export import ExportOptions, export_dataset, validate_export

    repo = completed_project(tmp_path, config)
    destination = tmp_path / "export"
    export_dataset(repo, destination, ExportOptions(formats=("ljspeech_parquet",)))
    card = destination / "ljspeech_hf/README.md"
    card.write_text(card.read_text(encoding="utf-8").replace("split: full", "split: train"), encoding="utf-8")
    result = validate_export(destination)
    assert not result["valid"] and any("dataset card" in issue for issue in result["issues"])

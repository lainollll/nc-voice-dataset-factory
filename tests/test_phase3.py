from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


def test_transcript_matching_numbers_unicode_and_insertions():
    from nc_factory.transcript import transcript_metrics

    assert transcript_metrics("We have 20 books.", "we have twenty books", "en")["wer"] == 0
    assert transcript_metrics("Åsa har 20 böcker.", "åsa har tjugo böcker", "sv")["cer"] == 0
    metrics = transcript_metrics("bring the red book", "bring the book", "en")
    assert metrics["wer"] == 0.25
    assert metrics["missing_words"] == ["red"]
    repeated = transcript_metrics("Please bring the book.", "please bring the book bring the book", "en")
    assert repeated["repeated_phrase"]


def test_asr_quality_mismatch_and_unreliable_review(tmp_path, config):
    from nc_factory.audio import write_wav_atomic
    from nc_factory.quality import ASRResult, MockASR, QualityController
    from nc_factory.tts import MockTTS

    result = MockTTS().synthesize("Please bring the red book tomorrow.", config.voice, config.synthesis, 0)
    path = tmp_path / "clip.wav"
    write_wav_atomic(path, result.samples, result.sample_rate)
    settings = replace(config.quality, verify_asr=True)
    good = QualityController(settings, asr=MockASR({str(path): "Please bring the red book tomorrow."}))
    assert good.check(path, "Please bring the red book tomorrow.", "en", config.voice).passed
    bad = QualityController(settings, asr=MockASR({str(path): "Some entirely unrelated hallucinated words."}))
    checked = bad.check(path, "Please bring the red book tomorrow.", "en", config.voice)
    assert not checked.passed and checked.wer > 0.1
    uncertain = MockASR(
        {
            str(path): ASRResult(
                "Please bring the red book tomorrow.", "MOCK-ASR", language="sv", language_probability=0.2
            )
        }
    )
    checked = QualityController(settings, asr=uncertain).check(
        path, "Please bring the red book tomorrow.", "en", config.voice
    )
    assert checked.needs_review and not checked.passed


def test_silence_clipping_and_corrupt_audio(tmp_path, config):
    from nc_factory.audio import write_wav_atomic
    from nc_factory.quality import QualityController

    checker = QualityController(config.quality)
    for name, samples, reason in (
        ("silence", np.zeros(24000), "low_signal"),
        ("clipping", np.ones(24000), "clipping"),
    ):
        path = tmp_path / f"{name}.wav"
        write_wav_atomic(path, samples.astype("float32"), 24000)
        result = checker.check(path, "A source sentence.", "en", config.voice)
        assert not result.passed and reason in result.reasons
    bad = tmp_path / "corrupt.wav"
    bad.write_bytes(b"not a wave")
    result = checker.check(bad, "A source sentence.", "en", config.voice)
    assert not result.passed and "invalid_audio" in result.reasons


def test_speaker_calibration_requires_separation_and_reference_binding(tmp_path, config):
    from nc_factory.speaker import calibrate_scores, load_calibration
    from nc_factory.storage import atomic_json

    with pytest.raises(ValueError, match="separat|overlap"):
        calibrate_scores([0.5, 0.6], [0.55, 0.7], "model", config.voice.reference_sha256)
    calibration = calibrate_scores(
        [0.86, 0.91, 0.89], [0.2, 0.4, 0.5], "model", config.voice.reference_sha256
    )
    assert 0.5 < calibration["threshold"] < 0.86
    file = tmp_path / "calibration.json"
    atomic_json(file, calibration)
    assert (
        load_calibration(file, "model", config.voice.reference_sha256)["threshold"]
        == calibration["threshold"]
    )
    with pytest.raises(ValueError, match="reference|model"):
        load_calibration(file, "other-model", config.voice.reference_sha256)


def test_text_edit_invalidates_acceptance_without_touching_audio(tmp_path, config):
    from nc_factory.audio import sha256_file
    from nc_factory.domain import TextItem
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    repo = Repository(tmp_path / "project")
    job = repo.create_job(
        replace(config, target_count=1),
        [TextItem("Please bring the red book.", "Please bring the red book.")],
    )
    Pipeline(repo, tts=MockTTS()).run(job)
    sample = next(repo.iter_samples())
    repo.edit_training_text(sample["id"], "Please bring the blue book.")
    edited = repo.get_sample(sample["id"])
    assert edited["spoken_text"] == "Please bring the red book."
    assert sha256_file(Path(edited["path"])) == sample["sha256"]
    assert edited["state"] == "REJECTED" and repo.statistics()["accepted"] == 0
    repo.review(sample["id"], "accept", "Reviewed and corrected transcript by listening")
    assert repo.statistics()["accepted"] == 1
    assert repo.events(sample["id"])[-1]["payload"]["actor"]

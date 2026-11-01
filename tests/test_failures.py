from __future__ import annotations

from dataclasses import replace
from pathlib import Path


def test_crash_after_published_wav_preserves_artifact_and_regenerates_unique(tmp_path, config):
    from nc_factory.audio import sha256_file, write_wav_atomic
    from nc_factory.domain import TextItem
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    repo = Repository(tmp_path / "project")
    cfg = replace(config, target_count=1)
    job = repo.create_job(
        cfg, [TextItem("Please bring the blue book tomorrow.", "Please bring the blue book tomorrow.")]
    )
    sample = repo.next_pending(job)
    attempt = repo.begin_attempt(sample["id"], 0, {})
    audio = MockTTS().synthesize(sample["spoken_text"], cfg.voice, cfg.synthesis, 0)
    orphan = repo.root / "audio" / f"{sample['id']}_{attempt}.wav"
    digest = write_wav_atomic(orphan, audio.samples, audio.sample_rate)
    report = Repository(repo.root).recover()
    assert str(orphan) in report["orphans"]
    result = Pipeline(Repository(repo.root), tts=MockTTS()).run(job)
    assert result["accepted"] == 1
    accepted = next(repo.iter_samples(state="ACCEPTED"))
    assert Path(accepted["path"]) != orphan
    assert sha256_file(orphan) == digest
    assert len(repo.attempts(sample["id"])) == 2


def test_all_bad_audio_stops_at_persisted_attempt_budget(tmp_path, config):
    import numpy as np
    from nc_factory.domain import AudioResult, TextItem
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    class BrokenTTS(MockTTS):
        def synthesize(self, *args):
            return AudioResult(np.array([float("nan")]), 24000, "BROKEN", {})

    repo = Repository(tmp_path / "project")
    cfg = replace(config, target_count=1, max_retries=5, max_attempts=2)
    job = repo.create_job(cfg, [TextItem("A source sentence.", "A source sentence.")])
    result = Pipeline(repo, tts=BrokenTTS()).run(job)
    assert result["accepted"] == 0 and result["attempts"] == 2
    assert result["status"] == "NEEDS_ATTENTION"
    assert not list((repo.root / "audio").glob("*.wav"))


def test_qc_disabled_does_not_load_optional_backends(tmp_path, config):
    from nc_factory.audio import write_wav_atomic
    from nc_factory.quality import QualityController
    from nc_factory.tts import MockTTS

    audio = MockTTS().synthesize("A source sentence.", config.voice, config.synthesis, 0)
    path = tmp_path / "clip.wav"
    write_wav_atomic(path, audio.samples, audio.sample_rate)
    checker = QualityController(config.quality)
    checker.prepare(config.voice)
    result = checker.check(path, "A source sentence.", "en", config.voice)
    assert result.passed and result.metrics["asr_check"] == "disabled"
    assert checker.asr is None and checker.speaker is None

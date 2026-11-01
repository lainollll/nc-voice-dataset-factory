from __future__ import annotations

from dataclasses import replace


def test_recovery_reconciles_committed_acceptance_with_unfinished_attempt_journal(tmp_path, config):
    from nc_factory.domain import TextItem
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    repo = Repository(tmp_path / "project")
    job = repo.create_job(
        replace(config, target_count=1),
        [TextItem("Please bring the blue book.", "Please bring the blue book.")],
    )
    Pipeline(repo, tts=MockTTS()).run(job)
    sample = next(repo.iter_samples(state="ACCEPTED"))
    # Simulate a crash after committing accepted sample/QC, before finalizing its attempt row.
    repo.finish_attempt(sample["latest_attempt_id"], state="GENERATED")
    repo.recover()
    assert repo.get_sample(sample["id"])["state"] == "ACCEPTED"
    assert repo.attempts(sample["id"])[-1]["state"] == "ACCEPTED"
    assert repo.statistics()["accepted"] == 1

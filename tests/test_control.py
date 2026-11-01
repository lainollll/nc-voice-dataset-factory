from __future__ import annotations

import threading
import time
from dataclasses import replace


def test_durable_external_pause_resume_stop(tmp_path, config):
    from nc_factory.control import request_job_control
    from nc_factory.domain import TextItem
    from nc_factory.pipeline import Pipeline
    from nc_factory.storage import Repository
    from nc_factory.tts import MockTTS

    repo = Repository(tmp_path / "project")
    job = repo.create_job(
        replace(config, target_count=1),
        [TextItem("A source sentence for control testing.", "A source sentence for control testing.")],
    )
    request_job_control(repo.root, job, "pause")
    pipeline = Pipeline(repo, tts=MockTTS())
    errors = []

    def run():
        try:
            pipeline.run(job)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 5
    while repo.get_job(job)["status"] != "PAUSED" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert repo.get_job(job)["status"] == "PAUSED"
    request_job_control(repo.root, job, "resume")
    thread.join(5)
    assert not thread.is_alive() and not errors
    assert repo.statistics()["accepted"] == 1


def test_chunked_template_source_preserves_category_balance(tmp_path, config):
    from collections import Counter
    from nc_factory.storage import Repository
    from nc_factory.text import ProjectTextSource

    repo = Repository(tmp_path / "project")
    job = repo.create_job(config, [])
    source = ProjectTextSource(config, repo, job)
    items = []
    for _ in range(10):
        items.extend(source(32))
    counts = Counter(item.category for item in items)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert len({item.spoken_text for item in items}) == 320

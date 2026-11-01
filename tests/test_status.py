from __future__ import annotations

from dataclasses import replace


def test_status_and_control_support_utf8_hash_paths(tmp_path, config):
    from nc_factory.control import read_project_status, request_job_control, take_job_control
    from nc_factory.storage import Repository

    repo = Repository(tmp_path / "öst#data")
    job = repo.create_job(replace(config, target_count=1), [])
    request_job_control(repo.root, job, "pause")
    assert take_job_control(repo.root, job) == "pause"
    status = read_project_status(repo.root)
    assert status["jobs"][0]["id"] == job
    assert status["accepted"] == 0

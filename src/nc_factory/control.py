"""Durable controls for a running CLI worker without acquiring its writer lock."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from pathlib import Path

from .storage import atomic_json, utcnow


def request_job_control(root: Path, job_id: str, command: str) -> None:
    if command not in {"pause", "resume", "stop"} or not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ValueError("A valid job ID and pause/resume/stop command are required")
    root = Path(root).resolve()
    db = sqlite3.connect((root / "factory.sqlite3").as_uri() + "?mode=ro", uri=True)
    try:
        if db.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone() is None:
            raise ValueError("Unknown project job")
    finally:
        db.close()
    atomic_json(root / ".controls" / f"{job_id}.json", {"command": command, "requested_at": utcnow()})


def read_project_status(root: Path) -> dict:
    """Read compact progress without taking the active generation writer lock."""
    db = sqlite3.connect((Path(root).resolve() / "factory.sqlite3").as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        jobs = [
            dict(row)
            for row in db.execute(
                "SELECT id,status,command,created_at,started_at,finished_at,error FROM jobs ORDER BY created_at DESC LIMIT 20"
            )
        ]
        accepted = db.execute(
            "SELECT COUNT(*),COALESCE(SUM(duration),0) FROM samples WHERE state='ACCEPTED' AND integrity_hold=0 AND training_revision=validated_revision"
        ).fetchone()
        states = {row[0]: row[1] for row in db.execute("SELECT state,COUNT(*) FROM samples GROUP BY state")}
        return {
            "jobs": jobs,
            "accepted": accepted[0],
            "accepted_seconds": accepted[1],
            "states": states,
            "attempts": db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
        }
    finally:
        db.close()


def take_job_control(root: Path, job_id: str) -> str | None:
    source = root / ".controls" / f"{job_id}.json"
    consumed = source.with_name(f".consumed-{uuid.uuid4().hex}.json")
    try:
        os.rename(source, consumed)
    except FileNotFoundError:
        return None
    try:
        command = json.loads(consumed.read_text(encoding="utf-8"))["command"]
        if command not in {"pause", "resume", "stop"}:
            raise ValueError("Invalid persisted job control")
        return command
    finally:
        consumed.unlink(missing_ok=True)

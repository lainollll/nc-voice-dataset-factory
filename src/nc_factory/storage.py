from __future__ import annotations

import getpass
import json
import os
import re
import sqlite3
import tempfile
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from filelock import FileLock

from .domain import ProjectConfig, TextItem, VoiceProfile

JSON_FIELDS = {"provenance", "metrics", "reasons", "parameters", "payload"}
SAMPLE_FIELDS = set(
    "state latest_attempt_id path original_path duration sha256 audio_fingerprint asr_text wer cer speaker_similarity metrics reasons parameters model_id integrity_hold training_revision validated_revision training_text".split()
)
ATTEMPT_FIELDS = set(
    "state path original_path duration sha256 metrics reasons parameters model_id error".split()
)
ACTIVE = {"GENERATING", "GENERATED", "VALIDATING"}
SAMPLE_FIELDS.add("generation_text")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold()))


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".nc-json-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)  # Config snapshots are replaceable; audio is not.
    finally:
        Path(name).unlink(missing_ok=True)


def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    data = dict(row)
    for key in JSON_FIELDS & data.keys():
        data[key] = json.loads(data[key]) if data[key] else ([] if key == "reasons" else {})
    return data


class Repository:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "factory.sqlite3"
        self._lock = FileLock(str(self.root / ".writer.lock"), timeout=0)
        with self.writer_lock():
            for name in ("audio", "original", "rejected", "references", "logs", "exports"):
                (self.root / name).mkdir(exist_ok=True)
            with self.connection() as db:
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version not in {0, 1, 2}:
                    raise ValueError(f"Unsupported project database version: {version}")
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS config(id INTEGER PRIMARY KEY CHECK(id=1), json TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS voices(id TEXT PRIMARY KEY, json TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS jobs(
                        id TEXT PRIMARY KEY, config_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'READY',
                        command TEXT NOT NULL DEFAULT 'run', created_at TEXT NOT NULL,
                        started_at TEXT, finished_at TEXT, error TEXT NOT NULL DEFAULT '');
                    CREATE TABLE IF NOT EXISTS samples(
                        id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                        original_text TEXT NOT NULL, spoken_text TEXT NOT NULL, spoken_key TEXT NOT NULL,
                        training_text TEXT NOT NULL, category TEXT, language TEXT NOT NULL,
                        provenance TEXT NOT NULL DEFAULT '{}', state TEXT NOT NULL DEFAULT 'PENDING',
                        attempt_count INTEGER NOT NULL DEFAULT 0, latest_attempt_id TEXT,
                        path TEXT, original_path TEXT, duration REAL DEFAULT 0, sha256 TEXT,
                        audio_fingerprint TEXT, asr_text TEXT, wer REAL, cer REAL, speaker_similarity REAL,
                        metrics TEXT NOT NULL DEFAULT '{}', reasons TEXT NOT NULL DEFAULT '[]',
                        parameters TEXT NOT NULL DEFAULT '{}', model_id TEXT,
                        integrity_hold INTEGER NOT NULL DEFAULT 0, training_revision INTEGER NOT NULL DEFAULT 0,
                        validated_revision INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                        UNIQUE(language, spoken_key));
                    CREATE INDEX IF NOT EXISTS samples_job_state ON samples(job_id,state,created_at);
                    CREATE INDEX IF NOT EXISTS samples_fingerprint ON samples(audio_fingerprint,state);
                    CREATE TABLE IF NOT EXISTS attempts(
                        id TEXT PRIMARY KEY, sample_id TEXT NOT NULL REFERENCES samples(id),
                        seed INTEGER NOT NULL, state TEXT NOT NULL, parameters TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL, finished_at TEXT, path TEXT, original_path TEXT,
                        duration REAL, sha256 TEXT, metrics TEXT DEFAULT '{}', reasons TEXT DEFAULT '[]',
                        model_id TEXT, error TEXT);
                    CREATE INDEX IF NOT EXISTS attempts_sample ON attempts(sample_id,created_at);
                    CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
                    CREATE INDEX IF NOT EXISTS events_sample ON events(json_extract(payload,'$.sample_id'));
                """)
                if "generation_text" not in {row[1] for row in db.execute("PRAGMA table_info(samples)")}:
                    db.execute("ALTER TABLE samples ADD COLUMN generation_text TEXT")
                if "attempt_budget_override" not in {row[1] for row in db.execute("PRAGMA table_info(jobs)")}:
                    db.execute(
                        "ALTER TABLE jobs ADD COLUMN attempt_budget_override INTEGER NOT NULL DEFAULT 0"
                    )
                db.execute("PRAGMA user_version=2")

    @contextmanager
    def writer_lock(self) -> Iterator[None]:
        # Thread-local reentrancy allows the pipeline's nested writes, not other threads.
        with self._lock:
            yield

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.db_path, timeout=5)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA busy_timeout=5000")
            with db:
                yield db
        finally:
            db.close()

    def _event(self, db: sqlite3.Connection, kind: str, payload: dict[str, Any]) -> None:
        db.execute(
            "INSERT INTO events(created_at,kind,payload) VALUES(?,?,?)",
            (utcnow(), kind, json.dumps(payload, ensure_ascii=False, allow_nan=False)),
        )

    def record_event(self, kind: str, payload: dict[str, Any]) -> None:
        with self.writer_lock(), self.connection() as db:
            self._event(db, kind, payload)

    def events(self, sample_id: str | None = None) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM events"
                + (" WHERE json_extract(payload,'$.sample_id')=?" if sample_id else "")
                + " ORDER BY id",
                [sample_id] if sample_id else [],
            ).fetchall()
        result = [_decode(row) for row in rows]
        return [
            r
            for r in result
            if r is not None and (sample_id is None or r["payload"].get("sample_id") == sample_id)
        ]

    def save_config(self, config: ProjectConfig) -> None:
        config.validate(check_reference=False)
        with self.writer_lock(), self.connection() as db:
            db.execute(
                "INSERT INTO config VALUES(1,?) ON CONFLICT(id) DO UPDATE SET json=excluded.json",
                (json.dumps(config.to_dict(), ensure_ascii=False, allow_nan=False),),
            )
            self._event(db, "configuration_saved", {"name": config.name})
        atomic_json(self.root / "project_config.json", config.to_dict())

    def load_config(self) -> ProjectConfig:
        with self.connection() as db:
            row = db.execute("SELECT json FROM config WHERE id=1").fetchone()
        if row is None:
            raise ValueError("No project configuration has been saved")
        return ProjectConfig.from_dict(json.loads(row[0]))

    def save_voice(self, voice: VoiceProfile) -> None:
        voice.validate()
        with self.writer_lock(), self.connection() as db:
            db.execute(
                "INSERT INTO voices VALUES(?,?) ON CONFLICT(id) DO UPDATE SET json=excluded.json",
                (voice.id, json.dumps(asdict(voice), ensure_ascii=False)),
            )

    def voices(self) -> list[dict[str, Any]]:
        with self.connection() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT json FROM voices ORDER BY id")]

    def create_job(self, config: ProjectConfig, items: Sequence[TextItem]) -> str:
        config.validate()
        job_id = uuid.uuid4().hex
        with self.writer_lock(), self.connection() as db:
            db.execute(
                "INSERT INTO jobs(id,config_json,created_at) VALUES(?,?,?)",
                (job_id, json.dumps(config.to_dict(), ensure_ascii=False), utcnow()),
            )
            self._event(db, "job_created", {"job_id": job_id})
        self.add_texts(job_id, items)
        return job_id

    def add_texts(self, job_id: str, items: Sequence[TextItem]) -> int:
        added = 0
        with self.writer_lock(), self.connection() as db:
            for item in items:
                key = canonical_text(item.spoken_text)
                if not key:
                    continue
                cursor = db.execute(
                    """INSERT OR IGNORE INTO samples(
                    id,job_id,original_text,spoken_text,spoken_key,training_text,category,language,provenance,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        uuid.uuid4().hex,
                        job_id,
                        item.original_text,
                        item.spoken_text,
                        key,
                        item.spoken_text,
                        item.category,
                        item.language,
                        json.dumps(item.provenance, ensure_ascii=False),
                        utcnow(),
                    ),
                )
                added += cursor.rowcount
        return added

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown job: {job_id}")
        return dict(row)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self.connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM jobs ORDER BY created_at DESC")]

    def update_job(self, job_id: str, **fields: Any) -> None:
        self._update("jobs", job_id, fields, {"status", "command", "error", "started_at", "finished_at"})

    def extend_attempt_budget(self, job_id: str, additional: int, reason: str) -> None:
        if not isinstance(additional, int) or not 1 <= additional <= 10_000_000 or not reason.strip():
            raise ValueError("A positive additional budget and review reason are required")
        with self.writer_lock(), self.connection() as db:
            job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job is None or job["status"] in {"RUNNING", "PAUSED"}:
                raise ValueError("Stop the selected job before extending its attempt budget")
            previous = max(json.loads(job["config_json"])["max_attempts"], job["attempt_budget_override"])
            db.execute(
                "UPDATE jobs SET attempt_budget_override=? WHERE id=?", (previous + additional, job_id)
            )
            self._event(
                db,
                "attempt_budget_extended",
                {
                    "job_id": job_id,
                    "previous": previous,
                    "new": previous + additional,
                    "reason": reason.strip(),
                    "actor": getpass.getuser(),
                },
            )

    def _update(self, table: str, identifier: str, fields: dict[str, Any], allowed: set[str]) -> None:
        if not fields or fields.keys() - allowed:
            raise ValueError(f"Invalid {table} fields: {fields.keys() - allowed}")
        values = [
            json.dumps(v, ensure_ascii=False, allow_nan=False) if k in JSON_FIELDS else v
            for k, v in fields.items()
        ]
        with self.writer_lock(), self.connection() as db:
            cursor = db.execute(
                f"UPDATE {table} SET {','.join(k + '=?' for k in fields)} WHERE id=?", [*values, identifier]
            )
            if cursor.rowcount != 1:
                raise KeyError(identifier)

    def list_samples(
        self,
        job_id: str | None = None,
        state: str | None = None,
        search: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        filters, args = [], []
        if job_id:
            filters.append("job_id=?")
            args.append(job_id)
        if state:
            filters.append("state=?")
            args.append(state)
        else:
            filters.append("state!='DELETED'")
        if search:
            filters.append("(training_text LIKE ? OR category LIKE ? OR asr_text LIKE ? OR path LIKE ?)")
            args.extend([f"%{search}%"] * 4)
        query = "SELECT * FROM samples WHERE " + " AND ".join(filters)
        with self.connection() as db:
            rows = db.execute(
                query + " ORDER BY created_at,id LIMIT ? OFFSET ?",
                [*args, min(max(limit, 1), 1000), max(0, offset)],
            ).fetchall()
        return [data for row in rows if (data := _decode(row)) is not None]

    def iter_samples(self, job_id: str | None = None, state: str | None = None) -> Iterator[dict[str, Any]]:
        offset = 0
        while rows := self.list_samples(job_id=job_id, state=state, limit=1000, offset=offset):
            yield from rows
            offset += len(rows)

    def get_sample(self, sample_id: str) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT * FROM samples WHERE id=?", (sample_id,)).fetchone()
        data = _decode(row)
        if data is None:
            raise KeyError(sample_id)
        return data

    def update_sample(self, sample_id: str, **fields: Any) -> None:
        self._update("samples", sample_id, fields, SAMPLE_FIELDS)

    def next_pending(self, job_id: str) -> dict[str, Any] | None:
        with self.connection() as db:
            return _decode(
                db.execute(
                    "SELECT * FROM samples WHERE job_id=? AND state='PENDING' AND integrity_hold=0 ORDER BY created_at,id LIMIT 1",
                    (job_id,),
                ).fetchone()
            )

    def begin_attempt(self, sample_id: str, seed: int, parameters: dict[str, Any]) -> str:
        attempt_id = uuid.uuid4().hex
        with self.writer_lock(), self.connection() as db:
            changed = db.execute(
                "UPDATE samples SET state='GENERATING',attempt_count=attempt_count+1,latest_attempt_id=?,metrics='{}',reasons='[]',asr_text=NULL,wer=NULL,cer=NULL,speaker_similarity=NULL WHERE id=? AND state='PENDING'",
                (attempt_id, sample_id),
            )
            if changed.rowcount != 1:
                raise ValueError("Only a pending sample can start a new attempt")
            db.execute(
                "INSERT INTO attempts(id,sample_id,seed,state,parameters,created_at) VALUES(?,?,?,'GENERATING',?,?)",
                (attempt_id, sample_id, seed, json.dumps(parameters), utcnow()),
            )
            self._event(
                db, "attempt_started", {"sample_id": sample_id, "attempt_id": attempt_id, "seed": seed}
            )
        return attempt_id

    def finish_attempt(self, attempt_id: str, **fields: Any) -> None:
        self._update(
            "attempts", attempt_id, {**fields, "finished_at": utcnow()}, ATTEMPT_FIELDS | {"finished_at"}
        )

    def attempts(self, sample_id: str) -> list[dict[str, Any]]:
        with self.connection() as db:
            return [
                _decode(row)
                for row in db.execute(
                    "SELECT * FROM attempts WHERE sample_id=? ORDER BY created_at,id", (sample_id,)
                )
            ]

    def find_audio_duplicate(self, value: str, excluding_id: str) -> bool:
        with self.connection() as db:
            return (
                db.execute(
                    "SELECT 1 FROM samples WHERE audio_fingerprint=? AND id!=? AND state='ACCEPTED' AND integrity_hold=0 LIMIT 1",
                    (value, excluding_id),
                ).fetchone()
                is not None
            )

    def statistics(self, job_id: str | None = None) -> dict[str, Any]:
        clause, args = (" AND job_id=?", [job_id]) if job_id else ("", [])
        stats: dict[str, Any] = dict(
            accepted=0,
            rejected=0,
            failed=0,
            pending=0,
            total=0,
            accepted_seconds=0.0,
            attempts=0,
            integrity_holds=0,
        )
        with self.connection() as db:
            rows = db.execute(
                "SELECT state,integrity_hold,training_revision=validated_revision AS valid,COUNT(*) AS n,SUM(duration) AS seconds,SUM(attempt_count) AS attempts FROM samples WHERE state!='DELETED'"
                + clause
                + " GROUP BY state,integrity_hold,valid",
                args,
            )
            for row in rows:
                stats["total"] += row["n"]
                stats["attempts"] += row["attempts"] or 0
                stats["integrity_holds"] += row["n"] if row["integrity_hold"] else 0
                key = row["state"].lower()
                if key in {"accepted", "rejected", "failed", "pending"}:
                    if key == "accepted" and (row["integrity_hold"] or not row["valid"]):
                        continue
                    stats[key] += row["n"]
                    if key == "accepted":
                        stats["accepted_seconds"] += row["seconds"] or 0
            query = "SELECT COUNT(*) FROM attempts a JOIN samples s ON s.id=a.sample_id"
            stats["attempts"] = db.execute(query + (" WHERE s.job_id=?" if job_id else ""), args).fetchone()[
                0
            ]
        return stats

    def _safe_audio(self, value: str) -> Path:
        path = Path(value).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Dataset audio path escapes the project directory")
        return path

    def review(self, sample_id: str, action: str, reason: str) -> dict[str, Any]:
        from .audio import read_audio, sha256_file

        if action not in {"accept", "reject", "retry", "regenerate", "delete"} or not reason.strip():
            raise ValueError("Review requires a valid action and non-empty reason")
        with self.writer_lock():
            sample = self.get_sample(sample_id)
            if sample["state"] in ACTIVE:
                raise ValueError("Cannot review an active sample")
            fields: dict[str, Any] = {
                "state": {
                    "accept": "ACCEPTED",
                    "reject": "REJECTED",
                    "retry": "PENDING",
                    "regenerate": "PENDING",
                    "delete": "DELETED",
                }[action],
                "reasons": [reason],
            }
            if action == "accept":
                path = self._safe_audio(sample["path"] or "")
                if sha256_file(path) != sample["sha256"]:
                    raise ValueError("Audio hash changed; manual approval refused")
                read_audio(path)
                fields.update(
                    metrics={**sample["metrics"], "manual_override": True},
                    integrity_hold=0,
                    validated_revision=sample["training_revision"],
                )
            elif action in {"retry", "regenerate"}:
                fields["integrity_hold"] = 0
                if action == "regenerate":
                    fields["generation_text"] = sample["training_text"]
            self.update_sample(sample_id, **fields)
            self.record_event(
                "sample_reviewed",
                {
                    "sample_id": sample_id,
                    "action": action,
                    "reason": reason.strip(),
                    "actor": getpass.getuser(),
                },
            )
        return self.get_sample(sample_id)

    def edit_training_text(self, sample_id: str, training_text: str) -> None:
        if not training_text.strip():
            raise ValueError("Training text cannot be empty")
        with self.writer_lock():
            sample = self.get_sample(sample_id)
            if sample["state"] in ACTIVE:
                raise ValueError("Cannot edit an active sample")
            self.update_sample(
                sample_id,
                training_text=training_text.strip(),
                training_revision=sample["training_revision"] + 1,
                state="REJECTED",
                reasons=["revalidation_required"],
            )
            self.record_event(
                "training_text_edited",
                {
                    "sample_id": sample_id,
                    "old": sample["training_text"],
                    "new": training_text.strip(),
                    "actor": getpass.getuser(),
                },
            )

    def recover(self) -> dict[str, Any]:
        from .audio import sha256_file

        report: dict[str, Any] = {"interrupted_reset": 0, "integrity_holds": 0, "orphans": []}
        with self.writer_lock(), self.connection() as db:
            active = list(
                db.execute(
                    "SELECT id,latest_attempt_id FROM samples WHERE state IN ('GENERATING','GENERATED','VALIDATING')"
                )
            )
            for row in active:
                db.execute("UPDATE samples SET state='PENDING',integrity_hold=0 WHERE id=?", (row["id"],))
                db.execute(
                    "UPDATE attempts SET state='FAILED',error='Interrupted before durable completion',finished_at=? WHERE id=?",
                    (utcnow(), row["latest_attempt_id"]),
                )
                self._event(db, "interrupted_recovered", {"sample_id": row["id"]})
                report["interrupted_reset"] += 1
            for row in db.execute("SELECT * FROM samples WHERE state='ACCEPTED'"):
                try:
                    bad = not row["path"] or sha256_file(self._safe_audio(row["path"])) != row["sha256"]
                except (OSError, ValueError):
                    bad = True
                if bad:
                    db.execute("UPDATE samples SET integrity_hold=1 WHERE id=?", (row["id"],))
                    self._event(db, "accepted_integrity_hold", {"sample_id": row["id"]})
                    report["integrity_holds"] += 1
                elif not row["integrity_hold"] and row["training_revision"] == row["validated_revision"]:
                    # The sample's accepted QC transaction may have committed immediately
                    # before a crash interrupted final attempt journaling. Reconcile only
                    # unfinished journal states, never historical manual review decisions.
                    changed = db.execute(
                        "UPDATE attempts SET state='ACCEPTED',path=?,sha256=?,metrics=?,reasons=?,finished_at=? "
                        "WHERE id=? AND state IN ('GENERATING','GENERATED','VALIDATING')",
                        (
                            row["path"],
                            row["sha256"],
                            row["metrics"],
                            row["reasons"],
                            utcnow(),
                            row["latest_attempt_id"],
                        ),
                    )
                    if changed.rowcount:
                        self._event(
                            db,
                            "accepted_attempt_reconciled",
                            {"sample_id": row["id"], "attempt_id": row["latest_attempt_id"]},
                        )
            known = {row[0] for row in db.execute("SELECT path FROM attempts WHERE path IS NOT NULL")}
            known.update(
                row[0]
                for row in db.execute("SELECT original_path FROM attempts WHERE original_path IS NOT NULL")
            )
            report["orphans"] = [
                str(p)
                for folder in ("audio", "original")
                for p in (self.root / folder).glob("*")
                if p.is_file() and str(p) not in known
            ]
            db.execute(
                "UPDATE jobs SET status='STOPPED',error='Recovered after interrupted run' WHERE status='RUNNING'"
            )
        return report

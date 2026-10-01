"""Persistent, single-writer processing runs with cooperative cancellation."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import json
import sqlite3

from .storage import file_hash, fingerprint


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def run_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        self.db = sqlite3.connect(path, timeout=15)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS items (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, state TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, error TEXT, record TEXT,
                artifacts TEXT, seconds REAL, updated TEXT NOT NULL
            );
        """)

    def close(self):
        self.db.close()

    def get(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key: str, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, json.dumps(value)))

    def initialize(self, config: dict, resume: bool):
        previous = self.get("config_hash")
        if previous:
            if not resume:
                raise ValueError("Run already exists; use --resume or a different output directory")
            if previous != fingerprint(config):
                raise ValueError("Inputs, code, assets or options changed; use a new output directory")
        else:
            self.set("config_hash", fingerprint(config))
            self.set("config", config)
            self.set("created", now())
        self.set("cancel_requested", False)
        self.set("state", "running")
        with self.db:
            self.db.execute("UPDATE items SET state='pending' WHERE state='running'")

    def completed(self, sample_id: str, directory: Path) -> bool:
        row = self.db.execute("SELECT state,artifacts FROM items WHERE id=?", (sample_id,)).fetchone()
        if row is None or row["state"] != "completed":
            return False
        artifacts = json.loads(row["artifacts"] or "{}")
        return bool(artifacts) and all((directory / name).is_file() and file_hash(directory / name) == sha
                                       for name, sha in artifacts.items())

    def begin(self, sample_id: str, source: str):
        with self.db:
            self.db.execute("""INSERT INTO items(id,source,state,attempts,updated) VALUES (?,?,'running',1,?)
                ON CONFLICT(id) DO UPDATE SET state='running',attempts=attempts+1,error=NULL,updated=excluded.updated""",
                            (sample_id, source, now()))

    def finish(self, sample_id: str, seconds: float, record: str, artifacts: dict):
        with self.db:
            self.db.execute("""UPDATE items SET state='completed',seconds=?,record=?,artifacts=?,error=NULL,
                updated=? WHERE id=?""", (seconds, record, json.dumps(artifacts), now(), sample_id))

    def fail(self, sample_id: str, error: str):
        with self.db:
            self.db.execute("UPDATE items SET state='failed',error=?,updated=? WHERE id=?", (error, now(), sample_id))

    def records(self):
        return self.db.execute("SELECT * FROM items ORDER BY rowid").fetchall()

    def summary(self) -> dict:
        counts = dict(self.db.execute("SELECT state,COUNT(*) FROM items GROUP BY state"))
        return {"state": self.get("state"), "created": self.get("created"),
                "updated": self.get("updated"), "cancel_requested": self.get("cancel_requested", False),
                "counts": counts, "discovered_items": sum(counts.values())}


def inspect_jobs(directory: Path, cancel: bool = False) -> dict:
    path = directory / "tasks.sqlite"
    if not path.is_file():
        raise FileNotFoundError(f"No run database at {path}")
    store = JobStore(path)
    try:
        if cancel:
            store.set("cancel_requested", True)
        return store.summary()
    finally:
        store.close()

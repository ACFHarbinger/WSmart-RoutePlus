"""SQLite control plane, for offline use, the local backend and tests.

Records are stored as JSON documents next to the columns the queue needs. Every
write runs in a ``BEGIN IMMEDIATE`` transaction, so leases stay exclusive across
threads and processes sharing the database file.

Attributes:
    SQLiteControlPlane: :class:`~logic.controllers.cloud.interfaces.ControlPlane` on SQLite.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from logic.controllers.cloud.interfaces import ControlPlane, StaleStateError
from logic.controllers.cloud.model import (
    Event,
    JobRecord,
    ResultFile,
    TaskRecord,
    TaskState,
    check_transition,
    job_from_dict,
    task_from_dict,
    to_dict,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, created_at REAL NOT NULL, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, seq INTEGER NOT NULL,
    state TEXT NOT NULL, lease_expires REAL, doc TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS tasks_queue ON tasks (state, seq);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, at REAL NOT NULL, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, at REAL NOT NULL, doc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS results (
    task_id TEXT NOT NULL, relpath TEXT NOT NULL, doc TEXT NOT NULL, PRIMARY KEY (task_id, relpath));
"""

_LEASED = (TaskState.PLACING.value,)


class SQLiteControlPlane(ControlPlane):
    """Control plane on one SQLite database (a file, or ``":memory:"``)."""

    def __init__(self, path: Union[str, Path] = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL" if str(path) != ":memory:" else "PRAGMA journal_mode=MEMORY")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def _read(self, sql: str, args: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    # Jobs -----------------------------------------------------------------

    def create_job(self, job: JobRecord, tasks: Sequence[TaskRecord]) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, created_at, doc) VALUES (?, ?, ?)",
                (job.job_id, job.created_at, json.dumps(to_dict(job))),
            )
            start = conn.execute("SELECT COALESCE(MAX(seq), 0) FROM tasks").fetchone()[0]
            for offset, task in enumerate(tasks, start=1):
                conn.execute(
                    "INSERT INTO tasks (task_id, job_id, seq, state, lease_expires, doc) VALUES (?, ?, ?, ?, ?, ?)",
                    (task.task_id, job.job_id, start + offset, TaskState(task.state).value, None, _doc(task)),
                )

    def get_job(self, job_id: str) -> JobRecord:
        rows = self._read("SELECT doc FROM jobs WHERE job_id = ?", (job_id,))
        if not rows:
            raise KeyError(job_id)
        return job_from_dict(json.loads(rows[0][0]))

    def list_jobs(self, limit: int = 50) -> List[JobRecord]:
        rows = self._read("SELECT doc FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,))
        return [job_from_dict(json.loads(row[0])) for row in rows]

    def request_cancel(self, job_id: str) -> None:
        with self._tx() as conn:
            row = conn.execute("SELECT doc FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            doc = json.loads(row[0])
            doc["cancel_requested"] = True
            conn.execute("UPDATE jobs SET doc = ? WHERE job_id = ?", (json.dumps(doc), job_id))

    # Tasks ----------------------------------------------------------------

    def get_task(self, task_id: str) -> TaskRecord:
        rows = self._read("SELECT doc FROM tasks WHERE task_id = ?", (task_id,))
        if not rows:
            raise KeyError(task_id)
        return task_from_dict(json.loads(rows[0][0]))

    def list_tasks(self, job_id: str) -> List[TaskRecord]:
        rows = self._read("SELECT doc FROM tasks WHERE job_id = ? ORDER BY seq", (job_id,))
        return [task_from_dict(json.loads(row[0])) for row in rows]

    def lease_task(self, owner: str, lease_seconds: float, job_id: Optional[str] = None) -> Optional[TaskRecord]:
        now = time.time()
        with self._tx() as conn:
            sql = "SELECT doc FROM tasks WHERE state = ?"
            args: List[Any] = [TaskState.QUEUED.value]
            if job_id is not None:
                sql += " AND job_id = ?"
                args.append(job_id)
            row = conn.execute(sql + " ORDER BY seq LIMIT 1", args).fetchone()
            if row is None:
                return None
            task = task_from_dict(json.loads(row[0]))
            task.state = TaskState.PLACING
            task.lease_owner = owner
            task.lease_expires = now + lease_seconds
            self._write_task(conn, task)
            return task

    def renew_lease(self, task_id: str, owner: str, lease_seconds: float) -> bool:
        with self._tx() as conn:
            task = self._task_in(conn, task_id)
            if task.lease_owner != owner or task.state.value not in _LEASED:
                return False
            task.lease_expires = time.time() + lease_seconds
            self._write_task(conn, task)
            return True

    def expire_leases(self, now: Optional[float] = None) -> List[str]:
        now = time.time() if now is None else now
        expired = []
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT doc FROM tasks WHERE state IN (%s) AND lease_expires < ?" % ",".join("?" * len(_LEASED)),
                (*_LEASED, now),
            ).fetchall()
            for row in rows:
                task = task_from_dict(json.loads(row[0]))
                task.state = TaskState.QUEUED
                task.lease_owner = None
                task.lease_expires = None
                self._write_task(conn, task)
                expired.append(task.task_id)
        return expired

    def update_task(
        self,
        task_id: str,
        state: Optional[TaskState] = None,
        expected: Optional[TaskState] = None,
        **fields: Any,
    ) -> TaskRecord:
        with self._tx() as conn:
            task = self._task_in(conn, task_id)
            if expected is not None and task.state != TaskState(expected):
                raise StaleStateError(f"{task_id}: expected {TaskState(expected).value}, found {task.state.value}")
            for name, value in fields.items():
                if not hasattr(task, name) or name in ("task_id", "job_id", "state"):
                    raise AttributeError(f"cannot update task field {name!r}")
                setattr(task, name, value)
            if state is not None and TaskState(state) != task.state:
                check_transition(task.state, TaskState(state))
                task.state = TaskState(state)
                if task.state.value not in _LEASED:
                    task.lease_expires = None
            self._write_task(conn, task)
            return task

    def _task_in(self, conn: sqlite3.Connection, task_id: str) -> TaskRecord:
        row = conn.execute("SELECT doc FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return task_from_dict(json.loads(row[0]))

    def _write_task(self, conn: sqlite3.Connection, task: TaskRecord) -> None:
        conn.execute(
            "UPDATE tasks SET state = ?, lease_expires = ?, doc = ? WHERE task_id = ?",
            (task.state.value, task.lease_expires, _doc(task), task.task_id),
        )

    # Events, metrics, results ---------------------------------------------

    def append_event(self, event: Event) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO events (job_id, at, doc) VALUES (?, ?, ?)",
                (event.job_id, event.at, json.dumps(to_dict(event))),
            )

    def list_events(self, job_id: str, since: Optional[float] = None) -> List[Event]:
        rows = self._read(
            "SELECT doc FROM events WHERE job_id = ? AND at > ? ORDER BY id",
            (job_id, -1.0 if since is None else since),
        )
        return [Event(**json.loads(row[0])) for row in rows]

    def put_metrics(self, task_id: str, records: Sequence[Dict[str, Any]]) -> None:
        with self._tx() as conn:
            conn.executemany(
                "INSERT INTO metrics (task_id, at, doc) VALUES (?, ?, ?)",
                [(task_id, float(record.get("at", time.time())), json.dumps(record)) for record in records],
            )

    def get_metrics(self, task_id: str, since: Optional[float] = None) -> List[Dict[str, Any]]:
        rows = self._read(
            "SELECT doc FROM metrics WHERE task_id = ? AND at > ? ORDER BY id",
            (task_id, -1.0 if since is None else since),
        )
        return [json.loads(row[0]) for row in rows]

    def record_results(self, task_id: str, files: Sequence[ResultFile]) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM results WHERE task_id = ?", (task_id,))
            conn.executemany(
                "INSERT INTO results (task_id, relpath, doc) VALUES (?, ?, ?)",
                [(task_id, record.relpath, json.dumps(to_dict(record))) for record in files],
            )

    def list_results(self, task_id: str) -> List[ResultFile]:
        rows = self._read("SELECT doc FROM results WHERE task_id = ? ORDER BY relpath", (task_id,))
        return [ResultFile(**json.loads(row[0])) for row in rows]


def _doc(task: TaskRecord) -> str:
    return json.dumps(to_dict(task))

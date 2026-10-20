"""Connection, schema and run-row helpers for TrackingStore.

Moved from ``logic.src.tracking.core.store``.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional


class TrackingStoreConnectionMixin:
    """Moved methods of :class:`TrackingStore`."""

    def __init__(self, db_path: str) -> None:
        """Initializes the storage backend and applies the schema.

        Args:
            db_path: Absolute path to the .db file. If the file does not exist,
                it will be created along with its parent directories.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Load queries dynamically to decouple SQL from Python
        queries_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries.json")
        with open(queries_path, "r", encoding="utf-8") as f:
            self.queries = json.load(f)

        self._apply_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Yields a short-lived SQLite connection with autocommit on success.

        Configures the connection for WAL mode and normal synchronous behavior
        to optimize performance while maintaining safety.

        Yields:
            sqlite3.Connection: A managed SQLite connection instance.

        Raises:
            Exception: Re-raises any exception after rolling back the transaction.
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous  = NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _apply_schema(self) -> None:
        """Initializes the database schema and performs required migrations.

        Reads the schema.sql file and executes it. Also applies conditional
        alterations for backward compatibility.
        """
        schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self._conn() as conn:
            conn.executescript(schema_sql)
            with suppress(sqlite3.OperationalError):
                conn.execute(self.queries["alter_dataset_events"])

    @staticmethod
    def _now() -> str:
        """Returns the current UTC timestamp in ISO 8601 format.

        Returns:
            str: ISO-formatted timestamp in UTC.
        """
        return datetime.now(timezone.utc).isoformat()

    def get_or_create_experiment(self, name: str, description: str = "", tags: Optional[Dict[str, Any]] = None) -> int:
        """Retrieves an existing experiment ID or creates a new record if absent.

        Args:
            name: Human-readable name of the experiment.
            description: Optional text describing the experiment goal/setup.
            tags: Optional dictionary of metadata tags for the experiment.

        Returns:
            int: The unique internal ID of the experiment.
        """
        tags_json = json.dumps(tags or {})
        with self._conn() as conn:
            conn.execute(self.queries["insert_experiment"], (name, self._now(), description, tags_json))
            row = conn.execute(self.queries["get_experiment_id"], (name,)).fetchone()
            return int(row["id"])

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Lists all existing experiments in the store.

        Returns:
            List[Dict[str, Any]]: Experiment records containing 'id', 'name',
                'created_at', and metadata.
        """
        with self._conn() as conn:
            rows = conn.execute(self.queries["list_experiments"]).fetchall()
            return [dict(r) for r in rows]

    def create_run(
        self, run_id: str, experiment_id: int, name: Optional[str], run_type: str, artifact_dir: str
    ) -> None:
        """Creates a new tracking run record.

        Args:
            run_id: Unique UUID string identifying the run.
            experiment_id: Internal ID of the parent experiment.
            name: Optional human-readable label for the run.
            run_type: Category identifier (e.g., 'training', 'simulation').
            artifact_dir: Filesystem path to the run's artifact storage.
        """
        with self._conn() as conn:
            conn.execute(self.queries["insert_run"], (run_id, experiment_id, name, run_type, self._now(), artifact_dir))

    def finish_run(self, run_id: str, status: str = "completed", error: Optional[str] = None) -> None:
        """Marks a run as completed and records the final status.

        Args:
            run_id: UUID string identifying the run.
            status: Terminal status (e.g., 'completed', 'failed', 'killed').
            error: Optional error message or traceback summarizing the failure.
        """
        with self._conn() as conn:
            conn.execute(self.queries["update_run_status"], (status, self._now(), error, run_id))

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the full record for a specific run.

        Args:
            run_id: UUID string identifying the run.

        Returns:
            Optional[Dict[str, Any]]: The run record dictionary, or None if not found.
        """
        with self._conn() as conn:
            row = conn.execute(self.queries["get_run"], (run_id,)).fetchone()
            return dict(row) if row else None

    def list_runs(
        self, experiment_id: Optional[int] = None, run_type: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves a filtered list of runs.

        Args:
            experiment_id: Optional parent experiment internal ID filter.
            run_type: Optional category label filter.
            status: Optional terminal status filter (e.g., 'completed').

        Returns:
            List[Dict[str, Any]]: Matching run records ordered by start time.
        """
        query = self.queries["list_runs_base"]
        args: List[Any] = []
        if experiment_id is not None:
            query += " AND experiment_id = ?"
            args.append(experiment_id)
        if run_type is not None:
            query += " AND run_type = ?"
            args.append(run_type)
        if status is not None:
            query += " AND status = ?"
            args.append(status)
        query += " ORDER BY start_time DESC"

        with self._conn() as conn:
            rows = conn.execute(query, args).fetchall()
            return [dict(r) for r in rows]

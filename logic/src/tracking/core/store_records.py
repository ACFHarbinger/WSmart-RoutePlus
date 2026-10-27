"""Tag, param, metric, artifact and dataset-event records for TrackingStore.

Moved from ``logic.src.tracking.core.store``.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import TYPE_CHECKING, Any, ContextManager, Dict, List, Optional, Protocol, Tuple, TypeVar

if TYPE_CHECKING:

    class _RecordHost(Protocol):
        """Attributes provided by TrackingStoreConnectionMixin."""

        queries: Dict[str, str]

        def _conn(self) -> ContextManager[sqlite3.Connection]: ...

        def _apply_schema(self) -> None: ...

        def _now(self) -> str: ...

    _RecT = TypeVar("_RecT", bound=_RecordHost)


class TrackingStoreRecordMixin:
    """Moved methods of :class:`TrackingStore`."""

    queries: Dict[str, str]

    def set_tag(self: _RecT, run_id: str, key: str, value: str) -> None:
        """Records a metadata tag for a run.

        Args:
            run_id: UUID string identifying the run.
            key: Tag key identifier.
            value: Tag string value.
        """
        with self._conn() as conn:
            conn.execute(self.queries["insert_tag"], (run_id, key, value))

    def set_tags(self: _RecT, run_id: str, tags: Dict[str, str]) -> None:
        """Records multiple metadata tags for a run in a single transaction.

        Args:
            run_id: UUID string identifying the run.
            tags: Mapping of tag keys to string values.
        """
        with self._conn() as conn:
            conn.executemany(self.queries["insert_tag"], [(run_id, k, str(v)) for k, v in tags.items()])

    def get_tags(self: _RecT, run_id: str) -> Dict[str, str]:
        """Retrieves all tags associated with a run.

        Args:
            run_id: UUID string identifying the run.

        Returns:
            Dict[str, str]: Mapping of retrieved tag keys to their string values.
        """
        with self._conn() as conn:
            rows = conn.execute(self.queries["get_tags"], (run_id,)).fetchall()
            return {r["key"]: r["value"] for r in rows}

    def log_param(self: _RecT, run_id: str, key: str, value: Any) -> None:
        """Records a configuration parameter.

        Args:
            run_id: UUID string identifying the run.
            key: Parameter key identifier.
            value: Parameter value. Will be JSON-serialized.
        """
        with self._conn() as conn:
            conn.execute(self.queries["insert_param"], (run_id, key, _safe_json_dumps(value)))

    def log_params(self: _RecT, run_id: str, params: Dict[str, Any]) -> None:
        """Records multiple configuration parameters in a single transaction.

        Args:
            run_id: UUID string identifying the run.
            params: Mapping of parameter keys to values.
        """
        try:
            with self._conn() as conn:
                conn.executemany(
                    self.queries["insert_param"], [(run_id, k, _safe_json_dumps(v)) for k, v in params.items()]
                )
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                self._apply_schema()
                with self._conn() as conn:
                    conn.executemany(
                        self.queries["insert_param"], [(run_id, k, _safe_json_dumps(v)) for k, v in params.items()]
                    )
            else:
                raise

    def get_params(self: _RecT, run_id: str) -> Dict[str, Any]:
        """Retrieves all configuration parameters associated with a run.

        Args:
            run_id: UUID string identifying the run.

        Returns:
            Dict[str, Any]: Mapping of parameter keys to deserialized values.
        """
        with self._conn() as conn:
            rows = conn.execute(self.queries["get_params"], (run_id,)).fetchall()
            return {r["key"]: json.loads(r["value"]) for r in rows}

    def log_metric(self: _RecT, run_id: str, key: str, value: float, step: int = 0) -> None:
        """Records a single scalar metric value.

        Args:
            run_id: UUID string identifying the run.
            key: Metric identifying key.
            value: Scalar metric value.
            step: Optional logical step index. Defaults to 0.
        """
        with self._conn() as conn:
            conn.execute(self.queries["insert_metric"], (run_id, key, float(value), step, self._now()))

    def log_metrics_batch(self: _RecT, run_id: str, metrics: List[Tuple[str, float, int]]) -> None:
        """Records a batch of metric values for improved performance.

        Args:
            run_id: UUID string identifying the run.
            metrics: List of (key, value, step) triples to be inserted.
        """
        now = self._now()
        try:
            with self._conn() as conn:
                conn.executemany(self.queries["insert_metric"], [(run_id, k, float(v), s, now) for k, v, s in metrics])
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                self._apply_schema()
                with self._conn() as conn:
                    conn.executemany(
                        self.queries["insert_metric"], [(run_id, k, float(v), s, now) for k, v, s in metrics]
                    )
            else:
                raise

    def get_metric_history(self: _RecT, run_id: str, key: str) -> List[Dict[str, Any]]:
        """Retrieves the full evolution history for a specific metric.

        Args:
            run_id: UUID string identifying the run.
            key: Metric identifying key.

        Returns:
            List[Dict[str, Any]]: Historical records ordered by step/time.
        """
        with self._conn() as conn:
            rows = conn.execute(self.queries["get_metric_history"], (run_id, key)).fetchall()
            return [dict(r) for r in rows]

    def get_latest_metrics(self: _RecT, run_id: str) -> Dict[str, float]:
        """Retrieves the most recent value for every metric logged in a run.

        Args:
            run_id: UUID string identifying the run.

        Returns:
            Dict[str, float]: Mapping of metric keys to their latest logged scalars.
        """
        with self._conn() as conn:
            rows = conn.execute(self.queries["get_latest_metrics"], (run_id,)).fetchall()
            return {r["key"]: r["value"] for r in rows}

    def log_artifact(
        self: _RecT,
        run_id: str,
        name: str,
        path: str,
        artifact_type: str = "file",
        file_hash: Optional[str] = None,
        size_bytes: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registers a local file as an artifact record.

        Args:
            run_id: UUID string identifying the run.
            name: Human-readable label for the artifact.
            path: Relative path to the file within the artifact storage.
            artifact_type: Category identifier (e.g., 'model', 'plot').
            file_hash: Optional SHA-256 hash of the file content.
            size_bytes: Optional file size in bytes.
            metadata: Optional dictionary of additional contextual metadata.
        """
        try:
            with self._conn() as conn:
                existing = conn.execute(self.queries["check_artifact"], (run_id, path)).fetchone()
                if existing:
                    return
                conn.execute(
                    self.queries["insert_artifact"],
                    (run_id, name, path, artifact_type, file_hash, size_bytes, self._now(), json.dumps(metadata or {})),
                )
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                self._apply_schema()
                with self._conn() as conn:
                    existing = conn.execute(self.queries["check_artifact"], (run_id, path)).fetchone()
                    if existing:
                        return
                    conn.execute(
                        self.queries["insert_artifact"],
                        (
                            run_id,
                            name,
                            path,
                            artifact_type,
                            file_hash,
                            size_bytes,
                            self._now(),
                            json.dumps(metadata or {}),
                        ),
                    )
            else:
                raise

    def get_artifacts(self: _RecT, run_id: str, artifact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves the metadata for artifacts registered in a run.

        Args:
            run_id: UUID string identifying the run.
            artifact_type: Optional category filter.

        Returns:
            List[Dict[str, Any]]: Matching artifact metadata records.
        """
        query = self.queries["get_artifacts_base"]
        args: List[Any] = [run_id]
        if artifact_type:
            query += " AND artifact_type = ?"
            args.append(artifact_type)
        with self._conn() as conn:
            rows = conn.execute(query, args).fetchall()
            return [dict(r) for r in rows]

    def log_dataset_event(
        self: _RecT,
        run_id: str,
        event_type: str,
        file_path: Optional[str] = None,
        file_hash: Optional[str] = None,
        prev_hash: Optional[str] = None,
        size_bytes: Optional[int] = None,
        shape: Optional[tuple] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Records a data-lifecycle event for lineage tracking.

        Args:
            run_id: UUID string identifying the run.
            event_type: Category (e.g., 'load', 'save', 'mutate').
            file_path: Path to the associated file.
            file_hash: Current SHA-256 hash of the file.
            prev_hash: Previous known SHA-256 hash for version tracking.
            size_bytes: Current file size in bytes.
            shape: Optional logical shape of the dataset if applicable.
            metadata: Optional dictionary of extra experiment-specific info.
        """
        with self._conn() as conn:
            conn.execute(
                self.queries["insert_dataset_event"],
                (
                    run_id,
                    event_type,
                    file_path,
                    file_hash,
                    prev_hash,
                    size_bytes,
                    str(tuple(shape)) if shape is not None else None,
                    json.dumps(metadata or {}),
                    self._now(),
                ),
            )

    def get_dataset_events(self: _RecT, run_id: str) -> List[Dict[str, Any]]:
        """Retrieves the data-lifecycle events for a specific run.

        Args:
            run_id: UUID string identifying the run.

        Returns:
            List[Dict[str, Any]]: Event records in chronological order.
        """
        with self._conn() as conn:
            rows = conn.execute(self.queries["get_dataset_events"], (run_id,)).fetchall()
            return [dict(r) for r in rows]

    def log_run_event(
        self: _RecT, run_id: str, kind: str, data: Optional[Dict[str, Any]] = None, at: Optional[float] = None
    ) -> None:
        """Records a lifecycle event of a run.

        Args:
            run_id: UUID string identifying the run.
            kind: Event kind (e.g., 'submitted', 'state', 'retry').
            data: JSON-serialisable payload.
            at: Unix time of the event. Defaults to now.
        """
        with self._conn() as conn:
            conn.execute(
                self.queries["insert_run_event"],
                (run_id, kind, _safe_json_dumps(data or {}), float(time.time() if at is None else at)),
            )

    def get_run_events(self: _RecT, run_id: str) -> List[Dict[str, Any]]:
        """Returns a run's lifecycle events, oldest first.

        Args:
            run_id: UUID string identifying the run.

        Returns:
            List of ``{"kind", "data", "timestamp"}`` dictionaries.
        """
        with self._conn() as conn:
            rows = conn.execute(self.queries["get_run_events"], (run_id,)).fetchall()
        return [{"kind": r[0], "data": json.loads(r[1] or "{}"), "timestamp": r[2]} for r in rows]

    def clear_run_data(self: _RecT, run_id: str) -> None:
        """Deletes a run's metrics and events, so an import can be repeated.

        Args:
            run_id: UUID string identifying the run.
        """
        with self._conn() as conn:
            conn.execute(self.queries["delete_run_metrics"], (run_id,))
            conn.execute(self.queries["delete_run_events"], (run_id,))


def _safe_json_dumps(value: Any) -> str:
    """JSON-serializes value with a fallback for non-serializable types.

    Args:
        value: The object to serialize.

    Returns:
        str: JSON-formatted string representing the value.
    """
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return json.dumps(str(value))

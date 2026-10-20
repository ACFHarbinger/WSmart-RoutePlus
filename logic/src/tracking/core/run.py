"""Run lifecycle management for the WSTracker.

Artifact logging lives in ``run_artifacts``.

Metric/param logging lives in ``run_logging``.

Provides the core recording interface for tracking individual experiments.
Manages buffered metric writes, parameter logging, tag management,
artifact registration, and dataset lifecycle event tracking.

Attributes:
    get_active_run: Function to retrieve the current process-level active run.
    set_active_run: Function to set the process-level active run.
    Run: Main class for interacting with an individual tracked run.

Example:
    >>> from logic.src.tracking.core.run import Run
    >>> from logic.src.tracking.core.store import TrackingStore
    >>> store = TrackingStore("tracking.db")
    >>> with Run("uuid-123", store, "./artifacts") as run:
    ...     run.log_param("lr", 0.001)
    ...     run.log_metric("loss", 0.5, step=1)
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any, Dict, List, Optional, Tuple

from typing_extensions import Literal

from logic.src.tracking.core.store import TrackingStore

from .run_artifacts import (
    RunArtifactMixin as RunArtifactMixin,
)
from .run_logging import (
    RunLoggingMixin as RunLoggingMixin,
)

# ---------------------------------------------------------------------------
# Module-level active-run registry (one per process)
# ---------------------------------------------------------------------------

_active_run: Optional["Run"] = None
_registry_lock = threading.Lock()


def get_active_run() -> Optional["Run"]:
    """Returns the active Run instance for the current process.

    Returns:
        Optional[Run]: The current process-global active run, or None if no
            run is active.
    """
    return _active_run


def set_active_run(run: Optional["Run"]) -> None:
    """Sets the active Run instance for the current process.

    This is usually called automatically by the Run context manager.

    Args:
        run: The Run instance to register as active, or None to clear the registry.
    """
    global _active_run
    with _registry_lock:
        _active_run = run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class Run(RunArtifactMixin, RunLoggingMixin):
    """A single tracked experiment run.

    Provides high-level methods to log metrics, parameters, tags, artifacts, and
    dataset lifecycle events. Metric writes are buffered in memory and flushed
    to SQLite in batches to maximize I/O performance.

    This class is thread-safe; multiple threads may log concurrently.

    Attributes:
        run_id: Unique UUID string identifying the run in the database.
        artifact_dir: Root directory where this run's artifacts are stored.
    """

    def __init__(
        self,
        run_id: str,
        store: TrackingStore,
        artifact_dir: str,
        buffer_size: int = 200,
    ) -> None:
        """Initializes a new tracking run.

        Args:
            run_id: UUID string identifying the run.
            store: Persistent storage backend instance.
            artifact_dir: Local directory for storing artifact files.
            buffer_size: Number of metrics to buffer before an automatic flush.
        """
        self.run_id = run_id
        self._store = store
        self.artifact_dir = artifact_dir
        self._buffer_size = buffer_size

        self._metric_buffer: List[Tuple[str, float, int]] = []
        self._buffer_lock = threading.Lock()

        # path -> last observed SHA-256 for change detection
        self._file_hashes: Dict[str, str] = {}
        self._closed = False

        # Optional secondary logging sinks (e.g. MLflowBridge).
        # Each sink must expose log_metric(key, value, step),
        # log_params(params), log_artifact(path), and finish(status).
        self._sinks: List[Any] = []

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Params
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Dataset events
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finish(self, status: str = "completed", error: Optional[str] = None) -> None:
        """Concludes the run, flushing buffers and updating state.

        Args:
            status: Final status (completed, failed, killed).
            error: Optional error message summarizing the failure.
        """
        if not self._closed:
            self.flush()
            self._store.finish_run(self.run_id, status=status, error=error)
            self._closed = True
            for sink in self._sinks:
                with contextlib.suppress(Exception):
                    sink.finish(status)
            if get_active_run() is self:
                set_active_run(None)

    # Context-manager support
    def __enter__(self) -> "Run":
        """Enters the tracking context, registering this as the active run.

        Returns:
            Run: The active run instance.
        """
        set_active_run(self)
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        _exc_tb: Any,
    ) -> Literal[False]:
        """Exits the context, automatically finishing the run.

        Handles exception logging if the context was exited via an error.

        Args:
            exc_type: Type of the exception if one occurred.
            exc_val: Exception instance.
            _exc_tb: Traceback object.

        Returns:
            Literal[False]: Always returns False to propagate exceptions.
        """
        if exc_type is not None:
            self.finish(status="failed", error=str(exc_val))
        else:
            self.finish(status="completed")
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Retrieves all configuration parameters logged for this run.

        Returns:
            Dict[str, Any]: Dictionary mapping parameter keys to values.
        """
        return self._store.get_params(self.run_id)

    def get_tags(self) -> Dict[str, str]:
        """Retrieves all metadata tags logged for this run.

        Returns:
            Dict[str, str]: Dictionary mapping tag keys to values.
        """
        return self._store.get_tags(self.run_id)

    def get_latest_metrics(self) -> Dict[str, float]:
        """Retrieves the most recent value for every logged metric.

        Returns:
            Dict[str, float]: Dictionary mapping metric keys to their latest scalars.
        """
        return self._store.get_latest_metrics(self.run_id)

    def get_metric_history(self, key: str) -> List[Dict[str, Any]]:
        """Retrieves the full evolution history for a specific metric.

        Args:
            key: Semantic name of the metric to query.

        Returns:
            List[Dict[str, Any]]: Step-indexed records containing 'value',
                'step', and 'timestamp'.
        """
        return self._store.get_metric_history(self.run_id, key)

    def get_artifacts(self, artifact_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves Registered artifacts for this run.

        Args:
            artifact_type: Optional filter by semantic category.

        Returns:
            List[Dict[str, Any]]: Artifact records mapping attributes like
                'name', 'path', and 'metadata'.
        """
        return self._store.get_artifacts(self.run_id, artifact_type=artifact_type)

    def get_dataset_events(self) -> List[Dict[str, Any]]:
        """Retrieves all data-lifecycle events in chronological order.

        Returns:
            List[Dict[str, Any]]: Event records containing type, path, and context.
        """
        return self._store.get_dataset_events(self.run_id)

    def __repr__(self) -> str:
        """Returns a string representation of the Run state.

        Returns:
            str: Developer-readable string summary.
        """
        return f"Run(id={self.run_id!r}, closed={self._closed})"

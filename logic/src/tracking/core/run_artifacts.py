"""Artifact and dataset-event logging for tracking Run.

Moved from ``logic.src.tracking.core.run``.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from logic.src.tracking.validation.hashing import hash_file

if TYPE_CHECKING:
    from .run import Run


class RunArtifactMixin:
    """Moved methods of :class:`Run`."""

    def log_artifact(
        self,
        path: str,
        name: Optional[str] = None,
        artifact_type: str = "file",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Run":
        """Registers a local file as an artifact of this run.

        Args:
            path: Local path to the artifact file.
            name: Human-readable name (defaults to file basename).
            artifact_type: Semantic category (e.g., 'model', 'plot').
            metadata: Additional JSON-serializable metadata.

        Returns:
            Run: Self for method chaining.
        """
        if self._closed or not os.path.exists(path):
            return self
        self._store.log_artifact(
            self.run_id,
            name or os.path.basename(path),
            path,
            artifact_type=artifact_type,
            file_hash=hash_file(path),
            size_bytes=os.path.getsize(path),
            metadata=metadata,
        )
        for sink in self._sinks:
            with contextlib.suppress(Exception):
                sink.log_artifact(path)
        return self

    def log_artifacts_dir(
        self,
        dir_path: str,
        artifact_type: str = "file",
    ) -> "Run":
        """Registers every file in a directory as an artifact.

        Note: This is non-recursive.

        Args:
            dir_path: Path to the directory.
            artifact_type: Semantic category applied to all files.

        Returns:
            Run: Self for method chaining.
        """
        if not os.path.isdir(dir_path):
            return self
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            if os.path.isfile(fpath):
                self.log_artifact(fpath, artifact_type=artifact_type)
        return self

    def log_dataset_event(
        self,
        event_type: str,
        file_path: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Run":
        """Records a data-lifecycle event (e.g., loading, mutation).

        Args:
            event_type: Semantic event (load, save, generate, hash_change).
            file_path: Path to the associated data file if applicable.
            shape: Logical shape of the dataset.
            metadata: Extra contextual information.

        Returns:
            Run: Self for method chaining.
        """
        if self._closed:
            return self

        file_hash: Optional[str] = None
        prev_hash: Optional[str] = None
        size_bytes: Optional[int] = None

        if file_path and os.path.exists(file_path):
            file_hash = hash_file(file_path)
            size_bytes = os.path.getsize(file_path)
            prev_hash = self._file_hashes.get(file_path)
            self._file_hashes[file_path] = file_hash  # type: ignore[assignment]

        self._store.log_dataset_event(
            self.run_id,
            event_type=event_type,
            file_path=file_path,
            file_hash=file_hash,
            prev_hash=prev_hash,
            size_bytes=size_bytes,
            shape=shape,
            metadata=metadata,
        )
        return self

    def watch_file(self, file_path: str) -> "Run":
        """Registers the current SHA-256 of a file for change detection.

        Args:
            file_path: Path to the file to monitor.

        Returns:
            Run: Self for method chaining.
        """
        if os.path.exists(file_path):
            h = hash_file(file_path)
            if h is not None:
                self._file_hashes[file_path] = h
        return self

    def check_file_changed(self, file_path: str) -> bool:
        """Determines if a watched file has changed on disk.

        Updating the baseline hash on every call ensures subsequent checks
        measure drift from the most recent confirmed state.

        Args:
            file_path: Path to the file to check.

        Returns:
            bool: True if the file hash has changed since the last observation.
        """
        if not os.path.exists(file_path):
            return False
        current = hash_file(file_path)
        prev = self._file_hashes.get(file_path)
        if current is not None:
            self._file_hashes[file_path] = current
        changed = prev is None or current != prev
        if changed and prev is not None:
            # Observed change — log it automatically
            self._store.log_dataset_event(
                self.run_id,
                event_type="hash_change",
                file_path=file_path,
                file_hash=current,
                prev_hash=prev,
                size_bytes=os.path.getsize(file_path),
            )
        return changed

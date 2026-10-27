"""Artifact and dataset-event logging for tracking Run.

Moved from ``logic.src.tracking.core.run``.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypeVar

from logic.src.tracking.validation.hashing import hash_file

if TYPE_CHECKING:
    from logic.src.tracking.core.store import TrackingStore

_ArtT = TypeVar("_ArtT", bound="RunArtifactMixin")


class RunArtifactMixin:
    """Moved methods of :class:`Run`."""

    run_id: str
    _store: "TrackingStore"
    _closed: bool
    _sinks: List[Any]
    _file_hashes: Dict[str, str]

    def log_artifact(
        self: _ArtT,
        path: str,
        name: Optional[str] = None,
        artifact_type: str = "file",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> _ArtT:
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
        self: _ArtT,
        dir_path: str,
        artifact_type: str = "file",
    ) -> _ArtT:
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
        self: _ArtT,
        event_type: str,
        file_path: Optional[str] = None,
        shape: Optional[Tuple[int, ...]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> _ArtT:
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

    def watch_file(self: _ArtT, file_path: str) -> _ArtT:
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

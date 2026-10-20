"""Filesystem-level change tracking for WSTracker.

This module provides the :class:`FilesystemTracker` which coordinates with a
tracking run to record file-level dataset events. It implements SHA-256
content hashing and OS-level stat collection (size, timestamps) to maintain
data lineage across training and evaluation runs.

Attributes:
    FilesystemTracker: Tracks filesystem-level data events and lineage.

Example:
    >>> from logic.src.tracking.integrations.filesystem import FilesystemTracker
    >>> tracker = FilesystemTracker(run)
    >>> tracker.on_load("data/vrp_50_test.pkl")
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from logic.src.tracking.core.run import Run


class FilesystemTracker:
    """Tracks filesystem-level data events and content lineage.

    Monitors file loading and saving events, capturing SHA-256 hashes and
    metadata to ensure the integrity and traceability of localized datasets.

    Attributes:
        _run: The active tracking run instance.
        _hash_cache: In-memory cache to avoid redundant hashing of large files.
    """

    def __init__(self, run: Optional[Run]) -> None:
        """Initializes the filesystem mutation tracker.

        Args:
            run: The active tracking run. If None, operations become no-ops.
        """
        self._run = run
        self._hash_cache: Dict[str, str] = {}

    def on_load(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Records a 'load' event for a specific file.

        Args:
            path: Local filesystem path to the dataset file.
            metadata: Extra contextual information (e.g., source_url, tags).

        Returns:
            str: The SHA-256 hash of the loaded file.
        """
        if self._run is None or not os.path.exists(path):
            return ""

        file_hash = self._hash_file(path)
        size_bytes, _ = self._get_stats(path)

        meta: Dict[str, Any] = {
            "event": "load",
            "size_bytes": size_bytes,
        }
        if metadata:
            meta.update(metadata)

        self._run.log_dataset_event(
            event_type="load",
            file_path=path,
            file_hash=file_hash,
            size_bytes=size_bytes,
            metadata=meta,
        )
        return file_hash

    def on_save(self, path: str, prev_hash: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Records a 'save' or 'mutate' event for a file.

        Args:
            path: Local filesystem path where the data was written.
            prev_hash: Previous known SHA-256 hash for lineage tracking.
            metadata: Extra context for the save event.

        Returns:
            str: The SHA-256 hash of the saved file.
        """
        if self._run is None or not os.path.exists(path):
            return ""

        # Bust the cache since the file was just written/mutated
        if path in self._hash_cache:
            del self._hash_cache[path]

        file_hash = self._hash_file(path)
        size_bytes, _ = self._get_stats(path)

        meta: Dict[str, Any] = {
            "event": "save",
            "size_bytes": size_bytes,
        }
        if metadata:
            meta.update(metadata)

        self._run.log_dataset_event(
            event_type="save",
            file_path=path,
            file_hash=file_hash,
            prev_hash=prev_hash,
            size_bytes=size_bytes,
            metadata=meta,
        )
        return file_hash

    def on_stat(self, path: str) -> Dict[str, Any]:
        """Provides high-level metadata statistics for a file without logging an event.

        Args:
            path: Local filesystem path.

        Returns:
            Dict[str, Any]: Dictionary containing 'hash' and 'size_bytes'.
        """
        if not os.path.exists(path):
            return {}

        return {
            "hash": self._hash_file(path),
            "size_bytes": os.path.getsize(path),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash_file(self, path: str) -> str:
        """Computes the SHA-256 hash of a file with internal caching.

        Args:
            path: Absolute or relative path to the file.

        Returns:
            str: Hexadecimal SHA-256 string.
        """
        if path in self._hash_cache:
            return self._hash_cache[path]

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)

        h = sha256.hexdigest()
        self._hash_cache[path] = h
        return h

    @staticmethod
    def _get_stats(path: str) -> Tuple[int, float]:
        """Retrieves size and modification time for a file.

        Args:
            path: Path to the file.

        Returns:
            Tuple[int, float]: (size_bytes, mtime).
        """
        stat = os.stat(path)
        return stat.st_size, stat.st_mtime

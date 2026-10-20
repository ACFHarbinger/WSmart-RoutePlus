"""SQLite storage backend for the WSTracker.

Record writers live in ``store_records``.

Connection helpers live in ``store_connection``.

Uses WAL (Write-Ahead Logging) journal mode so multiple processes can read/write
concurrently without corrupting the database. Every public method opens its own
short-lived connection and closes it before returning, providing a simple and
safe pattern for multi-process SQLite access.

Attributes:
    TrackingStore: Persistent SQLite store for experiment tracking data.

Example:
    >>> from logic.src.tracking.core.store import TrackingStore
    >>> store = TrackingStore("tracking.db")
    >>> store.get_or_create_experiment("VRPP-Optimization")
"""

from __future__ import annotations

from .store_connection import (
    TrackingStoreConnectionMixin as TrackingStoreConnectionMixin,
)
from .store_records import (
    TrackingStoreRecordMixin as TrackingStoreRecordMixin,
)
from .store_records import (
    _safe_json_dumps as _safe_json_dumps,
)

# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TrackingStore(TrackingStoreRecordMixin, TrackingStoreConnectionMixin):
    """Persistent SQLite store for experiment tracking data.

    Thread-safe and multi-process safe via SQLite WAL mode. Each operation
    acquires and releases its own connection to avoid long-lived locks and
    ensure robustness in distributed or multi-threaded environments.

    Attributes:
        db_path: Absolute path to the SQLite database file.
        queries: Dictionary holding loaded SQL query templates.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Tags & Params
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Artifacts & Dataset Events
    # ------------------------------------------------------------------

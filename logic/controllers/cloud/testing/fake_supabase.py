"""An in-memory Supabase client covering the API the Supabase control plane uses.

Tables follow ``infra/cloud/supabase/migrations/0001_cloud_runner.sql`` and the
``rpc`` functions re-implement its four functions with the same semantics. Like
PostgREST it rejects unknown tables and columns, missing NOT NULL values and
duplicate primary keys, and stores values as JSON, so non-JSON values fail.
``update`` returns the updated rows, like ``returning=representation``.

Attributes:
    FakeSupabaseClient: The client.
    PostgrestError: What PostgREST would report.
"""

from __future__ import annotations

import copy
import itertools
import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# table -> (columns, not-null columns without defaults, primary key, serial column)
SCHEMA: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Optional[str]]] = {
    "wsr_jobs": (("job_id", "created_at", "rev", "doc"), ("job_id", "created_at", "doc"), ("job_id",), None),
    "wsr_tasks": (
        ("task_id", "job_id", "seq", "state", "lease_expires", "rev", "doc"),
        ("task_id", "job_id", "seq", "state", "doc"),
        ("task_id",),
        None,
    ),
    "wsr_events": (("id", "job_id", "at", "doc"), ("job_id", "at", "doc"), ("id",), "id"),
    "wsr_metrics": (("id", "task_id", "at", "doc"), ("task_id", "at", "doc"), ("id",), "id"),
    "wsr_results": (("task_id", "relpath", "doc"), ("task_id", "relpath", "doc"), ("task_id", "relpath"), None),
}
_DEFAULTS = {"rev": 0}


class PostgrestError(Exception):
    """A request PostgREST would reject."""


@dataclass
class _Response:
    data: Any


def _json(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except TypeError as exc:
        raise PostgrestError(f"value is not JSON: {exc}") from exc


class _Query:
    def __init__(self, client: "FakeSupabaseClient", table: str) -> None:
        if table not in SCHEMA:
            raise PostgrestError(f'relation "{table}" does not exist')
        self._client, self._table = client, table
        self._op: Optional[str] = None
        self._payload: Any = None
        self._columns: Optional[List[str]] = None
        self._filters: List[Tuple[str, Callable[[Any, Any], bool], Any]] = []
        self._order: List[Tuple[str, bool]] = []
        self._limit: Optional[int] = None

    def _check(self, column: str) -> None:
        if column not in SCHEMA[self._table][0]:
            raise PostgrestError(f"column {self._table}.{column} does not exist")

    def select(self, columns: str = "*") -> "_Query":
        self._op = "select"
        self._columns = None if columns.strip() == "*" else [c.strip() for c in columns.split(",")]
        for column in self._columns or []:
            self._check(column)
        return self

    def insert(self, rows: Any) -> "_Query":
        self._op, self._payload = "insert", rows if isinstance(rows, list) else [rows]
        return self

    def update(self, values: Dict[str, Any]) -> "_Query":
        for column in values:
            self._check(column)
        self._op, self._payload = "update", values
        return self

    def delete(self) -> "_Query":
        self._op = "delete"
        return self

    def eq(self, column: str, value: Any) -> "_Query":
        self._check(column)
        self._filters.append((column, lambda a, b: a == b, value))
        return self

    def gt(self, column: str, value: Any) -> "_Query":
        self._check(column)
        self._filters.append((column, lambda a, b: a is not None and a > b, value))
        return self

    def order(self, column: str, desc: bool = False) -> "_Query":
        self._check(column)
        self._order.append((column, desc))
        return self

    def limit(self, count: int) -> "_Query":
        self._limit = count
        return self

    def _matching(self) -> List[Dict[str, Any]]:
        rows = [row for row in self._client._tables[self._table] if all(f(row.get(c), v) for c, f, v in self._filters)]
        for column, desc in reversed(self._order):
            rows.sort(key=lambda row: (row.get(column) is None, row.get(column)), reverse=desc)
        return rows[: self._limit] if self._limit is not None else rows

    def execute(self) -> _Response:
        with self._client._lock:
            if self._op == "select":
                rows = self._matching()
                keep = self._columns
                return _Response(
                    [copy.deepcopy({k: v for k, v in row.items() if keep is None or k in keep}) for row in rows]
                )
            if self._op == "insert":
                return _Response([self._client._insert(self._table, row) for row in self._payload])
            if self._op == "update":
                values = _json(self._payload)
                rows = self._matching()
                for row in rows:
                    row.update(copy.deepcopy(values))
                return _Response(copy.deepcopy(rows))
            if self._op == "delete":
                rows = self._matching()
                for row in rows:
                    self._client._tables[self._table].remove(row)
                return _Response(copy.deepcopy(rows))
        raise PostgrestError("no operation")


class _Rpc:
    def __init__(self, function: Callable[[], Any]) -> None:
        self._function = function

    def execute(self) -> _Response:
        return _Response(self._function())


class FakeSupabaseClient:
    """In-memory stand-in for ``supabase.Client``."""

    def __init__(self) -> None:
        self._tables: Dict[str, List[Dict[str, Any]]] = {name: [] for name in SCHEMA}
        self._serial = itertools.count(1)
        self._lock = threading.RLock()

    def table(self, name: str) -> _Query:
        return _Query(self, name)

    def _insert(self, table: str, row: Dict[str, Any]) -> Dict[str, Any]:
        columns, required, key, serial = SCHEMA[table]
        row = _json(row)
        for column in row:
            if column not in columns:
                raise PostgrestError(f"column {table}.{column} does not exist")
        missing = [c for c in required if row.get(c) is None]
        if missing:
            raise PostgrestError(
                f"null value in column {missing[0]!r} of relation {table!r} violates not-null constraint"
            )
        for column, default in _DEFAULTS.items():
            if column in columns:
                row.setdefault(column, default)
        if serial:
            row[serial] = next(self._serial)
        identity = tuple(row[c] for c in key)
        if any(tuple(existing[c] for c in key) == identity for existing in self._tables[table]):
            raise PostgrestError(f"duplicate key value violates unique constraint {table}_pkey")
        self._tables[table].append(row)
        return copy.deepcopy(row)

    # The migration's functions ---------------------------------------------

    def rpc(self, name: str, params: Dict[str, Any]) -> _Rpc:
        functions: Dict[str, Callable[..., Any]] = {
            "wsr_insert_job": self._insert_job,
            "wsr_lease_task": self._lease_task,
            "wsr_expire_leases": self._expire_leases,
            "wsr_replace_results": self._replace_results,
        }
        if name not in functions:
            raise PostgrestError(f"function {name} does not exist")
        params = _json(params)
        return _Rpc(lambda: functions[name](**params))

    def _insert_job(self, p_job: Dict[str, Any], p_tasks: List[Dict[str, Any]]) -> None:
        with self._lock:
            snapshot = copy.deepcopy(self._tables)
            try:
                self._insert("wsr_jobs", {"job_id": p_job["job_id"], "created_at": p_job["created_at"], "doc": p_job})
                for task in p_tasks:
                    self._insert(
                        "wsr_tasks",
                        {
                            "task_id": task["task_id"],
                            "job_id": task["job_id"],
                            "seq": task["seq"],
                            "state": task["state"],
                            "lease_expires": task.get("lease_expires"),
                            "rev": task.get("rev", 0),
                            "doc": task,
                        },
                    )
            except Exception:
                self._tables = snapshot  # the function runs in one transaction
                raise

    def _lease_task(
        self, p_owner: str, p_lease_expires: float, p_job_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            queued = [
                row
                for row in self._tables["wsr_tasks"]
                if row["state"] == "queued" and (p_job_id is None or row["job_id"] == p_job_id)
            ]
            if not queued:
                return None
            row = min(queued, key=lambda r: r["seq"])
            rev = row["rev"] + 1
            row.update(state="placing", lease_expires=p_lease_expires, rev=rev)
            row["doc"] = dict(row["doc"], state="placing", lease_owner=p_owner, lease_expires=p_lease_expires, rev=rev)
            return copy.deepcopy(row["doc"])

    def _expire_leases(self, p_now: float) -> List[str]:
        with self._lock:
            expired = []
            for row in self._tables["wsr_tasks"]:
                if row["state"] == "placing" and row["lease_expires"] is not None and row["lease_expires"] < p_now:
                    rev = row["rev"] + 1
                    row.update(state="queued", lease_expires=None, rev=rev)
                    row["doc"] = dict(row["doc"], state="queued", lease_owner=None, lease_expires=None, rev=rev)
                    expired.append(row["task_id"])
            return sorted(expired)

    def _replace_results(self, p_task_id: str, p_docs: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._tables["wsr_results"] = [r for r in self._tables["wsr_results"] if r["task_id"] != p_task_id]
            for doc in p_docs:
                self._insert("wsr_results", {"task_id": p_task_id, "relpath": doc["relpath"], "doc": doc})

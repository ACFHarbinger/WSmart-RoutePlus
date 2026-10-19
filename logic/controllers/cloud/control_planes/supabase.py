"""Supabase (Postgres) control plane.

Tables ``wsr_jobs``, ``wsr_tasks``, ``wsr_events``, ``wsr_metrics`` and
``wsr_results`` hold the JSON documents in a ``doc`` column next to the columns
the queries filter on. The migration in
``infra/cloud/supabase/migrations/0001_cloud_runner.sql`` creates them, their
row-level security, and four functions that make the multi-row operations atomic:

- ``wsr_insert_job(p_job, p_tasks)``: a job and its tasks in one transaction;
- ``wsr_lease_task(p_owner, p_lease_expires, p_job_id)``: ``FOR UPDATE SKIP LOCKED`` on the oldest queued task;
- ``wsr_expire_leases(p_now)``: requeue every expired ``placing`` task;
- ``wsr_replace_results(p_task_id, p_docs)``: swap a task's results index.

Other read-modify-writes are optimistic: the update is conditional on the row's
``rev`` and retried on conflict. Realtime subscriptions (the Studio) listen to
``wsr_tasks`` and ``wsr_events``.

The client (``supabase.create_client(url, key)``) is injected; tests pass
:class:`~logic.controllers.cloud.testing.fake_supabase.FakeSupabaseClient`.

URI: ``supabase://<project-ref>`` (the URL ``https://<project-ref>.supabase.co``;
the key from ``SUPABASE_KEY`` or the ``supabase/<profile>`` keychain entry, CR-08).

Attributes:
    SupabaseAdapter: :class:`~logic.controllers.cloud.control_planes.documents.DocumentAdapter` on Supabase.
    SupabaseControlPlane: The control plane.
    control_plane_from_uri: Factory for ``supabase://`` URIs.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, List, Optional, Sequence
from urllib.parse import ParseResult

from logic.controllers.cloud.control_planes.documents import (
    Doc,
    DocumentAdapter,
    DocumentControlPlane,
    Mutation,
)

JOBS, TASKS, EVENTS, METRICS, RESULTS = "wsr_jobs", "wsr_tasks", "wsr_events", "wsr_metrics", "wsr_results"
_APPEND = {"events": (EVENTS, "job_id"), "metrics": (METRICS, "task_id")}
_MAX_RETRIES = 25


class ConflictError(RuntimeError):
    """An optimistic update kept losing to concurrent writers."""


class SupabaseAdapter(DocumentAdapter):
    """Document primitives on a Supabase project."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def _one(self, table: str, column: str, value: str) -> Optional[Doc]:
        rows = self.client.table(table).select("doc").eq(column, value).limit(1).execute().data
        return rows[0]["doc"] if rows else None

    def _rmw(self, table: str, column: str, row_id: str, mutate: Mutation, columns: Callable[[Doc], Doc]) -> Doc:
        for _ in range(_MAX_RETRIES):
            rows = self.client.table(table).select("doc,rev").eq(column, row_id).limit(1).execute().data
            if not rows:
                raise KeyError(row_id)
            rev = rows[0]["rev"]
            updated = mutate(rows[0]["doc"])
            updated["rev"] = rev + 1
            written = (
                self.client.table(table)
                .update(dict(columns(updated), doc=updated, rev=rev + 1))
                .eq(column, row_id)
                .eq("rev", rev)
                .execute()
                .data
            )
            if written:
                return written[0]["doc"]
        raise ConflictError(f"{table}/{row_id}: {_MAX_RETRIES} conflicting updates in a row")

    # Jobs and tasks --------------------------------------------------------

    def insert_job(self, job: Doc, tasks: Sequence[Doc]) -> None:
        self.client.rpc("wsr_insert_job", {"p_job": job, "p_tasks": list(tasks)}).execute()

    def get_job(self, job_id: str) -> Optional[Doc]:
        return self._one(JOBS, "job_id", job_id)

    def list_jobs(self, limit: int) -> List[Doc]:
        rows = self.client.table(JOBS).select("doc").order("created_at", desc=True).limit(limit).execute().data
        return [row["doc"] for row in rows]

    def update_job(self, job_id: str, mutate: Mutation) -> Doc:
        return self._rmw(JOBS, "job_id", job_id, mutate, lambda doc: {})

    def get_task(self, task_id: str) -> Optional[Doc]:
        return self._one(TASKS, "task_id", task_id)

    def list_tasks(self, job_id: str) -> List[Doc]:
        rows = self.client.table(TASKS).select("doc").eq("job_id", job_id).order("seq").execute().data
        return [row["doc"] for row in rows]

    def update_task(self, task_id: str, mutate: Mutation) -> Doc:
        return self._rmw(
            TASKS,
            "task_id",
            task_id,
            mutate,
            lambda doc: {"state": doc["state"], "lease_expires": doc.get("lease_expires")},
        )

    def lease_task(self, owner: str, lease_expires: float, job_id: Optional[str]) -> Optional[Doc]:
        params = {"p_owner": owner, "p_lease_expires": lease_expires, "p_job_id": job_id}
        return self.client.rpc("wsr_lease_task", params).execute().data or None

    def expire_leases(self, now: float) -> List[str]:
        return list(self.client.rpc("wsr_expire_leases", {"p_now": now}).execute().data or [])

    # Events, metrics, results ----------------------------------------------

    def append(self, collection: str, doc: Doc) -> None:
        table, key = _APPEND[collection]
        self.client.table(table).insert({key: doc[key], "at": doc["at"], "doc": doc}).execute()

    def list_appended(self, collection: str, key: str, since: Optional[float]) -> List[Doc]:
        table, column = _APPEND[collection]
        query = self.client.table(table).select("doc").eq(column, key)
        if since is not None:
            query = query.gt("at", since)
        return [row["doc"] for row in query.order("id").execute().data]

    def replace_results(self, task_id: str, docs: Sequence[Doc]) -> None:
        self.client.rpc("wsr_replace_results", {"p_task_id": task_id, "p_docs": list(docs)}).execute()

    def list_results(self, task_id: str) -> List[Doc]:
        rows = self.client.table(RESULTS).select("doc").eq("task_id", task_id).order("relpath").execute().data
        return [row["doc"] for row in rows]


class SupabaseControlPlane(DocumentControlPlane):
    """The control plane on Supabase."""

    def __init__(self, client: Any, clock: Callable[[], float] = time.time) -> None:
        super().__init__(SupabaseAdapter(client), clock=clock)


def control_plane_from_uri(uri: ParseResult) -> SupabaseControlPlane:
    """``supabase://<project-ref>``; the API key comes from ``SUPABASE_KEY``."""
    from supabase import create_client

    key = os.environ.get("SUPABASE_KEY")
    if not key:
        raise ValueError("set SUPABASE_KEY (or use a keychain profile, CR-08) for the Supabase control plane")
    return SupabaseControlPlane(create_client(f"https://{uri.netloc}.supabase.co", key))

"""A control plane over a JSON document store, shared by the Firebase and Supabase implementations.

:class:`DocumentControlPlane` owns the record conversion, the state machine and the
conditional-update semantics of :class:`~logic.controllers.cloud.interfaces.ControlPlane`.
A :class:`DocumentAdapter` supplies the storage primitives, each of which must be
atomic on its service: a Firestore transaction, or a Postgres row lock or function.

Task documents are the task record (:func:`~logic.controllers.cloud.model.to_dict`)
plus two bookkeeping fields: ``seq`` (global FIFO order) and ``rev`` (incremented
on every write, for optimistic concurrency where the service needs it).

Attributes:
    DocumentAdapter: The storage primitives.
    DocumentControlPlane: :class:`ControlPlane` on an adapter.
"""

from __future__ import annotations

import abc
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

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

Doc = Dict[str, Any]
Mutation = Callable[[Doc], Doc]
_BOOKKEEPING = ("seq", "rev")


class DocumentAdapter(abc.ABC):
    """Storage primitives of a document store. Every method is atomic on its service."""

    @abc.abstractmethod
    def insert_job(self, job: Doc, tasks: Sequence[Doc]) -> None:
        """Store a job and its task documents."""

    @abc.abstractmethod
    def get_job(self, job_id: str) -> Optional[Doc]:
        """The job document, or None."""

    @abc.abstractmethod
    def list_jobs(self, limit: int) -> List[Doc]:
        """Job documents, newest ``created_at`` first."""

    @abc.abstractmethod
    def update_job(self, job_id: str, mutate: Mutation) -> Doc:
        """Read-modify-write a job document; ``KeyError`` if absent."""

    @abc.abstractmethod
    def get_task(self, task_id: str) -> Optional[Doc]:
        """The task document, or None."""

    @abc.abstractmethod
    def list_tasks(self, job_id: str) -> List[Doc]:
        """A job's task documents in ``seq`` order."""

    @abc.abstractmethod
    def update_task(self, task_id: str, mutate: Mutation) -> Doc:
        """Read-modify-write a task document; ``KeyError`` if absent. ``mutate`` may raise to abort."""

    @abc.abstractmethod
    def lease_task(self, owner: str, lease_expires: float, job_id: Optional[str]) -> Optional[Doc]:
        """Move the lowest-``seq`` queued task (of ``job_id`` if given) to ``placing`` under a lease.

        Sets ``state = "placing"``, ``lease_owner`` and ``lease_expires`` and increments ``rev``.
        """

    @abc.abstractmethod
    def expire_leases(self, now: float) -> List[str]:
        """Return ``placing`` tasks with ``lease_expires < now`` to ``queued`` (clearing the lease)."""

    @abc.abstractmethod
    def append(self, collection: str, doc: Doc) -> None:
        """Append to ``events`` (keyed by ``job_id``) or ``metrics`` (keyed by ``task_id``)."""

    @abc.abstractmethod
    def list_appended(self, collection: str, key: str, since: Optional[float]) -> List[Doc]:
        """Appended documents of one job or task, in append order, with ``at > since`` if given."""

    @abc.abstractmethod
    def replace_results(self, task_id: str, docs: Sequence[Doc]) -> None:
        """Replace a task's results index."""

    @abc.abstractmethod
    def list_results(self, task_id: str) -> List[Doc]:
        """A task's results index, sorted by ``relpath``."""


def _task(doc: Doc) -> TaskRecord:
    return task_from_dict({key: value for key, value in doc.items() if key not in _BOOKKEEPING})


def _job(doc: Doc) -> JobRecord:
    return job_from_dict({key: value for key, value in doc.items() if key not in _BOOKKEEPING})


class DocumentControlPlane(ControlPlane):
    """:class:`ControlPlane` on a :class:`DocumentAdapter`.

    Args:
        adapter: The storage primitives.
        clock: Time source for leases.
    """

    def __init__(self, adapter: DocumentAdapter, clock: Callable[[], float] = time.time) -> None:
        self.adapter = adapter
        self.clock = clock

    def create_job(self, job: JobRecord, tasks: Sequence[TaskRecord]) -> None:
        # Milliseconds x 1e5 leaves room for 100,000 tasks per job and stays far below int64.
        if len(tasks) > 100_000:
            raise ValueError("a job has at most 100,000 tasks")
        base = int(job.created_at * 1000) * 100_000
        docs = [dict(to_dict(task), seq=base + index, rev=0) for index, task in enumerate(tasks)]
        self.adapter.insert_job(to_dict(job), docs)

    def get_job(self, job_id: str) -> JobRecord:
        doc = self.adapter.get_job(job_id)
        if doc is None:
            raise KeyError(job_id)
        return _job(doc)

    def list_jobs(self, limit: int = 50) -> List[JobRecord]:
        return [_job(doc) for doc in self.adapter.list_jobs(limit)]

    def request_cancel(self, job_id: str) -> None:
        self.adapter.update_job(job_id, lambda doc: dict(doc, cancel_requested=True))

    def get_task(self, task_id: str) -> TaskRecord:
        doc = self.adapter.get_task(task_id)
        if doc is None:
            raise KeyError(task_id)
        return _task(doc)

    def list_tasks(self, job_id: str) -> List[TaskRecord]:
        return [_task(doc) for doc in self.adapter.list_tasks(job_id)]

    def lease_task(self, owner: str, lease_seconds: float, job_id: Optional[str] = None) -> Optional[TaskRecord]:
        doc = self.adapter.lease_task(owner, self.clock() + lease_seconds, job_id)
        return None if doc is None else _task(doc)

    def renew_lease(self, task_id: str, owner: str, lease_seconds: float) -> bool:
        def mutate(doc: Doc) -> Doc:
            if doc.get("lease_owner") != owner or doc.get("state") != TaskState.PLACING.value:
                raise _LeaseLost
            return dict(doc, lease_expires=self.clock() + lease_seconds, rev=doc.get("rev", 0) + 1)

        try:
            self.adapter.update_task(task_id, mutate)
        except _LeaseLost:
            return False
        return True

    def expire_leases(self, now: Optional[float] = None) -> List[str]:
        return self.adapter.expire_leases(self.clock() if now is None else now)

    def update_task(
        self,
        task_id: str,
        state: Optional[TaskState] = None,
        expected: Optional[TaskState] = None,
        **fields: Any,
    ) -> TaskRecord:
        for name in fields:
            if name in ("task_id", "job_id", "state") or name not in TaskRecord.__dataclass_fields__:
                raise AttributeError(f"cannot update task field {name!r}")

        def mutate(doc: Doc) -> Doc:
            task = _task(doc)
            if expected is not None and task.state != TaskState(expected):
                raise StaleStateError(f"{task_id}: expected {TaskState(expected).value}, found {task.state.value}")
            for name, value in fields.items():
                setattr(task, name, value)
            if state is not None and TaskState(state) != task.state:
                check_transition(task.state, TaskState(state))
                task.state = TaskState(state)
                if task.state != TaskState.PLACING:
                    task.lease_expires = None
            return dict(to_dict(task), seq=doc["seq"], rev=doc.get("rev", 0) + 1)

        return _task(self.adapter.update_task(task_id, mutate))

    def append_event(self, event: Event) -> None:
        self.adapter.append("events", to_dict(event))

    def list_events(self, job_id: str, since: Optional[float] = None) -> List[Event]:
        return [Event(**doc) for doc in self.adapter.list_appended("events", job_id, since)]

    def put_metrics(self, task_id: str, records: Sequence[Dict[str, Any]]) -> None:
        for record in records:
            self.adapter.append(
                "metrics", {"task_id": task_id, "at": float(record.get("at", self.clock())), "record": record}
            )

    def get_metrics(
        self,
        task_id: str,
        since: Optional[float] = None,
        limit: Optional[int] = None,
        reverse: bool = False,
    ) -> List[Dict[str, Any]]:
        docs = [doc["record"] for doc in self.adapter.list_appended("metrics", task_id, since)]
        if reverse:
            docs = list(reversed(docs))
        if limit is not None:
            docs = docs[:limit]
        return docs

    def record_results(self, task_id: str, files: Sequence[ResultFile]) -> None:
        self.adapter.replace_results(task_id, [dict(to_dict(record), task_id=task_id) for record in files])

    def list_results(self, task_id: str) -> List[ResultFile]:
        return [
            ResultFile(**{k: v for k, v in doc.items() if k != "task_id"}) for doc in self.adapter.list_results(task_id)
        ]


class _LeaseLost(Exception):
    """Internal: a renewal found the lease held by someone else."""

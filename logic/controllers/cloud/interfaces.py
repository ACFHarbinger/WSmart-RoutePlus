"""The contracts between the cloud runner and its providers.

Every provider package implements some of these, and passes the matching suite
in :mod:`logic.controllers.cloud.testing.contracts`:

- :class:`ControlPlane`: queue, task leases, job and task state, events, metrics
  and the results index (local SQLite, Firebase, Supabase).
- :class:`ComputeBackend`: starts, polls and stops one task attempt (local
  subprocess; batch, serverless and VM services on GCP, AWS and Azure).
- :class:`ObjectStore`: inputs, checkpoints, logs and results (filesystem, GCS, S3, Blob).
- :class:`SecretStore`: secret values by name (environment, OS keychain, provider secret managers).
- :class:`ImageRegistry`: the release image (Artifact Registry, ECR, ACR).

Provider SDKs are imported inside the implementations, never at module level, so
the core install does not need them.

Attributes:
    BackendStatus: What a backend reports for one attempt.
    LaunchSpec: Everything a backend needs to start one attempt.
    BackendCapabilities: What a backend can run.
    ControlPlane: Queue and state store.
    ComputeBackend: Runs task attempts.
    ObjectStore: Blob storage.
    StaleStateError: A conditional update lost a race.
    CapacityError: A backend cannot start an attempt now.
    SecretStore: Secret values by name.
    ImageRegistry: Container images.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from logic.controllers.cloud.model import (
    ComputeKind,
    Event,
    JobRecord,
    Provider,
    Resources,
    ResultFile,
    TaskRecord,
    TaskState,
)


class AttemptState(str, Enum):
    """A backend's view of one attempt."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PREEMPTED = "preempted"
    UNKNOWN = "unknown"


@dataclass
class BackendStatus:
    """What a backend reports for one attempt.

    Attributes:
        state: Attempt state.
        exit_code: Process exit code, when known.
        message: Backend message (failure reason, preemption notice, ...).
    """

    state: AttemptState
    exit_code: Optional[int] = None
    message: Optional[str] = None


@dataclass
class LaunchSpec:
    """Everything a backend needs to start one attempt.

    Attributes:
        task: The task (its cell, attempt number and ids).
        resources: Resources and limits.
        argv: Full command to run (``python main.py test_sim ...`` locally, the runtime entrypoint in a container).
        env: Environment variables. Secret values are never put here; ``secret_env`` names them instead.
        secret_env: Environment variable name -> secret-store name, resolved by the backend at start.
        workdir: Directory for the attempt's state and logs on the local backend; ignored by container backends.
        image: Container image, for container backends.
    """

    task: TaskRecord
    resources: Resources
    argv: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    secret_env: Dict[str, str] = field(default_factory=dict)
    workdir: Optional[str] = None
    image: Optional[str] = None


@dataclass
class BackendCapabilities:
    """What a backend can run.

    Attributes:
        provider: The provider.
        kinds: Compute kinds offered.
        max_wall_seconds: Longest attempt the service allows (None: unbounded).
        supports_spot: Spot or preemptible capacity is available.
        regions: Regions the backend may use.
        machine_types: Machine type -> ``{"cpus": ..., "memory_gb": ...}``.
    """

    provider: Provider
    kinds: Sequence[ComputeKind]
    max_wall_seconds: Optional[float] = None
    supports_spot: bool = False
    regions: Sequence[str] = ()
    machine_types: Dict[str, Dict[str, float]] = field(default_factory=dict)


class ControlPlane(abc.ABC):
    """Queue and state store shared by the CLI, the scheduler, workers and the Studio."""

    @abc.abstractmethod
    def create_job(self, job: JobRecord, tasks: Sequence[TaskRecord]) -> None:
        """Store a new job and its queued tasks."""

    @abc.abstractmethod
    def get_job(self, job_id: str) -> JobRecord:
        """The job; ``KeyError`` if unknown."""

    @abc.abstractmethod
    def list_jobs(self, limit: int = 50) -> List[JobRecord]:
        """Most recent jobs first."""

    @abc.abstractmethod
    def get_task(self, task_id: str) -> TaskRecord:
        """The task; ``KeyError`` if unknown."""

    @abc.abstractmethod
    def list_tasks(self, job_id: str) -> List[TaskRecord]:
        """A job's tasks in submission order."""

    @abc.abstractmethod
    def lease_task(self, owner: str, lease_seconds: float, job_id: Optional[str] = None) -> Optional[TaskRecord]:
        """Atomically move the oldest queued task (of ``job_id`` if given) to ``placing`` under a lease.

        Two concurrent callers never receive the same task. Returns None when nothing is queued.
        """

    @abc.abstractmethod
    def renew_lease(self, task_id: str, owner: str, lease_seconds: float) -> bool:
        """Extend a lease held by ``owner``; False if the lease was lost."""

    @abc.abstractmethod
    def expire_leases(self, now: Optional[float] = None) -> List[str]:
        """Return tasks whose ``placing`` lease ran out to the queue; the ids returned."""

    @abc.abstractmethod
    def update_task(
        self,
        task_id: str,
        state: Optional[TaskState] = None,
        expected: Optional[TaskState] = None,
        **fields: Any,
    ) -> TaskRecord:
        """Change a task's state and other fields.

        A state change must be legal (:func:`~logic.controllers.cloud.model.check_transition`).
        With ``expected``, the update only applies if the current state equals it;
        otherwise ``StaleStateError``. Leaving a leased state clears the lease.
        """

    @abc.abstractmethod
    def request_cancel(self, job_id: str) -> None:
        """Flag a job for cancellation."""

    @abc.abstractmethod
    def append_event(self, event: Event) -> None:
        """Append a lifecycle event."""

    @abc.abstractmethod
    def list_events(self, job_id: str, since: Optional[float] = None) -> List[Event]:
        """A job's events, oldest first, after ``since`` if given."""

    @abc.abstractmethod
    def put_metrics(self, task_id: str, records: Sequence[Dict[str, Any]]) -> None:
        """Append metric records (each with an ``at`` timestamp)."""

    @abc.abstractmethod
    def get_metrics(self, task_id: str, since: Optional[float] = None) -> List[Dict[str, Any]]:
        """A task's metric records, oldest first."""

    @abc.abstractmethod
    def record_results(self, task_id: str, files: Sequence[ResultFile]) -> None:
        """Replace a task's results index."""

    @abc.abstractmethod
    def list_results(self, task_id: str) -> List[ResultFile]:
        """A task's results index."""


class StaleStateError(RuntimeError):
    """A conditional task update found a different current state."""


class CapacityError(RuntimeError):
    """A backend cannot start an attempt now (no capacity, quota or memory); the task goes back to the queue."""


class ComputeBackend(abc.ABC):
    """Runs task attempts on one provider and compute kind."""

    provider: Provider
    kind: ComputeKind

    @abc.abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """What this backend can run."""

    @abc.abstractmethod
    def submit(self, launch: LaunchSpec) -> str:
        """Start one attempt; returns the backend's handle for it."""

    @abc.abstractmethod
    def poll(self, handle: str) -> BackendStatus:
        """Current status of an attempt."""

    @abc.abstractmethod
    def cancel(self, handle: str) -> None:
        """Stop an attempt; a no-op if it already finished."""

    @abc.abstractmethod
    def logs(self, handle: str, tail: Optional[int] = None) -> str:
        """The attempt's log, or its last ``tail`` lines."""

    def price_per_hour(self, machine_type: Optional[str], region: Optional[str], spot: bool) -> Optional[float]:
        """Hourly price hint in USD, or None when unknown."""
        return None


class ObjectStore(abc.ABC):
    """Blob storage addressed by ``/``-separated keys."""

    @abc.abstractmethod
    def put_file(self, local: Path, key: str) -> ResultFile:
        """Upload a file; the returned record carries its hash and size (``relpath`` is the key)."""

    @abc.abstractmethod
    def get_file(self, key: str, local: Path) -> None:
        """Download a key to a local path, creating parent directories."""

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """Whether the key exists."""

    @abc.abstractmethod
    def list(self, prefix: str) -> List[str]:
        """Keys under a prefix, sorted."""

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Delete a key; a no-op if absent."""

    def put_content_addressed(self, local: Path, namespace: str = "cas") -> ResultFile:
        """Upload a file under ``<namespace>/<sha256>`` once; later uploads of the same content are skipped."""
        from logic.controllers.cloud.store.storage import sha256_file

        digest = sha256_file(local)
        key = f"{namespace}/{digest}"
        if not self.exists(key):
            return self.put_file(local, key)
        return ResultFile(relpath=key, key=key, sha256=digest, size=Path(local).stat().st_size)


class SecretStore(abc.ABC):
    """Secret values by name. Values are never logged or stored in job records."""

    @abc.abstractmethod
    def get(self, name: str) -> Optional[str]:
        """The value, or None if absent."""

    @abc.abstractmethod
    def put(self, name: str, value: str) -> None:
        """Store or replace a value."""

    @abc.abstractmethod
    def delete(self, name: str) -> None:
        """Delete a value; a no-op if absent."""


class ImageRegistry(abc.ABC):
    """Container images of the release package."""

    @abc.abstractmethod
    def push(self, local_ref: str, tag: str) -> str:
        """Push a local image under ``tag``; returns its digest."""

    @abc.abstractmethod
    def resolve(self, tag: str) -> Optional[str]:
        """The digest behind a tag, or None."""

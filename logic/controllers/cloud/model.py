"""Job and task records of the cloud runner, and the task state machine.

A job is one ``test_sim`` command or a manifest of cells; it splits into one
task per cell. Task state moves only along :data:`TRANSITIONS`; a job's state
is derived from its tasks (:func:`job_state`).

Attributes:
    Provider: Where a task runs.
    ComputeKind: The kind of compute service on that provider.
    GurobiMode: How a task gets its Gurobi license.
    TaskState: The task state machine's states.
    TRANSITIONS: Legal task state changes.
    Resources: What one task asks for.
    CellSpec: One cell of a job (one ``test_sim`` invocation).
    JobSpec: What the user submits.
    TaskRecord: A task as the control plane stores it.
    JobRecord: A job as the control plane stores it.
    Event: A lifecycle event.
    ResultFile: One uploaded result file.

Example:
    >>> check_transition(TaskState.QUEUED, TaskState.PLACING)
    >>> job_state([TaskState.SUCCEEDED, TaskState.RUNNING], cancel_requested=False)
    <JobState.RUNNING: 'running'>
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class Provider(str, Enum):
    """Where a task runs."""

    LOCAL = "local"
    GCP = "gcp"
    AWS = "aws"
    AZURE = "azure"


class ComputeKind(str, Enum):
    """The kind of compute service on a provider."""

    PROCESS = "process"  # local subprocess
    BATCH = "batch"  # GCP Batch, AWS Batch, Azure Batch
    SERVERLESS = "serverless"  # Cloud Run Jobs, ECS Fargate, Container Apps Jobs
    VM = "vm"  # Compute Engine, EC2, Azure VMs


class GurobiMode(str, Enum):
    """How a task gets its Gurobi license."""

    NONE = "none"
    WLS_SECRET = "wls_secret"  # WLSACCESSID / WLSSECRET / LICENSEID from the provider's secret manager
    BYO = "byo"  # a license file or token supplied with the job


class TaskState(str, Enum):
    """States of one task."""

    QUEUED = "queued"
    PLACING = "placing"
    STARTING = "starting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobState(str, Enum):
    """States of a job, derived from its tasks."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = frozenset({TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED})
ACTIVE = frozenset({TaskState.PLACING, TaskState.STARTING, TaskState.RUNNING})

TRANSITIONS: Dict[TaskState, frozenset] = {
    TaskState.QUEUED: frozenset({TaskState.PLACING, TaskState.CANCELLED}),
    # A placement that finds no capacity returns the task to the queue.
    TaskState.PLACING: frozenset({TaskState.STARTING, TaskState.QUEUED, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.STARTING: frozenset({TaskState.RUNNING, TaskState.PREEMPTED, TaskState.FAILED, TaskState.CANCELLED}),
    TaskState.RUNNING: frozenset({TaskState.SUCCEEDED, TaskState.FAILED, TaskState.PREEMPTED, TaskState.CANCELLED}),
    TaskState.PREEMPTED: frozenset({TaskState.RETRYING, TaskState.FAILED, TaskState.CANCELLED}),
    # A failed task can be retried by the user or by the retry policy.
    TaskState.FAILED: frozenset({TaskState.RETRYING}),
    TaskState.RETRYING: frozenset({TaskState.QUEUED, TaskState.CANCELLED}),
    TaskState.SUCCEEDED: frozenset(),
    TaskState.CANCELLED: frozenset({TaskState.RETRYING}),
}


class TransitionError(ValueError):
    """An illegal task state change."""


def check_transition(old: TaskState, new: TaskState) -> None:
    """Raise :class:`TransitionError` unless ``old -> new`` is legal."""
    if new not in TRANSITIONS[TaskState(old)]:
        raise TransitionError(f"illegal task transition {TaskState(old).value} -> {TaskState(new).value}")


def job_state(states: Iterable[TaskState], cancel_requested: bool) -> JobState:
    """A job's state from its tasks' states."""
    states = [TaskState(s) for s in states]
    if not states:
        return JobState.CANCELLED if cancel_requested else JobState.SUCCEEDED
    if all(s in TERMINAL for s in states):
        if cancel_requested and any(s == TaskState.CANCELLED for s in states):
            return JobState.CANCELLED
        if all(s == TaskState.SUCCEEDED for s in states):
            return JobState.SUCCEEDED
        return JobState.FAILED
    if all(s == TaskState.QUEUED for s in states):
        return JobState.QUEUED
    return JobState.RUNNING


def new_id(prefix: str) -> str:
    """A short unique id such as ``job-1f3a9c2e5b7d``."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@dataclass
class Resources:
    """What one task asks for.

    Attributes:
        cpus: vCPUs.
        memory_gb: Memory in GiB.
        spot: Use spot or preemptible capacity.
        max_wall_seconds: Kill the task after this wall time (None: no limit).
        idle_seconds: Kill the task when no progress is reported for this long (None: no limit).
        max_retries: Automatic retries after a failure or preemption.
        machine_type: Provider machine type, or None for the backend's default.
        region: Provider region, or None for the backend's default.
    """

    cpus: float = 1.0
    memory_gb: float = 3.0
    spot: bool = False
    max_wall_seconds: Optional[float] = None
    idle_seconds: Optional[float] = None
    max_retries: int = 2
    machine_type: Optional[str] = None
    region: Optional[str] = None


@dataclass
class CellSpec:
    """One cell of a job: one ``main.py test_sim`` invocation.

    Attributes:
        cell_id: Stable id, unique within the job (for example ``30d/riomaior/100/emp/lm_cls/cf70/bpc``).
        argv: Arguments after ``main.py``, starting with ``test_sim``.
        policy_dir: Local directory with the cell's policy YAML (``WSR_POLICY_CONFIG_DIR``), or None.
        env: Extra environment variables (never secrets).
        expected_wall_seconds: The submitter's runtime estimate, or None.
        needs_gurobi: The cell runs a policy that needs a Gurobi license.
    """

    cell_id: str
    argv: List[str]
    policy_dir: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    expected_wall_seconds: Optional[float] = None
    needs_gurobi: bool = False


@dataclass
class JobSpec:
    """What the user submits.

    Attributes:
        name: Human-readable job name.
        cells: The job's cells; a single command is a one-cell job.
        provider: Where to run.
        compute: The kind of compute service.
        resources: Per-task resources and limits.
        max_parallel: Tasks of this job running at once.
        gurobi: How tasks get a Gurobi license.
        gurobi_secret: Secret-store name of the license (``wls_secret``) or of the supplied license (``byo``).
        image: Container image reference, or None for the local backend.
        labels: Free-form labels.
        inputs: Paths relative to the project root that remote tasks need (staged to object storage at submit).
        inputs_manifest: Object-store key of the staged inputs manifest (set by the service at submit).
    """

    name: str
    cells: List[CellSpec]
    provider: Provider = Provider.LOCAL
    compute: ComputeKind = ComputeKind.PROCESS
    resources: Resources = field(default_factory=Resources)
    max_parallel: int = 1
    gurobi: GurobiMode = GurobiMode.NONE
    gurobi_secret: Optional[str] = None
    image: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=lambda: ["data/simulator"])
    inputs_manifest: Optional[str] = None

    def validate(self) -> None:
        """Raise ``ValueError`` for an unrunnable spec."""
        if not self.cells:
            raise ValueError("a job needs at least one cell")
        ids = [cell.cell_id for cell in self.cells]
        if len(set(ids)) != len(ids):
            raise ValueError("cell ids must be unique within a job")
        for cell in self.cells:
            if not cell.argv or cell.argv[0] != "test_sim":
                raise ValueError(f"cell {cell.cell_id!r}: argv must start with 'test_sim'")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")
        if any(cell.needs_gurobi for cell in self.cells) and GurobiMode(self.gurobi) == GurobiMode.NONE:
            raise ValueError("a cell needs Gurobi but the job has no Gurobi license mode")
        if GurobiMode(self.gurobi) != GurobiMode.NONE and not self.gurobi_secret:
            raise ValueError("a Gurobi license mode needs gurobi_secret (a secret-store name, never the value)")


@dataclass
class TaskRecord:
    """A task as the control plane stores it.

    Attributes:
        task_id: Unique id.
        job_id: Owning job.
        cell: The cell the task runs.
        state: Current state.
        attempt: 1 for the first run, incremented on every retry.
        lease_owner: Worker or scheduler holding the task, if any.
        lease_expires: Unix time the lease ends.
        backend_handle: The backend's id for the running attempt.
        progress: Latest progress record (for example ``{"day": 12, "days": 30}``).
        heartbeat: Unix time of the latest progress or heartbeat.
        started_at: Unix time the current attempt started.
        finished_at: Unix time the task reached a terminal state.
        message: Latest failure or status message.
    """

    task_id: str
    job_id: str
    cell: CellSpec
    state: TaskState = TaskState.QUEUED
    attempt: int = 1
    lease_owner: Optional[str] = None
    lease_expires: Optional[float] = None
    backend_handle: Optional[str] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    heartbeat: Optional[float] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    message: Optional[str] = None


@dataclass
class JobRecord:
    """A job as the control plane stores it.

    Attributes:
        job_id: Unique id.
        spec: The submitted spec.
        created_at: Unix time of submission.
        cancel_requested: The user asked to cancel the job.
    """

    job_id: str
    spec: JobSpec
    created_at: float = field(default_factory=time.time)
    cancel_requested: bool = False


@dataclass
class Event:
    """A lifecycle event of a job or task.

    Attributes:
        job_id: Job the event belongs to.
        kind: Event kind (``submitted``, ``state``, ``retry``, ``cancel_requested``, ``result``, ...).
        task_id: Task the event belongs to, if any.
        data: Event payload.
        at: Unix time.
    """

    job_id: str
    kind: str
    task_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)


@dataclass
class ResultFile:
    """One uploaded result file.

    Attributes:
        relpath: Path relative to the task's output root (the ``outputs/`` layout).
        key: Object-store key.
        sha256: Content hash.
        size: Size in bytes.
    """

    relpath: str
    key: str
    sha256: str
    size: int


def to_dict(record: Any) -> Dict[str, Any]:
    """A JSON-ready dict of a record (enums as their values)."""

    def plain(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, list):
            return [plain(item) for item in value]
        return value

    return plain(asdict(record))


def cell_from_dict(data: Dict[str, Any]) -> CellSpec:
    """Inverse of :func:`to_dict` for a cell."""
    return CellSpec(**data)


def spec_from_dict(data: Dict[str, Any]) -> JobSpec:
    """Inverse of :func:`to_dict` for a job spec."""
    data = dict(data)
    data["cells"] = [cell_from_dict(cell) for cell in data["cells"]]
    data["provider"] = Provider(data["provider"])
    data["compute"] = ComputeKind(data["compute"])
    data["gurobi"] = GurobiMode(data["gurobi"])
    data["resources"] = Resources(**data["resources"])
    return JobSpec(**data)


def task_from_dict(data: Dict[str, Any]) -> TaskRecord:
    """Inverse of :func:`to_dict` for a task."""
    data = dict(data)
    data["cell"] = cell_from_dict(data["cell"])
    data["state"] = TaskState(data["state"])
    return TaskRecord(**data)


def job_from_dict(data: Dict[str, Any]) -> JobRecord:
    """Inverse of :func:`to_dict` for a job."""
    data = dict(data)
    data["spec"] = spec_from_dict(data["spec"])
    return JobRecord(**data)

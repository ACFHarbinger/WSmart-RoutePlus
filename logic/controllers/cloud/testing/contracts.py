"""Contract suites every implementation of the cloud interfaces must pass.

Each suite is a function taking a factory and raising ``AssertionError`` on the
first violation. Provider packages call them from their own tests, so the
contracts live in one place:

    def test_firestore_contract():
        control_plane_contract(lambda: FirestoreControlPlane(client=FakeFirestore()))

Attributes:
    control_plane_contract: Queue, leases, transitions, events, metrics and results.
    object_store_contract: Put, get, list, delete and content addressing.
    secret_store_contract: Put, get and delete.
    backend_contract: Submit, poll, cancel and logs.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, List

from logic.controllers.cloud.interfaces import (
    AttemptState,
    ComputeBackend,
    ControlPlane,
    LaunchSpec,
    ObjectStore,
    SecretStore,
    StaleStateError,
)
from logic.controllers.cloud.model import (
    CellSpec,
    Event,
    JobRecord,
    JobSpec,
    ResultFile,
    TaskRecord,
    TaskState,
    TransitionError,
    new_id,
)


def _job(n_tasks: int = 3):
    cells = [CellSpec(cell_id=f"c{i}", argv=["test_sim", f"sim.seed={i}"]) for i in range(n_tasks)]
    job = JobRecord(job_id=new_id("job"), spec=JobSpec(name="contract", cells=cells))
    tasks = [TaskRecord(task_id=new_id("task"), job_id=job.job_id, cell=cell) for cell in cells]
    return job, tasks


def control_plane_contract(make: Callable[[], ControlPlane]) -> None:
    """Check a :class:`ControlPlane` implementation."""
    cp = make()
    job, tasks = _job(3)
    cp.create_job(job, tasks)

    # Records round-trip.
    assert cp.get_job(job.job_id).spec.name == "contract"
    assert [t.task_id for t in cp.list_tasks(job.job_id)] == [t.task_id for t in tasks]
    assert cp.get_task(tasks[0].task_id).cell.argv == ["test_sim", "sim.seed=0"]
    assert job.job_id in [j.job_id for j in cp.list_jobs()]
    for missing in (lambda: cp.get_job("job-missing"), lambda: cp.get_task("task-missing")):
        try:
            missing()
        except KeyError:
            pass
        else:
            raise AssertionError("unknown ids must raise KeyError")

    # Leases are FIFO and exclusive.
    first = cp.lease_task("w1", 60.0)
    assert first is not None and first.task_id == tasks[0].task_id and first.state == TaskState.PLACING
    assert first.lease_owner == "w1"
    assert cp.renew_lease(first.task_id, "w1", 60.0)
    assert not cp.renew_lease(first.task_id, "w2", 60.0)

    # Expired leases go back to the queue.
    expired = cp.expire_leases(now=time.time() + 3600)
    assert expired == [first.task_id], expired
    assert cp.get_task(first.task_id).state == TaskState.QUEUED

    # Concurrent leasing never hands out a task twice.
    leased: List[str] = []
    lock = threading.Lock()

    def worker(name: str) -> None:
        while True:
            task = cp.lease_task(name, 60.0, job_id=job.job_id)
            if task is None:
                return
            with lock:
                leased.append(task.task_id)

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(leased) == sorted(t.task_id for t in tasks), leased
    assert cp.lease_task("w9", 60.0) is None

    # Transitions are enforced, and conditional updates detect races.
    task_id = tasks[1].task_id
    updated = cp.update_task(task_id, state=TaskState.STARTING, expected=TaskState.PLACING, backend_handle="h1")
    assert updated.state == TaskState.STARTING and updated.backend_handle == "h1"
    assert updated.lease_expires is None
    try:
        cp.update_task(task_id, state=TaskState.QUEUED)
    except TransitionError:
        pass
    else:
        raise AssertionError("starting -> queued must be rejected")
    try:
        cp.update_task(task_id, state=TaskState.RUNNING, expected=TaskState.PLACING)
    except StaleStateError:
        pass
    else:
        raise AssertionError("a stale expected state must be rejected")
    cp.update_task(task_id, state=TaskState.RUNNING, progress={"day": 3, "days": 30}, heartbeat=123.0)
    task = cp.get_task(task_id)
    assert task.progress == {"day": 3, "days": 30} and task.heartbeat == 123.0

    # Cancellation flag.
    cp.request_cancel(job.job_id)
    assert cp.get_job(job.job_id).cancel_requested

    # Events, metrics and results.
    cp.append_event(Event(job_id=job.job_id, kind="state", task_id=task_id, data={"to": "running"}, at=10.0))
    cp.append_event(Event(job_id=job.job_id, kind="retry", task_id=task_id, at=20.0))
    assert [e.kind for e in cp.list_events(job.job_id)] == ["state", "retry"]
    assert [e.kind for e in cp.list_events(job.job_id, since=15.0)] == ["retry"]
    cp.put_metrics(task_id, [{"at": 1.0, "cpu": 0.5}, {"at": 2.0, "cpu": 0.7}])
    assert [m["cpu"] for m in cp.get_metrics(task_id)] == [0.5, 0.7]
    assert [m["cpu"] for m in cp.get_metrics(task_id, since=1.5)] == [0.7]
    files = [ResultFile(relpath="a/log.json", key="results/x/a/log.json", sha256="0" * 64, size=3)]
    cp.record_results(task_id, files)
    assert cp.list_results(task_id) == files
    cp.record_results(task_id, [])
    assert cp.list_results(task_id) == []


def object_store_contract(make: Callable[[], ObjectStore]) -> None:
    """Check an :class:`ObjectStore` implementation."""
    store = make()
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "in.txt"
        source.write_text("hello")
        record = store.put_file(source, "runs/a/in.txt")
        assert record.key == "runs/a/in.txt" and record.size == 5
        assert record.sha256 == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        assert store.exists("runs/a/in.txt") and not store.exists("runs/a/missing.txt")
        store.put_file(source, "runs/b/in.txt")
        assert store.list("runs/") == ["runs/a/in.txt", "runs/b/in.txt"]
        assert store.list("runs/a") == ["runs/a/in.txt"]
        target = Path(tmp) / "out" / "nested" / "out.txt"
        store.get_file("runs/a/in.txt", target)
        assert target.read_text() == "hello"
        try:
            store.get_file("runs/missing", Path(tmp) / "x")
        except KeyError:
            pass
        else:
            raise AssertionError("a missing key must raise KeyError")
        cas = store.put_content_addressed(source)
        assert cas.key == f"cas/{record.sha256}" and store.exists(cas.key)
        assert store.put_content_addressed(source).key == cas.key
        store.delete("runs/b/in.txt")
        store.delete("runs/b/in.txt")
        assert store.list("runs/") == ["runs/a/in.txt"]


def secret_store_contract(make: Callable[[], SecretStore]) -> None:
    """Check a writable :class:`SecretStore` implementation."""
    secrets = make()
    assert secrets.get("gurobi-wls") is None
    secrets.put("gurobi-wls", "s3cr3t")
    assert secrets.get("gurobi-wls") == "s3cr3t"
    secrets.put("gurobi-wls", "rotated")
    assert secrets.get("gurobi-wls") == "rotated"
    secrets.delete("gurobi-wls")
    secrets.delete("gurobi-wls")
    assert secrets.get("gurobi-wls") is None


def backend_contract(
    make: Callable[[], ComputeBackend],
    launch: Callable[[str], LaunchSpec],
    timeout: float = 30.0,
    poll_seconds: float = 0.05,
) -> None:
    """Check a :class:`ComputeBackend` implementation.

    Args:
        make: Backend factory.
        launch: ``outcome -> LaunchSpec`` for ``"succeed"``, ``"fail"`` and ``"hang"``.
        timeout: Seconds to wait for an attempt to finish.
        poll_seconds: Seconds between polls.
    """
    backend = make()
    caps = backend.capabilities()
    assert backend.kind in caps.kinds and caps.provider == backend.provider

    def wait(handle: str) -> AttemptState:
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = backend.poll(handle).state
            if state not in (AttemptState.PENDING, AttemptState.RUNNING):
                return state
            time.sleep(poll_seconds)
        raise AssertionError(f"attempt {handle} did not finish in {timeout}s")

    ok = backend.submit(launch("succeed"))
    assert wait(ok) == AttemptState.SUCCEEDED
    assert isinstance(backend.logs(ok), str)
    failed = backend.submit(launch("fail"))
    status = None
    assert wait(failed) == AttemptState.FAILED
    status = backend.poll(failed)
    assert status.exit_code not in (None, 0) or status.message
    hung = backend.submit(launch("hang"))
    backend.cancel(hung)
    assert wait(hung) != AttemptState.SUCCEEDED
    backend.cancel(hung)  # cancelling a finished attempt is a no-op
    assert backend.poll("no-such-handle").state in (AttemptState.UNKNOWN, AttemptState.FAILED)

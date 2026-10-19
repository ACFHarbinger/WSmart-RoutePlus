"""The local implementations pass the cloud runner's contract suites."""

import sys

import pytest
from logic.controllers.cloud.interfaces import LaunchSpec
from logic.controllers.cloud.local import (
    FilesystemObjectStore,
    LocalProcessBackend,
    MemorySecretStore,
    SQLiteControlPlane,
)
from logic.controllers.cloud.model import CellSpec, Resources, TaskRecord
from logic.controllers.cloud.testing import (
    MemoryObjectStore,
    ScriptedBackend,
    backend_contract,
    control_plane_contract,
    object_store_contract,
    secret_store_contract,
)


@pytest.mark.unit
def test_sqlite_control_plane_in_memory():
    control_plane_contract(lambda: SQLiteControlPlane(":memory:"))


@pytest.mark.unit
def test_sqlite_control_plane_on_disk(tmp_path):
    control_plane_contract(lambda: SQLiteControlPlane(tmp_path / "state" / "control.db"))


@pytest.mark.unit
def test_sqlite_leases_are_exclusive_across_connections(tmp_path):
    from logic.controllers.cloud.testing.contracts import _job

    path = tmp_path / "control.db"
    first, second = SQLiteControlPlane(path), SQLiteControlPlane(path)
    job, tasks = _job(2)
    first.create_job(job, tasks)
    a, b = first.lease_task("a", 60.0), second.lease_task("b", 60.0)
    assert {a.task_id, b.task_id} == {t.task_id for t in tasks}
    assert first.lease_task("a", 60.0) is None and second.lease_task("b", 60.0) is None


@pytest.mark.unit
def test_object_stores(tmp_path):
    object_store_contract(lambda: FilesystemObjectStore(tmp_path / "objects"))
    object_store_contract(MemoryObjectStore)
    with pytest.raises(ValueError):
        FilesystemObjectStore(tmp_path / "o").exists("../escape")


@pytest.mark.unit
def test_memory_secret_store():
    secret_store_contract(MemorySecretStore)


def _launch(tmp_path, outcome):
    programs = {
        "succeed": "print('ok')",
        "fail": "import sys; sys.exit(3)",
        "hang": "import time; time.sleep(60)",
    }
    task = TaskRecord(task_id=f"task-{outcome}", job_id="job-x", cell=CellSpec(cell_id=outcome, argv=["test_sim"]))
    return LaunchSpec(
        task=task,
        resources=Resources(memory_gb=0.0),
        argv=[sys.executable, "-c", programs[outcome]],
        workdir=str(tmp_path / outcome),
    )


@pytest.mark.unit
def test_local_process_backend(tmp_path):
    backend_contract(lambda: LocalProcessBackend(cwd=tmp_path), lambda outcome: _launch(tmp_path, outcome))


@pytest.mark.unit
def test_local_backend_resolves_secrets_without_logging_them(tmp_path):
    backend = LocalProcessBackend(cwd=tmp_path, secrets=MemorySecretStore({"wls": "s3cr3t"}))
    launch = _launch(tmp_path, "succeed")
    launch.argv = [sys.executable, "-c", "import os; print(len(os.environ['WSR_GUROBI_LICENSE']))"]
    launch.secret_env = {"WSR_GUROBI_LICENSE": "wls"}
    handle = backend.submit(launch)
    import time

    for _ in range(200):
        if backend.poll(handle).state.value == "succeeded":
            break
        time.sleep(0.05)
    assert backend.logs(handle).strip() == "6"
    assert "s3cr3t" not in (tmp_path / "succeed" / "attempt-1" / "launch.json").read_text()
    launch.secret_env = {"X": "missing"}
    with pytest.raises(ValueError, match="not available"):
        backend.submit(launch)


@pytest.mark.unit
def test_scripted_backend(tmp_path):
    scripted = {"succeed": "succeed", "fail": "fail", "hang": "hang"}
    backend_contract(
        lambda: ScriptedBackend(script=lambda launch: scripted[launch.task.cell.cell_id]),
        lambda outcome: _launch(tmp_path, outcome),
        poll_seconds=0.0,
    )

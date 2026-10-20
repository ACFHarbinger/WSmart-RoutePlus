"""Crash/restart tests using durable SQLite ownership and filesystem journals."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest
from logic.controllers.cloud.local import SQLiteControlPlane, control_plane
from logic.controllers.cloud.model import CellSpec, ComputeKind, JobSpec, Provider, TaskState
from logic.controllers.cloud.providers.azure import AzureVmBackend
from logic.controllers.cloud.service import CloudService, RemoteConfig
from logic.controllers.cloud.testing import FakeClock, MemoryObjectStore


class Crash(BaseException):
    """Simulate process death: ordinary exception cleanup must not run."""


class Missing(Exception):
    status_code = 404


class RemoteResources:
    def __init__(self, stage: str = "") -> None:
        self.stage = stage
        self.vms: Dict[str, Any] = {}
        self.nics: Dict[str, Any] = {}
        self.deleted = []
        self.backend = None

    def die(self, stage: str) -> None:
        if self.stage == stage:
            self.stage = ""
            raise Crash(stage)

    def network(self) -> Any:
        def create(**kwargs: Any) -> Any:
            name = kwargs["network_interface_name"]
            assert any(record["nic_name"] == name for record in self.backend._load_provisions())
            self.die("before_nic")
            self.nics[name] = kwargs

            def result() -> Any:
                self.die("after_nic")
                return {"id": f"/nic/{name}"}

            return SimpleNamespace(result=result)

        def delete(**kwargs: Any) -> Any:
            name = kwargs["network_interface_name"]
            self.deleted.append(("nic", name))
            self.nics.pop(name, None)
            return SimpleNamespace(result=lambda: None)

        return SimpleNamespace(begin_create_or_update=create, begin_delete=delete)

    def compute(self) -> Any:
        def create(**kwargs: Any) -> Any:
            name = kwargs["vm_name"]
            assert any(record["vm_name"] == name for record in self.backend._load_provisions())
            self.vms[name] = kwargs

            def result() -> Any:
                self.die("after_vm")
                return {"name": name}

            return SimpleNamespace(result=result)

        def delete(**kwargs: Any) -> Any:
            name = kwargs["vm_name"]
            self.deleted.append(("vm", name))
            self.vms.pop(name, None)
            return SimpleNamespace(result=lambda: None)

        def view(**kwargs: Any) -> Any:
            if kwargs["vm_name"] not in self.vms:
                raise Missing()
            return {"statuses": [{"code": "PowerState/running"}]}

        return SimpleNamespace(begin_create_or_update=create, begin_delete=delete, instance_view=view)


def service_at(root: Path, remote: RemoteResources, clock: FakeClock) -> CloudService:
    backend = AzureVmBackend(
        client=remote.compute(), network_client=remote.network(), orphan_file=root / "orphans.json"
    )
    remote.backend = backend
    return CloudService(
        SQLiteControlPlane(root / "control.db"),
        MemoryObjectStore(),
        {(Provider.AZURE, ComputeKind.VM): backend},
        root / "work",
        clock=clock,
        remote=RemoteConfig(control_plane_uri=f"sqlite:///{root}/control.db", store_uri="memory://", project_root=root),
    )


def enqueue(service: CloudService) -> str:
    return service.submit(
        JobSpec(
            name="recover",
            provider=Provider.AZURE,
            compute=ComputeKind.VM,
            image="runner",
            inputs=[],
            cells=[CellSpec(cell_id="cell", argv=["test_sim"])],
        )
    ).job_id


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock(1000)
    monkeypatch.setattr(control_plane, "time", SimpleNamespace(time=fake))
    return fake


@pytest.mark.parametrize("stage", ["before_nic", "after_nic", "after_vm", "before_handoff"])
def test_restart_cleans_unowned_allocation_before_resubmitting(
    tmp_path: Path,
    clock: FakeClock,
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = RemoteResources(stage)
    first = service_at(tmp_path, remote, clock)
    job_id = enqueue(first)
    if stage == "before_handoff":
        original = first._set

        def die_before_handoff(task: Any, state: TaskState, **kwargs: Any) -> Any:
            if state == TaskState.STARTING:
                raise Crash(stage)
            return original(task, state, **kwargs)

        monkeypatch.setattr(first, "_set", die_before_handoff)
    with pytest.raises(Crash):
        first.tick()
    intent = remote.backend._load_provisions()[0]
    first.cp._conn.close()
    clock.advance(121)
    restarted = service_at(tmp_path, remote, clock)
    restarted.tick()
    assert ("vm", intent["vm_name"]) in remote.deleted
    assert ("nic", intent["nic_name"]) in remote.deleted
    assert intent["vm_name"] not in remote.vms
    assert intent["nic_name"] not in remote.nics
    assert len(remote.vms) == len(remote.nics) == 1
    assert restarted.cp.list_tasks(job_id)[0].state == TaskState.STARTING
    assert remote.backend._load_provisions() == []


def test_crash_after_durable_handoff_keeps_owned_vm(tmp_path: Path, clock: FakeClock) -> None:
    remote = RemoteResources()
    first = service_at(tmp_path, remote, clock)
    job_id = enqueue(first)

    def die(handle: str) -> None:
        raise Crash("after_handoff")

    remote.backend.confirm_provision = die
    with pytest.raises(Crash):
        first.tick()
    handle = first.cp.list_tasks(job_id)[0].backend_handle
    assert remote.backend._load_provisions()
    first.cp._conn.close()
    clock.advance(121)
    restarted = service_at(tmp_path, remote, clock)
    restarted.tick()
    assert restarted.cp.list_tasks(job_id)[0].backend_handle == handle
    assert len(remote.vms) == 1
    assert remote.deleted == []
    assert remote.backend._load_provisions() == []


def test_restart_does_not_delete_live_placing_owner(tmp_path: Path, clock: FakeClock) -> None:
    remote = RemoteResources("after_vm")
    first = service_at(tmp_path, remote, clock)
    enqueue(first)
    with pytest.raises(Crash):
        first.tick()
    first.cp._conn.close()
    restarted = service_at(tmp_path, remote, clock)
    restarted.tick()  # lease remains valid; recovery must wait
    assert len(remote.vms) == 1
    assert remote.deleted == []
    assert len(remote.backend._load_provisions()) == 1


def test_nic_lro_error_is_cleaned_and_journalled(tmp_path: Path, clock: FakeClock) -> None:
    remote = RemoteResources()
    service = service_at(tmp_path, remote, clock)
    backend = remote.backend
    original = backend._network_client.begin_create_or_update

    def create(**kwargs: Any) -> Any:
        original(**kwargs)  # allocate before failure

        def fail() -> Any:
            raise RuntimeError("NIC provisioning failed")

        return SimpleNamespace(result=fail)

    backend._network_client.begin_create_or_update = create
    job_id = enqueue(service)
    service.tick()
    assert service.status(job_id)["state"] == "failed"
    assert not remote.vms and not remote.nics
    assert backend._load_provisions() == []


def test_other_control_plane_cannot_reclaim_intent(tmp_path: Path, clock: FakeClock) -> None:
    remote = RemoteResources("after_vm")
    first = service_at(tmp_path, remote, clock)
    enqueue(first)
    with pytest.raises(Crash):
        first.tick()
    backend = remote.backend
    backend.configure_recovery(first.cp, "different-control-plane")
    backend.reconcile_provisions()
    assert len(remote.vms) == 1
    assert remote.deleted == []
    assert backend._load_provisions()


def test_expired_worker_cannot_overwrite_new_lease_owner(tmp_path: Path, clock: FakeClock) -> None:
    remote = RemoteResources()
    service = service_at(tmp_path, remote, clock)
    job_id = enqueue(service)
    backend = remote.backend
    original = backend._client.begin_create_or_update

    def create(**kwargs: Any) -> Any:
        poller = original(**kwargs)
        clock.advance(121)
        service.cp.expire_leases(clock())
        replacement = service.cp.lease_task("replacement-scheduler", 120, job_id=job_id)
        assert replacement is not None
        return poller

    backend._client.begin_create_or_update = create
    service.tick()
    task = service.cp.list_tasks(job_id)[0]
    assert task.state == TaskState.PLACING
    assert task.lease_owner == "replacement-scheduler"
    assert task.backend_handle is None
    assert not remote.vms and not remote.nics


def test_lease_is_renewed_while_provisioning_blocks() -> None:
    import threading

    from logic.controllers.cloud.model import TaskRecord
    from logic.controllers.cloud.provisioning_lease import renew_provisioning_lease

    renewed = threading.Event()
    cp = SimpleNamespace(renew_lease=lambda *args: renewed.set() or True)
    task = TaskRecord(task_id="task", job_id="job", cell=CellSpec(cell_id="cell", argv=["run"]), lease_owner="owner")
    with renew_provisioning_lease(cp, task, 0.03):
        assert renewed.wait(2)


def test_real_process_exit_retains_recoverable_intent(tmp_path: Path) -> None:
    import subprocess
    import sys
    import textwrap
    import time

    script = textwrap.dedent("""
        import os, sys, time
        from pathlib import Path
        from types import SimpleNamespace
        from logic.test.unit.controllers.cloud.providers.test_azure_provisioning_recovery import (
            RemoteResources, service_at, enqueue,
        )
        root = Path(sys.argv[1])
        remote = RemoteResources()
        service = service_at(root, remote, time.time)
        enqueue(service)
        backend = remote.backend
        original = backend._network_client.begin_create_or_update
        def create(**kwargs):
            original(**kwargs)
            (root / "allocated-nic").write_text(kwargs["network_interface_name"])
            return SimpleNamespace(result=lambda: os._exit(73))
        backend._network_client.begin_create_or_update = create
        service.tick()
    """)
    child = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, timeout=15)
    assert child.returncode == 73, child.stderr.decode()
    assert (tmp_path / "allocated-nic").exists()
    remote = RemoteResources()
    restarted = service_at(tmp_path, remote, FakeClock(time.time() + 121))
    backend = remote.backend
    original = backend._network_client.begin_delete

    def delete(**kwargs: Any) -> Any:
        (tmp_path / "allocated-nic").unlink(missing_ok=True)
        return original(**kwargs)

    backend._network_client.begin_delete = delete
    assert backend._load_provisions()
    restarted.tick()
    assert not (tmp_path / "allocated-nic").exists()
    assert backend._load_provisions() == []

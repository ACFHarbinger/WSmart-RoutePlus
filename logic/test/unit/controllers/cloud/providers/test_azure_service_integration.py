"""Integrated service tests driving Azure compute backends through CloudService.

Verifies lifecycle, cancellation, spot eviction retries, and cleanup ownership
without real cloud credentials (all tests use in-process fakes).
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from logic.controllers.cloud.local import SQLiteControlPlane
from logic.controllers.cloud.model import (
    CellSpec,
    ComputeKind,
    JobSpec,
    JobState,
    Provider,
    Resources,
    TaskState,
)
from logic.controllers.cloud.providers.azure import AzureVmBackend, ContainerAppsJobsBackend
from logic.controllers.cloud.service import CloudService, RemoteConfig
from logic.controllers.cloud.testing import FakeClock, MemoryObjectStore


class StatefulFakeVmClient:
    """In-memory Azure VM client that simulates real Azure ARM state transitions."""

    def __init__(self) -> None:
        self.vms: Dict[str, Dict[str, Any]] = {}
        self.virtual_machines = self
        self.deleted_calls: List[str] = []
        self.create_calls: List[str] = []

    def begin_create_or_update(self, resource_group_name: str, vm_name: str, parameters: Dict[str, Any]) -> Any:
        key = f"{resource_group_name}:{vm_name}"
        self.create_calls.append(vm_name)
        self.vms[key] = {
            "name": vm_name,
            "parameters": parameters,
            "status": "PowerState/starting",
            "exit_code": None,
            "deleted": False,
            "is_spot": parameters.get("priority") == "Spot",
        }

        class CreatePoller:
            def result(self) -> Dict[str, Any]:
                return {"name": vm_name}

        return CreatePoller()

    def instance_view(self, resource_group_name: str, vm_name: str) -> Optional[Dict[str, Any]]:
        key = f"{resource_group_name}:{vm_name}"
        if key not in self.vms:
            return None
        v = self.vms[key]
        tags = {}
        if v["exit_code"] is not None:
            tags["wsr-exit-code"] = str(v["exit_code"])

        return {
            "statuses": [{"code": v["status"]}],
            "tags": tags,
            "priority": "Spot" if v["is_spot"] else "Regular",
        }

    def get(self, resource_group_name: str, vm_name: str) -> Optional[Dict[str, Any]]:
        return self.instance_view(resource_group_name, vm_name)

    def begin_delete(self, resource_group_name: str, vm_name: str) -> Any:
        key = f"{resource_group_name}:{vm_name}"
        self.deleted_calls.append(key)
        if key in self.vms:
            self.vms[key]["deleted"] = True
            self.vms[key]["status"] = "PowerState/deallocated"
            if self.vms[key]["exit_code"] is None:
                self.vms[key]["exit_code"] = 143

        class DeletePoller:
            def result(self) -> None:
                return None

        return DeletePoller()


class FakeNetworkClient:
    """In-memory Azure Network client for NIC operations."""

    def __init__(self) -> None:
        self.network_interfaces = self
        self.nics: Dict[str, Dict[str, Any]] = {}
        self.deleted_nics: List[str] = []

    def begin_create_or_update(
        self, resource_group_name: str, network_interface_name: str, parameters: Dict[str, Any]
    ) -> Any:
        key = f"{resource_group_name}:{network_interface_name}"
        self.nics[key] = parameters

        class NicPoller:
            def result(self) -> Dict[str, Any]:
                return {
                    "id": f"/subscriptions/sub/resourceGroups/{resource_group_name}/providers/Microsoft.Network/networkInterfaces/{network_interface_name}"
                }

        return NicPoller()

    def begin_delete(self, resource_group_name: str, network_interface_name: str) -> Any:
        key = f"{resource_group_name}:{network_interface_name}"
        self.deleted_nics.append(key)
        self.nics.pop(key, None)

        class DeletePoller:
            def result(self) -> None:
                return None

        return DeletePoller()


def _make_azure_service(
    tmp_path: Path,
    backend: Any,
    compute_kind: ComputeKind = ComputeKind.VM,
    clock: Optional[FakeClock] = None,
) -> CloudService:
    key = (Provider.AZURE, compute_kind)
    remote = RemoteConfig(
        control_plane_uri="memory://",
        store_uri="memory://",
        project_root=tmp_path,
    )
    return CloudService(
        control_plane=SQLiteControlPlane(":memory:"),
        store=MemoryObjectStore(),
        backends={key: backend},
        workroot=tmp_path / "work",
        python="python",
        clock=clock or FakeClock(),
        remote=remote,
    )


@pytest.mark.unit
def test_azure_vm_service_lifecycle_success(tmp_path: Path) -> None:
    fake_vm_client = StatefulFakeVmClient()
    fake_net_client = FakeNetworkClient()
    orphan_file = tmp_path / "orphans.json"
    backend = AzureVmBackend(
        client=fake_vm_client,
        network_client=fake_net_client,
        orphan_file=orphan_file,
    )
    service = _make_azure_service(tmp_path, backend)

    spec = JobSpec(
        name="azure-test",
        provider=Provider.AZURE,
        compute=ComputeKind.VM,
        image="wsmart-route:v1.1.0",
        inputs=[],
        cells=[CellSpec(cell_id="c0", argv=["test_sim", "sim.seed=42"])],
    )
    job = service.submit(spec)

    # 1. Tick to submit task to Azure VM backend
    service.tick()
    status = service.status(job.job_id)
    assert status["tasks"][0]["state"] == TaskState.STARTING.value
    assert len(fake_vm_client.vms) == 1
    vm_key = next(iter(fake_vm_client.vms.keys()))
    assert fake_vm_client.vms[vm_key]["status"] == "PowerState/starting"

    # 2. Advance VM to PowerState/running
    fake_vm_client.vms[vm_key]["status"] = "PowerState/running"
    service.tick()
    status = service.status(job.job_id)
    assert status["tasks"][0]["state"] == TaskState.RUNNING.value

    # 3. Simulate VM execution finishing: stopped with exit code 0
    fake_vm_client.vms[vm_key]["status"] = "PowerState/deallocated"
    fake_vm_client.vms[vm_key]["exit_code"] = 0

    service.tick()
    status = service.status(job.job_id)
    assert status["state"] == JobState.SUCCEEDED.value
    assert status["tasks"][0]["state"] == TaskState.SUCCEEDED.value
    # Deletion was awaited and executed on completion
    assert vm_key in fake_vm_client.deleted_calls
    assert fake_vm_client.vms[vm_key]["deleted"] is True


@pytest.mark.unit
def test_azure_vm_service_cancellation(tmp_path: Path) -> None:
    fake_vm_client = StatefulFakeVmClient()
    fake_net_client = FakeNetworkClient()
    orphan_file = tmp_path / "orphans.json"
    backend = AzureVmBackend(
        client=fake_vm_client,
        network_client=fake_net_client,
        orphan_file=orphan_file,
    )
    service = _make_azure_service(tmp_path, backend)

    spec = JobSpec(
        name="azure-cancel",
        provider=Provider.AZURE,
        compute=ComputeKind.VM,
        image="wsmart-route:v1.1.0",
        inputs=[],
        cells=[CellSpec(cell_id="c0", argv=["test_sim"])],
    )
    job = service.submit(spec)
    service.tick()

    # Move to running
    vm_key = next(iter(fake_vm_client.vms.keys()))
    fake_vm_client.vms[vm_key]["status"] = "PowerState/running"
    service.tick()
    assert service.status(job.job_id)["tasks"][0]["state"] == TaskState.RUNNING.value

    # Request cancellation
    service.cancel(job.job_id)

    # Tick cancels active task and deletes remote VM
    service.tick()
    status = service.status(job.job_id)
    assert status["state"] == JobState.CANCELLED.value
    assert status["tasks"][0]["state"] == TaskState.CANCELLED.value
    assert vm_key in fake_vm_client.deleted_calls
    assert fake_vm_client.vms[vm_key]["deleted"] is True


@pytest.mark.unit
def test_azure_vm_service_spot_eviction_retry(tmp_path: Path) -> None:
    fake_vm_client = StatefulFakeVmClient()
    fake_net_client = FakeNetworkClient()
    orphan_file = tmp_path / "orphans.json"
    backend = AzureVmBackend(
        client=fake_vm_client,
        network_client=fake_net_client,
        orphan_file=orphan_file,
    )
    clock = FakeClock(100.0)
    service = _make_azure_service(tmp_path, backend, clock=clock)

    spec = JobSpec(
        name="azure-spot",
        provider=Provider.AZURE,
        compute=ComputeKind.VM,
        image="wsmart-route:v1.1.0",
        inputs=[],
        resources=Resources(spot=True, max_retries=2),
        cells=[CellSpec(cell_id="c0", argv=["test_sim"])],
    )
    job = service.submit(spec)

    # First attempt starts
    service.tick()
    vm_key_1 = next(iter(fake_vm_client.vms.keys()))
    fake_vm_client.vms[vm_key_1]["status"] = "PowerState/running"
    service.tick()
    assert service.status(job.job_id)["tasks"][0]["state"] == TaskState.RUNNING.value

    # Simulate spot eviction on Attempt 0
    fake_vm_client.vms[vm_key_1]["status"] = "ProvisioningState/failed/SpotEviction"
    service.tick()

    # Eviction triggers cleanup and immediately requeues for Attempt 2
    assert vm_key_1 in fake_vm_client.deleted_calls
    status = service.status(job.job_id)
    assert status["tasks"][0]["attempt"] == 2
    assert status["tasks"][0]["state"] == TaskState.STARTING.value

    # Second attempt was submitted
    assert len(fake_vm_client.create_calls) == 2

    # Second attempt succeeds
    vm_key_2 = next(key for key in fake_vm_client.vms if key != vm_key_1)
    fake_vm_client.vms[vm_key_2]["status"] = "PowerState/deallocated"
    fake_vm_client.vms[vm_key_2]["exit_code"] = 0
    service.tick()

    final_status = service.status(job.job_id)
    assert final_status["state"] == JobState.SUCCEEDED.value
    assert final_status["tasks"][0]["state"] == TaskState.SUCCEEDED.value
    assert vm_key_1 in fake_vm_client.deleted_calls


@pytest.mark.unit
def test_azure_vm_service_submit_capacity_and_partial_teardown(tmp_path: Path) -> None:
    orphan_file = tmp_path / "orphans.json"
    create_attempts = 0
    teardown_calls: List[str] = []

    class FailingCreatePoller:
        def result(self) -> None:
            raise RuntimeError("AllocationFailed: OverconstrainedAllocationRequest")

    class FlakyVmClient:
        def begin_create_or_update(self, **kwargs: Any) -> Any:
            nonlocal create_attempts
            create_attempts += 1
            if create_attempts == 1:
                return FailingCreatePoller()

            class SuccessPoller:
                def result(self) -> Dict[str, Any]:
                    return {"name": "vm"}

            return SuccessPoller()

        def begin_delete(self, **kwargs: Any) -> Any:
            teardown_calls.append("vm_delete")

            class DeletePoller:
                def result(self) -> None:
                    return None

            return DeletePoller()

        def instance_view(self, **kwargs: Any) -> Dict[str, Any]:
            return {
                "statuses": [{"code": "PowerState/deallocated"}],
                "tags": {"wsr-exit-code": "0"},
                "priority": "Regular",
            }

    class TrackingNetClient:
        def begin_create_or_update(self, **kwargs: Any) -> Any:
            class NicPoller:
                def result(self) -> Dict[str, Any]:
                    return {"id": "/nic"}

            return NicPoller()

        def begin_delete(self, **kwargs: Any) -> Any:
            teardown_calls.append("nic_delete")

            class DeletePoller:
                def result(self) -> None:
                    return None

            return DeletePoller()

    backend = AzureVmBackend(
        client=FlakyVmClient(),
        network_client=TrackingNetClient(),
        orphan_file=orphan_file,
    )
    service = _make_azure_service(tmp_path, backend)

    spec = JobSpec(
        name="azure-capacity-retry",
        provider=Provider.AZURE,
        compute=ComputeKind.VM,
        image="wsmart-route:v1.1.0",
        inputs=[],
        resources=Resources(max_retries=2),
        cells=[CellSpec(cell_id="c0", argv=["test_sim"])],
    )
    job = service.submit(spec)

    # First tick: submit fails with CapacityError
    service.tick()
    # Teardown was called: partial VM first, then NIC
    assert teardown_calls == ["vm_delete", "nic_delete"]
    # Task remains queued or requeued by scheduler
    status = service.status(job.job_id)
    assert status["tasks"][0]["state"] in (TaskState.QUEUED.value, TaskState.STARTING.value)

    # Second tick: submit succeeds, then finishes
    service.tick()
    service.tick()
    status = service.status(job.job_id)
    assert status["state"] == JobState.SUCCEEDED.value


@pytest.mark.unit
def test_azure_container_apps_service_lifecycle_and_cancel(tmp_path: Path) -> None:
    executions: Dict[str, Dict[str, Any]] = {}
    stopped_calls: List[str] = []

    class FakeContainerAppsClient:
        def __init__(self) -> None:
            self.jobs = self
            self.jobs_executions = self
            self.job_executions = self

        def begin_start(self, resource_group_name: str, job_name: str, **kwargs: Any) -> Any:
            exec_name = f"exec-{len(executions)}"
            executions[exec_name] = {"status": "Running"}

            class StartPoller:
                def result(self) -> Dict[str, Any]:
                    return {"name": exec_name}

            return StartPoller()

        def list(self, resource_group_name: str, job_name: str) -> List[Any]:
            return [
                SimpleNamespace(
                    name=k,
                    status=v["status"],
                    properties=SimpleNamespace(
                        status=v["status"],
                        start_time=None,
                        end_time=None,
                    ),
                )
                for k, v in executions.items()
            ]

        def begin_stop_execution(self, resource_group_name: str, job_name: str, job_execution_name: str) -> Any:
            stopped_calls.append(job_execution_name)
            if job_execution_name in executions:
                executions[job_execution_name]["status"] = "Stopped"

            class StopPoller:
                def result(self) -> None:
                    return None

            return StopPoller()

    fake_client = FakeContainerAppsClient()
    backend = ContainerAppsJobsBackend(client=fake_client)
    service = _make_azure_service(tmp_path, backend, compute_kind=ComputeKind.SERVERLESS)

    spec = JobSpec(
        name="aca-test",
        provider=Provider.AZURE,
        compute=ComputeKind.SERVERLESS,
        image="wsmart-route:v1.1.0",
        inputs=[],
        cells=[CellSpec(cell_id="c0", argv=["test_sim"])],
    )
    job = service.submit(spec)

    # 1. Start execution (tick 1 submits to ACA, tick 2 polls into running)
    service.tick()
    assert len(executions) == 1
    exec_name = next(iter(executions.keys()))
    assert service.status(job.job_id)["tasks"][0]["state"] == TaskState.STARTING.value

    service.tick()
    assert service.status(job.job_id)["tasks"][0]["state"] == TaskState.RUNNING.value

    # 2. Cancel running execution
    service.cancel(job.job_id)
    service.tick()

    assert service.status(job.job_id)["state"] == JobState.CANCELLED.value
    assert exec_name in stopped_calls


@pytest.mark.unit
def test_cancelled_vm_disappears_from_realistic_api(tmp_path: Path) -> None:
    class MissingResource(Exception):
        status_code = 404

    class DisappearingVmClient(StatefulFakeVmClient):
        def begin_delete(self, resource_group_name: str, vm_name: str) -> Any:
            self.vms.pop(f"{resource_group_name}:{vm_name}", None)
            return SimpleNamespace(result=lambda: None)

        def instance_view(self, resource_group_name: str, vm_name: str) -> Any:
            if f"{resource_group_name}:{vm_name}" not in self.vms:
                raise MissingResource()
            return super().instance_view(resource_group_name, vm_name)

    backend = AzureVmBackend(
        client=DisappearingVmClient(), network_client=FakeNetworkClient(), orphan_file=tmp_path / "orphans.json"
    )
    service = _make_azure_service(tmp_path, backend)
    job = service.submit(
        JobSpec(
            name="cancel",
            provider=Provider.AZURE,
            compute=ComputeKind.VM,
            image="runner",
            inputs=[],
            cells=[CellSpec(cell_id="cell", argv=["test_sim"])],
        )
    )
    service.tick()
    service.cancel(job.job_id)
    service.tick()
    assert service.status(job.job_id)["state"] == "cancelled"


@pytest.mark.unit
def test_scheduler_drains_orphans_without_active_jobs(tmp_path: Path) -> None:
    deleted = []
    backend = AzureVmBackend(
        client=SimpleNamespace(begin_delete=lambda **kwargs: deleted.append(kwargs)),
        orphan_file=tmp_path / "orphans.json",
    )
    backend._record_orphan("vm", "rg", "orphan")
    service = _make_azure_service(tmp_path, backend)
    assert service.tick() == 0
    assert len(deleted) == 1
    assert backend._load_orphans() == []

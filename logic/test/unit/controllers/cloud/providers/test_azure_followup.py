"""Regressions for Gemini's Azure follow-up; all clients remain offline."""

import shlex
from types import SimpleNamespace
from typing import Any

import pytest
from logic.controllers.cloud.interfaces import AttemptState, LaunchSpec
from logic.controllers.cloud.model import CellSpec, Resources, TaskRecord
from logic.controllers.cloud.providers.azure import AzureBatchBackend, AzureVmBackend


def test_batch_pending_model_without_execution_info() -> None:
    # Matches CloudTask from the SDK: attributes, not a dictionary, with None execution_info.
    client = SimpleNamespace(
        task=SimpleNamespace(get=lambda **kwargs: SimpleNamespace(state="active", execution_info=None))
    )
    assert AzureBatchBackend(client=client).poll("azure-batch:eastus:job:task").state == AttemptState.PENDING


def test_batch_preserves_argv_boundaries() -> None:
    recorded = {}
    client = SimpleNamespace(task=SimpleNamespace(add=lambda **kwargs: recorded.update(kwargs)))
    argv = ["python", "main.py", "path with spaces", "name=O'Brien", "value=$HOME"]
    launch = LaunchSpec(
        task=TaskRecord(task_id="task", job_id="job", cell=CellSpec(cell_id="cell", argv=argv)),
        argv=argv,
        env={},
        resources=Resources(),
    )
    AzureBatchBackend(client=client).submit(launch)
    assert shlex.split(recorded["task"]["command_line"]) == argv


@pytest.mark.parametrize("status", ["PowerState/stopped", "SpotEviction", "ProvisioningState/failed"])
def test_cleanup_lro_failure_keeps_attempt_pollable(status: str) -> None:
    waits = []

    class DeletePoller:
        def result(self) -> None:
            waits.append(True)
            if len(waits) == 1:
                raise RuntimeError("transient delete failure")

    client = SimpleNamespace(
        instance_view=lambda **kwargs: {"statuses": [{"code": status}], "tags": {"wsr-exit-code": "0"}},
        begin_delete=lambda **kwargs: DeletePoller(),
    )
    backend = AzureVmBackend(client=client)
    handle = "azure-vm:eastus:rg:vm"
    assert backend.poll(handle).state == AttemptState.UNKNOWN
    assert backend.poll(handle).state in (AttemptState.SUCCEEDED, AttemptState.PREEMPTED, AttemptState.FAILED)
    assert len(waits) == 2


def test_cancel_propagates_async_delete_failure() -> None:
    class DeletePoller:
        def result(self) -> None:
            raise RuntimeError("delete denied")

    backend = AzureVmBackend(client=SimpleNamespace(begin_delete=lambda **kwargs: DeletePoller()))
    with pytest.raises(RuntimeError, match="delete denied"):
        backend.cancel("azure-vm:eastus:rg:vm")


def test_supported_sdk_serialization_offline() -> None:
    appcontainers = pytest.importorskip("azure.mgmt.appcontainers")
    batch_models = pytest.importorskip("azure.batch.models")

    class OfflineCredential:
        def get_token(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("SDK inspection must not request a token or send HTTP")

    client = appcontainers.ContainerAppsAPIClient(OfflineCredential(), "offline")
    body = client.jobs._serialize.body(
        {
            "containers": [
                {
                    "name": "runner",
                    "image": "test",
                    "env": [
                        {"name": "KEY", "secretRef": "secret"},
                    ],
                }
            ]
        },
        "JobExecutionTemplate",
    )
    # The supported serializer accepts REST camelCase dictionary keys too.
    assert body["containers"][0]["env"][0]["secretRef"] == "secret"
    task = batch_models.CloudTask(state=batch_models.TaskState.active)
    backend = AzureBatchBackend(client=SimpleNamespace(task=SimpleNamespace(get=lambda **kwargs: task)))
    assert backend.poll("azure-batch:eastus:job:task").state == AttemptState.PENDING


def test_container_apps_cancel_propagates_stop_failure() -> None:
    from logic.controllers.cloud.providers.azure import ContainerAppsJobsBackend

    class StopPoller:
        def result(self) -> None:
            raise RuntimeError("stop denied")

    backend = ContainerAppsJobsBackend(client=SimpleNamespace(begin_stop_execution=lambda **kwargs: StopPoller()))
    with pytest.raises(RuntimeError, match="stop denied"):
        backend.cancel("azure-aca:eastus:rg:job:execution")


def test_create_lro_failure_waits_for_nic_cleanup() -> None:
    from logic.controllers.cloud.interfaces import CapacityError

    cleaned = []

    class CreatePoller:
        def result(self) -> None:
            raise RuntimeError("QuotaExceeded")

    class DeletePoller:
        def result(self) -> None:
            cleaned.append(True)

    network = SimpleNamespace(
        begin_create_or_update=lambda **kwargs: SimpleNamespace(result=lambda: {"id": "/nic"}),
        begin_delete=lambda **kwargs: DeletePoller(),
    )
    backend = AzureVmBackend(
        client=SimpleNamespace(begin_create_or_update=lambda **kwargs: CreatePoller()),
        network_client=network,
    )
    launch = LaunchSpec(
        task=TaskRecord(task_id="task", job_id="job", cell=CellSpec(cell_id="cell", argv=["run"])),
        argv=["run"],
        env={},
        resources=Resources(),
    )
    with pytest.raises(CapacityError, match="QuotaExceeded"):
        backend.submit(launch)
    assert cleaned == [True]


def test_failed_diagnostics_does_not_return_signed_url(monkeypatch: Any) -> None:
    signed_url = "https://logs.example.invalid/serial?sig=synthetic"
    backend = AzureVmBackend(
        client=SimpleNamespace(
            retrieve_boot_diagnostics_data=lambda **kwargs: SimpleNamespace(serial_console_log_blob_uri=signed_url),
        )
    )
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: SimpleNamespace(status_code=403, text="denied"))
    assert backend.logs("azure-vm:eastus:rg:vm") == ""


def test_create_lro_failure_tears_down_vm_before_nic(tmp_path: Any) -> None:
    from logic.controllers.cloud.interfaces import CapacityError

    order: list[str] = []

    class CreatePoller:
        def result(self) -> None:
            raise RuntimeError("AllocationFailed: Not enough compute capacity")

    class VmDeletePoller:
        def result(self) -> None:
            order.append("vm_delete")

    class NicDeletePoller:
        def result(self) -> None:
            order.append("nic_delete")

    client = SimpleNamespace(
        begin_create_or_update=lambda **kwargs: CreatePoller(),
        begin_delete=lambda **kwargs: VmDeletePoller(),
    )
    network = SimpleNamespace(
        begin_create_or_update=lambda **kwargs: SimpleNamespace(result=lambda: {"id": "/nic"}),
        begin_delete=lambda **kwargs: NicDeletePoller(),
    )
    backend = AzureVmBackend(
        client=client,
        network_client=network,
        orphan_file=tmp_path / "orphans.json",
    )
    launch = LaunchSpec(
        task=TaskRecord(task_id="task-order", job_id="job", cell=CellSpec(cell_id="cell", argv=["run"])),
        argv=["run"],
        env={},
        resources=Resources(),
    )
    with pytest.raises(CapacityError, match="AllocationFailed"):
        backend.submit(launch)

    assert order == ["vm_delete", "nic_delete"]


def test_submit_failure_records_orphans_and_drain_resolves(tmp_path: Any) -> None:
    from logic.controllers.cloud.interfaces import CapacityError

    orphan_file = tmp_path / "orphans.json"

    class CreatePoller:
        def result(self) -> None:
            raise RuntimeError("Capacity: regional compute quota exceeded")

    delete_calls: list[str] = []

    # First attempt: both VM delete and NIC delete fail with 500 error
    def failing_vm_delete(**kwargs: Any) -> Any:
        delete_calls.append("fail_vm")
        raise RuntimeError("ARM 500 InternalServerError")

    def failing_nic_delete(**kwargs: Any) -> Any:
        delete_calls.append("fail_nic")
        raise RuntimeError("Network 503 ServiceUnavailable")

    client = SimpleNamespace(
        begin_create_or_update=lambda **kwargs: CreatePoller(),
        begin_delete=failing_vm_delete,
    )
    network = SimpleNamespace(
        begin_create_or_update=lambda **kwargs: SimpleNamespace(result=lambda: {"id": "/nic"}),
        begin_delete=failing_nic_delete,
    )
    backend = AzureVmBackend(client=client, network_client=network, orphan_file=orphan_file)
    launch = LaunchSpec(
        task=TaskRecord(task_id="task-orphan", job_id="job", cell=CellSpec(cell_id="cell", argv=["run"])),
        argv=["run"],
        env={},
        resources=Resources(),
    )

    with pytest.raises(CapacityError, match="quota exceeded"):
        backend.submit(launch)

    # Both orphans must be recorded in the persistent file
    assert orphan_file.exists()
    orphans = backend._load_orphans()
    kinds = [o["kind"] for o in orphans]
    assert "vm" in kinds and "nic" in kinds
    assert any("task-orphan" in o["name"] for o in orphans)

    # Now fix deletion APIs and verify drain_orphans clears the queue
    drained_calls: list[str] = []

    class SuccessPoller:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def result(self) -> None:
            drained_calls.append(self.tag)

    client.begin_delete = lambda **kwargs: SuccessPoller("drained_vm")
    network.begin_delete = lambda **kwargs: SuccessPoller("drained_nic")

    cleaned = backend.drain_orphans()
    assert len(cleaned) == 2
    # VMs must be drained before NICs
    assert drained_calls == ["drained_vm", "drained_nic"]
    assert backend._load_orphans() == []


def test_idempotent_not_found_handling(tmp_path: Any) -> None:
    orphan_file = tmp_path / "orphans.json"

    class ResourceNotFoundError(Exception):
        status_code = 404
        error_code = "ResourceNotFound"

    class CreatePoller:
        def result(self) -> None:
            raise RuntimeError("Capacity unavailable")

    # Deletions return 404 (already deleted or never provisioned)
    def not_found_vm_delete(**kwargs: Any) -> Any:
        raise ResourceNotFoundError("ResourceNotFound: The Resource 'wsr-vm' was not found.")

    def not_found_nic_delete(**kwargs: Any) -> Any:
        raise ResourceNotFoundError("NotFound: NetworkInterface not found")

    client = SimpleNamespace(
        begin_create_or_update=lambda **kwargs: CreatePoller(),
        begin_delete=not_found_vm_delete,
    )
    network = SimpleNamespace(
        begin_create_or_update=lambda **kwargs: SimpleNamespace(result=lambda: {"id": "/nic"}),
        begin_delete=not_found_nic_delete,
    )
    backend = AzureVmBackend(client=client, network_client=network, orphan_file=orphan_file)
    launch = LaunchSpec(
        task=TaskRecord(task_id="task-404", job_id="job", cell=CellSpec(cell_id="cell", argv=["run"])),
        argv=["run"],
        env={},
        resources=Resources(),
    )

    with pytest.raises(RuntimeError, match="Capacity unavailable"):
        backend.submit(launch)

    # 404 not found is successful cleanup; no orphan should be recorded
    assert backend._load_orphans() == []


def test_vm_params_has_boot_diagnostics_enabled() -> None:
    recorded_params = {}

    class CreatePoller:
        def result(self) -> dict[str, Any]:
            return {"name": "vm"}

    client = SimpleNamespace(
        begin_create_or_update=lambda **kwargs: (
            recorded_params.update(kwargs.get("parameters", {})),
            CreatePoller(),
        )[1]
    )
    network = SimpleNamespace(begin_create_or_update=lambda **kwargs: SimpleNamespace(result=lambda: {"id": "/nic"}))
    backend = AzureVmBackend(client=client, network_client=network)
    launch = LaunchSpec(
        task=TaskRecord(task_id="task-diag", job_id="job", cell=CellSpec(cell_id="cell", argv=["run"])),
        argv=["run"],
        env={},
        resources=Resources(),
    )
    backend.submit(launch)

    assert recorded_params.get("diagnostics_profile") == {"boot_diagnostics": {"enabled": True}}


def test_container_apps_logs_via_azure_monitor_query() -> None:
    from logic.controllers.cloud.providers.azure import ContainerAppsJobsBackend

    queries: list[str] = []

    class FakeLogsQueryClient:
        def query_workspace(self, workspace_id: str, query: str, *, timespan: Any) -> Any:
            queries.append(query)
            return SimpleNamespace(
                tables=[
                    SimpleNamespace(
                        rows=[
                            ["[INFO] Starting WSmart-Route runner"],
                            ["[INFO] Optimization epoch 1 completed"],
                            ["[INFO] Job finished successfully"],
                        ]
                    )
                ]
            )

    backend = ContainerAppsJobsBackend(
        client=SimpleNamespace(),
        logs_client=FakeLogsQueryClient(),
        workspace_id="ws-uuid-1234",
    )
    logs = backend.logs("azure-aca:eastus:wsr-rg:my-job:my-exec", tail=2)
    assert len(queries) == 1
    assert "ContainerAppConsoleLogs_CL" in queries[0]
    assert 'ContainerJobName_s == "my-job"' in queries[0]
    assert 'ExecutionName_s == "my-exec"' in queries[0]
    assert logs == "[INFO] Optimization epoch 1 completed\n[INFO] Job finished successfully"


def test_container_apps_logs_fallback_and_empty() -> None:
    from logic.controllers.cloud.providers.azure import ContainerAppsJobsBackend

    # Fallback to get_log_stream when no logs_client/workspace_id is provided
    class FakeJobsWithStream:
        def get_log_stream(self, **kwargs: Any) -> Any:
            return b"stream line 1\nstream line 2\n"

    backend_stream = ContainerAppsJobsBackend(client=SimpleNamespace(jobs=FakeJobsWithStream()))
    assert backend_stream.logs("azure-aca:eastus:wsr-rg:job:exec") == "stream line 1\nstream line 2"

    # Empty string when neither is available
    backend_empty = ContainerAppsJobsBackend(client=SimpleNamespace())
    assert backend_empty.logs("azure-aca:eastus:wsr-rg:job:exec") == ""

"""Azure provider implementation passes all cloud runner contract suites."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

import pytest
from logic.controllers.cloud.interfaces import (
    AttemptState,
    CapacityError,
    LaunchSpec,
)
from logic.controllers.cloud.model import (
    CellSpec,
    ComputeKind,
    Provider,
    Resources,
    TaskRecord,
)
from logic.controllers.cloud.providers.azure import (
    AcrImageRegistry,
    AzureBatchBackend,
    AzureVmBackend,
    BlobObjectStore,
    ContainerAppsJobsBackend,
    KeyVaultSecretStore,
    price_per_hour,
)
from logic.controllers.cloud.testing import (
    backend_contract,
    object_store_contract,
    secret_store_contract,
)

# ---------------------------------------------------------------------------
# In-process Fakes for Azure
# ---------------------------------------------------------------------------


class FakeBlobDownloader:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def readall(self) -> bytes:
        return self.data


class FakeBlobItem:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeBlobClient:
    def __init__(self, store: Dict[str, bytes], key: str) -> None:
        self.store = store
        self.key = key

    def upload_blob(self, data: Any, overwrite: bool = True) -> None:
        if hasattr(data, "read"):
            self.store[self.key] = data.read()
        else:
            self.store[self.key] = bytes(data)

    def download_blob(self) -> FakeBlobDownloader:
        if self.key not in self.store:
            raise KeyError(self.key)
        return FakeBlobDownloader(self.store[self.key])

    def exists(self) -> bool:
        return self.key in self.store

    def get_blob_properties(self) -> Dict[str, Any]:
        if self.key not in self.store:
            raise KeyError(self.key)
        return {"size": len(self.store[self.key])}

    def delete_blob(self) -> None:
        self.store.pop(self.key, None)


class FakeContainerClient:
    def __init__(self) -> None:
        self.store: Dict[str, bytes] = {}

    def get_blob_client(self, blob_name: str) -> FakeBlobClient:
        return FakeBlobClient(self.store, blob_name)

    def list_blobs(self, name_starts_with: str = "") -> List[FakeBlobItem]:
        return [FakeBlobItem(k) for k in sorted(self.store.keys()) if k.startswith(name_starts_with)]


class FakeSecretItem:
    def __init__(self, value: str) -> None:
        self.value = value


class FakeKeyVaultClient:
    def __init__(self) -> None:
        self.secrets: Dict[str, str] = {}

    def get_secret(self, name: str) -> FakeSecretItem:
        if name not in self.secrets:
            raise KeyError(f"SecretNotFound: {name}")
        return FakeSecretItem(self.secrets[name])

    def set_secret(self, name: str, value: str) -> None:
        self.secrets[name] = value

    def begin_delete_secret(self, name: str) -> None:
        self.secrets.pop(name, None)


class FakeAzureBatchClient:
    def __init__(self) -> None:
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.task = self
        self.file = self

    def add(self, job_id: str, task: Dict[str, Any]) -> None:
        task_id = task["id"]
        cmd = task.get("command_line", "")
        outcome = "succeed"
        if "fail" in task_id or "fail" in cmd:
            outcome = "fail"
        elif "hang" in task_id or "hang" in cmd:
            outcome = "hang"
        elif "preempt" in task_id or "preempt" in cmd:
            outcome = "preempt"

        self.tasks[f"{job_id}:{task_id}"] = {
            "task_id": task_id,
            "outcome": outcome,
            "cancelled": False,
            "polls": 0,
            "_log_lines": [f"azure batch log for {task_id}"],
            "task_data": task,
        }

    def get(self, job_id: str, task_id: str) -> Optional[Dict[str, Any]]:
        key = f"{job_id}:{task_id}"
        if key not in self.tasks:
            return None
        t = self.tasks[key]
        t["polls"] += 1
        if t["cancelled"]:
            state = "completed"
            exec_info = {
                "exit_code": 143,
                "failure_info": "Task was terminated by user request",
                "preempted": False,
            }
        elif t["outcome"] == "hang":
            state = "running"
            exec_info = None
        elif t["outcome"] == "preempt":
            state = "completed"
            exec_info = {
                "exit_code": 137,
                "failure_info": "LowPriority node was preempted and reclaimed",
                "preempted": True,
            }
        elif t["outcome"] == "fail":
            state = "completed"
            exec_info = {
                "exit_code": 1,
                "failure_info": "Process exited with code 1",
                "preempted": False,
            }
        else:
            state = "completed"
            exec_info = {
                "exit_code": 0,
                "failure_info": None,
                "preempted": False,
            }

        return {
            "id": task_id,
            "state": state,
            "execution_info": exec_info,
            "_log_lines": t["_log_lines"],
        }

    def terminate(self, job_id: str, task_id: str) -> None:
        key = f"{job_id}:{task_id}"
        if key in self.tasks:
            self.tasks[key]["cancelled"] = True

    def get_from_task(self, job_id: str, task_id: str, file_path: str) -> io.BytesIO:
        key = f"{job_id}:{task_id}"
        if key in self.tasks:
            lines = self.tasks[key]["_log_lines"]
            return io.BytesIO("\n".join(lines).encode("utf-8"))
        return io.BytesIO(b"")


class FakeContainerAppsClient:
    def __init__(self) -> None:
        self.executions: Dict[str, Dict[str, Any]] = {}
        self.jobs = self
        self.job_executions = self

    def start(self, resource_group_name: str, job_name: str, template: Dict[str, Any]) -> Dict[str, Any]:
        container = template.get("containers", [{}])[0]
        cmd = " ".join(container.get("command", []))
        exec_name = f"exec-{len(self.executions) + 1}"

        outcome = "succeed"
        if "fail" in cmd:
            outcome = "fail"
        elif "hang" in cmd:
            outcome = "hang"
        elif "preempt" in cmd:
            outcome = "preempt"

        key = f"{resource_group_name}:{job_name}:{exec_name}"
        self.executions[key] = {
            "name": exec_name,
            "outcome": outcome,
            "cancelled": False,
            "polls": 0,
            "_log_lines": [f"container apps log for {exec_name}"],
        }
        return {"name": exec_name}

    def get(self, resource_group_name: str, job_name: str, job_execution_name: str) -> Optional[Dict[str, Any]]:
        key = f"{resource_group_name}:{job_name}:{job_execution_name}"
        if key not in self.executions:
            return None
        e = self.executions[key]
        e["polls"] += 1
        if e["cancelled"]:
            status = "Failed"
            msg = "Cancelled by user"
        elif e["outcome"] == "hang":
            status = "Running"
            msg = ""
        elif e["outcome"] == "preempt":
            status = "Failed"
            msg = "Container app job spot instance was preempted"
        elif e["outcome"] == "fail":
            status = "Failed"
            msg = "Container failed with exit code 1"
        else:
            status = "Succeeded"
            msg = ""

        return {
            "name": job_execution_name,
            "status": status,
            "message": msg,
            "_log_lines": e["_log_lines"],
        }

    def stop(self, resource_group_name: str, job_name: str, job_execution_name: str) -> None:
        key = f"{resource_group_name}:{job_name}:{job_execution_name}"
        if key in self.executions:
            self.executions[key]["cancelled"] = True


class FakeAzureVmClient:
    def __init__(self) -> None:
        self.vms: Dict[str, Dict[str, Any]] = {}
        self.virtual_machines = self

    def create_or_update(
        self,
        resource_group_name: str,
        vm_name: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        tags = parameters.get("tags", {})
        task_id = tags.get("wsr:task_id", "")
        custom_data_b64 = parameters.get("os_profile", {}).get("custom_data", "")
        try:
            custom_data = base64.b64decode(custom_data_b64).decode("utf-8")
        except Exception:
            custom_data = ""

        outcome = "succeed"
        if "fail" in task_id or "fail" in custom_data:
            outcome = "fail"
        elif "hang" in task_id or "hang" in custom_data:
            outcome = "hang"
        elif "preempt" in task_id or "preempt" in custom_data:
            outcome = "preempt"

        key = f"{resource_group_name}:{vm_name}"
        self.vms[key] = {
            "name": vm_name,
            "outcome": outcome,
            "cancelled": False,
            "polls": 0,
            "_log_lines": [f"azure vm console output for {vm_name}"],
        }
        return {"name": vm_name}

    def instance_view(self, resource_group_name: str, vm_name: str) -> Optional[Dict[str, Any]]:
        key = f"{resource_group_name}:{vm_name}"
        if key not in self.vms:
            return None
        v = self.vms[key]
        v["polls"] += 1
        if v["cancelled"]:
            status_code = "PowerState/stopped"
            status_tag = "failed"
            exit_code = "143"
        elif v["outcome"] == "hang":
            status_code = "PowerState/running"
            status_tag = "running"
            exit_code = ""
        elif v["outcome"] == "preempt":
            status_code = "ProvisioningState/failed/SpotEviction"
            status_tag = "preempted"
            exit_code = "137"
        elif v["outcome"] == "fail":
            status_code = "PowerState/stopped"
            status_tag = "failed"
            exit_code = "1"
        else:
            status_code = "PowerState/stopped"
            status_tag = "succeeded"
            exit_code = "0"

        return {
            "statuses": [{"code": status_code}],
            "tags": {
                "wsr:status": status_tag,
                "wsr:exit_code": exit_code,
            },
            "_log_lines": v["_log_lines"],
        }

    def delete(self, resource_group_name: str, vm_name: str) -> None:
        key = f"{resource_group_name}:{vm_name}"
        if key in self.vms:
            self.vms[key]["cancelled"] = True

    def get(self, resource_group_name: str, vm_name: str) -> Optional[Dict[str, Any]]:
        return self.instance_view(resource_group_name, vm_name)


class FakeAcrClient:
    def __init__(self) -> None:
        self.tags: Dict[str, str] = {}

    def upload_manifest(self, manifest: bytes, tag: str) -> None:
        self.tags[tag] = f"sha256:{tag}-acr-digest"

    def get_manifest_properties(self, repository_name: str, tag_or_digest: str) -> Any:
        if tag_or_digest in self.tags:
            class Props:
                digest = self.tags[tag_or_digest]
            return Props()
        raise KeyError(tag_or_digest)


# ---------------------------------------------------------------------------
# Launch helper
# ---------------------------------------------------------------------------


def _launch(outcome: str) -> LaunchSpec:
    task = TaskRecord(
        task_id=f"task-{outcome}",
        job_id="job-azure",
        cell=CellSpec(cell_id=outcome, argv=["test_sim", f"sim.outcome={outcome}"]),
    )
    return LaunchSpec(
        task=task,
        resources=Resources(cpus=1.0, memory_gb=2.0),
        argv=["python", "main.py", "test_sim", f"outcome={outcome}"],
        env={"TEST_VAR": "val"},
        secret_env={"GUROBI_KEY": "https://vault.azure.net/secrets/gurobi-secret-id"},
    )


# ---------------------------------------------------------------------------
# Contract Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_azure_blob_object_store_contract():
    container_client = FakeContainerClient()
    object_store_contract(lambda: BlobObjectStore(container="wsr-test", container_client=container_client))


@pytest.mark.unit
def test_azure_key_vault_secret_store_contract():
    client = FakeKeyVaultClient()
    secret_store_contract(lambda: KeyVaultSecretStore(client=client))


@pytest.mark.unit
def test_azure_batch_backend_contract():
    client = FakeAzureBatchClient()
    backend_contract(
        make=lambda: AzureBatchBackend(client=client),
        launch=_launch,
        poll_seconds=0.0,
    )


@pytest.mark.unit
def test_azure_batch_spot_preemption_and_capacity():
    client = FakeAzureBatchClient()
    backend = AzureBatchBackend(client=client)

    # Spot preemption
    handle = backend.submit(_launch("preempt"))
    status = backend.poll(handle)
    assert status.state == AttemptState.PREEMPTED
    assert "preempt" in (status.message or "").lower()

    # Capacity limit check
    excessive_launch = _launch("succeed")
    excessive_launch.resources = Resources(cpus=256.0, memory_gb=1024.0)
    with pytest.raises(CapacityError):
        backend.submit(excessive_launch)


@pytest.mark.unit
def test_azure_container_apps_jobs_contract():
    client = FakeContainerAppsClient()
    backend_contract(
        make=lambda: ContainerAppsJobsBackend(client=client),
        launch=_launch,
        poll_seconds=0.0,
    )


@pytest.mark.unit
def test_azure_container_apps_spot_and_capacity():
    client = FakeContainerAppsClient()
    backend = ContainerAppsJobsBackend(client=client)

    # Preemption check
    handle = backend.submit(_launch("preempt"))
    status = backend.poll(handle)
    assert status.state == AttemptState.PREEMPTED

    # Capacity limit check
    excessive_launch = _launch("succeed")
    excessive_launch.resources = Resources(cpus=32.0, memory_gb=128.0)
    with pytest.raises(CapacityError):
        backend.submit(excessive_launch)


@pytest.mark.unit
def test_azure_vm_backend_contract():
    client = FakeAzureVmClient()
    backend_contract(
        make=lambda: AzureVmBackend(client=client),
        launch=_launch,
        poll_seconds=0.0,
    )


@pytest.mark.unit
def test_azure_vm_spot_preemption():
    client = FakeAzureVmClient()
    backend = AzureVmBackend(client=client)
    handle = backend.submit(_launch("preempt"))
    status = backend.poll(handle)
    assert status.state == AttemptState.PREEMPTED


@pytest.mark.unit
def test_azure_acr_image_registry():
    client = FakeAcrClient()
    registry = AcrImageRegistry(client=client, registry_name="wsrregistry")
    assert registry.resolve("v1.1.0") is None
    digest = registry.push("local:v1.1.0", "v1.1.0")
    assert digest.startswith("sha256:")
    assert registry.resolve("v1.1.0") is not None


@pytest.mark.unit
def test_azure_pricing():
    assert price_per_hour("standard_d2s_v5", "eastus", spot=False) == 0.096
    assert price_per_hour("standard_d2s_v5", "eastus", spot=True) == 0.0192
    assert price_per_hour("container_apps", "eastus", spot=False) == 0.036
    assert price_per_hour("unknown-size") is None


@pytest.mark.unit
def test_azure_load_backend():
    from logic.controllers.cloud.providers import load_backend

    batch_be = load_backend(Provider.AZURE, ComputeKind.BATCH, client=FakeAzureBatchClient())
    assert isinstance(batch_be, AzureBatchBackend)
    serverless_be = load_backend(Provider.AZURE, ComputeKind.SERVERLESS, client=FakeContainerAppsClient())
    assert isinstance(serverless_be, ContainerAppsJobsBackend)
    vm_be = load_backend(Provider.AZURE, ComputeKind.VM, client=FakeAzureVmClient())
    assert isinstance(vm_be, AzureVmBackend)

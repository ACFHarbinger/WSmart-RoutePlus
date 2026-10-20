"""Azure provider implementation passes all cloud runner contract suites."""

from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

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
from logic.controllers.cloud.providers.azure.storage import store_from_uri
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
    """In-memory Azure Batch client for contract testing.

    Rejects Key Vault references passed as plain environment settings.
    """

    def __init__(self) -> None:
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.task = self
        self.file = self

    def add(self, job_id: str, task: Dict[str, Any]) -> None:
        task_id = task["id"]
        cmd = task.get("command_line", "")

        # Strict validation: Azure Batch does not resolve Key Vault references in plain settings
        for env_setting in task.get("environment_settings", []):
            val = str(env_setting.get("value", ""))
            if "@Microsoft.KeyVault" in val or ("akv://" in val and env_setting.get("name") != "WSR_SECRET_REFS"):
                raise ValueError(
                    "ParamValidationError: Azure Batch does not resolve Key Vault references in plain settings"
                )

        # Validate private ACR images require container_settings.registry with identity_reference
        c_settings = task.get("container_settings", {})
        img = c_settings.get("image_name", "")
        if ".azurecr.io" in img:
            reg = c_settings.get("registry")
            if not reg or not reg.get("registry_server"):
                raise ValueError("ParamValidationError: Azure Batch container_settings missing registry for ACR image")
            if not reg.get("identity_reference", {}).get("resource_id"):
                raise ValueError(
                    "ParamValidationError: Azure Batch container_settings registry missing identity_reference"
                )

        outcome = "succeed"
        if "fail" in task_id or "fail" in cmd:
            outcome = "fail"
        elif "hang" in task_id or "hang" in cmd:
            outcome = "hang"
        elif "preempt" in task_id or "preempt" in cmd:
            outcome = "preempt"
        elif "exit143" in task_id or "exit143" in cmd:
            outcome = "exit143"

        self.tasks[f"{job_id}:{task_id}"] = {
            "task_id": task_id,
            "outcome": outcome,
            "cancelled": False,
            "polls": 0,
            "log_lines": [f"azure batch log for {task_id}"],
            "task_data": task,
            "pool_id": task.get("pool_id", ""),
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
        elif t["outcome"] == "exit143":
            state = "completed"
            exec_info = {
                "exit_code": 143,
                "failure_info": "Container received SIGTERM and exited 143",
                "preempted": False,
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
            "pool_id": t["pool_id"],
        }

    def terminate(self, job_id: str, task_id: str) -> None:
        key = f"{job_id}:{task_id}"
        if key in self.tasks:
            self.tasks[key]["cancelled"] = True

    def get_from_task(self, job_id: str, task_id: str, file_path: str) -> io.BytesIO:
        key = f"{job_id}:{task_id}"
        if key in self.tasks:
            lines = self.tasks[key]["log_lines"]
            return io.BytesIO("\n".join(lines).encode("utf-8"))
        return io.BytesIO(b"")


class FakeContainerAppsClient:
    """In-memory Azure Container Apps client for contract testing.

    Enforces that referenced secrets are declared in job configuration.secrets.
    """

    def __init__(self) -> None:
        self.executions: Dict[str, Dict[str, Any]] = {}
        self.jobs = self
        self.job_executions = self
        self.job_definitions: Dict[str, Dict[str, Any]] = {
            "wsr-rg:wsr-job": {"configuration": {"secrets": [{"name": "GUROBI_KEY"}, {"name": "gurobi-secret"}]}}
        }

    def get(
        self, resource_group_name: str, job_name: str, job_execution_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        if job_execution_name is None:
            return self.job_definitions.get(f"{resource_group_name}:{job_name}")

        key = f"{resource_group_name}:{job_name}:{job_execution_name}"
        if key not in self.executions:
            return None
        e = self.executions[key]
        e["polls"] += 1
        if e["cancelled"]:
            status = "Failed"
            msg = "Cancelled by user"
            exit_code = 143
        elif e["outcome"] == "hang":
            status = "Running"
            msg = ""
            exit_code = None
        elif e["outcome"] == "preempt":
            status = "Failed"
            msg = "Container app job spot instance was preempted"
            exit_code = 137
        elif e["outcome"] == "exit143":
            status = "Failed"
            msg = "Container app spot node interrupted (exit 143)"
            exit_code = 143
        elif e["outcome"] == "fail":
            status = "Failed"
            msg = "Container failed with exit code 1"
            exit_code = 1
        else:
            status = "Succeeded"
            msg = ""
            exit_code = 0

        return {
            "name": job_execution_name,
            "status": status,
            "message": msg,
            "exit_code": exit_code,
            "spot": e.get("spot", False),
        }

    def start(self, resource_group_name: str, job_name: str, template: Dict[str, Any]) -> Dict[str, Any]:
        container = template.get("containers", [{}])[0]
        cmd = " ".join(container.get("command", []))
        exec_name = f"exec-{len(self.executions) + 1}"

        # Verify referenced secrets exist in configuration.secrets
        job_def = self.job_definitions.get(f"{resource_group_name}:{job_name}", {})
        known_secrets = {s["name"] for s in job_def.get("configuration", {}).get("secrets", [])}
        for env_entry in container.get("env", []):
            if "secretRef" in env_entry:
                ref = env_entry["secretRef"]
                if ref not in known_secrets:
                    raise ValueError(f"SecretRef '{ref}' not found in job configuration.secrets")

        outcome = "succeed"
        if "fail" in cmd:
            outcome = "fail"
        elif "hang" in cmd:
            outcome = "hang"
        elif "preempt" in cmd:
            outcome = "preempt"
        elif "exit143" in cmd:
            outcome = "exit143"

        key = f"{resource_group_name}:{job_name}:{exec_name}"
        self.executions[key] = {
            "name": exec_name,
            "outcome": outcome,
            "cancelled": False,
            "polls": 0,
            "log_lines": [f"container apps log for {exec_name}"],
            "spot": bool("preempt" in outcome or "exit143" in outcome),
        }
        return {"name": exec_name}

    def get_log_stream(self, resource_group_name: str, job_name: str, job_execution_name: str) -> io.BytesIO:
        key = f"{resource_group_name}:{job_name}:{job_execution_name}"
        lines = self.executions.get(key, {}).get("log_lines", [])
        return io.BytesIO("\n".join(lines).encode("utf-8"))

    def stop(self, resource_group_name: str, job_name: str, job_execution_name: str) -> None:
        key = f"{resource_group_name}:{job_name}:{job_execution_name}"
        if key in self.executions:
            self.executions[key]["cancelled"] = True


class FakeNetworkInterfacesClient:
    """In-memory Azure Network Interfaces client for VM testing."""

    def __init__(self) -> None:
        self.nics: Dict[str, Dict[str, Any]] = {}

    def begin_create_or_update(
        self,
        resource_group_name: str,
        network_interface_name: str,
        parameters: Dict[str, Any],
    ) -> Any:
        ip_configs = parameters.get("ip_configurations", [])
        if not ip_configs or not ip_configs[0].get("subnet", {}).get("id"):
            raise ValueError("NIC requires ip_configurations with valid subnet id")

        nic_id = f"/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/{resource_group_name}/providers/Microsoft.Network/networkInterfaces/{network_interface_name}"
        self.nics[f"{resource_group_name}:{network_interface_name}"] = {
            "id": nic_id,
            "parameters": parameters,
        }

        class Poller:
            def result(self) -> Dict[str, Any]:
                return {"id": nic_id, "name": network_interface_name}

        return Poller()

    create_or_update = begin_create_or_update


def _validate_azure_vm_parameters(parameters: Dict[str, Any]) -> None:
    """Strict Azure VM API parameter validation per real Azure Compute API."""
    storage_prof = parameters.get("storage_profile")
    if not storage_prof:
        raise ValueError("Invalid VM create request: missing storage_profile")
    img_ref = storage_prof.get("image_reference")
    if not img_ref or not img_ref.get("publisher") or not img_ref.get("offer"):
        raise ValueError("Invalid VM create request: storage_profile.image_reference is missing publisher/offer")
    os_disk = storage_prof.get("os_disk")
    if not os_disk or os_disk.get("delete_option") != "Delete":
        raise ValueError("Invalid VM create request: storage_profile.os_disk must have delete_option='Delete'")

    os_prof = parameters.get("os_profile")
    if not os_prof or not os_prof.get("admin_username"):
        raise ValueError("Invalid VM create request: os_profile.admin_username is required")
    linux_cfg = os_prof.get("linux_configuration")
    if not linux_cfg or linux_cfg.get("disable_password_authentication") is not True:
        raise ValueError("Invalid VM create request: password authentication must be disabled")
    ssh_cfg = linux_cfg.get("ssh", {})
    pub_keys = ssh_cfg.get("public_keys", [])
    if not pub_keys or not pub_keys[0].get("key_data"):
        raise ValueError("Invalid VM create request: at least one SSH public key is required")

    net_prof = parameters.get("network_profile")
    if not net_prof or not net_prof.get("network_interfaces"):
        raise ValueError("Invalid VM create request: missing network_profile.network_interfaces")
    nics = net_prof["network_interfaces"]
    if not nics:
        raise ValueError("Invalid VM create request: at least one NIC required")
    nic_props = nics[0].get("properties", nics[0])
    if nic_props.get("delete_option") != "Delete":
        raise ValueError("Invalid VM create request: network_profile NIC must specify delete_option='Delete'")

    identity = parameters.get("identity")
    if not identity or identity.get("type") != "UserAssigned" or not identity.get("user_assigned_identities"):
        raise ValueError("Invalid VM create request: user-assigned managed identity is required")


class FakeAzureVmClient:
    """In-memory Azure Virtual Machines client for contract testing."""

    def __init__(self) -> None:
        self.vms: Dict[str, Dict[str, Any]] = {}
        self.virtual_machines = self
        self.network_interfaces = FakeNetworkInterfacesClient()

    def create_or_update(
        self,
        resource_group_name: str,
        vm_name: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        _validate_azure_vm_parameters(parameters)

        tags = parameters.get("tags", {})
        task_id = tags.get("wsr:task_id", "")
        custom_data_b64 = parameters.get("os_profile", {}).get("custom_data", "")
        try:
            custom_data = base64.b64decode(custom_data_b64).decode("utf-8")
        except Exception:
            custom_data = ""

        outcome = "succeed"
        if "fail" in task_id or "test_sim_fail" in custom_data:
            outcome = "fail"
        elif "hang" in task_id or "test_sim_hang" in custom_data:
            outcome = "hang"
        elif "preempt" in task_id or "test_sim_preempt" in custom_data:
            outcome = "preempt"
        elif "exit143" in task_id or "test_sim_exit143" in custom_data:
            outcome = "exit143"

        key = f"{resource_group_name}:{vm_name}"
        self.vms[key] = {
            "name": vm_name,
            "outcome": outcome,
            "cancelled": False,
            "deleted": False,
            "polls": 0,
            "tags": dict(tags),
            "custom_data": custom_data,
            "parameters": parameters,
            "priority": parameters.get("priority"),
        }
        return {"name": vm_name}

    begin_create_or_update = create_or_update

    def instance_view(self, resource_group_name: str, vm_name: str) -> Optional[Dict[str, Any]]:
        key = f"{resource_group_name}:{vm_name}"
        if key not in self.vms:
            return None
        v = self.vms[key]
        v["polls"] += 1
        tags = dict(v["tags"])

        if v["cancelled"]:
            status_code = "PowerState/deallocated"
            tags["wsr-exit-code"] = "143"
        elif v["outcome"] == "hang":
            status_code = "PowerState/running"
        elif v["outcome"] == "preempt":
            status_code = "ProvisioningState/failed/SpotEviction"
            tags["wsr-exit-code"] = "137"
        elif v["outcome"] == "exit143":
            status_code = "PowerState/deallocated"
            tags["wsr-exit-code"] = "143"
        elif v["outcome"] == "fail":
            status_code = "PowerState/deallocated"
            tags["wsr-exit-code"] = "1"
        else:
            status_code = "PowerState/deallocated"
            tags["wsr-exit-code"] = "0"

        return {
            "statuses": [{"code": status_code}],
            "tags": tags,
            "priority": v["priority"],
        }

    def get(self, resource_group_name: str, vm_name: str) -> Optional[Dict[str, Any]]:
        return self.instance_view(resource_group_name, vm_name)

    def delete(self, resource_group_name: str, vm_name: str) -> None:
        key = f"{resource_group_name}:{vm_name}"
        if key in self.vms:
            self.vms[key]["deleted"] = True
            if self.vms[key]["outcome"] == "hang":
                self.vms[key]["cancelled"] = True

    begin_delete = delete

    def retrieve_boot_diagnostics_data(self, resource_group_name: str, vm_name: str) -> Any:
        class BootDiag:
            serial_console_log_blob_uri = f"azure vm serial console log for {vm_name}"

        return BootDiag()


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
# Test Helpers
# ---------------------------------------------------------------------------


def make_test_launch(
    task_id: str = "task-001",
    command: str = "test_sim",
    cpus: float = 1.0,
    memory_gb: float = 2.0,
    spot: bool = False,
    secret_env: Dict[str, str] | None = None,
    image: Optional[str] = None,
) -> LaunchSpec:
    return LaunchSpec(
        task=TaskRecord(
            task_id=task_id,
            job_id="job-001",
            cell=CellSpec(cell_id="c1", argv=[command]),
        ),
        argv=[command, "--days", "30"],
        env={"WSR_JOB_ID": "job-001"},
        secret_env=secret_env or {},
        image=image or "wsmart-route:v1.1.0",
        resources=Resources(cpus=cpus, memory_gb=memory_gb, spot=spot),
    )


def make_launch_for_outcome(outcome: str) -> LaunchSpec:
    secret_env = {"GUROBI_KEY": "GUROBI_KEY"} if outcome == "succeed" else {}
    return make_test_launch(
        task_id=f"task-{outcome}",
        command=f"test_sim_{outcome}",
        secret_env=secret_env,
    )


# ---------------------------------------------------------------------------
# Contract Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_azure_blob_object_store_contract():
    container_client = FakeContainerClient()
    object_store_contract(lambda: BlobObjectStore(container="wsr-test", container_client=container_client))


@pytest.mark.unit
def test_azure_blob_store_from_uri():
    store = store_from_uri(urlparse("az://myaccount/mycontainer/runs/prefix/"), container_client=FakeContainerClient())
    assert store.container == "mycontainer"
    assert store.prefix == "runs/prefix/"
    assert store.account_url == "https://myaccount.blob.core.windows.net"


@pytest.mark.unit
def test_azure_key_vault_secret_store_contract():
    client = FakeKeyVaultClient()
    secret_store_contract(lambda: KeyVaultSecretStore(client=client))


@pytest.mark.unit
def test_azure_batch_backend_contract():
    client = FakeAzureBatchClient()
    backend_contract(
        make=lambda: AzureBatchBackend(client=client),
        launch=make_launch_for_outcome,
    )


@pytest.mark.unit
def test_azure_batch_rejects_plain_secret_and_passes_wsr_secret_refs():
    client = FakeAzureBatchClient()
    backend = AzureBatchBackend(client=client, vault_name="my-vault")

    launch = make_test_launch(
        task_id="task-batch-sec",
        secret_env={"WSR_GUROBI_LICENSE": "gurobi-secret"},
    )
    handle = backend.submit(launch)
    assert handle.startswith("azure-batch:eastus:wsr-jobs:task-task-batch-sec")

    task = client.tasks["wsr-jobs:task-task-batch-sec"]["task_data"]
    # Image name should be set in container settings
    assert task["container_settings"]["image_name"] == "wsmart-route:v1.1.0"

    # Secrets must not be passed as plain @Microsoft.KeyVault references
    for s in task["environment_settings"]:
        assert "@Microsoft.KeyVault" not in s["value"]

    # Common secret path: WSR_SECRET_REFS must be present with akv:// URI
    secret_refs_entry = next(s for s in task["environment_settings"] if s["name"] == "WSR_SECRET_REFS")
    secret_refs = json.loads(secret_refs_entry["value"])
    assert secret_refs == {"WSR_GUROBI_LICENSE": "akv://my-vault/gurobi-secret"}

    # ACR image verification on Azure Batch
    acr_launch = make_test_launch(
        task_id="task-batch-acr",
        image="myregistry.azurecr.io/wsmart-route:v1.1.0",
    )
    backend.submit(acr_launch)
    acr_task = client.tasks["wsr-jobs:task-task-batch-acr"]["task_data"]
    c_settings = acr_task["container_settings"]
    assert c_settings["image_name"] == "myregistry.azurecr.io/wsmart-route:v1.1.0"
    assert c_settings["registry"]["registry_server"] == "myregistry.azurecr.io"
    assert c_settings["registry"]["identity_reference"]["resource_id"] == backend.identity_id

    # Fake rejects missing registry or identity_reference for ACR images
    with pytest.raises(ValueError, match="missing registry"):
        client.add(
            "wsr-jobs",
            {
                "id": "bad-task-1",
                "container_settings": {"image_name": "myregistry.azurecr.io/test:v1"},
            },
        )
    with pytest.raises(ValueError, match="missing identity_reference"):
        client.add(
            "wsr-jobs",
            {
                "id": "bad-task-2",
                "container_settings": {
                    "image_name": "myregistry.azurecr.io/test:v1",
                    "registry": {"registry_server": "myregistry.azurecr.io"},
                },
            },
        )


@pytest.mark.unit
def test_azure_batch_spot_preemption_and_exit_143():
    client = FakeAzureBatchClient()
    backend = AzureBatchBackend(client=client)

    # Spot preemption
    handle = backend.submit(make_test_launch(task_id="task-preempt", spot=True))
    status = backend.poll(handle)
    assert status.state == AttemptState.PREEMPTED
    assert "preempt" in (status.message or "").lower()

    # Exit code 143 on spot pool
    handle_143 = backend.submit(make_test_launch(task_id="task-exit143", spot=True))
    status_143 = backend.poll(handle_143)
    assert status_143.state == AttemptState.PREEMPTED

    # Capacity limit check
    excessive_launch = make_test_launch(cpus=256.0, memory_gb=1024.0)
    with pytest.raises(CapacityError):
        backend.submit(excessive_launch)


@pytest.mark.unit
def test_azure_container_apps_jobs_contract():
    client = FakeContainerAppsClient()
    backend_contract(
        make=lambda: ContainerAppsJobsBackend(client=client),
        launch=make_launch_for_outcome,
    )


@pytest.mark.unit
def test_azure_container_apps_rejects_undeclared_secret_and_spot_preemption():
    client = FakeContainerAppsClient()
    backend = ContainerAppsJobsBackend(client=client)

    # Undeclared secret should raise ValueError
    undeclared_launch = make_test_launch(
        task_id="task-undeclared",
        secret_env={"UNKNOWN_KEY": "undeclared-secret-name"},
    )
    with pytest.raises(ValueError, match="not declared"):
        backend.submit(undeclared_launch)

    # Declared secret passes
    declared_launch = make_test_launch(
        task_id="task-declared",
        secret_env={"GUROBI_KEY": "gurobi-secret"},
    )
    handle_ok = backend.submit(declared_launch)
    assert handle_ok.startswith("azure-aca:eastus:wsr-rg:wsr-job:")

    # Spot preemption
    handle_preempt = backend.submit(make_test_launch(command="test_sim_preempt", spot=True))
    assert backend.poll(handle_preempt).state == AttemptState.PREEMPTED

    # Exit code 143 on spot
    handle_143 = backend.submit(make_test_launch(command="test_sim_exit143", spot=True))
    assert backend.poll(handle_143).state == AttemptState.PREEMPTED

    # Capacity limit check
    excessive_launch = make_test_launch(cpus=32.0, memory_gb=128.0)
    with pytest.raises(CapacityError):
        backend.submit(excessive_launch)


@pytest.mark.unit
def test_azure_vm_backend_contract():
    client = FakeAzureVmClient()
    backend_contract(
        make=lambda: AzureVmBackend(client=client),
        launch=make_launch_for_outcome,
    )


@pytest.mark.unit
def test_azure_vm_custom_data_script_and_spot_preemption():
    client = FakeAzureVmClient()
    backend = AzureVmBackend(client=client, vault_name="my-vault")

    launch = make_test_launch(
        task_id="task-vm-run",
        secret_env={"WSR_GUROBI_LICENSE": "gurobi-secret"},
        spot=True,
    )
    handle = backend.submit(launch)
    assert handle.startswith("azure-vm:eastus:wsr-rg:wsr-vm-task-vm-run")

    # Verify custom_data script
    vm_record = client.vms["wsr-rg:wsr-vm-task-vm-run"]
    custom_data = vm_record["custom_data"]
    assert "#!/bin/bash" in custom_data
    assert "trap 'shutdown -h now' EXIT" in custom_data
    assert "docker run --rm" in custom_data
    assert "wsmart-route:v1.1.0" in custom_data
    assert "test_sim" in custom_data
    assert "shutdown -h now" in custom_data
    assert "WSR_SECRET_REFS" in custom_data
    assert "akv://my-vault/gurobi-secret" in custom_data
    assert "wsr-exit-code" in custom_data

    # Verify complete Azure VM creation parameters structure
    params = vm_record["parameters"]
    assert params["storage_profile"]["image_reference"]["publisher"] == "canonical"
    assert params["storage_profile"]["image_reference"]["offer"] == "0001-com-ubuntu-server-jammy"
    assert params["storage_profile"]["image_reference"]["sku"] == "22_04-lts-gen2"
    assert params["storage_profile"]["os_disk"]["delete_option"] == "Delete"
    assert params["os_profile"]["admin_username"] == "wsruser"
    assert params["os_profile"]["linux_configuration"]["disable_password_authentication"] is True
    assert len(params["os_profile"]["linux_configuration"]["ssh"]["public_keys"]) >= 1
    assert params["network_profile"]["network_interfaces"][0]["delete_option"] == "Delete"
    assert params["identity"]["type"] == "UserAssigned"
    assert backend.identity_id in params["identity"]["user_assigned_identities"]

    # Verify logs
    assert "azure vm serial console log" in backend.logs(handle)

    # Test polling states and billing halt (VM deletion on stopped/deallocated)
    assert vm_record["deleted"] is False
    st = backend.poll(handle)
    assert st.state == AttemptState.SUCCEEDED
    assert vm_record["deleted"] is True

    # Test ACR image in VM includes ACR login in custom_data
    acr_launch = make_test_launch(
        task_id="task-vm-acr",
        image="myacr.azurecr.io/wsmart-route:v1.1.0",
    )
    backend.submit(acr_launch)
    acr_record = client.vms["wsr-rg:wsr-vm-task-vm-acr"]
    assert "https://myacr.azurecr.io/oauth2/exchange" in acr_record["custom_data"]
    assert "myacr.azurecr.io" in acr_record["custom_data"]

    # Test spot interruption
    handle_preempt = backend.submit(make_test_launch(task_id="task-preempt", spot=True))
    assert backend.poll(handle_preempt).state == AttemptState.PREEMPTED

    # Test exit code 143 on spot
    handle_143 = backend.submit(make_test_launch(task_id="task-exit143", spot=True))
    assert backend.poll(handle_143).state == AttemptState.PREEMPTED


@pytest.mark.unit
def test_fake_azure_vm_client_rejects_incomplete_parameters():
    client = FakeAzureVmClient()

    valid_params: Dict[str, Any] = {
        "storage_profile": {
            "image_reference": {"publisher": "canonical", "offer": "0001-com-ubuntu-server-jammy"},
            "os_disk": {"delete_option": "Delete"},
        },
        "os_profile": {
            "admin_username": "wsruser",
            "linux_configuration": {
                "disable_password_authentication": True,
                "ssh": {"public_keys": [{"key_data": "ssh-rsa AAAA..."}]},
            },
        },
        "network_profile": {
            "network_interfaces": [{"id": "/sub/nic1", "properties": {"delete_option": "Delete"}}],
        },
        "identity": {
            "type": "UserAssigned",
            "user_assigned_identities": {"/sub/identity1": {}},
        },
    }

    # Missing storage_profile
    p = dict(valid_params)
    del p["storage_profile"]
    with pytest.raises(ValueError, match="missing storage_profile"):
        client.create_or_update("wsr-rg", "test-vm", p)

    # Missing image_reference publisher/offer
    p = dict(valid_params)
    p["storage_profile"] = {"os_disk": {"delete_option": "Delete"}}
    with pytest.raises(ValueError, match="image_reference is missing publisher/offer"):
        client.create_or_update("wsr-rg", "test-vm", p)

    # Missing os_disk delete_option
    p = dict(valid_params)
    p["storage_profile"] = {
        "image_reference": {"publisher": "canonical", "offer": "ubuntu"},
        "os_disk": {"delete_option": "Detach"},
    }
    with pytest.raises(ValueError, match="storage_profile.os_disk must have delete_option='Delete'"):
        client.create_or_update("wsr-rg", "test-vm", p)

    # Missing os_profile
    p = dict(valid_params)
    del p["os_profile"]
    with pytest.raises(ValueError, match="os_profile.admin_username is required"):
        client.create_or_update("wsr-rg", "test-vm", p)

    # Missing password auth disable
    p = dict(valid_params)
    p["os_profile"] = {
        "admin_username": "wsruser",
        "linux_configuration": {"disable_password_authentication": False},
    }
    with pytest.raises(ValueError, match="password authentication must be disabled"):
        client.create_or_update("wsr-rg", "test-vm", p)

    # Missing SSH public keys
    p = dict(valid_params)
    p["os_profile"] = {
        "admin_username": "wsruser",
        "linux_configuration": {
            "disable_password_authentication": True,
            "ssh": {"public_keys": []},
        },
    }
    with pytest.raises(ValueError, match="at least one SSH public key is required"):
        client.create_or_update("wsr-rg", "test-vm", p)

    # Missing network_profile NIC delete_option
    p = dict(valid_params)
    p["network_profile"] = {"network_interfaces": [{"id": "/sub/nic1", "properties": {}}]}
    with pytest.raises(ValueError, match="network_profile NIC must specify delete_option='Delete'"):
        client.create_or_update("wsr-rg", "test-vm", p)

    # Missing identity
    p = dict(valid_params)
    del p["identity"]
    with pytest.raises(ValueError, match="user-assigned managed identity is required"):
        client.create_or_update("wsr-rg", "test-vm", p)


@pytest.mark.unit
def test_fake_network_interfaces_client_subnet_validation():
    nic_client = FakeNetworkInterfacesClient()
    # Missing subnet id raises ValueError
    with pytest.raises(ValueError, match="valid subnet id"):
        nic_client.begin_create_or_update("wsr-rg", "nic-1", {"ip_configurations": [{"name": "ip1"}]})

    # Valid subnet id succeeds
    poller = nic_client.begin_create_or_update(
        "wsr-rg",
        "nic-1",
        {"ip_configurations": [{"name": "ip1", "subnet": {"id": "/subscriptions/sub1/subnets/s1"}}]},
    )
    assert poller.result()["name"] == "nic-1"


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

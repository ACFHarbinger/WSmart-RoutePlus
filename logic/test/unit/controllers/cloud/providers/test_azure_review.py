"""Exercise generated bootstrap with local commands, without cloud access."""

import base64
import json
import os
import subprocess
from pathlib import Path

import pytest
from logic.controllers.cloud.interfaces import AttemptState, LaunchSpec
from logic.controllers.cloud.model import CellSpec, Resources, TaskRecord
from logic.controllers.cloud.providers.azure import AzureVmBackend


@pytest.mark.parametrize("exit_code", [0, 7])
def test_bootstrap_exchanges_acr_token_and_records_actual_exit(tmp_path: Path, exit_code: int) -> None:
    launch = LaunchSpec(
        task=TaskRecord(task_id="task", job_id="job", cell=CellSpec(cell_id="cell", argv=["run"])),
        argv=["run", "argument with spaces"],
        env={},
        secret_env={},
        image="example.azurecr.io/runner:latest",
        resources=Resources(),
    )
    script = base64.b64decode(AzureVmBackend()._build_custom_data(launch)).decode()
    commands = {
        "systemctl": "exit 0",
        "shutdown": 'touch "$TEST_ROOT/shutdown"',
        "docker": """if [ "$1" = login ]; then
    read -r TOKEN || true
    test "$TOKEN" = acr-refresh || exit 90
    touch "$TEST_ROOT/login"
else
    test -f "$TEST_ROOT/login" || exit 91
    exit "$TEST_EXIT"
fi""",
        "curl": """case "$*" in
  *oauth2/token*) echo '{"access_token":"arm-token"}' ;;
  *oauth2/exchange*) echo '{"refresh_token":"acr-refresh"}' ;;
  *resourceId*) echo /subscriptions/test/resourceGroups/test/providers/Microsoft.Compute/virtualMachines/test ;;
  *tags/default*)
    while [ "$#" -gt 0 ]; do
      if [ "$1" = -d ]; then shift; printf '%s' "$1" > "$TEST_ROOT/tag.json"; fi
      shift
    done ;;
  *) exit 92 ;;
esac""",
    }
    for name, body in commands.items():
        command = tmp_path / name
        command.write_text("#!/bin/bash\n" + body + "\n")
        command.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}", TEST_ROOT=str(tmp_path), TEST_EXIT=str(exit_code))
    subprocess.run(["bash"], input=script, text=True, env=env, check=True, capture_output=True)
    assert json.loads((tmp_path / "tag.json").read_text()) == {
        "operation": "Merge",
        "properties": {"tags": {"wsr-exit-code": str(exit_code)}},
    }
    assert (tmp_path / "shutdown").exists()


def test_ssh_key_is_parseable() -> None:
    from cryptography.hazmat.primitives.serialization import load_ssh_public_key

    load_ssh_public_key(AzureVmBackend()._get_or_generate_ssh_key().encode())


def test_batch_credential_adapter_prepares_signed_request() -> None:
    from collections import namedtuple

    import requests
    from logic.controllers.cloud.providers.azure._sdk import (
        MsrestTokenCredentialAdapter,
    )

    AccessToken = namedtuple("AccessToken", ["token", "expires_on"])

    class SyntheticTokenCredential:
        def get_token(self, *scopes, **kwargs):
            return AccessToken(token="synthetic-batch-token-12345", expires_on=9999999999)

    adapter = MsrestTokenCredentialAdapter(SyntheticTokenCredential())
    session = requests.Session()
    adapter.signed_session(session)
    assert session.headers.get("Authorization") == "Bearer synthetic-batch-token-12345"


def test_container_apps_secret_patch_shape_and_await_poller() -> None:
    from logic.controllers.cloud.providers.azure import ContainerAppsJobsBackend

    recorded_patch = {}
    poller_awaited = False

    class FakeUpdatePoller:
        def result(self):
            nonlocal poller_awaited
            poller_awaited = True
            return {"properties": {"configuration": {"secrets": []}}}

    class FakeJobsApi:
        def get(self, resource_group_name, job_name):
            return {"properties": {"configuration": {"secrets": [{"name": "EXISTING_KEY"}]}}}

        def begin_update(self, resource_group_name, job_name, job_envelope):
            nonlocal recorded_patch
            recorded_patch = job_envelope
            return FakeUpdatePoller()

        def begin_start(self, resource_group_name, job_name, template):
            class StartPoller:
                def result(self):
                    class ExecBase:
                        name = "exec-test-patch"

                    return ExecBase()

            return StartPoller()

    backend = ContainerAppsJobsBackend(
        client=FakeJobsApi(),
        vault_url="https://myvault.vault.azure.net",
        identity_id="/sub/id/managedIdentities/my-id",
    )
    launch = LaunchSpec(
        task=TaskRecord(task_id="t1", job_id="j1", cell=CellSpec(cell_id="c1", argv=["echo"])),
        argv=["echo"],
        env={},
        secret_env={"NEW_KEY": "new-secret-val"},
        resources=Resources(),
    )
    handle = backend.submit(launch)
    assert handle.endswith("exec-test-patch")
    assert poller_awaited is True
    # Verify the patch envelope matches the SDK model shape
    assert "properties" in recorded_patch
    assert "configuration" in recorded_patch["properties"]
    secrets = recorded_patch["properties"]["configuration"]["secrets"]
    new_sec = next(s for s in secrets if s.get("name") == "new-secret-val")
    assert new_sec["key_vault_url"] == "https://myvault.vault.azure.net/secrets/new-secret-val"
    assert new_sec["identity"] == "/sub/id/managedIdentities/my-id"
    # Verify no camelCase keyVaultUrl
    assert "keyVaultUrl" not in new_sec


def test_batch_placement_submits_to_pool_specific_job_and_no_task_pool_id() -> None:
    from logic.controllers.cloud.providers.azure import AzureBatchBackend

    captured_tasks = {}

    class FakeBatchTasks:
        def add(self, job_id, task):
            captured_tasks[job_id] = task

        def get(self, job_id, task_id):
            return {"id": task_id, "state": "completed", "execution_info": {"exit_code": 0}}

    class FakeClient:
        task = FakeBatchTasks()

    backend = AzureBatchBackend(
        client=FakeClient(),
        job_id="wsr-regular-job",
        job_id_spot="wsr-spot-job",
    )

    # Submit regular task
    reg_launch = LaunchSpec(
        task=TaskRecord(task_id="reg-1", job_id="j1", cell=CellSpec(cell_id="c1", argv=["echo"])),
        argv=["echo"],
        env={},
        resources=Resources(spot=False),
    )
    reg_handle = backend.submit(reg_launch)
    assert "wsr-regular-job" in reg_handle
    assert "wsr-regular-job" in captured_tasks
    assert "pool_id" not in captured_tasks["wsr-regular-job"]

    # Submit spot task
    spot_launch = LaunchSpec(
        task=TaskRecord(task_id="spot-1", job_id="j1", cell=CellSpec(cell_id="c1", argv=["echo"])),
        argv=["echo"],
        env={},
        resources=Resources(spot=True),
    )
    spot_handle = backend.submit(spot_launch)
    assert "wsr-spot-job" in spot_handle
    assert "wsr-spot-job" in captured_tasks
    assert "pool_id" not in captured_tasks["wsr-spot-job"]


def test_vm_creation_failure_cleans_up_nic() -> None:
    from logic.controllers.cloud.interfaces import CapacityError

    deleted_nics = []

    class FailingVmClient:
        def begin_create_or_update(self, resource_group_name, vm_name, parameters):
            raise RuntimeError("Azure VM allocation failure: QuotaExceeded")

    class FakeNetworkClient:
        def begin_create_or_update(self, resource_group_name, network_interface_name, parameters):
            class Poller:
                def result(self):
                    return {"id": f"/nic/{network_interface_name}", "name": network_interface_name}

            return Poller()

        def begin_delete(self, resource_group_name, network_interface_name):
            deleted_nics.append(network_interface_name)

    backend = AzureVmBackend(client=FailingVmClient(), network_client=FakeNetworkClient())
    launch = LaunchSpec(
        task=TaskRecord(task_id="fail-alloc", job_id="j1", cell=CellSpec(cell_id="c1", argv=["echo"])),
        argv=["echo"],
        env={},
        resources=Resources(),
    )
    with pytest.raises(CapacityError, match="QuotaExceeded"):
        backend.submit(launch)

    # The created NIC must have been cleaned up
    assert "nic-wsr-vm-fail-alloc" in deleted_nics


def test_vm_deletion_failure_reported() -> None:
    class FailingDeleteVmClient:
        def instance_view(self, resource_group_name, vm_name):
            return {
                "statuses": [{"code": "PowerState/deallocated"}],
                "tags": {"wsr-exit-code": "0"},
                "priority": "Regular",
            }

        def begin_delete(self, resource_group_name, vm_name):
            raise RuntimeError("AccessDenied: subscription locked or missing Microsoft.Compute/virtualMachines/delete")

    backend = AzureVmBackend(client=FailingDeleteVmClient())
    st = backend.poll("azure-vm:eastus:wsr-rg:test-vm")
    assert st.state == AttemptState.UNKNOWN
    assert "deletion failed" in (st.message or "")


def test_vm_spot_eviction_and_async_failure_cleans_up() -> None:
    deleted_vms = []

    class SpotEvictedVmClient:
        def __init__(self, status_code: str):
            self.status_code = status_code

        def instance_view(self, resource_group_name, vm_name):
            return {
                "statuses": [{"code": self.status_code}],
                "tags": {},
                "priority": "Spot",
            }

        def begin_delete(self, resource_group_name, vm_name):
            deleted_vms.append(vm_name)

    # Spot eviction
    evicted_client = SpotEvictedVmClient("ProvisioningState/failed/SpotEviction")
    backend_evicted = AzureVmBackend(client=evicted_client)
    st = backend_evicted.poll("azure-vm:eastus:wsr-rg:evicted-vm")
    assert st.state == AttemptState.PREEMPTED
    assert "evicted-vm" in deleted_vms

    # Asynchronous allocation failure
    deleted_vms.clear()
    failed_client = SpotEvictedVmClient("ProvisioningState/failed/AllocationFailed")
    backend_failed = AzureVmBackend(client=failed_client)
    st_fail = backend_failed.poll("azure-vm:eastus:wsr-rg:failed-alloc-vm")
    assert st_fail.state == AttemptState.FAILED
    assert "failed-alloc-vm" in deleted_vms


def test_storage_and_secrets_re_raise_authorization_failures() -> None:
    from logic.controllers.cloud.providers.azure import BlobObjectStore, KeyVaultSecretStore

    class AuthError(Exception):
        status_code = 403
        error_code = "AuthorizationPermissionMismatch"

    class FailingContainerClient:
        def get_blob_client(self, blob_name):
            class BlobClient:
                def get_blob_properties(self):
                    raise AuthError("Forbidden: caller lacks Storage Blob Data Reader")

                def download_blob(self):
                    raise AuthError("Forbidden")

            return BlobClient()

        def list_blobs(self, name_starts_with=""):
            raise AuthError("Forbidden")

    store = BlobObjectStore(container="test", container_client=FailingContainerClient())
    with pytest.raises(AuthError):
        store.exists("test-key")
    with pytest.raises(AuthError):
        store.list("prefix")

    class FailingKeyVaultClient:
        def get_secret(self, name):
            raise AuthError("Forbidden: caller lacks Key Vault Secrets User")

    vault = KeyVaultSecretStore(client=FailingKeyVaultClient())
    with pytest.raises(AuthError):
        vault.get("test-secret")

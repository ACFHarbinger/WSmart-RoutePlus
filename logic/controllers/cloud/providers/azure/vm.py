"""Azure Virtual Machine compute backend implementation.

Attributes:
    AzureVmBackend: :class:`~logic.controllers.cloud.interfaces.ComputeBackend` for Azure Virtual Machines.
"""

from __future__ import annotations

import base64
import contextlib
import json
import shlex
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from logic.controllers.cloud.interfaces import (
    AttemptState,
    BackendCapabilities,
    BackendStatus,
    CapacityError,
    ComputeBackend,
    LaunchSpec,
)
from logic.controllers.cloud.model import ComputeKind, Provider
from logic.controllers.cloud.providers.azure._sdk import (
    make_compute_management_client,
    make_network_management_client,
)
from logic.controllers.cloud.providers.azure.pricing import price_per_hour


class AzureVmBackend(ComputeBackend):
    """Azure Virtual Machine compute backend.

    Args:
        client: Pre-configured ComputeManagementClient (injected for testing).
        network_client: Pre-configured NetworkManagementClient (injected for testing).
        resource_group: Azure Resource Group name.
        subscription_id: Azure Subscription ID.
        default_vm_size: Default Azure VM size (e.g. 'Standard_D2s_v5').
        subnet_id: Azure Virtual Network subnet resource ID for VM NICs.
        identity_id: User-assigned managed identity resource ID for Key Vault and ACR access.
        ssh_public_key: Optional OpenSSH public key for VM admin user authentication.
        admin_username: Linux administrator username (defaults to 'wsruser').
        vault_name: Azure Key Vault name for secret references.
        profile: Azure credentials profile name.
        region: Azure region.
    """

    provider = Provider.AZURE
    kind = ComputeKind.VM

    def __init__(
        self,
        client: Any = None,
        network_client: Any = None,
        resource_group: str = "wsr-rg",
        subscription_id: str = "00000000-0000-0000-0000-000000000000",
        default_vm_size: str = "Standard_D2s_v5",
        subnet_id: Optional[str] = None,
        identity_id: Optional[str] = None,
        ssh_public_key: Optional[str] = None,
        admin_username: str = "wsruser",
        vault_name: str = "wsr-vault",
        profile: str = "default",
        region: str = "eastus",
    ) -> None:
        self._client = client
        self._network_client = network_client
        self.resource_group = resource_group
        self.subscription_id = subscription_id
        self.default_vm_size = default_vm_size
        self.subnet_id = (
            subnet_id
            or f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Network/virtualNetworks/wsr-vnet/subnets/wsr-subnet"
        )
        self.identity_id = (
            identity_id
            or f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/wsr-identity"
        )
        self.ssh_public_key = ssh_public_key
        self.admin_username = admin_username
        self.vault_name = vault_name
        self.profile = profile
        self.region = region

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_compute_management_client(
                subscription_id=self.subscription_id,
                profile=self.profile,
            )
        return self._client

    @property
    def network_client(self) -> Any:
        if self._network_client is None:
            if hasattr(self.client, "network_interfaces"):
                return self.client
            self._network_client = make_network_management_client(
                subscription_id=self.subscription_id,
                profile=self.profile,
            )
        return self._network_client

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            provider=Provider.AZURE,
            kinds=[ComputeKind.VM],
            max_wall_seconds=86400 * 14,
            supports_spot=True,
            regions=[self.region],
            machine_types={
                "Standard_D2s_v5": {"cpus": 2.0, "memory_gb": 8.0},
                "Standard_D4s_v5": {"cpus": 4.0, "memory_gb": 16.0},
                "Standard_F2s_v2": {"cpus": 2.0, "memory_gb": 4.0},
                "Standard_F4s_v2": {"cpus": 4.0, "memory_gb": 8.0},
            },
        )

    def _parse_handle(self, handle: str) -> tuple[str, str]:
        """Return (resource_group, vm_name)."""
        if handle.startswith("azure-vm:"):
            parts = handle.split(":", 3)
            if len(parts) >= 4:
                return parts[2], parts[3]
        return self.resource_group, handle

    def _get_or_generate_ssh_key(self) -> str:
        """Return configured SSH public key or generate a throwaway key."""
        if self.ssh_public_key:
            return self.ssh_public_key.strip()

        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        return (
            private_key.public_key()
            .public_bytes(
                encoding=serialization.Encoding.OpenSSH,
                format=serialization.PublicFormat.OpenSSH,
            )
            .decode("utf-8")
        )

    def _build_custom_data(self, launch: LaunchSpec) -> str:
        """Construct UserData bootstrap script executing the task in Docker and shutting down."""
        env = dict(launch.env)
        env.setdefault("WSR_JOB_ID", launch.task.job_id)
        env.setdefault("WSR_TASK_ID", launch.task.task_id)

        # Inject secrets via WSR_SECRET_REFS (common secret path)
        if launch.secret_env:
            secret_refs = {
                var: f"akv://{self.vault_name}/{sec_name}" for var, sec_name in sorted(launch.secret_env.items())
            }
            env["WSR_SECRET_REFS"] = json.dumps(secret_refs)

        # Format docker run -e arguments
        env_flags = " ".join(f"-e {shlex.quote(f'{k}={v}')}" for k, v in sorted(env.items()))
        image = launch.image or "wsmart-route:latest"
        cmd = " ".join(shlex.quote(arg) for arg in launch.argv)

        registry = image.split("/", 1)[0] if "/" in image else ""
        # The stock Ubuntu image has no Azure CLI. Exchange the managed identity
        # token for an ACR refresh token; ARM tokens are not Docker passwords.
        acr_login = ""
        if registry.endswith(".azurecr.io"):
            acr_login = f"""ACR_TOKEN=$(curl -fsS -m 30 -X POST \\
    --data-urlencode grant_type=access_token \\
    --data-urlencode service={shlex.quote(registry)} \\
    --data-urlencode "access_token=$ACCESS_TOKEN" \\
    https://{registry}/oauth2/exchange | python3 -c 'import json,sys; print(json.load(sys.stdin)["refresh_token"])')
printf '%s' "$ACR_TOKEN" | docker login {shlex.quote(registry)} \\
    -u 00000000-0000-0000-0000-000000000000 --password-stdin
"""

        identity_query = quote(self.identity_id, safe="")
        script = f"""#!/bin/bash
set -euo pipefail
trap 'shutdown -h now' EXIT

# Dependencies for the selected Ubuntu image. Fail before launching if setup fails.
if ! command -v docker >/dev/null 2>&1; then
    apt-get update
    apt-get install -y docker.io curl python3
fi
systemctl start docker

ACCESS_TOKEN=$(curl -fsS -m 30 -H "Metadata:true" \\
    'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/&mi_res_id={identity_query}' \\
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
{acr_login}
EXIT_CODE=0
docker run --rm {env_flags} {shlex.quote(image)} {cmd} || EXIT_CODE=$?

# Refresh after execution: the initial token may have expired during the run.
ACCESS_TOKEN=$(curl -fsS -m 30 -H "Metadata:true" \\
    'http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/&mi_res_id={identity_query}' \\
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
VM_ID=$(curl -fsS -m 30 -H "Metadata:true" \\
    'http://169.254.169.254/metadata/instance/compute/resourceId?api-version=2021-02-01&format=text')
# Merge the actual exit code without replacing scheduler ownership tags.
TAG_BODY=$(printf '{{"operation":"Merge","properties":{{"tags":{{"wsr-exit-code":"%s"}}}}}}' "$EXIT_CODE")
curl -fsS -m 30 -X PATCH \\
    -H "Authorization: Bearer $ACCESS_TOKEN" \\
    -H "Content-Type: application/json" \\
    -d "$TAG_BODY" \\
    "https://management.azure.com$VM_ID/providers/Microsoft.Resources/tags/default?api-version=2021-04-01"
"""
        return base64.b64encode(script.encode("utf-8")).decode("utf-8")

    def submit(self, launch: LaunchSpec) -> str:
        """Create and start an Azure Virtual Machine executing the task."""
        vm_name = f"wsr-vm-{launch.task.task_id}"[:64]
        vm_size = launch.resources.machine_type or self.default_vm_size
        custom_data = self._build_custom_data(launch)

        # 1. Create or update Network Interface (NIC) attached to configured subnet
        nic_name = f"nic-{vm_name}"
        nic_api = getattr(self.network_client, "network_interfaces", self.network_client)
        nic_create_fn = getattr(nic_api, "begin_create_or_update", getattr(nic_api, "create_or_update", None))

        nic_id: Optional[str] = None
        if nic_create_fn is not None:
            try:
                nic_poller = nic_create_fn(
                    resource_group_name=self.resource_group,
                    network_interface_name=nic_name,
                    parameters={
                        "location": self.region,
                        "ip_configurations": [
                            {
                                "name": "ipconfig1",
                                "subnet": {"id": self.subnet_id},
                                "primary": True,
                            }
                        ],
                    },
                )
                nic_res = getattr(nic_poller, "result", lambda: nic_poller)()
                nic_id = nic_res.get("id") if isinstance(nic_res, dict) else getattr(nic_res, "id", None)
            except Exception as exc:
                msg = str(exc)
                if any(term in msg.lower() for term in ("quotaexceeded", "allocationfailed", "capacity")):
                    raise CapacityError(f"Azure NIC capacity unavailable: {msg}") from exc
                raise

        if not nic_id:
            nic_id = f"/subscriptions/{self.subscription_id}/resourceGroups/{self.resource_group}/providers/Microsoft.Network/networkInterfaces/{nic_name}"

        # 2. Build full VM create parameters
        ssh_key = self._get_or_generate_ssh_key()
        vm_params: Dict[str, Any] = {
            "location": self.region,
            "hardware_profile": {"vm_size": vm_size},
            "storage_profile": {
                "image_reference": {
                    "publisher": "canonical",
                    "offer": "0001-com-ubuntu-server-jammy",
                    "sku": "22_04-lts-gen2",
                    "version": "latest",
                },
                "os_disk": {
                    "name": f"osdisk-{vm_name}",
                    "caching": "ReadWrite",
                    "create_option": "FromImage",
                    "managed_disk": {"storage_account_type": "StandardSSD_LRS"},
                    "delete_option": "Delete",
                },
            },
            "os_profile": {
                "computer_name": vm_name[:15],
                "admin_username": self.admin_username,
                "custom_data": custom_data,
                "linux_configuration": {
                    "disable_password_authentication": True,
                    "ssh": {
                        "public_keys": [
                            {
                                "path": f"/home/{self.admin_username}/.ssh/authorized_keys",
                                "key_data": ssh_key,
                            }
                        ]
                    },
                },
            },
            "network_profile": {
                "network_interfaces": [
                    {
                        "id": nic_id,
                        "delete_option": "Delete",
                    }
                ]
            },
            "identity": {
                "type": "UserAssigned",
                "user_assigned_identities": {
                    self.identity_id: {},
                },
            },
            "tags": {
                "wsr-job-id": launch.task.job_id,
                "wsr-task-id": launch.task.task_id,
                "wsr:job_id": launch.task.job_id,
                "wsr:task_id": launch.task.task_id,
                "wsr:attempt": str(launch.task.attempt),
                "wsr:spot": "true" if launch.resources.spot else "false",
            },
        }

        # Spot market options
        if launch.resources.spot:
            vm_params["priority"] = "Spot"
            vm_params["eviction_policy"] = "Deallocate"

        vm_api = getattr(self.client, "virtual_machines", self.client)
        create_fn = getattr(vm_api, "begin_create_or_update", getattr(vm_api, "create_or_update", None))
        if create_fn is None:
            raise RuntimeError("Azure VM create API not available")

        try:
            create_fn(
                resource_group_name=self.resource_group,
                vm_name=vm_name,
                parameters=vm_params,
            )
        except Exception as exc:
            msg = str(exc)
            if any(term in msg.lower() for term in ("quotaexceeded", "allocationfailed", "capacity", "notavail")):
                raise CapacityError(f"Azure VM capacity unavailable: {msg}") from exc
            raise

        return f"azure-vm:{self.region}:{self.resource_group}:{vm_name}"

    def _resolve_terminal_status(self, exit_code_str: Optional[str], is_spot: bool) -> BackendStatus:
        """Derive final BackendStatus from exit code tag and spot configuration."""
        if exit_code_str is not None:
            try:
                ec = int(exit_code_str)
            except ValueError:
                ec = 1
            if ec == 0:
                return BackendStatus(AttemptState.SUCCEEDED, exit_code=0)
            if ec == 143 and is_spot:
                return BackendStatus(
                    AttemptState.PREEMPTED,
                    exit_code=143,
                    message="Spot instance interrupted (exit 143)",
                )
            return BackendStatus(
                AttemptState.FAILED,
                exit_code=ec,
                message=f"VM exited with code {ec}",
            )
        if is_spot:
            return BackendStatus(
                AttemptState.PREEMPTED,
                message="Spot instance evicted / deallocated",
            )
        return BackendStatus(
            AttemptState.FAILED,
            exit_code=1,
            message="VM stopped without exit code tag",
        )

    def poll(self, handle: str) -> BackendStatus:
        rg, vm_name = self._parse_handle(handle)
        vm_api = getattr(self.client, "virtual_machines", self.client)
        get_fn = getattr(vm_api, "instance_view", getattr(vm_api, "get", None))

        if get_fn is None:
            return BackendStatus(AttemptState.UNKNOWN, message="VM API not available")

        try:
            view = get_fn(resource_group_name=rg, vm_name=vm_name)
        except Exception as exc:
            return BackendStatus(AttemptState.UNKNOWN, message=str(exc))

        if not view:
            return BackendStatus(AttemptState.UNKNOWN, message="no such VM")

        statuses = getattr(view, "statuses", []) or (view.get("statuses", []) if isinstance(view, dict) else [])
        status_codes = [s.code if hasattr(s, "code") else s.get("code", "") for s in statuses]
        status_str = " ".join(status_codes).lower()

        # Check spot eviction from instance view statuses
        if any(term in status_str for term in ("spoteviction", "evicted", "preempt")):
            return BackendStatus(AttemptState.PREEMPTED, message="Azure spot VM evicted")

        tags = getattr(view, "tags", {}) or (view.get("tags", {}) if isinstance(view, dict) else {})
        if not tags and hasattr(vm_api, "get"):
            try:
                vm_obj = vm_api.get(resource_group_name=rg, vm_name=vm_name)
                tags = getattr(vm_obj, "tags", {}) or (vm_obj.get("tags", {}) if isinstance(vm_obj, dict) else {})
            except Exception:
                pass

        exit_code_str = tags.get("wsr-exit-code") or tags.get("wsr:exit_code")
        is_spot = (
            getattr(view, "priority", None) == "Spot"
            or (isinstance(view, dict) and view.get("priority") == "Spot")
            or tags.get("wsr:spot") == "true"
        )

        if "powerstate/running" in status_str:
            return BackendStatus(AttemptState.RUNNING)

        if "powerstate/starting" in status_str or "provisioningstate/creating" in status_str:
            return BackendStatus(AttemptState.PENDING)

        if "powerstate/deallocated" in status_str or "powerstate/stopped" in status_str:
            # Stopped VMs continue to bill while allocated; read exit status and trigger deletion
            final_status = self._resolve_terminal_status(exit_code_str, is_spot)

            # Delete the stopped VM to halt billing (NIC and disk delete via delete_option="Delete")
            delete_fn = getattr(vm_api, "begin_delete", getattr(vm_api, "delete", None))
            if delete_fn is not None:
                with contextlib.suppress(Exception):
                    delete_fn(resource_group_name=rg, vm_name=vm_name)

            return final_status

        return BackendStatus(AttemptState.UNKNOWN, message=f"unknown VM status: {status_codes}")

    def cancel(self, handle: str) -> None:
        rg, vm_name = self._parse_handle(handle)
        vm_api = getattr(self.client, "virtual_machines", self.client)
        delete_fn = getattr(vm_api, "begin_delete", getattr(vm_api, "delete", None))
        if delete_fn is not None:
            with contextlib.suppress(Exception):
                delete_fn(resource_group_name=rg, vm_name=vm_name)

    def logs(self, handle: str, tail: Optional[int] = None) -> str:
        rg, vm_name = self._parse_handle(handle)
        log_lines: List[str] = []

        try:
            vm_api = getattr(self.client, "virtual_machines", self.client)
            if hasattr(vm_api, "retrieve_boot_diagnostics_data"):
                data = vm_api.retrieve_boot_diagnostics_data(resource_group_name=rg, vm_name=vm_name)
                content = getattr(data, "serial_console_log_blob_uri", None)
                if content:
                    log_lines = str(content).splitlines()
        except Exception:
            pass

        if not log_lines:
            return ""

        if tail is not None and tail > 0:
            return "\n".join(log_lines[-tail:])
        return "\n".join(log_lines)

    def price_per_hour(
        self,
        machine_type: Optional[str] = None,
        region: Optional[str] = None,
        spot: bool = False,
    ) -> Optional[float]:
        return price_per_hour(machine_type or self.default_vm_size, region or self.region, spot)


__all__ = ["AzureVmBackend"]

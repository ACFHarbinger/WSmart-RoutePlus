"""Azure Virtual Machine compute backend implementation.

Attributes:
    AzureVmBackend: :class:`~logic.controllers.cloud.interfaces.ComputeBackend` for Azure Virtual Machines.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from logic.controllers.cloud.providers.azure.orphan_queue import AzureVmOrphanQueueMixin
from logic.controllers.cloud.providers.azure.pricing import price_per_hour
from logic.controllers.cloud.providers.azure.provisioning import AzureProvisioningMixin
from logic.controllers.cloud.providers.azure.vm_bootstrap import AzureVmBootstrapMixin


class AzureVmBackend(AzureProvisioningMixin, AzureVmOrphanQueueMixin, AzureVmBootstrapMixin, ComputeBackend):
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
        orphan_file: Optional[Path | str] = None,
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
        if orphan_file is not None:
            self.orphan_file = Path(orphan_file)
        else:
            base_dir = Path(os.environ.get("WSR_STATE_DIR", Path.home() / ".wsr"))
            scope = hashlib.sha256(f"{self.subscription_id}:{self.profile}".encode()).hexdigest()[:16]
            self.orphan_file = base_dir / f"azure_vm_orphans_{scope}.json"

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

    @staticmethod
    def _is_not_found(exc: Exception) -> bool:
        """Return True if exception indicates the Azure resource does not exist (404/NotFound)."""
        if getattr(exc, "status_code", None) == 404:
            return True
        resp = getattr(exc, "response", None)
        if resp is not None and getattr(resp, "status_code", None) == 404:
            return True
        error_code = getattr(exc, "error_code", None) or getattr(exc, "code", None)
        # Message text may describe a missing permission/credential, not a resource.
        return isinstance(error_code, str) and error_code.lower() in (
            "resourcenotfound",
            "notfound",
            "resourcegroupnotfound",
        )

    def _cleanup_nic_in_rg(self, rg: str, nic_name: str) -> None:
        """Attempt to delete a network interface in the specified resource group."""
        nic_api = getattr(self.network_client, "network_interfaces", self.network_client)
        del_fn = getattr(nic_api, "begin_delete", getattr(nic_api, "delete", None))
        if del_fn is None:
            raise RuntimeError("Azure NIC deletion API not available")
        try:
            poller = del_fn(resource_group_name=rg, network_interface_name=nic_name)
            if hasattr(poller, "result"):
                poller.result()
        except Exception as exc:
            if not self._is_not_found(exc):
                raise

    def _cleanup_nic(self, nic_name: str) -> None:
        """Attempt to delete a network interface if VM creation fails."""
        self._cleanup_nic_in_rg(self.resource_group, nic_name)

    def _delete_vm_resource(self, rg: str, vm_name: str) -> None:
        """Wait for deletion so asynchronous failures cannot masquerade as cleanup."""
        vm_api = getattr(self.client, "virtual_machines", self.client)
        delete_fn = getattr(vm_api, "begin_delete", getattr(vm_api, "delete", None))
        if delete_fn is None:
            raise RuntimeError("Azure VM deletion API not available")
        try:
            poller = delete_fn(resource_group_name=rg, vm_name=vm_name)
            if hasattr(poller, "result"):
                poller.result()
        except Exception as exc:
            if not self._is_not_found(exc):
                raise

    def _teardown_failed_submit(self, rg: str, vm_name: str, nic_name: str) -> None:
        """Teardown partial VM first, then NIC, recording orphans if cleanup fails."""
        # 1. Attempt partial VM deletion first (detaches NIC)
        try:
            self._delete_vm_resource(rg, vm_name)
        except Exception as exc:
            if not self._is_not_found(exc):
                self._record_orphan("vm", rg, vm_name, str(exc))

        # 2. Attempt NIC deletion
        try:
            self._cleanup_nic_in_rg(rg, nic_name)
        except Exception as exc:
            if not self._is_not_found(exc):
                self._record_orphan("nic", rg, nic_name, str(exc))

    def _finish_after_cleanup(self, rg: str, vm_name: str, status: BackendStatus) -> BackendStatus:
        """Keep the scheduler polling until cleanup succeeds, preserving the outcome."""
        try:
            self._delete_vm_resource(rg, vm_name)
        except Exception as exc:
            return BackendStatus(AttemptState.UNKNOWN, message=f"VM deletion failed; retry cleanup: {exc}")
        return status

    def submit(self, launch: LaunchSpec) -> str:
        """Persist intent before the first resource allocation and retain it through handoff."""
        self.reconcile_provisions()
        self.drain_orphans()
        suffix = f"-a{launch.task.attempt}" if launch.task.attempt > 1 else ""
        vm_name = f"wsr-vm-{launch.task.task_id}"[: 64 - len(suffix)] + suffix
        for orphan in self._load_orphans():
            if (
                orphan.get("subscription_id") == self.subscription_id
                and orphan.get("profile") == self.profile
                and orphan.get("resource_group") == self.resource_group
                and orphan.get("name") in (vm_name, f"nic-{vm_name}")
            ):
                raise CapacityError(
                    "Previous Azure resources still require cleanup before this attempt can be submitted"
                )
        if launch.task.lease_owner:
            vm_name = vm_name[:55] + "-" + uuid.uuid4().hex[:8]
        handle = f"azure-vm:{self.region}:{self.resource_group}:{vm_name}"
        token = self._begin_provision(launch, vm_name, handle)
        try:
            self._check_provision_lease(launch)
            self._allocate_vm(launch, vm_name)
            self._check_provision_lease(launch)
        except Exception:
            self._teardown_failed_submit(self.resource_group, vm_name, f"nic-{vm_name}")
            self._forget_provision(token)
            raise
        return handle

    def _allocate_vm(self, launch: LaunchSpec, vm_name: str) -> None:
        """Allocate one journalled VM; the submit wrapper owns failure recovery."""
        vm_size = launch.resources.machine_type or self.default_vm_size
        custom_data = self._build_custom_data(launch)
        ssh_key = self._get_or_generate_ssh_key()

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

        self._check_provision_lease(launch)
        # 2. Build full VM create parameters
        vm_params: Dict[str, Any] = {
            "location": self.region,
            "hardware_profile": {"vm_size": vm_size},
            "diagnostics_profile": {
                "boot_diagnostics": {
                    "enabled": True,
                }
            },
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
            poller = create_fn(
                resource_group_name=self.resource_group,
                vm_name=vm_name,
                parameters=vm_params,
            )
            if hasattr(poller, "result"):
                poller.result()
        except Exception as exc:
            msg = str(exc)
            if any(term in msg.lower() for term in ("quotaexceeded", "allocationfailed", "capacity", "notavail")):
                raise CapacityError(f"Azure VM capacity unavailable: {msg}") from exc
            raise

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
        self.drain_orphans()
        rg, vm_name = self._parse_handle(handle)
        vm_api = getattr(self.client, "virtual_machines", self.client)
        get_fn = getattr(vm_api, "instance_view", getattr(vm_api, "get", None))

        if get_fn is None:
            return BackendStatus(AttemptState.UNKNOWN, message="VM API not available")

        try:
            view = get_fn(resource_group_name=rg, vm_name=vm_name)
        except Exception as exc:
            if self._is_not_found(exc):
                return BackendStatus(AttemptState.FAILED, exit_code=1, message="VM resource no longer exists")
            return BackendStatus(AttemptState.UNKNOWN, message=str(exc))

        if not view:
            return BackendStatus(AttemptState.FAILED, exit_code=1, message="no such VM")

        statuses = getattr(view, "statuses", []) or (view.get("statuses", []) if isinstance(view, dict) else [])
        status_codes = [s.code if hasattr(s, "code") else s.get("code", "") for s in statuses]
        status_str = " ".join(status_codes).lower()

        # Every terminal path must complete deletion before releasing the attempt.
        if "provisioningstate/failed" in status_str:
            if any(term in status_str for term in ("spoteviction", "evicted", "preempt")):
                outcome = BackendStatus(AttemptState.PREEMPTED, message=f"Azure spot VM evicted: {status_str}")
            else:
                outcome = BackendStatus(
                    AttemptState.FAILED, exit_code=1, message=f"Azure VM provisioning failed: {status_str}"
                )
            return self._finish_after_cleanup(rg, vm_name, outcome)

        if any(term in status_str for term in ("spoteviction", "evicted", "preempt")):
            return self._finish_after_cleanup(
                rg, vm_name, BackendStatus(AttemptState.PREEMPTED, message="Azure spot VM evicted")
            )

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

            return self._finish_after_cleanup(rg, vm_name, final_status)

        return BackendStatus(AttemptState.UNKNOWN, message=f"unknown VM status: {status_codes}")

    def cancel(self, handle: str) -> None:
        rg, vm_name = self._parse_handle(handle)
        try:
            self._delete_vm_resource(rg, vm_name)
        except Exception as exc:
            if not self._is_not_found(exc):
                self._record_orphan("vm", rg, vm_name, str(exc))
                raise

    def logs(self, handle: str, tail: Optional[int] = None) -> str:
        rg, vm_name = self._parse_handle(handle)
        log_lines: List[str] = []

        try:
            vm_api = getattr(self.client, "virtual_machines", self.client)
            if hasattr(vm_api, "retrieve_boot_diagnostics_data"):
                data = vm_api.retrieve_boot_diagnostics_data(resource_group_name=rg, vm_name=vm_name)
                content = getattr(data, "serial_console_log_blob_uri", None)
                if content:
                    url = str(content)
                    if url.startswith("http://") or url.startswith("https://"):
                        try:
                            import requests

                            resp = requests.get(url, timeout=10)
                            if resp.status_code == 200:
                                log_lines = resp.text.splitlines()
                        except Exception:
                            pass
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

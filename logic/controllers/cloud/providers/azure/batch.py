"""Azure Batch compute backend implementation.

Attributes:
    AzureBatchBackend: :class:`~logic.controllers.cloud.interfaces.ComputeBackend` for Azure Batch.
"""

from __future__ import annotations

import contextlib
import json
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
from logic.controllers.cloud.providers.azure._sdk import make_batch_service_client
from logic.controllers.cloud.providers.azure.pricing import price_per_hour


class AzureBatchBackend(ComputeBackend):
    """Azure Batch compute backend.

    Note:
        The Azure Batch pool must be container-enabled and configured with the user-assigned
        managed identity referenced by ``identity_id``. This identity is required both for
        authenticating to private Azure Container Registry (ACR) instances and for resolving
        ``akv://`` Key Vault secret references at runtime.

    Args:
        client: Pre-configured BatchServiceClient (injected for testing).
        pool_id: Azure Batch pool identifier.
        pool_id_spot: Optional Azure Batch pool using Spot / Low-Priority nodes.
        job_id: Azure Batch job identifier.
        batch_url: Azure Batch account URL.
        vault_name: Azure Key Vault name for secret references.
        registry_server: Optional container registry server (e.g. 'myacr.azurecr.io').
        identity_id: User-assigned managed identity resource ID for ACR and Key Vault access.
        profile: Azure credentials profile name.
        region: Azure region.
    """

    provider = Provider.AZURE
    kind = ComputeKind.BATCH

    def __init__(
        self,
        client: Any = None,
        pool_id: str = "wsr-batch-pool",
        pool_id_spot: Optional[str] = None,
        job_id: str = "wsr-jobs",
        batch_url: Optional[str] = None,
        vault_name: str = "wsr-vault",
        registry_server: Optional[str] = None,
        identity_id: Optional[str] = None,
        profile: str = "default",
        region: str = "eastus",
    ) -> None:
        self._client = client
        self.pool_id = pool_id
        self.pool_id_spot = pool_id_spot or f"{pool_id}-spot"
        self.job_id = job_id
        self.batch_url = batch_url or "https://wsrbatch.eastus.batch.azure.com"
        self.vault_name = vault_name
        self.registry_server = registry_server
        self.identity_id = (
            identity_id
            or "/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/wsr-rg/providers/Microsoft.ManagedIdentity/userAssignedIdentities/wsr-batch-identity"
        )
        self.profile = profile
        self.region = region

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_batch_service_client(batch_url=self.batch_url, profile=self.profile)
        return self._client

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            provider=Provider.AZURE,
            kinds=[ComputeKind.BATCH],
            max_wall_seconds=86400 * 7,
            supports_spot=True,
            regions=[self.region],
            machine_types={
                "standard_d2s_v5": {"cpus": 2.0, "memory_gb": 8.0},
                "standard_d4s_v5": {"cpus": 4.0, "memory_gb": 16.0},
                "standard_f2s_v2": {"cpus": 2.0, "memory_gb": 4.0},
                "standard_f4s_v2": {"cpus": 4.0, "memory_gb": 8.0},
            },
        )

    def _parse_handle(self, handle: str) -> tuple[str, str]:
        """Return (job_id, task_id)."""
        if handle.startswith("azure-batch:"):
            parts = handle.split(":", 3)
            if len(parts) >= 4:
                return parts[2], parts[3]
        return self.job_id, handle

    def submit(self, launch: LaunchSpec) -> str:
        if launch.resources.cpus > 128 or launch.resources.memory_gb > 512:
            raise CapacityError(
                f"requested {launch.resources.cpus} CPUs and {launch.resources.memory_gb} GiB exceeds Azure Batch quota"
            )

        task_id = f"task-{launch.task.task_id}"
        pool_id = self.pool_id_spot if launch.resources.spot else self.pool_id

        env = dict(launch.env)
        env.setdefault("WSR_JOB_ID", launch.task.job_id)
        env.setdefault("WSR_TASK_ID", launch.task.task_id)

        # Inject secrets via WSR_SECRET_REFS (common secret path)
        if launch.secret_env:
            secret_refs = {
                var: f"akv://{self.vault_name}/{sec_name}" for var, sec_name in sorted(launch.secret_env.items())
            }
            env["WSR_SECRET_REFS"] = json.dumps(secret_refs)

        env_settings = [{"name": k, "value": str(v)} for k, v in sorted(env.items())]

        cmd_line = " ".join(launch.argv)
        task_data: Dict[str, Any] = {
            "id": task_id,
            "command_line": cmd_line,
            "environment_settings": env_settings,
            "pool_id": pool_id,
        }

        if launch.resources.max_wall_seconds is not None:
            task_data["constraints"] = {
                "max_wall_clock_time": f"PT{int(launch.resources.max_wall_seconds)}S",
            }

        image = launch.image or "wsmart-route:latest"
        registry = image.split("/", 1)[0] if "/" in image else ""
        container_settings: Dict[str, Any] = {"image_name": image}

        # Private ACR images require registry settings with identity reference on container-enabled pools
        if ".azurecr.io" in registry or self.registry_server:
            reg_server = self.registry_server or registry
            reg_dict: Dict[str, Any] = {"registry_server": reg_server}
            if self.identity_id:
                reg_dict["identity_reference"] = {"resource_id": self.identity_id}
            container_settings["registry"] = reg_dict

        task_data["container_settings"] = container_settings

        self.client.task.add(job_id=self.job_id, task=task_data)
        return f"azure-batch:{self.region}:{self.job_id}:{task_id}"

    def poll(self, handle: str) -> BackendStatus:
        job_id, task_id = self._parse_handle(handle)
        try:
            task = self.client.task.get(job_id=job_id, task_id=task_id)
        except Exception as exc:
            return BackendStatus(AttemptState.UNKNOWN, message=str(exc))

        if not task:
            return BackendStatus(AttemptState.UNKNOWN, message="no such task")

        state = getattr(task, "state", None) or task.get("state", "")
        pool_id = getattr(task, "pool_id", "") or (task.get("pool_id", "") if isinstance(task, dict) else "")
        is_spot = (pool_id == self.pool_id_spot) or ("-spot" in str(pool_id))

        exec_info = getattr(task, "execution_info", None) or task.get("execution_info", {})
        exit_code = getattr(exec_info, "exit_code", None) if exec_info else None
        if exit_code is None and isinstance(exec_info, dict):
            exit_code = exec_info.get("exit_code")

        failure_info = getattr(exec_info, "failure_info", None) or (
            exec_info.get("failure_info") if isinstance(exec_info, dict) else None
        )
        fail_msg = str(failure_info) if failure_info else ""

        if state in ("active", "preparing"):
            return BackendStatus(AttemptState.PENDING, message=fail_msg)

        if state == "running":
            return BackendStatus(AttemptState.RUNNING, message=fail_msg)

        if state == "completed":
            # Detect low priority spot preemption
            fail_lower = fail_msg.lower()
            is_preempted = (
                getattr(exec_info, "preempted", False)
                or (isinstance(exec_info, dict) and exec_info.get("preempted", False))
                or (exit_code == 143 and is_spot)
                or any(term in fail_lower for term in ("spot", "preempt", "lowpriority", "evicted", "node unallocated"))
            )
            if is_preempted:
                return BackendStatus(AttemptState.PREEMPTED, message=fail_msg or "Azure Batch node preempted")

            if exit_code == 0:
                return BackendStatus(AttemptState.SUCCEEDED, exit_code=0)

            code = exit_code if exit_code is not None else 1
            return BackendStatus(AttemptState.FAILED, exit_code=code, message=fail_msg or f"task failed (exit {code})")

        return BackendStatus(AttemptState.UNKNOWN, message=f"unknown state: {state}")

    def cancel(self, handle: str) -> None:
        job_id, task_id = self._parse_handle(handle)
        with contextlib.suppress(Exception):
            self.client.task.terminate(job_id=job_id, task_id=task_id)

    def logs(self, handle: str, tail: Optional[int] = None) -> str:
        job_id, task_id = self._parse_handle(handle)
        log_lines: List[str] = []

        try:
            if hasattr(self.client, "file"):
                stream = self.client.file.get_from_task(job_id=job_id, task_id=task_id, file_path="stdout.txt")
                data = stream.read() if hasattr(stream, "read") else bytes(stream)
                log_lines = data.decode("utf-8", errors="replace").splitlines()
        except Exception:
            pass

        if not log_lines:
            return ""

        if tail is not None:
            return "\n".join(log_lines[-tail:])
        return "\n".join(log_lines)

    def price_per_hour(
        self,
        machine_type: Optional[str] = None,
        region: Optional[str] = None,
        spot: bool = False,
    ) -> Optional[float]:
        return price_per_hour(machine_type, region or self.region, spot)


__all__ = ["AzureBatchBackend"]

"""Azure Container Apps Jobs compute backend implementation.

Attributes:
    ContainerAppsJobsBackend: :class:`~logic.controllers.cloud.interfaces.ComputeBackend` for Azure Container Apps Jobs.
"""

from __future__ import annotations

import contextlib
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
from logic.controllers.cloud.providers.azure._sdk import make_container_apps_client
from logic.controllers.cloud.providers.azure.pricing import price_per_hour


class ContainerAppsJobsBackend(ComputeBackend):
    """Azure Container Apps Jobs serverless compute backend.

    Args:
        client: Pre-configured ContainerAppsAPIClient (injected for testing).
        resource_group: Azure Resource Group name.
        job_name: Container Apps Job resource name.
        subscription_id: Azure Subscription ID.
        profile: Azure credentials profile name.
        region: Azure region.
    """

    provider = Provider.AZURE
    kind = ComputeKind.SERVERLESS

    def __init__(
        self,
        client: Any = None,
        resource_group: str = "wsr-rg",
        job_name: str = "wsr-job",
        subscription_id: str = "00000000-0000-0000-0000-000000000000",
        profile: str = "default",
        region: str = "eastus",
    ) -> None:
        self._client = client
        self.resource_group = resource_group
        self.job_name = job_name
        self.subscription_id = subscription_id
        self.profile = profile
        self.region = region

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_container_apps_client(
                subscription_id=self.subscription_id,
                profile=self.profile,
            )
        return self._client

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            provider=Provider.AZURE,
            kinds=[ComputeKind.SERVERLESS],
            max_wall_seconds=86400 * 7,
            supports_spot=True,
            regions=[self.region],
            machine_types={
                "container_apps.1vcpu": {"cpus": 1.0, "memory_gb": 2.0},
                "container_apps.2vcpu": {"cpus": 2.0, "memory_gb": 4.0},
            },
        )

    def _parse_handle(self, handle: str) -> tuple[str, str, str]:
        """Return (resource_group, job_name, execution_name)."""
        if handle.startswith("azure-aca:"):
            parts = handle.split(":", 4)
            if len(parts) >= 5:
                return parts[2], parts[3], parts[4]
        return self.resource_group, self.job_name, handle

    def submit(self, launch: LaunchSpec) -> str:
        if launch.resources.cpus > 16 or launch.resources.memory_gb > 64:
            raise CapacityError(
                f"requested {launch.resources.cpus} CPUs and {launch.resources.memory_gb} GiB exceeds Container Apps limits"
            )

        env_vars = [{"name": k, "value": str(v)} for k, v in sorted(launch.env.items())]
        # Native secret references
        if launch.secret_env:
            env_vars.extend([{"name": k, "secretRef": str(v)} for k, v in sorted(launch.secret_env.items())])

        container_override: Dict[str, Any] = {
            "name": "runner",
            "command": list(launch.argv),
            "env": env_vars,
            "resources": {
                "cpu": launch.resources.cpus,
                "memory": f"{launch.resources.memory_gb:.1f}Gi",
            },
        }
        if launch.image:
            container_override["image"] = launch.image

        template = {"containers": [container_override]}

        # If jobs client has start/begin_start
        jobs_api = getattr(self.client, "jobs", self.client)
        start_fn = getattr(jobs_api, "begin_start", getattr(jobs_api, "start", None))

        if start_fn is None:
            raise RuntimeError("Azure Container Apps jobs start API not available")

        resp = start_fn(
            resource_group_name=self.resource_group,
            job_name=self.job_name,
            template=template,
        )
        exec_name = getattr(resp, "name", None) or (resp.get("name") if isinstance(resp, dict) else f"exec-{launch.task.task_id}")

        return f"azure-aca:{self.region}:{self.resource_group}:{self.job_name}:{exec_name}"

    def poll(self, handle: str) -> BackendStatus:
        rg, job_name, exec_name = self._parse_handle(handle)
        jobs_api = getattr(self.client, "job_executions", getattr(self.client, "jobs", self.client))
        get_fn = getattr(jobs_api, "get", None)

        if get_fn is None:
            return BackendStatus(AttemptState.UNKNOWN, message="jobs API not available")

        try:
            execution = get_fn(resource_group_name=rg, job_name=job_name, job_execution_name=exec_name)
        except Exception as exc:
            return BackendStatus(AttemptState.UNKNOWN, message=str(exc))

        if not execution:
            return BackendStatus(AttemptState.UNKNOWN, message="no such execution")

        status = getattr(execution, "status", None) or (execution.get("status") if isinstance(execution, dict) else "")
        status_str = str(status).lower()

        if status_str in ("pending", "starting"):
            return BackendStatus(AttemptState.PENDING)

        if status_str == "running":
            return BackendStatus(AttemptState.RUNNING)

        if status_str == "succeeded":
            return BackendStatus(AttemptState.SUCCEEDED, exit_code=0)

        if status_str in ("failed", "degraded"):
            # Check for spot preemption
            msg = getattr(execution, "message", "") or (execution.get("message", "") if isinstance(execution, dict) else "")
            if any(term in str(msg).lower() for term in ("spot", "preempt", "evict", "reclaim")):
                return BackendStatus(AttemptState.PREEMPTED, message=str(msg) or "Container Apps spot preemption")
            return BackendStatus(AttemptState.FAILED, exit_code=1, message=str(msg) or "job execution failed")

        return BackendStatus(AttemptState.UNKNOWN, message=f"unknown status: {status}")

    def cancel(self, handle: str) -> None:
        rg, job_name, exec_name = self._parse_handle(handle)
        jobs_api = getattr(self.client, "job_executions", getattr(self.client, "jobs", self.client))
        stop_fn = getattr(jobs_api, "begin_stop", getattr(jobs_api, "stop", None))
        if stop_fn is not None:
            with contextlib.suppress(Exception):
                stop_fn(resource_group_name=rg, job_name=job_name, job_execution_name=exec_name)

    def logs(self, handle: str, tail: Optional[int] = None) -> str:
        rg, job_name, exec_name = self._parse_handle(handle)
        log_lines: List[str] = []

        try:
            jobs_api = getattr(self.client, "job_executions", getattr(self.client, "jobs", self.client))
            execution = jobs_api.get(resource_group_name=rg, job_name=job_name, job_execution_name=exec_name)
            if hasattr(execution, "_log_lines"):
                log_lines = list(execution._log_lines)
            elif isinstance(execution, dict) and "_log_lines" in execution:
                log_lines = list(execution["_log_lines"])
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
        return price_per_hour(machine_type or "container_apps", region or self.region, spot)


__all__ = ["ContainerAppsJobsBackend"]

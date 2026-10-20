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
        logs_client: Any = None,
        workspace_id: Optional[str] = None,
        resource_group: str = "wsr-rg",
        job_name: str = "wsr-job",
        subscription_id: str = "00000000-0000-0000-0000-000000000000",
        vault_url: Optional[str] = None,
        identity_id: Optional[str] = None,
        profile: str = "default",
        region: str = "eastus",
    ) -> None:
        self._client = client
        self._logs_client = logs_client
        self.workspace_id = workspace_id
        self.resource_group = resource_group
        self.job_name = job_name
        self.subscription_id = subscription_id
        self.vault_url = vault_url
        self.identity_id = identity_id
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

    @property
    def logs_client(self) -> Any:
        if self._logs_client is None and self.workspace_id:
            with contextlib.suppress(Exception):
                from azure.monitor.query import LogsQueryClient

                from logic.controllers.cloud.providers.azure._sdk import make_azure_credential

                self._logs_client = LogsQueryClient(make_azure_credential(self.profile))
        return self._logs_client

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

        jobs_api = getattr(self.client, "jobs", self.client)

        # Verify or declare secrets in job configuration
        if launch.secret_env and hasattr(jobs_api, "get"):
            job_def = jobs_api.get(resource_group_name=self.resource_group, job_name=self.job_name)
            props = getattr(job_def, "properties", None) or (
                job_def.get("properties") if isinstance(job_def, dict) else job_def
            )
            config = getattr(props, "configuration", None) or (
                props.get("configuration") if isinstance(props, dict) else {}
            )
            existing_secrets = getattr(config, "secrets", None) or (
                config.get("secrets") if isinstance(config, dict) else []
            )
            existing_names = {s.name if hasattr(s, "name") else s.get("name") for s in (existing_secrets or [])}
            missing = [sec for sec in launch.secret_env.values() if sec not in existing_names]
            if missing:
                update_fn = getattr(jobs_api, "begin_update", getattr(jobs_api, "update", None))
                if update_fn is None or not self.vault_url:
                    raise ValueError(
                        f"secrets {missing} not declared in Container Apps job {self.job_name} configuration.secrets"
                    )
                new_secrets = list(existing_secrets or [])
                for sec_name in missing:
                    entry = {"name": sec_name}
                    entry["key_vault_url"] = f"{self.vault_url.rstrip('/')}/secrets/{sec_name}"
                    if self.identity_id:
                        entry["identity"] = self.identity_id
                    new_secrets.append(entry)

                patch_envelope = {
                    "properties": {
                        "configuration": {
                            "secrets": new_secrets,
                        }
                    }
                }
                update_poller = update_fn(
                    resource_group_name=self.resource_group,
                    job_name=self.job_name,
                    job_envelope=patch_envelope,
                )
                if hasattr(update_poller, "result"):
                    update_poller.result()

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

        start_fn = getattr(jobs_api, "begin_start", getattr(jobs_api, "start", None))
        if start_fn is None:
            raise RuntimeError("Azure Container Apps jobs start API not available")

        poller = start_fn(
            resource_group_name=self.resource_group,
            job_name=self.job_name,
            template=template,
        )
        resp = poller.result() if hasattr(poller, "result") else poller
        exec_name = getattr(resp, "name", None) or (resp.get("name") if isinstance(resp, dict) else None)
        if not exec_name:
            raise RuntimeError("Container Apps start did not return execution name")

        return f"azure-aca:{self.region}:{self.resource_group}:{self.job_name}:{exec_name}"

    def poll(self, handle: str) -> BackendStatus:
        rg, job_name, exec_name = self._parse_handle(handle)
        executions_api = getattr(self.client, "jobs_executions", None)
        execution = None

        try:
            if executions_api is not None and hasattr(executions_api, "list"):
                for item in executions_api.list(resource_group_name=rg, job_name=job_name):
                    item_name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else None)
                    if item_name == exec_name:
                        execution = item
                        break
            else:
                fallback_api = getattr(self.client, "job_executions", getattr(self.client, "jobs", self.client))
                if hasattr(fallback_api, "get"):
                    execution = fallback_api.get(
                        resource_group_name=rg, job_name=job_name, job_execution_name=exec_name
                    )
                elif hasattr(fallback_api, "list"):
                    for item in fallback_api.list(resource_group_name=rg, job_name=job_name):
                        item_name = getattr(item, "name", None) or (
                            item.get("name") if isinstance(item, dict) else None
                        )
                        if item_name == exec_name:
                            execution = item
                            break
        except Exception as exc:
            return BackendStatus(AttemptState.UNKNOWN, message=str(exc))

        if not execution:
            return BackendStatus(AttemptState.UNKNOWN, message="no such execution")

        status = (
            getattr(execution, "status", None)
            or getattr(getattr(execution, "properties", None), "status", None)
            or (execution.get("status") if isinstance(execution, dict) else "")
            or (execution.get("properties", {}).get("status") if isinstance(execution, dict) else "")
        )
        status_str = str(status).lower()

        if status_str in ("pending", "starting"):
            return BackendStatus(AttemptState.PENDING)

        if status_str == "running":
            return BackendStatus(AttemptState.RUNNING)

        if status_str == "succeeded":
            return BackendStatus(AttemptState.SUCCEEDED, exit_code=0)

        if status_str in ("failed", "degraded", "stopped", "cancelled"):
            # Check for spot preemption or exit code 143
            msg = getattr(execution, "message", "") or (
                execution.get("message", "") if isinstance(execution, dict) else ""
            )
            exit_code = getattr(execution, "exit_code", None) or (
                execution.get("exit_code") if isinstance(execution, dict) else None
            )
            is_spot = (
                getattr(execution, "spot", False)
                or (isinstance(execution, dict) and execution.get("spot", False))
                or "spot" in str(msg).lower()
            )
            if (exit_code == 143 and is_spot) or any(
                term in str(msg).lower() for term in ("spot", "preempt", "evict", "reclaim")
            ):
                return BackendStatus(AttemptState.PREEMPTED, message=str(msg) or "Container Apps spot preemption")
            code = exit_code if exit_code is not None else 1
            return BackendStatus(
                AttemptState.FAILED, exit_code=code, message=str(msg) or f"job execution failed (exit {code})"
            )

        return BackendStatus(AttemptState.UNKNOWN, message=f"unknown status: {status}")

    def cancel(self, handle: str) -> None:
        rg, job_name, exec_name = self._parse_handle(handle)
        jobs_api = getattr(self.client, "jobs", self.client)
        stop_fn = getattr(jobs_api, "begin_stop_execution", getattr(jobs_api, "stop_execution", None))
        if stop_fn is None:
            stop_fn = getattr(jobs_api, "begin_stop", getattr(jobs_api, "stop", None))
        if stop_fn is None:
            raise RuntimeError("Azure Container Apps stop API not available")
        poller = stop_fn(resource_group_name=rg, job_name=job_name, job_execution_name=exec_name)
        if hasattr(poller, "result"):
            poller.result()

    def logs(self, handle: str, tail: Optional[int] = None) -> str:
        rg, job_name, exec_name = self._parse_handle(handle)
        log_lines: List[str] = []

        # 1. Query Azure Monitor Log Analytics workspace if configured
        if self.logs_client is not None and self.workspace_id:
            try:
                query = (
                    f"ContainerAppConsoleLogs_CL "
                    f"| where ContainerJobName_s == '{job_name}' and ExecutionName_s == '{exec_name}' "
                    f"| order by TimeGenerated asc "
                    f"| project Log_s"
                )
                response = self.logs_client.query_workspace(
                    workspace_id=self.workspace_id,
                    query=query,
                )
                tables = getattr(response, "tables", [])
                for table in tables:
                    rows = getattr(table, "rows", [])
                    for row in rows:
                        val = row[0] if isinstance(row, (list, tuple)) and row else row
                        if val is not None:
                            log_lines.append(str(val))
            except Exception:
                pass

        # 2. In-process test fakes fallback (if jobs_api provides get_log_stream)
        if not log_lines:
            try:
                jobs_api = getattr(self.client, "job_executions", getattr(self.client, "jobs", self.client))
                if hasattr(jobs_api, "get_log_stream"):
                    stream = jobs_api.get_log_stream(
                        resource_group_name=rg, job_name=job_name, job_execution_name=exec_name
                    )
                    data = (
                        stream.read()
                        if hasattr(stream, "read")
                        else bytes(stream)
                        if isinstance(stream, (bytes, bytearray))
                        else str(stream)
                    )
                    if isinstance(data, bytes):
                        data = data.decode("utf-8", errors="replace")
                    log_lines = data.splitlines()
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
        return price_per_hour(machine_type or "container_apps", region or self.region, spot)


__all__ = ["ContainerAppsJobsBackend"]

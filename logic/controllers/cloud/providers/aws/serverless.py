"""AWS ECS Fargate compute backend implementation.

Attributes:
    EcsFargateBackend: :class:`~logic.controllers.cloud.interfaces.ComputeBackend` for AWS ECS Fargate.
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
from logic.controllers.cloud.providers.aws._sdk import make_boto3_client
from logic.controllers.cloud.providers.aws.pricing import price_per_hour


class EcsFargateBackend(ComputeBackend):
    """AWS ECS Fargate serverless container backend.

    Args:
        client: Pre-configured boto3 ECS client (injected for testing).
        logs_client: Optional pre-configured boto3 CloudWatch Logs client.
        cluster: ECS cluster name.
        task_definition: ECS task definition name or family.
        container_name: Target container name within the task definition.
        profile: AWS credentials profile name.
        region: AWS region.
    """

    provider = Provider.AWS
    kind = ComputeKind.SERVERLESS

    def __init__(
        self,
        client: Any = None,
        logs_client: Any = None,
        cluster: str = "wsr-cluster",
        task_definition: str = "wsr-task-def",
        container_name: str = "runner",
        profile: str = "default",
        region: str = "us-east-1",
    ) -> None:
        self._client = client
        self._logs_client = logs_client
        self.cluster = cluster
        self.task_definition = task_definition
        self.container_name = container_name
        self.profile = profile
        self.region = region

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_boto3_client("ecs", profile=self.profile, region=self.region)
        return self._client

    @property
    def logs_client(self) -> Any:
        if self._logs_client is None:
            with contextlib.suppress(Exception):
                self._logs_client = make_boto3_client("logs", profile=self.profile, region=self.region)
        return self._logs_client

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            provider=Provider.AWS,
            kinds=[ComputeKind.SERVERLESS],
            max_wall_seconds=86400 * 7,
            supports_spot=True,
            regions=[self.region],
            machine_types={
                "fargate.1vcpu": {"cpus": 1.0, "memory_gb": 2.0},
                "fargate.2vcpu": {"cpus": 2.0, "memory_gb": 4.0},
                "fargate.4vcpu": {"cpus": 4.0, "memory_gb": 8.0},
            },
        )

    def _parse_handle(self, handle: str) -> tuple[str, str]:
        """Return (cluster, task_arn)."""
        if handle.startswith("aws-ecs:"):
            parts = handle.split(":", 3)
            if len(parts) >= 4:
                return parts[2], parts[3]
        return self.cluster, handle

    def submit(self, launch: LaunchSpec) -> str:
        # Fargate limits: up to 16 vCPUs, 120 GiB
        if launch.resources.cpus > 16 or launch.resources.memory_gb > 120:
            raise CapacityError(
                f"requested {launch.resources.cpus} vCPUs and {launch.resources.memory_gb} GiB exceeds ECS Fargate limits"
            )

        provider_name = "FARGATE_SPOT" if launch.resources.spot else "FARGATE"
        capacity_provider_strategy = [{"capacityProvider": provider_name, "weight": 1}]

        cpu_units = max(256, int(round(launch.resources.cpus * 1024)))
        memory_mb = max(512, int(round(launch.resources.memory_gb * 1024)))

        container_override: Dict[str, Any] = {
            "name": self.container_name,
            "command": list(launch.argv),
            "environment": [{"name": k, "value": str(v)} for k, v in sorted(launch.env.items())],
            "cpu": cpu_units,
            "memory": memory_mb,
        }

        # Native secret references
        if launch.secret_env:
            container_override["secrets"] = [
                {"name": k, "valueFrom": str(v)} for k, v in sorted(launch.secret_env.items())
            ]

        resp = self.client.run_task(
            cluster=self.cluster,
            taskDefinition=self.task_definition,
            capacityProviderStrategy=capacity_provider_strategy,
            overrides={"containerOverrides": [container_override]},
            tags=[
                {"key": "wsr:task_id", "value": launch.task.task_id},
                {"key": "wsr:job_id", "value": launch.task.job_id},
            ],
        )

        failures = resp.get("failures", [])
        if failures:
            reasons = "; ".join(f.get("reason", "unknown") for f in failures)
            if "capacity" in reasons.lower() or "limit" in reasons.lower() or "resource" in reasons.lower():
                raise CapacityError(f"ECS Fargate capacity unavailable: {reasons}")
            raise RuntimeError(f"ECS run_task failed: {reasons}")

        tasks = resp.get("tasks", [])
        if not tasks:
            raise RuntimeError("ECS run_task returned no tasks")

        task_arn = tasks[0]["taskArn"]
        return f"aws-ecs:{self.region}:{self.cluster}:{task_arn}"

    def poll(self, handle: str) -> BackendStatus:
        cluster, task_arn = self._parse_handle(handle)
        try:
            resp = self.client.describe_tasks(cluster=cluster, tasks=[task_arn])
        except Exception as exc:
            return BackendStatus(AttemptState.UNKNOWN, message=str(exc))

        tasks = resp.get("tasks", [])
        if not tasks:
            return BackendStatus(AttemptState.UNKNOWN, message="no such attempt")

        task = tasks[0]
        status = task.get("lastStatus", "")
        stop_code = task.get("stopCode", "")
        stopped_reason = task.get("stoppedReason", "")
        containers = task.get("containers", [])
        exit_code = containers[0].get("exitCode") if containers else None

        if status in ("PROVISIONING", "PENDING", "ACTIVATING"):
            return BackendStatus(AttemptState.PENDING, message=stopped_reason)

        if status in ("RUNNING", "DEACTIVATING"):
            return BackendStatus(AttemptState.RUNNING, message=stopped_reason)

        if status == "STOPPED":
            # Detect spot preemption
            reason_lower = (stopped_reason or "").lower()
            if stop_code == "SpotInterruption" or any(
                term in reason_lower for term in ("spot", "preempt", "reclaim", "interruption")
            ):
                return BackendStatus(
                    AttemptState.PREEMPTED,
                    message=stopped_reason or "Fargate spot interruption",
                )

            if exit_code == 0:
                return BackendStatus(AttemptState.SUCCEEDED, exit_code=0)

            code = exit_code if exit_code is not None else 1
            return BackendStatus(
                AttemptState.FAILED,
                exit_code=code,
                message=stopped_reason or f"task stopped with exit code {code}",
            )

        return BackendStatus(AttemptState.UNKNOWN, message=f"unknown status: {status}")

    def cancel(self, handle: str) -> None:
        cluster, task_arn = self._parse_handle(handle)
        with contextlib.suppress(Exception):
            self.client.stop_task(
                cluster=cluster,
                task=task_arn,
                reason="Cancelled by WSmart-Route",
            )

    def logs(self, handle: str, tail: Optional[int] = None) -> str:
        cluster, task_arn = self._parse_handle(handle)
        log_lines: List[str] = []

        try:
            resp = self.client.describe_tasks(cluster=cluster, tasks=[task_arn])
            tasks = resp.get("tasks", [])
            if tasks:
                task = tasks[0]
                if "_log_lines" in task:
                    log_lines = list(task["_log_lines"])
                elif self.logs_client is not None:
                    # CloudWatch logs integration
                    task_id = task_arn.split("/")[-1]
                    stream_name = f"ecs/{self.container_name}/{task_id}"
                    log_resp = self.logs_client.get_log_events(
                        logGroupName="/ecs/wsr",
                        logStreamName=stream_name,
                    )
                    log_lines = [ev.get("message", "") for ev in log_resp.get("events", [])]
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
        return price_per_hour(machine_type or "fargate", region or self.region, spot)


__all__ = ["EcsFargateBackend"]

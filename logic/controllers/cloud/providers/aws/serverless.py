"""AWS ECS Fargate compute backend implementation.

Attributes:
    EcsFargateBackend: :class:`~logic.controllers.cloud.interfaces.ComputeBackend` for AWS ECS Fargate.
"""

from __future__ import annotations

import contextlib
import hashlib
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
from logic.controllers.cloud.providers.aws.secrets import AwsSecretsManagerStore


class EcsFargateBackend(ComputeBackend):
    """AWS ECS Fargate serverless container backend.

    Args:
        client: Pre-configured boto3 ECS client (injected for testing).
        logs_client: Optional pre-configured boto3 CloudWatch Logs client.
        cluster: ECS cluster name.
        task_definition: Default ECS task definition family name prefix.
        container_name: Target container name within the task definition.
        subnets: VPC subnet IDs for Fargate awsvpc network configuration.
        security_groups: VPC security group IDs.
        execution_role_arn: IAM role ARN for pulling images and fetching Secrets Manager secrets.
        secrets_store: Optional AwsSecretsManagerStore for resolving secret ARNs.
        credentials: Optional ProviderCredentials instance or profile name.
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
        subnets: Optional[List[str]] = None,
        security_groups: Optional[List[str]] = None,
        execution_role_arn: Optional[str] = None,
        secrets_store: Optional[AwsSecretsManagerStore] = None,
        credentials: Any = None,
        profile: str = "default",
        region: str = "us-east-1",
    ) -> None:
        self._client = client
        self._logs_client = logs_client
        self.cluster = cluster
        self.task_definition = task_definition
        self.container_name = container_name
        self.subnets = subnets or ["subnet-default-1", "subnet-default-2"]
        self.security_groups = security_groups or ["sg-default"]
        self.execution_role_arn = execution_role_arn
        self.credentials = credentials
        self.profile = profile
        self.region = region
        self._secrets_store = secrets_store
        self._secret_arn_cache: Dict[str, str] = {}
        self._task_def_cache: Dict[str, str] = {}

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_boto3_client(
                "ecs",
                credentials=self.credentials,
                profile=self.profile,
                region=self.region,
            )
        return self._client

    @property
    def logs_client(self) -> Any:
        if self._logs_client is None:
            with contextlib.suppress(Exception):
                self._logs_client = make_boto3_client(
                    "logs",
                    credentials=self.credentials,
                    profile=self.profile,
                    region=self.region,
                )
        return self._logs_client

    @property
    def secrets_store(self) -> AwsSecretsManagerStore:
        if self._secrets_store is None:
            self._secrets_store = AwsSecretsManagerStore(
                credentials=self.credentials,
                profile=self.profile,
                region=self.region,
            )
        return self._secrets_store

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

    def _resolve_secret_arn(self, secret_name: str) -> str:
        if secret_name.startswith("arn:aws:secretsmanager:"):
            return secret_name
        if secret_name in self._secret_arn_cache:
            return self._secret_arn_cache[secret_name]
        arn = self.secrets_store.get_arn(secret_name)
        self._secret_arn_cache[secret_name] = arn
        return arn

    def _get_or_register_task_definition(self, launch: LaunchSpec, cpu_units: int, memory_mb: int) -> str:
        """Register or reuse a task definition per (image, sorted secrets, cpu, memory)."""
        image = launch.image or "wsmart-route:latest"
        secret_items = tuple(sorted(launch.secret_env.items())) if launch.secret_env else ()
        cache_key = f"{image}:{secret_items}:{cpu_units}:{memory_mb}"
        if cache_key in self._task_def_cache:
            return self._task_def_cache[cache_key]

        key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:12]
        family_name = f"wsr-{key_hash}"

        # Check existing
        if hasattr(self.client, "describe_task_definition"):
            try:
                desc = self.client.describe_task_definition(taskDefinition=family_name)
                td = desc.get("taskDefinition", {})
                if td.get("status") == "ACTIVE":
                    arn = td.get("taskDefinitionArn", family_name)
                    self._task_def_cache[cache_key] = arn
                    return arn
            except Exception:
                pass

        secrets_list = []
        for env_var, sec_name in secret_items:
            sec_arn = self._resolve_secret_arn(sec_name)
            secrets_list.append({"name": env_var, "valueFrom": sec_arn})

        container_def: Dict[str, Any] = {
            "name": self.container_name,
            "image": image,
            "essential": True,
        }
        if secrets_list:
            container_def["secrets"] = secrets_list

        task_def_kwargs: Dict[str, Any] = {
            "family": family_name,
            "requiresCompatibilities": ["FARGATE"],
            "networkMode": "awsvpc",
            "cpu": str(cpu_units),
            "memory": str(memory_mb),
            "containerDefinitions": [container_def],
        }
        if self.execution_role_arn:
            task_def_kwargs["executionRoleArn"] = self.execution_role_arn

        if hasattr(self.client, "register_task_definition"):
            reg_resp = self.client.register_task_definition(**task_def_kwargs)
            arn = reg_resp.get("taskDefinition", {}).get("taskDefinitionArn", family_name)
            self._task_def_cache[cache_key] = arn
            return arn

        self._task_def_cache[cache_key] = family_name
        return family_name

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

        task_def = self._get_or_register_task_definition(launch, cpu_units, memory_mb)

        # Fargate requires awsvpc networkConfiguration
        network_config = {
            "awsvpcConfiguration": {
                "subnets": self.subnets,
                "securityGroups": self.security_groups,
                "assignPublicIp": "ENABLED",
            }
        }

        # containerOverrides has ONLY command and environment (NO secrets or image)
        container_override: Dict[str, Any] = {
            "name": self.container_name,
            "command": list(launch.argv),
            "environment": [{"name": k, "value": str(v)} for k, v in sorted(launch.env.items())],
        }

        resp = self.client.run_task(
            cluster=self.cluster,
            taskDefinition=task_def,
            capacityProviderStrategy=capacity_provider_strategy,
            networkConfiguration=network_config,
            overrides={"containerOverrides": [container_override]},
            tags=[
                {"key": "wsr:task_id", "value": launch.task.task_id},
                {"key": "wsr:job_id", "value": launch.task.job_id},
                {"key": "wsr:spot", "value": "true" if launch.resources.spot else "false"},
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

        t = tasks[0]
        last_status = t.get("lastStatus", "")
        stop_code = t.get("stopCode", "")
        stopped_reason = t.get("stoppedReason", "")

        containers = t.get("containers", [])
        exit_code = containers[0].get("exitCode") if containers else None

        # Check spot tag / capacity provider
        is_spot = (
            any(tag.get("key") == "wsr:spot" and tag.get("value") == "true" for tag in t.get("tags", []))
            or t.get("capacityProviderName") == "FARGATE_SPOT"
        )

        if last_status in ("PROVISIONING", "PENDING", "ACTIVATING"):
            return BackendStatus(AttemptState.PENDING, message=last_status)

        if last_status == "RUNNING":
            return BackendStatus(AttemptState.RUNNING)

        if last_status == "STOPPED":
            # Check spot interruption or exit code 143 on spot
            if stop_code == "SpotInterruption" or (exit_code == 143 and is_spot):
                return BackendStatus(AttemptState.PREEMPTED, message=stopped_reason or "Fargate spot interruption")

            if exit_code == 0:
                return BackendStatus(AttemptState.SUCCEEDED, exit_code=0)

            code = exit_code if exit_code is not None else 1
            return BackendStatus(
                AttemptState.FAILED,
                exit_code=code,
                message=stopped_reason or f"task stopped with code {code}",
            )

        return BackendStatus(AttemptState.UNKNOWN, message=f"unknown status: {last_status}")

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
        task_id = task_arn.split("/")[-1]
        log_lines: List[str] = []

        if self.logs_client is not None:
            try:
                log_resp = self.logs_client.get_log_events(
                    logGroupName="/ecs/wsr",
                    logStreamName=f"runner/{self.container_name}/{task_id}",
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
        return price_per_hour(machine_type, region or self.region, spot)


__all__ = ["EcsFargateBackend"]

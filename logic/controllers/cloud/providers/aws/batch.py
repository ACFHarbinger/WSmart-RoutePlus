"""AWS Batch compute backend implementation.

Attributes:
    AwsBatchBackend: :class:`~logic.controllers.cloud.interfaces.ComputeBackend` for AWS Batch.
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


class AwsBatchBackend(ComputeBackend):
    """AWS Batch compute backend.

    Args:
        client: Pre-configured boto3 Batch client (injected for testing).
        logs_client: Optional pre-configured boto3 CloudWatch Logs client.
        job_queue: AWS Batch job queue name.
        job_queue_spot: Optional AWS Batch job queue for spot instances.
        job_definition: AWS Batch job definition name prefix.
        execution_role_arn: IAM role ARN for ECS/Batch container task execution.
        secrets_store: Optional AwsSecretsManagerStore for resolving secret ARNs.
        credentials: Optional ProviderCredentials instance or profile name.
        profile: AWS credentials profile name.
        region: AWS region.
    """

    provider = Provider.AWS
    kind = ComputeKind.BATCH

    def __init__(
        self,
        client: Any = None,
        logs_client: Any = None,
        job_queue: str = "wsr-batch-queue",
        job_queue_spot: Optional[str] = None,
        job_definition: str = "wsr-batch-job",
        execution_role_arn: Optional[str] = None,
        secrets_store: Optional[AwsSecretsManagerStore] = None,
        credentials: Any = None,
        profile: str = "default",
        region: str = "us-east-1",
    ) -> None:
        self._client = client
        self._logs_client = logs_client
        self.job_queue = job_queue
        self.job_queue_spot = job_queue_spot or f"{job_queue}-spot"
        self.job_definition = job_definition
        self.execution_role_arn = execution_role_arn
        self.credentials = credentials
        self.profile = profile
        self.region = region
        self._secrets_store = secrets_store
        self._secret_arn_cache: Dict[str, str] = {}
        self._job_def_cache: Dict[str, str] = {}

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_boto3_client(
                "batch",
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
            kinds=[ComputeKind.BATCH],
            max_wall_seconds=86400 * 7,
            supports_spot=True,
            regions=[self.region],
            machine_types={
                "c6i.large": {"cpus": 2.0, "memory_gb": 4.0},
                "c6i.xlarge": {"cpus": 4.0, "memory_gb": 8.0},
                "c6i.2xlarge": {"cpus": 8.0, "memory_gb": 16.0},
                "m6i.large": {"cpus": 2.0, "memory_gb": 8.0},
            },
        )

    def _parse_handle(self, handle: str) -> str:
        if handle.startswith("aws-batch:"):
            parts = handle.split(":")
            if len(parts) >= 3:
                return parts[2]
        return handle

    def _resolve_secret_arn(self, secret_name: str) -> str:
        if secret_name.startswith("arn:aws:secretsmanager:"):
            return secret_name
        if secret_name in self._secret_arn_cache:
            return self._secret_arn_cache[secret_name]
        arn = self.secrets_store.get_arn(secret_name)
        self._secret_arn_cache[secret_name] = arn
        return arn

    def _get_or_register_job_definition(self, launch: LaunchSpec) -> str:
        """Register or reuse a job definition per (image, sorted secret_env)."""
        image = launch.image or "wsmart-route:latest"
        secret_items = tuple(sorted(launch.secret_env.items())) if launch.secret_env else ()
        cache_key = f"{image}:{secret_items}"
        if cache_key in self._job_def_cache:
            return self._job_def_cache[cache_key]

        key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:12]
        def_name = f"wsr-{key_hash}"

        # Check if an active definition exists
        if hasattr(self.client, "describe_job_definitions"):
            try:
                desc = self.client.describe_job_definitions(jobDefinitionName=def_name, status="ACTIVE")
                defs = desc.get("jobDefinitions", [])
                if defs:
                    arn = defs[0].get("jobDefinitionArn", def_name)
                    self._job_def_cache[cache_key] = arn
                    return arn
            except Exception:
                pass

        # Register new job definition
        secrets_list = []
        for env_var, sec_name in secret_items:
            sec_arn = self._resolve_secret_arn(sec_name)
            secrets_list.append({"name": env_var, "valueFrom": sec_arn})

        container_properties: Dict[str, Any] = {
            "image": image,
            "resourceRequirements": [
                {"type": "VCPU", "value": str(max(1, int(round(launch.resources.cpus))))},
                {"type": "MEMORY", "value": str(max(512, int(round(launch.resources.memory_gb * 1024))))},
            ],
        }
        if secrets_list:
            container_properties["secrets"] = secrets_list
        if self.execution_role_arn:
            container_properties["executionRoleArn"] = self.execution_role_arn

        if hasattr(self.client, "register_job_definition"):
            reg_resp = self.client.register_job_definition(
                jobDefinitionName=def_name,
                type="container",
                containerProperties=container_properties,
            )
            arn = reg_resp.get("jobDefinitionArn", def_name)
            self._job_def_cache[cache_key] = arn
            return arn

        self._job_def_cache[cache_key] = def_name
        return def_name

    def submit(self, launch: LaunchSpec) -> str:
        # Enforce quota / capacity constraints
        if launch.resources.cpus > 128 or launch.resources.memory_gb > 512:
            raise CapacityError(
                f"requested {launch.resources.cpus} vCPUs and {launch.resources.memory_gb} GiB exceeds AWS Batch limit"
            )

        queue = self.job_queue_spot if launch.resources.spot else self.job_queue
        job_def = self._get_or_register_job_definition(launch)

        # In AWS Batch, containerOverrides ONLY supports command, environment, resourceRequirements
        # Image and secrets belong strictly in the job definition
        container_overrides: Dict[str, Any] = {
            "command": list(launch.argv),
            "environment": [{"name": k, "value": str(v)} for k, v in sorted(launch.env.items())],
            "resourceRequirements": [
                {"type": "VCPU", "value": str(max(1, int(round(launch.resources.cpus))))},
                {"type": "MEMORY", "value": str(max(512, int(round(launch.resources.memory_gb * 1024))))},
            ],
        }

        kwargs: Dict[str, Any] = {
            "jobName": f"wsr-{launch.task.task_id}",
            "jobQueue": queue,
            "jobDefinition": job_def,
            "containerOverrides": container_overrides,
            "parameters": {
                "taskId": launch.task.task_id,
                "jobId": launch.task.job_id,
                "spot": "true" if launch.resources.spot else "false",
            },
        }

        if launch.resources.max_wall_seconds is not None:
            kwargs["timeout"] = {
                "attemptDurationSeconds": max(60, int(round(launch.resources.max_wall_seconds))),
            }

        resp = self.client.submit_job(**kwargs)
        job_id = resp["jobId"]
        return f"aws-batch:{self.region}:{job_id}"

    def poll(self, handle: str) -> BackendStatus:
        job_id = self._parse_handle(handle)
        try:
            resp = self.client.describe_jobs(jobs=[job_id])
        except Exception as exc:
            return BackendStatus(AttemptState.UNKNOWN, message=str(exc))

        jobs = resp.get("jobs", [])
        if not jobs:
            return BackendStatus(AttemptState.UNKNOWN, message="no such attempt")

        job = jobs[0]
        status = job.get("status", "")
        status_reason = job.get("statusReason", "")
        container = job.get("container", {})
        exit_code = container.get("exitCode")

        if status in ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING"):
            return BackendStatus(AttemptState.PENDING, message=status_reason)

        if status == "RUNNING":
            return BackendStatus(AttemptState.RUNNING, message=status_reason)

        if status == "SUCCEEDED":
            return BackendStatus(AttemptState.SUCCEEDED, exit_code=0)

        if status == "FAILED":
            # Detect spot preemption: check statusReason text, or exit code 143 on spot capacity
            reason_lower = (status_reason or "").lower()
            is_spot = (
                job.get("parameters", {}).get("spot") == "true"
                or "spot" in job.get("jobQueue", "").lower()
            )
            is_preempted = (
                any(term in reason_lower for term in ("spot", "preempt", "reclaim", "host ec2 terminated"))
                or (exit_code == 143 and is_spot)
            )
            if is_preempted:
                return BackendStatus(AttemptState.PREEMPTED, message=status_reason or "spot preemption")

            code = exit_code if exit_code is not None else 1
            return BackendStatus(AttemptState.FAILED, exit_code=code, message=status_reason or "job failed")

        return BackendStatus(AttemptState.UNKNOWN, message=f"unknown status: {status}")

    def cancel(self, handle: str) -> None:
        job_id = self._parse_handle(handle)
        with contextlib.suppress(Exception):
            self.client.terminate_job(jobId=job_id, reason="Cancelled by WSmart-Route")

    def logs(self, handle: str, tail: Optional[int] = None) -> str:
        job_id = self._parse_handle(handle)
        log_lines: List[str] = []

        try:
            resp = self.client.describe_jobs(jobs=[job_id])
            jobs = resp.get("jobs", [])
            if jobs:
                job = jobs[0]
                container = job.get("container", {})
                stream_name = container.get("logStreamName")
                if stream_name and self.logs_client is not None:
                    log_resp = self.logs_client.get_log_events(
                        logGroupName="/aws/batch/job",
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
        return price_per_hour(machine_type, region or self.region, spot)


__all__ = ["AwsBatchBackend"]

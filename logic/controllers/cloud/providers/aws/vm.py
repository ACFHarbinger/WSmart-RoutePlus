"""AWS EC2 virtual machine compute backend implementation.

Attributes:
    Ec2Backend: :class:`~logic.controllers.cloud.interfaces.ComputeBackend` for AWS EC2 instances.
"""

from __future__ import annotations

import base64
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
from logic.controllers.cloud.providers.aws._sdk import make_boto3_client
from logic.controllers.cloud.providers.aws.pricing import price_per_hour


class Ec2Backend(ComputeBackend):
    """AWS EC2 instance compute backend.

    Args:
        client: Pre-configured boto3 EC2 client (injected for testing).
        ssm_client: Optional pre-configured boto3 SSM client.
        image_id: AMI ID for WSmart-Route runner VM.
        default_instance_type: Default EC2 instance type.
        profile: AWS credentials profile name.
        region: AWS region.
    """

    provider = Provider.AWS
    kind = ComputeKind.VM

    def __init__(
        self,
        client: Any = None,
        ssm_client: Any = None,
        image_id: str = "ami-wsr-latest",
        default_instance_type: str = "c6i.large",
        profile: str = "default",
        region: str = "us-east-1",
    ) -> None:
        self._client = client
        self._ssm_client = ssm_client
        self.image_id = image_id
        self.default_instance_type = default_instance_type
        self.profile = profile
        self.region = region

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_boto3_client("ec2", profile=self.profile, region=self.region)
        return self._client

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            provider=Provider.AWS,
            kinds=[ComputeKind.VM],
            max_wall_seconds=86400 * 14,
            supports_spot=True,
            regions=[self.region],
            machine_types={
                "c6i.large": {"cpus": 2.0, "memory_gb": 4.0},
                "c6i.xlarge": {"cpus": 4.0, "memory_gb": 8.0},
                "c6i.2xlarge": {"cpus": 8.0, "memory_gb": 16.0},
                "m6i.large": {"cpus": 2.0, "memory_gb": 8.0},
                "t3.medium": {"cpus": 2.0, "memory_gb": 4.0},
            },
        )

    def _parse_handle(self, handle: str) -> str:
        if handle.startswith("aws-ec2:"):
            parts = handle.split(":")
            if len(parts) >= 3:
                return parts[2]
        return handle

    def submit(self, launch: LaunchSpec) -> str:
        instance_type = launch.resources.machine_type or self.default_instance_type

        # Verify capacity
        if launch.resources.cpus > 128 or launch.resources.memory_gb > 512:
            raise CapacityError(f"requested resources exceed EC2 capabilities: {launch.resources}")

        kwargs: Dict[str, Any] = {
            "ImageId": self.image_id,
            "InstanceType": instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "wsr:task_id", "Value": launch.task.task_id},
                        {"Key": "wsr:job_id", "Value": launch.task.job_id},
                        {"Key": "wsr:attempt", "Value": str(launch.task.attempt)},
                    ],
                }
            ],
        }

        # Spot market options
        if launch.resources.spot:
            kwargs["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {"SpotInstanceType": "one-time"},
            }

        # Embed launch bootstrap into UserData script
        user_data_payload = {
            "argv": list(launch.argv),
            "env": dict(launch.env),
            "secret_env": dict(launch.secret_env),
            "image": launch.image,
            "task_id": launch.task.task_id,
            "job_id": launch.task.job_id,
        }
        user_data_script = f"#!/bin/bash\n# WSR-LAUNCH-SPEC\n# {json.dumps(user_data_payload)}\n"
        kwargs["UserData"] = base64.b64encode(user_data_script.encode("utf-8")).decode("utf-8")

        try:
            resp = self.client.run_instances(**kwargs)
        except Exception as exc:
            msg = str(exc)
            if any(term in msg.lower() for term in ("insufficientcapacity", "spotmaxpriceexceeded", "limitexceeded")):
                raise CapacityError(f"EC2 capacity unavailable: {msg}") from exc
            raise

        instances = resp.get("Instances", [])
        if not instances:
            raise RuntimeError("EC2 run_instances returned no instances")

        instance_id = instances[0]["InstanceId"]
        return f"aws-ec2:{self.region}:{instance_id}"

    def poll(self, handle: str) -> BackendStatus:
        instance_id = self._parse_handle(handle)
        try:
            resp = self.client.describe_instances(InstanceIds=[instance_id])
        except Exception as exc:
            return BackendStatus(AttemptState.UNKNOWN, message=str(exc))

        reservations = resp.get("Reservations", [])
        if not reservations or not reservations[0].get("Instances"):
            return BackendStatus(AttemptState.UNKNOWN, message="no such attempt")

        inst = reservations[0]["Instances"][0]
        state_info = inst.get("State", {})
        state_name = state_info.get("Name", "")
        state_reason = inst.get("StateReason", {})
        reason_code = state_reason.get("Code", "")
        reason_msg = state_reason.get("Message", "")

        # Check in-memory/test status override
        if "_status" in inst:
            return inst["_status"]

        if state_name == "pending":
            return BackendStatus(AttemptState.PENDING, message=reason_msg)

        if state_name == "running":
            # If instance recorded a completion tag, check it
            tags = {t.get("Key"): t.get("Value") for t in inst.get("Tags", [])}
            if tags.get("wsr:status") == "succeeded":
                return BackendStatus(AttemptState.SUCCEEDED, exit_code=0)
            if tags.get("wsr:status") == "failed":
                exit_code = int(tags.get("wsr:exit_code", "1"))
                return BackendStatus(AttemptState.FAILED, exit_code=exit_code, message=tags.get("wsr:reason"))
            return BackendStatus(AttemptState.RUNNING, message=reason_msg)

        if state_name in ("shutting-down", "terminated", "stopped"):
            # Check for spot interruption
            if "SpotInstanceTermination" in reason_code or "spot" in reason_msg.lower():
                return BackendStatus(AttemptState.PREEMPTED, message=reason_msg or "EC2 spot termination")

            tags = {t.get("Key"): t.get("Value") for t in inst.get("Tags", [])}
            if tags.get("wsr:status") == "succeeded":
                return BackendStatus(AttemptState.SUCCEEDED, exit_code=0)

            exit_code = int(tags.get("wsr:exit_code", "1")) if "wsr:exit_code" in tags else 1
            return BackendStatus(
                AttemptState.FAILED,
                exit_code=exit_code,
                message=reason_msg or f"instance {state_name}",
            )

        return BackendStatus(AttemptState.UNKNOWN, message=f"unknown EC2 state: {state_name}")

    def cancel(self, handle: str) -> None:
        instance_id = self._parse_handle(handle)
        with contextlib.suppress(Exception):
            self.client.terminate_instances(InstanceIds=[instance_id])

    def logs(self, handle: str, tail: Optional[int] = None) -> str:
        instance_id = self._parse_handle(handle)
        log_lines: List[str] = []

        try:
            # Check if instance object has _log_lines attached in tests
            resp = self.client.describe_instances(InstanceIds=[instance_id])
            reservations = resp.get("Reservations", [])
            if reservations and reservations[0].get("Instances"):
                inst = reservations[0]["Instances"][0]
                if "_log_lines" in inst:
                    log_lines = list(inst["_log_lines"])

            if not log_lines:
                console_resp = self.client.get_console_output(InstanceId=instance_id)
                output = console_resp.get("Output", "")
                if output:
                    log_lines = output.splitlines()
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
        return price_per_hour(machine_type or self.default_instance_type, region or self.region, spot)


__all__ = ["Ec2Backend"]

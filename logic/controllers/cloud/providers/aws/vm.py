"""AWS EC2 virtual machine compute backend implementation.

Attributes:
    Ec2Backend: :class:`~logic.controllers.cloud.interfaces.ComputeBackend` for AWS EC2 instances.
"""

from __future__ import annotations

import base64
import contextlib
import json
import shlex
from typing import Any, Dict, Optional

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
        image_id: AMI ID for WSmart-Route runner VM.
        default_instance_type: Default EC2 instance type.
        credentials: Optional ProviderCredentials instance or profile name.
        profile: AWS credentials profile name.
        region: AWS region.
    """

    provider = Provider.AWS
    kind = ComputeKind.VM

    def __init__(
        self,
        client: Any = None,
        image_id: str = "ami-wsr-latest",
        default_instance_type: str = "c6i.large",
        credentials: Any = None,
        profile: str = "default",
        region: str = "us-east-1",
    ) -> None:
        self._client = client
        self.image_id = image_id
        self.default_instance_type = default_instance_type
        self.credentials = credentials
        self.profile = profile
        self.region = region

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_boto3_client(
                "ec2",
                credentials=self.credentials,
                profile=self.profile,
                region=self.region,
            )
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

    def _build_user_data(self, launch: LaunchSpec) -> str:
        """Construct UserData bootstrap script executing the task in Docker and shutting down."""
        env = dict(launch.env)
        env.setdefault("WSR_JOB_ID", launch.task.job_id)
        env.setdefault("WSR_TASK_ID", launch.task.task_id)

        # Inject secrets via WSR_SECRET_REFS (common secret path)
        if launch.secret_env:
            secret_refs = {
                var: f"awssm://{self.region}/{sec_name}" for var, sec_name in sorted(launch.secret_env.items())
            }
            env["WSR_SECRET_REFS"] = json.dumps(secret_refs)

        # Format docker run -e arguments
        env_flags = " ".join(f"-e {shlex.quote(f'{k}={v}')}" for k, v in sorted(env.items()))
        image = launch.image or "wsmart-route:latest"
        cmd = " ".join(shlex.quote(arg) for arg in launch.argv)

        registry = image.split("/", 1)[0] if "/" in image else ""
        ecr_login = ""
        if ".dkr.ecr." in registry and ".amazonaws.com" in registry:
            ecr_login = (
                f"aws ecr get-login-password --region {shlex.quote(self.region)} "
                f"| docker login --username AWS --password-stdin {shlex.quote(registry)}"
            )

        script = f"""#!/bin/bash
set -euo pipefail
# Whatever happens below, stop the instance (InstanceInitiatedShutdownBehavior=terminate deletes it).
trap 'shutdown -h now' EXIT

# Ensure docker is installed and running
if ! command -v docker >/dev/null 2>&1; then
    (apt-get update && apt-get install -y docker.io) || yum install -y docker || true
fi
systemctl start docker || service docker start || true

# Query EC2 Instance ID
TOKEN=$(curl -s -m 5 -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 60" || echo "")
if [ -n "$TOKEN" ]; then
    INSTANCE_ID=$(curl -s -m 5 -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id || echo "")
else
    INSTANCE_ID=$(curl -s -m 5 http://169.254.169.254/latest/meta-data/instance-id || echo "")
fi

# Log in to ECR when the image is there (the instance role needs ecr:GetAuthorizationToken)
{ecr_login}

# Run the container image
EXIT_CODE=0
docker run --rm {env_flags} {shlex.quote(image)} {cmd} || EXIT_CODE=$?

# Record exit code via EC2 instance tag
if [ -n "$INSTANCE_ID" ]; then
    aws ec2 create-tags --region {shlex.quote(self.region)} --resources "$INSTANCE_ID" --tags Key=wsr:exit_code,Value="$EXIT_CODE" || true
fi

# The EXIT trap shuts the instance down.
"""
        return script

    def submit(self, launch: LaunchSpec) -> str:
        instance_type = launch.resources.machine_type or self.default_instance_type

        # Verify capacity
        if launch.resources.cpus > 128 or launch.resources.memory_gb > 512:
            raise CapacityError(f"requested resources exceed EC2 capabilities: {launch.resources}")

        user_data_script = self._build_user_data(launch)
        user_data_b64 = base64.b64encode(user_data_script.encode("utf-8")).decode("utf-8")

        kwargs: Dict[str, Any] = {
            "ImageId": self.image_id,
            "InstanceType": instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "InstanceInitiatedShutdownBehavior": "terminate",
            "UserData": user_data_b64,
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "wsr:task_id", "Value": launch.task.task_id},
                        {"Key": "wsr:job_id", "Value": launch.task.job_id},
                        {"Key": "wsr:attempt", "Value": str(launch.task.attempt)},
                        {"Key": "wsr:spot", "Value": "true" if launch.resources.spot else "false"},
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
            return BackendStatus(AttemptState.UNKNOWN, message="no such instance")

        instance = reservations[0]["Instances"][0]
        state = instance.get("State", {}).get("Name", "")
        state_reason = instance.get("StateReason", {}).get("Code", "")
        reason_msg = instance.get("StateReason", {}).get("Message", "")
        transition_reason = instance.get("StateTransitionReason", "")

        tags = {t.get("Key"): t.get("Value") for t in instance.get("Tags", [])}
        is_spot = tags.get("wsr:spot") == "true" or "InstanceMarketOptions" in instance

        if state == "pending":
            return BackendStatus(AttemptState.PENDING)

        if state == "running":
            return BackendStatus(AttemptState.RUNNING)

        if state in ("shutting-down", "terminated", "stopped"):
            # Check spot interruption
            if (
                state_reason == "Server.SpotInstanceTermination"
                or "spot" in transition_reason.lower()
                or "spot" in reason_msg.lower()
            ):
                return BackendStatus(
                    AttemptState.PREEMPTED,
                    message=reason_msg or "EC2 spot instance termination",
                )

            # Check recorded exit code tag
            exit_code_str = tags.get("wsr:exit_code")
            if exit_code_str is not None:
                try:
                    code = int(exit_code_str)
                    if code == 0:
                        return BackendStatus(AttemptState.SUCCEEDED, exit_code=0)
                    if code == 143 and is_spot:
                        return BackendStatus(
                            AttemptState.PREEMPTED,
                            message="EC2 spot preemption (exit code 143)",
                        )
                    return BackendStatus(
                        AttemptState.FAILED,
                        exit_code=code,
                        message=f"Container exited with code {code}",
                    )
                except ValueError:
                    pass

            return BackendStatus(
                AttemptState.FAILED,
                exit_code=1,
                message=reason_msg or f"instance terminated: {transition_reason}",
            )

        return BackendStatus(AttemptState.UNKNOWN, message=f"unknown EC2 state: {state}")

    def cancel(self, handle: str) -> None:
        instance_id = self._parse_handle(handle)
        with contextlib.suppress(Exception):
            self.client.terminate_instances(InstanceIds=[instance_id])

    def logs(self, handle: str, tail: Optional[int] = None) -> str:
        instance_id = self._parse_handle(handle)
        try:
            resp = self.client.get_console_output(InstanceId=instance_id)
            output = resp.get("Output", "")
            if output:
                lines = output.splitlines()
                if tail is not None:
                    return "\n".join(lines[-tail:])
                return output
        except Exception:
            pass
        return ""

    def price_per_hour(
        self,
        machine_type: Optional[str] = None,
        region: Optional[str] = None,
        spot: bool = False,
    ) -> Optional[float]:
        return price_per_hour(machine_type, region or self.region, spot)


__all__ = ["Ec2Backend"]

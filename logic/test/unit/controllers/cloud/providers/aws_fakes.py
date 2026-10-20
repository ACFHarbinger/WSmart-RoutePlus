"""In-process AWS client fakes for the AWS provider contract tests.

Fakes for the S3, Secrets Manager, CloudWatch Logs and Batch clients, and the
``make_test_launch`` / ``make_launch_for_outcome`` launch-spec helpers. Moved unchanged
from ``test_aws_contracts.py``.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List

from logic.controllers.cloud.interfaces import (
    LaunchSpec,
)
from logic.controllers.cloud.model import (
    CellSpec,
    Resources,
    TaskRecord,
)

# ---------------------------------------------------------------------------
# In-process Fakes
# ---------------------------------------------------------------------------


class FakeS3Client:
    """In-memory S3 client for contract testing."""

    def __init__(self) -> None:
        self.objects: Dict[str, bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: Any, **kwargs: Any) -> Dict[str, Any]:
        data = Body.read() if hasattr(Body, "read") else bytes(Body)
        self.objects[Key] = data
        return {}

    def get_object(self, Bucket: str, Key: str, **kwargs: Any) -> Dict[str, Any]:
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket: str, Key: str, **kwargs: Any) -> Dict[str, Any]:
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[Key])}

    def list_objects_v2(self, Bucket: str, Prefix: str = "", **kwargs: Any) -> Dict[str, Any]:
        contents = [{"Key": k} for k in sorted(self.objects.keys()) if k.startswith(Prefix)]
        return {"Contents": contents}

    def delete_object(self, Bucket: str, Key: str, **kwargs: Any) -> Dict[str, Any]:
        self.objects.pop(Key, None)
        return {}


class FakeSecretsManagerClient:
    """In-memory Secrets Manager client for contract testing."""

    def __init__(self) -> None:
        self.secrets: Dict[str, str] = {}

    def get_secret_value(self, SecretId: str, **kwargs: Any) -> Dict[str, Any]:
        if SecretId not in self.secrets:
            raise KeyError(f"ResourceNotFoundException: {SecretId}")
        return {"SecretString": self.secrets[SecretId]}

    def put_secret_value(self, SecretId: str, SecretString: str, **kwargs: Any) -> Dict[str, Any]:
        if SecretId not in self.secrets:
            raise KeyError(f"ResourceNotFoundException: {SecretId}")
        self.secrets[SecretId] = SecretString
        return {}

    def create_secret(self, Name: str, SecretString: str, **kwargs: Any) -> Dict[str, Any]:
        self.secrets[Name] = SecretString
        return {}

    def describe_secret(self, SecretId: str, **kwargs: Any) -> Dict[str, Any]:
        if SecretId not in self.secrets and not SecretId.startswith("arn:"):
            raise KeyError(f"ResourceNotFoundException: {SecretId}")
        return {"ARN": f"arn:aws:secretsmanager:us-east-1:123456789012:secret:{SecretId}"}

    def delete_secret(self, SecretId: str, **kwargs: Any) -> Dict[str, Any]:
        self.secrets.pop(SecretId, None)
        return {}


class FakeLogsClient:
    """In-memory CloudWatch Logs client for contract testing."""

    def __init__(self) -> None:
        self.streams: Dict[str, List[str]] = {}

    def get_log_events(self, logGroupName: str, logStreamName: str, **kwargs: Any) -> Dict[str, Any]:
        lines = self.streams.get(logStreamName, [f"log for {logStreamName}"])
        return {"events": [{"message": line} for line in lines]}


class FakeBatchClient:
    """In-memory AWS Batch client for contract testing.

    Rejects invalid API calls mirroring real AWS Batch behavior (e.g. secrets/image in containerOverrides).
    """

    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.submits: List[Dict[str, Any]] = []
        self.job_definitions: Dict[str, Dict[str, Any]] = {}

    def describe_job_definitions(self, jobDefinitionName: str, status: str = "ACTIVE", **kwargs: Any) -> Dict[str, Any]:
        defs = [
            d for name, d in self.job_definitions.items() if name == jobDefinitionName and d.get("status") == status
        ]
        return {"jobDefinitions": defs}

    def register_job_definition(
        self,
        jobDefinitionName: str,
        type: str,
        containerProperties: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        arn = f"arn:aws:batch:us-east-1:123456789012:job-definition/{jobDefinitionName}:1"
        data = {
            "jobDefinitionName": jobDefinitionName,
            "jobDefinitionArn": arn,
            "status": "ACTIVE",
            "type": type,
            "containerProperties": containerProperties,
        }
        self.job_definitions[jobDefinitionName] = data
        return {"jobDefinitionArn": arn, "jobDefinitionName": jobDefinitionName}

    def submit_job(
        self,
        jobName: str,
        jobQueue: str,
        jobDefinition: str,
        containerOverrides: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Strict API validation: real AWS Batch rejects image or secrets in containerOverrides
        if "image" in containerOverrides:
            raise ValueError("ParamValidationError: containerOverrides cannot contain 'image'")
        if "secrets" in containerOverrides:
            raise ValueError("ParamValidationError: containerOverrides cannot contain 'secrets'")

        job_id = f"batch-{len(self.jobs) + 1}"
        cmd_str = " ".join(containerOverrides.get("command", []))
        task_id = kwargs.get("parameters", {}).get("taskId", "")

        outcome = "succeed"
        if "fail" in task_id or "fail" in cmd_str:
            outcome = "fail"
        elif "hang" in task_id or "hang" in cmd_str:
            outcome = "hang"
        elif "preempt" in task_id or "preempt" in cmd_str:
            outcome = "preempt"
        elif "exit143" in task_id or "exit143" in cmd_str:
            outcome = "exit143"

        self.submits.append(
            {
                "jobName": jobName,
                "jobQueue": jobQueue,
                "jobDefinition": jobDefinition,
                "containerOverrides": containerOverrides,
                "kwargs": kwargs,
            }
        )
        self.jobs[job_id] = {
            "jobId": job_id,
            "jobQueue": jobQueue,
            "outcome": outcome,
            "cancelled": False,
            "polls": 0,
            "parameters": kwargs.get("parameters", {}),
            "logStreamName": f"stream-{job_id}",
        }
        return {"jobId": job_id}

    def describe_jobs(self, jobs: List[str]) -> Dict[str, Any]:
        result = []
        for jid in jobs:
            if jid in self.jobs:
                j = self.jobs[jid]
                j["polls"] += 1
                if j["cancelled"]:
                    status = "FAILED"
                    reason = "Cancelled by WSmart-Route"
                    exit_code = 143
                elif j["outcome"] == "hang":
                    status = "RUNNING"
                    reason = "Running simulation"
                    exit_code = None
                elif j["outcome"] == "preempt":
                    status = "FAILED"
                    reason = "Host EC2 (instance i-0123456789) terminated by spot reclaim"
                    exit_code = 137
                elif j["outcome"] == "exit143":
                    status = "FAILED"
                    reason = "Container received SIGTERM and exited 143"
                    exit_code = 143
                elif j["outcome"] == "fail":
                    status = "FAILED"
                    reason = "Container failed with exit code 1"
                    exit_code = 1
                else:
                    status = "SUCCEEDED"
                    reason = "Essential container exited"
                    exit_code = 0

                result.append(
                    {
                        "jobId": jid,
                        "jobQueue": j["jobQueue"],
                        "parameters": j["parameters"],
                        "status": status,
                        "statusReason": reason,
                        "container": {
                            "exitCode": exit_code,
                            "logStreamName": j["logStreamName"],
                        },
                    }
                )
        return {"jobs": result}

    def terminate_job(self, jobId: str, reason: str) -> Dict[str, Any]:
        if jobId in self.jobs:
            self.jobs[jobId]["cancelled"] = True
        return {}


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def make_test_launch(
    task_id: str = "task-001",
    command: str = "test_sim",
    cpus: float = 2.0,
    memory_gb: float = 4.0,
    spot: bool = False,
    secret_env: Dict[str, str] | None = None,
) -> LaunchSpec:
    return LaunchSpec(
        task=TaskRecord(
            task_id=task_id,
            job_id="job-001",
            cell=CellSpec(cell_id="c1", argv=[command]),
        ),
        argv=[command, "--days", "30"],
        env={"WSR_JOB_ID": "job-001"},
        secret_env=secret_env or {},
        image="wsmart-route:v1.1.0",
        resources=Resources(cpus=cpus, memory_gb=memory_gb, spot=spot),
    )


def make_launch_for_outcome(outcome: str) -> LaunchSpec:
    return make_test_launch(task_id=f"task-{outcome}", command=f"test_sim_{outcome}")

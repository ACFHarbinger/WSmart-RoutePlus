"""AWS provider implementation passes all cloud runner contract suites."""

from __future__ import annotations

import io
from typing import Any, Dict, List

import pytest
from logic.controllers.cloud.interfaces import (
    AttemptState,
    CapacityError,
    LaunchSpec,
)
from logic.controllers.cloud.model import (
    CellSpec,
    ComputeKind,
    Provider,
    Resources,
    TaskRecord,
)
from logic.controllers.cloud.providers.aws import (
    AwsBatchBackend,
    AwsSecretsManagerStore,
    Ec2Backend,
    EcrImageRegistry,
    EcsFargateBackend,
    S3ObjectStore,
    price_per_hour,
)
from logic.controllers.cloud.testing import (
    backend_contract,
    object_store_contract,
    secret_store_contract,
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

    def delete_secret(self, SecretId: str, **kwargs: Any) -> Dict[str, Any]:
        self.secrets.pop(SecretId, None)
        return {}


class FakeBatchClient:
    """In-memory AWS Batch client for contract testing."""

    def __init__(self) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.submits: List[Dict[str, Any]] = []

    def submit_job(
        self,
        jobName: str,
        jobQueue: str,
        jobDefinition: str,
        containerOverrides: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
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

        self.submits.append(
            {
                "jobName": jobName,
                "jobQueue": jobQueue,
                "containerOverrides": containerOverrides,
                "kwargs": kwargs,
            }
        )
        self.jobs[job_id] = {
            "jobId": job_id,
            "outcome": outcome,
            "cancelled": False,
            "polls": 0,
            "_log_lines": [f"batch log output for {jobName}"],
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
                        "status": status,
                        "statusReason": reason,
                        "container": {"exitCode": exit_code},
                        "_log_lines": j["_log_lines"],
                    }
                )
        return {"jobs": result}

    def terminate_job(self, jobId: str, reason: str) -> Dict[str, Any]:
        if jobId in self.jobs:
            self.jobs[jobId]["cancelled"] = True
        return {}


class FakeEcsClient:
    """In-memory AWS ECS client for contract testing."""

    def __init__(self) -> None:
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.runs: List[Dict[str, Any]] = []

    def run_task(
        self,
        cluster: str,
        taskDefinition: str,
        overrides: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        task_arn = f"arn:aws:ecs:us-east-1:123456789012:task/{cluster}/task-{len(self.tasks) + 1}"
        container_overrides = overrides.get("containerOverrides", [{}])[0]
        cmd_str = " ".join(container_overrides.get("command", []))

        outcome = "succeed"
        if "fail" in cmd_str:
            outcome = "fail"
        elif "hang" in cmd_str:
            outcome = "hang"
        elif "preempt" in cmd_str:
            outcome = "preempt"

        self.runs.append(
            {
                "cluster": cluster,
                "taskDefinition": taskDefinition,
                "overrides": overrides,
                "kwargs": kwargs,
            }
        )
        self.tasks[task_arn] = {
            "taskArn": task_arn,
            "outcome": outcome,
            "cancelled": False,
            "polls": 0,
            "_log_lines": ["ecs container log output line 1", "ecs container log output line 2"],
        }
        return {"tasks": [{"taskArn": task_arn}]}

    def describe_tasks(self, cluster: str, tasks: List[str]) -> Dict[str, Any]:
        result = []
        for arn in tasks:
            if arn in self.tasks:
                t = self.tasks[arn]
                t["polls"] += 1
                if t["cancelled"]:
                    status = "STOPPED"
                    code = "UserStop"
                    reason = "Cancelled by WSmart-Route"
                    exit_code = 143
                elif t["outcome"] == "hang":
                    status = "RUNNING"
                    code = ""
                    reason = ""
                    exit_code = None
                elif t["outcome"] == "preempt":
                    status = "STOPPED"
                    code = "SpotInterruption"
                    reason = "Fargate spot interruption"
                    exit_code = 137
                elif t["outcome"] == "fail":
                    status = "STOPPED"
                    code = "EssentialContainerExited"
                    reason = "Process exited with code 1"
                    exit_code = 1
                else:
                    status = "STOPPED"
                    code = "EssentialContainerExited"
                    reason = "Process exited cleanly"
                    exit_code = 0

                result.append(
                    {
                        "taskArn": arn,
                        "lastStatus": status,
                        "stopCode": code,
                        "stoppedReason": reason,
                        "containers": [{"exitCode": exit_code}],
                        "_log_lines": t["_log_lines"],
                    }
                )
        return {"tasks": result}

    def stop_task(self, cluster: str, task: str, reason: str) -> Dict[str, Any]:
        if task in self.tasks:
            self.tasks[task]["cancelled"] = True
        return {}


class FakeEc2Client:
    """In-memory AWS EC2 client for contract testing."""

    def __init__(self) -> None:
        self.instances: Dict[str, Dict[str, Any]] = {}
        self.runs: List[Dict[str, Any]] = []

    def run_instances(self, **kwargs: Any) -> Dict[str, Any]:
        inst_id = f"i-{len(self.instances) + 1:012d}"
        tags = {}
        for spec in kwargs.get("TagSpecifications", []):
            for t in spec.get("Tags", []):
                tags[t.get("Key")] = t.get("Value")
        task_id = tags.get("wsr:task_id", "")

        user_data_b64 = kwargs.get("UserData", "")
        try:
            import base64
            user_data = base64.b64decode(user_data_b64).decode("utf-8")
        except Exception:
            user_data = ""

        outcome = "succeed"
        if "fail" in task_id or "fail" in user_data:
            outcome = "fail"
        elif "hang" in task_id or "hang" in user_data:
            outcome = "hang"
        elif "preempt" in task_id or "preempt" in user_data:
            outcome = "preempt"

        self.runs.append(kwargs)
        self.instances[inst_id] = {
            "InstanceId": inst_id,
            "outcome": outcome,
            "cancelled": False,
            "polls": 0,
            "_log_lines": ["ec2 console output row 1", "ec2 console output row 2"],
        }
        return {"Instances": [{"InstanceId": inst_id}]}

    def describe_instances(self, InstanceIds: List[str]) -> Dict[str, Any]:
        reservations = []
        for iid in InstanceIds:
            if iid in self.instances:
                inst = self.instances[iid]
                inst["polls"] += 1
                if inst["cancelled"]:
                    state = "terminated"
                    code = "Client.UserInitiatedShutdown"
                    msg = "Cancelled by WSmart-Route"
                    status_tag = "failed"
                    exit_code = "143"
                elif inst["outcome"] == "hang":
                    state = "running"
                    code = ""
                    msg = ""
                    status_tag = "running"
                    exit_code = ""
                elif inst["outcome"] == "preempt":
                    state = "terminated"
                    code = "Server.SpotInstanceTermination"
                    msg = "Instance interrupted by spot price"
                    status_tag = "preempted"
                    exit_code = "137"
                elif inst["outcome"] == "fail":
                    state = "terminated"
                    code = "Client.InstanceTermination"
                    msg = "Container exited with code 1"
                    status_tag = "failed"
                    exit_code = "1"
                else:
                    state = "terminated"
                    code = "Client.InstanceTermination"
                    msg = "Container finished successfully"
                    status_tag = "succeeded"
                    exit_code = "0"

                reservations.append(
                    {
                        "Instances": [
                            {
                                "InstanceId": iid,
                                "State": {"Name": state},
                                "StateReason": {"Code": code, "Message": msg},
                                "Tags": [
                                    {"Key": "wsr:status", "Value": status_tag},
                                    {"Key": "wsr:exit_code", "Value": exit_code},
                                ],
                                "_log_lines": inst["_log_lines"],
                            }
                        ]
                    }
                )
        return {"Reservations": reservations}

    def terminate_instances(self, InstanceIds: List[str]) -> Dict[str, Any]:
        for iid in InstanceIds:
            if iid in self.instances:
                self.instances[iid]["cancelled"] = True
        return {}

    def get_console_output(self, InstanceId: str) -> Dict[str, Any]:
        if InstanceId in self.instances:
            lines = self.instances[InstanceId]["_log_lines"]
            return {"Output": "\n".join(lines)}
        return {"Output": ""}


class FakeEcrClient:
    """In-memory ECR client for contract testing."""

    def __init__(self) -> None:
        self.images: Dict[str, str] = {}

    def describe_images(self, repositoryName: str, imageIds: List[Dict[str, str]]) -> Dict[str, Any]:
        tag = imageIds[0].get("imageTag")
        if tag in self.images:
            return {"imageDetails": [{"imageDigest": self.images[tag]}]}
        raise KeyError(f"ImageNotFoundException: {tag}")

    def put_image(self, repositoryName: str, imageManifest: str, imageTag: str) -> Dict[str, Any]:
        digest = f"sha256:{imageTag}-digest"
        self.images[imageTag] = digest
        return {"image": {"imageDigest": digest}}


# ---------------------------------------------------------------------------
# Launch helper
# ---------------------------------------------------------------------------


def _launch(outcome: str) -> LaunchSpec:
    task = TaskRecord(
        task_id=f"task-{outcome}",
        job_id="job-aws",
        cell=CellSpec(cell_id=outcome, argv=["test_sim", f"sim.outcome={outcome}"]),
    )
    return LaunchSpec(
        task=task,
        resources=Resources(cpus=1.0, memory_gb=2.0),
        argv=["python", "main.py", "test_sim", f"outcome={outcome}"],
        env={"TEST_VAR": "val"},
        secret_env={"GUROBI_KEY": "gurobi-secret-id"},
    )


# ---------------------------------------------------------------------------
# Contract Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_aws_s3_object_store_contract():
    client = FakeS3Client()
    object_store_contract(lambda: S3ObjectStore(bucket="wsr-test-bucket", client=client))


@pytest.mark.unit
def test_aws_secrets_manager_contract():
    client = FakeSecretsManagerClient()
    secret_store_contract(lambda: AwsSecretsManagerStore(client=client))


@pytest.mark.unit
def test_aws_batch_backend_contract():
    client = FakeBatchClient()
    backend_contract(
        make=lambda: AwsBatchBackend(client=client),
        launch=_launch,
        poll_seconds=0.0,
    )


@pytest.mark.unit
def test_aws_batch_preemption_and_capacity():
    client = FakeBatchClient()
    backend = AwsBatchBackend(client=client)

    # Spot preemption detection
    handle = backend.submit(_launch("preempt"))
    status = backend.poll(handle)
    assert status.state == AttemptState.PREEMPTED
    assert "spot" in (status.message or "").lower()

    # Secret environment variable pass-through
    assert len(client.submits) > 0
    overrides = client.submits[-1]["containerOverrides"]
    assert overrides["secrets"] == [{"name": "GUROBI_KEY", "valueFrom": "gurobi-secret-id"}]

    # Capacity error on oversized resource request
    excessive_launch = _launch("succeed")
    excessive_launch.resources = Resources(cpus=256.0, memory_gb=1024.0)
    with pytest.raises(CapacityError):
        backend.submit(excessive_launch)


@pytest.mark.unit
def test_aws_ecs_fargate_backend_contract():
    client = FakeEcsClient()
    backend_contract(
        make=lambda: EcsFargateBackend(client=client),
        launch=_launch,
        poll_seconds=0.0,
    )


@pytest.mark.unit
def test_aws_ecs_fargate_spot_and_capacity():
    client = FakeEcsClient()
    backend = EcsFargateBackend(client=client)

    # Spot preemption detection
    handle = backend.submit(_launch("preempt"))
    status = backend.poll(handle)
    assert status.state == AttemptState.PREEMPTED

    # Capacity error
    huge_launch = _launch("succeed")
    huge_launch.resources = Resources(cpus=32.0, memory_gb=256.0)
    with pytest.raises(CapacityError):
        backend.submit(huge_launch)


@pytest.mark.unit
def test_aws_ec2_backend_contract():
    client = FakeEc2Client()
    backend_contract(
        make=lambda: Ec2Backend(client=client),
        launch=_launch,
        poll_seconds=0.0,
    )


@pytest.mark.unit
def test_aws_ec2_spot_preemption():
    client = FakeEc2Client()
    backend = Ec2Backend(client=client)
    handle = backend.submit(_launch("preempt"))
    status = backend.poll(handle)
    assert status.state == AttemptState.PREEMPTED


@pytest.mark.unit
def test_aws_ecr_image_registry():
    client = FakeEcrClient()
    registry = EcrImageRegistry(client=client, repository="wsr-release")
    assert registry.resolve("v1.1.0") is None
    digest = registry.push("local:v1.1.0", "v1.1.0")
    assert digest.startswith("sha256:")
    assert registry.resolve("v1.1.0") is not None


@pytest.mark.unit
def test_aws_pricing():
    assert price_per_hour("c6i.large", "us-east-1", spot=False) == 0.085
    assert price_per_hour("c6i.large", "us-east-1", spot=True) == 0.034
    assert price_per_hour("fargate", "us-east-1", spot=False) == 0.04048
    assert price_per_hour("unknown-type") is None


@pytest.mark.unit
def test_aws_load_backend():
    from logic.controllers.cloud.providers import load_backend

    batch_be = load_backend(Provider.AWS, ComputeKind.BATCH, client=FakeBatchClient())
    assert isinstance(batch_be, AwsBatchBackend)
    fargate_be = load_backend(Provider.AWS, ComputeKind.SERVERLESS, client=FakeEcsClient())
    assert isinstance(fargate_be, EcsFargateBackend)
    ec2_be = load_backend(Provider.AWS, ComputeKind.VM, client=FakeEc2Client())
    assert isinstance(ec2_be, Ec2Backend)


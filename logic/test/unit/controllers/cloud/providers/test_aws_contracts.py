"""AWS provider implementation passes all cloud runner contract suites."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List
from urllib.parse import urlparse

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
from logic.controllers.cloud.providers.aws.storage import store_from_uri
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
        defs = [d for name, d in self.job_definitions.items() if name == jobDefinitionName and d.get("status") == status]
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


class FakeEcsClient:
    """In-memory AWS ECS client for contract testing.

    Rejects missing networkConfiguration and illegal containerOverrides.
    """

    def __init__(self) -> None:
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.runs: List[Dict[str, Any]] = []
        self.task_definitions: Dict[str, Dict[str, Any]] = {}

    def describe_task_definition(self, taskDefinition: str, **kwargs: Any) -> Dict[str, Any]:
        if taskDefinition in self.task_definitions:
            return {"taskDefinition": self.task_definitions[taskDefinition]}
        return {}

    def register_task_definition(self, family: str, **kwargs: Any) -> Dict[str, Any]:
        arn = f"arn:aws:ecs:us-east-1:123456789012:task-definition/{family}:1"
        data = {
            "family": family,
            "taskDefinitionArn": arn,
            "status": "ACTIVE",
            **kwargs,
        }
        self.task_definitions[family] = data
        return {"taskDefinition": data}

    def run_task(
        self,
        cluster: str,
        taskDefinition: str,
        overrides: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Fargate requires awsvpc networkConfiguration
        if "networkConfiguration" not in kwargs or "awsvpcConfiguration" not in kwargs["networkConfiguration"]:
            raise ValueError("InvalidParameterException: networkConfiguration required for Fargate")

        container_overrides = overrides.get("containerOverrides", [{}])[0]
        if "secrets" in container_overrides:
            raise ValueError("ParamValidationError: containerOverrides cannot contain 'secrets'")
        if "image" in container_overrides:
            raise ValueError("ParamValidationError: containerOverrides cannot contain 'image'")

        task_arn = f"arn:aws:ecs:us-east-1:123456789012:task/{cluster}/task-{len(self.tasks) + 1}"
        cmd_str = " ".join(container_overrides.get("command", []))
        tags = kwargs.get("tags", [])
        task_id = next((t["value"] for t in tags if t.get("key") == "wsr:task_id"), "")

        outcome = "succeed"
        if "fail" in task_id or "fail" in cmd_str:
            outcome = "fail"
        elif "hang" in task_id or "hang" in cmd_str:
            outcome = "hang"
        elif "preempt" in task_id or "preempt" in cmd_str:
            outcome = "preempt"
        elif "exit143" in task_id or "exit143" in cmd_str:
            outcome = "exit143"

        self.runs.append(
            {
                "cluster": cluster,
                "taskDefinition": taskDefinition,
                "overrides": overrides,
                "networkConfiguration": kwargs.get("networkConfiguration"),
                "kwargs": kwargs,
            }
        )
        self.tasks[task_arn] = {
            "taskArn": task_arn,
            "outcome": outcome,
            "cancelled": False,
            "polls": 0,
            "tags": kwargs.get("tags", []),
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
                elif t["outcome"] == "exit143":
                    status = "STOPPED"
                    code = "EssentialContainerExited"
                    reason = "Container received SIGTERM and exited 143"
                    exit_code = 143
                elif t["outcome"] == "fail":
                    status = "STOPPED"
                    code = "EssentialContainerExited"
                    reason = "Task failed with exit code 1"
                    exit_code = 1
                else:
                    status = "STOPPED"
                    code = "EssentialContainerExited"
                    reason = "Task finished successfully"
                    exit_code = 0

                result.append(
                    {
                        "taskArn": arn,
                        "lastStatus": status,
                        "stopCode": code,
                        "stoppedReason": reason,
                        "tags": t["tags"],
                        "containers": [{"exitCode": exit_code}],
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
        # Validate shutdown behavior
        if kwargs.get("InstanceInitiatedShutdownBehavior") != "terminate":
            raise ValueError("InstanceInitiatedShutdownBehavior must be 'terminate'")

        instance_id = f"i-{len(self.instances) + 1:017d}"
        user_data_b64 = kwargs.get("UserData", "")
        user_data = base64.b64decode(user_data_b64).decode("utf-8") if user_data_b64 else ""

        tags = []
        for spec in kwargs.get("TagSpecifications", []):
            tags.extend(spec.get("Tags", []))
        task_id = next((t["Value"] for t in tags if t.get("Key") == "wsr:task_id"), "")

        outcome = "succeed"
        if "fail" in task_id or "test_sim_fail" in user_data:
            outcome = "fail"
        elif "hang" in task_id or "test_sim_hang" in user_data:
            outcome = "hang"
        elif "preempt" in task_id or "test_sim_preempt" in user_data:
            outcome = "preempt"
        elif "exit143" in task_id or "test_sim_exit143" in user_data:
            outcome = "exit143"

        self.runs.append({"kwargs": kwargs, "userData": user_data})
        self.instances[instance_id] = {
            "InstanceId": instance_id,
            "outcome": outcome,
            "cancelled": False,
            "polls": 0,
            "Tags": tags,
            "InstanceMarketOptions": kwargs.get("InstanceMarketOptions"),
        }
        return {"Instances": [{"InstanceId": instance_id}]}

    def create_tags(self, Resources: List[str], Tags: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        for res_id in Resources:
            if res_id in self.instances:
                existing = {t["Key"]: t["Value"] for t in self.instances[res_id]["Tags"]}
                for tag in Tags:
                    existing[tag["Key"]] = tag["Value"]
                self.instances[res_id]["Tags"] = [{"Key": k, "Value": v} for k, v in existing.items()]
        return {}

    def describe_instances(self, InstanceIds: List[str]) -> Dict[str, Any]:
        result = []
        for iid in InstanceIds:
            if iid in self.instances:
                inst = self.instances[iid]
                inst["polls"] += 1

                tags_dict = {t["Key"]: t["Value"] for t in inst["Tags"]}
                state = "running"
                reason_code = ""
                reason_msg = ""
                trans_reason = ""

                if inst["cancelled"]:
                    state = "terminated"
                    trans_reason = "User initiated shutdown"
                    tags_dict["wsr:exit_code"] = "143"
                elif inst["outcome"] == "hang":
                    state = "running"
                elif inst["outcome"] == "preempt":
                    state = "terminated"
                    reason_code = "Server.SpotInstanceTermination"
                    reason_msg = "Spot instance termination notice received"
                    trans_reason = "Server.SpotInstanceTermination"
                elif inst["outcome"] == "exit143":
                    state = "terminated"
                    trans_reason = "Container exited 143"
                    tags_dict["wsr:exit_code"] = "143"
                elif inst["outcome"] == "fail":
                    state = "terminated"
                    trans_reason = "Container exited 1"
                    tags_dict["wsr:exit_code"] = "1"
                else:
                    state = "terminated"
                    trans_reason = "Normal exit"
                    tags_dict["wsr:exit_code"] = "0"

                inst["Tags"] = [{"Key": k, "Value": v} for k, v in tags_dict.items()]

                result.append(
                    {
                        "InstanceId": iid,
                        "State": {"Name": state},
                        "StateReason": {"Code": reason_code, "Message": reason_msg},
                        "StateTransitionReason": trans_reason,
                        "Tags": inst["Tags"],
                        "InstanceMarketOptions": inst.get("InstanceMarketOptions"),
                    }
                )
        return {"Reservations": [{"Instances": result}]}

    def terminate_instances(self, InstanceIds: List[str]) -> Dict[str, Any]:
        for iid in InstanceIds:
            if iid in self.instances:
                self.instances[iid]["cancelled"] = True
        return {}

    def get_console_output(self, InstanceId: str) -> Dict[str, Any]:
        return {"Output": f"console logs for instance {InstanceId}\nline 2\n"}


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


# ---------------------------------------------------------------------------
# Contracts & Provider Unit Tests
# ---------------------------------------------------------------------------


def test_aws_s3_object_store_contract() -> None:
    client = FakeS3Client()
    object_store_contract(lambda: S3ObjectStore(bucket="wsr-test-bucket", client=client))


def test_aws_s3_store_from_uri() -> None:
    store = store_from_uri(urlparse("s3://my-bucket/runs/prefix/"), client=FakeS3Client())
    assert store.bucket == "my-bucket"
    assert store.prefix == "runs/prefix/"


def test_aws_secrets_manager_contract() -> None:
    client = FakeSecretsManagerClient()
    secret_store_contract(lambda: AwsSecretsManagerStore(client=client))


def test_aws_batch_backend_contract() -> None:
    backend_contract(
        lambda: AwsBatchBackend(
            client=FakeBatchClient(),
            logs_client=FakeLogsClient(),
            secrets_store=AwsSecretsManagerStore(client=FakeSecretsManagerClient()),
        ),
        launch=make_launch_for_outcome,
    )


def test_aws_batch_rejects_container_overrides_secrets_and_registers_job_def() -> None:
    batch_client = FakeBatchClient()
    sm_client = FakeSecretsManagerClient()
    sm_client.create_secret(Name="WSR_GUROBI_KEY", SecretString="dummy-license-key")
    secrets_store = AwsSecretsManagerStore(client=sm_client)

    backend = AwsBatchBackend(
        client=batch_client,
        logs_client=FakeLogsClient(),
        secrets_store=secrets_store,
        execution_role_arn="arn:aws:iam::123456789012:role/wsr-exec-role",
    )

    launch = make_test_launch(
        task_id="task-batch-def",
        secret_env={"WSR_GUROBI_LICENSE": "WSR_GUROBI_KEY"},
    )
    handle = backend.submit(launch)
    assert handle.startswith("aws-batch:us-east-1:")

    # Verify a job definition was registered with secrets and image
    assert len(batch_client.job_definitions) == 1
    def_data = next(iter(batch_client.job_definitions.values()))
    props = def_data["containerProperties"]
    assert props["image"] == "wsmart-route:v1.1.0"
    assert props["secrets"] == [
        {"name": "WSR_GUROBI_LICENSE", "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:WSR_GUROBI_KEY"}
    ]
    assert props["executionRoleArn"] == "arn:aws:iam::123456789012:role/wsr-exec-role"

    # Verify containerOverrides has NO image or secrets
    assert len(batch_client.submits) == 1
    co = batch_client.submits[0]["containerOverrides"]
    assert "image" not in co
    assert "secrets" not in co
    assert co["command"] == ["test_sim", "--days", "30"]


def test_aws_batch_spot_preemption_and_exit_143() -> None:
    batch_client = FakeBatchClient()
    backend = AwsBatchBackend(
        client=batch_client,
        logs_client=FakeLogsClient(),
        secrets_store=AwsSecretsManagerStore(client=FakeSecretsManagerClient()),
    )

    # Preemption status reason
    handle_preempt = backend.submit(make_test_launch(task_id="task-preempt", spot=True))
    st = backend.poll(handle_preempt)
    assert st.state == AttemptState.PREEMPTED

    # Exit code 143 on spot
    handle_143 = backend.submit(make_test_launch(task_id="task-exit143", spot=True))
    st_143 = backend.poll(handle_143)
    assert st_143.state == AttemptState.PREEMPTED

    # Capacity error on oversized resource
    with pytest.raises(CapacityError):
        backend.submit(make_test_launch(cpus=256.0, memory_gb=1024.0))


def test_aws_ecs_fargate_backend_contract() -> None:
    backend_contract(
        lambda: EcsFargateBackend(
            client=FakeEcsClient(),
            logs_client=FakeLogsClient(),
            secrets_store=AwsSecretsManagerStore(client=FakeSecretsManagerClient()),
        ),
        launch=make_launch_for_outcome,
    )


def test_aws_ecs_fargate_rejects_missing_network_and_registers_task_def() -> None:
    ecs_client = FakeEcsClient()
    sm_client = FakeSecretsManagerClient()
    sm_client.create_secret(Name="WSR_GUROBI_KEY", SecretString="dummy-license-key")
    secrets_store = AwsSecretsManagerStore(client=sm_client)

    backend = EcsFargateBackend(
        client=ecs_client,
        logs_client=FakeLogsClient(),
        subnets=["subnet-0123", "subnet-4567"],
        security_groups=["sg-9999"],
        execution_role_arn="arn:aws:iam::123456789012:role/ecs-task-exec",
        secrets_store=secrets_store,
    )

    launch = make_test_launch(
        task_id="task-fargate-def",
        secret_env={"WSR_GUROBI_LICENSE": "WSR_GUROBI_KEY"},
        spot=True,
    )
    handle = backend.submit(launch)
    assert handle.startswith("aws-ecs:us-east-1:wsr-cluster:")

    # Verify task definition registered
    assert len(ecs_client.task_definitions) == 1
    td = next(iter(ecs_client.task_definitions.values()))
    assert td["requiresCompatibilities"] == ["FARGATE"]
    assert td["networkMode"] == "awsvpc"
    container = td["containerDefinitions"][0]
    assert container["image"] == "wsmart-route:v1.1.0"
    assert container["secrets"] == [
        {"name": "WSR_GUROBI_LICENSE", "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:WSR_GUROBI_KEY"}
    ]

    # Verify networkConfiguration passed to run_task
    assert len(ecs_client.runs) == 1
    run_call = ecs_client.runs[0]
    awsvpc = run_call["networkConfiguration"]["awsvpcConfiguration"]
    assert awsvpc["subnets"] == ["subnet-0123", "subnet-4567"]
    assert awsvpc["securityGroups"] == ["sg-9999"]

    # Verify containerOverrides does not have secrets or image
    co = run_call["overrides"]["containerOverrides"][0]
    assert "secrets" not in co
    assert "image" not in co

    # Spot interruption and exit code 143 map to PREEMPTED
    st = backend.poll(handle)
    assert st.state == AttemptState.SUCCEEDED

    handle_preempt = backend.submit(make_test_launch(task_id="task-preempt", spot=True))
    assert backend.poll(handle_preempt).state == AttemptState.PREEMPTED

    handle_143 = backend.submit(make_test_launch(task_id="task-exit143", spot=True))
    assert backend.poll(handle_143).state == AttemptState.PREEMPTED


def test_aws_ec2_backend_contract() -> None:
    backend_contract(
        lambda: Ec2Backend(client=FakeEc2Client()),
        launch=make_launch_for_outcome,
    )


def test_aws_ec2_user_data_script_and_spot_preemption() -> None:
    ec2_client = FakeEc2Client()
    backend = Ec2Backend(client=ec2_client, region="us-west-2")

    launch = make_test_launch(
        task_id="task-ec2-run",
        secret_env={"WSR_GUROBI_LICENSE": "gurobi-secret-name"},
        spot=True,
    )
    handle = backend.submit(launch)
    assert handle.startswith("aws-ec2:us-west-2:")

    # Verify run_instances arguments and user data decoding
    assert len(ec2_client.runs) == 1
    run_info = ec2_client.runs[0]
    kwargs = run_info["kwargs"]
    assert kwargs["InstanceInitiatedShutdownBehavior"] == "terminate"

    user_data = run_info["userData"]
    assert "#!/bin/bash" in user_data
    assert "docker run --rm" in user_data
    assert "wsmart-route:v1.1.0" in user_data
    assert "test_sim" in user_data
    assert "shutdown -h now" in user_data
    assert "aws ec2 create-tags" in user_data
    assert "WSR_SECRET_REFS" in user_data
    assert "awssm://us-west-2/gurobi-secret-name" in user_data

    # Test polling states
    st = backend.poll(handle)
    assert st.state == AttemptState.SUCCEEDED

    # Test spot interruption
    handle_preempt = backend.submit(make_test_launch(task_id="task-preempt", spot=True))
    assert backend.poll(handle_preempt).state == AttemptState.PREEMPTED

    # Test exit code 143 on spot
    handle_143 = backend.submit(make_test_launch(task_id="task-exit143", spot=True))
    assert backend.poll(handle_143).state == AttemptState.PREEMPTED


def test_aws_ecr_image_registry() -> None:
    commands: List[List[str]] = []

    def mock_run(argv: List[str], input: Any = None, **kwargs: Any) -> Any:
        commands.append(argv)
        class MockResult:
            returncode = 0
            stdout = '["123456789012.dkr.ecr.us-east-1.amazonaws.com/wsmart-route@sha256:abcd1234efgh5678"]'
            stderr = ""
        if argv[:3] == ["docker", "manifest", "inspect"]:
            class ManifestResult:
                returncode = 0
                stdout = '{"Descriptor": {"digest": "sha256:abcd1234efgh5678"}}'
                stderr = ""
            return ManifestResult()
        return MockResult()

    registry = EcrImageRegistry(
        account="123456789012",
        region="us-east-1",
        image="wsmart-route",
        run=mock_run,
    )
    digest = registry.push("local:latest", "v1.1.0")
    assert digest == "sha256:abcd1234efgh5678"

    resolved = registry.resolve("v1.1.0")
    assert resolved == "sha256:abcd1234efgh5678"


def test_aws_pricing() -> None:
    p_ondemand = price_per_hour("c6i.large", "us-east-1", spot=False)
    p_spot = price_per_hour("c6i.large", "us-east-1", spot=True)
    assert p_ondemand is not None and p_spot is not None
    assert p_spot < p_ondemand

    f_ondemand = price_per_hour("fargate.1vcpu", "us-east-1", spot=False)
    f_spot = price_per_hour("fargate.1vcpu", "us-east-1", spot=True)
    assert f_ondemand is not None and f_spot is not None
    assert f_spot < f_ondemand


def test_aws_load_backend() -> None:
    from logic.controllers.cloud.providers import load_backend

    batch = load_backend(
        Provider.AWS,
        ComputeKind.BATCH,
        client=FakeBatchClient(),
        logs_client=FakeLogsClient(),
    )
    assert isinstance(batch, AwsBatchBackend)

    ecs = load_backend(
        Provider.AWS,
        ComputeKind.SERVERLESS,
        client=FakeEcsClient(),
        logs_client=FakeLogsClient(),
    )
    assert isinstance(ecs, EcsFargateBackend)

    ec2 = load_backend(Provider.AWS, ComputeKind.VM, client=FakeEc2Client())
    assert isinstance(ec2, Ec2Backend)

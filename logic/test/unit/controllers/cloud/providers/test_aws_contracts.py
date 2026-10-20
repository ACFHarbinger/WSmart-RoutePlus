"""AWS provider implementation passes all cloud runner contract suites.

The in-process client fakes live in ``aws_fakes.py`` and ``aws_compute_fakes.py``; the
ECS Fargate and EC2 tests live in ``test_aws_ecs_ec2_contracts.py``.
"""

from __future__ import annotations

from typing import Any, List
from urllib.parse import urlparse

import pytest
from logic.controllers.cloud.interfaces import (
    AttemptState,
    CapacityError,
)
from logic.controllers.cloud.model import (
    ComputeKind,
    Provider,
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
from logic.test.unit.controllers.cloud.providers.aws_compute_fakes import (
    FakeEc2Client,
    FakeEcsClient,
)
from logic.test.unit.controllers.cloud.providers.aws_fakes import (
    FakeBatchClient,
    FakeLogsClient,
    FakeS3Client,
    FakeSecretsManagerClient,
    make_launch_for_outcome,
    make_test_launch,
)

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
        {
            "name": "WSR_GUROBI_LICENSE",
            "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:WSR_GUROBI_KEY",
        }
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

"""AWS ECS Fargate and EC2 backends pass the cloud runner backend contract.

The ECS Fargate and EC2 tests, moved unchanged from ``test_aws_contracts.py``.
"""

from __future__ import annotations

from logic.controllers.cloud.interfaces import (
    AttemptState,
)
from logic.controllers.cloud.providers.aws import (
    AwsSecretsManagerStore,
    Ec2Backend,
    EcsFargateBackend,
)
from logic.controllers.cloud.testing import (
    backend_contract,
)
from logic.test.unit.controllers.cloud.providers.aws_compute_fakes import (
    FakeEc2Client,
    FakeEcsClient,
)
from logic.test.unit.controllers.cloud.providers.aws_fakes import (
    FakeLogsClient,
    FakeSecretsManagerClient,
    make_launch_for_outcome,
    make_test_launch,
)


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
        {
            "name": "WSR_GUROBI_LICENSE",
            "valueFrom": "arn:aws:secretsmanager:us-east-1:123456789012:secret:WSR_GUROBI_KEY",
        }
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
    assert "trap 'shutdown -h now' EXIT" in user_data  # the instance stops even if a step fails
    assert "docker login" not in user_data  # a public image needs no registry login

    ecr_launch = make_test_launch(task_id="task-ecr")
    ecr_launch.image = "123456789012.dkr.ecr.us-west-2.amazonaws.com/wsmart-route@sha256:" + "ab" * 32
    backend.submit(ecr_launch)
    ecr_script = ec2_client.runs[-1]["userData"]
    assert "aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin " in ecr_script
    assert ecr_script.index("docker login") < ecr_script.index("docker run")

    # Test polling states
    st = backend.poll(handle)
    assert st.state == AttemptState.SUCCEEDED

    # Test spot interruption
    handle_preempt = backend.submit(make_test_launch(task_id="task-preempt", spot=True))
    assert backend.poll(handle_preempt).state == AttemptState.PREEMPTED

    # Test exit code 143 on spot
    handle_143 = backend.submit(make_test_launch(task_id="task-exit143", spot=True))
    assert backend.poll(handle_143).state == AttemptState.PREEMPTED

"""In-process AWS ECS and EC2 client fakes for the AWS provider contract tests.

Moved unchanged from ``test_aws_contracts.py``.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List


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

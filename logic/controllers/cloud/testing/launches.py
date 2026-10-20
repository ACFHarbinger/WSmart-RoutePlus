"""Launch specs shared by the provider contract tests (CR-05).

:func:`contract_launch` builds the ``succeed`` / ``fail`` / ``hang`` (and ``preempt``)
launches :func:`.contracts.backend_contract` asks for. The outcome travels in the
``WSR_FAKE_OUTCOME`` environment variable, which every provider puts in its request
(Batch environment, Cloud Run env, VM startup script, ECS overrides), so a provider's
fake client can read it from the request it receives.

Attributes:
    OUTCOME_ENV: The variable holding the scripted outcome.
    contract_launch: A launch with that outcome.
    outcome_of: The outcome a fake reads back from a request's environment.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Mapping, Optional

from logic.controllers.cloud.interfaces import LaunchSpec
from logic.controllers.cloud.model import CellSpec, Resources, TaskRecord

OUTCOME_ENV = "WSR_FAKE_OUTCOME"
_COUNTER = itertools.count(1)


def contract_launch(
    outcome: str,
    secret_env: Optional[Dict[str, str]] = None,
    resources: Optional[Resources] = None,
    image: str = "registry.example/wsmart-route@sha256:feed",
) -> LaunchSpec:
    """A runtime launch whose fake attempt ends with ``outcome``."""
    n = next(_COUNTER)
    task = TaskRecord(
        task_id=f"task-{outcome}-{n}", job_id="job-contract", cell=CellSpec(cell_id=f"c{n}", argv=["test_sim"])
    )
    return LaunchSpec(
        task=task,
        resources=resources or Resources(cpus=2.0, memory_gb=4.0),
        argv=["python", "-m", "logic.controllers.cloud.runtime"],
        env={"WSR_CLOUD_JOB_ID": task.job_id, "WSR_CLOUD_TASK_ID": task.task_id, OUTCOME_ENV: outcome},
        secret_env=dict(secret_env or {}),
        image=image,
    )


def outcome_of(env: Any) -> str:
    """``WSR_FAKE_OUTCOME`` from a mapping or a ``[{"name", "value"}]`` list; ``succeed`` if absent."""
    if isinstance(env, Mapping):
        return str(env.get(OUTCOME_ENV, "succeed"))
    for item in env or []:
        if isinstance(item, Mapping) and item.get("name") == OUTCOME_ENV:
            return str(item.get("value", "succeed"))
    return "succeed"

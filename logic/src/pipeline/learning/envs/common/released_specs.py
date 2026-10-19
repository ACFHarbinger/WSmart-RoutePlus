"""Torch-free problem specs for the released classical simulator.

Importing this module does not import PyTorch or a neural environment.
``env_target`` stays a string until a neural caller asks for the class.
"""

from typing import Optional

from logic.src.pipeline.learning.envs.common.metrics import DEFAULT_METRICS
from logic.src.pipeline.learning.envs.common.objectives import distance_objective, profit_objective
from logic.src.pipeline.learning.envs.spec import ProblemSpec

PTP_SPEC = ProblemSpec(
    name="ptp",
    objective=profit_objective,
    metrics=DEFAULT_METRICS,
    env_target="logic.src.pipeline.learning.envs.ptp:PTPEnv",
    expand_pool=True,
)
CPTP_SPEC = ProblemSpec(
    name="cptp",
    objective=profit_objective,
    metrics=DEFAULT_METRICS,
    env_target="logic.src.pipeline.learning.envs.cptp:CPTPEnv",
    expand_pool=True,
)
TCPTP_SPEC = ProblemSpec(
    name="tcptp",
    objective=profit_objective,
    metrics=DEFAULT_METRICS + ("time_spent",),
    env_target="logic.src.pipeline.learning.envs.tcptp:TCPTPEnv",
    expand_pool=True,
    per_trip_shift=True,
)
CVRP_SPEC = ProblemSpec(
    name="cvrp",
    objective=distance_objective,
    metrics=DEFAULT_METRICS,
    env_target="logic.src.pipeline.learning.envs.cvrp:CVRPEnv",
    expand_pool=False,
)

_RELEASED = {spec.name: spec for spec in (PTP_SPEC, CPTP_SPEC, TCPTP_SPEC, CVRP_SPEC)}


def released_spec(name: str) -> Optional[ProblemSpec]:
    """Return a released spec, or None when the name is not a released problem."""
    return _RELEASED.get(name.lower())

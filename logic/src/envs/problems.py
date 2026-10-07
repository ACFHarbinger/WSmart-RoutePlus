"""
Facade for the problems package.

Attributes:
    BaseProblem: Base class for all problems
    COST_KM: Cost per kilometer
    REVENUE_KG: Revenue per kilogram
    BIN_CAPACITY: Bin capacity
    VEHICLE_CAPACITY: Vehicle capacity
    VRPP: Capacitated VRP
    CVRPP: Capacitated VRP with waste
    CTOP: VRPP with both per-trip capacity and time constraints

Example:
    from logic.src.envs.problems import VRPP
    env = VRPP(num_loc=50, cost_km=10)
    obs, _ = env.reset()
"""

from logic.src.constants.tasks import (
    BIN_CAPACITY,
    COST_KM,
    REVENUE_KG,
    VEHICLE_CAPACITY,
)
from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.ctop import CTOP
from logic.src.envs.tasks.cvrpp import CVRPP
from logic.src.envs.tasks.vrpp import VRPP

__all__ = [
    "BaseProblem",
    "COST_KM",
    "REVENUE_KG",
    "BIN_CAPACITY",
    "VEHICLE_CAPACITY",
    "VRPP",
    "CVRPP",
    "CTOP",
]

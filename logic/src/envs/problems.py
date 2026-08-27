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
    TTOP: VRPP with a per-trip time budget instead of a capacity constraint
    WCVRP: Waste-only CVRP
    CWCVRP: Capacitated VRP with waste and time windows
    SCWCVRP: Single-depot Capacitated VRP with waste and time windows

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
from logic.src.envs.tasks.cvrpp import CVRPP
from logic.src.envs.tasks.cwcvrp import CWCVRP
from logic.src.envs.tasks.scwcvrp import SCWCVRP
from logic.src.envs.tasks.ttop import TTOP
from logic.src.envs.tasks.vrpp import VRPP
from logic.src.envs.tasks.wcvrp import WCVRP

__all__ = [
    "BaseProblem",
    "COST_KM",
    "REVENUE_KG",
    "BIN_CAPACITY",
    "VEHICLE_CAPACITY",
    "VRPP",
    "CVRPP",
    "TTOP",
    "WCVRP",
    "CWCVRP",
    "SCWCVRP",
]

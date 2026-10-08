"""
Facade for the problems package.

Attributes:
    BaseProblem: Base class for all problems
    COST_KM: Cost per kilometer
    REVENUE_KG: Revenue per kilogram
    BIN_CAPACITY: Bin capacity
    VEHICLE_CAPACITY: Vehicle capacity
    PTP: Capacitated VRP
    CPTP: Capacitated VRP with waste
    TCPTP: PTP with both per-trip capacity and time constraints

Example:
    from logic.src.envs.problems import PTP
    env = PTP(num_loc=50, cost_km=10)
    obs, _ = env.reset()
"""

from logic.src.constants.tasks import (
    BIN_CAPACITY,
    COST_KM,
    REVENUE_KG,
    VEHICLE_CAPACITY,
)
from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.cptp import CPTP
from logic.src.envs.tasks.ptp import PTP
from logic.src.envs.tasks.tcptp import TCPTP

__all__ = [
    "BaseProblem",
    "COST_KM",
    "REVENUE_KG",
    "BIN_CAPACITY",
    "VEHICLE_CAPACITY",
    "PTP",
    "CPTP",
    "TCPTP",
]

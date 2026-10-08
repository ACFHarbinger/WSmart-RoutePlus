"""
Problem definitions sub-package.

Attributes:
    ATSP: ATSP class
    BaseProblem: BaseProblem class
    CVRP: CVRP class
    CPTP: CPTP class
    IRP: IRP class
    OP: OP class
    PCTSP: PCTSP class
    PDP: PDP class
    SPCTSP: SPCTSP class
    ThOP: ThOP class
    TSP: TSP class
    TCPTP: TCPTP class
    PTP: PTP class

Example:
    >>> from logic.src.envs import TSPEnv
    >>> env = TSPEnv(num_nodes=20, generator_params={"num_nodes": 20})
"""

from logic.src.constants.tasks import (
    BIN_CAPACITY,
    COST_KM,
    REVENUE_KG,
    VEHICLE_CAPACITY,
)

from .atsp import ATSP
from .base import BaseProblem
from .cvrp import CVRP
from .irp import IRP
from .cptp import CPTP
from .op import OP
from .pctsp import PCTSP
from .pdp import PDP
from .ptp import PTP
from .spctsp import SPCTSP
from .tcptp import TCPTP
from .thop import ThOP
from .tsp import TSP

__all__ = [
    "BaseProblem",
    "COST_KM",
    "REVENUE_KG",
    "BIN_CAPACITY",
    "VEHICLE_CAPACITY",
    "PTP",
    "CPTP",
    "IRP",
    "ATSP",
    "TSP",
    "CVRP",
    "OP",
    "PCTSP",
    "SPCTSP",
    "PDP",
    "ThOP",
    "TCPTP",
]

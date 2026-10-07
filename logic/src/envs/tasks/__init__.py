"""
Problem definitions sub-package.

Attributes:
    ATSP: ATSP class
    BaseProblem: BaseProblem class
    CVRP: CVRP class
    CVRPP: CVRPP class
    IRP: IRP class
    OP: OP class
    PCTSP: PCTSP class
    PDP: PDP class
    SPCTSP: SPCTSP class
    ThOP: ThOP class
    TSP: TSP class
    CTOP: CTOP class
    VRPP: VRPP class

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
from .ctop import CTOP
from .cvrp import CVRP
from .cvrpp import CVRPP
from .irp import IRP
from .op import OP
from .pctsp import PCTSP
from .pdp import PDP
from .spctsp import SPCTSP
from .thop import ThOP
from .tsp import TSP
from .vrpp import VRPP

__all__ = [
    "BaseProblem",
    "COST_KM",
    "REVENUE_KG",
    "BIN_CAPACITY",
    "VEHICLE_CAPACITY",
    "VRPP",
    "CVRPP",
    "IRP",
    "ATSP",
    "TSP",
    "CVRP",
    "OP",
    "PCTSP",
    "SPCTSP",
    "PDP",
    "ThOP",
    "CTOP",
]

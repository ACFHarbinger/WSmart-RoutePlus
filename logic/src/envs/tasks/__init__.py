"""Problem definitions, environments, and tasks sub-package.

Attributes:
    BaseProblem: Base problem class
    PTP, CPTP, TCPTP, CVRP: Released routing problem definitions
    TSP, ATSP, OP, PCTSP, SPCTSP, PDP, IRP, ThOP: Combinatorial problem definitions
    ESPPRCEnv: Canonical base environment
    PTPEnv, CPTPEnv, TCPTPEnv, CVRPEnv: Routing environment classes
    Resource, LoadResource, ShiftTimeResource: Declarative resource constraints
    ProblemSpec, PROBLEMS: Problem specifications and registry
"""

from typing import Any

from logic.src.constants.tasks import (
    BIN_CAPACITY,
    COST_KM,
    REVENUE_KG,
    VEHICLE_CAPACITY,
)
from logic.src.envs.tasks.atsp import ATSP, ATSPEnv, ATSPGenerator
from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.cvrp import CVRP, CVRPEnv, CVRPGenerator
from logic.src.envs.tasks.espprc import (
    ESPPRCEnv,
    ESPPRCGenerator,
    LoadResource,
    Resource,
    ShiftTimeResource,
)
from logic.src.envs.tasks.irp import IRP, IRPEnv, IRPGenerator
from logic.src.envs.tasks.op import OP, OPEnv, OPGenerator
from logic.src.envs.tasks.pctsp import PCTSP, PCTSPEnv, PCTSPGenerator
from logic.src.envs.tasks.pdp import PDP, PDPEnv, PDPGenerator
from logic.src.envs.tasks.profitable_tour.cptp import CPTP, CPTPEnv, CPTPGenerator
from logic.src.envs.tasks.profitable_tour.ptp import PTP, PTPEnv, PTPGenerator
from logic.src.envs.tasks.profitable_tour.tcptp import TCPTP, TCPTPEnv, TCPTPGenerator
from logic.src.envs.tasks.spctsp import SPCTSP, SPCTSPEnv, SPCTSPGenerator
from logic.src.envs.tasks.spec import PROBLEMS, ProblemRegistry, ProblemSpec
from logic.src.envs.tasks.thop import ThOP, ThOPEnv, ThOPGenerator
from logic.src.envs.tasks.tsp import TSP, TSPEnv, TSPGenerator


def get_env(name: str, **kwargs) -> Any:
    """Build an environment by name via the PROBLEMS registry."""
    key = name.lower()
    if key in PROBLEMS:
        return PROBLEMS[key].get_env(**kwargs)
    from logic.src.envs import get_env as root_get_env
    return root_get_env(name, **kwargs)


__all__ = [
    "BaseProblem",
    "COST_KM",
    "REVENUE_KG",
    "BIN_CAPACITY",
    "VEHICLE_CAPACITY",
    "Resource",
    "LoadResource",
    "ShiftTimeResource",
    "ESPPRCGenerator",
    "ESPPRCEnv",
    "PTP",
    "CPTP",
    "TCPTP",
    "CVRP",
    "TSP",
    "ATSP",
    "OP",
    "PCTSP",
    "SPCTSP",
    "PDP",
    "IRP",
    "ThOP",
    "PTPEnv",
    "CPTPEnv",
    "TCPTPEnv",
    "CVRPEnv",
    "TSPEnv",
    "ATSPEnv",
    "OPEnv",
    "PCTSPEnv",
    "SPCTSPEnv",
    "PDPEnv",
    "IRPEnv",
    "ThOPEnv",
    "PTPGenerator",
    "CPTPGenerator",
    "TCPTPGenerator",
    "CVRPGenerator",
    "TSPGenerator",
    "ATSPGenerator",
    "OPGenerator",
    "PCTSPGenerator",
    "SPCTSPGenerator",
    "PDPGenerator",
    "IRPGenerator",
    "ThOPGenerator",
    "ProblemSpec",
    "PROBLEMS",
    "ProblemRegistry",
    "get_env",
]

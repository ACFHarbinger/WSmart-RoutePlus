"""Lazy task exports; importing shared objectives or KPIs requires no neural extra."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_TARGETS = {
    "BIN_CAPACITY": "logic.src.constants.tasks:BIN_CAPACITY",
    "COST_KM": "logic.src.constants.tasks:COST_KM",
    "REVENUE_KG": "logic.src.constants.tasks:REVENUE_KG",
    "VEHICLE_CAPACITY": "logic.src.constants.tasks:VEHICLE_CAPACITY",
    "ATSP": "logic.src.envs.tasks.atsp:ATSP",
    "ATSPEnv": "logic.src.envs.tasks.atsp:ATSPEnv",
    "ATSPGenerator": "logic.src.envs.tasks.atsp:ATSPGenerator",
    "BaseProblem": "logic.src.envs.tasks.base:BaseProblem",
    "CVRP": "logic.src.envs.tasks.cvrp:CVRP",
    "CVRPEnv": "logic.src.envs.tasks.cvrp:CVRPEnv",
    "CVRPGenerator": "logic.src.envs.tasks.cvrp:CVRPGenerator",
    "ESPPRCEnv": "logic.src.envs.tasks.espprc:ESPPRCEnv",
    "ESPPRCGenerator": "logic.src.envs.tasks.espprc:ESPPRCGenerator",
    "LoadResource": "logic.src.envs.tasks.espprc:LoadResource",
    "Resource": "logic.src.envs.tasks.espprc:Resource",
    "ShiftTimeResource": "logic.src.envs.tasks.espprc:ShiftTimeResource",
    "IRP": "logic.src.envs.tasks.irp:IRP",
    "IRPEnv": "logic.src.envs.tasks.irp:IRPEnv",
    "IRPGenerator": "logic.src.envs.tasks.irp:IRPGenerator",
    "OP": "logic.src.envs.tasks.op:OP",
    "OPEnv": "logic.src.envs.tasks.op:OPEnv",
    "OPGenerator": "logic.src.envs.tasks.op:OPGenerator",
    "PCTSP": "logic.src.envs.tasks.pctsp:PCTSP",
    "PCTSPEnv": "logic.src.envs.tasks.pctsp:PCTSPEnv",
    "PCTSPGenerator": "logic.src.envs.tasks.pctsp:PCTSPGenerator",
    "PDP": "logic.src.envs.tasks.pdp:PDP",
    "PDPEnv": "logic.src.envs.tasks.pdp:PDPEnv",
    "PDPGenerator": "logic.src.envs.tasks.pdp:PDPGenerator",
    "CPTP": "logic.src.envs.tasks.profitable_tour.cptp:CPTP",
    "CPTPEnv": "logic.src.envs.tasks.profitable_tour.cptp:CPTPEnv",
    "CPTPGenerator": "logic.src.envs.tasks.profitable_tour.cptp:CPTPGenerator",
    "PTP": "logic.src.envs.tasks.profitable_tour.ptp:PTP",
    "PTPEnv": "logic.src.envs.tasks.profitable_tour.ptp:PTPEnv",
    "PTPGenerator": "logic.src.envs.tasks.profitable_tour.ptp:PTPGenerator",
    "TCPTP": "logic.src.envs.tasks.profitable_tour.tcptp:TCPTP",
    "TCPTPEnv": "logic.src.envs.tasks.profitable_tour.tcptp:TCPTPEnv",
    "TCPTPGenerator": "logic.src.envs.tasks.profitable_tour.tcptp:TCPTPGenerator",
    "SPCTSP": "logic.src.envs.tasks.spctsp:SPCTSP",
    "SPCTSPEnv": "logic.src.envs.tasks.spctsp:SPCTSPEnv",
    "SPCTSPGenerator": "logic.src.envs.tasks.spctsp:SPCTSPGenerator",
    "PROBLEMS": "logic.src.envs.tasks.spec:PROBLEMS",
    "ProblemRegistry": "logic.src.envs.tasks.spec:ProblemRegistry",
    "ProblemSpec": "logic.src.envs.tasks.spec:ProblemSpec",
    "ThOP": "logic.src.envs.tasks.thop:ThOP",
    "ThOPEnv": "logic.src.envs.tasks.thop:ThOPEnv",
    "ThOPGenerator": "logic.src.envs.tasks.thop:ThOPGenerator",
    "TSP": "logic.src.envs.tasks.tsp:TSP",
    "TSPEnv": "logic.src.envs.tasks.tsp:TSPEnv",
    "TSPGenerator": "logic.src.envs.tasks.tsp:TSPGenerator",
}


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _TARGETS, name)


def get_env(name: str, **kwargs: Any) -> Any:
    """Build an environment lazily through its problem specification."""
    from logic.src.envs.tasks.spec import PROBLEMS

    if name in PROBLEMS:
        return PROBLEMS[name].get_env(**kwargs)
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

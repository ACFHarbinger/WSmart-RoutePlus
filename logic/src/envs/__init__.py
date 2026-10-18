"""Environments. Concrete envs load when ``get_env`` is called."""

from typing import Any, Dict

from logic.src.utils.registry import import_target

ENV_TARGETS: Dict[str, str] = {
    "ptp": "logic.src.envs.tasks.ptp:PTPEnv",
    "cptp": "logic.src.envs.tasks.cptp:CPTPEnv",
    "tcptp": "logic.src.envs.tasks.tcptp:TCPTPEnv",
    "tsp": "logic.src.envs.tasks.tsp:TSPEnv",
    "tsp_kopt": "logic.src.envs.tsp_kopt:TSPkoptEnv",
    "irp": "logic.src.envs.tasks.irp:IRPEnv",
    "atsp": "logic.src.envs.tasks.atsp:ATSPEnv",
    "cvrp": "logic.src.envs.tasks.cvrp:CVRPEnv",
    "op": "logic.src.envs.tasks.op:OPEnv",
    "pctsp": "logic.src.envs.tasks.pctsp:PCTSPEnv",
    "spctsp": "logic.src.envs.tasks.spctsp:SPCTSPEnv",
    "pdp": "logic.src.envs.tasks.pdp:PDPEnv",
    "thop": "logic.src.envs.tasks.thop:ThOPEnv",
}

# Kept for callers that inspect keys. Values are resolved on get_env.
ENV_REGISTRY = ENV_TARGETS


def get_env(name: str, **kwargs: Any) -> Any:
    """Build an environment by name, importing only that env module."""
    key = name.lower()
    if key not in ENV_TARGETS:
        raise ValueError(f"Unknown environment: {name}. Available: {list(ENV_TARGETS.keys())}")
    env_cls = import_target(ENV_TARGETS[key])
    return env_cls(**kwargs)


def __getattr__(name: str) -> Any:
    mapping = {
        "RL4COEnvBase": "logic.src.envs.base.base:RL4COEnvBase",
        "ImprovementEnvBase": "logic.src.envs.base.improvement:ImprovementEnvBase",
        "RoutingEnv": "logic.src.envs.base.routing:RoutingEnv",
        "Resource": "logic.src.envs.base.resources:Resource",
        "LoadResource": "logic.src.envs.base.resources:LoadResource",
        "ShiftTimeResource": "logic.src.envs.base.resources:ShiftTimeResource",
        "RoutingGenerator": "logic.src.envs.base.routing:RoutingGenerator",
        "Generator": "logic.src.envs.generators.base:Generator",
        "PTPGenerator": "logic.src.envs.tasks.ptp:PTPGenerator",
        "CPTPGenerator": "logic.src.envs.tasks.cptp:CPTPGenerator",
        "TCPTPGenerator": "logic.src.envs.tasks.tcptp:TCPTPGenerator",
        "CVRPGenerator": "logic.src.envs.tasks.cvrp:CVRPGenerator",
        "TSPGenerator": "logic.src.envs.tasks.tsp:TSPGenerator",
        "ATSPGenerator": "logic.src.envs.tasks.atsp:ATSPGenerator",
        "IRPGenerator": "logic.src.envs.tasks.irp:IRPGenerator",
        "OPGenerator": "logic.src.envs.tasks.op:OPGenerator",
        "PCTSPGenerator": "logic.src.envs.tasks.pctsp:PCTSPGenerator",
        "SPCTSPGenerator": "logic.src.envs.tasks.spctsp:SPCTSPGenerator",
        "PDPGenerator": "logic.src.envs.tasks.pdp:PDPGenerator",
        "ThOPGenerator": "logic.src.envs.tasks.thop:ThOPGenerator",
        "PTPEnv": "logic.src.envs.tasks.ptp:PTPEnv",
        "CPTPEnv": "logic.src.envs.tasks.cptp:CPTPEnv",
        "TCPTPEnv": "logic.src.envs.tasks.tcptp:TCPTPEnv",
        "TSPEnv": "logic.src.envs.tasks.tsp:TSPEnv",
        "TSPkoptEnv": "logic.src.envs.tsp_kopt:TSPkoptEnv",
        "IRPEnv": "logic.src.envs.tasks.irp:IRPEnv",
        "ATSPEnv": "logic.src.envs.tasks.atsp:ATSPEnv",
        "CVRPEnv": "logic.src.envs.tasks.cvrp:CVRPEnv",
        "OPEnv": "logic.src.envs.tasks.op:OPEnv",
        "PCTSPEnv": "logic.src.envs.tasks.pctsp:PCTSPEnv",
        "SPCTSPEnv": "logic.src.envs.tasks.spctsp:SPCTSPEnv",
        "PDPEnv": "logic.src.envs.tasks.pdp:PDPEnv",
        "ThOPEnv": "logic.src.envs.tasks.thop:ThOPEnv",
        "ProblemSpec": "logic.src.envs.tasks.spec:ProblemSpec",
        "PROBLEMS": "logic.src.envs.tasks.spec:PROBLEMS",
    }
    if name not in mapping:
        raise AttributeError(f"module has no attribute {name!r}")
    obj = import_target(mapping[name])
    globals()[name] = obj
    return obj


__all__ = [
    "ENV_REGISTRY",
    "get_env",
    "RL4COEnvBase",
    "ImprovementEnvBase",
    "RoutingEnv",
    "Resource",
    "LoadResource",
    "ShiftTimeResource",
    "Generator",
    "RoutingGenerator",
    "PTPGenerator",
    "CPTPGenerator",
    "TCPTPGenerator",
    "CVRPGenerator",
    "TSPGenerator",
    "ATSPGenerator",
    "IRPGenerator",
    "OPGenerator",
    "PCTSPGenerator",
    "SPCTSPGenerator",
    "PDPGenerator",
    "ThOPGenerator",
    "PTPEnv",
    "CPTPEnv",
    "TCPTPEnv",
    "TSPEnv",
    "TSPkoptEnv",
    "IRPEnv",
    "ATSPEnv",
    "CVRPEnv",
    "OPEnv",
    "PCTSPEnv",
    "SPCTSPEnv",
    "PDPEnv",
    "ThOPEnv",
    "ProblemSpec",
    "PROBLEMS",
]

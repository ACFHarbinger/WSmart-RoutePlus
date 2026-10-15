"""Environments. Concrete envs load when ``get_env`` is called."""

from typing import Any, Dict

from logic.src.utils.registry import import_target

ENV_TARGETS: Dict[str, str] = {
    "ptp": "logic.src.envs.routing.ptp:PTPEnv",
    "cptp": "logic.src.envs.routing.cptp:CPTPEnv",
    "tcptp": "logic.src.envs.routing.tcptp:TCPTPEnv",
    "tsp": "logic.src.envs.routing.tsp:TSPEnv",
    "tsp_kopt": "logic.src.envs.tsp_kopt:TSPkoptEnv",
    "irp": "logic.src.envs.routing.irp:IRPEnv",
    "atsp": "logic.src.envs.routing.atsp:ATSPEnv",
    "cvrp": "logic.src.envs.routing.cvrp:CVRPEnv",
    "op": "logic.src.envs.routing.op:OPEnv",
    "pctsp": "logic.src.envs.routing.pctsp:PCTSPEnv",
    "spctsp": "logic.src.envs.routing.spctsp:SPCTSPEnv",
    "pdp": "logic.src.envs.routing.pdp:PDPEnv",
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
        "GENERATOR_REGISTRY": "logic.src.envs.generators:GENERATOR_REGISTRY",
        "ATSPGenerator": "logic.src.envs.generators:ATSPGenerator",
        "CVRPGenerator": "logic.src.envs.generators:CVRPGenerator",
        "Generator": "logic.src.envs.generators:Generator",
        "IRPGenerator": "logic.src.envs.generators:IRPGenerator",
        "OPGenerator": "logic.src.envs.generators:OPGenerator",
        "PCTSPGenerator": "logic.src.envs.generators:PCTSPGenerator",
        "PDPGenerator": "logic.src.envs.generators:PDPGenerator",
        "PTPGenerator": "logic.src.envs.generators:PTPGenerator",
        "get_generator": "logic.src.envs.generators:get_generator",
        "PTPEnv": "logic.src.envs.routing.ptp:PTPEnv",
        "CPTPEnv": "logic.src.envs.routing.cptp:CPTPEnv",
        "TCPTPEnv": "logic.src.envs.routing.tcptp:TCPTPEnv",
        "TSPEnv": "logic.src.envs.routing.tsp:TSPEnv",
        "TSPkoptEnv": "logic.src.envs.tsp_kopt:TSPkoptEnv",
        "IRPEnv": "logic.src.envs.routing.irp:IRPEnv",
        "ATSPEnv": "logic.src.envs.routing.atsp:ATSPEnv",
        "CVRPEnv": "logic.src.envs.routing.cvrp:CVRPEnv",
        "OPEnv": "logic.src.envs.routing.op:OPEnv",
        "PCTSPEnv": "logic.src.envs.routing.pctsp:PCTSPEnv",
        "SPCTSPEnv": "logic.src.envs.routing.spctsp:SPCTSPEnv",
        "PDPEnv": "logic.src.envs.routing.pdp:PDPEnv",
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
    "Generator",
    "PTPGenerator",
    "IRPGenerator",
    "ATSPGenerator",
    "CVRPGenerator",
    "OPGenerator",
    "PCTSPGenerator",
    "PDPGenerator",
    "get_generator",
    "GENERATOR_REGISTRY",
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
]

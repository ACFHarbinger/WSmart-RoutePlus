"""
Environment module for WSmart-Route.

This module provides RL4CO-style environment abstractions for
combinatorial optimization problems.

Attributes:
    PTPGenerator: Generator for Profitable Tour Problem instances.
    Generator: Abstract base class for all instance generators.
    IRPGenerator: Generator for Inventory Routing Problem instances.
    ATSPGenerator: Generator for Asymmetric TSP instances.
    CVRPGenerator: Generator for Capacitated VRP instances.
    OPGenerator: Generator for Orienteering Problem instances.
    PCTSPGenerator: Generator for Prize-Collecting TSP instances.
    PDPGenerator: Generator for Pickup and Delivery Problem instances.
    ThOPGenerator: Generator for Thief Orienteering Problem instances.
    TCMVPTPGenerator: Generator for Time-Constrained Multi-Vehicle Profitable Tour Problem instances.
    RL4COEnvBase: Abstract base class for all routing environments.
    ImprovementEnvBase: Base class for improvement-based routing environments.
    ENV_REGISTRY: Mapping of problem name strings to their environment classes.
    get_env: Factory function that instantiates an environment by name.

Example:
    >>> from logic.src.envs.routing import get_env
    >>> env = get_env("ptp", num_loc=50)
    >>> td = env.reset()
"""

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.envs.base.improvement import ImprovementEnvBase
from logic.src.envs.generators import (
    GENERATOR_REGISTRY,
    ATSPGenerator,
    CVRPGenerator,
    Generator,
    IRPGenerator,
    OPGenerator,
    PCTSPGenerator,
    PDPGenerator,
    PTPGenerator,
    TCMVPTPGenerator,
    ThOPGenerator,
    get_generator,
)
from logic.src.envs.routing.atsp import ATSPEnv
from logic.src.envs.routing.cvrp import CVRPEnv
from logic.src.envs.routing.irp import IRPEnv
from logic.src.envs.routing.mvptp import MVPTPEnv
from logic.src.envs.routing.op import OPEnv
from logic.src.envs.routing.pctsp import PCTSPEnv
from logic.src.envs.routing.pdp import PDPEnv
from logic.src.envs.routing.ptp import PTPEnv
from logic.src.envs.routing.spctsp import SPCTSPEnv
from logic.src.envs.routing.tcmvptp import TCMVPTPEnv
from logic.src.envs.routing.thop import ThOPEnv
from logic.src.envs.routing.tsp import TSPEnv
from logic.src.envs.tsp_kopt import TSPkoptEnv

# Environment registry
ENV_REGISTRY = {
    "ptp": PTPEnv,
    "mvptp": MVPTPEnv,
    "tsp": TSPEnv,
    "tsp_kopt": TSPkoptEnv,
    "thop": ThOPEnv,
    "irp": IRPEnv,
    "atsp": ATSPEnv,
    "cvrp": CVRPEnv,
    "op": OPEnv,
    "pctsp": PCTSPEnv,
    "spctsp": SPCTSPEnv,
    "pdp": PDPEnv,
    "tcmvptp": TCMVPTPEnv,
}


def get_env(name: str, **kwargs) -> RL4COEnvBase:
    """
    Factory function to get environment by name.

    Args:
        name: Environment name (ptp, mvptp, tcmvptp, irp, atsp, cvrp, op, etc.)
        kwargs: Environment configuration parameters.

    Returns:
        Initialized environment instance.

    Raises:
        ValueError: If environment name is not recognized.
    """
    name = name.lower()
    if name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment: {name}. Available: {list(ENV_REGISTRY.keys())}")
    return ENV_REGISTRY[name](**kwargs)


__all__ = [
    # Base classes
    "RL4COEnvBase",
    "ImprovementEnvBase",
    # Generators
    "Generator",
    "PTPGenerator",
    "IRPGenerator",
    "ATSPGenerator",
    "CVRPGenerator",
    "OPGenerator",
    "PCTSPGenerator",
    "PDPGenerator",
    "ThOPGenerator",
    "TCMVPTPGenerator",
    "get_generator",
    "GENERATOR_REGISTRY",
    # Environments
    "PTPEnv",
    "MVPTPEnv",
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
    "TCMVPTPEnv",
    # Registry
    "ENV_REGISTRY",
    "get_env",
]

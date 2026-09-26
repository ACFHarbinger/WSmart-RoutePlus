"""
Environment module for WSmart-Route.

This module provides RL4CO-style environment abstractions for
combinatorial optimization problems.

Attributes:
    WCVRPGenerator: Generator for Waste Collection VRP instances.
    VRPPGenerator: Generator for VRP with Profits instances.
    Generator: Abstract base class for all instance generators.
    IRPGenerator: Generator for Inventory Routing Problem instances.
    ATSPGenerator: Generator for Asymmetric TSP instances.
    CVRPGenerator: Generator for Capacitated VRP instances.
    OPGenerator: Generator for Orienteering Problem instances.
    PCTSPGenerator: Generator for Prize-Collecting TSP instances.
    PDPGenerator: Generator for Pickup and Delivery Problem instances.
    ThOPGenerator: Generator for Thief Orienteering Problem instances.
    CTOPGenerator: Generator for Capacitated Team Orienteering Problem instances.
    RL4COEnvBase: Abstract base class for all routing environments.
    ImprovementEnvBase: Base class for improvement-based routing environments.
    ENV_REGISTRY: Mapping of problem name strings to their environment classes.
    get_env: Factory function that instantiates an environment by name.

Example:
    >>> from logic.src.envs.routing import get_env
    >>> env = get_env("vrpp", num_loc=50)
    >>> td = env.reset()
"""

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.envs.base.improvement import ImprovementEnvBase
from logic.src.envs.generators import (
    GENERATOR_REGISTRY,
    ATSPGenerator,
    CTOPGenerator,
    CVRPGenerator,
    Generator,
    IRPGenerator,
    OPGenerator,
    PCTSPGenerator,
    PDPGenerator,
    ThOPGenerator,
    VRPPGenerator,
    WCVRPGenerator,
    get_generator,
)
from logic.src.envs.routing.atsp import ATSPEnv
from logic.src.envs.routing.ctop import CTOPEnv
from logic.src.envs.routing.cvrp import CVRPEnv
from logic.src.envs.routing.cvrpp import CVRPPEnv
from logic.src.envs.routing.cwcvrp import CWCVRPEnv
from logic.src.envs.routing.irp import IRPEnv
from logic.src.envs.routing.op import OPEnv
from logic.src.envs.routing.pctsp import PCTSPEnv
from logic.src.envs.routing.pdp import PDPEnv
from logic.src.envs.routing.spctsp import SPCTSPEnv
from logic.src.envs.routing.swcvrp import SCWCVRPEnv
from logic.src.envs.routing.thop import ThOPEnv
from logic.src.envs.routing.tsp import TSPEnv
from logic.src.envs.routing.vrpp import VRPPEnv
from logic.src.envs.routing.wcvrp import WCVRPEnv
from logic.src.envs.tsp_kopt import TSPkoptEnv

# Environment registry
ENV_REGISTRY = {
    "vrpp": VRPPEnv,
    "cvrpp": CVRPPEnv,
    "wcvrp": WCVRPEnv,
    "cwcvrp": CWCVRPEnv,
    "scwcvrp": SCWCVRPEnv,
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
    "ctop": CTOPEnv,
}


def get_env(name: str, **kwargs) -> RL4COEnvBase:
    """
    Factory function to get environment by name.

    Args:
        name: Environment name (vrpp, cvrpp, wcvrp, irp, atsp, cvrp, op, etc.)
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
    "VRPPGenerator",
    "WCVRPGenerator",
    "IRPGenerator",
    "ATSPGenerator",
    "CVRPGenerator",
    "OPGenerator",
    "PCTSPGenerator",
    "PDPGenerator",
    "ThOPGenerator",
    "CTOPGenerator",
    "get_generator",
    "GENERATOR_REGISTRY",
    # Environments
    "VRPPEnv",
    "CVRPPEnv",
    "WCVRPEnv",
    "CWCVRPEnv",
    "SCWCVRPEnv",
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
    "CTOPEnv",
    # Registry
    "ENV_REGISTRY",
    "get_env",
]

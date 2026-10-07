"""
Environment module for WSmart-Route.

This module provides RL4CO-style environment abstractions for
combinatorial optimization problems.

Attributes:
    ENV_REGISTRY: Registry of all environments.
    get_env: Factory function to get environment by name.

Example:
    >>> env = get_env("vrpp", num_loc=25, seed=42)
    >>> env
    <logic.src.envs.routing.vrpp.VRPPEnv object at 0x...>
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
    VRPPGenerator,
    get_generator,
)
from logic.src.envs.routing.atsp import ATSPEnv
from logic.src.envs.routing.cvrp import CVRPEnv
from logic.src.envs.routing.cvrpp import CVRPPEnv
from logic.src.envs.routing.irp import IRPEnv
from logic.src.envs.routing.op import OPEnv
from logic.src.envs.routing.pctsp import PCTSPEnv
from logic.src.envs.routing.pdp import PDPEnv
from logic.src.envs.routing.spctsp import SPCTSPEnv
from logic.src.envs.routing.tsp import TSPEnv
from logic.src.envs.routing.vrpp import VRPPEnv
from logic.src.envs.tsp_kopt import TSPkoptEnv

# Environment registry
ENV_REGISTRY = {
    "vrpp": VRPPEnv,
    "cvrpp": CVRPPEnv,
    "tsp": TSPEnv,
    "tsp_kopt": TSPkoptEnv,
    "irp": IRPEnv,
    "atsp": ATSPEnv,
    "cvrp": CVRPEnv,
    "op": OPEnv,
    "pctsp": PCTSPEnv,
    "spctsp": SPCTSPEnv,
    "pdp": PDPEnv,
}


def get_env(name: str, **kwargs) -> RL4COEnvBase:
    """
    Factory function to get environment by name.



    Args:
        name: Description of name.
        kwargs: Description of kwargs.

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
    "IRPGenerator",
    "ATSPGenerator",
    "CVRPGenerator",
    "OPGenerator",
    "PCTSPGenerator",
    "PDPGenerator",
    "get_generator",
    "GENERATOR_REGISTRY",
    # Environments
    "VRPPEnv",
    "CVRPPEnv",
    "TSPEnv",
    "TSPkoptEnv",
    "IRPEnv",
    "ATSPEnv",
    "CVRPEnv",
    "OPEnv",
    "PCTSPEnv",
    "SPCTSPEnv",
    "PDPEnv",
    # Registry
    "ENV_REGISTRY",
    "get_env",
]

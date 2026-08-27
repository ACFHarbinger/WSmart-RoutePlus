"""
Problem instance generators sub-package.

Attributes:
    GENERATOR_REGISTRY (dict[str, type[Generator]]): Registry of available generators.
    get_generator: Factory function for getting a generator.

Examples:
    >>> from src.envs.generators import get_generator
    >>> generator = get_generator("vrpp", num_loc=20)
    >>> problem = generator.generate()
    >>> problem
    <ProblemInstance: ...>
"""

from typing import Any

from .atsp import ATSPGenerator
from .base import Generator
from .cvrp import CVRPGenerator
from .irp import IRPGenerator
from .op import OPGenerator
from .pctsp import PCTSPGenerator
from .pdp import PDPGenerator
from .scwcvrp import SCWCVRPGenerator
from .thop import ThOPGenerator
from .tsp import TSPGenerator
from .ttop import TTOPGenerator
from .vrpp import VRPPGenerator
from .wcvrp import WCVRPGenerator

# Registry of available generators
GENERATOR_REGISTRY: dict[str, type[Generator]] = {
    "vrpp": VRPPGenerator,
    "cvrpp": VRPPGenerator,  # Same generator, different env handles capacity
    "ttop": TTOPGenerator,
    "wcvrp": WCVRPGenerator,
    "cwcvrp": WCVRPGenerator,
    "scwcvrp": SCWCVRPGenerator,
    "tsp": TSPGenerator,
    "irp": IRPGenerator,
    "atsp": ATSPGenerator,
    "cvrp": CVRPGenerator,
    "op": OPGenerator,
    "pctsp": PCTSPGenerator,
    "spctsp": PCTSPGenerator,  # SPCTSP reuses the PCTSP generator
    "pdp": PDPGenerator,
    "thop": ThOPGenerator,
}


def get_generator(name: str, **kwargs: Any) -> Generator:
    """
    Get a generator by name.



    Args:
        name: Generator name (e.g., "vrpp", "wcvrp", "tsp", "irp", "atsp", "cvrp").
        kwargs: Generator configuration parameters.

    Returns:
        Configured Generator instance.

    Raises:
        ValueError: If generator name is not found.
    """
    if name not in GENERATOR_REGISTRY:
        raise ValueError(f"Unknown generator: {name}. Available: {list(GENERATOR_REGISTRY.keys())}")

    return GENERATOR_REGISTRY[name](**kwargs)


__all__ = [
    "Generator",
    "VRPPGenerator",
    "WCVRPGenerator",
    "SCWCVRPGenerator",
    "TSPGenerator",
    "IRPGenerator",
    "ATSPGenerator",
    "CVRPGenerator",
    "OPGenerator",
    "PCTSPGenerator",
    "PDPGenerator",
    "ThOPGenerator",
    "TTOPGenerator",
    "GENERATOR_REGISTRY",
    "get_generator",
]

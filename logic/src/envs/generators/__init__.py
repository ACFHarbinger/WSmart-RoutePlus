"""
Problem instance generators sub-package.

Attributes:
    GENERATOR_REGISTRY (dict[str, type[Generator]]): Registry of available generators.
    get_generator: Factory function for getting a generator.

Examples:
    >>> from src.envs.generators import get_generator
    >>> generator = get_generator("ptp", num_loc=20)
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
from .ptp import PTPGenerator
from .tcmvptp import TCMVPTPGenerator
from .thop import ThOPGenerator
from .tsp import TSPGenerator

# Registry of available generators
GENERATOR_REGISTRY: dict[str, type[Generator]] = {
    "ptp": PTPGenerator,
    "mvptp": PTPGenerator,  # Same generator, different env handles capacity
    "tcmvptp": TCMVPTPGenerator,
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
        name: Generator name (e.g., "ptp", "mvptp", "tsp", "irp", "atsp", "cvrp").
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
    "PTPGenerator",
    "TSPGenerator",
    "IRPGenerator",
    "ATSPGenerator",
    "CVRPGenerator",
    "OPGenerator",
    "PCTSPGenerator",
    "PDPGenerator",
    "ThOPGenerator",
    "TCMVPTPGenerator",
    "GENERATOR_REGISTRY",
    "get_generator",
]

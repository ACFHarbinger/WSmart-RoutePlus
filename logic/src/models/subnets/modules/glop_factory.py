"""
GLOP Adapter Factory.

Attributes:
    ADAPTER_REGISTRY (dict): Mapping of environment names to their corresponding Adapter classes.
    get_adapter: Factory function for retrieving problem-specific adapters.

Example:
    >>> from logic.src.models.subnets.modules.glop_factory import get_adapter
    >>> adapter_cls = get_adapter("tsp")
"""

from __future__ import annotations

from .tsp_adapter import TSPAdapter
from .vrp_adapter import VRPAdapter

# Adapter registry
ADAPTER_REGISTRY = {
    "tsp": TSPAdapter,
    "cvrp": VRPAdapter,
    "vrpp": TSPAdapter,  # VRP variants can use TSP for subproblems
}


def get_adapter(env_name: str) -> type:
    """Retrieves the appropriate adapter class for a given environment.

    Args:
        env_name (str): Name of the environment (e.g., 'tsp', 'cvrp').

    Returns:
        type: The SubproblemAdapter subclass for the environment.
    """
    if env_name in ADAPTER_REGISTRY:
        return ADAPTER_REGISTRY[env_name]
    # Default to TSP for unknown environments
    return TSPAdapter

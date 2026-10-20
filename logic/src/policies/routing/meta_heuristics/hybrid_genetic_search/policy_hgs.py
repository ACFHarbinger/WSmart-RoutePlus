"""
HGS Policy Adapter.

Adapts the Hybrid Genetic Search (HGS) logic to the common policy interface.
Now agnostic to bin selection.

Attributes:
    HGSPolicy: Policy adapter class for the Hybrid Genetic Search.

Example:
    >>> policy = HGSPolicy(config)
    >>> routes, profit, cost = policy._run_solver(...)
"""

import math
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

from logic.src.configs.policies import HGSConfig
from logic.src.enums import PolicyTag
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.policies.routing.base.generic_policy import GenericRoutingPolicy
from logic.src.policies.routing.base.sub_problem import SubProblem
from logic.src.utils.registry import ALL

from .dispatcher import run_hgs


@ALL.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.POPULATION_BASED,
    PolicyTag.EVOLUTIONARY_ALGORITHM,
    PolicyTag.MEMETIC_SEARCH,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
    PolicyTag.PARALLELIZABLE,
)
@RoutingAlgorithmRegistry.register("hgs")
class HGSPolicy(GenericRoutingPolicy):
    """
    Hybrid Genetic Search (HGS) Policy - State-of-the-Art Evolutionary Routing.

    This policy implements the Hybrid Genetic Search algorithm, specifically
    optimized for Capacitated Vehicle Routing Problems (CVRP) and variants. It
    is widely considered one of the most powerful heuristics for VRP due to its
    synergistic combination of population management and aggressive local search.

    Algorithm Logic:
    1.  **Exploration (Genetic)**: Maintains a population of solutions and
        generates offspring using advanced crossover operators (e.g., OX)
        that preserve spatial clusters and route segments.
    2.  **Intensification (Local Search)**: Every offspring undergoes a rigorous
        "education" phase using a highly optimized Local Search engine (SWAP*,
        Relocate, 2-Opt) to reach a local optimum.
    3.  **Diversity Management**: Employs a biased fitness function that rewards
        both quality (low cost) and diversity (uniqueness in the population),
        preventing premature convergence and maintaining a healthy search.

    HGS is the primary engine for solving large-scale, static routing instances
    within the framework.

    Registry key: ``"hgs"``

    Attributes:
        config (Optional[Union[HGSConfig, Dict[str, Any]]]): Configuration object.

    Example:
        >>> policy = HGSPolicy(config)
    """

    _solver = run_hgs

    def __init__(self, config: Optional[Union[HGSConfig, Dict[str, Any]]] = None):
        """Initialize HGS policy with optional config.

        Args:
            config: HGSConfig dataclass, raw dict from YAML, or None.

        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """
        Get the configuration class for HGS.

        Returns:
            Optional[Type]: The HGSConfig class.
        """
        return HGSConfig

    def _get_config_key(self) -> str:
        """
        Return config key for HGS.

        Returns:
            str: The configuration key "hgs".
        """
        return "hgs"

    def _make_subproblem(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        kwargs: Dict[str, Any],
    ) -> SubProblem:
        """Assemble the sub-problem, projecting lat/lng to a local equirectangular plane.

        The projection is required for correct polar sector angles in SWAP*
        pruning: raw lat/lng cannot be used directly in atan2 — longitude
        degrees are shorter than latitude degrees by a factor of cos(lat),
        causing ~22% distortion at 39°N.
        """
        problem = super()._make_subproblem(
            sub_dist_matrix, sub_wastes, capacity, revenue, cost_unit, values, mandatory_nodes, kwargs
        )
        x_coords = problem.x_coords
        y_coords = problem.y_coords
        if x_coords is not None and y_coords is not None:
            depot_lat = float(y_coords[0])
            depot_lng = float(x_coords[0])
            cos_lat = math.cos(math.radians(depot_lat))
            problem.x_coords = (x_coords - depot_lng) * cos_lat
            problem.y_coords = y_coords - depot_lat
        return problem

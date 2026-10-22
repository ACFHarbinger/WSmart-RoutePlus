"""GA (Genetic Algorithm) Policy Adapter.

Attributes:
    GAPolicy: Policy class for Genetic Algorithm.

Example:
    >>> from logic.src.configs.policies.routing.meta_heuristics.ga import GAConfig
    >>> config = GAConfig(pop_size=50)
    >>> policy = GAPolicy(config)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.routing.meta_heuristics.ga import GAConfig
from logic.src.enums import PolicyTag
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.utils.registry import ALL

from .solver import GASolver


@ALL.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.POPULATION_BASED,
    PolicyTag.EVOLUTIONARY_ALGORITHM,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("ga")
class GAPolicy(BaseRoutingPolicy):
    """Genetic Algorithm (GA) Policy - Population-Based Evolutionary Routing.

    Attributes:
        config: Configuration for the policy.
    """

    def __init__(self, config: Optional[Union[GAConfig, Dict[str, Any]]] = None):
        """Initializes the GA policy.

        Args:
            config: Configuration source for the Genetic Algorithm.

        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Returns the configuration class for GA.

        Returns:
            The GAConfig class.
        """
        return GAConfig

    def _get_config_key(self) -> str:
        """Returns the configuration key for the GA policy.

        Returns:
            The registry key 'ga'.
        """
        return "ga"

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """Execute the Genetic Algorithm (GA) solver logic.

        GA is a population-based search heuristic inspired by natural selection.

        Args:
            sub_dist_matrix: Symmetric distance matrix.
            sub_wastes: Mapping of local node indices to waste levels.
            capacity: Maximum vehicle collection capacity.
            revenue: Revenue per kilogram of waste.
            cost_unit: Monetary cost per kilometer.
            values: Merged configuration dictionary.
            mandatory_nodes: Local indices of bins that MUST be collected.
            kwargs: Additional context.

        Returns:
            Tuple of (routes, profit, cost).
        """
        params = GAConfig(
            pop_size=int(values.get("pop_size", 30)),
            max_generations=int(values.get("max_generations", 100)),
            crossover_rate=float(values.get("crossover_rate", 0.8)),
            mutation_rate=float(values.get("mutation_rate", 0.1)),
            tournament_size=int(values.get("tournament_size", 3)),
            n_removal=int(values.get("n_removal", 2)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=values.get("seed", 42),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = GASolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()

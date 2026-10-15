r"""ABC Policy Adapter.

Adapts the Artificial Bee Colony (ABC) solver to the agnostic
BaseRoutingPolicy interface.

Attributes:
    ABCPolicy: Adapter class for the ABC solver.

Example:
    >>> from logic.src.configs.policies.abc import ABCConfig
    >>> from logic.src.policies.routing.meta_heuristics.artificial_bee_colony import ABCPolicy
    >>> config = ABCConfig(n_sources=20)
    >>> policy = ABCPolicy(config)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.abc import ABCConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from logic.src.configs.policies.abc import ABCConfig
from .solver import ABCSolver


@RoutingAlgorithmRegistry.register("abc")
class ABCPolicy(BaseRoutingPolicy):
    """
    ABC policy class.

    Visits bins using the Artificial Bee Colony algorithm.

    Attributes:
        config: Configuration for the policy.
    """

    def __init__(self, config: Optional[Union[ABCConfig, Dict[str, Any]]] = None):
        """Initializes the ABC policy.

        Args:
            config (Optional[Union[ABCConfig, Dict[str, Any]]]): Configuration
                source for the Artificial Bee Colony.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Returns the configuration class for ABC.

        Returns:
            Optional[Type]: The ABCConfig class.
        """
        return ABCConfig

    def _get_config_key(self) -> str:
        """Returns the configuration key for the ABC policy.

        Returns:
            str: The registry key 'abc'.
        """
        return "abc"

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
        """
        Execute the Artificial Bee Colony (ABC) solver logic.

        ABC is a population-based metaheuristic inspired by the foraging
        behavior of honey bees. It maintains a population of solutions
        (food sources), which are modified by three types of bees:
        - Employed Bees: Search around known food sources.
        - Onlooker Bees: Select high-quality food sources and search around
          them (exploitation).
        - Scout Bees: Abandon poor sources and find new random ones
          (exploration).
        The search process includes local search and removal operators for
        intensification.

        Args:
            sub_dist_matrix: Square distance matrix.
            sub_wastes: Node fill levels.
            capacity: Vehicle capacity.
            revenue: Revenue per kg.
            cost_unit: Cost per km.
            values: Merged config dictionary.
            mandatory_nodes: Nodes that must be visited.
            kwargs: Additional context.

        Returns:
            Tuple containing (routes, profit, cost).
        """
        params = ABCConfig(
            n_sources=int(values.get("n_sources", 20)),
            limit=int(values.get("limit", 10)),
            max_iterations=int(values.get("max_iterations", 200)),
            n_removal=int(values.get("n_removal", 1)),
            local_search_iterations=int(values.get("local_search_iterations", 500)),
            time_limit=float(values.get("time_limit", 60.0)),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = ABCSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost

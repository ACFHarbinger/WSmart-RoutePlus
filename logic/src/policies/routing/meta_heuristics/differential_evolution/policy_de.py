r"""Policy adapter for Memetic Differential Evolution (MDE/rand/1/exp).

Provides the interface between the MDE solver and the policy factory system.

Attributes:
    DEPolicyAdapter: Policy adapter for Memetic Differential Evolution.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.differential_evolution.policy_de import DEPolicyAdapter
    >>> policy = DEPolicyAdapter()
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.de import DEConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from logic.src.configs.policies.de import DEConfig
from .solver import DESolver


@RoutingAlgorithmRegistry.register("de")
class DEPolicyAdapter(BaseRoutingPolicy):
    """Policy adapter for Memetic Differential Evolution (MDE).

    MDE hybridizes the global exploratory power of DE/rand/1/exp (Storn & Price,
    1997) with memetic local search for discrete optimization. The adapter
    coordinates:

    - Global search via continuous Random Key mutation & Exponential crossover
    - Local reinforcement via discrete TSP-based refinement (memetic addition)
    - Dynamic population scaling to ensure mutual exclusivity axioms
    - Mathematical rigor in nomenclature (MDE/rand/1/exp)

    Reference:
        Storn, R., & Price, K. (1997). "Differential Evolution – A Simple and
        Efficient Heuristic for Global Optimization over Continuous Spaces."
        Journal of Global Optimization, 11(4), 341-359.

    Attributes:
        solver: Internal solver instance.
        params: Algorithm parameters.

    Example:
        >>> policy = DEPolicyAdapter()
    """

    def __init__(self, config: Optional[Union[DEConfig, Dict[str, Any]]] = None):
        """Initialize DE policy adapter.

        Args:
            config: DE configuration parameters. If None, uses defaults.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Returns the configuration class for DE.

        Returns:
            Optional[Type]: The DEConfig class.
        """
        return DEConfig

    def _get_config_key(self) -> str:
        """Returns the configuration key for the DE policy.

        Returns:
            str: The registry key 'de'.
        """
        return "de"

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
        Execute the Memetic Differential Evolution (MDE) solver logic.

        MDE (specifically MDE/rand/1/exp) adapts the continuous Differential
        Evolution algorithm for the discrete VRPP. It uses a Random Key
        representation where continuous vectors are mapped to discrete node
        sequences. Global exploration is achieved through mutation and
        exponential crossover on these vectors, while exploitation is
        strengthened by discrete local search (memetic phase) applied to the
        resulting tours.

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
        params = DEConfig(
            pop_size=values.get("pop_size", 50),
            mutation_factor=values.get("mutation_factor", 0.8),
            crossover_rate=values.get("crossover_rate", 0.9),
            n_removal=values.get("n_removal", 3),
            max_iterations=values.get("max_iterations", 1000),
            local_search_iterations=values.get("local_search_iterations", 100),
            evolution_strategy=values.get("evolution_strategy", "lamarckian"),
            time_limit=values.get("time_limit", 60.0),
            seed=values.get("seed", 42),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        solver = DESolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        return solver.solve()

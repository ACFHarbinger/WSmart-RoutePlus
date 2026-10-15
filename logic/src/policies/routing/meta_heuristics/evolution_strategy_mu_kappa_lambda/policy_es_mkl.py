r"""(μ,κ,λ) Evolution Strategy Policy Adapter with Age-Based Selection.

Adapts the (μ,κ,λ)-ES with age control for vehicle routing problems.
Individuals exceeding age κ are excluded from selection.

Attributes:
    MuKappaLambdaESPolicy: Policy class for (μ,κ,λ)-ES.

Example:
    >>> policy = MuKappaLambdaESPolicy()
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies import MuKappaLambdaESConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from logic.src.configs.policies.es_mkl import MuKappaLambdaESConfig
from .solver import MuKappaLambdaESSolver


@RoutingAlgorithmRegistry.register("es_mkl")
class MuKappaLambdaESPolicy(BaseRoutingPolicy):
    """(μ,κ,λ) Evolution Strategy policy with age-based selection.

    Age-limited parent survival prevents stagnation in long runs.

    Attributes:
        config: Configuration for the policy.
    """

    def __init__(self, config: Optional[Union[MuKappaLambdaESConfig, Dict[str, Any]]] = None):
        """Initialize (μ,κ,λ)-ES policy with optional config.

        Args:
            config: Configuration object or dictionary.

        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the configuration class for this policy.

        Returns:
            MuKappaLambdaESConfig class.
        """
        return MuKappaLambdaESConfig

    def _get_config_key(self) -> str:
        """Return the configuration key for this policy.

        Returns:
            The string key "es_mkl".
        """
        return "es_mkl"

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
        """Execute the (mu, kappa, lambda) Evolution Strategy (ES) solver logic.

        (mu, kappa, lambda)-ES is a sophisticated evolutionary algorithm that
        introduces an "age" control parameter (kappa):
        - mu: The number of parent individuals.
        - lambda: The number of offspring generated.
        - kappa: The maximum age (in generations) an individual can survive.
        - Selection: Parents can survive across generations but are strictly
          removed if they exceed age kappa.

        Args:
            sub_dist_matrix: Symmetric distance matrix.
            sub_wastes: Mapping of local node indices to waste quantities.
            capacity: Maximum vehicle collection capacity.
            revenue: Revenue obtained per kilogram of waste.
            cost_unit: Monetary cost incurred per kilometer.
            values: Merged configuration dictionary.
            mandatory_nodes: Local indices of bins that MUST be collected.
            kwargs: Additional context.

        Returns:
            Tuple of (routes, profit, cost).
        """
        tau_default: float = 1.0 / (2.0**0.5)
        params = MuKappaLambdaESConfig(
            mu=values.get("mu", 10),
            kappa=values.get("kappa", 5),
            lambda_=values.get("lambda_", 5),
            rho=values.get("rho", 2),
            tau_local=values.get("tau_local", tau_default),
            tau_global=values.get("tau_global", tau_default),
            initial_sigma=values.get("initial_sigma", 1.0),
            recombination_type=values.get("recombination_type", "intermediate"),
            max_iterations=values.get("max_iterations", 500),
            time_limit=values.get("time_limit", 60.0),
            min_sigma=values.get("min_sigma", 1e-10),
            max_sigma=values.get("max_sigma", 10.0),
            bounds_min=values.get("bounds_min", -5.0),
            bounds_max=values.get("bounds_max", 5.0),
            n_removal=values.get("n_removal", 3),
            stagnation_limit=values.get("stagnation_limit", 10),
            local_search_iterations=values.get("local_search_iterations", 100),
            seed=values.get("seed", 42),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
        )

        # Initialize the generalized self-adaptive solver
        solver = MuKappaLambdaESSolver(
            dist_matrix=sub_dist_matrix,
            wastes=sub_wastes,
            capacity=capacity,
            R=revenue,
            C=cost_unit,
            params=params,
            mandatory_nodes=mandatory_nodes,
        )

        routes, profit, cost = solver.solve()
        return routes, profit, cost

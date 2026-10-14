"""TS (Tabu Search) Policy Adapter.

Attributes:
    TSPolicy: Policy class for Tabu Search.

Example:
    >>> policy = TSPolicy(config)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.ts import TSConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.policies.routing.meta_heuristics.tabu_search.params import TSParams
from logic.src.policies.routing.meta_heuristics.tabu_search.solver import TSSolver


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.LOCAL_SEARCH,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("ts")
class TSPolicy(BaseRoutingPolicy):
    """Tabu Search (TS) Policy.

    Attributes:
        config: Configuration for the policy.
    """

    def __init__(self, config: Optional[Union[TSConfig, Dict[str, Any]]] = None):
        """Initializes the Tabu Search policy.

        Args:
            config: Configuration source.

        Returns:
            None.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Returns the configuration class for TS.

        Returns:
            TSConfig class.
        """
        return TSConfig

    def _get_config_key(self) -> str:
        """Returns the configuration key for the TS policy.

        Returns:
            The registry key 'ts'.
        """
        return "ts"

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
        """Execute the Tabu Search (TS) metaheuristic solver logic.

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
        params = TSParams(
            # Short-term memory
            tabu_tenure=int(values.get("tabu_tenure", 7)),
            dynamic_tenure=bool(values.get("dynamic_tenure", True)),
            min_tenure=int(values.get("min_tenure", 5)),
            max_tenure=int(values.get("max_tenure", 15)),
            # Aspiration criteria
            aspiration_enabled=bool(values.get("aspiration_enabled", True)),
            # Long-term memory
            intensification_enabled=bool(values.get("intensification_enabled", True)),
            diversification_enabled=bool(values.get("diversification_enabled", True)),
            intensification_interval=int(values.get("intensification_interval", 100)),
            diversification_interval=int(values.get("diversification_interval", 200)),
            elite_size=int(values.get("elite_size", 5)),
            frequency_penalty_weight=float(values.get("frequency_penalty_weight", 0.1)),
            # Candidate list
            candidate_list_enabled=bool(values.get("candidate_list_enabled", True)),
            candidate_list_size=int(values.get("candidate_list_size", 20)),
            # Strategic oscillation
            oscillation_enabled=bool(values.get("oscillation_enabled", False)),
            feasibility_tolerance=float(values.get("feasibility_tolerance", 0.1)),
            # General search
            max_iterations=int(values.get("max_iterations", 5000)),
            max_iterations_no_improve=int(values.get("max_iterations_no_improve", 500)),
            n_removal=int(values.get("n_removal", 3)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
            vrpp=values.get("vrpp", True),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
            # Neighborhood structure
            use_swap=bool(values.get("use_swap", True)),
            use_relocate=bool(values.get("use_relocate", True)),
            use_2opt=bool(values.get("use_2opt", True)),
            use_insertion=bool(values.get("use_insertion", True)),
        )

        solver = TSSolver(
            sub_dist_matrix,
            sub_wastes,
            capacity,
            revenue,
            cost_unit,
            params,
            mandatory_nodes,
        )

        return solver.solve()

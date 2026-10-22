"""
SS-HH Policy Adapter.

Adapts the Sequence-based Selection Hyper-Heuristic (SS-HH) solver to the
agnostic BaseRoutingPolicy interface.

Attributes:
    SSHHPolicy: Adapter for the SS-HH solver.

Example:
    >>> from logic.src.policies.routing.hyper_heuristics import SSHHPolicy
    >>> params = SSHHPolicy()
    >>> print(params)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.routing.hyper_heuristics.ss_hh import SSHHConfig
from logic.src.enums import PolicyTag
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.utils.registry import ALL

from .solver import SSHHSolver


@ALL.register(
    PolicyTag.HYPER_HEURISTIC,
    PolicyTag.TRAJECTORY_BASED,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("ss_hh")
class SSHHPolicy(BaseRoutingPolicy):
    """
    SS-HH policy class.

    Visits bins using the online-learning Sequence-based Selection Hyper-Heuristic.

    Attributes:
        config: Configuration parameters.
    """

    def __init__(self, config: Optional[Union[SSHHConfig, Dict[str, Any]]] = None):
        """
        Initializes the Sequence-Based Selection Hyper-Heuristic policy.

        Args:
            config: Configuration parameters.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """
        Get the configuration class.

        Returns:
            Optional[Type]: The configuration class.
        """
        return SSHHConfig

    def _get_config_key(self) -> str:
        """
        Get the configuration key.

        Returns:
            str: The configuration key.
        """
        return "ss_hh"

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
        Execute the Sequence-based Selection Hyper-Heuristic (SS-HH) solver logic.

        SS-HH is an online-learning hyper-heuristic that models the search as a
        sequence of moves. It learns the transition probabilities between
        low-level heuristics based on their historical performance in discovering
        improving or intensifying solutions.

        The solver progressively refines the current solution by selecting the
        most promising sequences of heuristics, guided by a decay-based
        acceptance threshold.

        Args:
            sub_dist_matrix: Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes: Mapping of local node indices to their
                current bin inventory levels.
            capacity: Maximum vehicle collection capacity.
            revenue: Revenue obtained per kilogram of waste collected.
            cost_unit: Monetary cost incurred per kilometer traveled.
            values: Merged configuration dictionary containing
                hyperparameters like `threshold_infeasible`, `threshold_decay_rate`, etc.
            mandatory_nodes: Local indices of bins that MUST be
                collected in this period.
            kwargs: Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes for the current day.
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        params = SSHHConfig(
            max_iterations=int(values.get("max_iterations", 500)),
            n_removal=int(values.get("n_removal", 2)),
            n_llh=int(values.get("n_llh", 5)),
            time_limit=float(values.get("time_limit", 60.0)),
            threshold_infeasible=float(values.get("threshold_infeasible", 0.001)),
            threshold_feasible_base=float(values.get("threshold_feasible_base", 0.0001)),
            threshold_decay_rate=float(values.get("threshold_decay_rate", 0.01)),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = SSHHSolver(
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

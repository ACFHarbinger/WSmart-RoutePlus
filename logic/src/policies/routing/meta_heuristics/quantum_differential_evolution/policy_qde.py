"""
QDE Policy Adapter.

Adapts the Quantum-Inspired Differential Evolution (QDE) logic to the
agnostic BaseRoutingPolicy interface.

Attributes:
    QDEPolicy: Policy class registering the QDE solver under key "qde".

Example:
    >>> from logic.src.policies.routing.meta_heuristics.quantum_differential_evolution.policy_qde import QDEPolicy
    >>> policy = QDEPolicy()
    >>> routes, profit, cost = policy.run(dist_matrix, wastes, capacity, revenue, cost_unit, values, mandatory_nodes)
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.routing.meta_heuristics.qde import QDEConfig
from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .solver import QDESolver


@RoutingAlgorithmRegistry.register("qde")
class QDEPolicy(BaseRoutingPolicy):
    """
    QDE policy class.

    Visits bins using Quantum-Inspired Differential Evolution.

    Attributes:
        config: QDEConfig or equivalent dict with solver hyperparameters.
    """

    def __init__(self, config: Optional[Union[QDEConfig, Dict[str, Any]]] = None):
        """
        Initializes the Quantum Differential Evolution policy.

        Args:
            config: Optional configuration dictionary or Hydra config.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the config dataclass used by this policy.

        Returns:
            Type: QDEConfig dataclass.
        """
        return QDEConfig

    def _get_config_key(self) -> str:
        """Return the YAML config key for this solver.

        Returns:
            str: Config key "qde".
        """
        return "qde"

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
        Execute the Quantum-Inspired Differential Evolution (QDE) solver logic.

        QDE is a hybrid metaheuristic that maps the principles of quantum
        computation (qubits and quantum gates) onto the Differential Evolution
        (DE) framework. In this implementation:
        - Quantum Population: Candidates are represented as quantum-state vectors.
        - Differential Evolution operators (mutation, crossover) are applied
          to the qubits' rotation angles.
        - Observation: Quantum states are collapsed ("measured") into discrete
          VRPP solutions periodically.
        This approach leverages the high exploration capability of quantum-state
        superpositions while utilizing DE's robust intensification mechanisms.

        Args:
            sub_dist_matrix: Symmetric distance matrix for the current
                sub-problem nodes.
            sub_wastes: Mapping of local node indices to their
                current bin inventory levels.
            capacity: Maximum vehicle collection capacity.
            revenue: Revenue obtained per kilogram of waste collected.
            cost_unit: Monetary cost incurred per kilometer traveled.
            values: Merged configuration dictionary containing
                QDE parameters (pop_size, F, CR, delta_theta).
            mandatory_nodes: Local indices of bins that MUST be
                collected in this period.
            kwargs: Additional context, including:
                - search_context (Optional[SearchContext]): Context for tracking
                  recursive solver statistics.
                - multi_day_context (Optional[MultiDayContext]): Context for
                  inter-day state propagation.

        Returns:
            Tuple[List[List[int]], float, float]: A 3-tuple containing:
                - routes: Optimized collection routes (list-of-lists, local indices).
                - profit: Total calculated net profit (Total Revenue - Total Cost).
                - cost: Total travel cost calculated by the solver.
        """
        params = QDEConfig(
            pop_size=int(values.get("pop_size", 20)),
            F=float(values.get("F", 0.5)),
            CR=float(values.get("CR", 0.7)),
            max_iterations=int(values.get("max_iterations", 200)),
            time_limit=float(values.get("time_limit", 60.0)),
            delta_theta=float(values.get("delta_theta", 0.01 * 3.14159)),
            local_search_iterations=int(values.get("local_search_iterations", 100)),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = QDESolver(
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

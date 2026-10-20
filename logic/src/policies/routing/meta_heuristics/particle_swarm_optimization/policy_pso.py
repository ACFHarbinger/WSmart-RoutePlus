"""Particle Swarm Optimization (PSO) policy.

Attributes:
    PSOPolicyAdapter: Policy class for Particle Swarm Optimization.

Example:
    >>>     >>> config = PSOConfig(pop_size=50)
    >>> policy = PSOPolicyAdapter(config)
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from logic.src.configs.policies.pso import PSOConfig
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.routing.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .solver import PSOSolver


@RoutingAlgorithmRegistry.register("pso")
class PSOPolicyAdapter(BaseMultiPeriodRoutingPolicy):
    """Policy adapter for Particle Swarm Optimization.

    Attributes:
        config: Configuration for the policy.
    """

    def __init__(self, config: Optional[Union[PSOConfig, Dict[str, Any]]] = None):
        """Initialize PSO policy adapter.

        Args:
            config: Configuration source.

        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Hydra-compliant configuration schema.

        Returns:
            PSOConfig class.
        """
        return PSOConfig

    def _get_config_key(self) -> str:
        """Identifier for this policy.

        Returns:
            The key 'pso'.
        """
        return "pso"

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """Execute the Particle Swarm Optimization (PSO) solver logic.

        Args:
            problem: Current ProblemContext.
            multi_day_ctx: Optional inter-day state propagation.

        Returns:
            Tuple of (SolutionContext, full_plan, stats).
        """
        values = asdict(self.config) if self.config else {}

        params = PSOConfig(
            pop_size=values.get("pop_size", 30),
            inertia_weight_start=values.get("inertia_weight_start", 0.9),
            inertia_weight_end=values.get("inertia_weight_end", 0.4),
            cognitive_coef=values.get("cognitive_coef", 2.0),
            social_coef=values.get("social_coef", 2.0),
            position_min=values.get("position_min", -1.0),
            position_max=values.get("position_max", 1.0),
            velocity_max=values.get("velocity_max", 0.5),
            max_iterations=values.get("max_iterations", 500),
            time_limit=values.get("time_limit", 60.0),
            profit_aware_operators=values.get("profit_aware_operators", False),
            seed=values.get("seed", 42),
        )

        solver = PSOSolver(
            dist_matrix=problem.distance_matrix,
            wastes=problem.wastes,
            capacity=problem.capacity,
            R=problem.revenue_per_kg,
            C=problem.cost_per_km,
            params=params,
            mandatory_nodes=problem.mandatory,
        )

        routes, profit, cost = solver.solve()

        # Wrap plan
        full_plan: List[List[List[int]]] = [[] for _ in range(self.horizon + 1)]
        full_plan[0] = routes

        # Extract primary route
        today_route = routes[0] if routes else []
        sol_ctx = SolutionContext.from_problem(problem, today_route)

        return sol_ctx, full_plan, {"cost": cost, "profit": profit}

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
        """Legacy fallback.

        Args:
            sub_dist_matrix: Dist matrix.
            sub_wastes: Waste dict.
            capacity: Capacity.
            revenue: Revenue.
            cost_unit: Cost.
            values: Config dict.
            mandatory_nodes: Mandatory nodes.
            kwargs: Extra args.

        Returns:
            Empty results.
        """
        return [], 0.0, 0.0

    def get_name(self) -> str:
        """Return policy name.

        Returns:
            Full name string.
        """
        return "Particle Swarm Optimization (PSO)"

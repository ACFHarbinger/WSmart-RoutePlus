r"""ALNS with Inter-Period Operators (ALNS-IPO) Policy Adapter.

Adapts the ALNS-IPO solver (``ALNSSolverIPO``) to the ``BaseMultiPeriodRoutingPolicy``
interface, enabling it to be invoked via ``test_sim`` with the key ``"alns_ipo"``.

Attributes:
    ALNSInterPeriodOperatorsPolicy: Policy class for ALNS-IPO.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators import ALNSInterPeriodOperatorsPolicy
    >>> policy = ALNSInterPeriodOperatorsPolicy()
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from logic.src.configs.policies.routing.meta_heuristics.alns_ipo import ALNSIPOConfig
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.routing.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .alns_ipo import ALNSSolverIPO


@RoutingAlgorithmRegistry.register("alns_ipo")
class ALNSInterPeriodOperatorsPolicy(BaseMultiPeriodRoutingPolicy):
    r"""ALNS policy with Inter-Period Operators (IPO).

    Maintains a full T-day horizon chromosome and uses ShiftVisitRemoval,
    PatternRemoval, and ForwardLookingInsertion operators to reshape the
    visit schedule across the entire planning horizon.

    Registry key: ``"alns_ipo"``

    Attributes:
        config: Configuration for the policy.

    Example:
        >>> policy = ALNSInterPeriodOperatorsPolicy()

    References:
        Coelho, L. C., Cordeau, J.-F., & Laporte, G. (2012).
        "The inventory-routing problem with transshipment."
        Computers & Operations Research, 39(11), 2537–2548.
    """

    def __init__(
        self,
        config: Optional[Union[ALNSIPOConfig, Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the ALNS-IPO policy adapter.

        Args:
            config: Configuration for the policy.
        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Type[ALNSIPOConfig]:
        """Return the configuration class for ALNS-IPO.

        Returns:
            The ALNSIPOConfig class.
        """
        return ALNSIPOConfig

    def _get_config_key(self) -> str:
        """Return the config key for ALNS-IPO.

        Returns:
            The string "alns_ipo".
        """
        return "alns_ipo"

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the multi-period ALNS over the T-day horizon.

        Args:
            problem: Current state including inventory and distance matrix.
            multi_day_ctx: Context for multi-day optimization.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Policy decisions for Day 0.
                - best_horizon: Collection plan spanning the entire horizon.
                - stats: Execution statistics and ALNS performance metadata.
        """
        tree = problem.scenario_tree

        # Initialize type-safe Params
        params = ALNSIPOConfig.from_config(asdict(self.config))  # type: ignore[arg-type]

        if params.acceptance_criterion is None:
            from logic.src.policies.acceptance_criteria.base.factory import (
                AcceptanceCriterionFactory,
            )

            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name="bmc",
                initial_temp=params.start_temp,
                alpha=params.cooling_rate,
                seed=params.seed,
            )

        solver = ALNSSolverIPO(
            dist_matrix=problem.distance_matrix,
            wastes=problem.wastes,
            capacity=problem.capacity,
            R=problem.revenue_per_kg,
            C=problem.cost_per_km,
            params=params,
            mandatory_nodes=problem.mandatory,
        )

        best_horizon, best_profit, best_cost = solver.solve_horizon(scenario_tree=tree)

        today_route = best_horizon[0][0] if best_horizon and best_horizon[0] else []
        sol = SolutionContext.from_problem(problem, today_route)

        return (
            sol,
            best_horizon,
            {
                "horizon_cost": best_cost,
                "horizon_profit": best_profit,
                "n_days": problem.horizon,
            },
        )

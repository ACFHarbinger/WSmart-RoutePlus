"""
ADP Rollout Policy Adapter.

Adapts the ``ADPRolloutEngine`` to the ``BaseMultiPeriodRoutingPolicy``
interface, enabling simulation-level evaluation via ``test_sim`` with the
registry key ``"adp_rollout"``.

Attributes:
    ADPRolloutPolicy: Approximate Dynamic Programming with Rollout Policy.

Example:
    >>> from logic.src.policies.routing.learning_algorithms import ADPRolloutPolicy
    >>> policy = ADPRolloutPolicy()
    >>> routes, metrics = policy.run_day(env)
    >>> print(f"Best routes: {routes}")
    >>> print(f"Metrics: {metrics}")
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from logic.src.configs.policies.adp import ADPRolloutConfig
from logic.src.enums import PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.routing.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.utils.registry import ALL

from .adp_engine import ADPRolloutEngine


@ALL.register(
    PolicyTag.REINFORCEMENT_LEARNING,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.ROUTING,
)
@RoutingAlgorithmRegistry.register("adp_rollout")
class ADPRolloutPolicy(BaseMultiPeriodRoutingPolicy):
    r"""Approximate Dynamic Programming (ADP) Rollout policy for stochastic IRP.

    Implements Powell's ADP framework with a configureable candidate-set
    strategy and truncated greedy rollout for estimating :math:`V(S_{t+1})`.

    Three candidate strategies are available (set via ``candidate_strategy``):

    * ``"threshold"`` — include all nodes above ``fill_threshold``.
    * ``"top_k"``     — include the top-K nodes by fill level.
    * ``"beam"``      — explore up to ``max_candidate_sets`` random subsets.

    Registry key: ``"adp_rollout"``

    References:
        Powell, W. B. (2011). *Approximate Dynamic Programming:
        Solving the Curses of Dimensionality.* Wiley-Interscience.

    Attributes:
        config: Configuration for the ADP Rollout policy.
    """

    def __init__(
        self,
        config: Optional[Union[ADPRolloutConfig, Dict[str, Any]]] = None,
    ) -> None:
        """Initialise the ADP Rollout policy adapter.

        Args:
            config: Configuration for the ADP Rollout policy.

        """
        super().__init__(config)

    @classmethod
    def _config_class(cls) -> Type[ADPRolloutConfig]:
        """Return the configuration class for this policy.

        Returns:
            Type[ADPRolloutConfig]: The configuration class for this policy.
        """
        return ADPRolloutConfig

    def _get_config_key(self) -> str:
        """Return the configuration key for this policy.

        Returns:
            str: The configuration key for this policy.
        """
        return "adp_rollout"

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the ADP rollout over the T-day horizon using Powell's framework.

        Args:
            problem: State data including current bin levels and scenario projections.
            multi_day_ctx: Context for multi-day optimization state.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Policy decisions for Day 0.
                - full_plan: Collection plan spanning the entire horizon.
                - stats: Metadata about candidate sets, rollout scores, and convergence.
        """
        tree = problem.scenario_tree
        if tree is None:
            raise ValueError("ADP requires a ScenarioTree in ProblemContext.")

        params = ADPRolloutConfig.from_config(asdict(self.config))  # type: ignore[arg-type]

        engine = ADPRolloutEngine(
            dist_matrix=problem.distance_matrix,
            wastes=problem.wastes,
            capacity=problem.capacity,
            R=problem.revenue_per_kg,
            C=problem.cost_per_km,
            params=params,
            mandatory_nodes=problem.mandatory,
        )

        full_plan, total_profit, metadata = engine.solve(
            scenario_tree=tree,
            horizon=problem.horizon,
        )

        today_route = full_plan[0][0] if full_plan and full_plan[0] else []
        sol = SolutionContext.from_problem(problem, today_route)

        return sol, full_plan, metadata

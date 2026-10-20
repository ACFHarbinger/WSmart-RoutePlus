"""
Matheuristic Hyper-Heuristic (MHH) policy implementation.

Reference:
    Hussain, A., Huda, M. S., Rownaghi, F. F., & Naji, S. (2025).
    Meta-Heuristic Performance Evaluation on Inventory Routing Problem With
    Profit Maximization. IEEE Access, 13, 99458-99473.
    DOI: 10.1109/ACCESS.2025.3465866

Attributes:
    MHHPolicy: Adapter for the MHH solver.

Example:
    >>> from logic.src.policies.routing.hyper_heuristics import MHHPolicy
    >>> policy = MHHPolicy()
"""

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.configs.policies.mhh import MHHConfig
from logic.src.enums import PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.routing.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.utils.policy.llh_pool import LLHPool
from logic.src.utils.policy.routes import route_profit
from logic.src.utils.policy.wrappers import greedy_day_route
from logic.src.utils.registry import ALL


@ALL.register(
    PolicyTag.HYPER_HEURISTIC,
    PolicyTag.MATHEURISTIC,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("mhh")
class MHHPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Matheuristic Hyper-Heuristic (MHH).

    MHH is a hybrid search strategy that combines the flexibility of
    hyper-heuristics with the mathematical rigor of exact solvers. It employs
    a sequence of Low-Level Heuristics (LLHs) for diversification (exploring
    different regions of the feasibility space) and periodically invokes an
    exact matheuristic (such as Relax-and-Fix) for intensification (optimizing
    the current region).

    Search Logic:
    1. **Diversification Phase**: Uses stochastic selection of LLHs to perturb
       the current solution and escape local optima.
    2. **Intensification Phase**: Leverages the ``RelaxFixOptimizePolicy`` to
       optimally solve sub-problems or the entire rolling horizon plan,
       ensuring high-quality local convergence.
    3. **Hybrid Control**: Manages the balance between heuristic exploration and
       exact exploitation based on the available computational budget.

    Registry key: ``"mhh"``

    Attributes:
        config: Configuration parameters.
        llhs: Pool of Low-Level Heuristics.
        params: Parameters for the MHH solver.
        iters: Number of iterations.
        seed: Random seed.
        rng: Random number generator.
    """

    @classmethod
    def _config_class(cls) -> Any:
        """Return the typed config of this policy."""
        return MHHConfig

    def __init__(self, config: Any = None):
        """
        Initialize MHHPolicy.

        Args:
            config: Configuration parameters.
        """
        super().__init__(config)
        self.params = self._effective_config(config or {})
        self.iters = self.params.iters
        self.seed = self.params.seed
        self.rng = random.Random(self.seed)
        self.llhs = LLHPool.get_all()

    def _evaluate(self, plan: List[List[List[int]]], problem: ProblemContext) -> float:
        """
        Evaluate the total profit of a plan.

        Args:
            plan: Collection plan.
            problem: Problem context.

        Returns:
            float: Total profit.
        """
        tot = 0.0
        cur_prob = problem
        for d in range(problem.horizon):
            rt = plan[d][0] if plan[d] else []
            tot += route_profit(rt, cur_prob)
            cur_prob = cur_prob.advance(rt)
        return tot

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the Matheuristic Hyper-Heuristic solver.

        Args:
            problem: The current ProblemContext containing state data.
            multi_day_ctx: Optional context for spanning multiple rolling days.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                - today_solution: Standardized solution context for Day 0.
                - best_plan: Combined optimized plan.
                - stats: Execution statistics (iterations, RFO usage).
        """
        np_rng = np.random.default_rng(self.seed)

        # Initial plan
        cur_plan = []
        cur_prob = problem
        for _d in range(problem.horizon):
            rt = greedy_day_route(cur_prob, np_rng)
            cur_plan.append([rt])
            cur_prob = cur_prob.advance(rt)

        best_plan = copy.deepcopy(cur_plan)
        best_prof = self._evaluate(best_plan, problem)

        try:
            from logic.src.policies.routing.matheuristics.relax_and_fix_and_optimize.policy_rfo import (
                RelaxFixOptimizePolicy,
            )

            rfo = RelaxFixOptimizePolicy(self.config)
        except Exception:
            rfo = None

        for _ in range(self.iters):
            # 1. Diversify with random LLH
            idx = self.rng.randint(0, len(self.llhs) - 1)
            p_div = self.llhs[idx](cur_plan, problem, self.rng)

            # 2. Intensify using exact solver
            if rfo:
                rfo_sol, rfo_plan, _ = rfo._run_multi_period_solver(problem, multi_day_ctx)
                p_int = rfo_plan
            else:
                p_int = copy.deepcopy(p_div)

            prof_int = self._evaluate(p_int, problem)
            if prof_int > best_prof:
                best_prof = prof_int
                best_plan = copy.deepcopy(p_int)

            # Random restart or keep going from int
            if self.rng.random() < 0.2:
                cur_plan = copy.deepcopy(p_int)

        today_route = best_plan[0][0] if best_plan[0] else []
        sol = SolutionContext.from_problem(problem, today_route)

        return sol, best_plan, {"mhh_iters": self.iters, "expected_profit": best_prof}

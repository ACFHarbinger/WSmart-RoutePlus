"""
Multi-Period Ant Colony Optimization (MP-ACO) policy.

Attributes:
    MultiPeriodACOPolicy: Policy class for the MP-ACO approach.

Example:
    >>> policy = MultiPeriodACOPolicy(config)
    >>> sol, plan, meta = policy._run_multi_period_solver(problem, multi_day_ctx)
"""

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.routing.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry
from logic.src.utils.policy.routes import route_cost, route_profit


@GlobalRegistry.register(
    PolicyTag.META_HEURISTIC,
    PolicyTag.SWARM_INTELLIGENCE,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.ROUTING,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("mp_aco")
class MultiPeriodACOPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Multi-Period Ant Colony Optimization (ACO).

    Pheromone matrix guides construction in each day based on multi-day reward.

    Attributes:
        params: MPACO specific parameters.
        n_ants: Number of ants per iteration.
        iters: Number of iterations.
        alpha: Influence of pheromone.
        beta: Influence of heuristic visibility.
        rho: Pheromone evaporation rate.
        seed: Random seed.
        rng: Random number generator.
    """

    def __init__(self, config: Any = None):
        """Initializes the Multi-Period ACO policy.

        Args:
            config: Optional configuration dictionary or Hydra config.

        """
        super().__init__(config)
        self.params = self._effective_config(config)
        self.n_ants = self.params.n_ants
        self.iters = self.params.iters
        self.alpha = self.params.alpha
        self.beta = self.params.beta
        self.rho = self.params.rho
        self.seed = self.params.seed
        self.rng = random.Random(self.seed)

    def _evaluate(self, plan: List[List[List[int]]], problem: ProblemContext) -> float:
        """Evaluate the multi-period plan and return total profit.

        Args:
            plan: The multi-period routing plan.
            problem: The problem context.

        Returns:
            float: Total net profit across all periods.
        """
        tot = 0.0
        cur_prob = problem
        for d in range(problem.horizon):
            rt = plan[d][0] if plan[d] else []
            tot += route_profit(rt, cur_prob)
            cur_prob = cur_prob.advance(rt)
        return tot

    def _build_ant_solution(self, problem: ProblemContext, pheromones: np.ndarray) -> List[List[List[int]]]:
        """Build a single ant's multi-period routing plan.

        Args:
            problem: The problem context.
            pheromones: The current pheromone matrix.

        Returns:
            List[List[List[int]]]: A generated multi-period plan.
        """
        D = problem.horizon
        N = len(problem.distance_matrix)
        plan = []
        cur_prob = problem

        for d in range(D):
            rt = []
            cand = set(range(1, N))
            curr = 0
            w_sum = 0.0

            # Simple heuristic visibility: profit ratio
            while cand:
                probs = []
                nodes = []
                for nxt in cand:
                    w = cur_prob.wastes.get(nxt, 0.0)
                    if w_sum + w <= cur_prob.capacity:
                        dist = max(1e-6, cur_prob.distance_matrix[curr, nxt])
                        # heuristic: w / dist
                        eta = w / dist
                        tau = pheromones[d, curr, nxt]
                        probs.append((tau**self.alpha) * (eta**self.beta))
                        nodes.append(nxt)

                if not nodes:
                    break

                sprobs = sum(probs)
                probs = [1.0 / len(probs)] * len(probs) if sprobs == 0 else [p / sprobs for p in probs]

                nxt = self.rng.choices(nodes, weights=probs)[0]
                rt.append(nxt)
                w_sum += cur_prob.wastes.get(nxt, 0.0)
                cand.remove(nxt)
                curr = nxt

            plan.append([rt])
            cur_prob = cur_prob.advance(rt)

        return plan

    def _update_pheromones(
        self, pheromones: np.ndarray, plans: List, best_plan: List, best_prof: float, problem: ProblemContext
    ):
        """Update pheromones based on the current iteration's plans.

        Args:
            pheromones: The pheromone matrix to update.
            plans: List of (profit, plan) tuples for all ants.
            best_plan: The best plan found so far.
            best_prof: The profit of the best plan.
            problem: The problem context.

        """
        # evaporation
        pheromones *= 1.0 - self.rho

        # deposit (elitist + all ants)
        # simplified deposit proportional to rank or just profit
        max_prof = max((p for p, _ in plans), default=1.0)
        if max_prof <= 0:
            max_prof = 1.0

        for prof, plan in plans:
            dep = max(0.0, prof / max_prof)
            for d in range(problem.horizon):
                rt = plan[d][0] if plan[d] else []
                path = [0] + rt + [0]
                for i in range(len(path) - 1):
                    pheromones[d, path[i], path[i + 1]] += dep * self.rho

        # elitist
        dep = max(0.0, best_prof / max_prof)
        for d in range(problem.horizon):
            rt = best_plan[d][0] if best_plan[d] else []
            path = [0] + rt + [0]
            for i in range(len(path) - 1):
                pheromones[d, path[i], path[i + 1]] += 2.0 * dep * self.rho

    def _run_multi_period_solver(
        self, problem: ProblemContext, multi_day_ctx: Optional[MultiDayContext]
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """Run the multi-period ACO solver.

        Args:
            problem: The initial problem context.
            multi_day_ctx: Optional multi-day context.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
                The solution for today, the full multi-period plan, and metadata.
        """
        D = problem.horizon
        N = len(problem.distance_matrix)

        # pheromones[day, i, j]
        pheromones = np.ones((D, N, N)) * 0.1

        best_plan = None
        best_prof = -float("inf")

        for _ in range(self.iters):
            plans = []
            for _ in range(self.n_ants):
                plan = self._build_ant_solution(problem, pheromones)
                prof = self._evaluate(plan, problem)
                plans.append((prof, plan))

                if prof > best_prof:
                    best_prof = prof
                    best_plan = copy.deepcopy(plan)

            if best_plan is not None:
                self._update_pheromones(pheromones, plans, best_plan, best_prof, problem)

        if not best_plan:
            best_plan = [[[]] for _ in range(D)]

        today_route = best_plan[0][0] if best_plan[0] else []
        sol = SolutionContext.from_single_route(
            route=today_route,
            profit=route_profit(today_route, problem),
            cost=route_cost(today_route, problem),
            metadata={},
        )
        return sol, best_plan, {"aco_iters": self.iters}

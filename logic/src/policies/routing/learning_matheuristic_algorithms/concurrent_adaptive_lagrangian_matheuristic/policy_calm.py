"""
Top-level multi-period Lagrangian matheuristic policy.

Inherits from :class:`BaseMultiPeriodRoutingPolicy`, which handles the
infrastructure for horizon management, scenario-tree ingestion, mandatory
node prediction, and SolutionContext plumbing.  Our override lives in
:meth:`_run_multi_period_solver`.

Algorithmic loop (high-level):

    state = initialise(problem)
    for outer_iter in range(max_outer_iterations):
        # 1. Bandit: pick (engine, cut_strategy) from context.
        arm = bandit.select_arm(context(state))

        # 2. Regret preprocessing: produce a RegretPlan.
        plan = regret_preprocessor.build_plan(tables.early_regret)

        # 3. Knapsack side: per-period selection solves, using
        #    Lagrangian-corrected coefficients + plan + arm.cut_strategy.
        selection_results = [solve_selection_period(...) for t in horizon]
        coordinator.set_knapsack_selection(x_K_from(selection_results))

        # 4. Routing side: per-period RS evaluations (concurrent).  Each
        #    worker updates oracle + coordinator in place.
        routing_results = concurrent_eval(selection_results)

        # 5. Outer-iteration commit: for the bundle tracker this triggers
        #    the QP-based multiplier update and serious/null-step decision.
        coordinator.commit_outer_iteration(full_lagrangian_value=...)

        # 6. Primal assembly: take best-so-far tours per period -> plan.
        primal = assemble_primal()

        # 7. Bandit update: reward = (primal delta) / (wallclock).
        bandit.update(arm, context, reward)

        # 8. Regret escalation observation.
        regret_preprocessor.observe_iteration(primal_improved=...)

        # Stopping: stagnation, dual-primal gap, or time limit.
        if stopping_criteria(): break

    return best_primal

Attributes:
    _RunState: Mutable state threaded through the outer loop.
    CALMPolicy: The CALM policy.

Example:
    >>> policy = CALMPolicy()
    >>> policy._run_multi_period_solver(problem_ctx, solution_ctx)
    <objective_value>

Outer-iteration state and the period loop live in ``calm_period``. Selection, routing, cuts and primal assembly live in ``calm_layers``. Both mixins are applied to ``CALMPolicy``, and ``_RunState`` is re-exported from ``calm_period``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np

from logic.src.enums import PolicyTag
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.routing.base.base_multi_period_policy import (
    BaseMultiPeriodRoutingPolicy,
)
from logic.src.policies.routing.base.factory import (
    RoutingAlgorithmRegistry,
)
from logic.src.utils.registry import ALL

from .bandit import build_context
from .calm_layers import CALMLayerMixin as CALMLayerMixin
from .calm_period import (
    CALMPeriodMixin as CALMPeriodMixin,
)
from .calm_period import (
    _RunState as _RunState,
)
from .params import CALMParams
from .routing import RoutingResult
from .selection import (
    SelectionResult,
)

# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@ALL.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.PROFIT_AWARE,
    PolicyTag.MULTI_PERIOD,
)
@RoutingAlgorithmRegistry.register("calm")
class CALMPolicy(CALMPeriodMixin, CALMLayerMixin, BaseMultiPeriodRoutingPolicy):
    """Multi-period Concurrent Adaptive Lagrangian Matheuristic (CALM) policy.

    The CALM (Concurrent Adaptive Lagrangian Matheuristic) policy is designed for multi-period vehicle routing problems where demand is uncertain
    and revealed over time. It addresses the challenge of coordinating routing decisions across multiple time periods under uncertainty by combining
    probabilistic lookahead with adaptive Lagrangian relaxation.

    Attributes:
        params (CALMParams): The parameters for the CALM policy.
        horizon (int): The horizon for the multi-period problem.
        stockout_penalty (float): The penalty for stockouts.
    """

    def __init__(self, config: Any = None):
        """
        Initialize the CALM policy.

        Args:
            config: The configuration for the policy.
        """
        super().__init__(config)
        self.params = CALMParams.from_config(self.config) if config else CALMParams()
        # Align BaseMultiPeriodRoutingPolicy defaults.
        self.horizon = self.params.lookahead.horizon
        self.stockout_penalty = self.params.stockout_penalty

    # ------------------------------------------------------------------
    # BaseMultiPeriodRoutingPolicy required overrides
    # ------------------------------------------------------------------

    @classmethod
    def _config_class(cls):
        """
        Get the configuration class.

        Returns:
            Type[CALMParams]: The configuration class.
        """
        return None

    def _get_config_key(self) -> str:
        """
        Get the configuration key.

        Returns:
            str: The configuration key.
        """
        return "lagrangian_matheuristic"

    def _stockout_penalty_estimate(self, state: _RunState, problem: ProblemContext) -> float:
        """
        Approximate stockout penalty: for each bin that is projected to
        exceed capacity_cap in expectation over the horizon and is NOT in any
        period's best-known selection, add the overflow amount to the penalty.

        Args:
            state: The internal state of the policy.
            problem: The problem context.

        Returns:
            float: The stockout penalty.
        """
        visited_any: set[int] = set()
        for sel in state.best_per_period_selection.values():
            visited_any.update(sel)

        cap = self.params.lookahead.capacity_cap
        penalty = 0.0
        expected_fill_final = state.tables.expected_fill[:, -1]
        for i in range(problem.n):
            bin_id = i + 1  # 1-based
            if bin_id in visited_any:
                continue
            overflow = max(0.0, expected_fill_final[i] - cap)
            penalty += overflow * self.params.stockout_penalty / max(1.0, cap)
        return penalty

    def _should_stop(self, state: _RunState) -> bool:
        """
        Determine whether to stop the algorithm.

        Args:
            state: The internal state of the policy.

        Returns:
            bool: True if the algorithm should stop, False otherwise.
        """
        if state.outer_iter >= self.params.lagrangian.max_outer_iterations:
            return True
        elapsed = time.perf_counter() - state.start_time
        if elapsed >= self.params.time_limit:
            return True
        if state.iters_since_improvement >= self.params.lagrangian.stagnation_patience:
            # Dual-primal gap check before giving up.
            dual = state.coordinator.current_dual_bound()
            primal = state.best_primal
            if np.isfinite(dual) and np.isfinite(primal):
                denom = max(1.0, abs(primal))
                gap = abs(primal - dual) / denom
                if gap < self.params.lagrangian.dual_bound_tolerance:
                    return True
            # Otherwise keep going if we have budget, but cap stagnation.
            if state.iters_since_improvement >= 2 * self.params.lagrangian.stagnation_patience:
                return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assemble_x_K(self, selection_results: List[SelectionResult], n_bins: int, horizon: int) -> np.ndarray:
        """
        Assemble the full selection matrix x_K.

        Args:
            selection_results: The selection results.
            n_bins: The number of bins.
            horizon: The horizon.

        Returns:
            np.ndarray: The selection matrix.
        """
        x_K = np.zeros((n_bins, horizon), dtype=float)
        for sr in selection_results:
            for b in sr.selection:
                if 1 <= b <= n_bins:
                    x_K[b - 1, sr.period] = 1.0
        return x_K

    def _compute_full_lagrangian(
        self,
        selection_results: List[SelectionResult],
        routing_results: List[RoutingResult],
    ) -> float:
        """
        Full Lagrangian value at current multipliers:
            L(lambda) = max_x^K  sum (V + lambda) x^K
                     + max_x^R  sum (V - lambda) x^R - routing_cost

        Under the convention used in lookahead.lagrangian_corrected_prizes,
        the per-period contributions are already computed on each side; we
        just sum.

        Args:
            selection_results: The selection results.
            routing_results: The routing results.

        Returns:
            float: The full Lagrangian value.
        """
        knapsack_side = sum(sr.lagrangian_objective for sr in selection_results)
        routing_side = sum(rr.lagrangian_value_contrib for rr in routing_results)
        return float(knapsack_side + routing_side)

    def _build_context(self, state: _RunState, problem: ProblemContext) -> np.ndarray:
        """
        Build the context for the neural network.

        Args:
            state: The internal state of the policy.
            problem: The problem context.

        Returns:
            np.ndarray: The context for the neural network.
        """
        dual = state.coordinator.current_dual_bound()
        primal = state.best_primal
        if np.isfinite(dual) and np.isfinite(primal):
            denom = max(1.0, abs(primal))
            gap = abs(primal - dual) / denom
        else:
            gap = 1.0

        # Dual progress proxy: EMA-tracker bookkeeping would give this more
        # accurately; here we use the ratio of current dual to a generous upper
        # bound.
        dual_progress_frac = 0.0
        if np.isfinite(dual) and np.isfinite(primal) and primal > 0:
            dual_progress_frac = np.clip(dual / primal, 0.0, 1.0)

        # Fraction of bins currently selected somewhere.
        total_selected = 0
        for sel in state.best_per_period_selection.values():
            total_selected += len(sel)
        frac_selected = total_selected / max(1, problem.n * self.horizon)

        # Fraction of periods where current expected fill has saturated somewhere.
        ef = state.tables.expected_fill  # (N, T+1)
        cap = self.params.lookahead.capacity_cap
        frac_saturated = float(np.mean(np.any(ef[:, 1:] >= cap, axis=0)))

        lambda_norm = float(np.linalg.norm(state.coordinator.lambdas_snapshot()))
        scale = max(
            abs(self.params.lagrangian.lambda_max),
            abs(self.params.lagrangian.lambda_min),
        )

        return build_context(
            outer_iter=state.outer_iter,
            max_outer=self.params.lagrangian.max_outer_iterations,
            primal_gap_frac=gap,
            dual_progress_frac=dual_progress_frac,
            iters_since_improvement=state.iters_since_improvement,
            stagnation_patience=self.params.lagrangian.stagnation_patience,
            fraction_bins_selected=frac_selected,
            fraction_periods_saturated=frac_saturated,
            lambda_norm=lambda_norm,
            lambda_norm_scale=scale,
            feature_dim=self.params.bandit.feature_dim,
        )

    def _log_iter(
        self,
        state: _RunState,
        arm_idx: int,
        engine: str,
        cut_strategy: str,
        primal: float,
        lagrangian: float,
        coord_stats: Dict[str, float],
    ) -> None:
        """
        Log the current iteration.

        Args:
            state: The internal state of the policy.
            arm_idx: The index of the arm.
            engine: The engine.
            cut_strategy: The cut strategy.
            primal: The primal value.
            lagrangian: The Lagrangian value.
            coord_stats: The coordinator statistics.
        """
        dual = state.coordinator.current_dual_bound()
        msg = (
            f"[lagrangian_matheuristic] iter={state.outer_iter} "
            f"arm={engine}/{cut_strategy} (idx={arm_idx}) "
            f"primal={primal:.4f} dual={dual:.4f} L(lambda)={lagrangian:.4f} "
            f"phase={state.regret_preprocessor.phase} "
            f"gamma_mean={coord_stats.get('gamma_mean', float('nan')):.4f}"
        )
        print(msg)

    # ------------------------------------------------------------------
    # Final solution packaging
    # ------------------------------------------------------------------

    def _package_solution(
        self,
        state: _RunState,
        problem: ProblemContext,
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Produce the triple expected by BaseMultiPeriodRoutingPolicy.execute:
            (today's SolutionContext, full_plan, telemetry)

        Args:
            state: The internal state of the policy.
            problem: The problem context.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]: The solution context, full plan, and telemetry.
        """
        full_plan: List[List[List[int]]] = []
        for t in range(self.horizon):
            tour = state.best_per_period_tours.get(t, [0, 0])
            # BaseMultiPeriodRoutingPolicy expects List[day][vehicle][node];
            # our single-vehicle model wraps the tour in a single-element list.
            full_plan.append([list(tour)])

        # Today's solution (day 0).
        today_tour = state.best_per_period_tours.get(0, [0, 0])
        today_cost = state.best_per_period_cost.get(0, 0.0)
        today_selection = state.best_per_period_selection.get(0, [])
        # Profit for today's route.
        V = state.tables.V
        today_idx = np.array([b - 1 for b in today_selection if 1 <= b <= problem.n], dtype=int)
        today_prize = float(np.sum(V[today_idx, 0])) if today_idx.size else 0.0
        today_profit = today_prize - problem.cost_unit * today_cost

        solution = SolutionContext(
            routes=[list(today_tour)],
            total_cost=today_cost,
            total_profit=today_profit,
        )

        telemetry: Dict[str, Any] = {
            "outer_iters": state.outer_iter,
            "best_primal": state.best_primal,
            "best_dual": state.coordinator.current_dual_bound(),
            "bandit_summary": state.bandit.summary(),
            "regret_phase": state.regret_preprocessor.phase,
            "wallclock_seconds": time.perf_counter() - state.start_time,
        }
        return solution, full_plan, telemetry

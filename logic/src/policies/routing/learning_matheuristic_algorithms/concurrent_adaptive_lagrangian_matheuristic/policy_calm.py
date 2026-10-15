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
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.pipeline.simulations.bins.prediction import ScenarioTree
from logic.src.policies.routing.base.base_multi_period_policy import (
    BaseMultiPeriodRoutingPolicy,
)
from logic.src.policies.routing.base.factory import (
    RoutingAlgorithmRegistry,
)

from .bandit import LinUCBBandit, build_context
from .dual import build_dual_bound_tracker
from .lagrangian import LagrangianCoordinator, LagrangianState
from .lookahead import LookaheadTables, LookaheadValuator
from .oracle import InsertionCostOracle
from .params import CALMParams
from .regret import RegretPlan, RegretPreprocessor
from .routing import RoutingResult, evaluate_period
from .selection import (
    SelectionResult,
    build_corrected_revenue,
    generate_lifted_cuts,
    generate_pareto_cuts,
    solve_selection_period,
)

# ---------------------------------------------------------------------------
# Internal state holder
# ---------------------------------------------------------------------------


@dataclass
class _RunState:
    """Mutable state threaded through the outer loop.

    Attributes:
        tables (LookaheadTables): The lookahead tables.
        scenario_tree (ScenarioTree): The scenario tree.
        lag_state (LagrangianState): The Lagrangian state.
        coordinator (LagrangianCoordinator): The Lagrangian coordinator.
        oracle (InsertionCostOracle): The insertion cost oracle.
        bandit (LinUCBBandit): The bandit.
        regret_preprocessor (RegretPreprocessor): The regret preprocessor.
        best_primal (float): The best primal.
        best_per_period_tours (Dict[int, List[int]]): The best per-period tours.
        best_per_period_selection (Dict[int, List[int]]): The best per-period selection.
        best_per_period_cost (Dict[int, float]): The best per-period cost.
        outer_iter (int): The outer iteration.
        iters_since_improvement (int): The number of iterations since improvement.
        prior_cuts (Dict[int, Dict[int, float]]): The prior cuts.
        selection_history (List[List[SelectionResult]]): The selection history.
        start_time (float): The start time.
    """

    tables: LookaheadTables
    scenario_tree: ScenarioTree
    lag_state: LagrangianState
    coordinator: LagrangianCoordinator
    oracle: InsertionCostOracle
    bandit: LinUCBBandit
    regret_preprocessor: RegretPreprocessor

    # Primal accumulator.
    best_primal: float = -np.inf
    best_per_period_tours: Dict[int, List[int]] = field(default_factory=dict)
    best_per_period_selection: Dict[int, List[int]] = field(default_factory=dict)
    best_per_period_cost: Dict[int, float] = field(default_factory=dict)

    # Iteration bookkeeping.
    outer_iter: int = 0
    iters_since_improvement: int = 0
    prior_cuts: Dict[int, Dict[int, float]] = field(default_factory=dict)  # period -> {bin: penalty}
    selection_history: List[List[SelectionResult]] = field(default_factory=list)

    # Telemetry.
    start_time: float = 0.0


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.MATH_PROGRAMMING,
    PolicyTag.PROFIT_AWARE,
    PolicyTag.MULTI_PERIOD,
)
@RoutingAlgorithmRegistry.register("calm")
class CALMPolicy(BaseMultiPeriodRoutingPolicy):
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

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Run the multi-period solver.

        Args:
            problem: The problem context.
            multi_day_ctx: The multi-day context.

        Returns:
            Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]: The solution context, routing results, and metadata.
        """
        # --- 1. Initialisation ---
        state = self._initialise(problem)

        # --- 2. Outer loop ---
        while True:
            if self._should_stop(state):
                break

            arm_idx, engine, cut_strategy, _ = state.bandit.select_arm(self._build_context(state, problem))

            plan = state.regret_preprocessor.build_plan(early_regret=state.tables.early_regret)

            iter_start = time.perf_counter()
            old_primal = state.best_primal
            old_dual = state.coordinator.current_dual_bound()

            selection_results = self._solve_selection_layer(
                state=state,
                problem=problem,
                engine=engine,
                plan=plan,
            )

            # Publish x_K to the coordinator.
            x_K = self._assemble_x_K(selection_results, n_bins=problem.n, horizon=self.horizon)
            state.coordinator.set_knapsack_selection(x_K)

            routing_results = self._solve_routing_layer(
                state=state,
                problem=problem,
                selection_results=selection_results,
            )

            # Generate cuts for the NEXT iteration based on arm's cut_strategy.
            state.prior_cuts = self._generate_cuts(
                strategy=cut_strategy,
                selection_results=selection_results,
                state=state,
                problem=problem,
            )
            state.selection_history.append(selection_results)
            if len(state.selection_history) > 5:
                state.selection_history.pop(0)

            # Commit outer iteration to the coordinator.
            full_lagr = self._compute_full_lagrangian(selection_results, routing_results)
            coord_stats = state.coordinator.commit_outer_iteration(full_lagr)

            # Assemble primal from best-known per-period tours.
            primal, improved = self._assemble_primal(state, problem)
            state.best_primal = max(state.best_primal, primal)

            # --- BANDIT REWARD: Relative Gap Reduction ---
            new_primal = state.best_primal
            new_dual = state.coordinator.current_dual_bound()

            # Fetch the old dual (we need to track this at the start of the loop)
            epsilon = 1e-6
            reward = 0.0

            # Check if both bounds are finite and valid
            if np.isfinite(old_dual) and np.isfinite(old_primal) and np.isfinite(new_dual):
                old_gap = abs(old_dual - old_primal)
                new_gap = abs(new_dual - new_primal)

                if old_gap > epsilon:
                    # How much of the gap did we close? (Usually between 0.0 and 1.0)
                    reward = (old_gap - new_gap) / old_gap
            else:
                # Fallback: Normalized Primal Improvement (if dual is still infinite)
                if np.isfinite(old_primal) and np.isfinite(new_primal):
                    primal_delta = max(0.0, new_primal - old_primal)
                    reward = primal_delta / (abs(old_primal) + epsilon)
                else:
                    # First iteration bootstrap: small constant reward if we found *any* valid primal
                    reward = 0.1 if np.isfinite(new_primal) else 0.0

            # Add a tiny penalty for wallclock time to break ties
            # between two arms that achieved the exact same gap reduction.
            elapsed = max(1e-6, time.perf_counter() - iter_start)
            reward -= 0.001 * elapsed
            state.bandit.update(
                arm_index=arm_idx,
                context=self._build_context(state, problem),
                reward=reward,
            )

            # Regret escalation observation.
            state.regret_preprocessor.observe_iteration(primal_improved=improved)
            state.iters_since_improvement = 0 if improved else state.iters_since_improvement + 1
            state.outer_iter += 1

            if self.params.verbose:
                self._log_iter(state, arm_idx, engine, cut_strategy, primal, full_lagr, coord_stats)

            # Reset x_R between iterations to avoid stale subgradients.
            state.coordinator.reset_outer_iteration()

        # --- 3. Package result ---
        return self._package_solution(state, problem)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise(self, problem: ProblemContext) -> _RunState:
        """
        Initialize the internal state of the policy.

        Args:
            problem: The problem context.

        Returns:
            _RunState: The internal state of the policy.
        """
        lookahead_params = self.params.lookahead
        # Populate revenue-conversion parameters from the problem if not set.
        if lookahead_params.volume is None:
            lookahead_params.volume = getattr(problem, "volume", 1.0)
        if lookahead_params.density is None:
            lookahead_params.density = getattr(problem, "density", 1.0)
        if lookahead_params.revenue_per_kg is None:
            lookahead_params.revenue_per_kg = getattr(problem, "revenue", 1.0)

        # Build the lookahead tables (V, rho, early_regret, expected_fill).
        valuator = LookaheadValuator(lookahead_params)
        current_wastes = problem.wastes
        bin_stats = {
            "means": getattr(problem, "fill_means", None),
            "stds": getattr(problem, "fill_stds", None),
        }
        if bin_stats["means"] is None:
            # Fall back to mean = 0 (pure static-fill scenarios).
            bin_stats = None
        tables = valuator.compute(
            current_wastes=current_wastes,
            bin_stats=bin_stats,
            truth_generator=getattr(problem, "truth_generator", None),
        )
        scenario_tree = valuator.get_scenario_tree()

        n_bins = problem.n
        horizon = self.horizon

        # Shared infrastructure.
        lag_state = LagrangianState(n_bins=n_bins, horizon=horizon)
        # Seed gamma with params.gamma_init (per-period).
        lag_state.gamma[:] = self.params.lagrangian.gamma_init

        tracker = build_dual_bound_tracker(
            dual_params=self.params.dual_bound,
            lag_params=self.params.lagrangian,
            n_bins=n_bins,
            horizon=horizon,
        )
        coordinator = LagrangianCoordinator(state=lag_state, tracker=tracker, lag_params=self.params.lagrangian)
        oracle = InsertionCostOracle(
            n_bins=n_bins,
            horizon=horizon,
            alpha=self.params.dual_bound.ema_alpha,
            quality_threshold=self.params.dual_bound.ema_quality_threshold,
        )

        rng = np.random.default_rng(self.params.seed)
        bandit = LinUCBBandit(params=self.params.bandit, rng=rng)

        regret_preprocessor = RegretPreprocessor(params=self.params.regret, n_bins=n_bins, horizon=horizon)

        return _RunState(
            tables=tables,
            scenario_tree=scenario_tree,
            lag_state=lag_state,
            coordinator=coordinator,
            oracle=oracle,
            bandit=bandit,
            regret_preprocessor=regret_preprocessor,
            start_time=time.perf_counter(),
        )

    # ------------------------------------------------------------------
    # Selection layer
    # ------------------------------------------------------------------

    def _solve_selection_layer(
        self,
        *,
        state: _RunState,
        problem: ProblemContext,
        engine: str,
        plan: RegretPlan,
    ) -> List[SelectionResult]:
        """
        Solve the selection layer.

        Args:
            state: The internal state of the policy.
            problem: The problem context.
            engine: The engine.
            plan: The regret plan.

        Returns:
            List[SelectionResult]: The selection results.
        """
        tpks_params = self.params.tpks
        insertion_snapshot = state.oracle.snapshot()
        lambdas_snapshot = state.coordinator.lambdas_snapshot()
        gamma_snapshot = state.coordinator.gamma_snapshot()
        V = state.tables.V

        # Per-period mandatory nodes (from BaseMultiPeriodRoutingPolicy helper).
        initial_mandatory = list(getattr(problem, "mandatory_nodes", []) or [])
        mandatory_map = self._predict_mandatory_nodes_for_horizon(
            tree=state.scenario_tree,
            initial_mandatory=initial_mandatory,
        )

        results: List[SelectionResult] = []
        for t in range(self.horizon):
            revenue_eff = build_corrected_revenue(
                V=V,
                lambdas=lambdas_snapshot,
                insertion_costs=insertion_snapshot,
                gamma=gamma_snapshot,
                regret_bias=plan.soft_bias,
                period=t,
            )

            # Hard-fixed bins for this period from the regret plan.
            hard_fix_bins_this_period: List[int] = []
            for bin_id, periods in plan.hard_fix.items():
                if t in periods:
                    hard_fix_bins_this_period.append(bin_id + 1)  # 1-based for TPKS

            result = solve_selection_period(
                period=t,
                dist_matrix=problem.dist_matrix,
                revenue_eff=revenue_eff,
                capacity=problem.capacity,
                routing_cost_unit=problem.cost_unit,
                mandatory_nodes=mandatory_map.get(t, []),
                tpks_params=tpks_params,
                hard_fix_bins=hard_fix_bins_this_period,
                engine=engine,
                prior_cuts=state.prior_cuts.get(t),
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Routing layer (concurrent)
    # ------------------------------------------------------------------

    def _solve_routing_layer(
        self,
        *,
        state: _RunState,
        problem: ProblemContext,
        selection_results: List[SelectionResult],
    ) -> List[RoutingResult]:
        """
        Solve the routing layer.

        Args:
            state: The internal state of the policy.
            problem: The problem context.
            selection_results: The selection results.

        Returns:
            List[RoutingResult]: The routing results.
        """
        lambdas_snapshot = state.coordinator.lambdas_snapshot()
        V = state.tables.V

        upper_bound = state.best_primal if np.isfinite(state.best_primal) else 0.0

        def _worker(sr: SelectionResult) -> RoutingResult:
            return evaluate_period(
                selection_result=sr,
                dist_matrix=problem.dist_matrix,
                n_bins=problem.n,
                V_column=V[:, sr.period],
                lambdas_column=lambdas_snapshot[:, sr.period],
                oracle=state.oracle,
                coordinator=state.coordinator,
                upper_bound=upper_bound,
                lkh3_improver=getattr(problem, "lkh3_improver", None),
            )

        results: List[RoutingResult] = []
        # Use a thread pool when async updates are enabled and the horizon
        # warrants it.  Gurobi releases the GIL during LP/MIP solves.
        if self.params.lagrangian.asynchronous_updates and self.horizon >= 2:
            max_workers = min(self.horizon, 8)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_worker, sr): sr for sr in selection_results}
                for fut in as_completed(futures):
                    results.append(fut.result())
            # Re-order by period for downstream consumers.
            results.sort(key=lambda r: r.period)
        else:
            for sr in selection_results:
                results.append(_worker(sr))
        return results

    # ------------------------------------------------------------------
    # Cut generation
    # ------------------------------------------------------------------

    def _generate_cuts(
        self,
        *,
        strategy: str,
        selection_results: List[SelectionResult],
        state: _RunState,
        problem: ProblemContext,
    ) -> Dict[int, Dict[int, float]]:
        """Return per-period {bin: penalty} dicts.

        Args:
            strategy: The cut generation strategy.
            selection_results: The selection results.
            state: The internal state of the policy.
            problem: The problem context.

        Returns:
            Dict[int, Dict[int, float]]: The per-period {bin: penalty} dicts.
        """
        if strategy == "plain":
            return {}
        out: Dict[int, Dict[int, float]] = {}
        V = state.tables.V
        for sr in selection_results:
            revenue_eff_col = V[:, sr.period]
            if strategy == "lifted":
                out[sr.period] = generate_lifted_cuts(
                    result=sr,
                    dist_matrix=problem.dist_matrix,
                    revenue_eff=revenue_eff_col,
                )
            elif strategy == "pareto":
                # Use the history of this period's selections.
                hist = [
                    iter_results[sr.period] for iter_results in state.selection_history if sr.period < len(iter_results)
                ]
                hist.append(sr)
                out[sr.period] = generate_pareto_cuts(
                    results=hist,
                    dist_matrix=problem.dist_matrix,
                    revenue_eff=revenue_eff_col,
                )
        return out

    # ------------------------------------------------------------------
    # Primal assembly and stopping criteria
    # ------------------------------------------------------------------

    def _assemble_primal(self, state: _RunState, problem: ProblemContext) -> Tuple[float, bool]:
        """
        Derive the current best primal from the oracle's incumbent tours.

        Primal objective = sum over periods of
            (sum_{i in S_t} V[i, t]) - C * cost(S_t)
        minus stockout penalty for any bin that overflows before being visited.

        Args:
            state: The internal state of the policy.
            problem: The problem context.

        Returns:
            Tuple[float, bool]: The primal value and whether it's an improvement.
        """
        V = state.tables.V
        cost_unit = problem.cost_unit
        total = 0.0
        for t in range(self.horizon):
            inc = state.oracle.get_incumbent(t)
            if not np.isfinite(inc.cost):
                continue
            sel = [b for b in inc.tour if b != 0]
            sel = sorted(set(sel))
            idx = np.array([b - 1 for b in sel if 1 <= b <= problem.n], dtype=int)
            prize = float(np.sum(V[idx, t])) if idx.size else 0.0
            total += prize - cost_unit * inc.cost

            prev_cost = state.best_per_period_cost.get(t, float("inf"))
            if inc.cost < prev_cost - 1e-9:
                state.best_per_period_tours[t] = list(inc.tour)
                state.best_per_period_selection[t] = sel
                state.best_per_period_cost[t] = inc.cost

        # Approximate overflow penalty: bins with expected_fill > cap that are
        # never visited across the horizon incur a stockout.
        penalty = self._stockout_penalty_estimate(state, problem)
        total -= penalty

        improved = total > state.best_primal + 1e-9
        return total, improved

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

"""Run state and the multi-period outer iteration for CALM, mixed into CALMPolicy as CALMPeriodMixin."""

from __future__ import annotations

import time
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.pipeline.simulations.bins.prediction import ScenarioTree

from .bandit import LinUCBBandit
from .dual import build_dual_bound_tracker
from .lagrangian import (
    LagrangianCoordinator,
    LagrangianState,
)
from .lookahead import (
    LookaheadTables,
    LookaheadValuator,
)
from .oracle import InsertionCostOracle
from .regret import RegretPreprocessor
from .selection import SelectionResult

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


class CALMPeriodMixin:
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

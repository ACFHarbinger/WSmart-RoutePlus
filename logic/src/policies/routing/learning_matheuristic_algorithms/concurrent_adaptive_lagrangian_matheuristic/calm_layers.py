"""Per-period selection, routing, cut generation and primal assembly for CALM, mixed into CALMPolicy as CALMLayerMixin."""

from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from typing import (
    Dict,
    List,
    Tuple,
)

import numpy as np

from logic.src.interfaces.context.problem_context import ProblemContext

from .calm_period import _RunState as _RunState
from .regret import RegretPlan
from .routing import (
    RoutingResult,
    evaluate_period,
)
from .selection import (
    SelectionResult,
    build_corrected_revenue,
    generate_lifted_cuts,
    generate_pareto_cuts,
    solve_selection_period,
)


class CALMLayerMixin:
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

"""HyperHeuristicSearch template hooks of the ACO hyper-heuristic.

:class:`HyperACOSearchHooksMixin` holds the hooks that the shared route-level
hyper-heuristic loop calls on :class:`HyperHeuristicACO` (incumbent seeding, candidate
ants, move selection and application, ranking, post-iteration elitism, stagnation
tracking, learning accumulation and the result), together with the per-ant
``_MoveTelemetry`` record and ``_elitism_sync_count``. Moved unchanged from
``hyper_aco.py``, which re-exports the two module-level names.
"""

import copy
import math
import time
from typing import List, NamedTuple, Tuple

import numpy as np

from logic.src.policies.routing.base.sub_problem import SolveResult

from .hyper_operators import (
    HYPER_OPERATORS,
)


class _MoveTelemetry(NamedTuple):
    """Per-ant telemetry flowing from ``_apply_move`` to the loop's bookkeeping."""

    journey_start: int
    sequence: List[str]
    eta_updates: np.ndarray
    end_op_idx: int
    start_obj: float
    final_obj: float


def _elitism_sync_count(n_ants: int, elitism_ratio: float) -> int:
    """Number of ants synced to the global best on improvement (ceil, per the flowchart)."""
    return max(1, math.ceil(n_ants * elitism_ratio))


class HyperACOSearchHooksMixin:
    """HyperHeuristicSearch hooks of :class:`HyperHeuristicACO` (moved unchanged from ``hyper_aco.py``)."""

    # ------------------------------------------------------------------
    # HyperHeuristicSearch hooks (P-35)
    # ------------------------------------------------------------------

    def _seed_incumbent(self) -> Tuple[List[List[int]], float]:
        # The incumbent is tracked at the real capacity: strategic oscillation may let ants
        # explore over-capacity plans, but only a feasible plan (all mandatory nodes visited,
        # no route over capacity) can become the returned best.
        best_routes = [r for r in copy.deepcopy(self.initial_solution) if r]
        best_objective = self._calculate_routing_cost(best_routes) if self._is_feasible(best_routes) else math.inf
        return best_routes, best_objective

    def _initial_candidates(self) -> List[List[List[int]]]:
        ant_solutions = [copy.deepcopy(self.initial_solution) for _ in range(self.cfg.n_ants)]

        # ----------------------------------------------------------------
        # Bug #2 fix: uniform ant placement on real operator vertices.
        # Paper: "Put m ants uniformly on the n vertices."
        # ----------------------------------------------------------------
        self._ant_positions: List[int] = [i % self.n_operators for i in range(self.cfg.n_ants)]

        # ----------------------------------------------------------------
        # Bug #2 fix (cont.): apply starting operator before main loop.
        # Paper: "For each ant k: apply low-level heuristic from ant k's
        #         current location to Sk."
        # ----------------------------------------------------------------
        effective_capacity = float("inf") if self.pv < self.initial_pv else self.capacity
        for ant_idx in range(self.cfg.n_ants):
            start_op_name = self.operator_names[self._ant_positions[ant_idx]]
            op_func = HYPER_OPERATORS.get(start_op_name)
            if op_func:
                ctx = self._make_context(ant_solutions[ant_idx], effective_capacity)
                op_func(ctx)
                ant_solutions[ant_idx] = ctx.routes
        return ant_solutions

    def _n_candidates(self) -> int:
        return self.cfg.n_ants

    def _select_move(self, candidate: List[List[int]], k: int) -> int:
        # The move is the ant's persistent operator-graph vertex; the actual
        # sequence construction happens inside ``build_solution``.
        return self._ant_positions[k]

    def _apply_move(self, candidate: List[List[int]], move: int, k: int) -> Tuple[List[List[int]], "_MoveTelemetry"]:
        start_obj = self._evaluate_objective(candidate)
        routes, sequence, eta_updates, end_op_idx = self.build_solution(candidate, move)
        final_obj = self._evaluate_objective(routes)
        self._ant_positions[k] = end_op_idx  # persistent across journeys
        telemetry = _MoveTelemetry(
            journey_start=move,
            sequence=sequence,
            eta_updates=eta_updates,
            end_op_idx=end_op_idx,
            start_obj=start_obj,
            final_obj=final_obj,
        )
        return routes, telemetry

    def _record(
        self, candidate: List[List[int]], k: int, telemetry: "_MoveTelemetry"
    ) -> Tuple[List[List[int]], float, float, List[str], int, int]:
        return (candidate, telemetry.start_obj, telemetry.final_obj, telemetry.sequence, k, telemetry.journey_start)

    def _rank(
        self, results: List[Tuple[List[List[int]], float, float, List[str], int, int]]
    ) -> List[Tuple[List[List[int]], float, float, List[str], int, int]]:
        # Minimisation ordering; the sort also ranks ants for elitism.
        return sorted(results, key=lambda x: x[2])

    def _post_iteration(
        self,
        results: List[Tuple[List[List[int]], float, float, List[str], int, int]],
        improved: bool,
        candidates: List[List[List[int]]],
        best_routes: List[List[int]],
    ) -> List[List[List[int]]]:
        if improved:
            # ----------------------------------------------------------------
            # Bug #4 context: elitism_ratio=1.0 (paper default) → all ants
            # sync to global best (exact paper behaviour).
            # elitism_ratio < 1.0 → hedged elitism (VRPP diversity option).
            # Paper: "For all ants: Sk = Sb"
            # ----------------------------------------------------------------
            sync_count = _elitism_sync_count(self.cfg.n_ants, self.cfg.elitism_ratio)
            for rank, (routes, _, _, _, ant_idx, _) in enumerate(results):
                if rank >= self.cfg.n_ants - sync_count:
                    candidates[ant_idx] = copy.deepcopy(best_routes)
                else:
                    candidates[ant_idx] = routes
        else:
            for routes, _, _, _, ant_idx, _ in results:
                candidates[ant_idx] = routes
        return candidates

    def _track_stagnation(self, improved: bool, stagnation: int) -> int:
        # Strategic Oscillation management (per paper §III.B)
        if improved:
            self.pv = self.initial_pv
            return 0
        stagnation += 1
        if stagnation >= self.cfg.stagnation_limit:
            self.pv /= 2.0
        return stagnation

    def _fresh_learning(self) -> np.ndarray:
        return np.zeros_like(self.eta)

    def _accumulate_learning(self, learning: np.ndarray, telemetry: "_MoveTelemetry") -> np.ndarray:
        return learning + telemetry.eta_updates

    def _update_selection_state(
        self, results: List[Tuple[List[List[int]], float, float, List[str], int, int]], learning: np.ndarray
    ) -> None:
        # Pure AS pheromone evaporation
        self._evaporate_pheromones()
        # ----------------------------------------------------------------
        # Bug #1 fix: pheromone deposit includes Q constant.
        # Paper: Δτ^k_ij = Q + I_k / L_k  when I_k > 0.
        # ----------------------------------------------------------------
        self._deposit_pheromones(results)
        # Visibility integration
        self.eta = (self.cfg.eta_decay * self.eta) + learning

    def _max_iterations(self) -> int:
        return self.cfg.max_iterations

    def _timed_out(self, start_time: float) -> bool:
        return self.cfg.time_limit > 0 and time.perf_counter() - start_time > self.cfg.time_limit

    def _result(self, best_routes: List[List[int]], best_objective: float) -> SolveResult:
        if math.isinf(best_objective):
            # No feasible plan was ever seen (the starting solution was already infeasible):
            # return the starting solution rather than an over-capacity ant plan.
            best_routes = [r for r in copy.deepcopy(self.initial_solution) if r]

        best_cost = self._calculate_routing_cost(best_routes)
        collected_rev = sum(self.wastes.get(n, 0) * self.R for r in best_routes for n in r)
        final_cost = best_cost / self.C if self.C > 0 else best_cost

        return SolveResult(best_routes, collected_rev - best_cost, final_cost)

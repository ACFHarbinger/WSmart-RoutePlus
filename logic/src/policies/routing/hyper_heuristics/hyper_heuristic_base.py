"""Route-level hyper-heuristic search template (P-35).

The released ACO-HH engine and the deferred route-level hyper-heuristics
(HMM-GD-HH, SS-HH, GIHH, HULK) all follow the same loop: maintain one or more
candidate solutions, repeatedly pick low-level heuristic move(s), produce a
new candidate, accept or reject it, and feed the outcome back into the
selection state, until a time/iteration budget expires.

:class:`HyperHeuristicSearch` is that loop. Engines plug in their specifics
through small hooks:

- ``_seed_incumbent`` / ``_initial_candidates`` — run-once setup (the base
  resets these on every ``solve()`` call; cross-solve learning state such as
  pheromone matrices lives in ``__init__``);
- ``_select_move`` — the selection strategy (ACO-HH: ant-system proportional
  rule over the operator graph);
- ``_apply_move`` — run the chosen low-level heuristic(s), returning the new
  candidate plus an opaque telemetry object;
- ``_accept`` — the move-acceptance strategy (ACO-HH: ants always adopt their
  constructed solution; the incumbent gate lives in ``_update_incumbent``);
- ``_rank`` / ``_update_incumbent`` — candidate ordering and best tracking;
- ``_post_iteration`` — population management (ACO-HH: elitism sync and
  strategic oscillation);
- ``_update_selection_state`` — learning update (ACO-HH: evaporation,
  deposit, visibility EMA);
- ``_max_iterations`` / ``_timed_out`` / ``_result`` — budget and mapping to
  the ``(routes, profit, cost)`` contract.

The multi-period hyper-heuristics (SHH/MHH/PHH/GP-MP/AMPHH) operate on
plans through ``BaseMultiPeriodRoutingPolicy`` and are out of scope for this
route-level base.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from logic.src.policies.routing.base.sub_problem import SolveResult

__all__ = ["HyperHeuristicSearch"]


class HyperHeuristicSearch(ABC):
    """Template for route-level hyper-heuristic engines."""

    # ------------------------------------------------------------------
    # Template
    # ------------------------------------------------------------------

    def solve(self) -> SolveResult:
        """Run the hyper-heuristic loop once and return the best result.

        Per-call state (incumbent, candidates, stagnation) is reset here;
        cross-solve learning state is owned by the engine's ``__init__``.
        """
        best_routes, best_objective = self._seed_incumbent()
        candidates = self._initial_candidates()
        stagnation = 0
        start_time = time.perf_counter()

        for _it in range(self._max_iterations()):
            if self._timed_out(start_time):
                break

            learning = self._fresh_learning()
            results: List[Any] = []

            for k in range(self._n_candidates()):
                move = self._select_move(candidates[k], k)
                candidate, telemetry = self._apply_move(candidates[k], move, k)
                learning = self._accumulate_learning(learning, telemetry)
                candidates[k] = self._accept(candidate, candidates[k])
                results.append(self._record(candidates[k], k, telemetry))

            results = self._rank(results)
            best_routes, best_objective, improved = self._update_incumbent(
                results, best_routes, best_objective
            )
            candidates = self._post_iteration(results, improved, candidates, best_routes)
            self._update_selection_state(results, learning)
            stagnation = self._track_stagnation(improved, stagnation)

        return self._result(best_routes, best_objective)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _seed_incumbent(self) -> Tuple[List[List[int]], float]:
        """Initialise the incumbent from the starting solution."""

    @abstractmethod
    def _initial_candidates(self) -> List[Any]:
        """Build the per-solve candidate population (e.g. ants on vertices)."""

    @abstractmethod
    def _n_candidates(self) -> int:
        """Population size for this solve."""

    @abstractmethod
    def _select_move(self, candidate: Any, k: int) -> Any:
        """Pick the next low-level heuristic move for candidate ``k``."""

    @abstractmethod
    def _apply_move(self, candidate: Any, move: Any, k: int) -> Tuple[Any, Any]:
        """Apply the move; return (new_candidate, telemetry)."""

    def _accept(self, candidate: Any, current: Any) -> Any:
        """Move-acceptance strategy: default adopts the candidate."""
        return candidate

    @abstractmethod
    def _record(self, candidate: Any, k: int, telemetry: Any) -> Any:
        """Build the per-candidate result record for this iteration."""

    @abstractmethod
    def _rank(self, results: List[Any]) -> List[Any]:
        """Order the iteration results (e.g. by objective, best first)."""

    @abstractmethod
    def _update_incumbent(
        self, results: List[Any], best_routes: List[List[int]], best_objective: float
    ) -> Tuple[List[List[int]], float, bool]:
        """Fold the iteration results into the incumbent; return improvement flag."""

    @abstractmethod
    def _post_iteration(
        self, results: List[Any], improved: bool, candidates: List[Any], best_routes: List[List[int]]
    ) -> List[Any]:
        """Population management after each iteration (elitism, oscillation)."""

    @abstractmethod
    def _update_selection_state(self, results: List[Any], learning: Any) -> None:
        """Learning update (pheromones, weights, matrices)."""

    def _track_stagnation(self, improved: bool, stagnation: int) -> int:
        """Stagnation bookkeeping; engines may override (ACO-HH halves pv)."""
        return 0 if improved else stagnation + 1

    @abstractmethod
    def _fresh_learning(self) -> Any:
        """Zero the per-iteration learning accumulator."""

    @abstractmethod
    def _accumulate_learning(self, learning: Any, telemetry: Any) -> Any:
        """Fold one candidate's telemetry into the accumulator."""

    @abstractmethod
    def _max_iterations(self) -> int:
        """Iteration budget."""

    @abstractmethod
    def _timed_out(self, start_time: float) -> bool:
        """Time-budget check."""

    @abstractmethod
    def _result(self, best_routes: List[List[int]], best_objective: float) -> SolveResult:
        """Map the incumbent to the (routes, profit, cost) contract."""

"""LNSBase hooks and loop helpers of the ALNS solver.

:class:`ALNSHooksMixin` holds the
:class:`~logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.alns.ALNSSolver`
methods that specialise the shared LNS loop for ALNS (removal bounds, segment scoring,
solution snapshots and hashing, novelty tracking, search preparation, move acceptance,
per-iteration bookkeeping) and the core loop helpers (noise, weight updates, segment end).
Moved unchanged from ``alns.py``.
"""

import random
from typing import Any, Callable, List, Set, Tuple

import numpy as np

from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.policies.routing.meta_heuristics.large_neighborhood_search.engine import (
    MIN_REMOVAL_LOWER_BOUND,
)
from logic.src.policies.routing.meta_heuristics.large_neighborhood_search.weight_updates import (
    RopkePisingerSegments,
)


class ALNSHooksMixin:
    """LNSBase hooks and core loop helpers of :class:`ALNSSolver` (moved unchanged from ``alns.py``)."""

    # Provided by ALNSSolver.__init__; annotations only.
    params: ALNSConfig
    cfg: ALNSConfig
    n_nodes: int
    dist_matrix: np.ndarray
    random: random.Random
    segment_size: int
    visited_solutions: Set[Tuple[Tuple[int, ...], ...]]
    acceptance_criterion: Any
    _viz_record: Callable[..., None]
    _weights: RopkePisingerSegments

    # ------------------------------------------------------------------
    # LNSBase semantic hooks (ALNS flavour)
    # ------------------------------------------------------------------

    def _removal_bounds(self, current_n_nodes: int) -> Tuple[int, int]:
        """Formal bounds per Ropke & Pisinger (2006) Section 4.3.1.

        Args:
            current_n_nodes: Number of nodes in the current solution.

        Returns:
            Tuple of (lower_bound, upper_bound) for the removal count.
        """
        lower_bound = min(current_n_nodes, MIN_REMOVAL_LOWER_BOUND)
        upper_bound = min(self.params.max_removal_cap, int(self.params.xi * self.n_nodes))
        upper_bound = max(lower_bound, upper_bound)
        return lower_bound, upper_bound

    def _scoring(
        self,
        new_profit: float,
        best_profit: float,
        current_profit: float,
        accepted: bool,
        is_new: bool,
    ) -> float:
        """Ropke & Pisinger (2005, Section 3.4) scoring.

        σ₁ — new global best (no "not visited" qualifier)
        σ₂ — better than current, not visited before, accepted
        σ₃ — worse than current, not visited before, accepted

        Called before the best-solution bookkeeping, so ``best_profit`` is the
        pre-update value.

        Args:
            new_profit: Profit of the candidate solution.
            best_profit: Best profit found so far (pre-update).
            current_profit: Profit of the current incumbent.
            accepted: Whether the candidate was accepted.
            is_new: Whether the candidate was not visited before.

        Returns:
            The score earned by this iteration's operators.
        """
        if new_profit > best_profit + 1e-6:
            return self.params.sigma_1
        if accepted and is_new:
            return self.params.sigma_2 if new_profit > current_profit + 1e-6 else self.params.sigma_3
        return 0.0

    def _snapshot_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """Shallow copy via list comprehension (replaces deepcopy): ~10x faster.

        Args:
            routes: Routes to copy.

        Returns:
            The copied routes.
        """
        return [r[:] for r in routes]

    def _hash_solution(self, routes: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
        """Create a canonical hash of a solution for tracking visited states.

        Args:
            routes: List of routes to hash.

        Returns:
            Tuple representation of sorted routes for hashing.
        """
        sorted_routes = sorted([tuple(route) for route in routes if route])
        return tuple(sorted_routes)

    def _track_and_check_new(self, solution_hash: Tuple[Tuple[int, ...], ...]) -> bool:
        """Check whether the hashed solution was visited before.

        Args:
            solution_hash: Output of ``_hash_solution``.

        Returns:
            True when the solution is not in the visited set.
        """
        return solution_hash not in self.visited_solutions

    def _track_accepted(self, solution_hash: Tuple[Tuple[int, ...], ...]) -> None:
        """Record an accepted solution in the visited set.

        Args:
            solution_hash: Output of ``_hash_solution``.
        """
        self.visited_solutions.add(solution_hash)

    def _prepare_search(self, best_routes: List[List[int]], current_profit: float) -> None:
        """Per-solve prologue: T_start calibration, visited reset, criterion setup.

        Dynamic T_start calibration (Ropke & Pisinger 2006, Section 3.5):
        non-positive initial profits used to skip calibration and leave T = 0,
        i.e. pure hill-climbing; the fallback scale=1.0 fixes that.

        Args:
            best_routes: Best routes after initialization.
            current_profit: Profit of the initial solution.
        """
        from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory

        self.acceptance_criterion = AcceptanceCriterionFactory.build(self.cfg.acceptance_criterion)

        self.visited_solutions.clear()
        self.visited_solutions.add(self._hash_solution(best_routes))

        if self.acceptance_criterion is not None:
            self.acceptance_criterion.setup(current_profit)

    def _accept_move(
        self,
        current_profit: float,
        new_profit: float,
        best_profit: float,
        iteration: int,
        **kwargs: Any,
    ) -> bool:
        """Acceptance decision via the injected criterion (hill-climb when None).

        Args:
            current_profit: Profit of the current incumbent.
            new_profit: Profit of the candidate solution.
            best_profit: Best profit found so far.
            iteration: Current iteration index.
            kwargs: Extra context forwarded to criterion.accept.

        Returns:
            True if the candidate is accepted.
        """
        if self.acceptance_criterion is None:
            return new_profit > current_profit + 1e-6
        accept, _ = self.acceptance_criterion.accept(
            current_obj=current_profit,
            candidate_obj=new_profit,
            f_best=best_profit,
            iteration=iteration,
            max_iterations=self.params.max_iterations,
            **kwargs,
        )
        return accept

    def _extra_iteration_end(self, **kwargs: Any) -> None:
        """Segment-end weight update and criterion cooling.

        Args:
            kwargs: Iteration context (iteration, current_profit, new_profit,
                best_profit, accepted, ...).
        """
        iteration = kwargs["iteration"]
        if (iteration + 1) % self.segment_size == 0:
            self._end_segment()

        if self.acceptance_criterion is not None:
            self.acceptance_criterion.step(
                current_obj=kwargs["current_profit"],
                candidate_obj=kwargs["new_profit"],
                accepted=kwargs["accepted"],
                f_best=kwargs["best_profit"],
                iteration=iteration,
                max_iterations=self.params.max_iterations,
            )

    def _record_iteration(self, **kwargs: Any) -> None:
        """Record one telemetry snapshot for the current iteration.

        Args:
            kwargs: Iteration context (iteration, d_idx, r_idx, best_profit,
                current_profit, accepted, score).
        """
        self._viz_record(
            iteration=kwargs["iteration"],
            d_idx=kwargs["d_idx"],
            r_idx=kwargs["r_idx"],
            best_profit=kwargs["best_profit"],
            current_profit=kwargs["current_profit"],
            acceptance_state=self.acceptance_criterion.get_state() if self.acceptance_criterion else {},
            accepted=int(kwargs["accepted"]),
            score=kwargs["score"],
        )

    # ------------------------------------------------------------------
    # Core solver loop helpers
    # ------------------------------------------------------------------

    def _get_noise(self) -> float:
        """
        Generate noise for repair operators (Ropke & Pisinger 2005, Section 3.4.3).

        Noise is scaled relative to the maximum distance in the problem instance:
        noise ~ U[-eta * max_dist, eta * max_dist]

        Returns:
            float: Noise value in range [-eta * max_dist, eta * max_dist]
        """
        max_dist = self.dist_matrix.max()
        return self.random.uniform(-self.params.noise_factor, self.params.noise_factor) * max_dist

    def _update_weights(self, d_idx: int, r_idx: int, score: float) -> None:
        """Update scores and usage counts for operators.

        Args:
            d_idx: Index of the destroy operator used.
            r_idx: Index of the repair operator used.
            score: Score to add based on solution quality.
        """
        self._weights.update(d_idx, r_idx, score)

    def _end_segment(self) -> None:
        """
        Update operator weights at the end of a segment (Ropke & Pisinger 2005, Eq. 4.1).

        Formula: w_{i,j+1} = w_{i,j} * (1-r) + r * (π_i / θ_i)
        where:
            - w_{i,j} is the weight of operator i in segment j
            - r is the reaction factor (learning rate)
            - π_i is the total score accumulated by operator i
            - θ_i is the number of times operator i was used
        """
        self._weights.end_segment()

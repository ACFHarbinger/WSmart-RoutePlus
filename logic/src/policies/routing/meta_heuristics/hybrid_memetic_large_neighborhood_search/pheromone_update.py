r"""Pheromone-update and elite-pool layer for the Memetic MMAS solver.

Moved unchanged from :mod:`maco`: the mixin holding the MMAS global
pheromone update (iteration-best / best-so-far / auto schedule with
elitist blending), the per-edge pheromone deposit, the stagnation restart
and the elite-pool maintenance of the memetic layer.
"""

from typing import List, Tuple

from logic.src.policies.routing.meta_heuristics.hybrid_memetic_large_neighborhood_search.params import MACOParams
from logic.src.policies.routing.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones import (
    SparsePheromoneTau,
)


class MACOPheromoneUpdateMixin:
    r"""Global pheromone update, deposit, restart and elite-pool upkeep."""

    # Provided by MemeticACOSolver.__init__; annotations only.
    params: MACOParams
    pheromone: SparsePheromoneTau
    elite_pool: List[Tuple[float, List[List[int]]]]

    # ====================================================================== #
    # Pheromone management                                                     #
    # ====================================================================== #

    def _global_pheromone_update(
        self,
        best_routes: List[List[int]],
        best_cost: float,
        iteration_best_routes: List[List[int]],
        iteration_best_cost: float,
        iteration: int,
        ib_phase: int,
    ) -> None:
        """Apply the MMAS global pheromone update.

        The reinforcing solution (and its weight) depends on
        ``params.update_schedule`` and ``params.elitist_weight``:

        * ``"iteration_best"``  – reinforce only with iteration-best (ib).
        * ``"best_so_far"``     – reinforce only with global best (bs).
        * ``"auto"``            – use ib for the first ``ib_phase`` iterations,
          then switch to bs.

        When ``elitist_weight < 1`` a *blended* deposit is applied::

            delta_deposit = w * delta_bs + (1−w) * delta_ib

        where ``w = elitist_weight``.

        Args:
            best_routes: Global best-so-far solution.
            best_cost: Cost of the global best.
            iteration_best_routes: Best solution in the current iteration.
            iteration_best_cost: Cost of the iteration best.
            iteration: Current iteration index (0-based).
            ib_phase: Number of iterations during which IB update is preferred.
        """
        schedule = self.params.update_schedule
        w = self.params.elitist_weight

        # Determine primary reinforcement source.
        use_ib_primary = schedule == "iteration_best" or (schedule == "auto" and iteration < ib_phase)

        if w >= 1.0 or use_ib_primary:
            # ---- Single-solution reinforcement ----
            if use_ib_primary and iteration_best_cost < float("inf"):
                primary_routes, primary_cost = iteration_best_routes, iteration_best_cost
            elif best_cost < float("inf"):
                primary_routes, primary_cost = best_routes, best_cost
            else:
                return
            self._deposit(primary_routes, primary_cost, weight=1.0)

            # Secondary (elitist blend) when weight is partial.
            if w < 1.0 and best_cost < float("inf") and use_ib_primary:
                self._deposit(best_routes, best_cost, weight=1.0 - w)
            elif w < 1.0 and iteration_best_cost < float("inf") and not use_ib_primary:
                self._deposit(iteration_best_routes, iteration_best_cost, weight=1.0 - w)
        else:
            # ---- Blended reinforcement: w * BS + (1−w) * IB ----
            if best_cost < float("inf"):
                self._deposit(best_routes, best_cost, weight=w)
            if iteration_best_cost < float("inf"):
                self._deposit(iteration_best_routes, iteration_best_cost, weight=1.0 - w)

    def _deposit(self, routes: List[List[int]], cost: float, weight: float) -> None:
        """Deposit pheromone on every edge of a solution.

        The deposit amount per edge is::

            delta = weight / cost

        Following Stützle & Hoos (2000) this is added *after* evaporation,
        then clamped to ``[tau_min, tau_max]`` inside ``SparsePheromoneTau.set``.

        Args:
            routes: Solution to reinforce.
            cost: Solution cost (used as denominator).
            weight: Multiplicative weight for the deposit.
        """
        if not routes or cost <= 0:
            return
        delta = weight / cost
        for route in routes:
            if not route:
                continue
            prev = 0
            for node in route:
                self.pheromone.deposit_edge(prev, node, delta)
                prev = node
            self.pheromone.deposit_edge(prev, 0, delta)

    def _restart(self, best_routes: List[List[int]], best_cost: float) -> None:
        """Trigger a pheromone reinitialization to escape stagnation.

        All pheromones are reset to ``tau_max`` (or the value specified by
        ``params.restart_pheromone``).  If ``params.use_restart_best`` is
        ``True``, the best-so-far solution is re-deposited once after reset so
        the search does not restart from a completely flat landscape.

        Args:
            best_routes: Best solution found before the restart.
            best_cost: Cost of that solution.
        """
        tau_reset = self.params.restart_pheromone  # None → tau_max inside reinitialize
        self.pheromone.reinitialize(tau_reset)

        if self.params.use_restart_best and best_routes and best_cost < float("inf"):
            self._deposit(best_routes, best_cost, weight=1.0)

    # ====================================================================== #
    # Elite pool (memetic layer)                                               #
    # ====================================================================== #

    def _update_elite_pool(self, routes: List[List[int]], cost: float) -> None:
        """Update the elite pool with a candidate solution.

        The pool is kept sorted ascending by cost and capped at
        ``params.elite_pool_size``.  A candidate enters if the pool has spare
        capacity or the candidate beats the worst member.

        Args:
            routes: Candidate solution.
            cost: Its routing cost.
        """
        pool_size = self.params.elite_pool_size
        if pool_size <= 0 or not routes or cost >= float("inf"):
            return

        if len(self.elite_pool) < pool_size:
            self.elite_pool.append((cost, routes))
            self.elite_pool.sort(key=lambda x: x[0])
        elif cost < self.elite_pool[-1][0]:
            self.elite_pool[-1] = (cost, routes)
            self.elite_pool.sort(key=lambda x: x[0])

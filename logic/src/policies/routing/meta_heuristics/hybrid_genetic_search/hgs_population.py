"""Population management, selection, offspring and penalty adaptation of the HGS solver.

:class:`HGSPopulationMixin` holds the
:class:`~logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.hgs.HGSSolver`
methods for the fitness cache and population insertion, population initialisation,
survivor selection and clone detection, binary-tournament parent selection, offspring
generation and its bandit signals, the coverage- and margin-driven penalty adaptation
(with its ``_MIN_PENALTY``, ``_MAX_PENALTY`` and ``_TARGET_*`` constants) and best-solution
extraction. Moved unchanged from ``hgs.py``, which re-exports the constants.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.configs.policies.routing.meta_heuristics.hgs import HGSConfig
from logic.src.envs.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.local_search.local_search_hgs import HGSLocalSearch
from logic.src.policies.helpers.operators.crossover_recombination import route_profit_gpx_crossover

from .evolution import _extract_edges, evaluate, update_biased_fitness
from .individual import Individual
from .split import LinearSplit

# Penalty-adaptation constants
_MIN_PENALTY = 0.05
_MAX_PENALTY = 10.0

# Target: visit roughly 40-70% of nodes (tune per instance class)
_TARGET_COVERAGE_LOW = 0.40
_TARGET_COVERAGE_HIGH = 0.70

# Target: average route margin above this means routes are healthy
_TARGET_MARGIN = 0.10  # 10% net margin


class HGSPopulationMixin:
    """Population, selection, offspring, penalty and extraction methods of :class:`HGSSolver` (moved unchanged from ``hgs.py``)."""

    # Provided by HGSSolver.__init__; annotations only, no runtime defaults.
    params: HGSConfig
    nodes: List[int]
    random: random.Random
    split_manager: LinearSplit
    ls: HGSLocalSearch
    d: np.ndarray
    wastes: Dict[int, float]
    Q: float
    R: float
    C: float
    mandatory_nodes: List[int]
    _feas_cache: Dict[Any, Any]
    _infeas_cache: Dict[Any, Any]
    _offspring_coverage: List[float]
    _offspring_margin: List[float]
    _offspring_feasibility: List[bool]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_cache(self, ind: Individual, cache: dict, inv: Dict[int, set]) -> None:
        """
        Remove all diversity distance entries that involve a specific individual.

        Uses an inverse index for O(P) eviction instead of O(P²) full cache scan.
        Called immediately before removing an individual from a subpopulation
        so that the cache does not keep entries for individuals that left the population.

        Args:
            ind: Individual about to be removed.
            cache: Distance cache to purge (modified in-place).
            inv: Inverse index mapping ind.serial -> set of cache keys (modified in-place).

        """
        ind_id = ind.serial
        for key in inv.pop(ind_id, ()):
            cache.pop(key, None)
            # Clean up the other individual's inverse entry for this key too.
            other_id = key[0] if key[1] == ind_id else key[1]
            other_keys = inv.get(other_id)
            if other_keys is not None:
                other_keys.discard(key)

    def _insert_into_pop(
        self,
        ind: Individual,
        pop_feasible: List[Individual],
        pop_infeasible: List[Individual],
    ) -> None:
        """
        Insert an individual into its respective subpopulation.

        Fitness is intentionally NOT updated here. The biased fitness update is
        deferred to _trim_pop, where it is computed once over the full population
        at peak size (μ+λ), amortising the O(P²) cost over λ insertions rather
        than paying it on every single insertion.

        Tournament selection operates on potentially stale fitness values between
        trim cycles. This is acceptable: binary tournament is robust to slightly
        stale ranks, and Vidal (2022) does not require fresh fitness before every
        selection — only before survivor eviction decisions.

        Args:
            ind: Individual to insert.
            pop_feasible: Feasible subpopulation (modified in-place).
            pop_infeasible: Infeasible subpopulation (modified in-place).

        """
        if ind.is_feasible:
            pop_feasible.append(ind)
        else:
            pop_infeasible.append(ind)

    # ------------------------------------------------------------------
    # Population initialization
    # ------------------------------------------------------------------

    def _initialize_population(self, penalty_capacity: float) -> Tuple[List[Individual], List[Individual]]:
        """
        Initialize dual subpopulations with 4μ random solutions improved by local search.

        Following Vidal (2022) Algorithm 1: "Initialize population with random
        solutions improved by local search." Fitness is updated once after all
        individuals are generated (bulk update is more efficient than incremental
        updates during initialization).

        Args:
            penalty_capacity: Initial penalty coefficient for capacity violations.

        Returns:
            Tuple[List[Individual], List[Individual]]: (feasible pop, infeasible pop).
        """
        pop_feasible: List[Individual] = []
        pop_infeasible: List[Individual] = []

        initial_size = 4 * self.params.mu
        for _ in range(initial_size):
            gt = self.nodes[:]
            self.random.shuffle(gt)
            ind = Individual(gt)
            evaluate(ind, self.split_manager, penalty_capacity)
            ind = self.ls.optimize(ind)
            evaluate(ind, self.split_manager, penalty_capacity)

            if ind.is_feasible:
                pop_feasible.append(ind)
            else:
                pop_infeasible.append(ind)

        # Bulk fitness update after initialization (uses the instance caches which
        # were reset in solve() / restart logic before this call).
        update_biased_fitness(pop_feasible, self.params.nb_elite, self.params.nb_close, self._feas_cache)
        update_biased_fitness(pop_infeasible, self.params.nb_elite, self.params.nb_close, self._infeas_cache)

        return pop_feasible, pop_infeasible

    # ------------------------------------------------------------------
    # Survivor selection
    # ------------------------------------------------------------------

    def _trim_pop(self, pop: List[Individual], cache: dict, inv: Dict[int, set]) -> None:
        """
        Trim a subpopulation from μ+λ down to μ, removing clones first then
        worst-fitness individuals.

        Following Vidal (2022) Section 2: "iteratively removing a clone solution
        (i.e., identical to another one) if such a solution exists, or otherwise
        the worst solution in terms of fitness according to Eq. (1)."

        Fitness is computed once at the start of trim (the only point where accurate
        ranks are needed for eviction decisions), then re-ranked cheaply after each
        removal using the cached pairwise distances — no O(P²) recompute per removal.

        Args:
            pop: Subpopulation to trim (modified in-place).
            cache: Per-subpopulation distance cache (modified in-place).
            inv: Inverse index for O(P) cache eviction (modified in-place).

        """
        if len(pop) < self.params.mu + self.params.n_offspring:
            return

        # Single full O(P²) fitness computation at peak population size.
        # All subsequent re-ranks inside the loop hit the cache and cost O(P).
        update_biased_fitness(pop, self.params.nb_elite, self.params.nb_close, cache, inv)

        while len(pop) > self.params.mu:
            clone_idx = self._find_clone(pop)
            if clone_idx is not None:
                self._evict_cache(pop[clone_idx], cache, inv)
                pop.pop(clone_idx)
            else:
                worst_idx = max(range(len(pop)), key=lambda i: pop[i].fitness)
                self._evict_cache(pop[worst_idx], cache, inv)
                pop.pop(worst_idx)

            # Re-rank with cached distances — O(P log P) sort, no distance recompute.
            update_biased_fitness(pop, self.params.nb_elite, self.params.nb_close, cache, inv)

    def _find_clone(self, pop: List[Individual]) -> Optional[int]:
        """
        Return the index of any individual whose solution is identical (by edge set)
        to an earlier individual in the population, or None if no clones exist.

        Uses edge-set hashing, which is the same undirected representation used in
        broken-pairs distance computation. This correctly identifies clones
        regardless of route traversal direction: route [1,2,3] and route [3,2,1]
        produce identical edge sets {(0,1),(1,2),(2,3),(3,0)} and are thus
        correctly detected as clones.

        Args:
            pop: Subpopulation to inspect.

        Returns:
            Optional[int]: Index of a clone, or None.
        """
        seen: set = set()
        for i, ind in enumerate(pop):
            if ind._edge_key is None:
                ind._edge_key = frozenset(_extract_edges(ind))
            if ind._edge_key in seen:
                return i
            seen.add(ind._edge_key)
        return None

    # ------------------------------------------------------------------
    # Parent selection
    # ------------------------------------------------------------------

    def _select_parents(
        self,
        pop_feasible: List[Individual],
        pop_infeasible: List[Individual],
    ) -> Tuple[Individual, Individual]:
        """
        Binary tournament selection from the combined population.

        Operates directly on both subpopulation lists without allocating a merged
        list, eliminating O(P) allocation per call. Fitness may be slightly stale
        between trim cycles; this is intentional and acceptable for tournament
        selection (see _insert_into_pop docstring).

        Args:
            pop_feasible: Feasible subpopulation.
            pop_infeasible: Infeasible subpopulation.

        Returns:
            Tuple[Individual, Individual]: Two selected parents (p1, p2).
        """
        n_feas = len(pop_feasible)
        n_total = n_feas + len(pop_infeasible)

        def get(idx: int) -> Individual:
            """
            Gets an individual from the combined population.

            Args:
                idx: Index of the individual.

            Returns:
                Individual: Selected individual.
            """
            return pop_feasible[idx] if idx < n_feas else pop_infeasible[idx - n_feas]

        def tournament() -> Individual:
            """
            Selects one individual using binary tournament selection.

            Returns:
                Individual: Selected individual.
            """
            idx1, idx2 = self.random.sample(range(n_total), 2)
            i1, i2 = get(idx1), get(idx2)
            return i1 if i1.fitness < i2.fitness else i2

        return tournament(), tournament()

    # ------------------------------------------------------------------
    # Offspring generation
    # ------------------------------------------------------------------

    def _generate_offspring(
        self,
        pop_feasible: List[Individual],
        pop_infeasible: List[Individual],
        penalty_capacity: float,
    ) -> Individual:
        """
        Generate and educate a new offspring via crossover and local search.

        Fitness is maintained as an invariant and is always current on entry,
        so no recomputation is needed here. The O(P²) pairwise distance
        computation now happens only when the population actually changes
        (inside _insert_into_pop and _trim_pop), not every iteration.

        crossover_rate is now applied (was declared but never used).
        When crossover is skipped, P1 is cloned so local search can still
        educate the individual — equivalent to a mutation-only step.

        Args:
            pop_feasible: Feasible subpopulation (fitness assumed current).
            pop_infeasible: Infeasible subpopulation (fitness assumed current).
            penalty_capacity: Current penalty coefficient.

        Returns:
            Individual: Educated offspring.
        """
        p1, p2 = self._select_parents(pop_feasible, pop_infeasible)

        if self.random.random() < self.params.crossover_rate:
            child = route_profit_gpx_crossover(
                p1,
                p2,
                self.d,
                self.wastes,
                self.Q,
                self.R,
                self.C,
                rng=self.random,
                mandatory_nodes=self.mandatory_nodes,
            )
        else:
            # No crossover: clone p1 so local search education still occurs.
            child = Individual(p1.giant_tour[:], expand_pool=p1.expand_pool)

        evaluate(child, self.split_manager, penalty_capacity)
        child = self.ls.optimize(child)
        evaluate(child, self.split_manager, penalty_capacity)
        return child

    def _record_offspring_signals(self, ind: Individual) -> None:
        """
        Record coverage and profitability margin signals from the latest offspring
        to support VRPP penalty adaptation.

        Reuses cost/revenue already computed by evaluate() — no route re-traversal.

        Args:
            ind: Recently generated offspring (post-crossover and local search).

        """
        n = len(ind.giant_tour)
        visited = sum(len(r) for r in ind.routes)
        self._offspring_coverage.append(visited / n if n > 0 else 0.0)

        # Reuse evaluate()'s already-computed ind.cost and ind.revenue.
        # Per-route breakdown is not needed; the aggregate margin is sufficient
        # for the quadrant penalty adaptation signal.
        margin = (ind.revenue - ind.cost) / ind.revenue if ind.revenue > 0.0 else 0.0
        self._offspring_margin.append(margin)

    # ------------------------------------------------------------------
    # Penalty adaptation
    # ------------------------------------------------------------------

    def _adjust_penalties(self, current_penalty: float) -> float:
        """
        Adapt the capacity penalty coefficient to maintain the target proportion
        of feasible offspring.

        Vidal (2022) Section 2 specifies "monitoring the number of feasible
        solutions *obtained*" - the feasibility rate of recently produced
        offspring after local-search education, not the accumulated population
        ratio. The population ratio is a lagging indicator that conflates the
        current penalty's effect with historical decisions. The offspring-feasibility
        rate directly reflects the current penalty's pressure on the search.

        Uses the last 100 offspring entries from self._offspring_feasibility,
        matching the every-100-iterations adaptation frequency.

        Args:
            current_penalty: Current capacity penalty coefficient.

        Returns:
            float: Updated penalty coefficient.
        """
        if not current_expand_pool():
            if not self._offspring_feasibility:
                return current_penalty

            recent = self._offspring_feasibility[-100:]
            feasible_ratio = sum(recent) / len(recent)

            if feasible_ratio > self.params.target_feasible + 0.05:
                # Too many feasible offspring: reduce penalty to allow more infeasibility
                return current_penalty * self.params.penalty_decrease
            elif feasible_ratio < self.params.target_feasible - 0.05:
                # Too few feasible offspring: increase penalty to push toward feasibility
                return current_penalty * self.params.penalty_increase
            return current_penalty
        else:
            # VRPP adaptation: balance coverage vs margin
            recent_cov = self._offspring_coverage[-100:]
            recent_mar = self._offspring_margin[-100:]
            if not recent_cov:
                return current_penalty

            avg_coverage = sum(recent_cov) / len(recent_cov)
            avg_margin = sum(recent_mar) / len(recent_mar)

            # Quadrant logic: coverage and margin jointly determine direction
            coverage_low = avg_coverage < _TARGET_COVERAGE_LOW
            coverage_high = avg_coverage > _TARGET_COVERAGE_HIGH
            margin_low = avg_margin < _TARGET_MARGIN
            if coverage_low and not margin_low:
                # Visiting too few nodes but routes are healthy: relax penalty so Split
                # considers more borderline nodes
                new = current_penalty * self.params.penalty_decrease
            elif coverage_high and margin_low:
                # Visiting too many nodes but profitability is poor: tighten penalty to
                # force Split to drop marginal nodes
                new = current_penalty * self.params.penalty_increase
            elif coverage_low and margin_low:
                # Both bad: routes are unprofitable AND too sparse. This usually means
                # the instance is hard or the tour ordering is poor. Nudge penalty up
                # slightly to consolidate what's being visited before expanding.
                new = current_penalty * (self.params.penalty_increase**0.5)
            else:
                # Healthy: coverage and margin both acceptable
                new = current_penalty

            return max(_MIN_PENALTY, min(_MAX_PENALTY, new))

    # ------------------------------------------------------------------
    # Solution extraction
    # ------------------------------------------------------------------

    def _get_best_solution(
        self, pop_feasible: List[Individual], pop_infeasible: List[Individual]
    ) -> Tuple[List[List[int]], float, float]:
        """
        Return the best feasible solution found, or the best infeasible solution
        if no feasible individuals remain in the population.

        Args:
            pop_feasible: Feasible subpopulation.
            pop_infeasible: Infeasible subpopulation.

        Returns:
            Tuple[List[List[int]], float, float]: Routes, profit, and cost.
        """
        if pop_feasible:
            best_ind = max(pop_feasible, key=lambda x: x.profit_score)
        elif pop_infeasible:
            best_ind = max(pop_infeasible, key=lambda x: x.profit_score)
        else:
            return [], 0.0, 0.0
        return best_ind.routes, best_ind.profit_score, best_ind.cost

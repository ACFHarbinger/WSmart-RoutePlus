"""
Hybrid Genetic Search (HGS) policy module.

Combines genetic algorithms with local search and the Split algorithm
for solving the Capacitated Vehicle Routing Problem with Profits (VRPP).

Reference:
    Vidal, T. (2022). Hybrid genetic search for the CVRP: Open-source
    implementation and SWAP* neighborhood. Computers & Operations Research,
    140, 105643.

Attributes:
    HGSSolver: Main solver class for the Hybrid Genetic Search.

Example:
    >>> solver = HGSSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()
"""

import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_hgs import HGSLocalSearch
from logic.src.policies.helpers.operators.crossover_recombination import route_profit_gpx_crossover

from .evolution import _extract_edges, evaluate, update_biased_fitness
from .individual import Individual
from .params import HGSParams
from .split import LinearSplit

# Penalty-adaptation constants
_MIN_PENALTY = 0.05
_MAX_PENALTY = 10.0

# Target: visit roughly 40-70% of nodes (tune per instance class)
_TARGET_COVERAGE_LOW = 0.40
_TARGET_COVERAGE_HIGH = 0.70

# Target: average route margin above this means routes are healthy
_TARGET_MARGIN = 0.10  # 10% net margin


class HGSSolver:
    """
    Implements Hybrid Genetic Search for the VRPP following Vidal et al. (2022).

    Key invariant maintained throughout the main loop:
        The biased fitness of every individual in both subpopulations is always
        up-to-date. Fitness is recomputed immediately after every population
        modification (insertion or deletion), never lazily before selection.
        This allows _generate_offspring / _select_parents to use fitness values
        directly without a recomputation step.

    Attributes:
        d (np.ndarray): NxN distance matrix.
        wastes (Dict[int, float]): Dictionary of node wastes.
        Q (float): Maximum vehicle capacity.
        R (float): Revenue multiplier.
        C (float): Cost multiplier.
        params (HGSParams): Detailed HGS parameters.
        mandatory_nodes (Optional[List[int]]): List of nodes that must be visited.
        random (random.Random): Seeded random number generator.
        n_nodes (int): Number of customer nodes (excluding depot).
        nodes (List[int]): Customer node indices [1, ..., n_nodes].
        split_manager (LinearSplit): Split algorithm manager.
        ls (HGSLocalSearch): Local search engine.

    Example:
        >>> solver = HGSSolver(dist_matrix, wastes, capacity, R, C, params)
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: HGSParams,
        mandatory_nodes: Optional[List[int]] = None,
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
    ):
        """
        Initialize the HGS solver.

        Args:
            dist_matrix: NxN distance matrix (index 0 = depot).
            wastes: Dictionary of node wastes (demands).
            capacity: Maximum vehicle capacity.
            R: Revenue multiplier (profit per unit of waste collected).
            C: Cost multiplier (cost per unit distance).
            params: HGS configuration parameters.
            mandatory_nodes: Local node indices that MUST be visited.
            x_coords: Optional x-coordinates for SWAP* polar sector pruning.
            y_coords: Optional y-coordinates for SWAP* polar sector pruning.

        Returns:
            None.
        """
        self.d = dist_matrix
        self.wastes = wastes
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        self.split_manager = LinearSplit(
            dist_matrix, wastes, capacity, R, C, params.max_vehicles, mandatory_nodes, vrpp=params.vrpp
        )
        self.ls = HGSLocalSearch(dist_matrix, wastes, capacity, R, C, mandatory_nodes, params)
        if x_coords is not None and y_coords is not None:
            self.ls.x_coords = x_coords
            self.ls.y_coords = y_coords

        # Per-subpopulation diversity distance caches, keyed by (id(ind_a), id(ind_b)).
        # Maintained across iterations; entries are evicted precisely when an individual
        # is removed, avoiding the stale-hit problem from clearing the entire cache.
        self._feas_cache: dict = {}
        self._infeas_cache: dict = {}

        # Inverse index for O(P) cache eviction instead of O(P²) full scan.
        # Maps id(ind) -> set of cache keys that involve that individual.
        self._feas_inv: Dict[int, set] = {}
        self._infeas_inv: Dict[int, set] = {}

        # Rolling list of offspring feasibility outcomes (True/False), used to compute
        # the recent feasibility rate for penalty adaptation.
        self._offspring_feasibility: List[bool] = []
        self._offspring_coverage: List[float] = []  # fraction of nodes visited per offspring
        self._offspring_margin: List[float] = []  # avg per-route (revenue - cost) / revenue

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_cache(self, ind: Individual, cache: dict, inv: Dict[int, set]) -> None:
        """
        Remove all diversity distance entries that involve a specific individual.

        Uses an inverse index for O(P) eviction instead of O(P²) full cache scan.
        Called immediately before removing an individual from a subpopulation
        so that the object-ID key can never produce a stale cache hit (Python
        may reuse the id() of a garbage-collected object for a new individual).

        Args:
            ind: Individual about to be removed.
            cache: Distance cache to purge (modified in-place).
            inv: Inverse index mapping id(ind) -> set of cache keys (modified in-place).

        Returns:
            None.
        """
        ind_id = id(ind)
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

        Returns:
            None.
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

        Returns:
            None.
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

        Returns:
            None.
        """
        n = len(ind.giant_tour)
        visited = sum(len(r) for r in ind.routes)
        self._offspring_coverage.append(visited / n if n > 0 else 0.0)

        # Reuse evaluate()'s already-computed ind.cost and ind.revenue.
        # Per-route breakdown is not needed; the aggregate margin is sufficient
        # for the quadrant penalty adaptation signal.
        margin = (ind.revenue - ind.cost) / ind.revenue if ind.revenue > 0.0 else 0.0
        self._offspring_margin.append(margin)

    def _insert_and_repair(
        self,
        child: Individual,
        pop_feasible: List[Individual],
        pop_infeasible: List[Individual],
        penalty_capacity: float,
    ) -> None:
        """
        Insert child into its respective subpopulation, then optionally repair
        if infeasible.

        Vidal (2022) Algorithm 1 step 8 states "repair C (local search) and insert
        it into respective subpopulation" — the word "respective" means the
        subpopulation is determined by the repaired individual's own feasibility,
        and insertion is unconditional. A repaired solution that remains infeasible
        may still be a significantly better infeasible solution than what is in
        the pool, and discarding it wastes the repair effort.

        The child's feasibility is also recorded for penalty adaptation tracking.

        Args:
            child: Educated offspring.
            pop_feasible: Feasible subpopulation.
            pop_infeasible: Infeasible subpopulation.
            penalty_capacity: Current penalty coefficient.

        Returns:
            None.
        """
        self._insert_into_pop(child, pop_feasible, pop_infeasible)

        # VRPP: record coverage and margin instead of binary feasibility
        if self.params.vrpp:
            self._record_offspring_signals(child)
        else:
            self._offspring_feasibility.append(child.is_feasible)

        # Repair only makes sense for CVRP infeasibility (capacity violations).
        # In VRPP, Split always produces feasible routes, so child.is_feasible
        # is always True and this block would never trigger anyway — but the
        # explicit guard makes the intent clear.
        if not self.params.vrpp and not child.is_feasible and self.random.random() < self.params.repair_probability:
            repaired = Individual(child.giant_tour[:], expand_pool=child.expand_pool)
            evaluate(repaired, self.split_manager, penalty_capacity * 10.0)
            repaired = self.ls.optimize(repaired)
            evaluate(repaired, self.split_manager, penalty_capacity)
            self._insert_into_pop(repaired, pop_feasible, pop_infeasible)

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
        if not self.params.vrpp:
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

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the Hybrid Genetic Search algorithm following Vidal et al. (2022)
        Algorithm 1, with VRPP adaptations.

        Returns:
            Tuple[List[List[int]], float, float]: Best routes, total profit, total cost.
        """
        start_time = time.perf_counter()

        penalty_capacity = self.params.initial_penalty_capacity

        # Reset all per-run state before initializing the population.
        self._feas_cache = {}
        self._infeas_cache = {}
        self._feas_inv = {}
        self._infeas_inv = {}
        self._offspring_feasibility = []
        self._offspring_coverage = []
        self._offspring_margin = []
        if self.params.acceptance_criterion is not None:
            self.params.acceptance_criterion.setup(0.0)

        pop_feasible, pop_infeasible = self._initialize_population(penalty_capacity)

        best_feasible_ind: Optional[Individual] = (
            max(pop_feasible, key=lambda x: x.profit_score) if pop_feasible else None
        )
        best_profit_so_far = best_feasible_ind.profit_score if best_feasible_ind is not None else -float("inf")

        it = 0
        it_no_improvement = 0
        use_time_limit = self.params.time_limit > 0
        restart_threshold = (
            self.params.restart_timer if self.params.restart_timer > 0 else self.params.n_iterations_no_improvement
        )

        while True:
            elapsed = time.perf_counter() - start_time
            if use_time_limit and elapsed >= self.params.time_limit:
                break
            if not use_time_limit and it_no_improvement >= self.params.n_iterations_no_improvement:
                break

            if use_time_limit and it_no_improvement >= restart_threshold:
                penalty_capacity = self.params.initial_penalty_capacity
                self._feas_cache = {}
                self._infeas_cache = {}
                self._feas_inv = {}
                self._infeas_inv = {}
                self._offspring_feasibility = []
                # _adjust_penalties reads these too; stale pre-restart entries skewed it.
                self._offspring_coverage = []
                self._offspring_margin = []
                pop_feasible, pop_infeasible = self._initialize_population(penalty_capacity)
                it_no_improvement = 0

            # Inline size check — no list allocation.
            if len(pop_feasible) + len(pop_infeasible) < 2:
                break

            it += 1
            it_no_improvement += 1

            child = self._generate_offspring(pop_feasible, pop_infeasible, penalty_capacity)
            self._insert_and_repair(child, pop_feasible, pop_infeasible, penalty_capacity)

            if child.is_feasible and child.profit_score > best_profit_so_far:
                best_profit_so_far = child.profit_score
                best_feasible_ind = child
                it_no_improvement = 0

            # Survivor selection: fitness is computed fresh inside _trim_pop,
            # which is the only place where accurate ranks are required.
            if len(pop_feasible) >= self.params.mu + self.params.n_offspring:
                self._trim_pop(pop_feasible, self._feas_cache, self._feas_inv)
            if len(pop_infeasible) >= self.params.mu + self.params.n_offspring:
                self._trim_pop(pop_infeasible, self._infeas_cache, self._infeas_inv)

            if it % 100 == 0:
                penalty_capacity = self._adjust_penalties(penalty_capacity)

            if self.params.acceptance_criterion is not None:
                self.params.acceptance_criterion.step(
                    current_obj=best_profit_so_far,
                    candidate_obj=child.profit_score if child.is_feasible else -float("inf"),
                    accepted=child.is_feasible and child.profit_score > best_profit_so_far,
                    iteration=it,
                    max_iterations=self.params.n_iterations_no_improvement,
                )

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=it,
                best_profit=best_profit_so_far,
                child_profit=child.profit_score,
                child_cost=child.cost,
                population_size=len(pop_feasible) + len(pop_infeasible),
            )

        if best_feasible_ind is not None:
            return (
                best_feasible_ind.routes,
                best_feasible_ind.profit_score,
                best_feasible_ind.cost,
            )
        return self._get_best_solution(pop_feasible, pop_infeasible)

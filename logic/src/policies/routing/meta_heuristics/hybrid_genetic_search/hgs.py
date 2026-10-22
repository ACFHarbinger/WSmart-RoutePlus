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

Population management, selection, offspring generation, penalty adaptation (with its
constants, re-exported here) and best-solution extraction live in ``hgs_population.py``
(:class:`HGSPopulationMixin`).
"""

import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from logic.src.configs.policies.hgs import HGSConfig
from logic.src.pipeline.learning.envs.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.local_search.local_search_hgs import HGSLocalSearch
from logic.src.policies.routing.base.sub_problem import SubProblem

from .evolution import evaluate
from .hgs_population import HGSPopulationMixin
from .individual import Individual
from .split import LinearSplit


class HGSSolver(HGSPopulationMixin):
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
        params (HGSConfig): Detailed HGS parameters.
        mandatory_nodes (Optional[List[int]]): List of nodes that must be visited.
        random (random.Random): Seeded random number generator.
        n_nodes (int): Number of customer nodes (excluding depot).
        nodes (List[int]): Customer node indices [1, ..., n_nodes].
        split_manager (LinearSplit): Split algorithm manager.
        ls (HGSLocalSearch): Local search engine.

    Example:
        >>> solver = HGSSolver(problem, params)
    """

    def __init__(
        self,
        problem: SubProblem,
        cfg: Optional[Union[HGSConfig, Dict[str, Any]]] = None,
    ):
        """
        Initialize the HGS solver.

        Args:
            problem: The routing sub-problem (distance matrix, wastes,
                capacity, revenue, cost unit, mandatory nodes, optional
                x/y coordinates for SWAP* polar sector pruning).
            cfg: HGS configuration dataclass or dictionary (coerced leniently:
                unknown keys are filtered, ``lambda_param`` aliases
                ``n_offspring``).

        """
        dist_matrix = problem.dist_matrix
        wastes = problem.wastes
        capacity = problem.capacity
        R = problem.revenue
        C = problem.cost_unit
        mandatory_nodes = problem.mandatory_nodes
        x_coords = problem.x_coords
        y_coords = problem.y_coords
        params = cfg

        self.problem = problem
        self.d = dist_matrix
        self.wastes = wastes
        self.Q = capacity
        self.R = R
        self.C = C

        if hasattr(params, "__dict__") and not isinstance(params, HGSConfig):
            params = vars(params)
        if isinstance(params, dict):
            import dataclasses

            p = dict(params)
            if "lambda_param" in p and "n_offspring" not in p:
                p["n_offspring"] = p["lambda_param"]
            valid_keys = {f.name for f in dataclasses.fields(HGSConfig)}
            cfg = HGSConfig(**{k: v for k, v in p.items() if k in valid_keys})
        elif params is None:
            cfg = HGSConfig()
        else:
            cfg = params

        self.cfg: HGSConfig = cfg
        self.params = cfg
        self.mandatory_nodes = mandatory_nodes
        self.random = random.Random(cfg.seed) if cfg.seed is not None else random.Random()

        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))

        self.split_manager = LinearSplit(dist_matrix, wastes, capacity, R, C, cfg.max_vehicles, mandatory_nodes)
        self.ls = HGSLocalSearch(dist_matrix, wastes, capacity, R, C, mandatory_nodes, cfg)
        if x_coords is not None and y_coords is not None:
            self.ls.x_coords = x_coords
            self.ls.y_coords = y_coords

        # Modular Acceptance Criterion - fresh per-solve instance (P-11)
        from logic.src.policies.acceptance_criteria.base.factory import (
            AcceptanceCriterionFactory,
        )

        acc_val = getattr(self.cfg, "acceptance_criterion", None)
        if hasattr(acc_val, "accept"):
            self.acceptance_criterion = acc_val
        elif acc_val is not None and hasattr(acc_val, "method"):
            self.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acc_val.method,
                config=acc_val.params,
            )
        elif isinstance(acc_val, dict) and "method" in acc_val:
            self.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acc_val["method"],
                config=acc_val.get("params"),
            )
        else:
            self.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")
        self.params.acceptance_criterion = self.acceptance_criterion

        # Per-subpopulation diversity distance caches, keyed by (ind_a.serial, ind_b.serial).
        # Maintained across iterations; entries are evicted precisely when an individual
        # is removed, avoiding the stale-hit problem from clearing the entire cache.
        self._feas_cache: dict = {}
        self._infeas_cache: dict = {}

        # Inverse index for O(P) cache eviction instead of O(P²) full scan.
        # Maps ind.serial -> set of cache keys that involve that individual.
        self._feas_inv: Dict[int, set] = {}
        self._infeas_inv: Dict[int, set] = {}

        # Rolling list of offspring feasibility outcomes (True/False), used to compute
        # the recent feasibility rate for penalty adaptation.
        self._offspring_feasibility: List[bool] = []
        self._offspring_coverage: List[float] = []  # fraction of nodes visited per offspring
        self._offspring_margin: List[float] = []  # avg per-route (revenue - cost) / revenue

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

        """
        self._insert_into_pop(child, pop_feasible, pop_infeasible)

        # VRPP: record coverage and margin instead of binary feasibility
        if current_expand_pool():
            self._record_offspring_signals(child)
        else:
            self._offspring_feasibility.append(child.is_feasible)

        # Repair only makes sense for CVRP infeasibility (capacity violations).
        # In VRPP, Split always produces feasible routes, so child.is_feasible
        # is always True and this block would never trigger anyway — but the
        # explicit guard makes the intent clear.
        if (
            not current_expand_pool()
            and not child.is_feasible
            and self.random.random() < self.params.repair_probability
        ):
            repaired = Individual(child.giant_tour[:], expand_pool=child.expand_pool)
            evaluate(repaired, self.split_manager, penalty_capacity * 10.0)
            repaired = self.ls.optimize(repaired)
            evaluate(repaired, self.split_manager, penalty_capacity)
            self._insert_into_pop(repaired, pop_feasible, pop_infeasible)

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
        if self.acceptance_criterion is not None:
            self.acceptance_criterion.setup(0.0)

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
            if self.params.max_iterations > 0 and it >= self.params.max_iterations:
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

            if self.acceptance_criterion is not None:
                self.acceptance_criterion.step(
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

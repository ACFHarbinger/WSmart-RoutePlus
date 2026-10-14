"""
Variable Neighborhood Search (VNS) for VRPP.

VNS systematically changes neighborhood structures to escape local optima.
Each outer iteration consists of two phases:

  1. Shaking: generate a random neighbour in the k-th shaking structure N_k
     (increasing severity as k grows).
  2. Local search descent: apply repeated LLH improvement from the shaken
     solution until no further improvement within the budget.

Attributes:
    VNSSolver (Type): Core solver class for the Variable Neighborhood Search.
    VNSParams (Type): Parameter dataclass for the solver.

Example:
    >>> solver = VNSSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()

Reference:
    Mladenović, N., & Hansen, P. "Variable Neighborhood Search", 1997.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.search_heuristics.destroy_repair_llh import (
    build_greedy_initial_routes,
    llh_cluster_greedy,
    llh_random_greedy,
    llh_random_regret_2,
    llh_worst_greedy,
    llh_worst_regret_2,
    routes_net_profit,
    routes_total_distance,
)

from .params import VNSParams


class VNSSolver:
    """
    Variable Neighborhood Search solver for VRPP.

    Attributes:
        dist_matrix (np.ndarray): Symmetric distance matrix.
        wastes (Dict[int, float]): Mapping of bin IDs to waste quantities.
        capacity (float): Maximum vehicle collection capacity.
        R (float): Revenue per kg of waste.
        C (float): Cost per km traveled.
        params (VNSParams): Algorithm-specific parameters.
        mandatory_nodes (List[int]): Nodes that must be visited.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: VNSParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """Initializes the Variable Neighborhood Search solver.

        Args:
            dist_matrix (np.ndarray): Symmetric distance matrix.
            wastes (Dict[int, float]): Mapping of bin IDs to waste quantities.
            capacity (float): Maximum vehicle collection capacity.
            R (float): Revenue per kg of waste.
            C (float): Cost per km traveled.
            params (VNSParams): Algorithm-specific parameters.
            mandatory_nodes (Optional[List[int]]): Nodes that must be visited.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes or []
        self.n_nodes = len(dist_matrix) - 1
        self.nodes = list(range(1, self.n_nodes + 1))
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()

        # Shaking neighborhoods N_1 ... N_{k_max} ordered by increasing severity
        self._neighborhoods = [
            self._shake_n1,
            self._shake_n2,
            self._shake_n3,
            self._shake_n4,
            self._shake_n5,
        ]

        # LLH pool for local search descent phase
        self._llh_pool = [
            self._llh0,
            self._llh1,
            self._llh2,
            self._llh3,
            self._llh4,
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run Variable Neighborhood Search.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.perf_counter()
        k_max = min(self.params.k_max, len(self._neighborhoods))

        # Initialize solution
        routes = self._build_initial_solution()
        profit = self._evaluate(routes)

        best_routes = copy.deepcopy(routes)
        best_profit = profit

        # Setup modular acceptance criterion
        self.params.acceptance_criterion.setup(profit)

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.perf_counter() - start > self.params.time_limit:
                break

            k = 0  # 0-based index into self._neighborhoods
            while k < k_max:
                if self.params.time_limit > 0 and time.perf_counter() - start > self.params.time_limit:
                    break

                # === Shaking phase ===
                try:
                    shaken = self._neighborhoods[k](copy.deepcopy(routes))
                except Exception:
                    k += 1
                    continue

                # === Local search descent phase ===
                ls_routes, ls_profit = self._local_search(shaken, start)

                # === Move or not (Modular acceptance criterion) ===
                is_accepted, _ = self.params.acceptance_criterion.accept(
                    current_obj=profit,
                    candidate_obj=ls_profit,
                    iteration=iteration,
                    max_iterations=self.params.max_iterations,
                )

                if is_accepted:
                    routes = ls_routes
                    profit = ls_profit
                    if profit > best_profit:
                        best_routes = copy.deepcopy(routes)
                        best_profit = profit
                    k = 0  # Reset to first neighborhood on acceptance
                else:
                    k += 1  # Advance to next neighborhood on rejection

                # Step the criterion after transition decision
                self.params.acceptance_criterion.step(
                    current_obj=profit,
                    candidate_obj=ls_profit,
                    accepted=is_accepted,
                    iteration=iteration,
                )

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=self._cost(best_routes),
            )

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Shaking neighborhoods (N_1 ... N_5, increasing severity)
    # ------------------------------------------------------------------

    def _shake_n1(self, routes: List[List[int]]) -> List[List[int]]:
        """
        N_1: Remove 1 node randomly, greedy reinsert.

        Args:
            routes: List of routes.
            n: Number of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        return llh_random_greedy(
            routes,
            1,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
            profit_aware=use_profit,
            rng=self.random,
        )

    def _shake_n2(self, routes: List[List[int]]) -> List[List[int]]:
        """
        N_2: Remove 2 nodes randomly, greedy reinsert.

        Args:
            routes: List of routes.
            n: Number of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        return llh_random_greedy(
            routes,
            2,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
            profit_aware=use_profit,
            rng=self.random,
        )

    def _shake_n3(self, routes: List[List[int]]) -> List[List[int]]:
        """
        N_3: Worst removal of 2 nodes, regret-2 reinsert.

        Args:
            routes: List of routes.
            n: Number of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        return llh_worst_regret_2(
            routes,
            2,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
            profit_aware=use_profit,
            rng=self.random,
        )

    def _shake_n4(self, routes: List[List[int]]) -> List[List[int]]:
        """
        N_4: Cluster removal of 3 nodes, greedy reinsert.

        Args:
            routes: List of routes.
            n: Number of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        return llh_cluster_greedy(
            routes,
            3,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
            profit_aware=use_profit,
            rng=self.random,
            nodes=self.nodes,
        )

    def _shake_n5(self, routes: List[List[int]]) -> List[List[int]]:
        """
        N_5: Remove 3 nodes randomly, regret-2 reinsert.

        Args:
            routes: List of routes.
            n: Number of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        return llh_random_regret_2(
            routes,
            3,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
            profit_aware=use_profit,
            rng=self.random,
        )

    # ------------------------------------------------------------------
    # Local search descent
    # ------------------------------------------------------------------

    def _local_search(
        self,
        routes: List[List[int]],
        start: float,
    ) -> Tuple[List[List[int]], float]:
        """
        Apply repeated LLH improvement until no further progress or budget reached.

        Args:
            routes: Starting solution for the descent.
            start: Wall-clock start time of the outer solve() call.

        Returns:
            (routes, profit) after descent.
        """
        profit = self._evaluate(routes)

        for _ in range(self.params.local_search_iterations):
            if self.params.time_limit > 0 and time.perf_counter() - start > self.params.time_limit:
                break

            llh_idx = self.random.randint(0, self.params.n_llh - 1)
            llh = self._llh_pool[llh_idx]

            try:
                new_routes = llh(copy.deepcopy(routes), self.params.n_removal)
                new_profit = self._evaluate(new_routes)
            except Exception:
                continue

            if new_profit > profit:
                routes = new_routes
                profit = new_profit

        return routes, profit

    # ------------------------------------------------------------------
    # LLH pool
    # ------------------------------------------------------------------

    def _llh0(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """
        Local neighborhood search.

        Args:
            routes: List of routes.
            n: Number of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        return llh_random_greedy(
            routes,
            n,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
            profit_aware=use_profit,
            rng=self.random,
        )

    def _llh1(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """
        Local neighborhood search.

        Args:
            routes: List of routes.
            n: Number of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        return llh_worst_regret_2(
            routes,
            n,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
            profit_aware=use_profit,
            rng=self.random,
        )

    def _llh2(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """
        Local neighborhood search.

        Args:
            routes: List of routes.
            n: Number of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        return llh_cluster_greedy(
            routes,
            n,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
            profit_aware=use_profit,
            rng=self.random,
            nodes=self.nodes,
        )

    def _llh3(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """
        Local neighborhood search.

        Args:
            routes: List of routes.
            n: Number of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        return llh_worst_greedy(
            routes,
            n,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
            profit_aware=use_profit,
            rng=self.random,
        )

    def _llh4(self, routes: List[List[int]], n: int) -> List[List[int]]:
        """
        Local neighborhood search.

        Args:
            routes: List of routes.
            n: Number of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        return llh_random_regret_2(
            routes,
            n,
            self.dist_matrix,
            self.wastes,
            self.capacity,
            self.R,
            self.C,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=expand_pool,
            profit_aware=use_profit,
            rng=self.random,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_initial_solution(self) -> List[List[int]]:
        """
        Build an initial solution.

        Args:
            routes: List of routes.

        Returns:
            List[List[int]]: List of routes.
        """
        return build_greedy_initial_routes(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
            rng=self.random,
        )

    def _evaluate(self, routes: List[List[int]]) -> float:
        """
        Calculate the total profit of the routes.

        Args:
            routes: List of routes.

        Returns:
            float: Total profit.
        """
        return routes_net_profit(routes, self.dist_matrix, self.wastes, self.R, self.C)

    def _cost(self, routes: List[List[int]]) -> float:
        """
        Calculate the total cost of the routes.

        Args:
            routes: List of routes.

        Returns:
            float: Total cost.
        """
        return routes_total_distance(routes, self.dist_matrix)

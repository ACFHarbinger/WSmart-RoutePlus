"""
Reinforcement Learning + Great Deluge Hyper-Heuristic (RL-GD-HH) for VRPP.

This solver implements the adaptive selection of Low-Level Heuristics (LLHs)
using a Reinforcement Learning (RL) mechanism paired with a Great Deluge (GD)
acceptance criterion, as specified in Ozcan et al. (2010).

Algorithm (Figure 2 in the paper):
    1. Generate an initial solution S_current; compute f0 = quality(S_current).
    2. Initialise utilities: u_i = 0.75 × maxUtilityValue for all i.
    3. Set level = f0;  qualityLB = 0.
    4. While t < totalTime:
        a. Select LLH i with highest utility (random tie-break).
        b. Apply LLH i → S_temp, f_temp.
        c. Reward / punish u_i & Accept / Reject S_temp:
             If f_temp > f_current:
                 Reward u_i.
                 Accept (S_current = S_temp).
             Else:
                 Punish u_i.
                 Accept (S_current = S_temp) iff f_temp >= level.
        d. Update level: level(t) = qualityLB + (f0 - qualityLB)×(1 - t/T).
    5. Return best solution seen.

RL Punishment Variants (Section 3.2):
    RL1 (default): u ← max(lb, u − penalty)          [subtractive]
    RL2:           u ← floor(u / 2)                   [divisional]
    RL3:           u ← floor(sqrt(u))                 [root]

References:
    Ozcan, E., Misir, M., Ochoa, G., & Burke, E. K. (2010).
    "A Reinforcement Learning – Great-Deluge Hyper-heuristic for
    Examination Timetabling." Informs Journal on Computing.

Attributes:
    RLGDHHSolver: Solver for RL-GD-HH.

Example:
    >>> from logic.src.policies.routing.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic import RLGDHHSolver
    >>> solver = RLGDHHSolver()
    >>> solution = solver.solve()
    >>> print(solution)
"""

import copy
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators import (
    build_grasp_routes,
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
    regret_2_insertion,
    regret_2_profit_insertion,
    shaw_profit_removal,
    shaw_removal,
    string_removal,
)

from logic.src.configs.policies.rl_gd_hh import RLGDHHConfig


class RLGDHHSolver:
    """
    RL-GD-HH solver implementation for VRPP.

    Implements the algorithm of Ozcan et al. (2010) adapted for the
    Vehicle Routing Problem with Profits (VRPP).

    Attributes:
        dist_matrix: Distance matrix.
        wastes: Waste at each node.
        capacity: Capacity of each vehicle.
        R: Recycling rate.
        C: Cost of recycling.
        params: Parameters for RL-GD-HH.
        mandatory_nodes: Mandatory nodes.
        n_nodes: Number of nodes.
        nodes: List of nodes.
        random: Random number generator.
        utilities: Utilities for each LLH.
        _llh_pool: Pool of LLHs.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: RLGDHHConfig,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """Initialize the RL-GD-HH solver.

        Args:
            dist_matrix (np.ndarray): Distance matrix.
            wastes (Dict[int, float]): Waste at each node.
            capacity (float): Capacity of each vehicle.
            R (float): Recycling rate.
            C (float): Cost of recycling.
            params (RLGDHHConfig): Parameters for RL-GD-HH.
            mandatory_nodes (Optional[List[int]]): Mandatory nodes.
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
        self.random = random.Random(params.seed)  # None → truly random seed per Python stdlib

        # Initialise LLH utilities: u_i = 0.75 × maxUtilityValue (paper p. 16)
        self.utilities = [self.params.initial_utility] * 4
        self._llh_pool = [
            self._llh_relocate,
            self._llh_shaw,
            self._llh_string,
            self._llh_regret2,
        ]

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """Runs the RL-GD-HH metaheuristic (Figure 2, Ozcan et al. 2010).

        Returns:
            Tuple[List[List[int]], float, float]: Routes, profit, and cost.
        """
        if not self.nodes:
            return [], 0.0, 0.0

        start_time = time.perf_counter()

        # Initialisation via GRASP (Randomized Complete Solution)
        current_routes = self._initialize_solution()
        current_profit = self._evaluate(current_routes)

        best_routes = copy.deepcopy(current_routes)
        best_profit = current_profit

        self.params.acceptance_criterion.setup(current_profit)

        for _iteration in range(self.params.max_iterations):
            time_elapsed = time.perf_counter() - start_time
            if self.params.time_limit > 0 and time_elapsed > self.params.time_limit:
                break

            llh_idx = self._select_llh()
            llh = self._llh_pool[llh_idx]

            new_routes = llh(current_routes)
            new_profit = self._evaluate(new_routes)

            # Pass critical timing data to the GD criterion
            is_accepted, _ = self.params.acceptance_criterion.accept(
                current_obj=current_profit,
                candidate_obj=new_profit,
                time_elapsed=time_elapsed,
                time_limit=self.params.time_limit,
            )

            if new_profit > current_profit:
                self.utilities[llh_idx] = self._apply_reward(self.utilities[llh_idx])
            else:
                self.utilities[llh_idx] = self._apply_punishment(self.utilities[llh_idx])

            if is_accepted:
                current_routes = new_routes
                current_profit = new_profit
                if current_profit > best_profit:
                    best_routes = copy.deepcopy(current_routes)
                    best_profit = current_profit

            # Step logic is preserved for potential interface needs
            self.params.acceptance_criterion.step(
                current_obj=current_profit,
                candidate_obj=new_profit,
                accepted=is_accepted,
            )

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # RL Adaptation Helpers
    # ------------------------------------------------------------------

    def _select_llh(self) -> int:
        """Selects LLH index using max-utility with ties broken randomly.

        Returns:
            int: Index of the selected LLH.
        """
        max_util = max(self.utilities)
        candidates = [i for i, u in enumerate(self.utilities) if u == max_util]
        return self.random.choice(candidates)

    def _apply_reward(self, u: float) -> float:
        """Additive reward: u ← min(UB, u + reward). (All RL variants share this.)

        Args:
            u (float): Current utility.

        Returns:
            float: New utility.
        """
        """Additive reward: u ← min(UB, u + reward). (All RL variants share this.)"""
        return min(self.params.utility_upper_bound, u + self.params.reward_improvement)

    def _apply_punishment(self, u: float) -> float:
        """
        Applies punishment according to the selected RL variant.
        Maintains raw float utilities to prevent absorbing states.

        RL1 (Section 3.2): u ← max(lb, u − penalty)   [subtractive]
        RL2 (Section 3.2): u ← floor(u / 2)            [divisional]
        RL3 (Section 3.2): u ← floor(sqrt(u))           [root]

        Args:
            u (float): Current utility.

        Returns:
            float: New utility.
        """
        variant = self.params.punishment_type
        lb = self.params.min_utility

        if variant == "RL2":
            return max(lb, u / 2.0)
        if variant == "RL3":
            # Handle the mathematical anomaly where sqrt(u) > u if u < 1.0
            return max(lb, math.sqrt(u) if u >= 1.0 else u / 2.0)

        # Default: RL1 (subtractive)
        return max(lb, u - self.params.penalty_worsening)

    # ------------------------------------------------------------------
    # Low-Level Heuristics (LLH Pool)
    # ------------------------------------------------------------------

    def _llh_relocate(self, routes: List[List[int]]) -> List[List[int]]:
        """H1: Single-node relocation (1-0 exchange) — ruin-and-recreate variant.

        Args:
            routes (List[List[int]]): Current routes.

        Returns:
            List[List[int]]: New routes.
        """
        if not routes:
            return copy.deepcopy(routes)
        new_routes = copy.deepcopy(routes)
        ridx = self.random.randint(0, len(new_routes) - 1)
        if not new_routes[ridx]:
            return new_routes

        n_idx = self.random.randint(0, len(new_routes[ridx]) - 1)
        node = new_routes[ridx].pop(n_idx)

        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        if use_profit:
            new_routes = greedy_profit_insertion(
                new_routes,
                [node],
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )
        else:
            new_routes = greedy_insertion(
                new_routes,
                [node],
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )
        return new_routes

    def _llh_shaw(self, routes: List[List[int]]) -> List[List[int]]:
        """H2: Shaw removal + regret-2 insertion.

        Args:
            routes (List[List[int]]): Current routes.

        Returns:
            List[List[int]]: New routes.
        """
        routes = copy.deepcopy(routes)
        n = max(1, min(len(self.nodes) // 4, 10))
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        if use_profit:
            partial, removed = shaw_profit_removal(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )
        else:
            partial, removed = shaw_removal(routes, n, self.dist_matrix)
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )

    def _llh_string(self, routes: List[List[int]]) -> List[List[int]]:
        """H3: String removal + greedy insertion.

        Args:
            routes (List[List[int]]): Current routes.

        Returns:
            List[List[int]]: New routes.
        """
        routes = copy.deepcopy(routes)
        n = max(1, min(len(self.nodes) // 4, 10))
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        partial, removed = string_removal(routes, n, self.dist_matrix, rng=self.random)

        if use_profit:
            return greedy_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )
        else:
            return greedy_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )

    def _llh_regret2(self, routes: List[List[int]]) -> List[List[int]]:
        """H4: Random removal + regret-2 insertion.

        Args:
            routes (List[List[int]]): Current routes.

        Returns:
            List[List[int]]: New routes.
        """
        routes = copy.deepcopy(routes)
        n = max(1, min(len(self.nodes) // 5, 5))
        use_profit = self.params.profit_aware_operators
        expand_pool = self.params.vrpp

        partial, removed = random_removal(routes, n, rng=self.random)
        if use_profit:
            return regret_2_profit_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                self.R,
                self.C,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )
        else:
            return regret_2_insertion(
                partial,
                removed,
                self.dist_matrix,
                self.wastes,
                self.capacity,
                mandatory_nodes=self.mandatory_nodes,
                expand_pool=expand_pool,
            )

    # ------------------------------------------------------------------
    # Functional Helpers
    # ------------------------------------------------------------------

    def _initialize_solution(self) -> List[List[int]]:
        """Randomized greedy construction of a starting feasible routing.

        Returns:
            List[List[int]]: Initial feasible routes.
        """
        nodes = copy.copy(self.nodes)
        self.random.shuffle(nodes)
        try:
            return build_grasp_routes(
                dist_matrix=self.dist_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                R=self.R,
                C=self.C,
                mandatory_nodes=self.mandatory_nodes,
                alpha=0.5,  # Moderate randomness to ensure a chaotic but feasible start
                rng=self.random,
            )
        except Exception:
            # Fallback to single-node routes if GRASP fails
            nodes = copy.copy(self.nodes)
            return [[n] for n in nodes]

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Calculates fitness F(g) as Net Profit ($).

        Args:
            routes (List[List[int]]): Routes.

        Returns:
            float: Net profit.
        """
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Calculates distance-based travel cost (km).

        Args:
            routes (List[List[int]]): Routes.

        Returns:
            float: Travel cost.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for i in range(len(route) - 1):
                total += self.dist_matrix[route[i]][route[i + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total

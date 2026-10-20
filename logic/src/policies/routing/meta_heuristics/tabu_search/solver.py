"""Tabu Search (TS) for VRPP.

Attributes:
    TSSolver: Main solver class for Tabu Search.

Example:
    >>> solver = TSSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()
"""

import copy
import random
import time
from collections import defaultdict, deque
from typing import Any, DefaultDict, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from logic.src.configs.policies.ts import TSConfig
from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory
from logic.src.policies.helpers.operators.solution_initialization.greedy_si import build_greedy_routes
from logic.src.policies.routing.meta_heuristics.tabu_search.memory import TSMemoryMixin
from logic.src.policies.routing.meta_heuristics.tabu_search.moves import TSMoveGenerationMixin


class TSSolver(TSMemoryMixin, TSMoveGenerationMixin):
    """Tabu Search solver for VRPP with comprehensive memory structures.

    Key Features:
    - Short-term memory: Recency-based tabu list with dynamic tenure
    - Aspiration criteria: Override tabu status for globally improving moves
    - Long-term memory: Frequency-based memory for intensification/diversification
    - Elite solutions: Maintain pool of best solutions for path relinking
    - Strategic oscillation: Explore feasible/infeasible boundaries (optional)
    - Candidate lists: Restrict neighborhood exploration for efficiency


    Attributes:
        tabu_list: Short-term memory for tabu moves.
        move_frequency: Frequency of moves.
        node_frequency: Frequency of node usage.
        elite_solutions: Pool of elite solutions.
        best_routes: Best routes found so far.
        best_profit: Best profit found so far.
        best_solution_time: Time to find the best solution.
        iteration: Current iteration number.
        iterations_since_improvement: Number of iterations without improvement.
        time_limit: Time limit for the search.
        diversification_level: Current diversification level.
        rrng: Random number generator.
        params: Tabu search parameters.
        dist_matrix: Distance matrix.
        wastes: Mapping of bin IDs to waste quantities.
        capacity: Maximum vehicle collection capacity.
        R: Revenue per kg of waste.
        C: Cost per km traveled.
        mandatory_nodes: List of mandatory nodes.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: TSConfig,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """Initializes the Tabu Search solver.

        Args:
            dist_matrix: Symmetric distance matrix.
            wastes: Mapping of bin IDs to waste quantities.
            capacity: Maximum vehicle collection capacity.
            R: Revenue per kg of waste.
            C: Cost per km traveled.
            params: Algorithm-specific parameters.
            mandatory_nodes: Nodes that must be visited.
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

        # Short-term memory: Tabu list stores (move_type, move_attributes, expiration_iter)
        # move_attributes is a tuple representing the move (e.g., (node1, node2) for swap)
        self.tabu_list: Deque[Tuple[str, Any, int]] = deque()

        # Long-term memory: Frequency matrices
        # Track how often nodes appear in solutions and how often moves are performed
        self.node_frequency: DefaultDict[int, int] = defaultdict(int)
        self.move_frequency: DefaultDict[Tuple[str, Any], int] = defaultdict(int)

        # Elite solutions pool for intensification and path relinking
        self.elite_solutions: List[Tuple[List[List[int]], float]] = []

        # Iteration counters
        self.iteration = 0
        self.iterations_since_improvement = 0

        # LLH pool for destroy-repair
        self._llh_pool = [
            self._llh_random_greedy,
            self._llh_worst_regret,
            self._llh_cluster_greedy,
            self._llh_worst_greedy,
            self._llh_random_regret,
        ]

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run Tabu Search with adaptive memory structures.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start_time = time.perf_counter()

        # Initialize solution
        routes = self._build_initial_solution()
        profit = self._evaluate(routes)
        current_routes = copy.deepcopy(routes)
        current_profit = profit
        best_routes = copy.deepcopy(routes)
        best_profit = profit

        # Setup modular acceptance criterion
        self.acceptance = AcceptanceCriterionFactory.build(self.params.acceptance_criterion)
        self.acceptance.setup(profit)

        # Add initial solution to elite pool
        self._update_elite_pool(routes, profit)
        self._update_frequency_memory(routes)

        for iteration in range(self.params.max_iterations):
            self.iteration = iteration
            # Time limit check
            if self.params.time_limit > 0 and time.perf_counter() - start_time > self.params.time_limit:
                break

            # Restart if stuck
            if self.iterations_since_improvement > self.params.max_iterations_no_improve:
                if self.params.diversification_enabled:
                    current_routes = self._diversification_restart()
                    current_profit = self._evaluate(current_routes)
                    self.iterations_since_improvement = 0
                else:
                    break

            # Periodic intensification
            if (
                self.params.intensification_enabled
                and self.iteration > 0
                and self.iteration % self.params.intensification_interval == 0
            ):
                current_routes = self._intensification_phase(best_routes)
                current_profit = self._evaluate(current_routes)

            # Periodic diversification
            if (
                self.params.diversification_enabled
                and self.iteration > 0
                and self.iteration % self.params.diversification_interval == 0
            ):
                current_routes = self._diversification_phase(current_routes)
                current_profit = self._evaluate(current_routes)

            # Generate candidate neighborhood and select best move
            candidates = self._generate_candidates(current_routes)
            best_candidate, best_candidate_profit, best_move_desc = self._select_best_candidate(
                candidates, current_profit, best_profit
            )

            # Move to best candidate
            if best_candidate is not None:
                # Determine if this specific move was accepted (using the same logic as selection)
                # Note: selection already filtered by acceptance, but we need the flag for step()
                is_tabu = self._is_tabu(best_move_desc) if best_move_desc else False
                is_accepted, _ = self.acceptance.accept(
                    current_obj=current_profit,
                    candidate_obj=best_candidate_profit,
                    is_tabu=is_tabu,
                    best_profit=best_profit,
                    iteration=iteration,
                )

                current_routes = best_candidate
                current_profit = best_candidate_profit

                # Update tabu list
                if best_move_desc is not None:
                    self._add_to_tabu_list(best_move_desc)
                    self._update_move_frequency(best_move_desc)

                # Update frequency memory
                self._update_frequency_memory(current_routes)

                # Update best solution
                if current_profit > best_profit:
                    best_routes = copy.deepcopy(current_routes)
                    best_profit = current_profit
                    self.iterations_since_improvement = 0
                    self._update_elite_pool(current_routes, current_profit)
                else:
                    self.iterations_since_improvement += 1

                # Step criterion
                self.acceptance.step(
                    current_obj=current_profit,
                    candidate_obj=best_candidate_profit,
                    accepted=is_accepted,
                    iteration=iteration,
                )

            # Record telemetry
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=self.iteration,
                best_profit=best_profit,
                current_profit=current_profit,
                tabu_size=len(self.tabu_list),
                elite_size=len(self.elite_solutions),
            )

        best_cost = self._cost(best_routes)
        return best_routes, best_profit, best_cost

    # ========================================================================
    # Candidate Selection
    # ========================================================================

    def _select_best_candidate(
        self,
        candidates: Sequence[Tuple[List[List[int]], Tuple[str, Any]]],
        current_profit: float,
        best_profit: float,
    ) -> Tuple[Optional[List[List[int]]], float, Optional[Tuple[str, Any]]]:
        """
        Select best accepted candidate based on modular criterion (non-tabu or aspirated).

        Args:
            candidates: Candidates to evaluate.
            current_profit: Current profit.
            best_profit: Best profit found so far.

        Returns:
            Tuple[Optional[List[List[int]]], float, Optional[Tuple[str, Any]]]: Best candidate, best candidate profit, and best move descriptor.
        """
        best_candidate = None
        best_candidate_profit = float("-inf")
        best_move_desc = None
        least_rejected = None
        least_rejected_profit = float("-inf")
        least_rejected_desc = None

        for candidate_routes, move_desc in candidates:
            candidate_profit = self._evaluate(candidate_routes)
            is_tabu = self._is_tabu(move_desc)

            # Accept if not tabu or if aspiration criteria met
            # Delegate decision to injected criterion
            is_accepted, _ = self.acceptance.accept(
                current_obj=current_profit,
                candidate_obj=candidate_profit,
                is_tabu=is_tabu,
                best_profit=best_profit,
                iteration=self.iteration,
            )

            if not is_accepted:
                if candidate_profit > least_rejected_profit:
                    least_rejected = candidate_routes
                    least_rejected_profit = candidate_profit
                    least_rejected_desc = move_desc
                continue

            # Apply frequency-based penalty for diversification (Objective selection phase)
            penalty = 0.0
            if self.params.diversification_enabled:
                penalty = self._compute_frequency_penalty(move_desc)

            adjusted_profit = candidate_profit - penalty

            if adjusted_profit > best_candidate_profit:
                best_candidate = candidate_routes
                best_candidate_profit = candidate_profit
                best_move_desc = move_desc

        # If all moves are rejected, force the least-rejected one (Tabu search convention).
        # That is the highest-profit neighbour, not whichever candidate the shuffle left first.
        if best_candidate is None and least_rejected is not None:
            best_candidate = least_rejected
            best_candidate_profit = least_rejected_profit
            best_move_desc = least_rejected_desc

        return best_candidate, best_candidate_profit, best_move_desc

    # ========================================================================
    # Tabu List Management (Short-term Memory)
    # ========================================================================

    # ========================================================================
    # Frequency Memory (Long-term Memory)
    # ========================================================================

    # ========================================================================
    # Elite Solutions & Intensification
    # ========================================================================

    # ========================================================================
    # Diversification
    # ========================================================================

    # ========================================================================
    # Neighborhood Generation
    # ========================================================================

    # ========================================================================
    # LLH Pool (Destroy-Repair Heuristics)
    # ========================================================================

    # ========================================================================
    # Helpers
    # ========================================================================

    def _build_initial_solution(self) -> List[List[int]]:
        """
        Build initial solution using greedy profit-aware heuristic.

        Returns:
            List[List[int]]: Initial solution.
        """
        return build_greedy_routes(
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
        Evaluate solution profit (revenue - cost).

        Args:
            routes: Routes to evaluate.

        Returns:
            float: Solution profit.
        """
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """
        Compute total routing cost.

        Args:
            routes: Routes to evaluate.

        Returns:
            float: Total routing cost.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += self.dist_matrix[0][route[0]]
            for k in range(len(route) - 1):
                total += self.dist_matrix[route[k]][route[k + 1]]
            total += self.dist_matrix[route[-1]][0]
        return total


__all__ = ["TSSolver"]

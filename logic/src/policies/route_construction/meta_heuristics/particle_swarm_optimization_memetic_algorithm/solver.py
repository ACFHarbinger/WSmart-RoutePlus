"""
Particle Swarm Optimization Memetic Algorithm (PSOMA) for VRPP.

Particles navigate the discrete routing space via swap-based velocity.
A low inertia weight (ω≈0.4) forces intensive local exploitation.

Attributes:
    PSOMASolver (Type): Core solver class for PSOMA.
    PSOMAParams (Type): Parameter dataclass for the solver.

Example:
    >>> solver = PSOMASolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()

Reference:
    Liu, B., Wang, L., Jin, Y., & Huang, D. (2006).
    "An Effective PSO-Based Memetic Algorithm for TSP"
"""

import copy
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.split import LinearSplit

from .params import PSOMAParams
from .particle import PSOMAParticle


class PSOMASolver:
    """
    PSOMASolver class for Vehicle Routing Problem with Pickup and Delivery (VRPP).
    Implements a Particle Swarm Optimization (PSO) algorithm with a Memetic Algorithm (MA).

    Attributes:
        dist_matrix (np.ndarray): Symmetric distance matrix.
        wastes (Dict[int, float]): Mapping of bin IDs to waste quantities.
        capacity (float): Maximum vehicle collection capacity.
        R (float): Revenue per kg of waste.
        C (float): Cost per km traveled.
        params (PSOMAParams): Algorithm-specific parameters.
        mandatory_nodes (Optional[List[int]]): Nodes that must be visited.
        n_nodes (int): Number of nodes (excluding depot).
        clients (List[int]): List of client IDs.
        random (random.Random): Random number generator.
        split_solver (LinearSplit): Solver for partitioning tours into routes.
        acceptance_criterion (AcceptanceCriterion): Criterion for accepting worse solutions.
        metropolis_steps (int): Number of metropolis steps per iteration.
        operators (List[Callable]): List of mutation operators.
        probabilities (np.ndarray): Probabilities for selecting mutation operators.
        rewards (np.ndarray): Rewards for each mutation operator.
        swarm (List[PSOMAParticle]): The swarm of particles.
        gbest_X (np.ndarray): Global best position vector.
        gbest_giant_tour (np.ndarray): Global best giant tour.
        gbest_routes (List[List[int]]): Global best routes.
        gbest_profit (float): Global best profit.
        gbest_mapping (np.ndarray): Global best mapping.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: PSOMAParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """Initializes the PSOMA solver.

        Args:
            dist_matrix (np.ndarray): Symmetric distance matrix.
            wastes (Dict[int, float]): Mapping of bin IDs to waste quantities.
            capacity (float): Maximum vehicle collection capacity.
            R (float): Revenue per kg of waste.
            C (float): Cost per km traveled.
            params (PSOMAParams): Algorithm-specific parameters.
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
        self.clients = list(range(1, self.n_nodes + 1))
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()
        # One seeded numpy stream for every vector draw, so a run is repeatable outside the simulator's global reseed.
        self.np_rng = np.random.default_rng(params.seed)
        self.split_solver = LinearSplit(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            max_vehicles=0,
            mandatory_nodes=self.mandatory_nodes,
            vrpp=self.params.vrpp,
        )

        # Modular Acceptance Criterion
        assert params.acceptance_criterion is not None, "Acceptance criterion must be provided."
        self.acceptance_criterion = params.acceptance_criterion

        self.metropolis_steps = max(1, self.n_nodes * (self.n_nodes - 1))
        self.operators = [self._swap, self._insert, self._inverse]
        self.probabilities = np.array([1 / 3, 1 / 3, 1 / 3])
        self.rewards = np.zeros(3)

        self.swarm: List[PSOMAParticle] = []
        self.gbest_X = np.zeros(self.n_nodes)
        self.gbest_giant_tour = np.zeros(self.n_nodes, dtype=int)
        self.gbest_routes: List[List[int]] = []
        self.gbest_profit = -float("inf")
        self.gbest_mapping = np.zeros(self.n_nodes, dtype=int)

        # Construct the Profit-Biased Distance Matrix for O(1) Surrogate Evaluation
        self.mu = 10.0  # Tuning parameter: higher values pull profitable nodes closer
        self.biased_dist_matrix = np.copy(self.dist_matrix)

        # Calculate profit-to-cost ratio (rho) for each node
        rho = np.zeros(self.n_nodes + 1)
        for i in self.clients:
            marginal_cost = self.dist_matrix[0, i] + self.dist_matrix[i, 0]
            if marginal_cost > 0:
                rho[i] = (self.wastes[i] * self.R) / (marginal_cost * self.C)

        # Transform the matrix
        for i in range(1, self.n_nodes + 1):
            for j in range(1, self.n_nodes + 1):
                if i != j:
                    self.biased_dist_matrix[i, j] -= self.mu * ((rho[i] + rho[j]) / 2.0)

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """Executes the PSO Memetic Algorithm to find an optimal routing solution.

        The algorithm iterates between a global exploration phase (PSO) and
        a local exploitation phase (SA) for a fixed number of iterations or until a time limit is reached.

        Returns:
            Tuple[List[List[int]], float, float]: A tuple containing:
                - The best routing solution found (list of routes).
                - The total profit of the best solution.
                - The total travel cost (distance) of the best solution.
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start_time = time.perf_counter()
        self._init_swarm()

        self.acceptance_criterion.setup(self.gbest_profit)
        self._training_phase()
        stagnation_counter = 0

        for _iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.perf_counter() - start_time > self.params.time_limit:
                break

            old_gbest_profit = self.gbest_profit

            # PSO Exploration
            for p in self.swarm:
                r1, r2 = self.np_rng.random(self.n_nodes), self.np_rng.random(self.n_nodes)
                p.V = (
                    self.params.omega * p.V
                    + self.params.c1 * r1 * (p.pbest_X - p.X)
                    + self.params.c2 * r2 * (self.gbest_X - p.X)
                )
                p.V = np.clip(p.V, self.params.v_min, self.params.v_max)
                p.X = np.clip(p.X + p.V, self.params.x_min, self.params.x_max)

                p.giant_tour, p.mapping_indices = p._rov_rule(p.X)
                p.routes, p.profit = self.split_solver.split(p.giant_tour.tolist())
                if p.profit > p.pbest_profit:
                    p.pbest_X = np.copy(p.X)
                    p.pbest_giant_tour = np.copy(p.giant_tour)
                    p.pbest_mapping_indices = np.copy(p.mapping_indices)
                    p.pbest_profit = p.profit

                if p.profit > self.gbest_profit:
                    self._set_gbest(p.X, p.giant_tour, p.mapping_indices, p.routes, p.profit)

            # SA Local Search
            self._non_training_phase()
            self.acceptance_criterion.step(
                current_obj=self.gbest_profit, candidate_obj=self.gbest_profit, accepted=True
            )

            if self.gbest_profit <= old_gbest_profit:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            if stagnation_counter >= self.params.L:
                break

        final_cost = self._calculate_cost(self.gbest_routes)
        return self.gbest_routes, self.gbest_profit, final_cost

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_swarm(self):
        """Initialises the swarm with random feasible solutions.

        Returns:
            List[PSOMAParticle]: Initialized swarm of particles.
        """
        self.swarm = [PSOMAParticle(self.clients, self.params, self.split_solver, rng=self.np_rng) for _ in range(self.params.pop_size)]
        for p in self.swarm:
            if p.profit > self.gbest_profit:
                self._set_gbest(p.X, p.giant_tour, p.mapping_indices, p.routes, p.profit)

    def _set_gbest(
        self,
        X: np.ndarray,
        giant_tour: np.ndarray,
        mapping: np.ndarray,
        routes: List[List[int]],
        profit: float,
    ):
        """Updates the global best solution.

        Args:
            X (np.ndarray): Global best position.
            giant_tour (np.ndarray): Global best tour.
            mapping (np.ndarray): Global best mapping.
            routes (List[List[int]]): Global best routes.
            profit (float): Global best profit.
        """
        self.gbest_X = np.copy(X)
        self.gbest_giant_tour = np.copy(giant_tour)
        self.gbest_mapping = np.copy(mapping)
        self.gbest_routes = copy.deepcopy(routes)
        self.gbest_profit = profit

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculates the total travel cost for a given set of routes.

        Args:
            routes (List[List[int]]): The routing solution to evaluate.

        Returns:
            float: The total travel cost.
        """
        cost = 0.0
        for r in routes:
            if not r:
                continue
            cost += self.dist_matrix[0, r[0]]
            for k in range(len(r) - 1):
                cost += self.dist_matrix[r[k], r[k + 1]]
            cost += self.dist_matrix[r[-1], 0]
        return cost * self.C

    def _training_phase(self) -> None:
        """Executes the training phase of the algorithm.

        In this phase, each operator is applied a fixed number of times to evaluate their performance
        based on the improvement they bring to the current best solution. The rewards are updated
        to reflect the effectiveness of each operator.

        Returns:
            None
        """
        for i, op in enumerate(self.operators):
            best_pf, _, _, _ = self._sa_search(op)
            # Only improvements earn reward, as in the main loop (abs() rewarded worsening).
            self.rewards[i] = max(0.0, best_pf - self.gbest_profit) / self.metropolis_steps
        self._update_probabilities()

    def _non_training_phase(self) -> None:
        """Executes the non-training phase of the algorithm.

        In this phase, an operator is selected based on the current reward probabilities,
        applied to the global best solution, and the rewards are updated based on the
        improvement achieved.

        Returns:
            None
        """
        op_idx = int(self.np_rng.choice(3, p=self.probabilities))
        op = self.operators[op_idx]
        initial_profit = self.gbest_profit

        best_pf, _, _, _ = self._sa_search(op)

        # Calculate improvement rate
        delta_eta = max(0.0, best_pf - initial_profit) / self.metropolis_steps

        # EMA Update: gamma = 0.5 for balanced recency weighting
        gamma = 0.5
        self.rewards[op_idx] = (1 - gamma) * self.rewards[op_idx] + gamma * delta_eta

        self._update_probabilities()

    def _sa_search(self, operator: Callable) -> Tuple[float, np.ndarray, np.ndarray, List[List[int]]]:
        """Performs simulated annealing search starting from the global best solution.

        Args:
            operator (Callable): The operator to apply to the current solution.

        Returns:
            Tuple[float, np.ndarray, np.ndarray, List[List[int]]]: A tuple containing:
                - The best profit found during the search.
                - The best position vector.
                - The best tour.
                - The best routes.
        """
        current_tour, current_X = np.copy(self.gbest_giant_tour), np.copy(self.gbest_X)
        current_mapping = np.copy(self.gbest_mapping)
        current_pf = self.gbest_profit

        best_tour, best_X, best_pf = np.copy(current_tour), np.copy(current_X), current_pf
        best_routes = copy.deepcopy(self.gbest_routes)
        best_mapping = np.copy(current_mapping)

        for _ in range(self.metropolis_steps):
            # 1. Generate neighborhood move and O(1) Surrogate Delta simultaneously
            new_tour, new_X, new_mapping, delta_dist = operator(current_tour, current_X, current_mapping)

            # 2. SA Acceptance on Surrogate (Minimizing Distance)
            if delta_dist > 0:
                surrogate_accepted, _ = self.acceptance_criterion.accept(current_obj=0.0, candidate_obj=-delta_dist)
                if not surrogate_accepted:
                    continue  # Reject early! Bypass the O(N^2) Split!

            # 3. True Evaluation (Lazy Decoding)
            new_routes, new_pf = self.split_solver.split(new_tour.tolist())

            # 4. Final SA Acceptance on true Profit (Maximization)
            accepted, _ = self.acceptance_criterion.accept(current_obj=current_pf, candidate_obj=new_pf)
            if accepted:
                current_tour, current_X, current_pf = new_tour, new_X, new_pf
                current_mapping = new_mapping

                if current_pf > best_pf:
                    best_tour, best_X, best_pf = np.copy(current_tour), np.copy(current_X), current_pf
                    best_routes, best_mapping = copy.deepcopy(new_routes), np.copy(new_mapping)

                    if best_pf > self.gbest_profit:
                        self._set_gbest(best_X, best_tour, best_mapping, best_routes, best_pf)

        return best_pf, best_X, best_tour, best_routes

    def _update_probabilities(self) -> None:
        """Updates the probabilities for operator selection based on rewards.

        Returns:
            None
        """
        epsilon = 0.05  # 5% minimum probability for any operator

        # Apply epsilon bound to raw rewards to ensure exploration
        bounded_rewards = np.maximum(epsilon, self.rewards)
        total_reward = np.sum(bounded_rewards)

        self.probabilities = bounded_rewards / total_reward

    def _swap(
        self,
        tour: np.ndarray,
        X: np.ndarray,
        mapping: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Performs swap mutation on the tour and position vector.

        Args:
            tour (np.ndarray): The tour to perform swap mutation on.
            X (np.ndarray): The position vector to perform swap mutation on.
            mapping (np.ndarray): The mapping vector to perform swap mutation on.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]: A tuple containing:
                - The tour after swap mutation.
                - The position vector after swap mutation.
                - The mapping vector after swap mutation.
                - The delta of the objective function.
        """
        if self.n_nodes < 2:
            return tour, X, mapping, 0.0

        i, j = sorted(self.random.sample(range(self.n_nodes), 2))
        N = self.n_nodes
        D = self.biased_dist_matrix

        # Calculate O(1) Delta BEFORE modifying the arrays
        p_i, v_i, s_i = tour[(i - 1) % N], tour[i], tour[(i + 1) % N]
        p_j, v_j, s_j = tour[(j - 1) % N], tour[j], tour[(j + 1) % N]

        if j == i + 1 or (i == 0 and j == N - 1):  # Adjacent swap
            if i == 0 and j == N - 1:
                # If circularly adjacent, j is conceptually "before" i
                delta = D[p_j, v_i] + D[v_j, s_i] - D[p_j, v_j] - D[v_i, s_i]
            else:
                delta = D[p_i, v_j] + D[v_i, s_j] - D[p_i, v_i] - D[v_j, s_j]
        else:  # Disjoint swap
            delta = (
                D[p_i, v_j]
                + D[v_j, s_i]
                + D[p_j, v_i]
                + D[v_i, s_j]
                - D[p_i, v_i]
                - D[v_i, s_i]
                - D[p_j, v_j]
                - D[v_j, s_j]
            )

        # Discrete manipulations
        new_tour, new_mapping = np.copy(tour), np.copy(mapping)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        new_mapping[i], new_mapping[j] = new_mapping[j], new_mapping[i]

        new_X = np.empty_like(X)
        new_X[new_mapping] = np.sort(X)

        return new_tour, new_X, new_mapping, delta

    def _insert(
        self,
        tour: np.ndarray,
        X: np.ndarray,
        mapping: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Performs insert mutation on the tour and position vector.

        Args:
            tour (np.ndarray): The tour to perform insert mutation on.
            X (np.ndarray): The position vector to perform insert mutation on.
            mapping (np.ndarray): The mapping vector to perform insert mutation on.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]: A tuple containing:
                - The tour after insert mutation.
                - The position vector after insert mutation.
                - The mapping vector after insert mutation.
                - The delta of the objective function.
        """
        if self.n_nodes < 2:
            return tour, X, mapping, 0.0

        # Choose two distinct positions; i < j implies moving a node 'backwards'
        i, j = sorted(self.random.sample(range(self.n_nodes), 2))
        N = self.n_nodes
        D = self.biased_dist_matrix

        # Identify nodes involved in the move
        v = tour[j]  # The node being moved
        p_j, s_j = tour[(j - 1) % N], tour[(j + 1) % N]
        p_i, v_i = tour[(i - 1) % N], tour[i]

        # --- O(1) Surrogate Delta Calculation ---
        # Case 1: Circular Adjacency (e.g., moving last node to the front)
        # In a circular tour, this move results in zero change to the edge set.
        if i == 0 and j == N - 1:
            delta = 0.0

        # Case 2: Standard Adjacency (j is the immediate successor of i)
        # This effectively swaps the positions of v_i and v.
        elif j == i + 1:
            delta = D[p_i, v] + D[v, v_i] + D[v_i, s_j] - D[p_i, v_i] - D[v_i, v] - D[v, s_j]

        # Case 3: Disjoint Insertion
        # We break 3 edges (around i and j) and form 3 new ones.
        else:
            delta = D[p_j, s_j] + D[p_i, v] + D[v, v_i] - D[p_j, v] - D[v, s_j] - D[p_i, v_i]

        # --- Physical Mutation ---
        val_tour, val_map = tour[j], mapping[j]
        # Remove from old position and insert at new position i
        new_tour = np.insert(np.delete(tour, j), i, val_tour)
        new_mapping = np.insert(np.delete(mapping, j), i, val_map)

        # --- ROV Repair Logic ---
        # Guarantees the continuous space X matches the new discrete permutation[cite: 15, 46].
        new_X = np.empty_like(X)
        new_X[new_mapping] = np.sort(X)

        return new_tour, new_X, new_mapping, delta

    def _inverse(
        self,
        tour: np.ndarray,
        X: np.ndarray,
        mapping: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Performs inverse mutation on the tour and position vector.

        Args:
            tour (np.ndarray): The tour to perform inverse mutation on.
            X (np.ndarray): The position vector to perform inverse mutation on.
            mapping (np.ndarray): The mapping vector to perform inverse mutation on.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]: A tuple containing:
                - The tour after inverse mutation.
                - The position vector after inverse mutation.
                - The mapping vector after inverse mutation.
                - The delta of the objective function.
        """
        if self.n_nodes < 2:
            return tour, X, mapping, 0.0

        i, j = sorted(self.random.sample(range(self.n_nodes), 2))
        N = self.n_nodes
        D = self.biased_dist_matrix

        # Calculate O(1) Delta
        p_i = tour[(i - 1) % N]
        s_j = tour[(j + 1) % N]

        delta = D[p_i, tour[j]] + D[tour[i], s_j] - D[p_i, tour[i]] - D[tour[j], s_j]

        new_tour, new_mapping = np.copy(tour), np.copy(mapping)
        new_tour[i : j + 1] = new_tour[i : j + 1][::-1]
        new_mapping[i : j + 1] = new_mapping[i : j + 1][::-1]

        new_X = np.empty_like(X)
        new_X[new_mapping] = np.sort(X)

        return new_tour, new_X, new_mapping, delta

"""Swarm initialisation, cost evaluation and SA local search of the PSOMA solver.

:class:`PSOMASwarmSearchMixin` holds the private helpers of
:class:`~logic.src.policies.routing.meta_heuristics.particle_swarm_optimization_memetic_algorithm.solver.PSOMASolver`:
swarm initialisation and particle seeding (nearest-neighbour order), global-best tracking,
cost evaluation, the training and non-training phases, the simulated-annealing search and
its swap, insert and inverse neighbourhood operators. Moved unchanged from ``solver.py``.
"""

import copy
from typing import Callable, List, Tuple

import numpy as np

from .particle import PSOMAParticle


class PSOMASwarmSearchMixin:
    """Swarm, evaluation and SA local-search helpers of :class:`PSOMASolver` (moved unchanged from ``solver.py``)."""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_swarm(self):
        """Initialises the swarm with random feasible solutions.

        Returns:
            List[PSOMAParticle]: Initialized swarm of particles.
        """
        self.swarm = [
            PSOMAParticle(self.clients, self.params, self.split_solver, rng=self.np_rng)
            for _ in range(self.params.pop_size)
        ]
        if self.params.seed_nearest_neighbour and self.swarm:
            self._seed_particle(self.swarm[0])
        for p in self.swarm:
            if p.profit > self.gbest_profit:
                self._set_gbest(p.X, p.giant_tour, p.mapping_indices, p.routes, p.profit)

    def _nearest_neighbour_order(self) -> List[int]:
        """Client order of a nearest-neighbour tour from the depot (true distances)."""
        unvisited = set(self.clients)
        order, current = [], 0
        while unvisited:
            nxt = min(unvisited, key=lambda j: self.dist_matrix[current, j])
            order.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return order

    def _seed_particle(self, p) -> None:
        """Place one particle on the nearest-neighbour giant tour (ROV keys ranked along the tour)."""
        keys = np.empty(self.n_nodes)
        ranks = np.linspace(self.params.x_min, self.params.x_max, self.n_nodes)
        for rank, node in enumerate(self._nearest_neighbour_order()):
            keys[node - 1] = ranks[rank]
        p.X = keys
        p.giant_tour, p.mapping_indices = p._rov_rule(p.X)
        p.routes, p.profit = self.split_solver.split(p.giant_tour.tolist())
        p.pbest_X = np.copy(p.X)
        p.pbest_giant_tour = np.copy(p.giant_tour)
        p.pbest_mapping_indices = np.copy(p.mapping_indices)
        p.pbest_profit = p.profit

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
            X: Global best position.
            giant_tour: Global best tour.
            mapping: Global best mapping.
            routes: Global best routes.
            profit: Global best profit.
        """
        self.gbest_X = np.copy(X)
        self.gbest_giant_tour = np.copy(giant_tour)
        self.gbest_mapping = np.copy(mapping)
        self.gbest_routes = copy.deepcopy(routes)
        self.gbest_profit = profit
        # Mirror onto the template-level incumbent so PopulationSearch can track stagnation.
        self.best_routes = self.gbest_routes
        self.best_profit = profit

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculates the total travel cost for a given set of routes.

        Args:
            routes: The routing solution to evaluate.

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

        """
        for i, op in enumerate(self.operators):
            if self._out_of_time():
                break
            best_pf, _, _, _ = self._sa_search(op)
            # Only improvements earn reward, as in the main loop (abs() rewarded worsening).
            self.rewards[i] = max(0.0, best_pf - self.gbest_profit) / self.metropolis_steps
        self._update_probabilities()

    def _non_training_phase(self) -> None:
        """Executes the non-training phase of the algorithm.

        In this phase, an operator is selected based on the current reward probabilities,
        applied to the global best solution, and the rewards are updated based on the
        improvement achieved.

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
            operator: The operator to apply to the current solution.

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
            if self._out_of_time():
                break
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

    def _swap(
        self,
        tour: np.ndarray,
        X: np.ndarray,
        mapping: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Performs swap mutation on the tour and position vector.

        Args:
            tour: The tour to perform swap mutation on.
            X: The position vector to perform swap mutation on.
            mapping: The mapping vector to perform swap mutation on.

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
            tour: The tour to perform insert mutation on.
            X: The position vector to perform insert mutation on.
            mapping: The mapping vector to perform insert mutation on.

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
            tour: The tour to perform inverse mutation on.
            X: The position vector to perform inverse mutation on.
            mapping: The mapping vector to perform inverse mutation on.

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

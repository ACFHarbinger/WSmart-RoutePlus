r"""K-Sparse ACO Solver Module.

This module implements the main loop of the K-Sparse Ant Colony Optimization
algorithm using MMAS_exp methodology. It manages the ant colony, global
pheromone updates, and coordinates the search process.

Attributes:
    KSparseACOSolver: K-Sparse MAX-MIN Ant System (MMAS_exp) solver for CVRP/VRPP.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.ant_colony_optimization_k_sparse.solver import KSparseACOSolver
    >>> solver = KSparseACOSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> result = solver.solve()

Reference:
    Hale, D. "Investigation of Ant Colony Optimization Implementation
    Strategies For Low-Memory Operating Environments", 2021.
    Implements MMAS_exp with scale-based sparse pheromone matrix and
    global-only pheromone updates.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.ckdtree import cKDTree

from logic.src.configs.policies.aco_ks import KSparseACOConfig
from logic.src.policies.helpers.local_search.local_search_aco import ACOLocalSearch

from .construction import SolutionConstructor
from .pheromones import SparsePheromoneTau


class KSparseACOSolver:
    """K-Sparse MAX-MIN Ant System (MMAS_exp) solver for CVRP/VRPP.

    Implements MMAS with scale-based sparse pheromone storage for memory
    efficiency and fast computation on large problem instances. Follows
    the experimental MMAS variant (MMAS_exp) from Hale (2021), featuring:
    - Global-only pheromone updates (no local updates during construction)
    - Pure roulette-wheel selection (no q0 exploitation)
    - Dynamic default_value with precision-based pruning

    Attributes:
        node_coords: Node coordinates.
        wastes: Mapping of bin IDs to waste quantities.
        capacity: Maximum vehicle collection capacity.
        R: Revenue multiplier for waste collected.
        C: Cost multiplier for distance traveled.
        params: Algorithm-specific hyperparameters.
        mandatory_nodes: Nodes that must be visited.
        n_nodes: Total number of nodes.
        nodes: List of customer nodes.
        tau_0: Initial pheromone level.
        pheromone: Sparse pheromone matrix.
        ls: Local search optimizer.
        candidate_lists: K-sparse candidate neighbor lists.
        constructor: Solution constructor for ants.
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: KSparseACOConfig,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """Initializes the K-Sparse Ant Colony Optimization solver.

        Args:
            node_coords: Node coordinates.
            wastes: Mapping of bin IDs to waste quantities.
            capacity: Maximum vehicle collection capacity.
            R: Revenue multiplier for waste collected.
            C: Cost multiplier for distance traveled.
            params: Algorithm-specific hyperparameters.
            mandatory_nodes: Nodes that must be visited.
        """
        self.node_coords = np.array(node_coords)
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes

        self.n_nodes = len(self.node_coords)
        self.nodes = list(range(1, self.n_nodes))

        # Build spatial index for O(N log N) nearest neighbor queries
        self.tree = cKDTree(self.node_coords)

        # Build candidate lists efficiently without an N x N matrix
        self.candidate_lists = self._build_candidate_lists()

        if params.tau_0 is None:
            nn_cost = self._nearest_neighbor_cost()
            self.tau_0 = params.rho / nn_cost if nn_cost > 0 else params.tau_max
        else:
            self.tau_0 = params.tau_0

        self.pheromone = SparsePheromoneTau(
            n_nodes=self.n_nodes,
            tau_0=self.tau_0,
            scale=params.scale,
            tau_min=params.tau_min,
            tau_max=params.tau_max,
        )

        self.ls = ACOLocalSearch(
            self.node_coords,
            wastes,
            capacity,
            R,
            C,
            params,
            neighbors=self.candidate_lists,
        )

        self.constructor = SolutionConstructor(
            node_coords=self.node_coords,
            wastes=wastes,
            capacity=capacity,
            pheromone=self.pheromone,
            candidate_lists=self.candidate_lists,
            nodes=self.nodes,
            params=params,
            tau_0=self.tau_0,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
        )

    def _build_candidate_lists(self) -> Dict[int, List[int]]:
        """Build k-nearest neighbor candidate lists in O(N log N) time.

        Returns:
            Dict[int, List[int]]: Dictionary mapping node index to list of nearest neighbors.
        """
        candidates: Dict[int, List[int]] = {}

        # Query k+1 to account for the node finding itself
        k = min(self.params.k_sparse + 1, self.n_nodes)
        _, indices = self.tree.query(self.node_coords, k=k)

        for i in range(self.n_nodes):
            # Filter out the self-reference
            cands = [j for j in indices[i] if j != i][: self.params.k_sparse]

            # THE CVRP RELAXATION: Guarantee depot access.
            # If we are not the depot, and the depot isn't in our k-nearest neighbors,
            # we sacrifice the furthest neighbor to insert the depot.
            if i != 0 and 0 not in cands:
                if len(cands) == self.params.k_sparse:
                    cands[-1] = 0
                else:
                    cands.append(0)

            candidates[i] = cands
        return candidates

    def _nearest_neighbor_cost(self) -> float:
        """Compute baseline cost using a VRP-aware greedy heuristic.
        Respects vehicle capacity and returns to the depot, providing a mathematically
        sound baseline for tau_0 in a multi-route environment.

        Returns:
            float: Cost of the nearest neighbor tour.
        """
        # A bin heavier than the capacity can never be served; keeping it would loop forever.
        unvisited = {n for n in self.nodes if self.wastes.get(n, 0) <= self.capacity}
        log_n_limit = max(1, int(np.log2(self.n_nodes)))
        total_cost = 0.0
        while unvisited:
            current = 0
            load = 0.0
            while unvisited:
                best_next = None
                best_dist = float("inf")

                # Part (a): k-sparse candidates
                for cand in self.candidate_lists.get(current, []):
                    if cand in unvisited and load + self.wastes.get(cand, 0) <= self.capacity:
                        d = float(np.linalg.norm(self.node_coords[current] - self.node_coords[cand]))
                        if d < best_dist:
                            best_next = cand
                            best_dist = d

                # Part (b): Strict O(N log N) fallback
                if best_next is None and unvisited:
                    valid_unvisited = [n for n in unvisited if load + self.wastes.get(n, 0) <= self.capacity]
                    if valid_unvisited:
                        sample_size = min(log_n_limit, len(valid_unvisited))
                        sampled = self.constructor.random.sample(valid_unvisited, sample_size)
                        for cand in sampled:
                            d = float(np.linalg.norm(self.node_coords[current] - self.node_coords[cand]))
                            if d < best_dist:
                                best_next = cand
                                best_dist = d

                if best_next is not None:
                    total_cost += best_dist
                    unvisited.remove(best_next)
                    load += self.wastes.get(best_next, 0)
                    current = best_next
                else:
                    # Vehicle is full or no valid moves, return to depot
                    break

            # Add cost to return to depot
            total_cost += float(np.linalg.norm(self.node_coords[current] - self.node_coords[0]))

        return total_cost

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the K-Sparse ACO algorithm.

        Returns:
            Tuple[List[List[int]], float, float]: (best_routes, profit, cost)
        """
        best_routes: List[List[int]] = []
        best_cost = float("inf")
        start_time = time.perf_counter()
        for _iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.perf_counter() - start_time > self.params.time_limit:
                break

            iteration_best_routes: List[List[int]] = []
            iteration_best_cost = float("inf")

            # Each ant constructs a solution
            iteration_solutions = []
            for _ in range(self.params.n_ants):
                # Use delegated constructor
                routes = self.constructor.construct()

                # Optional local search
                if self.params.local_search:
                    routes = self.ls.optimize(routes)

                cost = self._calculate_cost(routes)
                iteration_solutions.append((routes, cost))

                if cost < iteration_best_cost:
                    iteration_best_cost = cost
                    iteration_best_routes = routes

            # Sort iteration solutions by cost
            iteration_solutions.sort(key=lambda x: x[1])

            # Update global best and dynamic MMAS bounds
            if iteration_best_cost < best_cost:
                best_cost = iteration_best_cost
                best_routes = iteration_best_routes

                # Dynamic MMAS Bound Update (Stützle & Hoos, 2000)
                # We update tau_max based on the new best cost, and scale tau_min proportionately.
                # A zero-cost best (no bin worth its round trip) gives no finite bound.
                if best_cost > 0:
                    new_tau_max = 1.0 / (self.params.rho * best_cost)
                    self.pheromone.tau_max = new_tau_max

                    # tau_min is typically set to tau_max / a, where a depends on the problem size.
                    # Hale uses a fixed tau_min, but scaling it is theoretically safer.
                    # For now, we enforce that the default_value respects the new bounds.
                    self.pheromone.default_value = min(self.pheromone.default_value, new_tau_max)

            # Global pheromone update: MMAS (evaporate all, then reinforce best)
            self._global_pheromone_update(best_routes, best_cost)

            _tau_vals = [v for nbrs in self.pheromone._pheromone.values() for v in nbrs.values()]
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=_iteration,
                best_cost=best_cost,
                iter_best_cost=iteration_best_cost,
                tau_mean=float(sum(_tau_vals) / len(_tau_vals)) if _tau_vals else self.pheromone.default_value,
                tau_max=float(max(_tau_vals)) if _tau_vals else self.pheromone.default_value,
            )

        # Calculate profit
        collected_revenue = sum(self.wastes.get(node, 0) * self.R for route in best_routes for node in route)
        profit = collected_revenue - best_cost * self.C

        return best_routes, profit, best_cost

    def _global_pheromone_update(self, best_routes: List[List[int]], best_cost: float) -> None:
        """
        Apply MMAS global pheromone update.

        This implements the standard MMAS update rule:
        1. Evaporate ALL pheromones globally (including default_value)
        2. Reinforce edges in the best-so-far solution
        3. Precision-based pruning automatically occurs during evaporation

        The update formula for edges in the best solution is:
            tau(i,j) = evaporated_tau(i,j) + delta_bs
        where delta_bs = 1 / best_cost

        Args:
            best_routes: Best solution found so far.
            best_cost: Cost of the best solution.

        Returns:
            None.

        Reference:
            Hale (2021), Section 4.2.2: MMAS global update with evaporate-then-reinforce.
        """
        if not best_routes or best_cost <= 0:
            return

        # Step 1: Reinforce edges in best solution FIRST
        # Depositing before evaporation ensures the new edges are immediately
        # subjected to the precision check relative to the shifting default value.
        delta_bs = 1.0 / best_cost

        for route in best_routes:
            if not route:
                continue

            # Depot to first node
            prev = 0
            for node in route:
                self.pheromone.deposit_edge(prev, node, delta_bs)
                prev = node

            # Last node back to depot
            self.pheromone.deposit_edge(prev, 0, delta_bs)

        # Steps 2 & 3: Global evaporation, bounding, and precision-based pruning
        self.pheromone.evaporate_all(self.params.rho)

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """Calculate total routing cost.

        Args:
            routes: List of routes to evaluate.

        Returns:
            Total cost of the routes.
        """
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += np.linalg.norm(self.node_coords[0] - self.node_coords[route[0]])
            for k in range(len(route) - 1):
                total += np.linalg.norm(self.node_coords[route[k]] - self.node_coords[route[k + 1]])
            total += np.linalg.norm(self.node_coords[route[-1]] - self.node_coords[0])
        return float(total)

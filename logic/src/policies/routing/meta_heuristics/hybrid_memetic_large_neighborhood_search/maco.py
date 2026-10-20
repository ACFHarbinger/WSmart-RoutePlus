r"""Memetic Ant Colony Optimization (MACO) Solver Module.

This module implements a fully-correct Memetic MAX-MIN Ant System (MMAS) for
CVRP and VRPP.  It follows the canonical MMAS formulation of Stützle & Hoos
(2000) augmented with the sparse-pheromone MMAS_exp representation from Hale
(2021) and a memetic elite-pool layer.

Canonical MMAS mechanics implemented here
-----------------------------------------
* **Dynamic pheromone bounds** – ``tau_max`` and ``tau_min`` are recomputed
  whenever a new global best is found (§4 of Stützle & Hoos 2000).
* **Update schedule** – three modes selectable via ``MACOParams.update_schedule``:

    - ``"iteration_best"``  – reinforce with the best solution of the current
      iteration (good exploration, used early).
    - ``"best_so_far"``     – reinforce with the global best (good
      intensification, used late).
    - ``"auto"``            – switches from iteration-best to best-so-far after
      ``ib_phase_length`` iterations; this is the canonical MMAS schedule.

* **Elitist blending** – when ``elitist_weight < 1``, both the iteration-best
  and the best-so-far solutions deposit pheromone in proportion
  ``(w, 1−w)`` (elitist-ant MMAS variant).
* **Stagnation detection & restart** – if no improvement is seen for
  ``stagnation_limit`` iterations, all pheromones are reset to ``tau_max``
  and optionally re-reinforced with the best-so-far solution.

Memetic layer
-------------
An *elite pool* of the ``elite_pool_size`` best solutions seen so far is
maintained.  After local search, solutions that improve upon the worst pool
member replace it.  The pool is used to drive the pheromone update: the best
pool member always acts as the best-so-far solution, providing a richer signal
than a single trajectory.

Attributes:
    MemeticACOSolver: Memetic MAX-MIN Ant System solver for CVRP/VRPP.

Example:
    >>> solver = MemeticACOSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()

References:
    Stützle, T. & Hoos, H. H. (2000). MAX-MIN Ant System.
        Future Generation Computer Systems, 16(8), 889–914.
    Hale, D. (2021). Investigation of Ant Colony Optimization Implementation
        Strategies For Low-Memory Operating Environments.

The global pheromone update, deposit, restart and elite-pool maintenance
methods live in ``pheromone_update.py`` as the ``MACOPheromoneUpdateMixin``,
mixed into ``MemeticACOSolver`` here.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.local_search.local_search_general import GeneralLocalSearch

from .construction import SolutionConstructor
from .params import MACOParams
from .pheromone_update import MACOPheromoneUpdateMixin
from .pheromones import SparsePheromoneTau


class MemeticACOSolver(MACOPheromoneUpdateMixin):
    """Memetic MAX-MIN Ant System (MMAS) solver for CVRP/VRPP.

    The solver integrates four cooperating mechanisms:

    1. **Colony construction** – ants build solutions via proportional
       roulette-wheel selection biased by pheromone and heuristic information.
    2. **Local search** – each ant's solution is optionally improved before
       cost evaluation (controlled by ``params.local_search``).
    3. **Pheromone management** – MMAS-compliant global evaporation, dynamic
       bounds recomputation, and configurable reinforcement schedule.
    4. **Elite pool** – the best ``elite_pool_size`` solutions are tracked and
       used to seed the pheromone update, providing a memetic signal across
       iterations.

    Attributes:
        dist_matrix: NxN distance matrix.
        wastes: Mapping of node IDs to waste quantities.
        capacity: Maximum vehicle collection capacity.
        R: Revenue multiplier for waste collected.
        C: Cost multiplier for distance travelled.
        params: Algorithm-specific hyperparameters (``MACOParams``).
        mandatory_nodes: Nodes that must be visited.
        n_nodes: Total number of nodes (including depot).
        nodes: List of customer node indices (depot excluded).
        eta: Heuristic visibility matrix (inverse distances).
        tau_0: Actual initial pheromone level used.
        pheromone: Sparse pheromone matrix.
        ls: Local search optimizer.
        candidate_lists: K-sparse candidate neighbour lists.
        constructor: Solution constructor for ants.
        elite_pool: Sorted list of (cost, routes) for the best solutions seen.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: MACOParams,
        mandatory_nodes: Optional[List[int]] = None,
    ) -> None:
        """Initialise the Memetic ACO solver.

        Args:
            dist_matrix: NxN distance matrix.
            wastes: Mapping of node IDs to waste quantities.
            capacity: Maximum vehicle collection capacity.
            R: Revenue multiplier for waste collected.
            C: Cost multiplier for distance travelled.
            params: Algorithm-specific hyperparameters.
            mandatory_nodes: Nodes that must be visited in every solution.
        """
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.R = R
        self.C = C
        self.params = params
        self.mandatory_nodes = mandatory_nodes

        self.n_nodes = len(dist_matrix)
        self.nodes = list(range(1, self.n_nodes))  # exclude depot (0)

        # ------------------------------------------------------------------ #
        # Heuristic visibility matrix  eta(i,j) = 1 / d(i,j)                 #
        # ------------------------------------------------------------------ #
        with np.errstate(divide="ignore", invalid="ignore"):
            self.eta = np.where(dist_matrix > 0, 1.0 / dist_matrix, 0.0)

        # ------------------------------------------------------------------ #
        # tau_0 initialisation via nearest-neighbour heuristic                #
        # tau_0 = 1 / (n * C_nn)  (Stützle & Hoos 2000, §4)                  #
        # ------------------------------------------------------------------ #
        if params.tau_0 is None:
            nn_cost = self._nearest_neighbor_cost()
            # When tau_max is dynamic, tau_0 acts as a reasonable warm start.
            initial_tau_max = params.tau_max if params.tau_max is not None else 10.0
            self.tau_0 = 1.0 / (self.n_nodes * nn_cost) if nn_cost > 0 else initial_tau_max
        else:
            self.tau_0 = params.tau_0

        # ------------------------------------------------------------------ #
        # Sparse pheromone matrix                                              #
        # avg_candidates ≈ k_sparse / 2 + 1 for the tau_min formula          #
        # ------------------------------------------------------------------ #
        avg_candidates = params.k_sparse / 2.0 + 1.0
        self.pheromone = SparsePheromoneTau(
            n_nodes=self.n_nodes,
            tau_0=self.tau_0,
            scale=params.scale,
            rho=params.rho,
            tau_min_fixed=params.tau_min,  # None → dynamic
            tau_max_fixed=params.tau_max,  # None → dynamic
            p_best=params.p_best,
            avg_candidates=avg_candidates,
        )

        # ------------------------------------------------------------------ #
        # Local search                                                         #
        # ------------------------------------------------------------------ #
        self.ls = GeneralLocalSearch(
            dist_matrix,
            wastes,
            capacity,
            R,
            C,
            params,
        )

        # ------------------------------------------------------------------ #
        # Candidate lists (k-nearest neighbours per node)                     #
        # ------------------------------------------------------------------ #
        self.candidate_lists = self._build_candidate_lists()

        # ------------------------------------------------------------------ #
        # Solution constructor                                                 #
        # ------------------------------------------------------------------ #
        self.constructor = SolutionConstructor(
            dist_matrix,
            wastes,
            capacity,
            self.pheromone,
            self.eta,
            self.candidate_lists,
            self.nodes,
            params,
            self.tau_0,
            R=self.R,
            C=self.C,
            mandatory_nodes=self.mandatory_nodes,
        )

        # ------------------------------------------------------------------ #
        # Elite pool  (memetic layer)                                         #
        # Stored as a sorted list of (cost, routes) ascending by cost.        #
        # ------------------------------------------------------------------ #
        self.elite_pool: List[Tuple[float, List[List[int]]]] = []

    # ====================================================================== #
    # Public interface                                                         #
    # ====================================================================== #

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """Run the Memetic MMAS algorithm.

        Main loop structure per iteration
        ----------------------------------
        1. Each ant constructs a solution (+ optional local search).
        2. Elite pool is updated.
        3. Pheromone bounds are updated if a new global best was found.
        4. Pheromone is evaporated globally.
        5. Pheromone is reinforced according to the active update schedule.
        6. Stagnation is checked; restart is triggered when necessary.

        Returns:
            Tuple of (best_routes, profit, cost) where:
                best_routes: list of routes (each route = list of node indices).
                profit: total revenue minus total cost (VRPP objective).
                cost: total routing distance of the best solution.
        """
        best_routes: List[List[int]] = []
        best_cost = float("inf")

        # -- Determine iteration-best phase length --------------------------
        ib_phase = self.params.ib_phase_length
        if ib_phase <= 0:
            ib_phase = max(1, self.params.max_iterations // 4)

        stagnation_counter = 0
        start_time = time.perf_counter()

        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.perf_counter() - start_time > self.params.time_limit:
                break

            # ----------------------------------------------------------------
            # 1. Colony construction + optional local search
            # ----------------------------------------------------------------
            iteration_best_routes: List[List[int]] = []
            iteration_best_cost = float("inf")

            for _ in range(self.params.n_ants):
                routes = self.constructor.construct()

                if self.params.local_search:
                    routes = self.ls.optimize(routes)

                cost = self._calculate_cost(routes)

                if cost < iteration_best_cost:
                    iteration_best_cost = cost
                    iteration_best_routes = routes

            # ----------------------------------------------------------------
            # 2. Elite pool update
            # ----------------------------------------------------------------
            self._update_elite_pool(iteration_best_routes, iteration_best_cost)

            # ----------------------------------------------------------------
            # 3. Global best update + dynamic pheromone bounds
            # ----------------------------------------------------------------
            if iteration_best_cost < best_cost:
                best_cost = iteration_best_cost
                best_routes = iteration_best_routes
                stagnation_counter = 0
                # Recompute tau_max = 1/(rho*C_bs) and tau_min accordingly.
                self.pheromone.update_bounds(best_cost)
            else:
                stagnation_counter += 1

            # ----------------------------------------------------------------
            # 4 & 5. Global evaporation + reinforcement
            # ----------------------------------------------------------------
            self.pheromone.evaporate_all(self.params.rho)
            self._global_pheromone_update(
                best_routes=best_routes,
                best_cost=best_cost,
                iteration_best_routes=iteration_best_routes,
                iteration_best_cost=iteration_best_cost,
                iteration=iteration,
                ib_phase=ib_phase,
            )

            # ----------------------------------------------------------------
            # 6. Stagnation detection and restart
            # ----------------------------------------------------------------
            if self.params.stagnation_limit > 0 and stagnation_counter >= self.params.stagnation_limit:
                self._restart(best_routes, best_cost)
                stagnation_counter = 0

            # Optional telemetry hook (no-op unless subclass overrides).
            _tau_vals = [v for nbrs in self.pheromone._pheromone.values() for v in nbrs.values()]
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_cost=best_cost,
                iter_best_cost=iteration_best_cost,
                tau_mean=(float(sum(_tau_vals) / len(_tau_vals)) if _tau_vals else self.pheromone.default_value),
                tau_max=float(max(_tau_vals)) if _tau_vals else self.pheromone.default_value,
            )

        # ------------------------------------------------------------------
        # Final profit computation
        # ------------------------------------------------------------------
        collected_revenue = sum(self.wastes.get(node, 0) * self.R for route in best_routes for node in route)
        profit = collected_revenue - best_cost * self.C
        return best_routes, profit, best_cost

    # ====================================================================== #
    # Initialisation helpers                                                   #
    # ====================================================================== #

    def _nearest_neighbor_cost(self) -> float:
        """Compute a nearest-neighbour tour cost for tau_0 initialisation.

        Constructs a single greedy tour from the depot and returns its total
        distance, which is used as an approximation C_nn for::

            tau_0 = 1 / (n * C_nn)

        Returns:
            Approximate cost of a complete nearest-neighbour tour.
        """
        visited: set = {0}
        current = 0
        cost = 0.0
        for _ in range(len(self.nodes)):
            best_next = None
            best_dist = float("inf")
            for node in self.nodes:
                if node not in visited:
                    d = self.dist_matrix[current][node]
                    if d < best_dist:
                        best_dist = d
                        best_next = node
            if best_next is not None:
                cost += best_dist
                visited.add(best_next)
                current = best_next
        cost += self.dist_matrix[current][0]  # return to depot
        return cost

    def _build_candidate_lists(self) -> Dict[int, List[int]]:
        """Build k-nearest neighbour candidate lists for each node.

        Returns:
            Dictionary mapping each node index to its k nearest neighbours,
            sorted ascending by distance.
        """
        candidates: Dict[int, List[int]] = {}
        k = min(self.params.k_sparse, len(self.nodes))
        for i in range(self.n_nodes):
            distances = [(self.dist_matrix[i][j], j) for j in range(self.n_nodes) if j != i]
            distances.sort()
            candidates[i] = [j for _, j in distances[:k]]
        return candidates

    # ====================================================================== #
    # Cost evaluation                                                          #
    # ====================================================================== #

    def _calculate_cost(self, routes: List[List[int]]) -> float:
        """Compute the total routing distance of a solution.

        Args:
            routes: List of routes, each a sequence of customer node indices.

        Returns:
            Total distance (depot→route→depot for each route).
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

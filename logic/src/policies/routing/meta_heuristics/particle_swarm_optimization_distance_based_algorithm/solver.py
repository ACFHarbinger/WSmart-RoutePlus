"""
Particle Swarm Optimization with Velocity Momentum for VRPP.

**TRUE PSO IMPLEMENTATION** with inertia-weighted velocity updates.
This replaces the deceptive "Firefly Algorithm" which lacked velocity momentum.

Core PSO Algorithm (Kennedy & Eberhart 1995, Ai & Kachitvichyanukul 2009):
    1. Initialize swarm of particles (solutions) with random velocities
    2. For each iteration:
        a. For each particle i:
            - Update velocity: v(t+1) = w*v(t) + c₁*r₁*(pbest - x(t)) + c₂*r₂*(gbest - x(t))
            - Update position: x(t+1) = x(t) + v(t+1)
            - Evaluate fitness f(x(t+1))
            - Update personal best if f(x(t+1)) > f(pbest_i)
        b. Update global best from all personal bests
        c. Decrease inertia weight w linearly

Velocity in Discrete Space (Ai & Kachitvichyanukul 2009):
    - Velocity magnitude → mutation strength
    - Cognitive term (pbest - x) → reintroduce nodes from personal best
    - Social term (gbest - x) → reintroduce nodes from global best
    - Inertia term w*v → preserve previous move direction

Complexity:
    - Time: O(T × N × n²) where T = iterations, N = pop_size, n = nodes
    - Space: O(N × n) for swarm + velocities + personal bests
    - Velocity update: O(n) per particle (node set differences)

References:
    Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization."
    Proceedings of ICNN'95 - International Conference on Neural Networks.

    Ai, T. J., & Kachitvichyanukul, V. (2009). "A particle swarm optimization
    for the vehicle routing problem with simultaneous pickup and delivery."
    Computers & Operations Research, 36(6), 1693-1702.

Attributes:
    DistancePSOSolver: PSO solver class with velocity momentum for VRPP.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.particle_swarm_optimization_distance_based_algorithm.solver import DistancePSOSolver
    >>> from logic.src.configs.policies.routing.meta_heuristics.psoda import DistancePSOConfig
    >>> params = DistancePSOConfig()
    >>> solver = DistancePSOSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()

The velocity/position update layer (velocity equation, greedy repair,
insertion-cost scoring and random walk) lives in ``velocity_updates.py``
as the ``PSOVelocityUpdateMixin``, mixed into the solver here.
"""

import copy
import random
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.configs.policies.routing.meta_heuristics.aco_ks import KSparseACOConfig
from logic.src.configs.policies.routing.meta_heuristics.psoda import DistancePSOConfig
from logic.src.policies.helpers.local_search.local_search_general import GeneralLocalSearch
from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import build_nn_routes

from .velocity_updates import PSOVelocityUpdateMixin


class DistancePSOSolver(PSOVelocityUpdateMixin):
    """
    Particle Swarm Optimization solver with velocity momentum for VRPP.

    **TRUE PSO** with inertia-weighted velocity updates (Kennedy & Eberhart 1995).
    Each particle maintains:
    - Current position x(t): routing solution
    - Velocity v(t): tendency to change solution (represented as node sets)
    - Personal best pbest: best solution this particle has found

    Attributes:
        dist_matrix: Distance matrix including depot at index 0.
        wastes: Node-to-waste-quantity mapping.
        capacity: Vehicle capacity constraint.
        R: Revenue per unit of waste collected.
        C: Cost per unit of distance.
        params: PSO hyperparameter configuration.
        mandatory_nodes: Nodes that must be visited.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: DistancePSOConfig,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the PSO solver with velocity momentum.

        Args:
            dist_matrix: Distance matrix [n+1 × n+1] including depot at index 0.
            wastes: Dictionary mapping node IDs to waste quantities.
            capacity: Vehicle capacity constraint.
            R: Revenue per unit waste collected.
            C: Cost per unit distance traveled.
            params: PSO configuration parameters.
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

        # PSO State: Velocities and Personal Bests
        self.velocities: List[Set[int]] = []  # Velocity as set of nodes to add/remove
        self.personal_bests: List[List[List[int]]] = []  # pbest for each particle
        self.personal_best_fitness: List[float] = []  # f(pbest) for each particle

        # Initialize Local Search once for reuse
        aco_params = KSparseACOConfig(
            local_search_iterations=self.params.local_search_iterations,
            time_limit=self.params.time_limit,
            profit_aware_operators=self.params.profit_aware_operators,
            seed=self.params.seed,
        )
        self.ls = GeneralLocalSearch(
            dist_matrix=self.dist_matrix,
            waste=self.wastes,
            capacity=self.capacity,
            R=self.R,
            C=self.C,
            params=aco_params,
        )

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Execute Particle Swarm Optimization with velocity momentum.

        Implements standard PSO algorithm (Kennedy & Eberhart 1995):
        1. Initialize swarm with random positions and velocities
        2. Iteratively update velocities and positions
        3. Track personal and global bests

        Returns:
            Tuple of (best_routes, best_profit, best_cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.perf_counter()

        # Initialize swarm (particle positions)
        swarm = [self._initialize_particle() for _ in range(self.params.population_size)]
        fitness_values = [self._evaluate(particle) for particle in swarm]

        # Initialize velocities (empty sets = no movement initially)
        self.velocities = [set() for _ in range(self.params.population_size)]

        # Initialize personal bests (pbest)
        self.personal_bests = [copy.deepcopy(particle) for particle in swarm]
        self.personal_best_fitness = list(fitness_values)

        # Track global best (gbest)
        best_idx = int(np.argmax(fitness_values))
        best_routes = copy.deepcopy(swarm[best_idx])
        best_profit = fitness_values[best_idx]
        best_cost = self._cost(best_routes)

        # PSO main loop with velocity momentum
        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.perf_counter() - start > self.params.time_limit:
                break

            # Compute dynamic inertia weight (linearly decreasing)
            inertia_weight = self.params.get_inertia_weight(iteration)

            # Update each particle using PSO velocity equation
            for i in range(self.params.population_size):
                # Update velocity: v(t+1) = w*v(t) + c₁*r₁*(pbest - x) + c₂*r₂*(gbest - x)
                new_velocity = self._update_velocity(
                    current_position=swarm[i],
                    current_velocity=self.velocities[i],
                    personal_best=self.personal_bests[i],
                    global_best=best_routes,
                    inertia_weight=inertia_weight,
                )
                self.velocities[i] = new_velocity

                # Update position: x(t+1) = x(t) + v(t+1)
                new_position = self._apply_velocity(swarm[i], new_velocity)
                new_fitness = self._evaluate(new_position)

                # Accept new position
                swarm[i] = new_position
                fitness_values[i] = new_fitness

                # Update personal best if improved
                if new_fitness > self.personal_best_fitness[i]:
                    self.personal_bests[i] = copy.deepcopy(new_position)
                    self.personal_best_fitness[i] = new_fitness

                # Update global best if improved
                if new_fitness > best_profit:
                    best_routes = copy.deepcopy(new_position)
                    best_profit = new_fitness
                    best_cost = self._cost(best_routes)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                population_size=self.params.population_size,
            )

        return best_routes, best_profit, best_cost

    # ------------------------------------------------------------------
    # Particle Initialization
    # ------------------------------------------------------------------

    def _initialize_particle(self) -> List[List[int]]:
        """
        Initialize a single particle using nearest-neighbor heuristic.

        Randomized node ordering creates diverse initial swarm distribution.

        Returns:
            A feasible routing solution as list of routes.

        Complexity: O(n²) for nearest-neighbor construction.
        """
        routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.random,
        )
        return routes

    # ------------------------------------------------------------------
    # Fitness Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, routes: List[List[int]]) -> float:
        """
        Evaluate particle fitness (net profit).

        Fitness = Revenue - Cost × C
        Revenue = Σ(waste_collected × R)
        Cost = Total distance traveled

        Args:
            routes: Routing solution.

        Returns:
            Net profit (higher is better).

        Complexity: O(n) for route traversal.
        """
        if not routes:
            return 0.0
        revenue = sum(self.wastes.get(n, 0.0) * self.R for route in routes for n in route)
        return revenue - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """
        Calculate total routing distance.

        Args:
            routes: List of routes.

        Returns:
            Total distance traveled.

        Complexity: O(n) for route traversal.
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

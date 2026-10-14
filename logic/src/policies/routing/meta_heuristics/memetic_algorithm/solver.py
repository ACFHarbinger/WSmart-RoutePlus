"""Memetic Algorithm (MA) for VRPP.

Combines population-based stochastic search with intensive individual refinement.

Attributes:
    MASolver: Main solver class for the Memetic Algorithm.

Example:
    >>> solver = MASolver(dist_matrix, wastes, capacity, R, C, params)
    >>> best_routes, best_profit = solver.solve()
"""

import contextlib
import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators import greedy_insertion, greedy_profit_insertion

from .params import MAParams


class MASolver:
    """Memetic Algorithm solver for VRPP.

    Attributes:
        dist_matrix: Symmetric distance matrix.
        wastes: Mapping of bin IDs to waste quantities.
        capacity: Maximum vehicle collection capacity.
        R: Revenue per kg of waste.
        C: Cost per kg traveled.
        params: Algorithm-specific parameters.
        mandatory_nodes: Nodes that must be visited.
        n_nodes: Number of customer nodes.
        nodes: List of node indices.
        random: Random number generator.
        start_time: Process start time.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: MAParams,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the Memetic Algorithm solver.

        Args:
            dist_matrix: Distance matrix (n_nodes+1 x n_nodes+1).
            wastes: Dictionary mapping node indices to waste levels.
            capacity: Maximum capacity constraint for the vehicle.
            R: Revenue multiplier per unit of waste collected.
            C: Cost multiplier per unit of distance traveled.
            params: Configuration objects containing hyper-parameters.
            mandatory_nodes: Indices of nodes that MUST be visited.

        Returns:
            None.
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
        # self.random is used throughout for consistent reproducibility when a seed is provided.
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()
        self.start_time = time.perf_counter()

    def solve(self) -> Tuple[List[List[int]], float]:
        """
        Executive main loop of the Memetic Algorithm.

        Provides the high-level orchestration of the evolutionary process.

        Algorithm 1: Memetic Generational Step (Moscato 2004, Fig. 3.1)
        -------------------------------------------------------------
        1. Initialize Population: Construct initial feasible solutions.
        2. Evaluate: Measure the fitness (profit) of each individual.
        3. Loop until termination (max generations or time limit):
            a. Selection: Sample breeders based on fitness competition.
            b. Generation: Create offspring via recombination and mutation.
            c. Local Search: Apply "Individual Search" (Memetic phase).
            d. Replacement: Update population with top-tier individuals.

        Returns:
            Tuple[List[List[int]], float]:
                - best_individual: The most profitable routing found during the search.
                - best_fitness: The net profit of the best individual.
        """
        # Phase 1: Initialize population with randomized greedy construction.
        population = self._initialize_population()

        # Track global best across generations.
        best_individual = max(population, key=self._evaluate)
        best_fitness = self._evaluate(best_individual)

        # Main generational evolution loop.
        for generation in range(self.params.max_generations):
            # Check for resource exhaustion (Time Limit).
            if time.perf_counter() - self.start_time > self.params.time_limit:
                break

            # 1. Selection (Fig. 3.1: Identify the 'breeders')
            # Select-From-Population(pop) using tournament selection.
            breeders = self._select_from_population(population)

            # 2. Reproduction & Individual Search (Fig. 3.1 & Fig. 3.2: Create newpop)
            # Generate-New-Population(breeders) via operators and hill-climbing.
            new_population = self._generate_new_population(breeders)

            # 3. Replacement (Fig. 3.1: Update pop)
            # Update-Population(pop, newpop) using the 'Plus' replacement strategy.
            population = self._update_population(population, new_population)

            # Monitor the best solver in this generation.
            current_best = max(population, key=self._evaluate)
            current_fitness = self._evaluate(current_best)

            # Preserving the best overall solution (Elitism).
            if current_fitness > best_fitness:
                best_individual = copy.deepcopy(current_best)
                best_fitness = current_fitness

            # Record metrics for telemetry and visualization.
            getattr(self, "_viz_record", lambda **k: None)(
                iteration=generation,
                best_profit=best_fitness,
                best_cost=self._cost(best_individual),
            )

        return best_individual, best_fitness

    # -------------------------------------------------------------------------
    # FIG 3.1: The Generational Step Components
    # -------------------------------------------------------------------------

    def _select_from_population(self, population: List[List[List[int]]]) -> List[List[List[int]]]:
        """Tournament selection to identify breeders.

        Args:
            population: Current generation.

        Returns:
            Selected breeders.
        """
        breeders = []
        for _ in range(self.params.pop_size):
            # Sample tournament_size unique individuals from the population.
            candidates = self.random.sample(population, min(self.params.tournament_size, len(population)))
            # Find the best performer in this sub-group.
            winner = max(candidates, key=self._evaluate)
            breeders.append(copy.deepcopy(winner))
        return breeders

    def _generate_new_population(self, breeders: List[List[List[int]]]) -> List[List[List[int]]]:
        """Generating the new population via recombination and mutation.

        Args:
            breeders: Selected parents.

        Returns:
            The newly created offspring population.
        """
        new_pop = []

        # Process breeders pairwise to apply recombination.
        for i in range(0, len(breeders), 2):
            p1 = breeders[i]
            # Use same parent if odd number of breeders.
            p2 = breeders[i + 1] if i + 1 < len(breeders) else breeders[i]

            # 1. Recombination (Crossover)
            # Paper definition (p. 4): "exchange of information acquired".
            if self.random.random() < self.params.crossover_rate:
                child = self._recombination(p1, p2)
            else:
                child = copy.deepcopy(p1 if self.random.random() < 0.5 else p2)

            # 2. Mutation
            # Paper definition (p. 4): "injecting new material in the population".
            if self.random.random() < self.params.mutation_rate:
                child = self._mutation(child)

            # 3. Local-Improver (The Individual Search / Memetic Stage)
            # Implements FIG 3.3 by finding local optima in the neighborhood.
            if self.random.random() < self.params.local_search_rate:
                child = self._local_improver(child)

            new_pop.append(child)
            # Duplicate the child to maintain population size if we have space.
            if len(new_pop) < self.params.pop_size and i + 1 < len(breeders):
                new_pop.append(copy.deepcopy(child))

        return new_pop[: self.params.pop_size]

    def _update_population(
        self, old_pop: List[List[List[int]]], new_pop: List[List[List[int]]]
    ) -> List[List[List[int]]]:
        """Update-Population(pop, newpop) via Elitist Replacement.

        Args:
            old_pop: Current generation.
            new_pop: Offspring population.

        Returns:
            The merged, elite-selected population.
        """
        # Combine existing and new individuals.
        combined = old_pop + new_pop
        # Sort by fitness (descending).
        combined.sort(key=self._evaluate, reverse=True)
        # Keep only the top performers.
        return combined[: self.params.pop_size]

    # -------------------------------------------------------------------------
    # FIG 3.3: The Local-Improver (The "Memetic" Engine)
    # -------------------------------------------------------------------------

    def _local_improver(self, solution: List[List[int]]) -> List[List[int]]:
        """Intensive Local Search (2-Opt) refinement.

        Args:
            solution: Candidate routing solution.

        Returns:
            Locally optimized routing solution.
        """
        current = copy.deepcopy(solution)
        improved = True

        # Continue searching until no more 'uphill' moves are found.
        while improved:
            improved = False
            # Optimize each vehicle route independently.
            for r_idx, route in enumerate(current):
                if len(route) < 3:
                    continue

                best_route = route
                best_cost = self._cost([route])

                # 2-Opt Search Loop.
                for i in range(1, len(route) - 1):
                    for j in range(i + 1, len(route)):
                        # Create a new candidate by reversing the segment between i and j.
                        new_route = route[:i] + route[i:j][::-1] + route[j:]
                        new_cost = self._cost([new_route])

                        # Acceptance logic for deterministic improvement.
                        if new_cost < best_cost:
                            best_route = new_route
                            best_cost = new_cost
                            improved = True

                # Replace route in solution if improvement was found.
                current[r_idx] = best_route

        return current

    # -------------------------------------------------------------------------
    # Reproductive Operators (Section 3.1 Design Principles)
    # -------------------------------------------------------------------------

    def _recombination(self, parent1: List[List[int]], parent2: List[List[int]]) -> List[List[int]]:
        """Recombination (Crossover) operator.

        Args:
            parent1: First parent solution.
            parent2: Second parent solution.

        Returns:
            A new child solution.
        """
        # Flatten routes to operate on the sequences of nodes.
        p1_flat = [node for route in parent1 for node in route]
        p2_flat = [node for route in parent2 for node in route]

        if not p1_flat or not p2_flat:
            return copy.deepcopy(parent1)

        # Determine a random crossover point (C-Point).
        cut = self.random.randint(1, min(len(p1_flat), len(p2_flat)))

        # Build child: Take head from parent 1, fill remaining from parent 2 without duplicates.
        child_flat = p1_flat[:cut]
        seen = set(child_flat)
        for node in p2_flat:
            if node not in seen:
                child_flat.append(node)
                seen.add(node)

        # Constraint Enforcement: Ensure all mandatory nodes are in the solution.
        for mn in self.mandatory_nodes:
            if mn not in seen:
                child_flat.append(mn)
                seen.add(mn)

        # Mapping back to the Routing space via greedy reconstruction.
        child_routes = []
        with contextlib.suppress(Exception):
            if self.params.profit_aware_operators:
                child_routes = greedy_profit_insertion(
                    [],
                    child_flat,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
            else:
                child_routes = greedy_insertion(
                    [],
                    child_flat,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
        # Fallback to copy if construction fails for any reason.
        return child_routes if child_routes else copy.deepcopy(parent1)

    def _mutation(self, solution: List[List[int]]) -> List[List[int]]:
        """Mutation (Exploration) operator.

        Args:
            solution: Candidate routing solution.

        Returns:
            The mutated routing solution.
        """
        # Collect all visited nodes.
        flat_nodes = [node for route in solution for node in route]
        if not flat_nodes:
            return solution

        # Select a random subset to relocate.
        n_to_remove = min(self.params.n_removal, len(flat_nodes))
        nodes_to_relocate = self.random.sample(flat_nodes, n_to_remove)

        # Clear selected nodes from existing routes.
        mutated_routes = [[n for n in r if n not in nodes_to_relocate] for r in solution]
        mutated_routes = [r for r in mutated_routes if r]

        # Re-insert the nodes greedily to maintain feasibility.
        with contextlib.suppress(Exception):
            if self.params.profit_aware_operators:
                mutated_routes = greedy_profit_insertion(
                    mutated_routes,
                    nodes_to_relocate,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
            else:
                mutated_routes = greedy_insertion(
                    mutated_routes,
                    nodes_to_relocate,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=self.params.vrpp,
                )
        return mutated_routes

    # -------------------------------------------------------------------------
    # Helper Utilities: Population, Evaluation, and Cost
    # -------------------------------------------------------------------------

    def _initialize_population(self) -> List[List[List[int]]]:
        """Constructs diverse initial solutions.

        Returns:
            Initial feasible population.
        """
        starting_pop = []
        for _ in range(self.params.pop_size):
            # Start with a random permutation of all available customer nodes.
            potential_nodes = copy.copy(self.nodes)
            self.random.shuffle(potential_nodes)

            initial_routes = []
            with contextlib.suppress(Exception):
                # Construct routes while honoring vehicle capacity.
                if self.params.profit_aware_operators:
                    initial_routes = greedy_profit_insertion(
                        [],
                        potential_nodes,
                        self.dist_matrix,
                        self.wastes,
                        self.capacity,
                        self.R,
                        self.C,
                        mandatory_nodes=self.mandatory_nodes,
                        expand_pool=self.params.vrpp,
                    )
                else:
                    initial_routes = greedy_insertion(
                        [],
                        potential_nodes,
                        self.dist_matrix,
                        self.wastes,
                        self.capacity,
                        mandatory_nodes=self.mandatory_nodes,
                        expand_pool=self.params.vrpp,
                    )
            # Ensure we always return at least individual node visits if greedy fails.
            starting_pop.append(initial_routes if initial_routes else [[n] for n in potential_nodes])
        return starting_pop

    def _evaluate(self, solution: List[List[int]]) -> float:
        """Fitness evaluation (Net Profit).

        Args:
            solution: Candidate routing solution.

        Returns:
            Total profit value.
        """
        if not solution:
            return 0.0
        # Sum revenue from all bins visited across all routes.
        total_revenue = sum(self.wastes.get(node, 0.0) * self.R for route in solution for node in route)
        # Subtract the distance cost.
        return total_revenue - self._cost(solution) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Calculates total travel distance.

        Args:
            routes: Candidate routing solution.

        Returns:
            Total distance traveled.
        """
        total_distance = 0.0
        for individual_route in routes:
            if not individual_route:
                continue
            # Distance from depot to the first customer.
            total_distance += self.dist_matrix[0][individual_route[0]]
            # Inter-customer distances within the route.
            for k in range(len(individual_route) - 1):
                total_distance += self.dist_matrix[individual_route[k]][individual_route[k + 1]]
            # Return distance back to the depot.
            total_distance += self.dist_matrix[individual_route[-1]][0]
        return total_distance

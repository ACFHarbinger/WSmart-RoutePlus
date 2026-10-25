"""Velocity and position update layer for the distance-based PSO solver.

Moved unchanged from :mod:`solver`: the mixin holding the discrete PSO
velocity update (inertia, cognitive and social terms), the position
update that destroys and greedily repairs routes, the node-set helper,
the best-insertion-cost scoring and the random-walk exploration move.
"""

import copy
from typing import List, Set

from logic.src.envs.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.operators import greedy_insertion, greedy_profit_insertion, random_removal


class PSOVelocityUpdateMixin:
    """PSO velocity/position updates, insertion scoring and random walk."""

    # ------------------------------------------------------------------
    # PSO Velocity Update Methods
    # ------------------------------------------------------------------

    def _update_velocity(
        self,
        current_position: List[List[int]],
        current_velocity: Set[int],
        personal_best: List[List[int]],
        global_best: List[List[int]],
        inertia_weight: float,
    ) -> Set[int]:
        """
        Update particle velocity using PSO equation (discrete adaptation).

        PSO Velocity Equation (Kennedy & Eberhart 1995):
            v(t+1) = w*v(t) + c₁*r₁*(pbest - x(t)) + c₂*r₂*(gbest - x(t))

        Discrete Interpretation (Ai & Kachitvichyanukul 2009):
            - Velocity = set of nodes representing movement tendency
            - Inertia term w*v(t): preserve fraction of current velocity
            - Cognitive term c₁*(pbest - x): nodes in pbest but not in x
            - Social term c₂*(gbest - x): nodes in gbest but not in x

        Args:
            current_position: Current particle position x(t).
            current_velocity: Current velocity v(t) as node set.
            personal_best: Personal best position pbest.
            global_best: Global best position gbest.
            inertia_weight: Inertia weight w ∈ [0,1].

        Returns:
            New velocity v(t+1) as set of nodes.

        Complexity: O(n) for set operations.
        """
        # Extract node sets
        current_nodes = self._get_node_set(current_position)
        pbest_nodes = self._get_node_set(personal_best)
        gbest_nodes = self._get_node_set(global_best)

        # Inertia term: w * v(t) - probabilistically keep nodes in current velocity
        inertia_contribution = {node for node in current_velocity if self.random.random() < inertia_weight}

        # Cognitive term: c₁ * r₁ * (pbest - x) - nodes in pbest but not in current
        cognitive_difference = pbest_nodes - current_nodes
        r1 = self.random.random()
        cognitive_prob = self.params.c1 * r1
        cognitive_contribution = {node for node in cognitive_difference if self.random.random() < cognitive_prob}

        # Social term: c₂ * r₂ * (gbest - x) - nodes in gbest but not in current
        social_difference = gbest_nodes - current_nodes
        r2 = self.random.random()
        social_prob = self.params.c2 * r2
        social_contribution = {node for node in social_difference if self.random.random() < social_prob}

        # Combine all velocity components
        new_velocity = inertia_contribution | cognitive_contribution | social_contribution

        return new_velocity

    def _apply_velocity(self, current_position: List[List[int]], velocity: Set[int]) -> List[List[int]]:
        """
        Apply velocity to current position to get new position.

        Position Update: x(t+1) = x(t) + v(t+1)

        In discrete space:
        1. Velocity nodes = nodes to add to solution
        2. Remove random nodes (destroy)
        3. Insert velocity nodes greedily (repair)
        4. Apply local search refinement

        Args:
            current_position: Current routing solution x(t).
            velocity: Velocity vector v(t+1) as node set.

        Returns:
            New routing solution x(t+1).

        Complexity: O(n²) for insertion + local search.
        """
        if not velocity:
            # No velocity → small random perturbation for exploration
            return self._random_walk(current_position)

        # Extract current nodes
        current_nodes = self._get_node_set(current_position)

        # Separate velocity into nodes to add vs nodes already present
        nodes_to_add = [n for n in velocity if n not in current_nodes]

        if not nodes_to_add:
            # All velocity nodes already in solution → small perturbation
            return self._random_walk(current_position)

        # Destroy: remove nodes to make room (proportional to velocity magnitude)
        n_remove = min(len(nodes_to_add), max(3, int(len(velocity) * self.params.velocity_to_mutation_rate)))

        try:
            partial_routes, removed_nodes = random_removal(current_position, n_remove, self.random)
        except Exception:
            partial_routes = copy.deepcopy(current_position)
            removed_nodes = []

        # Repair: insert velocity nodes + removed nodes greedily
        all_nodes_to_insert = nodes_to_add + [n for n in removed_nodes if n not in velocity]

        # Score and sort nodes by attraction (same as original attract_toward logic)
        scored_nodes = []
        for node in all_nodes_to_insert:
            profit = self.wastes.get(node, 0.0) * self.R
            fill_level = self.wastes.get(node, 0.0)
            insertion_cost = self._compute_best_insertion_cost(node, partial_routes)
            score = (
                self.params.alpha_profit * profit
                + self.params.beta_will * fill_level
                - self.params.gamma_cost * insertion_cost
            )
            scored_nodes.append((score, node))

        scored_nodes.sort(reverse=True)
        sorted_nodes = [node for _, node in scored_nodes]

        # Greedy insertion
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()
        try:
            if use_profit:
                new_routes = greedy_profit_insertion(
                    partial_routes,
                    sorted_nodes,
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
                    partial_routes,
                    sorted_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
            # Apply local search refinement
            return self.ls.optimize(new_routes)
        except Exception:
            return copy.deepcopy(current_position)

    @staticmethod
    def _get_node_set(routes: List[List[int]]) -> Set[int]:
        """
        Extract set of nodes visited in routing solution.

        Args:
            routes: List of routes.

        Returns:
            Set of node indices.

        Complexity: O(n).
        """
        return {node for route in routes for node in route}

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _compute_best_insertion_cost(self, node: int, routes: List[List[int]]) -> float:
        """
        Compute minimum insertion cost for a node into existing routes.

        Finds the position that minimizes additional distance when inserting node.

        Args:
            node: Node index to insert.
            routes: Existing routes.

        Returns:
            Minimum additional distance required.

        Complexity: O(n) for trying all insertion positions.
        """
        min_cost = float("inf")
        for route in routes:
            for i in range(len(route) + 1):
                prev = 0 if i == 0 else route[i - 1]
                nxt = 0 if i == len(route) else route[i]
                cost = self.dist_matrix[prev][node] + self.dist_matrix[node][nxt] - self.dist_matrix[prev][nxt]
                if cost < min_cost:
                    min_cost = cost

        # If no routes exist, cost = depot → node → depot
        if not routes:
            min_cost = self.dist_matrix[0][node] + self.dist_matrix[node][0]

        return max(0.0, min_cost)

    def _random_walk(self, particle: List[List[int]]) -> List[List[int]]:
        """
        Apply random walk exploration to particle.

        Removes random nodes and reinserts them greedily. This maintains
        swarm diversity and prevents premature convergence.

        Args:
            particle: Current particle (routing solution).

        Returns:
            Perturbed particle after random walk.

        Complexity: O(n²) for destroy-repair operation.
        """
        try:
            n_remove = max(3, self.params.n_removal)
            partial_routes, removed_nodes = random_removal(particle, n_remove, self.random)
            if self.params.profit_aware_operators:
                repaired_routes = greedy_profit_insertion(
                    partial_routes,
                    removed_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    self.R,
                    self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=current_expand_pool(),
                )
            else:
                repaired_routes = greedy_insertion(
                    partial_routes,
                    removed_nodes,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=current_expand_pool(),
                )
            # Apply local search refinement
            return self.ls.optimize(repaired_routes)
        except Exception:
            return copy.deepcopy(particle)

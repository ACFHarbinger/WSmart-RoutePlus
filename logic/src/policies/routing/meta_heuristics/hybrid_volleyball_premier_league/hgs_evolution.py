"""HGS evolution operators for the HVPL solver.

Moved unchanged from :mod:`solver`: the mixin holding the Phase-2
population-evolution layer -- HGS crossover and mutation driven by
tournament selection, the elitist survival selection and the substitution
phase that injects diversity from the passive reserve population.
"""

import copy
from typing import List, Tuple

from logic.src.pipeline.learning.envs.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.operators import (
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
)


class HVPLEvolutionMixin:
    """HGS evolution, substitution and survival-selection operators."""

    # ------------------------------------------------------------------
    # Private: Phase 2 - HGS Evolution
    # ------------------------------------------------------------------

    def _hgs_evolution(self, active_teams: List[List[List[int]]], active_profits: List[float]) -> List[List[List[int]]]:
        """Apply HGS genetic operators for population evolution.

        Replaces deterministic VPL coaching with crossover and mutation.

        Args:
            active_teams: Current active team solutions.
            active_profits: Fitness values for active teams.

        Returns:
            List[List[List[int]]]: Offspring population.
        """
        offspring = []

        for _ in range(self.params.n_teams):
            # Tournament selection
            parent1 = self._tournament_select(active_teams, active_profits)
            parent2 = self._tournament_select(active_teams, active_profits)

            # Crossover
            if self.random.random() < self.params.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)

            # Mutation
            if self.random.random() < self.params.mutation_rate:
                child = self._mutate(child)

            offspring.append(child)

        return offspring

    def _tournament_select(self, teams: List[List[List[int]]], profits: List[float], k: int = 3) -> List[List[int]]:
        """Select best solution from k random candidates.

        Args:
            teams: Population of solutions.
            profits: Fitness values of solutions.
            k: Tournament size.

        Returns:
            List[List[int]]: Best solution from the tournament.
        """
        candidates = self.random.sample(range(len(teams)), min(k, len(teams)))
        best_idx = max(candidates, key=lambda i: profits[i])
        return copy.deepcopy(teams[best_idx])

    def _crossover(self, parent1: List[List[int]], parent2: List[List[int]]) -> List[List[int]]:
        """Order-crossover (OX) style operator for routing solutions.

        Combines node selections from both parents.

        Args:
            parent1: First parent solution.
            parent2: Second parent solution.

        Returns:
            List[List[int]]: Recombined child solution.
        """
        nodes_p1 = {node for route in parent1 for node in route}
        nodes_p2 = {node for route in parent2 for node in route}

        # Combine nodes with preference for profitable ones
        child_nodes = list(nodes_p1 | nodes_p2)

        # If too many nodes, prune low-profit ones
        if len(child_nodes) > len(nodes_p1):
            scored = [(self.wastes.get(n, 0.0), n) for n in child_nodes if n not in self.mandatory_nodes]
            scored.sort(reverse=True)
            keep_count = int(len(nodes_p1) * 1.1)  # Allow 10% growth
            child_nodes = self.mandatory_nodes + [n for _, n in scored[:keep_count]]

        # Rebuild routes
        try:
            if self.params.profit_aware_operators:
                return greedy_profit_insertion(
                    routes=[],
                    removed_nodes=child_nodes,
                    dist_matrix=self.dist_matrix,
                    wastes=self.wastes,
                    capacity=self.capacity,
                    R=self.R,
                    C=self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=current_expand_pool(),
                )
            else:
                return greedy_insertion(
                    routes=[],
                    removed_nodes=child_nodes,
                    dist_matrix=self.dist_matrix,
                    wastes=self.wastes,
                    capacity=self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=current_expand_pool(),
                )
        except Exception:
            return copy.deepcopy(parent1)

    def _mutate(self, routes: List[List[int]]) -> List[List[int]]:
        """Mutation operator using destroy-repair.

        Args:
            routes: Routing solution to mutate.

        Returns:
            List[List[int]]: Mutated solution.
        """
        try:
            n_remove = max(2, int(sum(len(r) for r in routes) * 0.2))
            partial, removed = random_removal(routes, n_remove, self.random)
            if self.params.profit_aware_operators:
                return greedy_profit_insertion(
                    routes=partial,
                    removed_nodes=removed,
                    dist_matrix=self.dist_matrix,
                    wastes=self.wastes,
                    capacity=self.capacity,
                    R=self.R,
                    C=self.C,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=current_expand_pool(),
                )
            else:
                return greedy_insertion(
                    routes=partial,
                    removed_nodes=removed,
                    dist_matrix=self.dist_matrix,
                    wastes=self.wastes,
                    capacity=self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=current_expand_pool(),
                )
        except Exception:
            return copy.deepcopy(routes)

    def _selection(
        self, teams: List[List[List[int]]], profits: List[float]
    ) -> Tuple[List[List[List[int]]], List[float]]:
        """Select next generation with elitism.

        Args:
            teams: Combined parent and offspring teams.
            profits: Fitness values.

        Returns:
            Tuple[List[List[List[int]]], List[float]]: (selected_teams, selected_profits).
        """
        # Sort by profit
        sorted_indices = sorted(range(len(profits)), key=lambda i: profits[i], reverse=True)

        # Select top N
        selected_indices = sorted_indices[: self.params.n_teams]
        selected_teams = [teams[i] for i in selected_indices]
        selected_profits = [profits[i] for i in selected_indices]

        return selected_teams, selected_profits

    # ------------------------------------------------------------------
    # Private: Phase 2 - Substitution
    # ------------------------------------------------------------------

    def _substitution_phase(
        self, active_teams: List[List[List[int]]], passive_teams: List[List[List[int]]]
    ) -> List[List[List[int]]]:
        """Inject diversity by replacing weak teams with passive teams.

        Args:
            active_teams: Current active teams.
            passive_teams: Reserve passive teams.

        Returns:
            List[List[List[int]]]: Active teams with injected diversity.
        """
        n_substitute = max(1, int(self.params.n_teams * self.params.substitution_rate))

        # Replace weakest teams
        for i in range(self.params.n_teams - n_substitute, self.params.n_teams):
            # Select random passive team as replacement
            replacement = self.random.choice(passive_teams)
            active_teams[i] = copy.deepcopy(replacement)

        return active_teams

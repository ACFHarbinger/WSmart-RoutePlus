"""
Soccer League Competition (SLC) algorithm for VRPP.

Models a population of routing solutions as players organised into
hierarchical teams.  The globally best solution is the "superstar".

Dual competition structure:
  1. Intra-team: Each player locally perturbs its own solution; replaces
     itself if improved.
  2. Inter-team: Teams play probabilistic matches; the loser's weakest player
     adopts structure from the winner's best player via OX-inspired recombination.

Stagnation detection: if a team's aggregate profit does not improve for
`stagnation_limit` seasons, the entire team is regenerated from scratch.

The globally best solution is the "superstar", which exerts a coaching
influence on all teams via the Coaching phase.

Reference:
    Moosavian, N., & Rppdsarou, B. K. (2014).
    "Soccer league competition algorithm: A novel meta-heuristic
    algorithm for optimal design of water distribution networks."

Attributes:
    SLCSolver: Solver for the Soccer League Competition algorithm.

Example:
    >>> solver = SLCSolver()
    >>> routes, profit, cost = solver.solve()
"""

import copy
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.configs.policies.aco_ks import KSparseACOConfig
from logic.src.configs.policies.slc import SLCConfig
from logic.src.pipeline.learning.envs.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.local_search.local_search_general import GeneralLocalSearch
from logic.src.policies.helpers.operators import (
    greedy_insertion,
    greedy_profit_insertion,
    random_removal,
)
from logic.src.policies.helpers.operators.destroy_ruin.shaw import shaw_profit_removal
from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import (
    build_nn_routes,
)


class SLCSolver:
    """
    Soccer League Competition solver for VRPP.

    Attributes:
        dist_matrix (np.ndarray): Distance matrix.
        wastes (Dict[int, float]): Waste quantities.
        capacity (float): Vehicle capacity.
        R (float): Revenue per kg of waste.
        C (float): Cost per km traveled.
        params (SLCConfig): Algorithm parameters.
        mandatory_nodes (Optional[List[int]]): Mandatory nodes.
        random (np.random.Generator): Random number generator.
        ls (GeneralLocalSearch): Local search instance.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: SLCConfig,
        mandatory_nodes: Optional[List[int]] = None,
    ):
        """Initializes the Soccer League Competition solver.

        Args:
            dist_matrix: Symmetric distance matrix.
            wastes: Mapping of bin IDs to waste quantities.
            capacity: Maximum vehicle collection capacity.
            R: Revenue per kg of waste.
            C: Cost per km traveled.
            params: Algorithm-specific parameters (teams, size).
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

        # Pre-instantiate Local Search for reuse
        aco_params = KSparseACOConfig(
            local_search_iterations=self.params.local_search_iterations,
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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[List[List[int]], float, float]:
        """
        Run the SLC algorithm.

        Returns:
            Tuple of (routes, profit, cost).
        """
        if self.n_nodes == 0:
            return [], 0.0, 0.0

        start = time.perf_counter()

        # Initialise teams: list of lists of (routes, profit)
        teams: List[List[Tuple[List[List[int]], float]]] = [self._new_team() for _ in range(self.params.n_teams)]
        stagnation: List[int] = [0] * self.params.n_teams
        team_best: List[float] = [max(p for _, p in team) for team in teams]

        # Global best (superstar)
        self.superstars: List[Tuple[List[List[int]], float]] = []
        self._update_superstars(teams)

        best_routes = self.superstars[0][0]
        best_profit = self.superstars[0][1]
        best_cost = self._cost(best_routes)
        for iteration in range(self.params.max_iterations):
            if self.params.time_limit > 0 and time.perf_counter() - start > self.params.time_limit:
                break

            # 1. Racing and Interplays (Intra-team competition)
            # Moosavian (2014): Players improve individually, then teams compete.
            for _t_idx, team in enumerate(teams):
                for p_idx in range(len(team)):
                    routes, profit = team[p_idx]
                    # Intra-team perturbation: localized improvement
                    new_routes = self._perturb(routes)
                    new_profit = self._evaluate(new_routes)
                    if new_profit > profit:
                        team[p_idx] = (new_routes, new_profit)

            # 2. Inter-team competition: Weekly Matches
            # Teams are ranked and play matches (implied by inter-team recombination)
            team_indices = list(range(self.params.n_teams))
            self.random.shuffle(team_indices)
            for k in range(0, len(team_indices) - 1, 2):
                a_idx, b_idx = team_indices[k], team_indices[k + 1]
                self._play_match(teams[a_idx], teams[b_idx])

            # 3. Coaching and Learning Phase (Superstar Influence)
            # Moosavian (2014): Weaker players learn from the "Superstar" (global best).
            self._update_superstars(teams)
            superstar = self.superstars[0]  # Global best

            for _t_idx, team in enumerate(teams):
                for p_idx in range(len(team)):
                    # Coaching: probabilistic learning from the superstar
                    if self.random.random() < 0.2:
                        team[p_idx] = self._coach(team[p_idx], superstar)

            # 4. Substitution Operator (Diversity injection)
            # Replace the worst players globally with new learners
            for t_idx in range(self.params.n_teams):
                if stagnation[t_idx] >= self.params.stagnation_limit:
                    teams[t_idx] = self._new_team()
                    stagnation[t_idx] = 0

            # Stagnation tracking (simplified)
            for _t_idx, team in enumerate(teams):
                current_best = max(p for _, p in team)
                if current_best > team_best[t_idx] + 1e-9:
                    team_best[t_idx] = current_best
                    stagnation[t_idx] = 0
                else:
                    stagnation[t_idx] += 1

            best_routes, best_profit = self.superstars[0]
            best_cost = self._cost(best_routes)

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=iteration,
                best_profit=best_profit,
                best_cost=best_cost,
                n_teams=self.params.n_teams,
            )

        return best_routes, best_profit, best_cost

    def _update_superstars(self, teams: List[List[Tuple[List[List[int]], float]]]) -> None:
        """Maintain top 3 global solutions (Superstars).

        Args:
            teams: List of teams.

        """
        all_players = [p for team in teams for p in team]
        all_players.sort(key=lambda x: x[1], reverse=True)
        self.superstars = all_players[:3]

    def _play_match(
        self,
        team_a: List[Tuple[List[List[int]], float]],
        team_b: List[Tuple[List[List[int]], float]],
    ) -> None:
        """Teams compete; loser's weakest player learns from winner's best (Moosavian 2014).

        Args:
            team_a: First team.
            team_b: Second team.

        """
        fit_a = sum(p for _, p in team_a)
        fit_b = sum(p for _, p in team_b)

        if fit_a >= fit_b:
            winner, loser = team_a, team_b
        else:
            winner, loser = team_b, team_a

        # Winner's best player (Coach for the match)
        winner_best = max(winner, key=lambda x: x[1])[0]

        # Loser's weakest player
        loser_worst_idx = int(np.argmin([p for _, p in loser]))

        # Learning: Loser's weakest player is replaced by a recombined offspring
        child = self._recombine(loser[loser_worst_idx][0], winner_best)
        child_profit = self._evaluate(child)
        loser[loser_worst_idx] = (child, child_profit)

    def _coach(
        self, player: Tuple[List[List[int]], float], superstar: Tuple[List[List[int]], float]
    ) -> Tuple[List[List[int]], float]:
        """
        Superstar Influence (Coaching).

        Moosavian (2014): Weaker players imitate the best solutions (superstars).
        In discrete VRR, this corresponds to recombining with the global best.

        Args:
            player: Player to be coached.
            superstar: Best player.

        Returns:
            Coached player.
        """
        current_routes, current_profit = player
        superstar_routes, _ = superstar

        # Recombine with global superstar
        new_routes = self._recombine(current_routes, superstar_routes)
        new_profit = self._evaluate(new_routes)

        # Moosavian (2014): Player moves towards superstar if it improves their fitness
        return (new_routes, new_profit) if new_profit > current_profit else player

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _new_team(self) -> List[Tuple[List[List[int]], float]]:
        """Create a fresh team of `team_size` random players.

        Returns:
            Team of random players.
        """
        team = []
        for _ in range(self.params.team_size):
            routes = self._build_random_solution()
            profit = self._evaluate(routes)
            team.append((routes, profit))
        return team

    def _build_random_solution(self) -> List[List[int]]:
        """Order-dependent sequential construction (matches ALNS style).

        Returns:
            Randomly constructed routes.
        """
        optimized_routes = build_nn_routes(
            nodes=self.nodes,
            mandatory_nodes=self.mandatory_nodes,
            wastes=self.wastes,
            capacity=self.capacity,
            dist_matrix=self.dist_matrix,
            R=self.R,
            C=self.C,
            rng=self.random,
        )
        return optimized_routes

    def _perturb(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Intra-team perturbation: remove one node and reinsert greedily.

        Args:
            routes: Current player routes.

        Returns:
            Perturbed routes.
        """
        n = max(3, self.params.n_removal)
        use_profit = self.params.profit_aware_operators
        expand_pool = current_expand_pool()

        try:
            if use_profit:
                partial, removed = shaw_profit_removal(
                    routes,
                    n,
                    self.dist_matrix,
                    self.wastes,
                    self.R,
                    self.C,
                    randomization_factor=2.0,
                    rng=self.random,
                )
                repaired = greedy_profit_insertion(
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
                partial, removed = random_removal(routes, n, self.random)
                repaired = greedy_insertion(
                    partial,
                    removed,
                    self.dist_matrix,
                    self.wastes,
                    self.capacity,
                    mandatory_nodes=self.mandatory_nodes,
                    expand_pool=expand_pool,
                )
            # Apply comprehensive local search (reusing instance)
            return self.ls.optimize(repaired)
        except Exception:
            return copy.deepcopy(routes)

    def _recombine(self, loser_routes: List[List[int]], winner_routes: List[List[int]]) -> List[List[int]]:
        """
        Recombine loser's route with winner's best route via OX-inspired injection.

        Extracts a random segment from the winner's flat tour and inserts
        unvisited nodes from that segment into the loser's solution.

        Args:
            loser_routes: Losing player's route set.
            winner_routes: Winner's best player route set.

        Returns:
            New child routing solution.
        """
        winner_flat = [n for r in winner_routes for n in r]
        loser_flat = [n for r in loser_routes for n in r]

        if len(winner_flat) < 2:
            return copy.deepcopy(loser_routes)

        # Extract a random segment from the winner
        a = self.random.randint(0, len(winner_flat) - 1)
        b = self.random.randint(a, min(a + max(1, len(winner_flat) // 3), len(winner_flat)))
        segment = winner_flat[a:b]
        segment_set = set(segment)

        # Order Crossover (OX): fill remaining from loser preserving order
        remaining = [n for n in loser_flat if n not in segment_set]
        insert_pos = min(a, len(remaining))
        child_flat = remaining[:insert_pos] + segment + remaining[insert_pos:]

        # Split into routes by capacity
        child_routes: List[List[int]] = []
        curr_route: List[int] = []
        load = 0.0
        for node in child_flat:
            waste = self.wastes.get(node, 0.0)
            if load + waste <= self.capacity:
                curr_route.append(node)
                load += waste
            else:
                if curr_route:
                    child_routes.append(curr_route)
                curr_route = [node]
                load = waste
        if curr_route:
            child_routes.append(curr_route)

        # Ensure mandatory nodes
        visited = {n for r in child_routes for n in r}
        for n in self.mandatory_nodes:
            if n not in visited:
                child_routes.append([n])

        # Apply comprehensive local search (reusing instance)
        return self.ls.optimize(child_routes)

    def _evaluate(self, routes: List[List[int]]) -> float:
        """Net profit for a set of routes.

        Args:
            routes: List of routes.

        Returns:
            Net profit.
        """
        if not routes:
            return 0.0
        rev = sum(self.wastes.get(n, 0.0) * self.R for r in routes for n in r)
        return rev - self._cost(routes) * self.C

    def _cost(self, routes: List[List[int]]) -> float:
        """Total routing distance.

        Args:
            routes: List of routes.

        Returns:
            Total routing distance.
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

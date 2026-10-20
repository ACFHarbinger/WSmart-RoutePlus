"""
HGS Individual Module.

This module defines the Individual class used in the Hybrid Genetic Search (HGS)
algorithm. It represents a single solution in the population, storing genotype
(giant tour) and phenotype (routes), along with cost and fitness metrics.

Attributes:
    Individual: Class representing a single solution in the population.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.individual import Individual
    >>> ind = Individual(giant_tour=[1, 5, 2])
"""

from typing import List, Optional, Set


class Individual:
    """
    Individual solution representation for genetic algorithm.
    Follows Vidal et al. (2022) HGS-CVRP implementation with VRPP adaptations.

    Genotype/Phenotype Mapping:
        - Genotype: giant_tour (ordered list of all customer nodes, including unvisited)
        - Phenotype: routes (actual vehicle routes after Split algorithm)

    For VRPP (Vehicle Routing Problem with Profits):
        The Split algorithm may skip unprofitable nodes, creating a mismatch between
        genotype and phenotype. To maintain genetic consistency for crossover:

        1. giant_tour contains ALL nodes in a specific order (genotype)
        2. routes contains only visited nodes organized into vehicle routes (phenotype)
        3. Nodes in giant_tour but not in routes are implicitly "unvisited"
        4. Local search operators may move nodes between visited and unvisited sets
        5. Crossover operates on giant_tour, preserving genetic material for all nodes

    In adapting the classical Hybrid Genetic Search (HGS) framework to the Vehicle Routing Problem with
    Profits (VRPP), we must manage nodes that are economically unviable to visit. To do so, we employ a
    continuous giant-tour genotype that encompasses the entire set of available customer nodes, regardless
    of their inclusion in the final routing phenotype. Nodes excluded from the active routes by the Split
    algorithm are retained in their sequence positions, effectively constituting an "implicit dummy route."

    Consequently, we formally define the evolutionary dynamics of the unvisited nodes as follows: the relative
    ordering of unvisited nodes within the giant tour evolves exclusively under the selection pressure exerted
    by the crossover operator. Because the local search strictly optimizes the phenotype (the active routes),
    it does not exert direct optimization pressure on the unvisited sequence. Instead, the eventual inclusion
    of an unvisited node into the active phenotype is governed dynamically by the linear-time Split DP decoder.

    As the genetic algorithm progresses, the dynamic adaptation of capacity penalties and the ongoing
    refinement of the visited-route structure continuously alter the economic landscape of the DP state
    space. It is this evolving state space that periodically renders unvisited nodes profitable, allowing
    the Split algorithm to harvest them from the implicit dummy route into the active phenotype. This design
    choice maintains strict internal consistency and pure evolutionary recombination. We acknowledge, however,
    that this represents a deliberate theoretical trade-off: because the unvisited gene pool is shielded from
    localized exploitation, its sequence convergence is inherently slower than that of the actively routed nodes.

    Attributes:
        giant_tour (List[int]): Ordered list of all customer nodes (genotype).
        fitness (float): Biased fitness score (to be minimized).
        profit_score (float): Net profit of the solution.
        cost (float): Total routing cost.
        revenue (float): Total revenue collected.
        is_feasible (bool): True if solution satisfies all constraints.
        capacity_violation (float): Total capacity excess.
        penalized_cost (float): Cost including penalty for violations.
        dist_to_parents (float): Average diversity distance to parents.
        rank_profit (int): Rank based on profit.
        rank_diversity (int): Rank based on diversity.
        is_coached (bool): True if individual has been improved by local search.
        expand_pool (bool): True if repair operators consider all unvisited nodes.

    Example:
        >>> ind = Individual(giant_tour=[1, 2, 3])
    """

    def __init__(self, giant_tour: List[int], expand_pool: bool = False):
        """
        Initialize an individual with a giant tour.

        Args:
            giant_tour: A list representing the visit order of all clients (genotype).
                       For VRPP, this includes both visited and potentially unvisited nodes.
            expand_pool: If True (VRPP mode), repair operators consider all unvisited nodes.
                        If False (CVRP mode), repair operators only consider removed nodes.

        """
        self.giant_tour = giant_tour
        self._routes: List[List[int]] = []
        self._visited_cache: Optional[Set[int]] = None
        # Fix 8: Fitness is minimized, so initialize to +inf.
        self.fitness = float("inf")
        self.profit_score = -float("inf")
        self.cost = 0.0
        self.revenue = 0.0

        # Feasibility tracking
        self.is_feasible = True
        self.capacity_violation = 0.0  # Total capacity excess across all routes
        self.penalized_cost = 0.0  # Cost including penalty for violations

        # Diversity and ranking
        self.dist_to_parents = 0.0
        self.rank_profit = 0
        self.rank_diversity = 0
        self.is_coached = False

        # VRPP-specific: controls whether repair operators expand node pool
        self.expand_pool = expand_pool

        # Genetic Diversity
        self._edge_key: Optional[frozenset] = None

    @property
    def routes(self) -> List[List[int]]:
        """
        Actual vehicle routes after Split algorithm (phenotype).

        Returns:
            List[List[int]]: List of routes.
        """
        return self._routes

    @routes.setter
    def routes(self, value: List[List[int]]) -> None:
        """
        Set routes and invalidate visited nodes cache.

        Args:
            value: List of routes to set.

        """
        self._routes = value
        self._visited_cache = None
        self._edge_key = None

    @property
    def penalized_profit(self) -> float:
        """
        Revenue minus penalized cost. Used for quality ranking in HGS.
        (Fix 10: Named property for penalised profit expression)

        Returns:
            float: The penalized profit value.
        """
        # penalized_cost = cost + penalty * violation
        # profit_score = revenue - cost
        # revenue - cost - penalty * violation = profit_score - penalized_cost + cost
        return self.profit_score - self.penalized_cost + self.cost

    def get_visited_nodes(self) -> Set[int]:
        """
        Get the set of nodes that are currently visited in the solution.

        Returns:
            Set of node IDs present in routes (phenotype).
        """
        # Fix 9: Cache visited nodes for performance in tight local search loops.
        if self._visited_cache is None:
            self._visited_cache = {node for route in self._routes for node in route}
        return self._visited_cache

    def __lt__(self, other: "Individual") -> bool:
        """
        Comparison based on biased fitness (used for sorting).

        Args:
            other: Other individual to compare with.

        Returns:
            bool: True if this individual's fitness is less than other's.
        """
        return self.fitness < other.fitness

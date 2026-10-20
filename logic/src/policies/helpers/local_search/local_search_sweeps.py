"""
Local Search Operators:
    Exchange:
        - λ-interchange (cross-exchange)
        - OR-opt (relocate chains)
        - Ejection chains

    Route:
        - 2-opt* (inter-route tail exchange)
        - 3-opt* (inter-route 3-route tail exchange)
        - 4-opt* (inter-route 4-route tail exchange)
        - SWAP* (inter-route node swap)
        - 2-opt (intra-route reversal)
        - 3-opt (intra-route reconnection)
        - 4-opt (intra-route segment permutation)

    Move:
        - Relocate single nodes
        - Swap within route

    Heuristics:
        - Lin-Kernighan-Helsgaun (LKH) for TSP sub-problems

Each operator method sweeps every position of the current routes and applies
the first improving move (first improvement); callers chain the methods into
their own descent.

Attributes:
    SweepLocalSearch: The operator sweeps on the :class:`LocalSearch` state.

Example:
    >>> from logic.src.policies.helpers.local_search import SweepLocalSearch
    >>> ls = SweepLocalSearch(dist_matrix, wastes, capacity, R, C, seed=42)
    >>> ls.set_routes(routes)
    >>> ls.or_opt()
    True
    >>> ls.get_routes()
    [[1, 2], [3, 4, 5]]

The exchange-operator sweeps live in ``local_search_sweeps_exchange.py``
(:class:`SweepExchangeMixin`) and the route-operator sweeps in
``local_search_sweeps_route.py`` (:class:`SweepRouteMixin`); the move and heuristic
operators stay here.
"""

from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

from logic.src.policies.helpers.operators.intra_route_local_search import (
    move_relocate,
    move_swap,
    relocate_chain,
)

# Import operators from their respective modules
from logic.src.policies.helpers.operators.search_heuristics import apply_ges, apply_lns, solve_lkh

from .local_search_base import LocalSearch
from .local_search_sweeps_exchange import SweepExchangeMixin
from .local_search_sweeps_route import SweepRouteMixin


class SweepLocalSearch(SweepExchangeMixin, SweepRouteMixin, LocalSearch):
    """
    Local search driven by named operator sweeps over the current routes.

    Shares the :class:`LocalSearch` state (routes, loads, node map, penalties)
    and atomic operators; instead of the granular neighbour loop of
    ``LocalSearch._optimize_internal`` each public method enumerates every
    position for one operator, so callers pick and order the operators.

    Attributes:
        improvement_threshold (float): Minimum improvement a caller requires.
        seed (Optional[int]): Seed of ``random`` and ``rng``.
        rng (np.random.Generator): Random number generator (LKH refinement).
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        improvement_threshold: float = 1e-6,
        seed: Optional[int] = None,
        mandatory_nodes: Optional[List[int]] = None,
        penalty_capacity: float = 1.0,
    ):
        """
        Initialize the sweeps with problem data and configuration.

        Args:
            dist_matrix: Matrix of distances between nodes.
            wastes: Dictionary of wastes for each node.
            capacity: Capacity of the vehicle.
            R: Revenue per unit of waste.
            C: Cost per unit of distance.
            improvement_threshold: Minimum improvement required to accept a move.
            seed: Seed for the random number generators.
            mandatory_nodes: Nodes the drop operator never removes.
            penalty_capacity: Capacity-violation penalty weight of the penalised operators.
        """
        # The sweeps enumerate positions themselves: no granular neighbour lists,
        # hence no top-insertion cache upkeep on each move.
        super().__init__(
            dist_matrix,
            wastes,
            capacity,
            R,
            C,
            params=SimpleNamespace(seed=seed),
            mandatory_nodes=mandatory_nodes,
            neighbors={},
            penalty_capacity=penalty_capacity,
        )
        self.improvement_threshold = improvement_threshold
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def optimize(
        self, solution: List[List[int]], operators: Optional[List[str]] = None, max_iterations: int = 1000
    ) -> List[List[int]]:
        """
        Descend with the named sweeps until none improves.

        After each improving move the descent restarts from the first operator.

        Args:
            solution: Routes to improve.
            operators: Method names to apply, in order. Defaults to relocate, swap,
                2-opt, or-opt, 2-opt* and SWAP*.
            max_iterations: Cap on improving moves.

        Returns:
            List[List[int]]: The improved routes.
        """
        names = operators or ["relocate", "swap", "two_opt_intra", "or_opt", "two_opt_star", "swap_star"]
        moves = [getattr(self, name) for name in names]
        self.set_routes(solution)
        for _ in range(max_iterations):
            if not any(move() for move in moves):
                break
        return self.get_routes()

    def set_routes(self, routes: List[List[int]]) -> None:
        """
        Update the current working set of routes.

        Args:
            routes: New set of routes to optimize.
        """
        self.routes = [r[:] for r in routes]
        self.route_loads = [self._calc_load_fresh(r) for r in self.routes]
        self.node_map = {node: (ri, pi) for ri, r in enumerate(self.routes) for pi, node in enumerate(r)}
        self.route_sectors.clear()
        self.top_insertions.clear()

    def get_routes(self) -> List[List[int]]:
        """
        Get a copy of the current routes.

        Returns:
            List[List[int]]: Deep copy of the current routes.
        """
        return [r[:] for r in self.routes]

    # ===== Move Operators =====

    def relocate(self) -> bool:
        """Relocate: move single nodes to better positions.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        for r_u in range(len(self.routes)):
            route_u = self.routes[r_u]
            if not route_u:
                continue

            for p_u in range(len(route_u)):
                u = route_u[p_u]

                # Try relocating to all routes
                for r_v in range(len(self.routes)):
                    route_v = self.routes[r_v]

                    for p_v in range(len(route_v)):
                        v = route_v[p_v]
                        if move_relocate(self, u, v, r_u, p_u, r_v, p_v):
                            return True
        return False

    def relocate_chain_op(self, max_chain_len: int = 3) -> bool:
        """
        Relocate chain: move a chain of consecutive nodes to another position.

        Args:
            max_chain_len: Maximum length of the chain to relocate. Defaults to 3.

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        for r_u in range(len(self.routes)):
            route_u = self.routes[r_u]
            if not route_u:
                continue

            for chain_len in range(2, max_chain_len + 1):
                if len(route_u) < chain_len:
                    break
                for p_u in range(len(route_u) - chain_len + 1):
                    for r_v in range(len(self.routes)):
                        route_v = self.routes[r_v]
                        for p_v in range(len(route_v)):
                            if relocate_chain(self, r_u, p_u, r_v, p_v, chain_len=chain_len):
                                return True
        return False

    def swap(self) -> bool:
        """Swap: exchange positions of two nodes.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        for r_u in range(len(self.routes)):
            route_u = self.routes[r_u]
            if not route_u:
                continue

            for p_u in range(len(route_u)):
                u = route_u[p_u]

                # Try swapping with nodes in all routes
                for r_v in range(r_u, len(self.routes)):
                    route_v = self.routes[r_v]
                    start_p_v = p_u + 1 if r_v == r_u else 0

                    for p_v in range(start_p_v, len(route_v)):
                        v = route_v[p_v]
                        if move_swap(self, u, v, r_u, p_u, r_v, p_v):
                            return True
        return False

    # ===== Heuristic Operators =====

    def lns(self) -> bool:
        """
        Large Neighbourhood Search (LNS).

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        old_routes = self.get_routes()
        new_routes = apply_lns(old_routes, self.d, self.waste, self.Q, self.R, self.C, self.random)
        if new_routes != old_routes:
            self.set_routes(new_routes)
            return True
        return False

    def ges(self) -> bool:
        """
        Guided Ejection Search (GES).

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        old_routes = self.get_routes()
        new_routes = apply_ges(old_routes, self.d, self.waste, self.Q, self.random)
        if new_routes != old_routes:
            self.set_routes(new_routes)
            return True
        return False

    def lkh_refinement(self) -> bool:
        """
        Apply Lin-Kernighan-Helsgaun heuristic to refine individual routes.

        Uses LKH to optimize each route as a TSP subproblem.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        improved = False
        waste_array = np.zeros(len(self.d))
        for node, w in self.waste.items():
            if node < len(waste_array):
                waste_array[node] = w

        for r_idx, route in enumerate(self.routes):
            if len(route) < 3:
                continue

            # Build TSP instance for this route
            nodes = [0] + route + [0]
            initial_cost = sum(self.d[nodes[i], nodes[i + 1]] for i in range(len(nodes) - 1))

            # Apply LKH
            improved_tour, improved_cost = solve_lkh(
                self.d, initial_tour=nodes, max_iterations=20, np_rng=self.rng, seed=self.seed
            )
            if improved_cost < initial_cost - 1e-6:
                # Extract route from tour (remove depot duplicates)
                new_route = [n for n in improved_tour if n != 0]
                self.routes[r_idx] = new_route
                self._update_map({r_idx})
                improved = True

        return improved


__all__ = ["SweepLocalSearch"]

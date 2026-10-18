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
"""

from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

from logic.src.policies.helpers.operators.inter_route_local_search import (
    cross_exchange,
    cyclic_transfer,
    ejection_chain,
    exchange_2_0,
    exchange_2_1,
    improved_cross_exchange,
    lambda_interchange,
    move_2opt_star,
    move_3opt_star,
    move_kopt_star,
    move_swap_star,
)
from logic.src.policies.helpers.operators.intra_route_local_search import (
    move_2opt_intra,
    move_3opt_intra,
    move_kopt_intra,
    move_or_opt,
    move_relocate,
    move_swap,
    relocate_chain,
    three_permutation,
)

# Import operators from their respective modules
from logic.src.policies.helpers.operators.search_heuristics import apply_ges, apply_lns, solve_lkh

from .local_search_base import LocalSearch


class SweepLocalSearch(LocalSearch):
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

    # ===== Exchange Operators =====

    def or_opt(self, chain_len: int = 2) -> bool:
        """
        OR-opt: relocate chain of nodes using the imported operator.

        Tries all possible OR-opt moves and applies the first improving one.

        Args:
            chain_len: Length of the chain to relocate. Defaults to 2.

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if not self.routes:
            return False

        for r_idx, route in enumerate(self.routes):
            if len(route) < chain_len:
                continue

            for pos in range(len(route)):
                if pos + chain_len > len(route):
                    continue

                if move_or_opt(self, r_idx, pos, chain_len):
                    return True

        return False

    def cross_exchange_op(self, max_seg_len: int = 2) -> bool:
        """
        Cross-exchange: swap segments between routes.

        Systematically tries cross-exchange moves between route pairs.

        Args:
            max_seg_len: Maximum length of segments to swap. Defaults to 2.

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 2:
            return False

        for r_a in range(len(self.routes)):
            for r_b in range(r_a + 1, len(self.routes)):
                for seg_a_len in range(max_seg_len + 1):
                    for seg_b_len in range(max_seg_len + 1):
                        if seg_a_len == 0 and seg_b_len == 0:
                            continue

                        for seg_a_start in range(len(self.routes[r_a]) - seg_a_len + 1):
                            for seg_b_start in range(len(self.routes[r_b]) - seg_b_len + 1):
                                if cross_exchange(self, r_a, seg_a_start, seg_a_len, r_b, seg_b_start, seg_b_len):
                                    return True
        return False

    def improved_cross_exchange_op(self, max_seg_len: int = 2) -> bool:
        """
        Improved Cross-exchange: swap segments between routes with optional internal reversal.

        Args:
            max_seg_len: Maximum length of segments to swap. Defaults to 2.

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 2:
            return False

        for r_a in range(len(self.routes)):
            for r_b in range(r_a + 1, len(self.routes)):
                for seg_a_len in range(max_seg_len + 1):
                    for seg_b_len in range(max_seg_len + 1):
                        if seg_a_len == 0 and seg_b_len == 0:
                            continue

                        for seg_a_start in range(len(self.routes[r_a]) - seg_a_len + 1):
                            for seg_b_start in range(len(self.routes[r_b]) - seg_b_len + 1):
                                if improved_cross_exchange(
                                    self, r_a, seg_a_start, seg_a_len, r_b, seg_b_start, seg_b_len
                                ):
                                    return True
        return False

    def lambda_interchange_op(self, lambda_max: int = 2) -> bool:
        """
        λ-interchange: generalized cross-exchange with segment lengths up to λ.

        Args:
            lambda_max: Maximum length of segments to swap. Defaults to 2.

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        return lambda_interchange(self, lambda_max)

    def ejection_chain_op(self, max_depth: int = 3) -> bool:
        """
        Ejection chain: attempt to empty routes for fleet minimization.

        Args:
            max_depth: Maximum depth of the ejection chain. Defaults to 3.

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 2:
            return False

        # Try to eject from smallest routes first
        route_sizes = [(i, len(r)) for i, r in enumerate(self.routes) if r]
        route_sizes.sort(key=lambda x: x[1])

        # Try ejection on up to 3 smallest routes
        return any(ejection_chain(self, r_idx, max_depth) for r_idx, _ in route_sizes[:3])

    def cyclic_transfer_op(self) -> bool:
        """
        Cyclic transfer: 3-way cyclical node exchange.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 3:
            return False

        # Attempt to find an improving 3-way cycle
        for r_a in range(len(self.routes)):
            if not self.routes[r_a]:
                continue
            for r_b in range(r_a + 1, len(self.routes)):
                if not self.routes[r_b]:
                    continue
                for r_c in range(r_b + 1, len(self.routes)):
                    if not self.routes[r_c]:
                        continue

                    for p_a in range(len(self.routes[r_a])):
                        for p_b in range(len(self.routes[r_b])):
                            for p_c in range(len(self.routes[r_c])):
                                if cyclic_transfer(self, [(r_a, p_a), (r_b, p_b), (r_c, p_c)]):
                                    return True
        return False

    def exchange_chain_op(self) -> bool:
        """
        Exchange chain: evaluate block transfers composed of exchange_2_0 and exchange_2_1.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 2:
            return False

        for r_a in range(len(self.routes)):
            for r_b in range(len(self.routes)):
                if r_a == r_b:
                    continue
                # Try exchange (2, 0)
                for p_a in range(len(self.routes[r_a]) - 1):
                    for p_b in range(len(self.routes[r_b]) + 1):
                        if exchange_2_0(self, r_a, p_a, r_b, p_b):
                            return True

                # Try exchange (2, 1) mutually
                for p_a in range(len(self.routes[r_a]) - 1):
                    for p_b in range(len(self.routes[r_b])):
                        if exchange_2_1(self, r_a, p_a, r_b, p_b):
                            return True
        return False

    # ===== Route Operators =====

    def two_opt_star(self) -> bool:
        """2-opt* inter-route operator using imported implementation.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 2:
            return False

        for r_u in range(len(self.routes)):
            for r_v in range(r_u + 1, len(self.routes)):
                route_u = self.routes[r_u]
                route_v = self.routes[r_v]

                if not route_u or not route_v:
                    continue

                for p_u in range(len(route_u)):
                    u = route_u[p_u]
                    for p_v in range(len(route_v)):
                        v = route_v[p_v]
                        if move_2opt_star(self, u, v, r_u, p_u, r_v, p_v):
                            return True

        return False

    def swap_star(self) -> bool:
        """SWAP* inter-route operator using imported implementation.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 2:
            return False

        for r_u in range(len(self.routes)):
            for r_v in range(r_u + 1, len(self.routes)):
                route_u = self.routes[r_u]
                route_v = self.routes[r_v]

                if not route_u or not route_v:
                    continue

                for p_u in range(len(route_u)):
                    u = route_u[p_u]
                    for p_v in range(len(route_v)):
                        v = route_v[p_v]
                        if move_swap_star(self, u, v, r_u, p_u, r_v, p_v):
                            return True

        return False

    def two_opt_intra(self) -> bool:
        """2-opt intra-route: reverse segments within routes.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        for r_u in range(len(self.routes)):
            route = self.routes[r_u]
            if len(route) < 3:
                continue

            for p_u in range(len(route) - 1):
                u = route[p_u]
                for p_v in range(p_u + 2, len(route)):
                    v = route[p_v]
                    if move_2opt_intra(self, u, v, r_u, p_u, r_u, p_v):
                        return True
        return False

    def three_opt_intra(self) -> bool:
        """3-opt intra-route: reconnect three segments within routes.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        for r_u in range(len(self.routes)):
            route = self.routes[r_u]
            if len(route) < 4:
                continue

            for p_u in range(len(route) - 2):
                u = route[p_u]
                for p_v in range(p_u + 2, len(route)):
                    v = route[p_v]
                    if move_3opt_intra(self, u, v, r_u, p_u, r_u, p_v, self.random):
                        return True
        return False

    def four_opt_intra(self) -> bool:
        """4-opt intra-route: permute four segments within routes.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        for r_u in range(len(self.routes)):
            route = self.routes[r_u]
            if len(route) < 5:
                continue

            for p_u in range(len(route) - 2):
                u = route[p_u]
                for p_v in range(p_u + 2, len(route)):
                    v = route[p_v]
                    if move_kopt_intra(self, u, v, r_u, p_u, r_u, p_v, k=4, rng=self.random):
                        return True
        return False

    def three_permutation_op(self) -> bool:
        """3-permutation intra-route: reorder 3 consecutive nodes.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        for r_u in range(len(self.routes)):
            route = self.routes[r_u]
            if len(route) < 3:
                continue

            for p_u in range(len(route) - 3 + 1):
                if three_permutation(self, r_u, p_u):
                    return True
        return False

    def three_opt_star(self) -> bool:
        """3-opt* inter-route: exchange tails among three routes.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 3:
            return False

        for r_u in range(len(self.routes)):
            route_u = self.routes[r_u]
            if not route_u:
                continue
            for r_v in range(r_u + 1, len(self.routes)):
                route_v = self.routes[r_v]
                if not route_v:
                    continue
                for r_w in range(r_v + 1, len(self.routes)):
                    route_w = self.routes[r_w]
                    if not route_w:
                        continue

                    for p_u in range(len(route_u)):
                        u = route_u[p_u]
                        for p_v in range(len(route_v)):
                            v = route_v[p_v]
                            for p_w in range(len(route_w)):
                                w = route_w[p_w]
                                if move_3opt_star(self, u, v, w, r_u, p_u, r_v, p_v, r_w, p_w):
                                    return True
        return False

    def four_opt_star(self) -> bool:
        """4-opt* inter-route: exchange tails among four routes.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 4:
            return False

        for r_u in range(len(self.routes)):
            route_u = self.routes[r_u]
            if not route_u:
                continue
            for r_v in range(r_u + 1, len(self.routes)):
                route_v = self.routes[r_v]
                if not route_v:
                    continue
                for r_w in range(r_v + 1, len(self.routes)):
                    route_w = self.routes[r_w]
                    if not route_w:
                        continue
                    for r_x in range(r_w + 1, len(self.routes)):
                        route_x = self.routes[r_x]
                        if not route_x:
                            continue

                        for p_u in range(len(route_u)):
                            u = route_u[p_u]
                            for p_v in range(len(route_v)):
                                v = route_v[p_v]
                                for p_w in range(len(route_w)):
                                    w = route_w[p_w]
                                    for p_x in range(len(route_x)):
                                        x = route_x[p_x]
                                        if move_kopt_star(
                                            self,
                                            [(u, r_u, p_u), (v, r_v, p_v), (w, r_w, p_w), (x, r_x, p_x)],
                                        ):
                                            return True
        return False

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

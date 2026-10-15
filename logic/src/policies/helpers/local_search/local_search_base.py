"""
Local Search Base Module.

This module defines the abstract base class for local search algorithms.
It handles common initialization, neighbor lists, and provides wrappers
for atomic move operators.

Attributes:
    LocalSearch: Abstract base class for local search algorithms.

Example:
    >>> # Cannot be instantiated directly
    >>> class MyLS(LocalSearch): ...
"""

from __future__ import annotations

import math
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.interfaces import IAcceptanceCriterion
from logic.src.policies.helpers.operators.inter_route_local_search import (
    move_2opt_star,
    move_cross,
    move_swap_star,
    shift_2_0,
    swap_2_1,
    swap_2_2,
)
from logic.src.policies.helpers.operators.inter_route_local_search.cross_exchange import (
    cross_exchange,
    improved_cross_exchange,
    lambda_interchange,
)
from logic.src.policies.helpers.operators.inter_route_local_search.cyclic_transfer import cyclic_transfer
from logic.src.policies.helpers.operators.inter_route_local_search.ejection_chain import ejection_chain
from logic.src.policies.helpers.operators.inter_route_local_search.exchange_chain import exchange_2_0, exchange_2_1
from logic.src.policies.helpers.operators.intra_route_local_search import (
    move_2opt_intra,
    move_3opt_intra,
    move_or_opt,
    move_relocate,
    move_swap,
    relocate_chain,
)
from logic.src.policies.helpers.operators.intra_route_local_search.k_permutation import three_permutation


class LocalSearch(ABC):
    """Abstract base class for Local Search algorithms.

    Provides common infrastructure for neighbor lists, move operators, and
    acceptance criteria management for routing optimization.

    Attributes:
        d (np.ndarray): Distance matrix.
        waste (Dict[int, float]): Dictionary mapping node IDs to their waste/demand.
        Q (float): Vehicle capacity.
        R (float): Revenue factor per unit of waste collected.
        C (float): Cost factor per unit of distance traveled.
        params (Any): Configuration parameters for the local search.
        penalty_capacity (float): Current penalty factor for capacity violations.
        random (random.Random): Local random number generator.
        acceptance_criterion (IAcceptanceCriterion): Strategy for move acceptance.
        neighbors (Dict[int, List[int]]): Granular neighbor list for each node.
        node_map (Dict[int, Tuple[int, int]]): Maps node IDs to (route_idx, pos).
        route_loads (List[float]): Cached total waste for each route.
        routes (List[List[int]]): Current set of routes in the solution.
        top_insertions (Dict[int, Dict[int, List[Tuple[float, int]]]]): Cache of
            best insertion positions for each node into each route.
        x_coords (Optional[np.ndarray]): X-coordinates for polar sector calculations.
        y_coords (Optional[np.ndarray]): Y-coordinates for polar sector calculations.
        route_sectors (Dict[int, Optional[Tuple[float, float]]]): Cached polar
            sectors (min_angle, max_angle) for each route.
        current_profit (float): Current objective value (total profit).
    """

    def __init__(
        self,
        dist_matrix: Optional[np.ndarray],
        waste: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: Any,
        mandatory_nodes: Optional[List[int]] = None,
        neighbors: Optional[Dict[int, List[int]]] = None,
        penalty_capacity: float = 1.0,
        acceptance_criterion: Optional[IAcceptanceCriterion] = None,
        node_coords: Optional[np.ndarray] = None,
    ):
        """Initialize Local Search base class.

        Args:
            dist_matrix: N x N distance matrix.
            waste: Dictionary of node waste demands.
            capacity: Vehicle load capacity.
            R: Reward factor for collected waste.
            C: Cost factor for travel distance.
            params: Hyperparameters and configuration.
            neighbors: Precomputed granular neighbors. Defaults to None.
            penalty_capacity: Initial capacity violation penalty weight. Defaults to 1.0.
            acceptance_criterion: Custom acceptance strategy. Defaults to None.
            node_coords: (N, 2) array of node coordinates.
        """
        # Store matrix only if provided (O(n^2) memory)
        self.d = np.array(dist_matrix) if dist_matrix is not None else None

        # Store coordinates for on-demand calculation (O(n) memory)
        self.node_coords = node_coords
        if node_coords is not None:
            self.x_coords = node_coords[:, 0]
            self.y_coords = node_coords[:, 1]
        else:
            self.x_coords = np.array([])
            self.y_coords = np.array([])

        self.waste = waste
        self.Q = capacity
        self.R = R
        self.C = C
        self.params = params
        self.penalty_capacity = penalty_capacity
        self.random = random.Random(params.seed) if params.seed is not None else random.Random()
        self.mandatory_nodes = set(mandatory_nodes) if mandatory_nodes else set()

        # Modular Acceptance Criterion Injection
        self.acceptance_criterion = acceptance_criterion or getattr(params, "acceptance_criterion", None)
        if self.acceptance_criterion is None:
            # Fallback for base LocalSearch (elitist hill-climbing)
            from logic.src.policies.acceptance_criteria.base.factory import (
                AcceptanceCriterionFactory,
            )

            self.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

        if neighbors is not None:
            self.neighbors = neighbors
        elif self.d is not None:
            # Traditional O(n^2) neighbor generation
            n_nodes = len(self.d)
            self.neighbors = {}
            nb_granular = getattr(params, "nb_granular", 20)
            for i in range(1, n_nodes):
                row = self.d[i]
                order = np.argsort(row)
                cands = [c for c in order if c not in (i, 0)][:nb_granular]
                self.neighbors[i] = cands
        else:
            raise ValueError("K-Sparse optimization requires precomputed neighbors if dist_matrix is None.")

        # Initialize caches
        self.node_map: Dict[int, Tuple[int, int]] = {}
        self.route_loads: List[float] = []
        self.routes: List[List[int]] = []
        self.top_insertions: Dict[int, Dict[int, List[Tuple[float, int]]]] = {}
        self.route_sectors: Dict[int, Optional[Tuple[float, float]]] = {}
        self.current_profit: float = 0.0
        self._target_neighborhood: str = "all"

    def get_dist(self, i: int, j: int) -> float:
        """Abstraction to fetch distance from matrix or calculate on-demand.

        This follows the MMAS_exp strategy of recomputing
        values to avoid intractable matrix sizes.

        Args:
            i: The index of the first node.
            j: The index of the second node.

        Returns:
            float: The distance between the two nodes.
        """
        if self.d is not None:
            return float(self.d[i, j])
        if self.node_coords is not None:
            return float(np.linalg.norm(self.node_coords[i] - self.node_coords[j]))
        raise RuntimeError("Neither distance matrix nor coordinates available.")

    def _compute_top_insertions(self, route_idx: Optional[int] = None):
        """Modified to use get_dist abstraction.

        Args:
            route_idx: The index of the route to compute the top insertions for.
                                     If None, computes for all routes.
        """
        routed = set(self.node_map.keys())  # O(1) set lookup below
        routes_to_update = [route_idx] if route_idx is not None else range(len(self.routes))
        for r_idx in routes_to_update:
            route = self.routes[r_idx]
            if not route:
                continue
            for node in self.neighbors:
                if node in routed:  # O(1) instead of O(route_length)
                    continue
                if node not in self.top_insertions:
                    self.top_insertions[node] = {}

                insertion_costs = []
                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0
                    # Use get_dist to allow O(n) execution
                    delta = self.get_dist(prev, node) + self.get_dist(node, nxt) - self.get_dist(prev, nxt)
                    insertion_costs.append((delta, pos))

                insertion_costs.sort(key=lambda x: x[0])
                self.top_insertions[node][r_idx] = insertion_costs[:3]

    def _cost(self, routes: List[List[int]]) -> float:
        """Calculate total cost of all routes.

        Args:
            routes: List of routes, where each route is a list of node IDs.

        Returns:
            float: Total distance traveled across all routes, including returns to depot.
        """
        total = 0.0
        for r in routes:
            if not r:
                continue
            total += self.get_dist(0, r[0])
            for i in range(len(r) - 1):
                total += self.get_dist(r[i], r[i + 1])
            total += self.get_dist(r[-1], 0)
        return total

    @abstractmethod
    def optimize(self, solution: Any) -> Any:
        """Optimize the given solution.

        Args:
            solution: The initial solution to optimize.

        Returns:
            Any: The optimized solution.
        """
        pass

    def _optimize_internal(
        self,
        target_neighborhood: Optional[str] = None,
        active_nodes: Optional[Set[int]] = None,
        initial_profit: Optional[float] = None,
    ):
        """
        Core local search loop. Assumes self.routes is populated.

        Args:
            target_neighborhood: If provided, only applies the specified neighborhood operator.
                               If None or "all", applies all available operators.
            active_nodes: If provided, restricts search to moves involving at least one active node.
                        This enables O(1) localization per iteration (FILO).
            initial_profit: Optional absolute profit of the starting solution.
        """
        self.route_loads = [self._calc_load_fresh(r) for r in self.routes]
        if initial_profit is not None:
            self.current_profit = initial_profit
        else:
            # Fallback slow calculation (should be avoided in tight loops)
            self.current_profit = (
                sum(self.waste.get(n, 0) * self.R for r in self.routes for n in r) - self._cost(self.routes) * self.C
            )

        # Initialize node map and sector cache
        self.node_map.clear()
        self.route_sectors.clear()
        for ri, r in enumerate(self.routes):
            for pi, node in enumerate(r):
                self.node_map[node] = (ri, pi)

        # Store target neighborhood for _process_pair to filter operators
        self._target_neighborhood = target_neighborhood if target_neighborhood else "all"

        # Initialize Top-3 Insertion Cache for O(1) SWAP* evaluation (Vidal 2022)
        if any(n not in self.node_map for n in self.neighbors):
            self._compute_top_insertions()

        # Store active nodes set for localization (FILO)
        self._active_nodes = active_nodes

        improved = True
        pairs = [(u, v) for u in self.neighbors for v in self.neighbors[u]]  # Build the full candidate pair list
        it = 0
        t_start = time.perf_counter()
        max_ls_iters = getattr(self.params, "local_search_iterations", 500)
        time_limit = getattr(self.params, "time_limit", 0.0)
        while improved and it < max_ls_iters:
            if time_limit > 0 and time.perf_counter() - t_start > time_limit:
                break

            improved = False
            it += 1
            self.random.shuffle(pairs)
            for u, v in pairs:
                if self._active_nodes is not None and u not in self._active_nodes and v not in self._active_nodes:
                    continue
                if self._process_pair(u, v):
                    improved = True
                    break

            getattr(self, "_viz_record", lambda **k: None)(
                iteration=it,
                n_routes=len(self.routes),
                improved=int(improved),
            )

    def _calc_load_fresh(self, r: List[int]) -> float:
        """Calculate total load of a route.

        Args:
            r: The sequence of node IDs in the route.

        Returns:
            float: Sum of waste for all nodes in the route.
        """
        return sum(self.waste.get(x, 0) for x in r)

    def _should_try_operator(self, op_name: str) -> bool:
        """Check if the operator should be tried based on target_neighborhood filter.

        Args:
            op_name: Name of the operator (e.g., 'relocate', 'swap').

        Returns:
            bool: True if the operator should be executed, False otherwise.
        """
        target = getattr(self, "_target_neighborhood", "all")
        return target == "all" or target == op_name

    def _polar_sector(self, route_idx: int) -> Optional[Tuple[float, float]]:
        """
        Return the (min_angle, max_angle) polar sector of a route in radians,
        measured from depot (node 0). Returns None for empty routes.
        Uses caching to avoid redundant calculations.

        Args:
            route_idx: Index of the route in `self.routes`.

        Returns:
            Optional[Tuple[float, float]]: (min_angle, max_angle) in radians,
                or None if route is empty.
        """
        if route_idx in self.route_sectors:
            return self.route_sectors[route_idx]

        route = self.routes[route_idx]
        if not route:
            self.route_sectors[route_idx] = None
            return None

        x_coords = self.x_coords
        y_coords = self.y_coords
        if x_coords is None or y_coords is None:
            res = (-math.pi, math.pi)
            self.route_sectors[route_idx] = res
            return res

        angles = []
        for node in route:
            angle = math.atan2(y_coords[node] - y_coords[0], x_coords[node] - x_coords[0])
            angles.append(angle)

        res = (min(angles), max(angles))
        self.route_sectors[route_idx] = res
        return res

    def accept_move(self, current_obj: float, candidate_obj: float, **kwargs: Any) -> bool:
        """
        Consult the registered acceptance criterion to decide if a move should be applied.

        Args:
            current_obj: Objective value of the current solution.
            candidate_obj: Objective value of the proposed solution.
            kwargs: Additional context for the acceptance decision.

        Returns:
            bool: True if the move should be accepted, False otherwise.
        """
        if self.acceptance_criterion is None:
            return candidate_obj > current_obj + 1e-4
        accepted, _ = self.acceptance_criterion.accept(current_obj=current_obj, candidate_obj=candidate_obj, **kwargs)
        return accepted

    def step_move(self, current_obj: float, candidate_obj: float, accepted: bool, **kwargs: Any) -> None:
        """
        Perform internal state updates (e.g., cooling, counting) after a move decision.

        Args:
            current_obj: Objective value before the move attempt.
            candidate_obj: Objective value after the move attempt.
            accepted: Whether the move was actually accepted and applied.
            kwargs: Additional context for the step update.
        """
        if self.acceptance_criterion is not None:
            self.acceptance_criterion.step(
                current_obj=current_obj, candidate_obj=candidate_obj, accepted=accepted, **kwargs
            )

    def _sectors_overlap(self, s1: Optional[Tuple[float, float]], s2: Optional[Tuple[float, float]]) -> bool:
        """Return True if two polar sectors overlap, handling the ±π wraparound.

        Args:
            s1: First polar sector.
            s2: Second polar sector.

        Returns:
            bool: True if the sectors overlap (including circular wraparound).
        """
        if s1 is None or s2 is None:
            return True
        # Standard interval overlap
        if s1[0] <= s2[1] and s2[0] <= s1[1]:
            return True
        # Wraparound: shift one sector by 2π and re-test
        shifted = (s1[0] + 2 * math.pi, s1[1] + 2 * math.pi)
        if shifted[0] <= s2[1] and s2[0] <= shifted[1]:
            return True
        shifted = (s2[0] + 2 * math.pi, s2[1] + 2 * math.pi)
        return s1[0] <= shifted[1] and shifted[0] <= s1[1]

    def _process_pair(self, u: int, v: int) -> bool:
        """
        Attempt all applicable move operators on the (u, v) pair.
        Returns True if any improving move was applied.

        Args:
            u: First node ID.
            v: Second node ID.

        Returns:
            bool: True if any move was successfully applied.
        """
        u_loc = self.node_map.get(u)
        v_loc = self.node_map.get(v)

        if not u_loc and not v_loc:
            return False

        # --- NEW VRPP LOGIC: Try Drop Operator ---
        if self._should_try_operator("drop"):
            # Try dropping u if it's currently routed
            if u_loc and self._move_drop(u, u_loc[0], u_loc[1]):
                return True

            # Re-fetch v_loc in case dropping u shifted indices
            v_loc = self.node_map.get(v)
            # Try dropping v if it's currently routed
            if v_loc and self._move_drop(v, v_loc[0], v_loc[1]):
                return True
        # -----------------------------------------

        # Rest of your original logic remains exactly the same
        if not u_loc and v_loc:
            r_v, p_v = v_loc
            return self._should_try_operator("unrouted_insert") and self._move_unrouted_insert(u, r_v, p_v)

        if u_loc and not v_loc:
            r_u, p_u = u_loc
            return self._should_try_operator("unrouted_insert") and self._move_unrouted_insert(v, r_u, p_u)

        r_u, p_u = u_loc  # type: ignore[misc]
        r_v, p_v = v_loc  # type: ignore[misc]

        if r_u != r_v:
            return self._process_inter_route(u, v, r_u, p_u, r_v, p_v)
        else:
            return self._process_intra_route(u, v, r_u, p_u, r_v, p_v)

    def _process_inter_route(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Process moves between different routes.

        Args:
            u: Node from the first route.
            v: Node from the second route.
            r_u: Index of the first route.
            p_u: Position of node `u` in `r_u`.
            r_v: Index of the second route.
            p_v: Position of node `v` in `r_v`.

        Returns:
            bool: True if an improving inter-route move was applied.
        """
        if (
            self._should_try_operator("relocate") or self._should_try_operator("inter_relocate")
        ) and self._move_relocate(u, v, r_u, p_u, r_v, p_v):
            return True

        if (self._should_try_operator("swap") or self._should_try_operator("inter_swap")) and self._move_swap(
            u, v, r_u, p_u, r_v, p_v
        ):
            return True

        if self._should_try_operator("inter_2opt_star") and self._move_2opt_star(u, v, r_u, p_u, r_v, p_v):
            return True
        if self._should_try_operator("cross") and self._move_cross(u, v, r_u, p_u, r_v, p_v):
            return True
        if self._should_try_operator("shift_2_0") and self._move_shift_2_0(r_u, p_u, r_v, p_v):
            return True
        if self._should_try_operator("swap_2_1") and self._move_swap_2_1(r_u, p_u, r_v, p_v):
            return True
        if self._should_try_operator("swap_2_2") and self._move_swap_2_2(r_u, p_u, r_v, p_v):
            return True

        return (
            self._target_neighborhood == "inter_swap_star"
            and self._sectors_overlap(self._polar_sector(r_u), self._polar_sector(r_v))
            and self._move_swap_star(u, v, r_u, p_u, r_v, p_v)
        )

    def _process_intra_route(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Process moves within the same route.

        Args:
            u: First node ID.
            v: Second node ID.
            r_u: Index of the route containing both nodes.
            p_u: Position of node `u`.
            r_v: Position of node `v`.
            p_v: Alias for `p_v` (redundant but kept for API consistency).

        Returns:
            bool: True if an improving intra-route move was applied.
        """
        if (
            self._should_try_operator("relocate") or self._should_try_operator("intra_relocate")
        ) and self._move_relocate(u, v, r_u, p_u, r_v, p_v):
            return True

        if (self._should_try_operator("swap") or self._should_try_operator("intra_swap")) and self._move_swap(
            u, v, r_u, p_u, r_v, p_v
        ):
            return True

        if self._should_try_operator("intra_2opt") and self._move_2opt_intra(u, v, r_u, p_u, r_v, p_v):
            return True
        if self._should_try_operator("intra_or_opt") and self._move_or_opt(r_u, p_u, self.random.choice([1, 2, 3])):
            return True
        return (
            self._should_try_operator("intra_3opt")
            and getattr(self.params, "use_3opt", False)
            and self._move_3opt_intra(u, v, r_u, p_u, r_v, p_v, self.random)
        )

    def _update_map(self, affected_indices: Set[int]):
        """
        Update node map and route loads after route modifications.
        Also updates Top-3 Insertion Cache for affected routes (Vidal 2022).

        Args:
            affected_indices: Set of route indices that were modified.
        """
        # 1. Clean up stale mappings for these routes first.
        # If we don't do this, nodes that were removed/dropped from these routes
        # will still appear as "routed" in node_map at their old (now invalid) positions.
        for node, loc in list(self.node_map.items()):
            if loc[0] in affected_indices:
                del self.node_map[node]

        # 2. Rebuild node_map for affected routes, so the routed set is current
        # before _compute_top_insertions uses it for O(1) skip checks.
        for ri in sorted(affected_indices):
            for pi, node in enumerate(self.routes[ri]):
                self.node_map[node] = (ri, pi)
            self.route_loads[ri] = self._calc_load_fresh(self.routes[ri])
            self.route_sectors.pop(ri, None)  # invalidate stale sector cache

        # Only recompute insertion cache if unvisited nodes actually exist.
        # After a move that only relocates routed nodes (relocate, swap, 2-opt),
        # every neighbor is in node_map and the cache update is pure waste.
        if any(n not in self.node_map for n in self.neighbors):
            for ri in sorted(affected_indices):
                self._compute_top_insertions(route_idx=ri)

    def _get_load_cached(self, ri: int) -> float:
        """Get cached total load of a route.

        Args:
            ri: Route index.

        Returns:
            float: Cached total load of the route.
        """
        return self.route_loads[ri]

    def _move_relocate(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Relocate a node to a new position.

        Args:
            u: Node ID to relocate.
            v: Target neighbor node ID.
            r_u: Current route index of `u`.
            p_u: Current position of `u`.
            r_v: Target route index.
            p_v: Target position (usually after `v`).

        Returns:
            bool: True if the relocate move was accepted.
        """
        return move_relocate(self, u, v, r_u, p_u, r_v, p_v)

    def _move_swap(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Swap two nodes.

        Args:
            u: First node ID.
            v: Second node ID.
            r_u: Route index of `u`.
            p_u: Position of `u`.
            r_v: Route index of `v`.
            p_v: Position of `v`.

        Returns:
            bool: True if the swap move was accepted.
        """
        return move_swap(self, u, v, r_u, p_u, r_v, p_v)

    def _move_swap_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply SWAP* operator (best insertion swap).

        Args:
            u: First node ID.
            v: Second node ID.
            r_u: Route index of `u`.
            p_u: Position of `u`.
            r_v: Route index of `v`.
            p_v: Position of `v`.

        Returns:
            bool: True if the SWAP* move was accepted.
        """
        return move_swap_star(self, u, v, r_u, p_u, r_v, p_v)

    def _move_3opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int, rng: random.Random) -> bool:
        """Apply intra-route 3-opt move.

        Args:
            u: First node ID.
            v: Second node ID.
            r_u: Route index.
            p_u: First position.
            r_v: Route index (same as `r_u`).
            p_v: Second position.
            rng: Random number generator for tie-breaking.

        Returns:
            bool: True if the 3-opt move was accepted.
        """
        return move_3opt_intra(self, u, v, r_u, p_u, r_v, p_v, rng)

    def _move_2opt_star(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply inter-route 2-opt* move.

        Args:
            u: First node ID.
            v: Second node ID.
            r_u: First route index.
            p_u: First position.
            r_v: Second route index.
            p_v: Second position.

        Returns:
            bool: True if the 2-opt* move was accepted.
        """
        return move_2opt_star(self, u, v, r_u, p_u, r_v, p_v)

    def _move_2opt_intra(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply intra-route 2-opt move.

        Args:
            u: First node ID.
            v: Second node ID.
            r_u: Route index.
            p_u: First position.
            r_v: Route index (same as `r_u`).
            p_v: Second position.

        Returns:
            bool: True if the intra-route 2-opt move was accepted.
        """
        return move_2opt_intra(self, u, v, r_u, p_u, r_v, p_v)

    def _move_or_opt(self, r_idx: int, pos: int, chain_len: int) -> bool:
        """Apply Or-opt move.

        Args:
            r_idx: Route index.
            pos: Starting position.
            chain_len: Length of the chain to relocate.

        Returns:
            bool: True if the Or-opt move was accepted.
        """
        return move_or_opt(self, r_idx, pos, chain_len)

    def _move_cross(self, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply Cross move.

        Args:
            u: First node ID.
            v: Second node ID.
            r_u: First route index.
            p_u: First position.
            r_v: Second route index.
            p_v: Second position.

        Returns:
            bool: True if the Cross move was accepted.
        """
        return move_cross(self, u, v, r_u, p_u, r_v, p_v)

    def _move_shift_2_0(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply Shift(2,0) move (relocate two nodes).

        Args:
            r_u: Source route index.
            p_u: Starting position of the 2-node chain.
            r_v: Target route index.
            p_v: Target insertion position.

        Returns:
            bool: True if the Shift(2,0) move was accepted.
        """
        return shift_2_0(self, r_u, p_u, r_v, p_v)

    def _move_swap_2_1(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply Swap(2,1) move (swap 2 nodes with 1 node).

        Args:
            r_u: First route index.
            p_u: Starting position of the 2-node chain.
            r_v: Second route index.
            p_v: Position of the single node.

        Returns:
            bool: True if the Swap(2,1) move was accepted.
        """
        return swap_2_1(self, r_u, p_u, r_v, p_v)

    def _move_swap_2_2(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply Swap(2,2) move (swap two pairs of nodes).

        Args:
            r_u: First route index.
            p_u: Starting position of the first 2-node chain.
            r_v: Second route index.
            p_v: Starting position of the second 2-node chain.

        Returns:
            bool: True if the Swap(2,2) move was accepted.
        """
        return swap_2_2(self, r_u, p_u, r_v, p_v)

    def _try_cross_exchange(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply Cross-Exchange operator.

        Args:
            r_u: First route index.
            p_u: First position.
            r_v: Second route index.
            p_v: Second position.

        Returns:
            bool: True if the Cross-Exchange move was accepted.
        """
        return cross_exchange(self, r_u, p_u, 1, r_v, p_v, 1)

    def _try_improved_cross_exchange(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply Improved Cross-Exchange operator.

        Args:
            r_u: First route index.
            p_u: First position.
            r_v: Second route index.
            p_v: Second position.

        Returns:
            bool: True if the Improved Cross-Exchange move was accepted.
        """
        return improved_cross_exchange(self, r_u, p_u, 1, r_v, p_v, 1)

    def _try_lambda_interchange(self, r_u: int, r_v: int) -> bool:
        """Apply Lambda-Interchange operator.

        Args:
            r_u: First route index.
            r_v: Second route index.

        Returns:
            bool: True if the Lambda-Interchange move was accepted.
        """
        return lambda_interchange(self, getattr(self.params, "lambda_max", 2))

    def _try_cyclic_transfer(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Apply Cyclic Transfer move (3-route exchange).

        Args:
            r_u: First route index.
            p_u: First position.
            r_v: Second route index.
            p_v: Second position.

        Returns:
            bool: True if the Cyclic Transfer move was accepted.
        """
        if len(self.routes) < 3:
            return False
        candidates = [r for r in range(len(self.routes)) if r != r_u and r != r_v and self.routes[r]]
        if not candidates:
            return False
        r_w = self.random.choice(candidates)
        p_w = self.random.randint(0, len(self.routes[r_w]) - 1)
        return cyclic_transfer(self, [(r_u, p_u), (r_v, p_v), (r_w, p_w)])

    def _try_exchange_chains(self, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Try applying various Exchange Chain operators.

        Args:
            r_u: First route index.
            p_u: First position.
            r_v: Second route index.
            p_v: Second position.

        Returns:
            bool: True if any Exchange Chain move was accepted.
        """
        if exchange_2_0(self, r_u, p_u, r_v, p_v):
            return True
        return exchange_2_1(self, r_u, p_u, r_v, p_v)

    def _try_ejection_chain(self, r_u: int) -> bool:
        """Docstring.

        Args:
            r_u: Starting route index for the ejection chain.

        Returns:
            bool: True if the Ejection Chain move was accepted.
        """
        return ejection_chain(self, r_u)

    def _move_relocate_chain(self, u: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
        """Docstring.

        Args:
            u: Starting node ID for the chain.
            r_u: Source route index.
            p_u: Starting position in `r_u`.
            r_v: Target route index.
            p_v: Target position in `r_v`.

        Returns:
            bool: True if the Relocate Chain move was accepted.
        """
        if relocate_chain(self, r_u, p_u, r_v, p_v, chain_len=2):
            return True
        return relocate_chain(self, r_u, p_u, r_v, p_v, chain_len=3)

    def _move_three_permutation(self, u: int, r_u: int, p_u: int) -> bool:
        """Apply 3-Permutation move.

        Args:
            u: Node ID to permute.
            r_u: Route index.
            p_u: Position of node `u`.

        Returns:
            bool: True if the 3-Permutation move was accepted.
        """
        return three_permutation(self, r_u, p_u)

    def _move_unrouted_insert(self, node: int, route_idx: int, _hint_pos: int) -> bool:
        """
        Try inserting an unrouted node back into a route in O(1) time (VRPP operator).

        Evaluates inserting 'node' (currently unvisited) into 'route_idx' at the best position.
        Uses Top-3 Insertion Cache for O(1) position lookup instead of O(N) loop.
        Uses penalized search: considers both profit gain and capacity penalties.

        Args:
            node: Unvisited node to potentially insert.
            route_idx: Route to insert into.
            _hint_pos: Position hint (unused).

        Returns:
            bool: True if insertion improves penalized profit.
        """
        route = self.routes[route_idx]

        # 1. Base Case: Empty route
        if not route:
            cost_delta = 2 * self.get_dist(0, node) * self.C
            best_pos = 0
        else:
            # 2. O(1) Cache Query - No loops required!
            # Since 'node' is unrouted, all cached positions are valid (no removal offsets)
            if node in self.top_insertions and route_idx in self.top_insertions[node]:
                # Always use first cached position (guaranteed valid for unrouted nodes)
                best_delta_raw, best_pos = self.top_insertions[node][route_idx][0]
                cost_delta = best_delta_raw * self.C
            else:
                # Cache miss: compute insertion cost directly
                best_pos, best_delta = 0, float("inf")
                for pos in range(len(route) + 1):
                    prev = route[pos - 1] if pos > 0 else 0
                    nxt = route[pos] if pos < len(route) else 0
                    delta = self.get_dist(prev, node) + self.get_dist(node, nxt) - self.get_dist(prev, nxt)
                    if delta < best_delta:
                        best_delta, best_pos = delta, pos
                cost_delta = best_delta * self.C

        profit_delta = self.waste.get(node, 0) * self.R
        load_delta = self.waste.get(node, 0)

        # 3. Penalties & Evaluation (Standard HGS logic)
        penalty_weight = getattr(self, "penalty_capacity", 1.0)
        current_load = self._get_load_cached(route_idx)
        new_load = current_load + load_delta

        old_penalty = penalty_weight * max(0.0, current_load - self.Q)
        new_penalty = penalty_weight * max(0.0, new_load - self.Q)
        penalty_delta = new_penalty - old_penalty

        # Total change in objective: profit gain - cost increase - penalty increase
        total_change = profit_delta - cost_delta - penalty_delta

        # Accept if improves penalized profit
        if total_change > 1e-4:
            if not route:
                self.routes[route_idx] = [node]
            else:
                self.routes[route_idx].insert(best_pos, node)
            self._update_map({route_idx})
            return True

        return False

    def _move_drop(self, node: int, route_idx: int, pos: int) -> bool:
        """
        Try dropping a routed node in O(1) time (VRPP operator).

        Evaluates removing 'node' from 'route_idx' at 'pos'.
        Uses penalized search: considers both profit loss and capacity penalty reduction.

        Args:
            node: Routed node to potentially drop.
            route_idx: Route to drop from.
            pos: Position of the node in the route.

        Returns:
            bool: True if dropping improves penalized profit.
        """
        route = self.routes[route_idx]

        # Validation
        if pos < 0 or pos >= len(route) or route[pos] != node:
            return False

        if getattr(self, "mandatory_nodes", None) and node in self.mandatory_nodes:
            return False

        prev = route[pos - 1] if pos > 0 else 0
        nxt = route[pos + 1] if pos < len(route) - 1 else 0

        # 1. Cost & Profit Deltas
        # Cost saving is negative delta (we save distance by bypassing the node)
        delta_dist = self.get_dist(prev, nxt) - (self.get_dist(prev, node) + self.get_dist(node, nxt))
        cost_delta = delta_dist * self.C  # Guaranteed to be <= 0 by triangle inequality

        # Profit loss is negative delta
        profit_delta = -self.waste.get(node, 0) * self.R
        load_delta = -self.waste.get(node, 0)

        # 2. Penalties (Standard HGS logic)
        penalty_weight = getattr(self, "penalty_capacity", 1.0)
        current_load = self._get_load_cached(route_idx)
        new_load = current_load + load_delta

        old_penalty = penalty_weight * max(0.0, current_load - self.Q)
        new_penalty = penalty_weight * max(0.0, new_load - self.Q)
        penalty_delta = new_penalty - old_penalty

        # Total change in objective: profit gain - cost increase - penalty increase
        total_change = profit_delta - cost_delta - penalty_delta

        # 3. Acceptance
        # Accept if dropping the node strictly improves the penalized objective
        if total_change > 1e-4:
            route.pop(pos)
            self._update_map({route_idx})
            return True

        return False

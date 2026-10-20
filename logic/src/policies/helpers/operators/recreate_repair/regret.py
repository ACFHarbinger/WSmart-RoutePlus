"""
Regret Insertion Operator Module.

This module implements the Regret-k insertion heuristic for VRP repair.
It calculates a regret value for each unassigned node based on the cost difference
between its best and k-th best insertion positions.

Note:
    Profit-aware regret insertion operators for VRPP are located in :mod:`.regret_profit`.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.repair.regret import RegretInsertion
    >>> operator = RegretInsertion(k=2)
    >>> new_routes = operator.repair(destroyed_routes, unassigned_nodes)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .regret_profit import (
    _get_insertion_options_with_profit as _get_insertion_options_with_profit,
)
from .regret_profit import (
    regret_2_profit_insertion as regret_2_profit_insertion,
)
from .regret_profit import (
    regret_3_profit_insertion as regret_3_profit_insertion,
)
from .regret_profit import (
    regret_4_profit_insertion as regret_4_profit_insertion,
)
from .regret_profit import (
    regret_k_profit_insertion as regret_k_profit_insertion,
)

__all__ = [
    "regret_2_insertion",
    "regret_k_insertion",
    "regret_3_insertion",
    "regret_4_insertion",
    "regret_2_profit_insertion",
    "regret_k_profit_insertion",
    "regret_3_profit_insertion",
    "regret_4_profit_insertion",
    "_get_insertion_options_with_profit",
]


def regret_2_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    noise: float = 0.0,
) -> List[List[int]]:
    """
    Insert removed nodes based on the regret-2 criterion.
    Regret-2 Insertion Heuristic.

    Prioritizes inserting nodes that would be much more expensive to insert
    later (high regret = cost difference between best and 2nd best option).

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to be re-inserted.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        mandatory_nodes: Optional list of mandatory node indices.
        expand_pool: If True, all unvisited nodes are candidates.
        noise: Random noise level for cost calculation.

    Returns:
        List[List[int]]: New routes after insertion.
    """
    return regret_k_insertion(
        routes,
        removed_nodes,
        dist_matrix,
        wastes,
        capacity,
        k=2,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_pool,
        noise=noise,
    )


def regret_k_insertion(  # noqa: C901
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    k: int = 2,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    noise: float = 0.0,
) -> List[List[int]]:
    """
    Insert removed nodes using the regret-k heuristic.

    Implements Algorithm 1 of Pisinger & Ropke (2007), *A general heuristic for
    vehicle routing problems*, §3.2.2 (Regret-k insertion):

    1. Compute for each unassigned request *i* its *k* cheapest insertion costs
       across all current routes: ``c_i^1 \u2264 c_i^2 \u2264 ... \u2264 c_i^k``.
    2. Compute regret: ``regret(i) = \u03a3_{h=2}^{k} (c_i^h - c_i^1)``.
       If fewer than *k* routes can accept *i*, the missing differences are set
       to ``max_observed_cost + large_constant`` so that hard-to-place requests
       are strongly prioritised.
    3. Insert the request with the **maximum** regret.
    4. Tie-breaking (paper, §3.2.2): if two nodes share the same regret, insert
       the one with the **minimum** best insertion cost ``c_i^1``.

    Efficiency: O(U * N * L_avg) incremental evaluation using a RoutingCache.
    Only re-evaluates the modified route after each insertion instead of recalculating
    all possible options (O(U^2 * N * L_avg)).

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to be re-inserted.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        k: Regret degree (2, 3, etc.).
        mandatory_nodes: Optional list of mandatory node indices.
        expand_pool: If True, all unvisited nodes are candidates.
        noise: Random noise level for cost calculation.

    Returns:
        List[List[int]]: New routes after insertion.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    # Calculate current loads and track visited
    loads = []
    visited = set()
    for route in routes:
        loads.append(sum(wastes.get(node, 0) for node in route))
        visited.update(route)

    if expand_pool:
        # All unvisited nodes (including those previously removed) are candidates
        n_nodes = len(dist_matrix) - 1
        unassigned_set = set(range(1, n_nodes + 1)) - visited
    else:
        unassigned_set = set(removed_nodes)
        if mandatory_nodes:
            unassigned_set.update(set(mandatory_nodes) - visited)

    unassigned = sorted(list(unassigned_set))
    if not unassigned:
        return routes

    # --- Routing Cache Initialization ---
    # node_route_cache[node][r_idx] = (cost, pos)
    node_route_cache: Dict[int, List[Tuple[float, int]]] = {}

    def get_best_for_route(node_id: int, r_idx: int) -> Tuple[float, int]:
        """Calculates the best insertion cost and position for a node in a specific route.

        Args:
            node_id: ID of the node to insert.
            r_idx: Index of the target route.

        Returns:
            Tuple[float, int]: (Best insertion cost, best position index).
        """
        route = routes[r_idx]
        node_waste = wastes.get(node_id, 0)
        if loads[r_idx] + node_waste > capacity:
            return float("inf"), -1

        best_r_cost = float("inf")
        best_r_pos = -1
        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0
            cost = dist_matrix[prev, node_id] + dist_matrix[node_id, nxt] - dist_matrix[prev, nxt]
            if noise != 0:
                cost = max(0.0, cost + noise)
            if cost < best_r_cost:
                best_r_cost = cost
                best_r_pos = pos
        return best_r_cost, best_r_pos

    # Initial population
    for node in unassigned:
        node_route_cache[node] = []
        for i in range(len(routes)):
            cost, pos = get_best_for_route(node, i)
            node_route_cache[node].append((cost, pos))

    # --- Main Loop ---
    while unassigned:
        all_candidates = []
        unprofitable_nodes = []

        for node in unassigned:
            options = sorted([c for c in node_route_cache[node] if c[0] != float("inf")], key=lambda x: x[0])

            if not options:
                is_mandatory = node in mandatory_nodes_set
                if not is_mandatory:
                    unprofitable_nodes.append(node)
                continue

            # Formal Lexicographical Metrics
            missing_routes = max(0, k - len(options))

            if len(options) > 1:
                target_k = min(k, len(options))
                regret = sum(options[j][0] - options[0][0] for j in range(1, target_k))
            else:
                regret = 0.0  # Missing routes metric entirely dominates this state

            best_opt = options[0]
            r_idx = -1
            for i, (c, _) in enumerate(node_route_cache[node]):
                if c == best_opt[0]:
                    r_idx = i
                    break

            # Store tuple: (missing_routes, regret, node, (best_cost, r_idx, pos))
            all_candidates.append((missing_routes, regret, node, (best_opt[0], r_idx, best_opt[1])))

        for node in unprofitable_nodes:
            unassigned.remove(node)
            if node in node_route_cache:
                del node_route_cache[node]

        if not all_candidates:
            mandatory_remaining = [n for n in unassigned if n in mandatory_nodes_set]
            if mandatory_remaining:
                node = mandatory_remaining[0]
                # Skip mandatory nodes that exceed capacity
                if wastes.get(node, 0) > capacity:
                    unassigned.remove(node)
                    continue
                routes.append([node])
                loads.append(wastes.get(node, 0))
                unassigned.remove(node)
                # Expand cache for ALL remaining nodes
                new_r_idx = len(routes) - 1
                for u_node in unassigned:
                    cost, pos = get_best_for_route(u_node, new_r_idx)
                    node_route_cache[u_node].append((cost, pos))
                continue
            else:
                break

        # RIGOROUS TIE-BREAKING:
        # 1. Maximize missing routes (hardest to place)
        # 2. Maximize regret (cost difference)
        # 3. Minimize best insertion cost (equivalent to maximizing -best_cost)
        all_candidates.sort(key=lambda x: (x[0], x[1], -x[3][0]), reverse=True)
        _, _, best_node, (best_cost, r_idx, pos) = all_candidates[0]

        # Apply insertion
        routes[r_idx].insert(pos, best_node)
        loads[r_idx] += wastes.get(best_node, 0)
        unassigned.remove(best_node)
        del node_route_cache[best_node]

        # Incremental Update
        for u_node in unassigned:
            new_p_cost, new_p_pos = get_best_for_route(u_node, r_idx)
            node_route_cache[u_node][r_idx] = (new_p_cost, new_p_pos)

    return routes


def regret_3_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    noise: float = 0.0,
) -> List[List[int]]:
    """Convenience wrapper for regret-3 insertion.

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to be re-inserted.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        mandatory_nodes: Optional list of mandatory node indices.
        expand_pool: If True, all unvisited nodes are candidates.
        noise: Random noise level.

    Returns:
        List[List[int]]: New routes after insertion.
    """
    return regret_k_insertion(
        routes,
        removed_nodes,
        dist_matrix,
        wastes,
        capacity,
        k=3,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_pool,
        noise=noise,
    )


def regret_4_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    noise: float = 0.0,
) -> List[List[int]]:
    """Convenience wrapper for regret-4 insertion.

    Args:
        routes: Partial routes.
        removed_nodes: Nodes to be re-inserted.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        mandatory_nodes: Optional list of mandatory node indices.
        expand_pool: If True, all unvisited nodes are candidates.
        noise: Random noise level.

    Returns:
        List[List[int]]: New routes after insertion.
    """
    return regret_k_insertion(
        routes,
        removed_nodes,
        dist_matrix,
        wastes,
        capacity,
        k=4,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_pool,
        noise=noise,
    )


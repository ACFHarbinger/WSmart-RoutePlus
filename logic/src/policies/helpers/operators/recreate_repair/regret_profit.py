"""Profit-aware Regret Insertion Operator Module for VRPP.

This module implements the Regret-k profit-maximising insertion heuristic for VRPP repair.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.utils.policy.routes import (
    prune_unprofitable_routes,
)
from logic.src.utils.policy.seed_hurdle import _dynamic_seed_hurdle


def _get_insertion_options_with_profit(
    node: int,
    routes: List[List[int]],
    loads: List[float],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    is_mandatory: bool,
    noise: float,
    max_dist: float,
    unassigned: Optional[List[int]] = None,
    n_total_optional: int = 0,
    mandatory_nodes_set: Optional[set] = None,
) -> List[Tuple[float, int, int]]:
    """Helper to calculate insertion options for a node with profit logic.

    Args:
        node: Node index.
        routes: List of routes.
        loads: Current route loads.
        dist_matrix: Distance matrix.
        wastes: Node demands.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        is_mandatory: Mandatory status.
        noise: Random noise level.
        max_dist: Max distance (unused).
        unassigned: List of unassigned nodes.
        n_total_optional: Total number of optional nodes.
        mandatory_nodes_set: Set of mandatory nodes.

    Returns:
        List[Tuple[float, int, int]]: List of (profit, route_idx, position).
    """
    node_waste = wastes.get(node, 0)
    revenue = node_waste * R
    node_options = []

    for i, route in enumerate(routes):
        if loads[i] + node_waste > capacity:
            continue

        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0

            cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]

            if noise != 0:
                cost = max(0.0, cost + noise)

            profit = revenue - cost * C

            # Requirement: profit > 0 or is_mandatory
            if is_mandatory or profit > -1e-4:
                node_options.append((profit, i, pos))

    # Evaluate new route (Speculative Seeding Heuristic)
    # See greedy_profit_insertion for theoretical justification
    new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
    if noise != 0:
        new_cost = max(0.0, new_cost + noise)

    new_profit = revenue - (new_cost * C)
    if unassigned is not None and n_total_optional > 0:
        seed_hurdle = _dynamic_seed_hurdle(
            node,
            unassigned,
            mandatory_nodes_set or set(),
            dist_matrix,
            new_cost,
            C,
            n_total_optional,
        )
    else:
        seed_hurdle = -0.5 * (new_cost * C)  # legacy fallback

    if is_mandatory or new_profit >= seed_hurdle:
        # A new route has its best (and only) insertion at pos 0 in a new index
        node_options.append((new_profit, len(routes), 0))

    return node_options


def regret_2_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
    noise: float = 0.0,
) -> List[List[int]]:
    """Insert removed nodes based on the regret-2 criterion maximizing profit.

    Wrapper around regret_k_profit_insertion with k=2.

    Args:
        routes: List of routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        mandatory_nodes: List of mandatory node indices.
        expand_pool: If True, consider all unvisited nodes.
        noise: Random noise level.

    Returns:
        List[List[int]]: Updated routes.
    """
    return regret_k_profit_insertion(
        routes,
        removed_nodes,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        k=2,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_pool,
        noise=noise,
    )


def regret_k_profit_insertion(  # noqa: C901
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    k: int = 2,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
    noise: float = 0.0,
) -> List[List[int]]:
    """
    Regret-k insertion maximizing profit (revenue - cost).

    Efficiency: O(U * N * L_avg) incremental evaluation using a RoutingCache.
    VRPP logic: Instead of minimizing cost, we calculate profit for each position.
    A node is only considered if its best insertion is profitable or if it's mandatory.
    Regret is calculated as the difference between the best and k-th best profits.

    Args:
        routes: List of routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        k: Regret degree.
        mandatory_nodes: List of mandatory node indices.
        expand_pool: If True, consider all unvisited nodes.
        noise: Random noise level for cost calculation.

    Returns:
        List[List[int]]: Updated routes.
    """
    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()
    loads = [sum(wastes.get(node, 0) for node in r) for r in routes]

    # Calculate total optional nodes
    n_total_optional = len(dist_matrix) - 1 - len(mandatory_nodes_set)

    visited = set()
    for r in routes:
        visited.update(r)

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
    node_route_cache: Dict[int, List[Tuple[float, int]]] = {}

    def get_best_for_route_profit(node_id: int, r_idx: int) -> Tuple[float, int]:
        """Calculates the best insertion profit and position for a node in a specific route.

        Args:
            node_id: ID of the node to insert.
            r_idx: Index of the target route.

        Returns:
            Tuple[float, int]: (Best insertion profit, best position index).
        """
        route = routes[r_idx]
        node_waste = wastes.get(node_id, 0)
        if loads[r_idx] + node_waste > capacity:
            return -float("inf"), -1

        revenue = node_waste * R
        is_mandatory = node_id in mandatory_nodes_set
        best_r_profit = -float("inf")
        best_r_pos = -1

        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0
            cost = dist_matrix[prev, node_id] + dist_matrix[node_id, nxt] - dist_matrix[prev, nxt]
            if noise != 0:
                cost = max(0.0, cost + noise)
            profit = revenue - cost * C

            # Skip unprofitable positions for non-mandatory nodes before comparison
            if not is_mandatory and profit < -1e-4:
                continue

            if profit > best_r_profit:
                best_r_profit = profit
                best_r_pos = pos
        return best_r_profit, best_r_pos

    def get_seed_profit_regret(node_id: int) -> float:
        """Calculates the profit of starting a new route with the given node.

        Args:
            node_id: ID of the node to insert.

        Returns:
            float: Profit of the new route.
        """
        node_waste = wastes.get(node_id, 0)
        # Reject nodes that exceed vehicle capacity
        if node_waste > capacity:
            return -float("inf")
        revenue = node_waste * R
        new_cost = dist_matrix[0, node_id] + dist_matrix[node_id, 0]
        new_profit = revenue - (new_cost * C)

        # --- UPDATED DYNAMIC SEED HURDLE ---
        seed_hurdle = _dynamic_seed_hurdle(
            node=node_id,
            unassigned=unassigned,
            mandatory_nodes_set=mandatory_nodes_set,
            dist_matrix=dist_matrix,
            new_cost=new_cost,
            C=C,
            n_total_optional=n_total_optional,
        )

        is_mandatory = node_id in mandatory_nodes_set
        if is_mandatory or new_profit >= seed_hurdle:
            return new_profit
        return -float("inf")

    # Initial population
    for node in unassigned:
        node_route_cache[node] = []
        for i in range(len(routes)):
            p, pos = get_best_for_route_profit(node, i)
            node_route_cache[node].append((p, pos))

    # --- Main Loop ---
    while unassigned:
        all_candidates = []
        skipped_nodes = []

        for node in unassigned:
            is_mandatory = node in mandatory_nodes_set
            seed_p = get_seed_profit_regret(node)
            options = [(p, i, pos) for i, (p, pos) in enumerate(node_route_cache[node]) if p != -float("inf")]
            if seed_p != -float("inf"):
                options.append((seed_p, len(routes), 0))

            if not options:
                if not is_mandatory:
                    skipped_nodes.append(node)
                continue

            options.sort(key=lambda x: x[0], reverse=True)

            # Formal Lexicographical Metrics
            missing_routes = max(0, k - len(options))

            if len(options) > 1:
                target_k = min(k, len(options))
                regret = sum(options[0][0] - options[j][0] for j in range(1, target_k))
            else:
                regret = 0.0

            best_option = options[0]
            all_candidates.append((missing_routes, regret, node, best_option))

        for out_node in skipped_nodes:
            unassigned.remove(out_node)
            if out_node in node_route_cache:
                del node_route_cache[out_node]

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
                # Expand cache
                new_r_idx = len(routes) - 1
                for u_node in unassigned:
                    p, pos = get_best_for_route_profit(u_node, new_r_idx)
                    node_route_cache[u_node].append((p, pos))
                continue
            else:
                break

        # RIGOROUS TIE-BREAKING:
        # 1. Maximize missing routes (hardest to place)
        # 2. Maximize regret (cost difference)
        # 3. Minimize best insertion cost (equivalent to maximizing -best_cost)
        all_candidates.sort(key=lambda x: (x[0], x[1], -x[3][0]), reverse=True)
        _, _, best_node, (best_cost, r_idx, pos) = all_candidates[0]

        if r_idx == len(routes):
            # Seed new route
            routes.append([best_node])
            loads.append(wastes.get(best_node, 0))
            unassigned.remove(best_node)
            del node_route_cache[best_node]
            # Update all nodes with the new route
            new_r_idx = len(routes) - 1
            for u_node in unassigned:
                p, ppos = get_best_for_route_profit(u_node, new_r_idx)
                node_route_cache[u_node].append((p, ppos))
        else:
            # Insert into existing
            routes[r_idx].insert(pos, best_node)
            loads[r_idx] += wastes.get(best_node, 0)
            unassigned.remove(best_node)
            del node_route_cache[best_node]
            # Update modified route for all
            for u_node in unassigned:
                p, ppos = get_best_for_route_profit(u_node, r_idx)
                node_route_cache[u_node][r_idx] = (p, ppos)

    # Clean up any routes that failed to become profitable after speculative seeding
    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_nodes_set)


def regret_3_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
    noise: float = 0.0,
) -> List[List[int]]:
    """Convenience wrapper for regret-3 profit-aware insertion.

    Args:
        routes: List of routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        mandatory_nodes: List of mandatory node indices.
        expand_pool: If True, consider all unvisited nodes.
        noise: Random noise level.

    Returns:
        List[List[int]]: Updated routes.
    """
    return regret_k_profit_insertion(
        routes,
        removed_nodes,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        k=3,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_pool,
        noise=noise,
    )


def regret_4_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
    noise: float = 0.0,
) -> List[List[int]]:
    """Convenience wrapper for regret-4 profit-aware insertion.

    Args:
        routes: List of routes.
        removed_nodes: List of unassigned node indices.
        dist_matrix: Distance matrix.
        wastes: waste look-up.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        mandatory_nodes: List of mandatory node indices.
        expand_pool: If True, consider all unvisited nodes.
        noise: Random noise level.

    Returns:
        List[List[int]]: Updated routes.
    """
    return regret_k_profit_insertion(
        routes,
        removed_nodes,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        k=4,
        mandatory_nodes=mandatory_nodes,
        expand_pool=expand_pool,
        noise=noise,
    )

"""Branch-and-Bound profit-maximising removal (destroy) operator for VRPP.

This module provides the LDS-guided profit removal operator for VRPP.
"""

from random import Random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def _marginal_detour(
    node: int,
    r_idx: int,
    pos: int,
    routes: List[List[int]],
    dist_matrix: np.ndarray,
) -> float:
    """Compute the detour cost saved by removing *node* from *routes[r_idx][pos]*.

    Detour saved = dist(prev, node) + dist(node, next) - dist(prev, next)

    Args:
        node: The customer node to evaluate.
        r_idx: Index of the route containing *node*.
        pos: Position of *node* within *routes[r_idx]*.
        routes: Current plan.
        dist_matrix: Distance matrix (depot at index 0).

    Returns:
        Non-negative detour cost saved by removing *node*.
    """
    route = routes[r_idx]
    prev = route[pos - 1] if pos > 0 else 0
    nxt = route[pos + 1] if pos < len(route) - 1 else 0
    return float(dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt])


def _profit_contribution(
    node: int,
    r_idx: int,
    pos: int,
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
) -> float:
    """Marginal profit contribution of *node* in its current position.

    profit(v) = revenue(v) - C * detour_saved(v)

    A lower (or negative) profit indicates a weak or unprofitable node —
    ideal candidate for removal in VRPP.

    Args:
        node: The customer node.
        r_idx: Route index.
        pos: Position in route.
        routes: Current plan.
        dist_matrix: Distance matrix.
        wastes: Node demand lookup.
        R: Revenue per unit waste.
        C: Cost per unit distance.

    Returns:
        Marginal profit of keeping *node* in its current position.
    """
    detour = _marginal_detour(node, r_idx, pos, routes, dist_matrix)
    revenue = wastes.get(node, 0.0) * R
    return revenue - detour * C


def _lds_remove_profit(  # noqa: C901
    routes: List[List[int]],
    loads: List[float],
    node_positions: Dict[int, Tuple[int, int]],
    candidates: List[Tuple[float, int]],  # sorted [(neg_profit_asc, node), ...]
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    discrep: int,
    current_removal: List[int],
    current_profit_freed: float,
    best_state: List[Optional[Tuple[float, List[int]]]],
) -> None:
    """Recursive LDS tree search to select the best removal set (VRPP).

    Analogous to ``_lds_remove`` but scores candidates by their marginal
    profit contribution.  Nodes with the lowest (or most negative) profit are
    removed first, maximising the freed-profit of the removal set.

    Args:
        routes: Mutable current plan.
        loads: Mutable per-route loads.
        node_positions: Node-to-position mapping.
        candidates: Sorted list of ``(profit_contribution_asc, node)`` —
            nodes sorted by ascending profit (worst/lowest first).
        n_remove: Remaining nodes to select.
        dist_matrix: Distance matrix.
        wastes: Node demand lookup.
        capacity: Vehicle capacity.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        discrep: Remaining LDS discrepancy budget.
        current_removal: Nodes selected so far.
        current_profit_freed: Cumulative freed profit of current removal.
        best_state: ``[(best_freed_profit, best_removal_list)]`` or ``[None]``.
    """
    if n_remove == 0:
        current_best = best_state[0]
        if current_best is None or current_profit_freed > current_best[0]:
            best_state[0] = (current_profit_freed, list(current_removal))
        return

    if not candidates:
        current_best = best_state[0]
        if current_best is None or current_profit_freed > current_best[0]:
            best_state[0] = (current_profit_freed, list(current_removal))
        return

    for i, (profit, node) in enumerate(candidates):
        remaining_discrep = discrep - i
        if remaining_discrep < 0:
            break

        loc = node_positions.get(node)
        if loc is None:
            continue

        r_idx, pos = loc
        node_waste = wastes.get(node, 0.0)

        # Tentatively remove
        route = routes[r_idx]
        route.pop(pos)
        loads[r_idx] -= node_waste

        # Update position map
        for shift_pos in range(pos, len(route)):
            node_positions[route[shift_pos]] = (r_idx, shift_pos)
        del node_positions[node]

        # Rebuild candidate list with updated profit scores
        new_candidates: List[Tuple[float, int]] = []
        for _, c_node in candidates:
            if c_node == node:
                continue
            c_loc = node_positions.get(c_node)
            if c_loc is None:
                continue
            c_r_idx, c_pos = c_loc
            c_profit = _profit_contribution(c_node, c_r_idx, c_pos, routes, dist_matrix, wastes, R, C)
            new_candidates.append((c_profit, c_node))
        # Ascending by profit (worst/lowest first)
        new_candidates.sort(key=lambda x: x[0])

        _lds_remove_profit(
            routes,
            loads,
            node_positions,
            new_candidates,
            n_remove - 1,
            dist_matrix,
            wastes,
            capacity,
            R,
            C,
            remaining_discrep,
            current_removal + [node],
            current_profit_freed + (-profit),  # profit is negative for worst nodes
            best_state,
        )

        # Backtrack
        route.insert(pos, node)
        loads[r_idx] += node_waste
        for shift_pos in range(pos + 1, len(route)):
            node_positions[route[shift_pos]] = (r_idx, shift_pos)
        node_positions[node] = (r_idx, pos)


def bb_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    max_discrepancy: int = 1,
    rng: Optional[Random] = None,
    noise: float = 0.0,
) -> Tuple[List[List[int]], List[int]]:
    """Select nodes to remove via LDS, biasing toward low-profit customers (VRPP).

    VRPP adaptation of ``bb_removal``.  Instead of scoring nodes by
    worst-recovery cost, candidates are scored by their **marginal profit
    contribution** — nodes with the lowest (or most negative) profit are ranked
    first for removal, focusing the search on economically weak customers.

    The LDS branching scheme is identical to ``bb_removal``: the first
    candidate (lowest profit contribution) costs 0 discrepancy; alternatives
    cost 1 discrepancy unit each.

    Args:
        routes: Current plan (list of customer sequences, depot implicit).
        n_remove: Number of nodes to remove.
        dist_matrix: Square distance matrix (depot at index 0).
        wastes: Node demand lookup.
        capacity: Maximum vehicle load.
        R: Revenue per unit of waste collected.
        C: Cost per unit of distance.
        max_discrepancy: LDS discrepancy budget.  Recommended: 0–2.
        rng: Optional random number generator for tie-breaking noise.
        noise: Magnitude of Gaussian noise added to profit scores when *rng*
            is provided.  Set to 0 for deterministic mode.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and removed node IDs.

    Raises:
        ValueError: If *n_remove* or *max_discrepancy* are negative.
    """
    if n_remove < 0:
        raise ValueError(f"n_remove must be non-negative, got {n_remove}")
    if max_discrepancy < 0:
        raise ValueError(f"max_discrepancy must be non-negative, got {max_discrepancy}")

    if n_remove == 0:
        return routes, []

    node_positions: Dict[int, Tuple[int, int]] = {}
    all_nodes: List[int] = []
    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            node_positions[node] = (r_idx, pos)
            all_nodes.append(node)

    if not all_nodes:
        return routes, []

    n_remove = min(n_remove, len(all_nodes))
    loads: List[float] = [sum(wastes.get(n, 0.0) for n in r) for r in routes]

    routes_sim = [list(r) for r in routes]
    loads_sim = list(loads)
    np_sim = dict(node_positions)

    initial_scores: List[Tuple[float, int]] = []
    for node in all_nodes:
        loc = node_positions[node]
        r_idx, pos = loc
        profit = _profit_contribution(node, r_idx, pos, routes, dist_matrix, wastes, R, C)
        if rng is not None and noise > 0.0:
            profit = profit + rng.gauss(0.0, noise)
        initial_scores.append((profit, node))

    # Ascending: lowest profit first (worst nodes first)
    initial_scores.sort(key=lambda x: x[0])

    best_state: List[Optional[Tuple[float, List[int]]]] = [None]

    _lds_remove_profit(
        routes_sim,
        loads_sim,
        np_sim,
        initial_scores,
        n_remove,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        max_discrepancy,
        [],
        0.0,
        best_state,
    )

    if best_state[0] is None:
        removed_ids = [node for _, node in initial_scores[:n_remove]]
    else:
        _, removed_ids = best_state[0]

    # Apply removal
    removed_set: Set[int] = set(removed_ids)
    final_removed: List[int] = []
    modified_routes: List[List[int]] = []

    for route in routes:
        new_route: List[int] = []
        for node in route:
            if node in removed_set:
                final_removed.append(node)
            else:
                new_route.append(node)
        if new_route:
            modified_routes.append(new_route)

    return modified_routes, final_removed

"""Branch-and-Bound profit-maximising (LDS) insertion repair operator for VRPP.

This module provides the profit-maximising variant of the Shaw (1998) LDS insertion operator.
"""

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.utils.policy.routes import (
    prune_unprofitable_routes,
)

# Mutable plan state threaded through the LDS recursion
_Plan = List[List[int]]
_Loads = List[float]


def _get_sorted_positions_profit(
    node: int,
    routes: _Plan,
    loads: _Loads,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    is_mandatory: bool,
    seed_hurdle_factor: float = 0.5,
) -> List[Tuple[float, int, int]]:
    """Return feasible insertion positions for *node*, sorted by descending profit.

    For VRPP, positions are scored by ``profit = revenue - cost_delta * C``.
    Non-mandatory nodes at positions with negative profit are filtered out
    (unless no profitable option exists, in which case all are returned so
    that the caller can decide whether to include the node at all).

    Args:
        node: Customer node to be inserted.
        routes: Current partial plan.
        loads: Cumulative load per route.
        dist_matrix: Distance matrix.
        wastes: Node demand lookup.
        capacity: Maximum vehicle load.
        R: Revenue per unit of waste collected.
        C: Cost per unit of distance travelled.
        is_mandatory: If True, the node must be inserted even if unprofitable.
        seed_hurdle_factor: Minimum net profit expressed as a fraction of the
            new-route cost (default 0.5 → at least -50 % of new-route cost).
            Mirrors the speculative seeding hurdle used in other operators.

    Returns:
        List of ``(neg_profit, route_idx, position)`` triples sorted ascending
        by *neg_profit* (i.e. descending true profit) so that the LDS engine
        can treat cheapest-first and most-profitable-first uniformly using the
        same ascending-cost logic.
    """
    node_waste = wastes.get(node, 0.0)
    revenue = node_waste * R
    positions: List[Tuple[float, int, int]] = []

    for r_idx, route in enumerate(routes):
        if loads[r_idx] + node_waste > capacity:
            continue
        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0
            delta = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
            profit = revenue - delta * C
            if is_mandatory or profit >= -1e-4:
                # Store as -profit so ascending sort → highest profit first
                positions.append((-profit, r_idx, pos))

    # New route: speculative seeding
    if node_waste <= capacity:
        new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
        new_profit = revenue - new_cost * C
        seed_hurdle = -seed_hurdle_factor * new_cost * C
        if is_mandatory or new_profit >= seed_hurdle:
            positions.append((-new_profit, len(routes), 0))

    positions.sort(key=lambda x: x[0])
    return positions


def _lds_reinsert_profit(  # noqa: C901
    routes: _Plan,
    loads: _Loads,
    unrouted: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_set: Set[int],
    discrep: int,
    best_state: List[Optional[Tuple[float, _Plan, _Loads]]],
    seed_hurdle_factor: float,
) -> None:
    """Recursive LDS tree search for profit-maximising re-insertion (VRPP).

    Analogous to ``_lds_reinsert`` but objective is *maximise profit* rather
    than *minimise distance*.  The best_state tuple carries
    ``(-profit, routes_copy, loads_copy)`` so the same ``<`` comparison works
    for maximisation (a larger negative value indicates a better plan).

    Non-mandatory nodes with no profitable position in any route or new route
    are silently dropped from the unrouted list.

    Args:
        routes: Mutable list of routes.
        loads: Mutable per-route accumulated load.
        unrouted: Nodes not yet assigned.
        dist_matrix: Distance matrix.
        wastes: Node demand lookup.
        capacity: Maximum vehicle capacity.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        mandatory_set: Node IDs that must be inserted regardless of profit.
        discrep: Remaining discrepancy budget.
        best_state: Single-element list containing ``(-profit, routes, loads)``
            or ``None``.
        seed_hurdle_factor: Fraction of new-route cost used as seed hurdle.
    """
    if not unrouted:
        # Compute total profit for the complete plan
        total_profit = 0.0
        for route in routes:
            if not route:
                continue
            # Route cost
            route_cost = dist_matrix[0, route[0]]
            for k in range(len(route) - 1):
                route_cost += dist_matrix[route[k], route[k + 1]]
            route_cost += dist_matrix[route[-1], 0]
            # Route revenue
            route_revenue = sum(wastes.get(n, 0.0) for n in route) * R
            total_profit += route_revenue - route_cost * C

        # Negate for minimisation-compatible comparison
        neg_profit = -total_profit
        current_best = best_state[0]
        if current_best is None or neg_profit < current_best[0]:
            best_state[0] = (neg_profit, [list(r) for r in routes], list(loads))
        return

    # --- Variable selection: Farthest Insertion (most constrained first) ---
    # Here "min_cost" is replaced by "max achievable profit" (choose node with
    # the smallest best profit → hardest to place profitably).
    farthest_node: Optional[int] = None
    farthest_best_profit = math.inf  # We want the *minimum* best profit

    droppable: List[int] = []  # Non-mandatory nodes with no feasible profitable positions

    for candidate in unrouted:
        is_mandatory = candidate in mandatory_set
        positions = _get_sorted_positions_profit(
            candidate, routes, loads, dist_matrix, wastes, capacity, R, C, is_mandatory, seed_hurdle_factor
        )
        if not positions:
            if not is_mandatory:
                droppable.append(candidate)
                continue
            # Mandatory node: force a new route as a fallback position
            new_cost = dist_matrix[0, candidate] + dist_matrix[candidate, 0]
            new_profit = wastes.get(candidate, 0.0) * R - new_cost * C
            best_profit_candidate = new_profit
        else:
            # Recall positions stored as (-profit, ...), so positions[0][0] is -best_profit
            best_profit_candidate = -positions[0][0]

        if best_profit_candidate < farthest_best_profit:
            farthest_best_profit = best_profit_candidate
            farthest_node = candidate

    # Remove permanently unprofitable optional nodes before recursing
    remaining_unrouted = [n for n in unrouted if n not in droppable]

    if farthest_node is None or farthest_node in droppable:
        # All remaining nodes are droppable; recurse with cleared list
        _lds_reinsert_profit(
            routes,
            loads,
            [],
            dist_matrix,
            wastes,
            capacity,
            R,
            C,
            mandatory_set,
            discrep,
            best_state,
            seed_hurdle_factor,
        )
        return

    v = farthest_node
    is_mandatory_v = v in mandatory_set
    remaining = [n for n in remaining_unrouted if n != v]

    positions = _get_sorted_positions_profit(
        v, routes, loads, dist_matrix, wastes, capacity, R, C, is_mandatory_v, seed_hurdle_factor
    )

    if not positions and is_mandatory_v:
        # Mandatory node: force a new route
        new_cost = dist_matrix[0, v] + dist_matrix[v, 0]
        positions = [(-((wastes.get(v, 0.0) * R) - new_cost * C), len(routes), 0)]

    # --- LDS branching (most-profitable first) ---
    current_best = best_state[0]

    for i, (neg_profit_delta, r_idx, pos) in enumerate(positions):
        remaining_discrep = discrep - i
        if remaining_discrep < 0:
            break

        # Pruning: upper bound on the maximum achievable profit from this branch
        # Current plan profit + best possible profit for v (a weak upper bound)
        # For VRPP, pruning by lower bound on neg_profit:
        # if we can establish current neg_profit + neg_profit_delta >= best neg_profit, prune
        if current_best is not None:
            # An upper bound on profit remaining: assume all other nodes achieve
            # their best possible profit (weak but cheap bound)
            lb_neg_profit = current_best[0]  # Best neg-profit so far (most negative = best plan)
            # If inserting v at this position yields neg_profit_delta > current_best[0],
            # and all future choices are at their best, this branch cannot improve.
            # Simplified bound: prune if this position's profit alone is worse than best plan
            if -neg_profit_delta <= lb_neg_profit and not is_mandatory_v:
                continue  # Prune

        node_waste = wastes.get(v, 0.0)
        new_route_opened = r_idx == len(routes)

        if new_route_opened:
            routes.append([v])
            loads.append(node_waste)
        else:
            routes[r_idx].insert(pos, v)
            loads[r_idx] += node_waste

        _lds_reinsert_profit(
            routes,
            loads,
            remaining,
            dist_matrix,
            wastes,
            capacity,
            R,
            C,
            mandatory_set,
            remaining_discrep,
            best_state,
            seed_hurdle_factor,
        )

        # Backtrack
        if new_route_opened:
            routes.pop()
            loads.pop()
        else:
            routes[r_idx].pop(pos)
            loads[r_idx] -= node_waste


def bb_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    max_discrepancy: int = 2,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = False,
    seed_hurdle_factor: float = 0.5,
) -> List[List[int]]:
    """Re-insert removed visits via LDS tree search, maximising route profit.

    VRPP adaptation of ``bb_insertion``.  Extends the Shaw (1998) framework
    with profit-aware scoring (revenue - cost) and speculative seeding of new
    routes via a hurdle threshold.

    The search objective is to maximise the aggregate plan profit:

        profit = Σ_route [ R * Σ_v waste(v)  -  C * route_distance ]

    Non-mandatory nodes that have no profitable insertion option are silently
    dropped (consistent with the opt-in philosophy of VRPP).

    Args:
        routes: Partial plan (list of customer sequences, depot implicit).
        removed_nodes: Customer indices to be re-inserted.
        dist_matrix: Square distance matrix (depot at index 0).
        wastes: Node demand lookup.
        capacity: Maximum vehicle load.
        R: Revenue per unit of waste collected.
        C: Cost per unit of distance.
        max_discrepancy: LDS discrepancy budget.  Values of 1–3 are practical
            for ALNS repair calls.  Defaults to 2.
        mandatory_nodes: Nodes that must appear in the final plan.
        expand_pool: If True, all unvisited nodes are candidates (VRPP
            exploration mode).  Defaults to False (strict ALNS mode).
        seed_hurdle_factor: Fraction of new-route round-trip cost used as the
            minimum acceptable profit for opening a new route.  A value of 0.5
            means the node's profit must cover at least 50 % of its travel
            cost.  Defaults to 0.5 (matching other operators in the package).

    Returns:
        List[List[int]]: Updated routes after profit-maximising insertion.
            Economically unviable routes (negative profit, no mandatory nodes)
            are pruned from the final result.

    Raises:
        ValueError: If *max_discrepancy* is negative.
    """
    if max_discrepancy < 0:
        raise ValueError(f"max_discrepancy must be non-negative, got {max_discrepancy}")

    mandatory_set: Set[int] = set(mandatory_nodes) if mandatory_nodes else set()
    loads: List[float] = [sum(wastes.get(n, 0.0) for n in r) for r in routes]

    visited: Set[int] = {n for r in routes for n in r}
    if expand_pool:
        n_nodes = len(dist_matrix) - 1
        unrouted: List[int] = sorted(set(range(1, n_nodes + 1)) - visited)
    else:
        unrouted = sorted(set(removed_nodes) - visited)
        if mandatory_nodes:
            unrouted = sorted(set(unrouted) | (mandatory_set - visited))

    if not unrouted:
        return routes

    best_state: List[Optional[Tuple[float, _Plan, _Loads]]] = [None]

    _lds_reinsert_profit(
        routes,
        loads,
        unrouted,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        mandatory_set,
        max_discrepancy,
        best_state,
        seed_hurdle_factor,
    )

    if best_state[0] is not None:
        _, best_routes, _ = best_state[0]
        routes.clear()
        routes.extend([r for r in best_routes if r])
    else:
        # Fallback: mandatory nodes get their own route
        for orphan in unrouted:
            if orphan in mandatory_set:
                routes.append([orphan])

    # Remove economically unviable routes (no mandatory content)
    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_set)

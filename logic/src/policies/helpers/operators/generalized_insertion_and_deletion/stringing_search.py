"""Search and move application routines for stringing insertion operators.

This module provides the move evaluation and topological search routines for GENIUS stringing operators.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.stringing_i import (
    apply_type_i_s,
    apply_type_i_s_profit,
)
from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.stringing_ii import (
    apply_type_ii_s,
    apply_type_ii_s_profit,
)
from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.stringing_iii import (
    apply_type_iii_s,
    apply_type_iii_s_profit,
)
from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.stringing_iv import (
    apply_type_iv_s,
    apply_type_iv_s_profit,
)
from logic.src.utils.policy.neighborhood import get_p_neighborhood


def _evaluate_routes(routes: List[List[int]], dist_matrix: np.ndarray) -> float:
    """
    Evaluate total route cost.

    Args:
        routes: List of routes.
        dist_matrix: Distance matrix.

    Returns:
        float: Total negative cost.
    """
    cost = 0.0
    for rt in routes:
        if not rt:
            continue
        d = dist_matrix[0][rt[0]]
        for idx in range(len(rt) - 1):
            d += dist_matrix[rt[idx]][rt[idx + 1]]
        d += dist_matrix[rt[-1]][0]
        cost += d
    return -cost


def _apply_stringing_op(
    route: List[int],
    node: int,
    string_type: int,
    params: Tuple,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    profit_mode: bool,
    current_load: float,
) -> Tuple[List[int], float]:
    """
    Helper to apply stringing operation and get new route and immediate profit val.

    Args:
        route: Current route.
        node: Node to insert.
        string_type: Stringing type (1-4).
        params: Parameters for the move.
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        profit_mode: Whether to use profit-based moves.
        current_load: Current route load.

    Returns:
        Tuple[List[int], float]: (new_route, delta_profit).
    """
    if string_type in (2, 4):
        i, j, k, l = params
        if profit_mode:
            if string_type == 2:
                return apply_type_ii_s_profit(
                    route, node, i, j, k, l, dist_matrix, wastes, capacity, R, C, current_load
                )
            return apply_type_iv_s_profit(route, node, i, j, k, l, dist_matrix, wastes, capacity, R, C, current_load)
        new_route = (
            apply_type_ii_s(route, node, i, j, k, l, current_load)
            if string_type == 2
            else apply_type_iv_s(route, node, i, j, k, l, current_load)
        )
        return new_route, 0.0

    i, j, k = params
    if profit_mode:
        if string_type == 1:
            return apply_type_i_s_profit(route, node, i, j, k, dist_matrix, wastes, capacity, R, C, current_load)
        return apply_type_iii_s_profit(route, node, i, j, k, dist_matrix, wastes, capacity, R, C, current_load)
    new_route = (
        apply_type_i_s(route, node, i, j, k, current_load)
        if string_type == 1
        else apply_type_iii_s(route, node, i, j, k, current_load)
    )
    return new_route, 0.0


def _try_string_insertion(
    routes: List[List[int]],
    node: int,
    r_idx: int,
    string_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    rng: random.Random,
    current_load: float,
    profit_mode: bool = False,
) -> Optional[Tuple[List[List[int]], float]]:
    """
    Try inserting a node using the specified stringing operator, returning the best result.

    Args:
        routes: Current routes.
        node: Node to insert.
        r_idx: Route index.
        string_type: Stringing type.
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        rng: Random generator.
        current_load: Current route load.
        profit_mode: Whether to use profit-based moves.

    Returns:
        Optional[Tuple[List[List[int]], float]]: (best_routes, best_value) if successful.
    """
    route = routes[r_idx]
    if len(route) < 3:
        return None

    best_routes = None
    best_value = float("-inf")

    for _ in range(min(5, len(route))):
        valid_positions = list(range(len(route)))
        if len(valid_positions) < (4 if string_type in (2, 4) else 3):
            continue

        try:
            i = rng.choice(valid_positions)
            valid_j = [p for p in valid_positions if p != i]
            if not valid_j:
                continue
            j = rng.choice(valid_j)
            valid_k = [p for p in valid_positions if p not in (i, j)]
            if not valid_k:
                continue
            k = rng.choice(valid_k)

            if string_type in (2, 4):
                valid_l = [p for p in valid_positions if p not in (i, j, k)]
                if not valid_l:
                    continue
                l = rng.choice(valid_l)
                params: Tuple[int, ...] = (i, j, k, l)
            else:
                params = (i, j, k)  # type: ignore[assignment]

            new_route, val = _apply_stringing_op(
                route, node, string_type, params, dist_matrix, wastes, capacity, R, C, profit_mode, current_load
            )

            # Validate insertion (capacity and existence)
            # Optimization: since we do early pruning for capacity, we don't need the O(N) sum() here
            # anymore, stringing_wrapper enforces the check before trying the topological strings.
            if node not in new_route:
                continue

            test_routes = [list(r) for r in routes]
            test_routes[r_idx] = new_route

            if not profit_mode:
                val = _evaluate_routes(test_routes, dist_matrix)

            if val > best_value:
                best_value = val
                best_routes = test_routes

        except Exception:
            continue

    return (best_routes, best_value) if best_routes is not None else None


def _try_string_insertion_deterministic(  # noqa: C901
    routes: List[List[int]],
    node: int,
    r_idx: int,
    string_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    current_load: float,
    neighborhood_size: int,
    profit_mode: bool = False,
) -> Optional[Tuple[List[List[int]], float]]:
    """
    Try inserting a node using deterministic p-neighborhood search (bidirectional).

    Args:
        routes: Current routes.
        node: Node to insert.
        r_idx: Route index.
        string_type: Stringing type.
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        current_load: Current route load.
        neighborhood_size: Size of p-neighborhood.
        profit_mode: Whether to use profit-based moves.

    Returns:
        Optional[Tuple[List[List[int]], float]]: (best_routes, best_value) if successful.
    """
    route = routes[r_idx]
    if len(route) < 3:
        return None

    best_routes = None
    best_value = float("-inf")

    # Evaluate both orientations
    for target_route in [route, route[::-1]]:
        # Get p-neighborhood of the target node (u)
        p_neighbors_u = get_p_neighborhood(node, target_route, dist_matrix, neighborhood_size)

        for v_i in p_neighbors_u:
            i = target_route.index(v_i)
            v_ip1 = target_route[(i + 1) % len(target_route)]
            p_neighbors_ip1 = get_p_neighborhood(v_ip1, target_route, dist_matrix, neighborhood_size)

            for v_j in p_neighbors_u:
                j = target_route.index(v_j)
                if j == i:
                    continue
                v_jp1 = target_route[(j + 1) % len(target_route)]
                p_neighbors_jp1 = get_p_neighborhood(v_jp1, target_route, dist_matrix, neighborhood_size)

                try:
                    if string_type in (2, 4):
                        # Phase 3: Decouple neighborhoods
                        # v_k from N_p(v_{i+1})
                        for v_k in p_neighbors_ip1:
                            k = target_route.index(v_k)
                            if k in (i, j):
                                continue
                            # v_l from N_p(v_{j+1})
                            for v_l in p_neighbors_jp1:
                                l = target_route.index(v_l)
                                if l in (i, j, k):
                                    continue

                                params: Tuple[int, ...] = (i, j, k, l)
                                new_route, val = _apply_stringing_op(
                                    target_route,
                                    node,
                                    string_type,
                                    params,
                                    dist_matrix,
                                    wastes,
                                    capacity,
                                    R,
                                    C,
                                    profit_mode,
                                    current_load,
                                )
                                if node in new_route:
                                    test_routes = [list(r) for r in routes]
                                    test_routes[r_idx] = new_route
                                    if not profit_mode:
                                        val = _evaluate_routes(test_routes, dist_matrix)
                                    if val > best_value:
                                        best_value, best_routes = val, test_routes
                    else:
                        # Types I and III: v_k \in N_p(v_{i+1})
                        for v_k in p_neighbors_ip1:
                            k = target_route.index(v_k)
                            if k in (i, j):
                                continue

                            params = (i, j, k)  # type: ignore[assignment]
                            new_route, val = _apply_stringing_op(
                                target_route,
                                node,
                                string_type,
                                params,
                                dist_matrix,
                                wastes,
                                capacity,
                                R,
                                C,
                                profit_mode,
                                current_load,
                            )
                            if node in new_route:
                                test_routes = [list(r) for r in routes]
                                test_routes[r_idx] = new_route
                                if not profit_mode:
                                    val = _evaluate_routes(test_routes, dist_matrix)
                                if val > best_value:
                                    best_value, best_routes = val, test_routes
                except Exception:
                    continue

    return (best_routes, best_value) if best_routes is not None else None

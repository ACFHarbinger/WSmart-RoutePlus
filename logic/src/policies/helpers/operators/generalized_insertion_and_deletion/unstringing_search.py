"""Search and move application routines for unstringing removal operators.

This module provides the move evaluation and topological search routines for GENIUS unstringing operators.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.unstringing_i import (
    apply_type_i_us,
    apply_type_i_us_profit,
)
from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.unstringing_ii import (
    apply_type_ii_us,
    apply_type_ii_us_profit,
)
from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.unstringing_iii import (
    apply_type_iii_us,
    apply_type_iii_us_profit,
)
from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.unstringing_iv import (
    apply_type_iv_us,
    apply_type_iv_us_profit,
)
from logic.src.utils.policy.neighborhood import get_p_neighborhood


def _evaluate_routes(routes: List[List[int]], dist_matrix: np.ndarray) -> float:
    """
    Evaluate total route cost (negative for maximization).

    Args:
        routes: Current routes.
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


def _apply_unstring_op(
    route: List[int],
    unstring_type: int,
    params: Tuple,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
    profit_mode: bool,
) -> Tuple[List[int], float]:
    """
    Helper to apply unstringing operation and get new route and immediate profit val.

    Args:
        route: Current route.
        unstring_type: Unstringing type (1-4).
        params: Parameters for the move.
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        R: Revenue multiplier.
        C: Cost multiplier.
        profit_mode: Whether to use profit-based moves.

    Returns:
        Tuple[List[int], float]: (new_route, delta_profit).
    """
    if unstring_type in (2, 4):
        i, j, k, l = params
        if profit_mode:
            if unstring_type == 2:
                return apply_type_ii_us_profit(route, i, j, k, l, dist_matrix, wastes, R, C)
            return apply_type_iv_us_profit(route, i, j, k, l, dist_matrix, wastes, R, C)
        new_route = apply_type_ii_us(route, i, j, k, l) if unstring_type == 2 else apply_type_iv_us(route, i, j, k, l)
        return new_route, 0.0

    i, j, k = params
    if profit_mode:
        if unstring_type == 1:
            return apply_type_i_us_profit(route, i, j, k, dist_matrix, wastes, R, C)
        return apply_type_iii_us_profit(route, i, j, k, dist_matrix, wastes, R, C)
    new_route = apply_type_i_us(route, i, j, k) if unstring_type == 1 else apply_type_iii_us(route, i, j, k)
    return new_route, 0.0


def _try_unstring_removal(
    routes: List[List[int]],
    r_idx: int,
    n_idx: int,
    unstring_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
    rng: random.Random,
    profit_mode: bool = False,
) -> Optional[Tuple[List[List[int]], float]]:
    """
    Try removing a node using the specified unstringing operator (bidirectional).

    Args:
        routes: Current solution.
        r_idx: Route index.
        n_idx: Node index in route.
        unstring_type: Unstringing type.
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        R: Revenue multiplier.
        C: Cost multiplier.
        rng: Random generator.
        profit_mode: Profit mode flag.

    Returns:
        Optional[Tuple[List[List[int]], float]]: (best_routes, best_value) if successful.
    """
    route = routes[r_idx]
    if len(route) < 3:
        return None

    best_routes = None
    best_value = -float("inf")

    # Evaluate both orientations
    for target_route in [route, route[::-1]]:
        # Find index of target_node in target_route
        target_node = route[n_idx]
        try:
            curr_i = target_route.index(target_node)
        except ValueError:
            continue

        for _ in range(min(5, len(target_route))):
            valid_positions = [p for p in range(len(target_route)) if p != curr_i]

            try:
                if unstring_type in (1, 3):
                    if len(valid_positions) < 2:
                        continue
                    j = rng.choice(valid_positions)
                    valid_k = [p for p in valid_positions if p != j]
                    k = rng.choice(valid_k)
                    params: Tuple[int, ...] = (curr_i, j, k)
                else:
                    if len(valid_positions) < 3:
                        continue
                    j = rng.choice(valid_positions)
                    valid_k = [p for p in valid_positions if p != j]
                    k = rng.choice(valid_k)
                    valid_l = [p for p in valid_positions if p not in (j, k)]
                    l = rng.choice(valid_l)
                    params = (curr_i, j, k, l)  # type: ignore[assignment]

                new_route, val = _apply_unstring_op(
                    target_route, unstring_type, params, dist_matrix, wastes, R, C, profit_mode
                )

                if target_node in new_route:
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


def _try_unstring_removal_deterministic(  # noqa: C901
    routes: List[List[int]],
    r_idx: int,
    n_idx: int,
    unstring_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
    neighborhood_size: int,
    profit_mode: bool = False,
) -> Optional[Tuple[List[List[int]], float]]:
    """
    Try removing a node using deterministic p-neighborhood search (bidirectional).

    Args:
        routes: Current solution.
        r_idx: Route index.
        n_idx: Node index in route.
        unstring_type: Unstringing type.
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        R: Revenue multiplier.
        C: Cost multiplier.
        neighborhood_size: P-neighborhood size.
        profit_mode: Profit mode flag.

    Returns:
        Optional[Tuple[List[List[int]], float]]: (best_routes, best_value) if successful.
    """
    route = routes[r_idx]
    if len(route) < 3:
        return None

    best_routes = None
    best_value = -float("inf")

    # Evaluate both orientations
    for target_route in [route, route[::-1]]:
        target_node = route[n_idx]
        try:
            curr_i = target_route.index(target_node)
        except ValueError:
            continue

        v_i_minus_1 = 0 if curr_i == 0 else target_route[curr_i - 1]
        v_i_plus_1 = 0 if curr_i == len(target_route) - 1 else target_route[curr_i + 1]

        valid_positions = [p for p in range(len(target_route)) if p != curr_i]
        valid_nodes = [target_route[p] for p in valid_positions]

        p_neighbors_v_i_plus_1 = get_p_neighborhood(v_i_plus_1, valid_nodes, dist_matrix, neighborhood_size)
        p_neighbors_v_i_minus_1 = get_p_neighborhood(v_i_minus_1, valid_nodes, dist_matrix, neighborhood_size)

        for v_j in p_neighbors_v_i_plus_1:
            j = target_route.index(v_j)
            for v_k in p_neighbors_v_i_minus_1:
                k = target_route.index(v_k)
                if k in (curr_i, j):
                    continue

                try:
                    if unstring_type in (1, 3):
                        params: Tuple[int, ...] = (curr_i, j, k)
                        new_route, val = _apply_unstring_op(
                            target_route, unstring_type, params, dist_matrix, wastes, R, C, profit_mode
                        )
                        if target_node not in new_route:
                            test_routes = [list(r) for r in routes]
                            test_routes[r_idx] = new_route
                            if not profit_mode:
                                val = _evaluate_routes(test_routes, dist_matrix)
                            if val > best_value:
                                best_value, best_routes = val, test_routes
                    else:
                        v_k_plus_1 = 0 if k == len(target_route) - 1 else target_route[k + 1]
                        valid_l_nodes = [target_route[p] for p in valid_positions if p not in (j, k)]
                        p_neighbors_v_k_plus_1 = get_p_neighborhood(
                            v_k_plus_1, valid_l_nodes, dist_matrix, neighborhood_size
                        )

                        for v_l in p_neighbors_v_k_plus_1:
                            l = target_route.index(v_l)
                            params = (curr_i, j, k, l)  # type: ignore[assignment]
                            new_route, val = _apply_unstring_op(
                                target_route, unstring_type, params, dist_matrix, wastes, R, C, profit_mode
                            )
                            if target_node not in new_route:
                                test_routes = [list(r) for r in routes]
                                test_routes[r_idx] = new_route
                                if not profit_mode:
                                    val = _evaluate_routes(test_routes, dist_matrix)
                                if val > best_value:
                                    best_value, best_routes = val, test_routes
                except Exception:
                    continue

    return (best_routes, best_value) if best_routes is not None else None

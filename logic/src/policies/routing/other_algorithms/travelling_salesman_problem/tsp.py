"""
TSP Policy Module.

This module provides helper functions for solving single-vehicle routing problems
(TSP) including construction, local search optimization, and capacity management.
It typically wraps exact methods (like the bottom-up Held-Karp algorithm) or
(meta-)heuristics (like 2-opt and Stochastic Local Search).

Key Functions:
- find_route: Solve TSP using fast_tsp library
- get_multi_tour: Split single tour into multiple depot trips (capacity aware)
- get_route_cost: Calculate total tour distance
- get_partial_tour: Remove bins to meet capacity constraints
- dist_matrix_from_graph: Compute all-pairs shortest paths from NetworkX graph

Attributes:
    None

Example:
    >>> from logic.src.policies.routing.other_algorithms.travelling_salesman_problem.tsp import find_route
    >>> tour, length = find_route(dist_matrix)

Reference:
    Held, M., & Karp, R. M. (1961).
    "A dynamic programming approach to sequencing problems"
    Hoos, H. H., & Stützle, T. (2005).
    "Stochastic local search: Foundations and applications"
"""

import logging
from typing import List

import fast_tsp
import numpy as np

from logic.src.constants.routing import SCALE
from logic.src.policies.helpers.operators.intra_route_local_search.k_opt import directed_two_opt_route

from .two_opt import solve_tsp_2opt

logger = logging.getLogger(__name__)

# fast-tsp documents its distances as uint16; keep every scaled edge within that range.
_FAST_TSP_MAX_DIST = 65535


def find_route(C, to_collect, time_limit=2.0, seed=42, engine="fast_tsp"):
    """
    Find a TSP route through the depot and ``to_collect`` with fast_tsp.

    Args:
        C: Distance matrix (km).
        to_collect: Bin indices to visit.
        time_limit: fast_tsp search budget in seconds.
        seed: Accepted for API compatibility but not used: fast-tsp 0.1.5's
            find_tour has no seed and its time-budgeted search is not repeatable
            (different tours across runs on the same input).
        engine: Only "fast_tsp" is available in this export.

    Returns:
        List[int]: Tour starting and ending at depot. Format: [0, node1, node2, ..., 0].
        The fast_tsp order is driven in its cheaper direction on ``C`` and polished by a
        directed 2-opt, since fast_tsp treats ``C`` as symmetric. If fast_tsp fails, the
        input order ``[0, *to_collect, 0]`` is returned unchanged.
    """
    if engine == "custom":
        return solve_tsp_2opt(C, list(to_collect), depot=0)

    to_collect_tmp = [0] + list(to_collect)
    tmpC = C[to_collect_tmp, :][:, to_collect_tmp]
    # fast_tsp needs integer distances within uint16: use SCALE unless the longest edge would overflow.
    max_edge = float(np.max(tmpC)) if tmpC.size else 0.0
    scale = SCALE if max_edge * SCALE <= _FAST_TSP_MAX_DIST else _FAST_TSP_MAX_DIST / max_edge
    tmpC_int = np.round(tmpC * scale).astype(int)
    try:
        tour = fast_tsp.find_tour(tmpC_int, duration_seconds=time_limit)
    except Exception as exc:  # the improver must never lose a feasible trip
        logger.warning("fast_tsp failed (%s); keeping the input order", exc)
        return to_collect_tmp + [0]
    zero_index = tour.index(0)
    tour = tour[zero_index:] + tour[:zero_index]
    bins = [to_collect_tmp[i] for i in tour[1:]]
    # fast-tsp prices its tours as if distances were symmetric: for up to 20 nodes it
    # returns the optimum of the transposed matrix, and its local search ignores the
    # direction of reversed segments. Drive the cheaper direction on the true matrix,
    # then polish with a 2-opt that prices every reversal with the directed costs.
    if _directed_length(C, bins[::-1]) < _directed_length(C, bins):
        bins = bins[::-1]
    bins = directed_two_opt_route(bins, C)
    return [0, *bins, 0]


def _directed_length(C, bins):
    """Length of the depot-to-depot route through ``bins`` on the directed matrix ``C``."""
    path = [0, *bins, 0]
    return float(sum(C[a, b] for a, b in zip(path, path[1:], strict=False)))


def get_path_cost(G, p):
    """
    Calculate path cost in a NetworkX graph.

    Args:
        G: Graph with edge weights
        p: Path as sequence of node IDs

    Returns:
        float: Total path cost (sum of edge weights)
    """
    last_node = p[0]
    c = 0
    for id_i in range(1, len(p)):
        try:
            c += G.get_edge_data(last_node, p[id_i])["weight"]
        except Exception:
            c += 1
        last_node = p[id_i]
    return c


# Create matrix will all distances
def calculate_tour_cost(distance_matrix: np.ndarray, tour: List[int]) -> float:
    """
    Calculate the total distance of a given tour sequence.

    Useful for validating the combined cost of concatenated sector tours.

    Args:
        distance_matrix: NxN matrix of shortest path distances.
        tour: Sequence of node indices representing the path.

    Returns:
        float: Sum of distances between consecutive nodes in the tour.
    """
    cost = 0.0
    for i in range(len(tour) - 1):
        cost += distance_matrix[tour[i], tour[i + 1]]
    return float(cost)

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
    >>> from logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem.tsp import find_route
    >>> tour, length = find_route(dist_matrix)

Reference:
    Held, M., & Karp, R. M. (1961).
    "A dynamic programming approach to sequencing problems"
    Hoos, H. H., & Stützle, T. (2005).
    "Stochastic local search: Foundations and applications"
"""

import logging
from typing import List, Tuple

import fast_tsp
import networkx as nx
import numpy as np
from networkx.algorithms.shortest_paths.weighted import dijkstra_path

from logic.src.constants.routing import SCALE
from logic.src.utils.routing.tours import get_multi_tour, get_route_cost  # noqa: F401  (re-exported)

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
        If fast_tsp fails, the input order ``[0, *to_collect, 0]`` is returned unchanged.
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
    # cost = fast_tsp.compute_cost(tour, tmpC)
    tour2 = []
    for ii in range(0, len(tour) - 1):
        current_node = to_collect_tmp[tour[ii]]
        next_node = to_collect_tmp[tour[ii + 1]]
        tour2.append(current_node)
    tour2.extend([next_node, 0])
    return tour2


def get_path_cost(G, p):
    """
    Calculate path cost in a NetworkX graph.

    Args:
        G (networkx.Graph): Graph with edge weights
        p (List[int]): Path as sequence of node IDs

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


def get_partial_tour(
    tour: List[int],
    bins: np.ndarray,
    max_capacity: float,
    distance_matrix: np.ndarray,
    cost: float,
) -> Tuple[np.ndarray, float]:
    """
    Reduce a tour to fit within vehicle capacity by removing bins with minimal waste.

    Args:
        tour (List[int]): Current tour.
        bins (np.ndarray): Waste amounts for each bin.
        max_capacity (float): Vehicle capacity limit.
        distance_matrix (np.ndarray): Distance matrix.
        cost (float): Current routing cost.

    Returns:
        Tuple[np.ndarray, float]: (Reduced tour, updated cost).
    """
    tmp_tour = np.array([x - 1 for x in tour if x != 0])
    total_waste = np.sum(bins[tmp_tour])
    while total_waste > max_capacity:
        min_waste_bin_idx = np.argmin(bins[tmp_tour])
        bin_to_remove = tmp_tour[min_waste_bin_idx]
        total_waste -= bins[bin_to_remove]
        cost -= float(distance_matrix[tmp_tour[min_waste_bin_idx - 1], bin_to_remove])
        tmp_tour = np.delete(tmp_tour, min_waste_bin_idx)
    return tmp_tour, cost


# Create matrix will all distances
def dist_matrix_from_graph(G: nx.Graph) -> Tuple[np.ndarray, List[List[List[int]]]]:
    """
    Compute all-pairs shortest path distances and paths from a NetworkX graph.

    Args:
        G (nx.Graph): Input graph with nodes 0..N-1 and weighted edges.

    Returns:
        Tuple[np.ndarray, List[List[List[int]]]]: (Distance matrix, Path matrix).
            Distance matrix is N x N numpy array of shortest path lengths.
            Path matrix contains the sequence of nodes for each shortest path.
    """
    paths_between_states: List[List[List[int]]] = []
    n_vertices = len(G.nodes)
    dist_matrix = np.zeros((n_vertices, n_vertices), int)
    for id_i in range(n_vertices):
        paths_between_states.append([])
        for id_j in range(n_vertices):
            if id_i == id_j:
                paths_between_states[id_i].append([])
                continue
            p = dijkstra_path(G, source=id_i, target=id_j)
            paths_between_states[id_i].append(p)
            dist_matrix[id_i, id_j] = int(get_path_cost(G, p))
    return dist_matrix, paths_between_states


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

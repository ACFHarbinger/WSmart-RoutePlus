"""Local-search primitives shared by the TSP solvers in the cluster-first route-second algorithm.

These helpers were split out of ``tsp_metaheuristics``: tour-distance evaluation,
2-opt segment reversal, and the swap-sequence operators used by the PSO and GA
solvers.
"""

from typing import List, Tuple

import numpy as np


def _calculate_tour_distance(tour: List[int], distance_matrix: np.ndarray) -> float:
    """Calculate total distance of a TSP tour.

    Args:
        tour: Sequence of node indices representing the tour.
        distance_matrix: Distance matrix of shape (n_nodes, n_nodes).

    Returns:
        float: Sum of edge distances along the tour.
    """
    total = 0.0
    for i in range(len(tour) - 1):
        total += distance_matrix[tour[i], tour[i + 1]]
    return total


def _two_opt_swap(tour: List[int], i: int, k: int) -> List[int]:
    """
    Perform 2-opt swap on tour segment between indices i and k.

    2-opt removes two edges and reconnects the path by reversing the segment.
    This is the fundamental local search operator for TSP.

    Args:
        tour: Current tour (includes depot at start and end)
        i: Start index of segment to reverse
        k: End index of segment to reverse

    Returns:
        New tour with reversed segment
    """
    new_tour = tour[:i] + tour[i : k + 1][::-1] + tour[k + 1 :]
    return new_tour


def _get_swap_operators(tour1: List[int], tour2: List[int]) -> List[Tuple[int, int]]:
    """Compute sequence of swap operations to transform tour1 into tour2.

    Args:
        tour1: Source tour to transform.
        tour2: Target tour to reach.

    Returns:
        List[Tuple[int, int]]: Index-pair swaps that convert tour1 to tour2.
    """
    swaps = []
    t1 = list(tour1)
    for i in range(len(t1)):
        if t1[i] != tour2[i]:
            for j in range(i + 1, len(t1)):
                if t1[j] == tour2[i]:
                    swaps.append((i, j))
                    t1[i], t1[j] = t1[j], t1[i]
                    break
    return swaps

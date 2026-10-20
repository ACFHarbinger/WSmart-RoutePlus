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


def _apply_swap_sequence(tour: List[int], swap_ops: List[Tuple[int, int]], max_swaps: int = 3) -> List[int]:
    """
    Apply a sequence of swap operations to move toward a target tour.

    This implements the discrete PSO velocity update mechanism by
    probabilistically applying swaps that move the current particle
    toward its personal best and the global best.

    Args:
        tour: Current tour
        swap_ops: List of (i, j) swap operations
        max_swaps: Maximum number of swaps to apply

    Returns:
        Modified tour
    """
    result = tour.copy()
    num_swaps = min(len(swap_ops), max_swaps)

    for i in range(num_swaps):
        idx1, idx2 = swap_ops[i]
        if 0 < idx1 < len(result) - 1 and 0 < idx2 < len(result) - 1:
            result[idx1], result[idx2] = result[idx2], result[idx1]

    return result


def _get_swap_sequence(source: List[int], target: List[int]) -> List[Tuple[int, int]]:
    """
    Compute sequence of swap operations to transform source tour into target tour.

    This is used to compute the "velocity" in discrete PSO by finding
    the swaps needed to move from current position to pbest/gbest.

    Args:
        source: Current tour
        target: Target tour

    Returns:
        List of (i, j) index pairs to swap
    """
    swaps = []
    temp = source.copy()

    for i in range(1, len(temp) - 1):  # Exclude depot at start and end
        if temp[i] != target[i]:
            # Find where target[i] is in temp
            for j in range(i + 1, len(temp) - 1):
                if temp[j] == target[i]:
                    swaps.append((i, j))
                    temp[i], temp[j] = temp[j], temp[i]
                    break

    return swaps


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


def _apply_2opt_local_search(
    tour: List[int],
    distance_matrix: np.ndarray,
    n_customers: int,
    max_swaps: int = 10,
) -> List[int]:
    """Apply random 2-opt swaps to improve a tour.

    Args:
        tour: Current tour including depot at start and end.
        distance_matrix: Distance matrix of shape (n_nodes, n_nodes).
        n_customers: Number of customer nodes (excluding depot).
        max_swaps: Maximum number of 2-opt swaps to attempt.

    Returns:
        List[int]: Improved tour after applying the best 2-opt moves found.
    """
    best_tour = tour.copy()
    best_dist = _calculate_tour_distance(best_tour, distance_matrix)

    # Try a limited number of 2-opt swaps
    for _ in range(min(max_swaps, n_customers)):
        i = np.random.randint(1, n_customers)
        k = np.random.randint(i + 1, n_customers + 1)
        new_tour = _two_opt_swap(best_tour, i, k)
        new_dist = _calculate_tour_distance(new_tour, distance_matrix)

        if new_dist < best_dist:
            best_tour = new_tour
            best_dist = new_dist

    return best_tour

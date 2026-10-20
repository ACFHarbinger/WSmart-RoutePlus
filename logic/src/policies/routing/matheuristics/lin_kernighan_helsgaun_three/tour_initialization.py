"""Tour construction, perturbation and topology factory for LKH-3.

Holds the starting-tour construction, the double-bridge kick, the
successor-array helper and the k-opt topology factory, split out of
``tour_construction``.
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.perturbation_shaking.double_bridge import double_bridge
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.tour_adapter import TourAdapter


def _double_bridge_kick(
    tour: List[int],
    distance_matrix: np.ndarray,
    rng: Random,
) -> List[int]:
    """
    Apply a double-bridge (random 4-opt) perturbation via the shared operator.

    Delegates to :func:`double_bridge` from
    ``logic.src.policies.helpers.operators.perturbation.double_bridge``, which
    slices the route at three random cut points and reconnects as A+C+B+D.

    A :class:`TourAdapter` bridges the flat tour representation used by
    this solver with the ``ls``-style interface expected by the operator.

    Args:
        tour: Current closed tour (first node repeated at end).
        distance_matrix: (n × n) cost matrix (forwarded to the adapter).
        rng: ``random.Random`` instance for reproducible cut selection.

    Returns:
        List[int]: New perturbed closed tour, or original if too short for move.
    """
    adapter = TourAdapter(tour, distance_matrix)
    applied = double_bridge(adapter, r_idx=0, rng=rng)
    if not applied:
        return tour
    return adapter.to_closed_tour()


def _initialize_tour(
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]],
) -> List[int]:
    """
    Build or validate the starting tour.

    If no tour is provided a nearest-neighbour construction heuristic is used
    (Helsgaun 2000, Section 4.4).  The returned tour is always closed (first
    node repeated at end).

    Args:
        distance_matrix: (n × n) cost matrix.
        initial_tour: Optional existing tour.

    Returns:
        List[int]: A closed tour over all n nodes.
    """
    n = len(distance_matrix)
    if initial_tour is None:
        curr = 0
        path = [0]
        unvisited = set(range(1, n))
        while unvisited:
            next_node = min(unvisited, key=lambda x: distance_matrix[curr, x])
            path.append(next_node)
            unvisited.remove(next_node)
            curr = next_node
        path.append(0)
        return path

    else:
        t = initial_tour[:]
        if t[0] != t[-1]:
            t.append(t[0])
        return t


def _get_tour_array(start_node: int, tour_dict: Dict[int, int]) -> List[int]:
    """Retrieve the tour as a list starting from `start_node`.

    Args:
        start_node: Starting node.
        tour_dict: Tour dictionary.

    Returns:
        List[int]: Tour as ordered node list.
    """
    tour = [start_node]
    curr = tour_dict[start_node]
    while curr != start_node:
        tour.append(curr)
        curr = tour_dict[curr]
    return tour


class KoptTopologyFactory:
    """
    Programmatic generator for exhaustive non-sequential k-opt topologies.
    Memoizes the exact valid Hamiltonian reconnections to avoid hardcoding.

    Attributes:
        _cache: Cache for storing valid topologies.
    """

    _cache: Dict[int, List[List[Tuple[int, int]]]] = {}

    @classmethod
    def get_topologies(cls, k: int) -> List[List[Tuple[int, int]]]:  # noqa: C901
        """
        Get all valid added-edge topologies for a given k.
        Returns a list of topologies. Each topology is a list of exactly k edge tuples.
        The tuples contain 0-based indices into the broken nodes array (t1..t2k).

        Matches LKH pure non-sequential counts:
        - 2-opt: 1
        - 3-opt: 7 (including 2-opt sub-moves as requested)
        - 4-opt: 25 pure
        - 5-opt: exhaustive pure valid Hamiltonian reconnections

        Args:
            k: Number of edges to remove (k-opt).

        Returns:
            List[List[Tuple[int, int]]]: Valid added-edge topologies for k-opt.

        Yields:
            Valid topologies.
        """
        if k in cls._cache:
            return cls._cache[k]

        endpoints = list(range(2 * k))
        # Segments: (t_2, t_3), (t_4, t_5), ..., (t_2k, t_1) -> indices: (1, 2), (3, 4), ..., (2k-1, 0)
        segments = [(2 * i + 1, (2 * i + 2) % (2 * k)) for i in range(k)]
        broken = set()
        for i in range(k):
            u, v = 2 * i, 2 * i + 1
            broken.add((u, v))
            broken.add((v, u))

        def find_matchings(unmatched: List[int]):
            """Generates all possible perfect matchings for the given nodes.

            Args:
                unmatched: List of nodes to match.

            Yields:
                Perfect matchings.
            """
            if not unmatched:
                yield []
            else:
                u = unmatched[0]
                for i in range(1, len(unmatched)):
                    v = unmatched[i]
                    for m in find_matchings(unmatched[1:i] + unmatched[i + 1 :]):
                        yield [(u, v)] + m

        valid_topologies = []
        for m in find_matchings(endpoints):
            # Exclude the original configuration (where all added edges are broken edges)
            is_original = True
            for u, v in m:
                if (u, v) in broken:
                    is_original = False
                    break
            if is_original:
                continue

            # Build adjacency list
            adj: Dict[int, List[int]] = {v: [] for v in endpoints}
            for u, v in segments:
                adj[u].append(v)
                adj[v].append(u)
            for u, v in m:
                adj[u].append(v)
                adj[v].append(u)

            # Check for a single Hamiltonian cycle
            visited = set()
            curr = 0
            while curr not in visited:
                visited.add(curr)
                nxt = None
                for n in adj[curr]:
                    if n not in visited:
                        nxt = n
                        break
                if nxt is None:
                    break
                curr = nxt

            if len(visited) == 2 * k:
                # Filter pure moves for k >= 4
                pure = True
                if k >= 4:
                    for u, v in m:
                        if (u, v) in broken:
                            pure = False
                            break
                if pure:
                    valid_topologies.append(m)

        cls._cache[k] = valid_topologies
        return valid_topologies

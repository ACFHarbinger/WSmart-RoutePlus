"""Tour construction, perturbation and topology factory for LKH-3.

Holds the starting-tour construction, the double-bridge kick, the
successor-array helper and the k-opt topology factory, split out of
``tour_construction``.
"""

from __future__ import annotations

from random import Random
from typing import List, Optional

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

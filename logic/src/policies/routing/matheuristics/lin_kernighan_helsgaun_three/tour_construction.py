"""
LKH-3 Gain-Computation and Segment-Building Module.

Provides exact gain formulae for 2-, 3-, 4-, and 5-opt moves together with
the k-opt acceptance gate and the segment-reconstruction helper used to
materialise accepted moves.

Exact gain functions
--------------------
The gain functions compute the net reduction in tour length if the named
edges are removed and the segments are reconnected in the specified pattern.
A positive value means the move is improving.

_2opt_gain(t1, t2, t3, t4, d) -> float
    Gain for removing (t1,t2) and (t3,t4):
    ``c(t1,t2) + c(t3,t4) − c(t1,t3) − c(t2,t4)``.

_3opt_gains(t1, t2, t3, t4, t5, t6, d) -> List[float]
    Gains for all seven non-trivial 3-opt reconnection patterns (cases 0–6).

_4opt_gains(t1, …, t8, d) -> List[float]
    Gains for three common 4-opt reconnection patterns (cases 0–2).

_5opt_gains(t1, …, t10, d) -> List[float]
    Gains for five common 5-opt reconnection patterns (cases 0–4), covering
    the sequential 5-opt moves central to Helsgaun (2000), Section 4.3.

The tour initialisation, double-bridge perturbation and topology factory moved
to :mod:`tour_initialization`; the elite-tour merging routines moved to
:mod:`tour_merging`. All of them are re-imported here so every existing
``from tour_construction import ...`` keeps working.

Attributes:
    KoptTopologyFactory: Factory for generating k-opt topologies.
    merge_tours: Helper function for merging tours.
    merge_tours_ip: Helper function for merging tours using IP.
    merge_tours_best: Helper function for merging tours using best-first search.
    build_tour_from_segments: Helper function for building tours from segments.

Example:
    >>> from logic.src.policies.helpers.operators.heuristics._tour_construction import (  # noqa: E402
    ...     _initialize_tour, _double_bridge_kick, _2opt_gain, _3opt_gains
    ... )
    >>> tour = _initialize_tour(dist, initial_tour=None)
    >>> gain = _2opt_gain(t1, t2, t3, t4, dist)
    >>> if gain > 1e-6:
    ...     tour = _double_bridge_kick(tour, dist, rng)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.tour_initialization import (
    KoptTopologyFactory as KoptTopologyFactory,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.tour_initialization import (
    _double_bridge_kick as _double_bridge_kick,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.tour_initialization import (
    _get_tour_array as _get_tour_array,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.tour_initialization import (
    _initialize_tour as _initialize_tour,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.tour_merging import (
    merge_tours as merge_tours,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.tour_merging import (
    merge_tours_best as merge_tours_best,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.tour_merging import (
    merge_tours_ip as merge_tours_ip,
)


def build_tour_from_segments(
    curr_tour: List[int],
    pos: List[int],
    added_edges: List[Tuple[int, int]],
    k: int,
) -> List[int]:
    """
    Constructs the exact topological candidate bypassing heuristic operator ambiguity.

    Given k broken edges (t1,t2), (t3,t4) ... defined by their indices in `pos`,
    extracts the k segments and connects them according to `added_edges`.

    Args:
        curr_tour: The original closed tour (length N+1, starts/ends with 0).
        pos: List of indices in curr_tour for [t1, t2, ..., t2k].
        added_edges: Matching of endpoints [0..2k-1] defining the reconnection.
        k: The k-opt order.

    Returns:
        List[int]: The new closed tour array.
    """
    # 1. Extract the k segments
    # S_i goes from t_{2i+1} to t_{2i+2} (0-indexed pos: 2i+1 to (2i+2)%2k)
    segments = []
    for i in range(k):
        start_idx = pos[2 * i + 1]
        end_idx = pos[(2 * i + 2) % (2 * k)]

        if start_idx <= end_idx:
            seg = curr_tour[start_idx : end_idx + 1]
        else:
            # Wrap around (only possible for the last segment t_2k -> t_1)
            seg = curr_tour[start_idx:-1] + curr_tour[: end_idx + 1]
        segments.append(seg)

    # Mapping endpoint -> (segment_idx, is_start)
    endpoint_to_seg = {}
    for i in range(k):
        endpoint_to_seg[2 * i + 1] = (i, True)
        endpoint_to_seg[(2 * i + 2) % (2 * k)] = (i, False)

    # Build fast adjacency for added edges
    adj = {}
    for u, v in added_edges:
        adj[u] = v
        adj[v] = u

    # 2. Trace the single cycle starting from the depot (node 0)
    # The depot is ALWAYS in the last segment (k-1) if t1..t2k follows order
    # but we can just find which segment contains the first node of curr_tour.
    # Actually, segments[k-1] wraps around, so it contains 0!
    # Wait, the first element of curr_tour is 0, so 0 is in the segment that wraps around.

    new_tour_open = []

    # We start traversal from endpoint 0 (t1, end of segment k-1).
    # Wait, the new tour must start and end with 0! It's easier to build any
    # valid cyclic array, then rotate it so 0 is at the front.

    curr_endpoint = 0
    for _ in range(k):
        # We are at curr_endpoint (which is the end of some segment).
        # We cross the added edge to next_endpoint
        next_endpoint = adj[curr_endpoint]

        seg_idx, is_start = endpoint_to_seg[next_endpoint]
        seg_nodes = segments[seg_idx]

        if is_start:
            # Traverse forward
            new_tour_open.extend(seg_nodes)
            curr_endpoint = (2 * seg_idx + 2) % (2 * k)  # arrive at end of segment
        else:
            # Traverse backward
            new_tour_open.extend(reversed(seg_nodes))
            curr_endpoint = 2 * seg_idx + 1  # arrive at start of segment

    # Remove consecutive duplicates that occur from gluing
    cleaned: List[int] = []
    for node in new_tour_open:
        if not cleaned or cleaned[-1] != node:
            cleaned.append(node)
    if cleaned and cleaned[-1] == cleaned[0]:
        cleaned.pop()

    # Rotate so 0 is first
    zero_idx = cleaned.index(0)
    rotated = cleaned[zero_idx:] + cleaned[:zero_idx] + [0]
    return rotated


# ---------------------------------------------------------------------------
# Exact gain computations for each k-opt level
# ---------------------------------------------------------------------------


def _2opt_gain(
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    d: np.ndarray,
) -> float:
    """
    Exact distance gain for a 2-opt move that removes (t1,t2) and (t3,t4).

    Gain = c(t1,t2) + c(t3,t4) − c(t1,t3) − c(t2,t4).

    A positive gain means the 2-opt move strictly reduces tour length.

    NOTE: For lexicographic objectives, this function only computes ΔC (cost change).
    The caller must separately evaluate ΔP (penalty change) to determine if the
    move should be accepted. See _should_accept_kopt_move() for the full logic.

    Args:
        t1: Node 1.
        t2: Node 2.
        t3: Node 3.
        t4: Node 4.
        d: Distance matrix.

    Returns:
        float: Exact distance gain for a 2-opt move.
    """
    return d[t1, t2] + d[t3, t4] - d[t1, t3] - d[t2, t4]


def _should_accept_kopt_move(
    delta_penalty: float,
    delta_cost: float,
) -> bool:
    """
    Lexicographic pre-screening for k-opt moves.

    A k-opt move should be passed to the operator for full evaluation if:
    1. It reduces capacity violations (ΔP < 0), OR
    2. It maintains feasibility (ΔP == 0) AND improves cost (ΔC > 0)

    This allows the search to transit through infeasible space by accepting
    moves that reduce penalty, even if they increase routing cost.

    Args:
        delta_penalty: Change in capacity penalty (P_new - P_old).
                       Negative values mean reduced violations.
        delta_cost: Change in routing cost (C_new - C_old).
                    Positive values mean cost reduction.

    Returns:
        bool: True if the move should be accepted for evaluation, False otherwise.

    Mathematical Formulation:
        Accept iff: (ΔP < -ε) OR (|ΔP| ≤ ε AND ΔC > ε)
        where ε = 1e-6 is numerical tolerance.

    Examples:
        >>> # Case 1: Reduces penalty (even if cost increases)
        >>> _should_accept_kopt_move(delta_penalty=-5.0, delta_cost=-2.0)
        True
        >>>
        >>> # Case 2: Maintains feasibility and improves cost
        >>> _should_accept_kopt_move(delta_penalty=0.0, delta_cost=3.0)
        True
        >>>
        >>> # Case 3: Increases penalty (blocked)
        >>> _should_accept_kopt_move(delta_penalty=2.0, delta_cost=5.0)
        False
        >>>
        >>> # Case 4: No change (blocked)
        >>> _should_accept_kopt_move(delta_penalty=0.0, delta_cost=0.0)
        False
    """
    eps = 1e-6

    # Condition 1: Reduces capacity violation
    if delta_penalty < -eps:
        return True

    # Condition 2: Maintains feasibility AND improves cost
    return bool(abs(delta_penalty) <= eps and delta_cost > eps)


def _3opt_gains(
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    t5: int,
    t6: int,
    d: np.ndarray,
) -> List[float]:
    """
    Exact gains for all seven non-trivial 3-opt reconnection cases.

    Removes edges (t1,t2), (t3,t4), (t5,t6).  The original tour cost
    contribution is d(t1,t2) + d(t3,t4) + d(t5,t6).

    Returns a list of seven gain values corresponding to cases 0–6 of
    ``move_kopt_intra`` (k=3).

    Gain > 0 means an improving move for that case.

    NOTE: Cases 1 and 2 are pure 2-opt sub-moves (single segment reversal).
    The calling code in _try_3opt_move deliberately excludes them because
    the preceding _try_2opt_move search has already proven them unprofitable
    for the current (i, j) pair.

    Args:
        t1: Node 1.
        t2: Node 2.
        t3: Node 3.
        t4: Node 4.
        t5: Node 5.
        t6: Node 6.
        d: Distance matrix.

    Returns:
        List[float]: Exact gains for all seven non-trivial 3-opt reconnection cases.
    """
    base = d[t1, t2] + d[t3, t4] + d[t5, t6]
    return [
        # case 0: A + B' + C' + D  →  t1-t3, t2-t5, t4-t6
        base - (d[t1, t3] + d[t2, t5] + d[t4, t6]),
        # case 1: A + B  + C' + D  →  pure reversal of C (2-opt sub-move)
        base - (d[t1, t2] + d[t3, t5] + d[t4, t6]),
        # case 2: A + B' + C  + D  →  pure reversal of B (2-opt sub-move)
        base - (d[t1, t3] + d[t2, t4] + d[t5, t6]),
        # case 3: A + C  + B  + D  →  t1-t4, t5-t2, t3-t6
        base - (d[t1, t4] + d[t5, t2] + d[t3, t6]),
        # case 4: A + C  + B' + D  →  t1-t4, t5-t3, t2-t6
        base - (d[t1, t4] + d[t5, t3] + d[t2, t6]),
        # case 5: A + C' + B  + D  →  t1-t5, t4-t2, t3-t6
        base - (d[t1, t5] + d[t4, t2] + d[t3, t6]),
        # case 6: A + C' + B' + D  →  t1-t5, t4-t3, t2-t6
        base - (d[t1, t5] + d[t4, t3] + d[t2, t6]),
    ]


def _4opt_gains(
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    t5: int,
    t6: int,
    t7: int,
    t8: int,
    d: np.ndarray,
) -> List[float]:
    """
    Exact gains for three common 4-opt reconnection cases.

    Removes edges (t1,t2), (t3,t4), (t5,t6), (t7,t8).

    Returns a list of three gain values (indices 0–2).

    Args:
        t1: Node 1.
        t2: Node 2.
        t3: Node 3.
        t4: Node 4.
        t5: Node 5.
        t6: Node 6.
        t7: Node 7.
        t8: Node 8.
        d: Distance matrix.

    Returns:
        List[float]: Exact gains for the three 4-opt reconnection cases.
    """
    base = d[t1, t2] + d[t3, t4] + d[t5, t6] + d[t7, t8]
    return [
        # case 0: double-bridge  A+C+B+D+E  →  t1-t5, t6-t3, t4-t7, t2-t8
        base - (d[t1, t5] + d[t6, t3] + d[t4, t7] + d[t2, t8]),
        # case 1: reverse B and D  →  t1-t3, t2-t4, t5-t7, t6-t8
        base - (d[t1, t3] + d[t2, t4] + d[t5, t7] + d[t6, t8]),
        # case 2: swap B and D  →  t1-t7, t8-t5, t6-t3, t4-t2
        base - (d[t1, t7] + d[t8, t5] + d[t6, t3] + d[t4, t2]),
    ]


def _5opt_gains(
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    t5: int,
    t6: int,
    t7: int,
    t8: int,
    t9: int,
    t10: int,
    d: np.ndarray,
) -> List[float]:
    """
    Exact gains for five representative 5-opt reconnection cases.

    Removes edges (t1,t2), (t3,t4), (t5,t6), (t7,t8), (t9,t10).

    Returns a list of five gain values (indices 0–4).  These cover the
    sequential 5-opt moves central to Helsgaun (2000), Section 4.3.
    The full 5-opt exhaustive enumeration has 60 non-trivial cases;
    only these 5 representative patterns are evaluated for efficiency.

    Args:
        t1: Node 1.
        t2: Node 2.
        t3: Node 3.
        t4: Node 4.
        t5: Node 5.
        t6: Node 6.
        t7: Node 7.
        t8: Node 8.
        t9: Node 9.
        t10: Node 10.
        d: Distance matrix.

    Returns:
        List[float]: Exact gains for the five 5-opt reconnection cases.
    """
    base = d[t1, t2] + d[t3, t4] + d[t5, t6] + d[t7, t8] + d[t9, t10]
    return [
        # case 0: A + B' + C + D' + E + F
        base - (d[t1, t3] + d[t2, t4] + d[t5, t7] + d[t6, t8] + d[t9, t10]),
        # case 1: A + C + B' + D + E + F
        base - (d[t1, t5] + d[t6, t3] + d[t4, t2] + d[t7, t8] + d[t9, t10]),
        # case 2: A + B + D + C' + E + F
        base - (d[t1, t2] + d[t3, t7] + d[t8, t5] + d[t6, t4] + d[t9, t10]),
        # case 3: A + C + D + B + E + F
        base - (d[t1, t5] + d[t6, t7] + d[t8, t3] + d[t4, t2] + d[t9, t10]),
        # case 4: A + D + B' + C + E + F
        base - (d[t1, t7] + d[t8, t3] + d[t4, t5] + d[t6, t2] + d[t9, t10]),
    ]

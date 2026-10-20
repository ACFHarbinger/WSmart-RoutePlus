"""
Lin-Kernighan Heuristic (1973) — Full Implementation.

Implements the original variable-depth sequential k-opt local search of
Lin & Kernighan (1973) with the following properties faithful to the paper:

1. **Gain criterion** (Section 4, LK 1973):
   The partial gain G_i = Σ_{r=1}^{i} [c(t_{2r-1}, t_{2r}) - c(t_{2r}, t_{2r+1})]
   must remain strictly positive at every level. This prunes the search tree
   aggressively without sacrificing solution quality.

2. **Sequential constraint** (Section 3, LK 1973):
   The edge (t_{2i-1}, t_{2i}) removed at step i must be incident to t_{2i-1}
   in the *current* tour, and t_{2i} must be chosen so that closing the chain
   (t_{2i}, t_1) yields a valid (non-degenerate) tour.

3. **Candidate list restriction** (Section 5, LK 1973):
   The node t_{2i+1} — the endpoint of the added edge — is restricted to the
   α-nearest neighbours of t_{2i}. This reduces worst-case complexity from
   O(n^k) to O(n · |cand|^{k-1}) per pass.

4. **Backtracking** (Section 4, LK 1973):
   If no improving closing edge is found at depth i, all valid choices of
   t_{2i+1} are tried before abandoning t_1. The search returns the first
   improvement found at any depth (first-improvement strategy).

5. **Don't-look bits** (LKH extension; Helsgaun 2000 §5.3):
   Nodes whose neighbourhood yielded no improvement are masked. A bit is
   cleared whenever an adjacent move is accepted.

Architecture
------------
The implementation is self-contained but reuses:

- :func:`compute_alpha_measures` and :func:`get_candidate_set` from
  `lin_kernighan_helsgaun.py` for α-nearest-neighbour candidate lists.
- :func:`_initialize_tour` from `_tour_construction.py` for the starting
  nearest-neighbour tour.
- :func:`get_cost`, :func:`is_better` from `_objective.py`.
- :func:`_double_bridge_kick` from `_tour_construction.py` for ILS
  perturbation (the outer restart loop follows the same ILS pattern as
  `solve_lkh` in `lin_kernighan_helsgaun.py`).

It does NOT use `move_kopt_intra` / `TourAdapter` internally because the
sequential search requires direct manipulation of position arrays, which is
incompatible with the operator's route-list interface. Segment reversals are
performed with in-place slice reversal on the position-indexed tour array.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.search_heuristics.lin_kernighan import solve_lk
    >>> tour, cost = solve_lk(dist_matrix)

References:
----------
Lin, S., & Kernighan, B. W. (1973). An effective heuristic algorithm for the
  travelling-salesman problem. Operations Research, 21(2), 498–516.

Helsgaun, K. (2000). An effective implementation of the Lin-Kernighan
  travelling salesman heuristic. EJOR 126, 106–130.
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.search_heuristics._objective import (
    get_cost,
    is_better,
)
from logic.src.policies.helpers.operators.search_heuristics._tour_construction import (
    _double_bridge_kick,
    _initialize_tour,
)
from logic.src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun import (
    compute_alpha_measures,
    get_candidate_set,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .lin_kernighan_search import (
    _apply_2opt_positions as _apply_2opt_positions,
)
from .lin_kernighan_search import (
    _build_pos_map as _build_pos_map,
)
from .lin_kernighan_search import (
    _lk_extend as _lk_extend,
)
from .lin_kernighan_search import (
    _lk_search as _lk_search,
)
from .lin_kernighan_search import (
    _pred as _pred,
)
from .lin_kernighan_search import (
    _succ as _succ,
)

__all__ = [
    "solve_lk",
    "_build_pos_map",
    "_apply_2opt_positions",
    "_succ",
    "_pred",
    "_lk_search",
    "_lk_extend",
    "_improve_tour_lk",
]

# ---------------------------------------------------------------------------
# One improvement pass over the full tour
# ---------------------------------------------------------------------------


def _improve_tour_lk(
    tour: List[int],
    cost: float,
    candidates: Dict[int, List[int]],
    d: np.ndarray,
    dont_look: np.ndarray,
    max_depth: int,
) -> Tuple[List[int], float, bool, np.ndarray]:
    """
    Execute one complete sequential LK improvement pass over all nodes.

    Iterates over every node t1 (skipping those masked by don't-look bits)
    and calls :func:`_lk_search` to find an improving move. Returns after
    the first improvement found (first-improvement strategy).

    Don't-look bit management:
    - Bit set to True when _lk_search finds no improvement for t1.
    - Bit set to False for all tour-neighbours of any improved node.

    Args:
        tour:       Closed tour (length n+1).
        cost:       Current tour cost.
        candidates: α-candidate lists.
        d:          Distance matrix.
        dont_look:  Boolean array of length n (indexed by node).
        max_depth:  Sequential search depth limit (2..5).

    Returns:
        (new_closed_tour, new_cost, improved, dont_look)
    """
    open_tour = tour[:-1]  # work on the open prefix
    n = len(open_tour)
    pos = _build_pos_map(tour)
    improved = False

    for idx in range(n):
        t1 = open_tour[idx]
        if dont_look[t1]:
            continue

        result = _lk_search(t1, open_tour, pos, n, d, candidates, max_depth)
        if result is not None:
            new_open, new_pos, new_cost = result
            # Update don't-look bits for changed neighbourhood
            dont_look[t1] = False
            for nb_pos in (_succ(int(new_pos[t1]), n), _pred(int(new_pos[t1]), n)):
                dont_look[new_open[nb_pos]] = False

            open_tour = new_open
            pos = new_pos
            cost = new_cost
            improved = True
            # Restart from the beginning (first-improvement)
            break
        else:
            dont_look[t1] = True

    return open_tour + [open_tour[0]], cost, improved, dont_look


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------


def solve_lk(
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 50,
    max_depth: int = 5,
    n_candidates: int = 5,
    recorder: Optional[PolicyStateRecorder] = None,
    np_rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[List[int], float]:
    """
    Solve a TSP instance using the original Lin-Kernighan (1973) heuristic.

    The outer loop is an Iterated Local Search (ILS) with double-bridge
    perturbation (Helsgaun 2000, §5) to escape local optima.

    The inner loop is the variable-depth sequential LK search:
    (1) For each root t1, attempt a gain-criterion-guided sequential exchange
        up to depth ``max_depth``.
    (2) Repeat until no improving move exists (local optimum).
    (3) Perturb with double-bridge and re-run.

    Candidate lists are built via α-nearness (MST sensitivity), which is
    Helsgaun's key contribution over the original LK paper's simpler
    nearest-neighbour candidate lists.

    Args:
        distance_matrix: (n × n) symmetric cost matrix.
        initial_tour: Optional starting tour (closed).
            Nearest-neighbour is used when not provided.
        max_iterations: Maximum ILS perturbation restarts.
        max_depth: Maximum sequential search depth (2–5). Higher values
            find better solutions but scale as O(n · k^{depth-1}).
        n_candidates: α-nearest-neighbour candidate list size per node.
        recorder: Optional recorder for telemetry.
        np_rng: NumPy Generator (seeded from
            ``seed`` if not provided).
        seed: Alternative seed (overridden by ``np_rng`` if given).

    Returns:
        Tuple[List[int], float]: (best_tour, best_cost) — closed node sequence and
            its total length.
    """
    n = len(distance_matrix)
    if n < 3:
        t = list(range(n)) + [0]
        return t, float(get_cost(t, distance_matrix))

    if np_rng is None:
        np_rng = np.random.default_rng(seed if seed is not None else 42)
    stdlib_rng = Random(int(np_rng.integers(0, 2**31)))

    # 1. Initialisation
    curr_tour = _initialize_tour(distance_matrix, initial_tour)

    # 2. α-candidate sets (MST-based; identical to solve_lkh)
    alpha = compute_alpha_measures(distance_matrix)
    candidates = get_candidate_set(distance_matrix, alpha, max_candidates=n_candidates)

    curr_cost = get_cost(curr_tour, distance_matrix)
    best_tour = curr_tour[:]
    best_cost = curr_cost
    dont_look = np.zeros(n, dtype=bool)

    # 3. ILS outer loop
    for restart in range(max_iterations):
        # Inner LK passes until local optimum
        while True:
            curr_tour, curr_cost, improved, dont_look = _improve_tour_lk(
                curr_tour, curr_cost, candidates, distance_matrix, dont_look, max_depth
            )
            if not improved:
                break

        if is_better(curr_cost, best_cost):
            best_cost = curr_cost
            best_tour = curr_tour[:]

        if recorder is not None:
            recorder.record(engine="lk", restart=restart, best_cost=best_cost)

        # Perturbation: double-bridge kick
        curr_tour = _double_bridge_kick(best_tour, distance_matrix, stdlib_rng)
        curr_cost = get_cost(curr_tour, distance_matrix)
        dont_look.fill(False)

    return best_tour, best_cost

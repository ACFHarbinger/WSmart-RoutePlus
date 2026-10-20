"""
LKH-1 Solver — Full Implementation (Helsgaun 2000).

Implements the complete Lin-Kernighan-Helsgaun (LKH-1) algorithm as described
in Helsgaun (2000), including the components absent from the existing
``lin_kernighan_helsgaun.py::solve_lkh``:

1. **Alpha-measure candidate sets** (Section 4.1, Helsgaun 2000):
   Edges pruned using sensitivity analysis on minimum spanning 1-trees. Only
   α-nearest neighbours are considered for inclusion in the tour.

2. **Minimum 1-Tree (1T) computation** (Section 3, Helsgaun 2000):
   A 1-tree rooted at node 0 is the minimum spanning tree on nodes {1..n-1}
   plus the two shortest edges incident to node 0.  Its weight W(π) is a
   Held-Karp lower bound on the optimal tour length when the node penalties
   π are optimal (Held & Karp 1970).

3. **Subgradient optimisation of π** (Section 3.1, Helsgaun 2000):
   The penalty vector π ∈ ℝ^n is updated to maximise the 1-tree lower bound:

       W(π) = min-1-tree(c̃_{ij}) + 2 · Σ_i π_i
       c̃_{ij} = c_{ij} + π_i + π_j          (penalty-modified costs)
       g_i    = d_i(T) − 2                   (degree-defect subgradient)
       π ← π + t · g,   t = μ · (W* − W) / ||g||²  (Polyak step)

   where d_i(T) is the degree of node i in the 1-tree T.

4. **Penalty-modified candidate sets** (Section 4.1, Helsgaun 2000):
   The α-nearness and candidate lists are recomputed using c̃ rather than c
   after subgradient convergence.  This produces far tighter candidates than
   MST-only α-measures on the raw distances.

5. **Sequential k-opt moves (k = 2..5)** (Section 4.3):
   Exact gain computation for every reconnection case at each level, with
   early-termination via the positive-gain criterion.  Routines for higher-
   order moves (3 to 5-opt) are delegated to components in
   ``._tour_improvement``, with all final segment swaps executed via
   :func:`move_kopt_intra` from
   ``logic.src.policies.helpers.operators.intra_route.k_opt``.

6. **Candidate-set restricted search** (Section 3.2 / 4.1):
   Inner loops restricted to the α-nearest neighbours of each node, giving
   O(n · k_cand) search per node rather than O(n²).

7. **Don't-look bits** (Section 5.3):
   Nodes whose neighbourhood yielded no improvement are skipped until a
   neighbouring move changes their tour-adjacency.

6. **Double-bridge kick** (ILS perturbation):
   4-opt non-sequential perturbation that escapes local optima, delegated to
   :func:`double_bridge` from
   ``logic.src.policies.helpers.operators.perturbation.double_bridge``.

7. **Tour merging** (pool-based recombination):
   Combines shared edges from several elite tours to seed new searches.

8. **_improve_tour** — outer driver.  Iterates over every non-masked node
   t1 and tries 2-opt first; if unsuccessful it attempts 3-, 4-, and 5-opt
   in order (each gated by an instance-size threshold).  Accepts the first
   improvement found (first-improvement strategy) and returns immediately.
   Uses *don't-look bits* (Helsgaun 2000, Section 5.3) to skip nodes whose
   neighbourhood was exhausted since the last move touched them.

Architecture
------------
This module owns:
- 1-tree computation     → :func:`compute_1tree`
- Subgradient optimiser  → :func:`run_subgradient`
- Top-level LKH-1 solver → :func:`solve_lkh1`

The inner k-opt improvement loop is delegated to:
- :func:`_improve_tour` from ``lin_kernighan_helsgaun.py`` (k = 2..5,
  don't-look bits, first-improvement).

Candidate set / α-measure utilities are reused from:
- :func:`compute_alpha_measures`, :func:`get_candidate_set` in
  ``lin_kernighan_helsgaun.py``.

The double-bridge perturbation is reused from:
- :func:`_double_bridge_kick` in ``_tour_construction.py``.

References
----------
Held, M., & Karp, R. M. (1970). The traveling-salesman problem and minimum
  spanning trees. Operations Research, 18(6), 1138–1162.

Helsgaun, K. (2000). An effective implementation of the Lin-Kernighan
  traveling salesman heuristic. EJOR 126, 106–130.

Attributes:
    None

Example:
    >>> tour, cost = solve_lkh(distance_matrix)
"""

from __future__ import annotations

import random
from random import Random
from typing import List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.search_heuristics._objective import (
    get_cost,
    is_better,
)
from logic.src.policies.helpers.operators.search_heuristics._tour_construction import (
    _double_bridge_kick,
    _initialize_tour,
    merge_tours,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder

from .lin_kernighan_helsgaun_alpha import (
    compute_alpha_measures as compute_alpha_measures,
)
from .lin_kernighan_helsgaun_alpha import (
    get_candidate_set as get_candidate_set,
)
from .lin_kernighan_helsgaun_alpha import (
    run_subgradient as run_subgradient,
)
from .lin_kernighan_helsgaun_search import (
    _improve_tour as _improve_tour,
)

__all__ = ["solve_lkh", "solve_lkh1", "compute_alpha_measures", "get_candidate_set", "run_subgradient", "_improve_tour"]


"""
LKH-1 Solver — Full Implementation (Helsgaun 2000).

Implements the complete Lin-Kernighan-Helsgaun (LKH-1) algorithm as described
in Helsgaun (2000), including the components absent from the existing
``lin_kernighan_helsgaun.py::solve_lkh``:

1. **Minimum 1-Tree (1T) computation** (Section 3, Helsgaun 2000):
   A 1-tree rooted at node 0 is the minimum spanning tree on nodes {1..n-1}
   plus the two shortest edges incident to node 0.  Its weight W(π) is a
   Held-Karp lower bound on the optimal tour length when the node penalties
   π are optimal (Held & Karp 1970).

2. **Subgradient optimisation of π** (Section 3.1, Helsgaun 2000):
   The penalty vector π ∈ ℝ^n is updated to maximise the 1-tree lower bound:

       W(π) = min-1-tree(c̃_{ij}) + 2 · Σ_i π_i
       c̃_{ij} = c_{ij} + π_i + π_j          (penalty-modified costs)
       g_i    = d_i(T) − 2                   (degree-defect subgradient)
       π ← π + t · g,   t = μ · (W* − W) / ||g||²  (Polyak step)

   where d_i(T) is the degree of node i in the 1-tree T.

3. **Penalty-modified candidate sets** (Section 4.1, Helsgaun 2000):
   The α-nearness and candidate lists are recomputed using c̃ rather than c
   after subgradient convergence.  This produces far tighter candidates than
   MST-only α-measures on the raw distances.

4. **Full ILS loop** (Section 5, Helsgaun 2000):
   Combines the penalized k-opt local search (delegated to
   :func:`_improve_tour` from ``._tour_improvement``) with double-bridge
   perturbation and elite-pool recombination.

Architecture
------------
This module owns:
- 1-tree computation     → :func:`compute_1tree`
- Subgradient optimiser  → :func:`run_subgradient`
- Top-level LKH-1 solver → :func:`solve_lkh1`

The inner k-opt improvement loop is delegated to:
- :func:`_improve_tour` from ``lin_kernighan_helsgaun.py`` (k = 2..5,
  don't-look bits, first-improvement).

Candidate set / α-measure utilities are reused from:
- :func:`compute_alpha_measures`, :func:`get_candidate_set` in
  ``lin_kernighan_helsgaun.py``.

The double-bridge perturbation is reused from:
- :func:`_double_bridge_kick` in ``_tour_construction.py``.

References
----------
Held, M., & Karp, R. M. (1970). The traveling-salesman problem and minimum
  spanning trees. Operations Research, 18(6), 1138–1162.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun import solve_lkh
    >>> tour, cost = solve_lkh(dist_matrix)

References:
----------
Helsgaun, K. (2000). An effective implementation of the Lin-Kernighan traveling
  salesman heuristic. European Journal of Operational Research, 126(1), 106–130.
"""


# ---------------------------------------------------------------------------
# Top-level LKH-1 solver
# ---------------------------------------------------------------------------


def solve_lkh1(
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 100,
    max_k: int = 5,
    n_candidates: int = 5,
    sg_max_iter: int = 100,
    sg_mu_init: float = 1.0,
    sg_patience: int = 20,
    pool_size: int = 5,
    recorder: Optional[PolicyStateRecorder] = None,
    np_rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[List[int], float, float]:
    """
    Solve a TSP instance using the full LKH-1 algorithm (Helsgaun 2000).

    This function orchestrates the complete LKH-1 pipeline:

    Phase 1 — Subgradient (Section 3.1):
        Run :func:`run_subgradient` to find penalty vector π* that maximises
        the Held-Karp 1-tree lower bound W(π).

    Phase 2 — Penalized candidate sets (Section 4.1):
        Compute α-nearness and candidate lists on the *penalized* distance
        matrix c̃_{ij} = c_{ij} + π*_i + π*_j.  These candidates are far
        tighter than those derived from raw distances.

    Phase 3 — ILS local-search loop (Section 5):
        Repeat until the iteration budget is exhausted:
        a. Run sequential k-opt local search via ``_improve_tour`` (k=2..max_k,
           don't-look bits) until a local optimum is reached.
        b. Update the global best solution.
        c. Every 10 iterations: try elite-pool tour merging.
        d. Otherwise: apply double-bridge perturbation.

    Args:
        distance_matrix: (n x n) symmetric raw cost matrix.
        initial_tour: Optional starting closed tour.
        max_iterations: ILS restart budget.
        max_k: Maximum k-opt depth (2–5).
        n_candidates: Candidate list size per node.
        sg_max_iter: Maximum subgradient iterations.
        sg_mu_init: Initial Polyak step multiplier.
        sg_patience: Patience for μ halving.
        pool_size: Elite tour pool size (for merge recombination).
        recorder: Optional telemetry recorder.
        np_rng: NumPy Generator.
        seed: Alternative seed.

    Returns:
        Tuple[List[int], float, float]: (best_tour, best_cost, hk_lower_bound)
            - best_tour: Best closed tour found.
            - best_cost: Its total raw distance.
            - hk_lower_bound: Held-Karp lower bound W(π*) from Phase 1.
    """
    n = len(distance_matrix)
    if n < 3:
        t = list(range(n)) + [0]
        return t, float(get_cost(t, distance_matrix)), 0.0

    if np_rng is None:
        np_rng = np.random.default_rng(seed if seed is not None else 42)
    stdlib_rng = Random(int(np_rng.integers(0, 2**31)))

    # -----------------------------------------------------------------
    # Phase 1: Subgradient optimisation → π*, W*(Held-Karp bound)
    # -----------------------------------------------------------------
    pi, hk_bound, penalized_dm = run_subgradient(
        distance_matrix,
        max_iter=sg_max_iter,
        mu_init=sg_mu_init,
        patience=sg_patience,
    )

    if recorder is not None:
        recorder.record(engine="lkh1_subgradient", hk_bound=hk_bound)

    # -----------------------------------------------------------------
    # Phase 2: Build α-candidate sets on the penalized matrix
    # -----------------------------------------------------------------
    alpha_penalized = compute_alpha_measures(penalized_dm)
    candidates = get_candidate_set(penalized_dm, alpha_penalized, max_candidates=n_candidates)

    # -----------------------------------------------------------------
    # Phase 3: ILS local-search on raw distances with penalized candidates
    # -----------------------------------------------------------------
    curr_tour = _initialize_tour(distance_matrix, initial_tour)
    curr_cost = get_cost(curr_tour, distance_matrix)
    best_tour = curr_tour[:]
    best_cost = curr_cost

    dont_look_bits: Optional[np.ndarray] = None
    tour_pool: List[List[int]] = [best_tour[:]]

    for restart in range(max_iterations):
        # Inner k-opt passes until local optimum
        while True:
            curr_tour, curr_cost, improved_local, dont_look_bits = _improve_tour(
                curr_tour,
                curr_cost,
                candidates,
                distance_matrix,
                stdlib_rng,
                dont_look_bits,
                max_k,
            )
            if not improved_local:
                break

        if is_better(curr_cost, best_cost):
            best_cost = curr_cost
            best_tour = curr_tour[:]
            tour_pool.append(best_tour[:])
            if len(tour_pool) > pool_size:
                tour_pool.pop(0)

        if recorder is not None:
            recorder.record(
                engine="lkh1",
                restart=restart,
                best_cost=best_cost,
                hk_bound=hk_bound,
                gap=(best_cost - hk_bound) / max(hk_bound, 1.0),
            )

        # Early termination if optimality gap is closed
        if hk_bound > 0 and (best_cost - hk_bound) / hk_bound < 1e-4:
            break

        # Every 10 restarts: try elite-pool merging
        if restart > 0 and restart % 10 == 0 and len(tour_pool) >= 2:
            idx1, idx2 = random.sample(range(len(tour_pool)), 2)
            merged = merge_tours(tour_pool[idx1], tour_pool[idx2], distance_matrix)
            merged_cost = get_cost(merged, distance_matrix)
            if is_better(merged_cost, best_cost):
                best_cost = merged_cost
                best_tour = merged[:]
                tour_pool.append(best_tour[:])
                if len(tour_pool) > pool_size:
                    tour_pool.pop(0)
            curr_tour = merged
            curr_cost = merged_cost
            dont_look_bits = None
        else:
            curr_tour = _double_bridge_kick(best_tour, distance_matrix, stdlib_rng)
            curr_cost = get_cost(curr_tour, distance_matrix)
            dont_look_bits = None

    return best_tour, best_cost, hk_bound


# ---------------------------------------------------------------------------
# Convenience wrapper matching the solve_lkh interface
# ---------------------------------------------------------------------------


def solve_lkh(
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    max_iterations: int = 100,
    max_k: int = 5,
    recorder: Optional[PolicyStateRecorder] = None,
    np_rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Tuple[List[int], float]:
    """
    Thin convenience wrapper for :func:`solve_lkh1` matching the two-return
    interface of the existing ``lin_kernighan_helsgaun.solve_lkh``.

    Args:
        distance_matrix: (n x n) symmetric cost matrix.
        initial_tour: Optional starting tour (closed).
        max_iterations: Maximum LKH-1 iterations.
        max_k: Maximum k-opt move size (2 to 5).
        recorder: Optional telemetry recorder.
        np_rng: NumPy Generator.
        seed: Alternative seed.

    Returns:
        Tuple[List[int], float]: (best_tour, best_cost) without the Held-Karp bound.
    """
    tour, cost, _ = solve_lkh1(
        distance_matrix,
        initial_tour=initial_tour,
        max_iterations=max_iterations,
        max_k=max_k,
        recorder=recorder,
        np_rng=np_rng,
        seed=seed,
    )
    return tour, cost

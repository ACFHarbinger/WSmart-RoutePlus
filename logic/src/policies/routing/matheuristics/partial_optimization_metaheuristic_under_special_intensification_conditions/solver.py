"""
Core implementation of POPMUSIC (Partial Optimization Metaheuristic Under Special
Intensification Conditions).

POPMUSIC is a matheuristic framework that decomposes a large combinatorial
optimization problem into subproblems (sets of routes) and iteratively optimizes
them until a local optimum is reached across all parts.

Reference:
    Taillard, É., & Voss, S. (2002). POPMUSIC — partial optimization metaheuristic
    under special intensification conditions. In Essays and Surveys in Metaheuristics
    (pp. 613–629). Kluwer, Boston.

    Alvim, A. C. F., & Taillard, É. D. (2013). POPMUSIC for the world location
    routing problem. EURO Journal on Transportation and Logistics, 2, 231–254.

The route-centroid, nearest-route lookup and O(r) subproblem-assembly helpers
moved to :mod:`popmusic_utils` and are re-imported here; the Issue-2
adjacency-walk details are documented in that module.

Architectural notes (four-issue refactor):
    Issue 1 — Asymptotic initialization:
        Initial solution is built via a two-level p-median alternating heuristic
        (p = floor(sqrt(n))) running in O(n^{3/2}) with profit-biased assignment.

    Issue 3 — VRPP subproblem purity:
        Every unvisited node u ∈ U is represented as a singleton part s_u = {u}
        with centroid = coord(u), inserted into G and the seed stack U whenever
        a node becomes unvisited. The r-part invariant is preserved exactly.

    Issue 4 — Ghost routes / data structure:
        A pre-allocated stable centroid buffer coords[MAX_PARTS, 2] is maintained.
        Deleted slots are pushed onto a free_slots stack in O(1). New parts pop a slot
        in O(1). G uses raw slot indices — slots are never compacted, so all G entries
        remain permanently valid. Compaction is never performed.

Attributes:
    run_popmusic: Entry point for the POPMUSIC solver.
    find_route_neighbors: KDTree-based nearest-route neighbor lookup.

Example:
    >>> solver = POPSolver(
    >>>     X=np.array([[0,0], [1,1], [2,2]]),
    >>>     wastes={1: 1.0, 2: 2.0},
    >>>     capacity=10.0,
    >>>     R=3.0, C=1.0,
    >>>     K=1,
    >>>     seed=42,
>>> )
    >>> routes, profit = solver.run_popmusic()
    >>> routes
    [[0, 1, 2, 0]]
    >>> profit
    5.393398113205676
"""

from __future__ import annotations

import warnings
from collections import deque
from random import Random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from logic.src.envs.common.expand_pool import current_expand_pool
from logic.src.policies.routing.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.pmedian import (
    _compute_d_prox,
    _initialize_pmedian,
)
from logic.src.policies.routing.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.popmusic_utils import (
    _assemble_subproblem,
    _route_centroid,
)
from logic.src.policies.routing.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.subproblem import (
    _evaluate_slots,
    _optimize_subproblem,
)
from logic.src.utils.routing.tours import get_route_cost

from .state import POPMUSICState

# ---------------------------------------------------------------------------
# Issue 1 — p-median initialization with profit-biased assignment
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main POPMUSIC loop
# ---------------------------------------------------------------------------


def run_popmusic(  # noqa: C901
    coords: pd.DataFrame,
    mandatory: List[int],
    distance_matrix: np.ndarray,
    n_vehicles: int,
    subproblem_size: int = 3,
    max_iterations: Optional[int] = None,
    base_solver: str = "fast_tsp",
    base_solver_config: Optional[Any] = None,
    cluster_solver: str = "fast_tsp",
    cluster_solver_config: Optional[Any] = None,
    initial_solver: str = "pmedian",
    seed: int = 42,
    wastes: Optional[Dict[int, float]] = None,
    capacity: float = 1.0e9,
    R: float = 1.0,
    C: float = 0.0,
    profit_aware_operators: bool = False,
    k_prox: int = 10,
    seed_strategy: str = "lifo",
) -> Tuple[List[List[int]], float, float, Dict[str, Any]]:
    """
    Solve VRPP using the POPMUSIC matheuristic (Taillard & Voss, 2002).

    Architectural guarantees (post-refactor):
        - Initialization: O(n^{3/2}) via p-median alternating heuristic.
        - Proximity network G: built once, O(r) subproblem assembly.
        - VRPP purity: unvisited nodes are singleton parts; r-part invariant
          is preserved exactly.
        - Data structure: stable-slot free-stack; no compaction; O(1)
          insert/delete; G never invalidated.

    Args:
        coords: DataFrame with columns [Lat, Lng], indexed by global node ID.
        mandatory: Global indices of nodes that must be visited.
        distance_matrix: Symmetric distance matrix, shape (n+1, n+1), depot at 0.
        n_vehicles: Maximum number of vehicles (upper-bounds p).
        subproblem_size: r — number of parts per subproblem. The unique
            POPMUSIC parameter (Taillard & Voss, 2002, §Conclusion).
        max_iterations: Optional hard iteration cap (None = run to convergence).
        base_solver: Subproblem solver ("fast_tsp", "hgs", "alns").
        base_solver_config: Configuration passed to the base solver.
        cluster_solver: Solver for warm-starting initial clusters (unused
            unless WARMSTART_INITIAL=True).
        cluster_solver_config: Configuration for cluster_solver.
        initial_solver: Initialization method. "pmedian" (canonical, O(n^{3/2}))
            is the only fully paper-faithful option. Legacy "greedy" and
            "nearest_neighbor" are retained for backward compatibility but
            do not construct G and fall back to a KDTree (O(p log p) rebuild).
        seed: Random seed for reproducibility.
        wastes: Mapping global node index → waste level.
        capacity: Vehicle capacity.
        R: Revenue per unit waste collected.
        C: Monetary cost per unit distance travelled.
            nodes are represented as singleton parts (Issue 3).
        profit_aware_operators: If True, uses profit-biased d_assign metric
            during p-median initialization (Issue 1).
        k_prox: Retained for API compatibility. In the refactored code,
            G degree is set to subproblem_size + 2 during initialization.
        seed_strategy: Seed selection strategy ("lifo", "fifo", "random").

    Returns:
        routes: List of optimized routes (each a list of global node indices).
        total_routing_cost: Total travel cost (distance * C).
        total_profit: Total net profit (revenue − travel cost).
        info: Dictionary with diagnostic keys: iterations, num_routes,
            num_singletons_final, p_initial.
    """
    if not mandatory:
        return [[0, 0]], 0.0, 0.0, {}

    wastes_dict: Dict[int, float] = wastes if wastes is not None else {}
    coord_array = coords[["Lat", "Lng"]].values.astype(np.float64)  # (n+1, 2), depot at 0
    n_nodes = len(coord_array) - 1  # excludes depot

    # ------------------------------------------------------------------ #
    # 1. INITIALIZATION                                                    #
    # ------------------------------------------------------------------ #

    if initial_solver == "pmedian":
        init_routes, center_coords, G_cluster = _initialize_pmedian(
            coords_df=coords,
            wastes_dict=wastes_dict,
            capacity=capacity,
            R=R,
            C=C,
            n_vehicles=n_vehicles,
            subproblem_size=subproblem_size,
            seed=seed,
            profit_aware=profit_aware_operators,
            distance_matrix=distance_matrix,
            mandatory=mandatory,
        )
    else:
        # Legacy fallback — does not produce G; G is built from centroids below.
        from logic.src.policies.helpers.operators.solution_initialization.greedy_si import (
            build_greedy_routes,
        )
        from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import (
            build_nn_routes,
        )

        rng_legacy = Random(seed)
        if initial_solver == "greedy":
            init_routes = build_greedy_routes(
                dist_matrix=distance_matrix,
                wastes=wastes_dict,
                capacity=capacity,
                R=R,
                C=C,
                mandatory_nodes=mandatory,
                rng=rng_legacy,
            )
        elif initial_solver == "nearest_neighbor":
            init_routes = build_nn_routes(
                nodes=list(range(1, len(distance_matrix))),
                mandatory_nodes=mandatory,
                wastes=wastes_dict,
                capacity=capacity,
                dist_matrix=distance_matrix,
                R=R,
                C=C,
                rng=rng_legacy,
            )
        else:
            raise ValueError(f"Unknown initial_solver: {initial_solver!r}")
        G_cluster = {}  # built lazily from centroids below

    p_initial = len(init_routes)

    # ------------------------------------------------------------------ #
    # 2. BUILD POPMUSICState (Issue 4)                                    #
    # ------------------------------------------------------------------ #

    state = POPMUSICState(n_nodes=n_nodes + len(init_routes) + n_nodes, coord_array=coord_array)
    # n_nodes extra headroom: worst case all nodes become singleton parts simultaneously

    # Slot map: cluster index (from G_cluster) → state slot
    cluster_to_slot: Dict[int, int] = {}
    route_slots: List[int] = []

    for k, route in enumerate(init_routes):
        centroid = _route_centroid(route, coord_array)
        slot = state.insert_route(route, centroid)
        cluster_to_slot[k] = slot
        route_slots.append(slot)

    # Remap G_cluster indices to state slot indices — O(p * k)
    if G_cluster:
        for cluster_idx, neighbor_clusters in G_cluster.items():
            mapped_slot: Optional[int] = cluster_to_slot.get(cluster_idx)
            if mapped_slot is None:
                continue
            neighbor_slots = [cluster_to_slot[nc] for nc in neighbor_clusters if nc in cluster_to_slot]
            state.G[mapped_slot] = neighbor_slots
    else:
        # Legacy fallback: build G from initial centroids using geometric d_prox — O(p^2)
        active_slots = sorted(state.active)
        p = len(active_slots)
        if p > 1:
            c_coords = state.coords[active_slots]
            k_nb = min(subproblem_size + 2, p - 1)
            d_prox = _compute_d_prox(c_coords)
            for local_i, slot in enumerate(active_slots):
                row = d_prox[local_i].copy()
                row[local_i] = np.inf
                nb_local = np.argsort(row)[:k_nb].tolist()
                state.G[slot] = [active_slots[j] for j in nb_local]

    # VRPP: all customers not in any route become singleton parts (Issue 3)
    assigned_nodes: Set[int] = {n for r in init_routes for n in r if n != 0}
    all_customers: Set[int] = {int(i) for i in coords.index if i != 0}

    if current_expand_pool():
        for unode in all_customers - assigned_nodes:
            singleton_slot = state.insert_singleton(unode)
            # Insert singleton into G: connect to k nearest active route slots
            active_route_list = sorted(state.active_route_slots)
            if active_route_list:
                k_nb = min(subproblem_size + 2, len(active_route_list))
                dists = np.linalg.norm(state.coords[active_route_list] - state.coords[singleton_slot], axis=1)
                nb_idx = np.argsort(dists)[:k_nb]
                state.G[singleton_slot] = [active_route_list[j] for j in nb_idx]

    # ------------------------------------------------------------------ #
    # 3. SEED STACK U (Paper Algorithm 1)                                 #
    # ------------------------------------------------------------------ #

    rng = Random(seed)
    all_active = sorted(state.active)

    if seed_strategy == "random":
        rng.shuffle(all_active)
        active_stack: deque[int] = deque(all_active)
    elif seed_strategy == "fifo":
        active_stack = deque(all_active)
    else:  # lifo (default, recommended by paper)
        active_stack = deque(all_active)

    active_in_stack: Set[int] = set(active_stack)

    # ------------------------------------------------------------------ #
    # 4. MAIN POPMUSIC LOOP                                               #
    # ------------------------------------------------------------------ #

    iteration = 0

    while active_stack:
        if max_iterations is not None and iteration >= max_iterations:
            warnings.warn(
                f"POPMUSIC terminated early: max_iterations={max_iterations} reached.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        # Select seed s_g (Paper Algorithm 1, line 3)
        seed_slot = active_stack.pop() if seed_strategy == "lifo" else active_stack.popleft()
        active_in_stack.discard(seed_slot)

        # Skip slots that were deleted since being enqueued
        if seed_slot not in state.active:
            continue

        # ---- Assemble subproblem R: O(r) walk on G (Issue 2) ----------
        subproblem_slots = _assemble_subproblem(seed_slot, state, subproblem_size)

        # Collect nodes from subproblem slots (routes + singletons, Issue 3)
        subproblem_nodes: List[int] = []
        for sl in subproblem_slots:
            subproblem_nodes.extend(state.customer_nodes(sl))

        if not subproblem_nodes:
            continue

        # ---- Evaluate current solution quality -------------------------
        old_profit = _evaluate_slots(subproblem_slots, state, distance_matrix, wastes_dict, R, C)

        # ---- Optimize subproblem ---------------------------------------
        new_routes, new_profit = _optimize_subproblem(
            base_solver=base_solver,
            base_solver_config=base_solver_config,
            subproblem_nodes=subproblem_nodes,
            distance_matrix=distance_matrix,
            wastes_dict=wastes_dict,
            capacity=capacity,
            R=R,
            C=C,
            neighborhood_indices=subproblem_slots,  # kept for HGS/ALNS compat
            mandatory=mandatory,
            seed=seed,
            profit_aware_operators=profit_aware_operators,
        )

        # ---- Accept improvement and update state (Paper Algorithm 1) ---
        if new_profit > old_profit + 1e-6:
            # Match new_routes to old slots by route content identity.
            # Paper Algorithm 1 line 7: "Remove from U the parts that do NOT
            # belong to s anymore."  A part "belongs to s" if its node-set
            # appears unchanged in the new solution.  Retaining unchanged slots
            # preserves their G connectivity and avoids a wasted slot allocation.
            #
            # Implementation:
            #   - Represent each old slot's content as a frozenset of its
            #     customer nodes (order-independent identity).
            #   - Any new_route whose frozenset matches an old slot's frozenset
            #     is considered "unchanged" — that slot is kept, not replaced.
            #   - Old slots with no matching new_route are deleted (they left s).
            #   - New routes with no matching old slot get a fresh slot.

            old_content: Dict[int, frozenset] = {sl: frozenset(state.customer_nodes(sl)) for sl in subproblem_slots}
            # Index old slots by their content for O(1) lookup
            content_to_old_slot: Dict[frozenset, int] = {
                fs: sl
                for sl, fs in old_content.items()
                if fs  # skip empty routes
            }

            new_slots: List[int] = []
            matched_old_slots: Set[int] = set()

            for nr in new_routes:
                nr_content = frozenset(n for n in nr if n != 0)
                if not nr_content:
                    continue

                if nr_content in content_to_old_slot:
                    # Route unchanged — retain slot, update centroid in-place
                    old_sl = content_to_old_slot.pop(nr_content)  # pop: each slot matched once
                    matched_old_slots.add(old_sl)
                    # Centroid may have drifted if route order changed but nodes
                    # are identical — recompute for accuracy, O(|route|).
                    state.update_coords(old_sl, _route_centroid(nr, coord_array))
                    state.route_nodes[old_sl] = nr
                    new_slots.append(old_sl)
                else:
                    # Genuinely new route — allocate a fresh slot
                    centroid = _route_centroid(nr, coord_array)
                    new_slot = state.insert_route(nr, centroid)
                    new_slots.append(new_slot)

            # Delete old slots whose content no longer appears in s
            for sl in subproblem_slots:
                if sl not in matched_old_slots:
                    state.delete_slot(sl)
                    active_in_stack.discard(sl)

            # Handle nodes that became unvisited (VRPP, Issue 3)
            if current_expand_pool():
                nodes_in_new = {n for nr in new_routes for n in nr if n != 0}
                newly_unvisited = set(subproblem_nodes) - nodes_in_new
                for unode in newly_unvisited:
                    if unode in set(mandatory):
                        # Mandatory nodes cannot be dropped
                        new_route = [0, unode, 0]
                        centroid = _route_centroid(new_route, coord_array)
                        recov_slot = state.insert_route(new_route, centroid)
                        new_slots.append(recov_slot)
                    else:
                        sing_slot = state.insert_singleton(unode)
                        new_slots.append(sing_slot)

            # Update G for genuinely new slots only (retained slots keep their
            # existing G entries which remain valid — Gap-9 fix).
            # Candidate pool = union of deleted slots' former neighbors + new
            # slots' own neighbors (bounded by O(r * degree(G)) = O(r)).
            deleted_slots = set(subproblem_slots) - matched_old_slots
            candidate_pool: Set[int] = set()
            for dl in deleted_slots:
                # Neighbors of deleted slots in G (before deletion) were already
                # pruned by delete_slot; use subproblem_slots neighbors instead.
                for nb in state.G.get(dl, []):
                    if nb in state.active:
                        candidate_pool.add(nb)
            # Also include all new_slots themselves as mutual candidates
            for ns in new_slots:
                candidate_pool.add(ns)

            candidate_list = sorted(candidate_pool | set(new_slots))
            k_nb = min(subproblem_size + 2, max(1, len(candidate_list) - 1))

            for new_slot in new_slots:
                if new_slot in matched_old_slots:
                    continue  # retained slot: G entry still valid
                if not candidate_list:
                    state.G[new_slot] = []
                    continue
                dists = np.linalg.norm(state.coords[candidate_list] - state.coords[new_slot], axis=1)
                nb_idx = np.argsort(dists)[: k_nb + 1]
                state.G[new_slot] = [candidate_list[j] for j in nb_idx if candidate_list[j] != new_slot][:k_nb]

            # Paper Algorithm 1 line 8: "Insert in U the parts of optimized
            # subproblem R."  This means ALL parts of R — both changed and
            # unchanged — are re-inserted so they can be re-examined as seeds
            # from a different neighborhood context (Gap-4 fix).
            for sl in new_slots:
                if sl not in active_in_stack and sl in state.active:
                    if seed_strategy == "lifo":
                        active_stack.append(sl)
                    else:
                        active_stack.appendleft(sl)
                    active_in_stack.add(sl)

        iteration += 1

    # ------------------------------------------------------------------ #
    # 5. EXTRACT FINAL SOLUTION                                           #
    # ------------------------------------------------------------------ #

    final_routes: List[List[int]] = []
    for slot in state.active_route_slots:
        route = state.route_nodes[slot]
        customers = [n for n in route if n != 0]
        if customers:
            final_routes.append(route)

    total_routing_cost = sum(get_route_cost(distance_matrix, r) for r in final_routes) * C
    total_rev = sum(wastes_dict.get(n, 0.0) * R for r in final_routes for n in r if n != 0)
    total_profit = total_rev - total_routing_cost

    return (
        final_routes,
        total_routing_cost,
        total_profit,
        {
            "iterations": iteration,
            "num_routes": len(final_routes),
            "num_singletons_final": len(state.active_singleton_slots),
            "p_initial": p_initial,
        },
    )


# ---------------------------------------------------------------------------
# Subproblem evaluation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Subproblem optimization (unchanged interface, internal only)
# ---------------------------------------------------------------------------

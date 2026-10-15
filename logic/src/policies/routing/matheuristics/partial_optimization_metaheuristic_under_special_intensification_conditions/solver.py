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

Architectural notes (four-issue refactor):
    Issue 1 — Asymptotic initialization:
        Initial solution is built via a two-level p-median alternating heuristic
        (p = floor(sqrt(n))) running in O(n^{3/2}). Cluster membership uses a
        profit-biased assignment metric d_assign(i,c) = d(i,c)/d_max / (rev(i)/rev_max + ε)
        (ratio form, dimensionless, parameter-free). Cluster centers are recomputed as
        the geographic median of assigned nodes (stable centroid semantics). The proximity
        network G is built on pure geometric distance d_prox between cluster centers
        (two-metric architecture), preserving spatial locality for subproblem assembly.

    Issue 2 — KDTree inefficiency:
        The proximity network G is built once during initialization as a static
        adjacency list dict[int, list[int]] sorted by inter-center distance. Subproblem
        assembly walks G in O(r). Centroids are updated in O(r) after each improvement.
        G topology is never rebuilt — consistent with the paper's proximate optimality
        assumption.

    Issue 3 — VRPP subproblem purity:
        Every unvisited node u ∈ U is represented as a singleton part s_u = {u} with
        centroid = coord(u), inserted into G and the seed stack U at initialization and
        whenever a node becomes unvisited. The r-part invariant is preserved exactly:
        a subproblem of size r may contain a mix of proper routes and singleton parts.
        p = |active_routes| + |singleton_unvisited| is dynamic.

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
    >>>     vrpp=True
    >>> )
    >>> routes, profit = solver.run_popmusic()
    >>> routes
    [[0, 1, 2, 0]]
    >>> profit
    5.393398113205676
"""

from __future__ import annotations

import dataclasses
import warnings
from collections import deque
from random import Random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from logic.src.configs.policies.alns import ALNSConfig
from logic.src.configs.policies.hgs import HGSConfig
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.alns import ALNSSolver
from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.hgs import HGSSolver
from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.split import LinearSplit
from logic.src.policies.routing.other_algorithms.travelling_salesman_problem.tsp import (
    find_route,
    get_route_cost,
)

from .state import POPMUSICState

# ---------------------------------------------------------------------------
# Issue 1 — p-median initialization with profit-biased assignment
# ---------------------------------------------------------------------------


def _compute_d_assign(
    node_coords: np.ndarray,
    center_coords: np.ndarray,
    revenues: np.ndarray,
    d_max: float,
    rev_max: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute the profit-biased assignment dissimilarity matrix.

    Uses the dimensionless ratio form (Issue 1, Tension 3):

        d_assign(i, c) = (d(i,c) / d_max) / (rev(i) / rev_max + ε)

    where d_max and rev_max are instance-level normalization constants
    computed once in O(n), making the metric parameter-free.

    High-revenue nodes are gravitationally attracted to cluster centers
    (small d_assign) while the metric remains strictly non-negative.

    Time complexity: O(n * p) via vectorized broadcasting.
    Space complexity: O(n * p).

    Args:
        node_coords: Shape (n, 2) — customer coordinates.
        center_coords: Shape (p, 2) — current cluster center coordinates.
        revenues: Shape (n,) — per-node revenue (rev(i) = waste(i) * R).
        d_max: Maximum pairwise distance in the instance (normalization).
        rev_max: Maximum node revenue (normalization).
        eps: Small positive constant preventing division by zero.

    Returns:
        d_assign_matrix: Shape (n, p) — profit-biased dissimilarity.
    """
    # Euclidean distances: (n, p) via broadcasting
    diff = node_coords[:, np.newaxis, :] - center_coords[np.newaxis, :, :]  # (n,p,2)
    geo_dist = np.linalg.norm(diff, axis=-1)  # (n, p)
    norm_dist = geo_dist / (d_max + eps)  # (n, p)  — normalized to [0,1]

    # Revenue factor: (n,) → (n,1) for broadcasting
    rev_factor = (revenues / (rev_max + eps) + eps)[:, np.newaxis]  # (n, 1)

    return norm_dist / rev_factor  # (n, p)


def _compute_d_prox(center_coords: np.ndarray) -> np.ndarray:
    """
    Compute the pure geometric pairwise distance matrix between cluster centers.

    Used exclusively for building the proximity network G (Issue 1, two-metric
    architecture). Spatial coherence of subproblems requires G to reflect
    geographic nearness, not profit-biased nearness.

    Time complexity: O(p^2).
    Space complexity: O(p^2).

    Args:
        center_coords: Shape (p, 2) — cluster center coordinates.

    Returns:
        d_prox_matrix: Shape (p, p) — symmetric Euclidean distance matrix.
    """
    diff = center_coords[:, np.newaxis, :] - center_coords[np.newaxis, :, :]  # (p,p,2)
    return np.linalg.norm(diff, axis=-1)  # (p, p)


def _pmedian_alternating(
    node_coords: np.ndarray,
    revenues: np.ndarray,
    p: int,
    rng: Random,
    max_iter: int = 30,
    profit_aware: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate p-median via the alternating (Voronoi) heuristic.

    Runs in O(n * p * max_iter) = O(n^{3/2}) since p = floor(sqrt(n)) and
    max_iter is a small constant (empirically 10–20 suffice for convergence).

    Assignment step uses d_assign (profit-biased) when profit_aware=True,
    else pure Euclidean. Center recomputation always uses the geographic
    median of assigned nodes — mixing profit-bias into center recomputation
    would drift centers toward revenue hotspots and destabilize convergence
    (Issue 1, Point 3 decision).

    Args:
        node_coords: Shape (n, 2) — customer coordinates (excluding depot).
        revenues: Shape (n,) — per-node revenue for profit-bias.
        p: Number of cluster centers = floor(sqrt(n)).
        rng: Seeded random instance for reproducibility.
        max_iter: Maximum alternating iterations (constant, does not affect O).
        profit_aware: Whether to use profit-biased d_assign for Voronoi step.

    Returns:
        assignments: Shape (n,) int — cluster index for each node.
        center_coords: Shape (p, 2) — final cluster center coordinates.
        d_final: Shape (n, p) — distance matrix from the final assignment
            step, returned for reuse in _build_proximity_network (avoids
            redundant O(n*p) recomputation of the same matrix).
    """
    n = len(node_coords)
    if p >= n:
        # Degenerate: each node is its own cluster
        return np.arange(n), node_coords.copy(), _compute_d_prox(node_coords)

    # Normalisation constants — computed once, O(n)
    # d_max: use max distance from a random sample to avoid O(n^2) full matrix
    sample_size = min(200, n)
    sample_idx = rng.sample(range(n), sample_size)
    sample_coords = node_coords[sample_idx]
    diff_sample = sample_coords[:, np.newaxis, :] - sample_coords[np.newaxis, :, :]
    d_max = float(np.linalg.norm(diff_sample, axis=-1).max()) + 1e-8
    rev_max = float(revenues.max()) + 1e-8

    # Initialise p centers via maximin seeding: O(n * p)
    # Each new center maximises the minimum distance to existing centers.
    center_indices = [rng.randrange(n)]
    for _ in range(p - 1):
        # Distance of every node to its nearest current center
        current_centers = node_coords[center_indices]
        diff = node_coords[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1).min(axis=1)  # (n,)
        center_indices.append(int(np.argmax(dists)))

    center_coords = node_coords[center_indices].copy()  # (p, 2)
    assignments = np.zeros(n, dtype=np.int64)

    for _ in range(max_iter):
        # --- Assignment step: Voronoi w/ profit-biased metric ---
        if profit_aware:
            d_matrix = _compute_d_assign(node_coords, center_coords, revenues, d_max, rev_max)
        else:
            diff = node_coords[:, np.newaxis, :] - center_coords[np.newaxis, :, :]
            d_matrix = np.linalg.norm(diff, axis=-1)  # (n, p)

        new_assignments = np.argmin(d_matrix, axis=1)  # (n,)

        # Convergence check
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments

        # --- Center recomputation: geographic centroid (stable semantics) ---
        for k in range(p):
            members = np.where(assignments == k)[0]
            if len(members) > 0:
                center_coords[k] = node_coords[members].mean(axis=0)
            # If cluster is empty, keep previous center (avoids collapse)

    # Capture the distance matrix from the final assignment step for reuse
    # in _build_proximity_network (second-closest-center criterion).
    d_final: np.ndarray
    if profit_aware:
        d_final = _compute_d_assign(node_coords, center_coords, revenues, d_max, rev_max)
    else:
        diff_f = node_coords[:, np.newaxis, :] - center_coords[np.newaxis, :, :]
        d_final = np.linalg.norm(diff_f, axis=-1)

    return assignments, center_coords, d_final


def _build_proximity_network(
    center_coords: np.ndarray,
    assignments: np.ndarray,
    d_assign_matrix: np.ndarray,
) -> Dict[int, List[int]]:
    """
    Build the static adjacency list proximity network G using the canonical
    second-closest-center criterion (Taillard & Voss, 2002, p.692; Alvim &
    Taillard, 2013).

    Definition (paper p.692):
        Clusters a and b are neighbors if an entity of one of these clusters
        has the center of the other one as the second closest center.

    Formally: edge (a, b) ∈ G iff ∃ i ∈ cluster_a s.t. b = argmin_{c ≠ a} d(i, c),
    or ∃ j ∈ cluster_b s.t. a = argmin_{c ≠ b} d(j, c).  The relation is
    symmetrised so both G[a] contains b and G[b] contains a.

    This is strictly more faithful than k-NN on center coordinates because it
    encodes which clusters share "contested" entities — exactly the nodes most
    likely to benefit from joint re-optimisation.  k-NN on centers ignores
    entity distribution entirely; the second-closest criterion is informed by
    the Voronoi structure of the current solution.

    After adding edges by the second-closest criterion, each adjacency list is
    sorted in ascending order of inter-center d_prox so that BFS in
    _assemble_subproblem visits spatially nearest parts first.

    Time complexity: O(n * p) — one argsort of length p per entity, dominated
        by the d_assign_matrix already computed in _pmedian_alternating.
    Space complexity: O(n + p * avg_degree).  avg_degree is empirically O(1)
        for spatially distributed instances.

    Args:
        center_coords: Shape (p, 2) — final cluster center coordinates.
        assignments: Shape (n,) int — cluster index for each entity (0..p-1).
        d_assign_matrix: Shape (n, p) — distance matrix used in the final
            Voronoi assignment step of _pmedian_alternating.  Reused here to
            avoid recomputation; colum j gives d(entity_i, center_j).

    Returns:
        G: dict[int, list[int]] — cluster index → adjacency list sorted by
            ascending inter-center Euclidean distance.
    """
    p = len(center_coords)
    # Adjacency sets — use sets during construction for O(1) membership test,
    # then convert to sorted lists at the end.
    adj: Dict[int, Set[int]] = {k: set() for k in range(p)}

    # For each entity i: find its second-nearest center (≠ its assigned center).
    # d_assign_matrix[i] gives distances to all p centers.
    # The nearest is assignments[i]; the second-nearest is the smallest among the rest.
    for i in range(len(assignments)):
        a = int(assignments[i])
        row = d_assign_matrix[i].copy()
        row[a] = np.inf  # mask the nearest (own cluster)
        b = int(np.argmin(row))  # second-nearest center index
        # Add symmetric edge
        adj[a].add(b)
        adj[b].add(a)

    # Sort each adjacency list by ascending inter-center Euclidean distance
    # so BFS in _assemble_subproblem greedily visits the spatially closest
    # neighbors first, maximising locality of assembled subproblems.
    d_prox = _compute_d_prox(center_coords)  # (p, p)
    G: Dict[int, List[int]] = {}
    for k in range(p):
        neighbors = sorted(adj[k], key=lambda nb: d_prox[k, nb])
        G[k] = neighbors

    return G


def _initialize_pmedian(
    coords_df: pd.DataFrame,
    wastes_dict: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    n_vehicles: int,
    subproblem_size: int,
    seed: int,
    profit_aware: bool,
    distance_matrix: np.ndarray,
    mandatory: List[int],
) -> Tuple[List[List[int]], np.ndarray, Dict[int, List[int]]]:
    """
    Build initial solution via p-median heuristic in O(n^{3/2}).

    Sets p = floor(sqrt(n_customers)), runs the alternating heuristic, and
    constructs the initial routes by assigning each cluster's nodes to a
    vehicle route (greedy capacity-aware assignment within each cluster).
    Returns the proximity network G for use in the main loop (Issue 2).

    Args:
        coords_df: DataFrame with columns [Lat, Lng], indexed by node index.
        wastes_dict: Mapping node → waste level.
        capacity: Vehicle capacity.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        n_vehicles: Hint for maximum number of vehicles.
        subproblem_size: r parameter — used as k_neighbors for G construction.
        seed: Random seed.
        profit_aware: Whether to use profit-biased assignment metric.
        distance_matrix: Full distance matrix (n+1, n+1) including depot at 0.
        mandatory: Mandatory node indices.

    Returns:
        routes: Initial list of routes (each a list of global node indices).
        center_coords: Shape (p, 2) — cluster center coordinates (for state init).
        G_cluster: Proximity network on cluster indices (remapped to route slots
            in run_popmusic after POPMUSICState construction).
    """
    rng = Random(seed)

    # Node indices: all customers excluding depot (index 0)
    all_nodes = [i for i in coords_df.index if i != 0]
    n = len(all_nodes)

    if n == 0:
        return [], np.empty((0, 2)), {}

    node_array = np.array(all_nodes)  # (n,) — maps local index → global node ID
    node_coords = coords_df.loc[all_nodes, ["Lat", "Lng"]].values.astype(np.float64)  # (n,2)

    # Revenue vector for profit-biased metric
    revenues = np.array([wastes_dict.get(node, 0.0) * R for node in all_nodes], dtype=np.float64)

    # p = floor(sqrt(n)) per Alvim & Taillard (2013) — the unique parameter
    # that governs O(n^{3/2}) complexity
    p = max(1, min(int(np.floor(np.sqrt(n))), n_vehicles if n_vehicles > 0 else n))

    # Run alternating p-median heuristic — O(n^{3/2})
    # Returns d_final for reuse in proximity network construction (Gap-2 fix).
    assignments, center_coords, d_final = _pmedian_alternating(
        node_coords=node_coords,
        revenues=revenues,
        p=p,
        rng=rng,
        profit_aware=profit_aware,
    )

    # Build proximity network G using the canonical second-closest-center
    # criterion (paper p.692) — O(n * p) = O(n^{3/2}).
    # d_final is reused directly; no recomputation needed.
    G_cluster = _build_proximity_network(center_coords, assignments, d_final)

    # Construct initial routes: one route per cluster (capacity-aware greedy split)
    mandatory_set = set(mandatory)
    routes: List[List[int]] = []

    for k in range(p):
        cluster_local_idx = np.where(assignments == k)[0]
        cluster_nodes = node_array[cluster_local_idx].tolist()

        if not cluster_nodes:
            continue

        # Sort cluster nodes: mandatory first, then by distance from cluster center
        cluster_nodes.sort(
            key=lambda nd: (
                0 if nd in mandatory_set else 1,
                float(distance_matrix[0, nd]),
            )
        )

        # Greedy capacity-aware split into sub-routes within the cluster
        current_route: List[int] = [0]
        current_load = 0.0

        for nd in cluster_nodes:
            load = wastes_dict.get(nd, 0.0)
            if current_load + load > capacity and len(current_route) > 1:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0]
                current_load = 0.0
            current_route.append(nd)
            current_load += load

        if len(current_route) > 1:
            current_route.append(0)
            routes.append(current_route)

    # Ensure all mandatory nodes appear in at least one route
    assigned_nodes: Set[int] = {n for r in routes for n in r if n != 0}
    for mnd in mandatory:
        if mnd not in assigned_nodes:
            routes.append([0, mnd, 0])

    return routes, center_coords, G_cluster


# ---------------------------------------------------------------------------
# Utility: centroid computation
# ---------------------------------------------------------------------------


def _route_centroid(
    nodes: List[int],
    coord_array: np.ndarray,
) -> np.ndarray:
    """
    Compute the geographic centroid (mean coordinate) of non-depot nodes.

    Args:
        nodes: Route node list including depot sentinels (0).
        coord_array: Shape (n+1, 2) — all node coordinates.

    Returns:
        centroid: Shape (2,) float64.
    """
    customers = [n for n in nodes if n != 0]
    if not customers:
        return coord_array[0].copy()
    return coord_array[customers].mean(axis=0)


def find_route_neighbors(
    seed_idx: int,
    centroids: List[np.ndarray],
    k: int = 5,
    kdtree: Any = None,
    k_prox: int = 10,
) -> List[int]:
    """
    Find the k nearest route neighbors of a seed route using a KDTree.

    Queries the KDTree for k_prox candidates around the seed centroid, then
    returns the k nearest indices (including the seed itself at position 0,
    since KDTree includes the query point).

    Args:
        seed_idx: Index of the seed route in centroids.
        centroids: List of route centroid coordinate arrays.
        k: Number of neighbors to return (including the seed itself).
        kdtree: Pre-built scipy.spatial.KDTree over the centroid coordinates.
        k_prox: Number of candidates to fetch from the KDTree before selecting k.

    Returns:
        List of route indices (length <= k), sorted by ascending distance.
    """
    n_query = min(k_prox, len(centroids))
    _, indices = kdtree.query(np.asarray(centroids[seed_idx]), k=n_query)
    if np.ndim(indices) == 0:
        return [int(indices)]
    return list(indices[:k])


# ---------------------------------------------------------------------------
# Issue 2 — O(r) subproblem assembly via static adjacency list walk
# ---------------------------------------------------------------------------


def _assemble_subproblem(
    seed_slot: int,
    state: POPMUSICState,
    r: int,
) -> List[int]:
    """
    Assemble a subproblem of at most r parts by walking the adjacency list G.

    Time complexity: O(r) — independent of n (Issue 2 guarantee).

    The walk is a BFS/greedy expansion from seed_slot: at each step, the
    unvisited neighbor with the smallest d_prox (pre-sorted in G) is added
    until r parts are collected or G is exhausted.

    Args:
        seed_slot: Slot index of the seed part.
        state: Current POPMUSICState.
        r: Subproblem size (number of parts).

    Returns:
        List of slot indices forming the subproblem (length ≤ r).
    """
    selected: List[int] = [seed_slot]
    visited: Set[int] = {seed_slot}
    frontier: deque[int] = deque([seed_slot])

    while frontier and len(selected) < r:
        current = frontier.popleft()
        for neighbor in state.G.get(current, []):
            if neighbor not in visited and neighbor in state.active:
                visited.add(neighbor)
                selected.append(neighbor)
                frontier.append(neighbor)
                if len(selected) >= r:
                    break

    return selected


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
    vrpp: bool = True,
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
        vrpp: If True, nodes may be left unvisited (VRPP mode). Unvisited
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
            slot = cluster_to_slot.get(cluster_idx)
            if slot is None:
                continue
            neighbor_slots = [cluster_to_slot[nc] for nc in neighbor_clusters if nc in cluster_to_slot]
            state.G[slot] = neighbor_slots
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

    if vrpp:
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
            vrpp=vrpp,
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
            if vrpp:
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


def _evaluate_slots(
    slots: List[int],
    state: POPMUSICState,
    distance_matrix: np.ndarray,
    wastes_dict: Dict[int, float],
    R: float,
    C: float,
) -> float:
    """
    Compute net profit for a set of slots (routes + singletons).

    Singleton slots contribute revenue but zero travel cost (they are not
    currently routed). This is consistent with the VRPP objective where
    unvisited nodes generate no revenue.

    Args:
        slots: List of slot indices to evaluate.
        state: Current POPMUSICState.
        distance_matrix: Full distance matrix.
        wastes_dict: Waste levels per node.
        R: Revenue per unit waste.
        C: Cost per unit distance.

    Returns:
        Net profit (float).
    """
    total_cost = 0.0
    total_rev = 0.0

    for slot in slots:
        nodes = state.route_nodes.get(slot, [])
        customers = [n for n in nodes if n != 0]

        if slot in state.singleton_slots:
            # Singleton: contributes 0 profit (not yet routed)
            pass
        else:
            route_cost = get_route_cost(distance_matrix, nodes) * C
            route_rev = sum(wastes_dict.get(n, 0.0) * R for n in customers)
            total_cost += route_cost
            total_rev += route_rev

    return total_rev - total_cost


# ---------------------------------------------------------------------------
# Subproblem optimization (unchanged interface, internal only)
# ---------------------------------------------------------------------------


def _optimize_subproblem(
    base_solver: Optional[str],
    base_solver_config: Optional[Any],
    subproblem_nodes: List[int],
    distance_matrix: np.ndarray,
    wastes_dict: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    neighborhood_indices: List[int],
    mandatory: List[int],
    seed: int,
    vrpp: bool = True,
    profit_aware_operators: bool = False,
) -> Tuple[List[List[int]], float]:
    """
    Dispatch to the appropriate base solver for subproblem optimization.

    The subproblem_nodes list is guaranteed to satisfy the r-part invariant
    (Issue 3): it is exactly the union of customer nodes from r parts, where
    each part is either a proper route or a singleton unvisited node.

    Args:
        base_solver: "fast_tsp", "hgs", or "alns".
        base_solver_config: Solver-specific configuration.
        subproblem_nodes: Customer nodes in the subproblem (global indices).
        distance_matrix: Full distance matrix.
        wastes_dict: Waste levels per node.
        capacity: Vehicle capacity.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        neighborhood_indices: Slot indices of the subproblem parts (for
            HGS/ALNS vehicle count hinting).
        mandatory: Global mandatory node indices.
        seed: Random seed.
        vrpp: VRPP mode flag.
        profit_aware_operators: Profit-aware operator flag.

    Returns:
        Tuple of (new_routes, new_profit).
    """
    actual_config = base_solver_config
    if base_solver and base_solver_config:
        if hasattr(base_solver_config, base_solver):
            actual_config = getattr(base_solver_config, base_solver)
        elif isinstance(base_solver_config, dict) and base_solver in base_solver_config:
            actual_config = base_solver_config[base_solver]
            if isinstance(actual_config, list) and actual_config and isinstance(actual_config[0], dict):
                actual_config = actual_config[0]

    if base_solver == "fast_tsp" or base_solver is None:
        return _optimize_with_fast_tsp(
            subproblem_nodes,
            distance_matrix,
            wastes_dict,
            capacity,
            R,
            C,
            actual_config,
            1.0,
            seed,
            vrpp,
        )
    elif base_solver == "hgs":
        return _optimize_with_hgs(
            distance_matrix,
            wastes_dict,
            capacity,
            R,
            C,
            neighborhood_indices,
            mandatory,
            actual_config,
            1.0,
            seed,
            vrpp,
            profit_aware_operators,
            subproblem_nodes,
        )
    elif base_solver == "alns":
        return _optimize_with_alns(
            distance_matrix,
            wastes_dict,
            capacity,
            R,
            C,
            mandatory,
            actual_config,
            1.0,
            seed,
            vrpp,
            profit_aware_operators,
            subproblem_nodes,
        )
    else:
        raise ValueError(f"Unsupported base_solver: {base_solver!r}")


def _optimize_with_fast_tsp(
    subproblem_nodes: List[int],
    distance_matrix: np.ndarray,
    wastes_dict: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    config: Optional[Any],
    time_limit: float,
    seed: int,
    vrpp: bool = False,
) -> Tuple[List[List[int]], float]:
    """Optimize subproblem using FastTSP + LinearSplit.

    Args:
        subproblem_nodes: Subproblem nodes.
        distance_matrix: Distance matrix.
        wastes_dict: Wastes dictionary.
        capacity: Vehicle capacity.
        R: Revenue factor.
        C: Profit factor.
        config: FastTSP configuration.
        time_limit: Time limit.
        seed: Random seed.
        vrpp: VRPP mode flag.

    Returns:
        Tuple of (new_routes, new_profit).
    """
    actual_time_limit = time_limit
    if config and hasattr(config, "time_limit"):
        actual_time_limit = config.time_limit
    elif isinstance(config, dict):
        actual_time_limit = config.get("time_limit", actual_time_limit)

    new_tour = find_route(distance_matrix, subproblem_nodes, time_limit=actual_time_limit, seed=seed)
    giant_tour = [n for n in new_tour if n != 0]

    splitter = LinearSplit(
        dist_matrix=distance_matrix,
        wastes=wastes_dict,
        capacity=capacity,
        R=R,
        C=C,
        vrpp=vrpp,
    )
    return splitter.split(giant_tour)


def _optimize_with_hgs(
    distance_matrix: np.ndarray,
    wastes_dict: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    neighborhood_indices: List[int],
    mandatory: List[int],
    config: Optional[Any],
    time_limit: float,
    seed: int,
    vrpp: bool = True,
    profit_aware_operators: bool = False,
    subproblem_nodes: Optional[List[int]] = None,
) -> Tuple[List[List[int]], float]:
    """Optimize subproblem using Hybrid Genetic Search (HGS).

    Args:
        distance_matrix: Distance matrix.
        wastes_dict: Wastes dictionary.
        capacity: Vehicle capacity.
        R: Revenue factor.
        C: Profit factor.
        neighborhood_indices: Neighborhood indices.
        mandatory: Mandatory nodes.
        config: HGS configuration.
        time_limit: Time limit.
        seed: Random seed.
        vrpp: VRPP mode flag.
        profit_aware_operators: Profit-aware operator flag.
        subproblem_nodes: Subproblem nodes.

    Returns:
        Tuple of (new_routes, new_profit).
    """
    if subproblem_nodes is None:
        raise ValueError("subproblem_nodes must be provided for HGS optimization.")

    local_mandatory = [n for n in mandatory if n in subproblem_nodes]
    local_nodes = [0] + subproblem_nodes
    node_to_local = {node: i for i, node in enumerate(local_nodes)}

    local_dist = distance_matrix[np.ix_(local_nodes, local_nodes)]
    local_wastes = {node_to_local[n]: wastes_dict.get(n, 0.0) for n in subproblem_nodes}
    local_mandatory_idx = [node_to_local[n] for n in local_mandatory]

    if isinstance(config, HGSConfig):
        params = dataclasses.replace(config)
    elif isinstance(config, dict):
        valid_keys = {f.name for f in dataclasses.fields(HGSConfig)}
        params = HGSConfig(**{k: v for k, v in config.items() if k in valid_keys})
    else:
        params = HGSConfig(
            max_vehicles=max(1, len(neighborhood_indices)),
            time_limit=time_limit,
            vrpp=vrpp,
            profit_aware_operators=profit_aware_operators,
            seed=seed,
        )

    if params.max_vehicles <= 0:
        params.max_vehicles = max(1, len(neighborhood_indices))
    if params.time_limit < 0:
        params.time_limit = time_limit

    solver = HGSSolver(
        build_subproblem(
            dist_matrix=local_dist,
            wastes=local_wastes,
            capacity=capacity,
            revenue=R,
            cost_unit=C,
            mandatory_nodes=local_mandatory_idx,
        ),
        params,
    )
    local_routes, profit, _ = solver.solve()

    return (
        [[local_nodes[i] if i < len(local_nodes) else 0 for i in lr] for lr in local_routes],
        profit,
    )


def _optimize_with_alns(
    distance_matrix: np.ndarray,
    wastes_dict: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory: List[int],
    config: Optional[Any],
    time_limit: float,
    seed: int,
    vrpp: bool = True,
    profit_aware_operators: bool = False,
    subproblem_nodes: Optional[List[int]] = None,
) -> Tuple[List[List[int]], float]:
    """Optimize subproblem using Adaptive Large Neighborhood Search (ALNS).

    Args:
        distance_matrix: Distance matrix.
        wastes_dict: Wastes dictionary.
        capacity: Vehicle capacity.
        R: Revenue factor.
        C: Profit factor.
        mandatory: Mandatory nodes.
        config: ALNS configuration.
        time_limit: Time limit.
        seed: Random seed.
        vrpp: VRPP mode flag.
        profit_aware_operators: Profit-aware operator flag.
        subproblem_nodes: Subproblem nodes.

    Returns:
        Tuple of (new_routes, new_profit).
    """
    if subproblem_nodes is None:
        raise ValueError("subproblem_nodes must be provided for ALNS optimization.")

    local_mandatory = [n for n in mandatory if n in subproblem_nodes]
    local_nodes = [0] + subproblem_nodes
    node_to_local = {node: i for i, node in enumerate(local_nodes)}

    local_dist = distance_matrix[np.ix_(local_nodes, local_nodes)]
    local_wastes = {node_to_local[n]: wastes_dict.get(n, 0.0) for n in subproblem_nodes}
    local_mandatory_idx = [node_to_local[n] for n in local_mandatory]

    if config:
        if isinstance(config, ALNSConfig):
            params = dataclasses.replace(config)
        elif isinstance(config, dict):
            valid_keys = {f.name for f in dataclasses.fields(ALNSConfig)}
            params = ALNSConfig(**{k: v for k, v in config.items() if k in valid_keys})
        else:
            params = ALNSConfig(
                max_iterations=1000,
                time_limit=time_limit,
                vrpp=vrpp,
                profit_aware_operators=profit_aware_operators,
                seed=seed,
            )
    else:
        params = ALNSConfig(max_iterations=1000, time_limit=time_limit, vrpp=vrpp)

    if params.time_limit < 0:
        params.time_limit = time_limit

    solver = ALNSSolver(
        build_subproblem(
            dist_matrix=local_dist,
            wastes=local_wastes,
            capacity=capacity,
            revenue=R,
            cost_unit=C,
            mandatory_nodes=local_mandatory_idx,
        ),
        params,
    )
    local_routes, profit, _ = solver.solve()

    return (
        [[local_nodes[i] if i < len(local_nodes) else 0 for i in lr] for lr in local_routes],
        profit,
    )

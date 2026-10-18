"""p-median initialisation and proximity network of POPMUSIC.

Attributes:
    _initialize_pmedian: Initial routes from an alternating p-median clustering.
    _build_proximity_network: Route proximity graph used to pick sub-problems.

Example:
    >>> routes = _initialize_pmedian(dist, wastes, capacity, R, C, n_routes, rng)
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


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

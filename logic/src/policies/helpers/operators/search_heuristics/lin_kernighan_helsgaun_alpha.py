"""Minimum 1-tree, subgradient optimization, and alpha-nearness candidates for LKH."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def compute_alpha_measures(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute α-nearness for every edge using minimum-spanning-tree sensitivity.

    α(i, j) = c(i, j) − β(i, j),  where β(i, j) is the weight of the heaviest
    edge on the unique MST path between i and j (Helsgaun 2000, Section 4.1).

    Edges with α = 0 belong to some MST; edges with small α are "nearly
    spanning" and hence strong candidates for an optimal tour.

    Args:
        distance_matrix: (n × n) symmetric cost matrix.

    Returns:
        (n × n) array of α-values (symmetric, non-negative).
    """
    n = len(distance_matrix)
    mst_sparse = minimum_spanning_tree(distance_matrix)
    # Convert directed result to symmetric dense MST
    mst = mst_sparse.toarray()
    mst = np.maximum(mst, mst.T)

    # Build explicit adjacency list for faster MST traversal (fixes O(V^3) bottleneck)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    rows, cols = np.where(mst > 0)
    for r, c in zip(rows, cols, strict=False):
        adj[r].append((c, float(mst[r, c])))

    alpha = np.zeros((n, n), dtype=float)

    for i in range(n):
        # BFS to find max edge weight from root i to all other nodes
        max_edge_to = np.zeros(n, dtype=float)
        visited = np.zeros(n, dtype=bool)
        queue: Deque[int] = deque([i])
        visited[i] = True

        while queue:
            current = queue.popleft()
            for neighbor, weight in adj[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    # Max edge on path to neighbor is max(max edge to current, weight)
                    max_edge_to[neighbor] = max(max_edge_to[current], weight)
                    queue.append(neighbor)

        # After BFS for root i, populate alpha row for pairs (i, j)
        for j in range(i + 1, n):
            alpha_val = distance_matrix[i, j] - max_edge_to[j]
            alpha[i, j] = alpha[j, i] = max(0.0, alpha_val)

    return alpha


def get_candidate_set(
    distance_matrix: np.ndarray,
    alpha_measures: np.ndarray,
    max_candidates: int = 5,
) -> Dict[int, List[int]]:
    """
    Build per-node candidate lists sorted by α-nearness (ties broken by cost).

    Restricting the inner LK search to these lists reduces worst-case
    complexity from O(n²) to O(n · max_candidates) per pass.

    Args:
        distance_matrix: (n × n) cost matrix.
        alpha_measures: (n × n) α-nearness matrix from :func:`compute_alpha_measures`.
        max_candidates: Maximum number of candidates per node (default 5).

    Returns:
        dict mapping each node index to its sorted candidate list.
    """
    n = len(distance_matrix)
    candidates: Dict[int, List[int]] = {}
    for i in range(n):
        indices = sorted(
            [j for j in range(n) if j != i],
            key=lambda j: (alpha_measures[i, j], distance_matrix[i, j]),
        )
        candidates[i] = indices[:max_candidates]
    return candidates


def compute_1tree(
    penalized_distances: np.ndarray,
    root: int = 0,
) -> Tuple[float, np.ndarray]:
    """
    Compute a minimum 1-tree rooted at ``root`` (Held & Karp 1970).

    A 1-tree consists of:
    - The minimum spanning tree (MST) on all nodes *except* the root.
    - The two cheapest edges incident to the root.

    Its cost is a valid lower bound on the TSP optimum.

    Args:
        penalized_distances: (n × n) penalty-modified cost matrix
                             c̃_{ij} = c_{ij} + π_i + π_j.
        root:                Root node (default 0).

    Returns:
        (tree_weight, degree_array):
            tree_weight  — total edge weight of the 1-tree.
            degree_array — integer array of shape (n,); degree[i] = degree
                           of node i in the 1-tree.
    """
    n = len(penalized_distances)

    # Compute MST on nodes {0, ..., n-1} \ {root}
    non_root = [i for i in range(n) if i != root]
    sub_dist = penalized_distances[np.ix_(non_root, non_root)]
    mst_sparse = minimum_spanning_tree(sub_dist)
    mst_dense = mst_sparse.toarray()
    mst_dense = np.maximum(mst_dense, mst_dense.T)

    degree = np.zeros(n, dtype=np.int32)
    tree_weight = 0.0

    # Accumulate MST edges (re-indexed to global node indices)
    for local_i, global_i in enumerate(non_root):
        for local_j, global_j in enumerate(non_root):
            if local_j > local_i and mst_dense[local_i, local_j] > 1e-12:
                w = float(mst_dense[local_i, local_j])
                tree_weight += w
                degree[global_i] += 1
                degree[global_j] += 1

    # Add the two shortest edges incident to root
    root_edges = sorted(
        [(penalized_distances[root, j], j) for j in range(n) if j != root],
        key=lambda x: x[0],
    )
    for w, j in root_edges[:2]:
        tree_weight += w
        degree[root] += 1
        degree[j] += 1

    return tree_weight, degree


def run_subgradient(
    distance_matrix: np.ndarray,
    max_iter: int = 100,
    mu_init: float = 1.0,
    patience: int = 20,
    root: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Maximise the Held-Karp lower bound W(π) via subgradient optimisation.

    At each step:
    1. Compute c̃_{ij} = c_{ij} + π_i + π_j (penalized distances).
    2. Compute minimum 1-tree weight W̃ and node degrees d_i.
    3. Recover the un-penalized bound: W(π) = W̃ + 2 · Σ_i π_i.
    4. Subgradient: g_i = d_i − 2 (positive → overconstrained).
    5. Polyak step: t = μ · (W* − W(π_current)) / ||g||²  where W* is the
       best known lower bound (updated greedily as we improve).
    6. Update: π ← π + t · g  (no non-negativity constraint; π can be negative).
    7. Halve μ if no improvement for ``patience`` consecutive iterations.

    Args:
        distance_matrix: (n x n) symmetric raw distance matrix.
        max_iter: Maximum subgradient iterations.
        mu_init: Initial step-size multiplier (μ₀ in Helsgaun 2000).
        patience: Iterations without improvement before halving μ.
        root: 1-tree root node (typically 0).
        verbose: Print per-iteration diagnostics.

    Returns:
        Tuple[np.ndarray, float, np.ndarray]: (pi, best_W, penalized_dm)
            - pi: Optimal penalty vector, shape (n,).
            - best_W: Best (highest) 1-tree lower bound achieved.
            - penalized_dm: Final penalized distance matrix c̃_{ij}.
    """
    n = len(distance_matrix)
    pi = np.zeros(n, dtype=float)
    best_W = -np.inf
    mu = mu_init
    no_improve_count = 0

    for k in range(max_iter):
        # Build penalized matrix: c̃_ij = c_ij + π_i + π_j
        pi_mat = pi[:, None] + pi[None, :]
        penalized = distance_matrix + pi_mat

        # 1-tree lower bound in penalized space
        W_tilde, degree = compute_1tree(penalized, root)
        W_pi = W_tilde + 2.0 * np.sum(pi)

        if W_pi > best_W + 1e-9:
            best_W = W_pi
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                mu /= 2.0
                no_improve_count = 0
                if mu < 1e-8:
                    break  # Converged

        # Subgradient: g_i = d_i(T) - 2  (degree defect)
        g = degree.astype(float) - 2.0
        g_sq = float(np.dot(g, g))
        if g_sq < 1e-12:
            break  # Perfect tour degree (all nodes have degree 2)

        # Polyak step size
        step = mu * max(W_pi - best_W, 1e-6) / g_sq
        # Avoid oscillation: clamp step to reasonable range
        step = min(step, 10.0 / max(np.abs(g).max(), 1.0))
        pi += step * g

        if verbose:
            print(f"  [subgradient k={k:3d}] W(π)={W_pi:.4f}  best_W={best_W:.4f}  μ={mu:.6f}")

    # Final penalized matrix
    pi_mat = pi[:, None] + pi[None, :]
    penalized_dm = distance_matrix + pi_mat
    return pi, best_W, penalized_dm

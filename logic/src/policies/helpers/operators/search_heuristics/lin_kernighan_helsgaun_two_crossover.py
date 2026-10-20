"""Iterative Partial Transcription (IPT) crossover operator for LKH-2."""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.helpers.operators.search_heuristics._objective import get_cost
from logic.src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun import (
    _improve_tour,
)

from .lin_kernighan_helsgaun_two_population import TourPopulation


def ipt_crossover(  # noqa: C901
    parent1: List[int],
    parent2: List[int],
    distance_matrix: np.ndarray,
    candidates: Dict[int, List[int]],
    stdlib_rng: Random,
    max_k: int = 5,
    max_gap_iter: int = 20,
) -> Tuple[List[int], float]:
    """
    Iterative Partial Transcription (IPT) crossover operator (Helsgaun 2009).

    Algorithm:
    1. Compute the "backbone" B = edges present in BOTH parent1 and parent2.
    2. Extract the connected backbone chains (maximal paths/cycles whose edges
       are all in B).  Nodes not covered by any backbone edge are "free".
    3. Assemble an offspring tour:
       a. Start from a chain endpoint (or any free node).
       b. Walk along backbone chains greedily, then bridge each gap between
          chain endpoints and free nodes using the nearest-neighbour rule.
       c. Ensure each node is visited exactly once.
    4. Refine the offspring via LKH-1 k-opt with ``fixed_edges=backbone``
       so that backbone edges are never removed during gap-filling.
    5. If no backbone exists (parents share no edges), return the better parent.

    The use of ``fixed_edges`` is the key departure from a plain nearest-
    neighbour re-assembly: it preserves the parental backbone structure
    throughout the refinement step, faithfully implementing the IPT algorithm
    of Helsgaun (2009, Section 3).

    Args:
        parent1: First parent (closed tour, starts and ends at 0).
        parent2: Second parent (closed tour, starts and ends at 0).
        distance_matrix: (n x n) raw cost matrix.
        candidates: α-candidate lists (penalized).
        stdlib_rng: Random number generator.
        max_k: k-opt depth for gap-filling local search.
        max_gap_iter: Maximum improvement passes for each gap.

    Returns:
        Tuple[List[int], float]: (offspring_tour, offspring_cost).
    """
    n = len(parent1) - 1  # number of real nodes

    # --- 1. Compute backbone (edges common to both parents) ---
    es1 = TourPopulation._to_edge_set(parent1)
    es2 = TourPopulation._to_edge_set(parent2)
    backbone: Set[Tuple[int, int]] = es1 & es2

    if not backbone:
        # No shared edges: return the better parent unchanged
        c1 = get_cost(parent1, distance_matrix)
        c2 = get_cost(parent2, distance_matrix)
        return (parent1[:], c1) if c1 <= c2 else (parent2[:], c2)

    # --- 2. Build backbone adjacency and extract chains ---
    # Adjacency in the backbone sub-graph
    bb_adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for a, b in backbone:
        bb_adj[a].append(b)
        bb_adj[b].append(a)

    # Find chain endpoints: nodes with backbone degree 1 (endpoints) or 0 (free)
    # Nodes with backbone degree 2 are interior chain nodes.
    visited = [False] * n
    chains: List[List[int]] = []  # each chain is an ordered list of nodes

    for start in range(n):
        if visited[start] or len(bb_adj[start]) == 0:
            continue

        # Walk the chain starting from 'start'
        chain = [start]
        visited[start] = True
        prev = -1
        curr = start

        while True:
            # Prefer the unvisited neighbour that is not the previous node
            next_node = None
            for nb in bb_adj[curr]:
                if nb != prev and not visited[nb]:
                    next_node = nb
                    break

            if next_node is None:
                # No more unvisited neighbours: chain terminates
                # Check if the last node connects back to start (closed loop)
                if start in bb_adj[curr] and start != prev and len(chain) > 2:
                    chain.append(start)  # Mark it as closed by re-appending start
                break

            visited[next_node] = True
            chain.append(next_node)
            prev = curr
            curr = next_node

        if len(chain) >= 2:
            chains.append(chain)

    # --- 3. Assemble offspring by stitching chains and free nodes ---
    # Nodes already placed by backbone chains
    placed_nodes: Set[int] = set()
    for chain in chains:
        for node in chain:
            if node != chain[0] or chain[-1] != chain[0]:  # avoid double-adding closed chain start
                placed_nodes.add(node)
        # For closed chains (cycle), the start appears twice but it IS placed
        if chains and chain[0] == chain[-1]:
            placed_nodes.add(chain[0])

    # Re-compute properly: any node that appears in any chain is placed
    placed_nodes = set()
    for chain in chains:
        for node in chain:
            placed_nodes.add(node)

    free_nodes = set(range(n)) - placed_nodes

    # Build offspring as an ordered list (open path, will be closed at end)
    offspring_path: List[int] = []

    # If the whole tour is a single closed backbone cycle, we're done
    if len(chains) == 1 and chains[0][0] == chains[0][-1] and len(placed_nodes) == n:
        offspring = chains[0]
        # Ensure it starts at 0
        if offspring[0] != 0:
            idx = offspring.index(0)
            offspring = offspring[idx:] + offspring[1 : idx + 1]
            if offspring[-1] != 0:
                offspring.append(0)
        return offspring, get_cost(offspring, distance_matrix)

    # Greedy stitching: walk chains in order and bridge gaps via nearest-neighbour
    remaining_chains = list(chains)
    remaining_free = set(free_nodes)

    # Start with the chain containing the depot (node 0) if possible
    depot_chain_idx = None
    for idx, chain in enumerate(remaining_chains):
        if 0 in chain:
            depot_chain_idx = idx
            break

    if depot_chain_idx is not None:
        first_chain = remaining_chains.pop(depot_chain_idx)
    elif remaining_chains:
        first_chain = remaining_chains.pop(0)
    else:
        first_chain = []

    # If chain starts with depot, use as-is; otherwise rotate so depot is first
    if first_chain and first_chain[0] != 0 and 0 in first_chain:
        rot = first_chain.index(0)
        first_chain = first_chain[rot:] + first_chain[:rot]

    offspring_path = list(first_chain)
    # For closed chains (cycle detected), strip the duplicated last node
    if len(offspring_path) > 1 and offspring_path[0] == offspring_path[-1]:
        offspring_path = offspring_path[:-1]

    def nearest_free(current: int, free: Set[int]) -> int:
        """
        Return the nearest free node to `current`.

        Args:
            current: Current node index.
            free: Set of available free nodes.

        Returns:
            int: Index of the nearest free node.
        """
        return min(free, key=lambda v: distance_matrix[current, v])

    def nearest_chain_endpoint(current: int, chains_left: List[List[int]]) -> Tuple[int, int, bool]:
        """
        Find the closest chain endpoint to `current`.

        Args:
            current: Current node index.
            chains_left: List of remaining backbone chains.

        Returns:
            Tuple[int, int, bool]: (chain_idx, end_which, reversed)
                - chain_idx: index in chains_left
                - end_which: 0 = connect to chain[0], 1 = connect to chain[-1]
                - reversed:  whether to traverse the chain in reverse
        """
        best_dist = float("inf")
        best = (0, 0, False)
        for ci, ch in enumerate(chains_left):
            d0 = distance_matrix[current, ch[0]]
            d1 = distance_matrix[current, ch[-1]]
            if d0 < best_dist:
                best_dist = d0
                best = (ci, 0, False)  # attach ch forward: ch[0] connected to current
            if d1 < best_dist:
                best_dist = d1
                best = (ci, 1, True)  # attach ch backward: ch[-1] connected to current
        return best

    # Greedy stitching loop
    while remaining_chains or remaining_free:
        current = offspring_path[-1] if offspring_path else 0

        # Prefer adding a nearby chain endpoint over a lone free node
        nearest_chain_dist = float("inf")
        nearest_free_dist = float("inf")

        if remaining_chains:
            ci, _, _ = nearest_chain_endpoint(current, remaining_chains)
            ch = remaining_chains[ci]
            nearest_chain_dist = min(
                distance_matrix[current, ch[0]],
                distance_matrix[current, ch[-1]],
            )
        if remaining_free:
            nf = nearest_free(current, remaining_free)
            nearest_free_dist = distance_matrix[current, nf]

        if remaining_chains and nearest_chain_dist <= nearest_free_dist:
            # Attach the nearest chain
            ci, end_which, reversed_chain = nearest_chain_endpoint(current, remaining_chains)
            ch = remaining_chains.pop(ci)
            if reversed_chain:
                ch = list(reversed(ch))
            # Strip the connecting node from the chain if it duplicates the path end
            if offspring_path and ch[0] == offspring_path[-1]:
                ch = ch[1:]
            offspring_path.extend(ch)
        else:
            # Insert nearest free node
            nf = nearest_free(current, remaining_free)
            remaining_free.discard(nf)
            offspring_path.append(nf)

    # --- 4. Close the tour and ensure it starts at depot (0) ---
    # Rotate to start at node 0
    if offspring_path and offspring_path[0] != 0 and 0 in offspring_path:
        rot = offspring_path.index(0)
        offspring_path = offspring_path[rot:] + offspring_path[:rot]

    # Guarantee each non-depot node appears exactly once (safety net)
    seen: Set[int] = set()
    deduped: List[int] = []
    for node in offspring_path:
        if node not in seen:
            seen.add(node)
            deduped.append(node)
    # Add any missing nodes (should not happen in correct implementations)
    for node in range(n):
        if node not in seen:
            deduped.append(node)
    offspring_path = deduped

    # Close the tour
    offspring = offspring_path + [offspring_path[0]]

    # --- 5. Refine with backbone-preserving LKH-1 k-opt ---
    offspring_cost = get_cost(offspring, distance_matrix)
    dont_look: Optional[np.ndarray] = np.zeros(n, dtype=bool)
    # Pass backbone as fixed_edges so refinement does not remove backbone edges
    backbone_fixed: Set[Tuple[int, int]] = backbone  # already sorted pairs
    for _ in range(max_gap_iter):
        offspring, offspring_cost, improved, dont_look = _improve_tour(
            offspring,
            offspring_cost,
            candidates,
            distance_matrix,
            stdlib_rng,
            dont_look,
            max_k,
            fixed_edges=backbone_fixed,
        )
        if not improved:
            break

    return offspring, offspring_cost

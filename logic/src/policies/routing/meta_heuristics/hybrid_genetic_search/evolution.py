"""
Evolutionary operators for Hybrid Genetic Search (HGS).

This module contains the crossover, evaluation, and fitness calculation
logic for maintaining and improving the HGS population.

Attributes:
    update_biased_fitness: Updates biased fitness for a subpopulation.
    evaluate: Decodes giant tour and calculates metrics.

Example:
    >>> evaluate(individual, split_manager)
    >>> update_biased_fitness(population, nb_elite=4)
"""

from typing import List, Optional

import numpy as np

from .individual import Individual
from .split import LinearSplit


def _extract_edges(individual: Individual) -> set:
    """
    Extract all undirected edges from an individual's routes.

    An edge {A, B} exists if node B immediately follows node A in a route
    (or vice-versa), including edges from/to the depot (node 0).
    Using frozenset ensures edges are undirected (Fix 8).

    Args:
        individual: Individual whose edges to extract.

    Returns:
        Set of undirected edges (frozensets of node pairs).
    """
    edges = set()
    for route in individual.routes:
        if not route:
            continue
        # Edge from depot to first node
        edges.add(frozenset({0, route[0]}))
        # Edges within route
        for i in range(len(route) - 1):
            edges.add(frozenset({route[i], route[i + 1]}))
        # Edge from last node back to depot
        edges.add(frozenset({route[-1], 0}))
    return edges


def _compute_broken_pairs_distance(ind1: Individual, ind2: Individual) -> float:
    """
    Compute the broken pairs distance (edge-based diversity) between two individuals.

    Following Vidal (2022), the distance is defined as:
    Δ(A,B) = 1 - |E(A) ∩ E(B)| / max(|E(A)|, |E(B)|)

    Args:
        ind1: First individual.
        ind2: Second individual.

    Returns:
        Broken pairs distance between 0.0 and 1.0.
    """
    edges1 = _extract_edges(ind1)
    edges2 = _extract_edges(ind2)

    if not edges1 and not edges2:
        return 0.0

    if not edges1 or not edges2:
        return 1.0

    intersection = len(edges1 & edges2)
    max_edges = max(len(edges1), len(edges2))

    return 1.0 - (intersection / max_edges)


def update_biased_fitness(
    population: List[Individual],
    nb_elite: int,
    neighbor_size: int = 5,
    distance_cache: Optional[dict] = None,
    inv: Optional[dict] = None,
):
    """
    Update biased fitness based on profit rank and diversity rank.
    Follows Vidal et al. (2022) HGS-CVRP implementation with parameterless
    diversity weighting.

    Args:
        population: List of individuals to update (single subpopulation).
        nb_elite: Number of elite individuals to protect.
        neighbor_size: Number of nearest neighbors (nbClose) to consider for diversity.
        distance_cache: Optional cache for pairwise diversity distances.
        inv: Optional inverse index mapping id(ind) -> set of cache keys involving
             that individual. When provided alongside distance_cache, newly computed
             distances are registered in the index so _evict_cache can purge them
             in O(P) instead of O(P²).

    Returns:
        None.
    """
    if not population:
        return
    pop_size = len(population)

    population.sort(key=lambda x: x.penalized_profit, reverse=True)
    for i, ind in enumerate(population):
        ind.rank_profit = i + 1

    n = len(population)
    dist_matrix_local = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if distance_cache is not None:
                key = (id(population[i]), id(population[j]))
                if key not in distance_cache:
                    d = _compute_broken_pairs_distance(population[i], population[j])
                    rev_key = (id(population[j]), id(population[i]))
                    distance_cache[key] = d
                    distance_cache[rev_key] = d
                    # Register both directions in the inverse index so eviction
                    # can find and remove these entries in O(P) per individual.
                    if inv is not None:
                        inv.setdefault(id(population[i]), set()).add(key)
                        inv.setdefault(id(population[i]), set()).add(rev_key)
                        inv.setdefault(id(population[j]), set()).add(key)
                        inv.setdefault(id(population[j]), set()).add(rev_key)
                else:
                    d = distance_cache[key]
            else:
                d = _compute_broken_pairs_distance(population[i], population[j])

            dist_matrix_local[i][j] = d
            dist_matrix_local[j][i] = d

    for i, ind in enumerate(population):
        dists = sorted(dist_matrix_local[i][j] for j in range(n) if j != i)
        if dists:
            n_neighbors = min(neighbor_size, len(dists))
            ind.dist_to_parents = float(np.mean(dists[:n_neighbors]))
        else:
            ind.dist_to_parents = 0.0

    population.sort(key=lambda x: x.dist_to_parents, reverse=True)
    for i, ind in enumerate(population):
        ind.rank_diversity = i + 1

    diversity_weight = max(0.0, 1.0 - (nb_elite / pop_size))  # negative when pop_size < nb_elite
    for ind in population:
        ind.fitness = ind.rank_profit + diversity_weight * ind.rank_diversity


def evaluate(ind: Individual, split_manager: LinearSplit, penalty_capacity: float = 1.0):
    """
    Decode the ACTIVE sequence and calculate metrics, including feasibility and penalties.

    This function transforms the genotype (active_sequence) into the phenotype (routes)
    using the Split algorithm. For VRPP, the Split algorithm may skip unprofitable nodes,
    resulting in routes that contain a subset of the nodes in active_sequence.

    The active_sequence is preserved unchanged to maintain genetic material for crossover.
    Nodes not in routes are considered "unvisited" and remain available for future
    insertion by local search operators.

    Args:
        ind: Individual to evaluate.
        split_manager: Split algorithm manager.
        penalty_capacity: Penalty coefficient for capacity violations.

    Returns:
        None.
    """
    # 1. Isolate the active sequence
    if ind.routes:
        # If routes exist (e.g., from RP-GPX), extract only the active nodes in sequence order
        active_set = {n for route in ind.routes for n in route}
        active_sequence = [n for n in ind.giant_tour if n in active_set]
    else:
        # Fallback for initialization or pure OX crossover:
        # Pre-filter the giant_tour using a standalone break-even hurdle.
        # If revenue doesn't cover the baseline to/from depot cost, exclude it from the Split DP.
        active_sequence = []
        for n in ind.giant_tour:
            rev = split_manager.wastes.get(n, 0) * split_manager.R
            cost = (split_manager.dist_matrix[0, n] + split_manager.dist_matrix[n, 0]) * split_manager.C
            if rev >= cost or n in split_manager.mandatory_nodes:
                active_sequence.append(n)

    # 2. Feed ONLY the active sequence to the hostage-prone Split DP
    routes, profit = split_manager.split(active_sequence)
    ind.routes = routes
    ind.profit_score = profit

    # Calculate cost/revenue and check feasibility
    rev = 0.0
    cost = 0.0
    total_capacity_violation = 0.0

    for r in routes:
        if not r:
            continue

        # Calculate route distance and revenue
        d = split_manager.dist_matrix[0, r[0]]
        route_load = split_manager.wastes[r[0]]

        for i in range(len(r) - 1):
            d += split_manager.dist_matrix[r[i], r[i + 1]]
            route_load += split_manager.wastes[r[i + 1]]

        d += split_manager.dist_matrix[r[-1], 0]
        cost += d * split_manager.C

        # Calculate revenue for all nodes in route
        for node in r:
            rev += split_manager.wastes[node] * split_manager.R

        # Check capacity constraint
        if route_load > split_manager.capacity:
            total_capacity_violation += route_load - split_manager.capacity

    ind.cost = cost
    ind.revenue = rev
    ind.capacity_violation = total_capacity_violation
    ind.is_feasible = total_capacity_violation == 0.0

    # Penalized cost includes penalty for capacity violations
    ind.penalized_cost = cost + penalty_capacity * total_capacity_violation

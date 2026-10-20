"""Genetic Algorithm (GA) TSP solver for the cluster-first route-second algorithm.

These components were split out of ``tsp_metaheuristics``: the enhanced edge
recombination (EER) crossover operator and the ``find_route_ga`` solver.
"""

import random
import time
from typing import Dict, List, Set

import numpy as np

from logic.src.policies.routing.matheuristics.cluster_first_route_second.tsp_local_search import (
    _calculate_tour_distance,
)


def _eer_crossover(p1: List[int], p2: List[int], customers: List[int]) -> List[int]:
    """Enhanced Edge Recombination (EER) crossover.

    Args:
        p1: First parent tour (customer nodes only, no depot).
        p2: Second parent tour (customer nodes only, no depot).
        customers: List of customer node indices.

    Returns:
        List[int]: Child tour produced by EER crossover.
    """
    n_cust = len(customers)
    edge_table: Dict[int, Set[int]] = {node: set() for node in customers}
    for parent in [p1, p2]:
        for i in range(n_cust):
            edge_table[parent[i]].add(parent[(i - 1) % n_cust])
            edge_table[parent[i]].add(parent[(i + 1) % n_cust])

    child: List[int] = []
    curr = random.choice([p1[0], p2[0]])
    while len(child) < n_cust:
        child.append(curr)
        for neighbors in edge_table.values():
            neighbors.discard(curr)

        if len(child) == n_cust:
            break

        neighbors = list(edge_table[curr])  # type: ignore[assignment]
        if not neighbors:
            remaining = [node for node in customers if node not in child]
            curr = random.choice(remaining)
        else:
            min_neighbors = min(len(edge_table[n]) for n in neighbors)
            candidates = [n for n in neighbors if len(edge_table[n]) == min_neighbors]
            curr = random.choice(candidates)
    return child


def find_route_ga(
    distance_matrix: np.ndarray,
    cluster: List[int],
    time_limit: float = 60.0,
    seed: int = 42,
    pop_size: int = 50,
    mutation_rate: float = 0.2,
) -> List[int]:
    """
    Solve TSP using Genetic Algorithm (GA).

    Args:
        distance_matrix: Pre-computed all-pairs distance matrix.
        cluster: List of node indices.
        time_limit: Maximum time.
        seed: Random seed.
        pop_size: Population size.
        mutation_rate: Mutation probability.

    Returns:
        List[int]: Best TSP tour found as a list of node indices.
    """
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()

    if not cluster:
        return [0, 0]
    if len(cluster) <= 3:
        return list(cluster) + [0] if cluster[-1] != 0 else cluster

    customers = [node for node in cluster if node != 0]
    n_cust = len(customers)

    def _get_tour(perm):
        return [0] + perm + [0]

    def _get_dist(perm):
        return _calculate_tour_distance(_get_tour(perm), distance_matrix)

    # Initialize population
    population = []
    for _ in range(pop_size):
        perm = np.random.permutation(customers).tolist()
        population.append(perm)

    best_perm = min(population, key=_get_dist)
    best_dist = _get_dist(best_perm)

    while time.time() - start_time < time_limit:
        population.sort(key=_get_dist)
        new_pop = population[:2]  # Elitism

        while len(new_pop) < pop_size:
            if time.time() - start_time >= time_limit:
                break
            # Selection (Tournament)
            p1 = min(random.sample(population, min(5, len(population))), key=_get_dist)
            p2 = min(random.sample(population, min(5, len(population))), key=_get_dist)

            # Crossover (Enhanced Edge Recombination - EER)
            # Ref: Sultana & Akhand (2017) specify EER for adjacency preservation
            child = _eer_crossover(p1, p2, customers)

            # Mutation (Swap)
            if random.random() < mutation_rate and n_cust >= 2:
                i1, i2 = random.sample(range(n_cust), 2)
                child[i1], child[i2] = child[i2], child[i1]

            new_pop.append(child)

            d = _get_dist(child)
            if d < best_dist:
                best_dist = d
                best_perm = list(child)

        population = new_pop

    return _get_tour(best_perm)

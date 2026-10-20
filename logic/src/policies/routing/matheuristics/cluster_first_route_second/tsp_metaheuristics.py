"""
TSP Metaheuristic Solvers for Cluster-First Route-Second Algorithm.

This module provides metaheuristic algorithms for solving the Traveling Salesman Problem
in the routing phase of the VFJ algorithm.

Supported Metaheuristics (as evaluated in Sultana & Akhand, 2017):
- Particle Swarm Optimization (PSO) - FULLY IMPLEMENTED
- Ant Colony Optimization (ACO) - Placeholder
- Genetic Algorithm (GA) - Placeholder

The paper concludes that VFJ + PSO provides the best performance for CVRP benchmarks.

Reference:
    Sultana, T., Akhand, M. A. H., & Rahman, M. M. H. (2017).
    Section 4: "VFJ algorithm with PSO performs better than VFJ algorithm with
    ACO and VFJ algorithm with GA"

The Genetic Algorithm components moved to :mod:`tsp_genetic` and the shared
local-search helpers moved to :mod:`tsp_local_search`; this module re-exports
``find_route_ga`` and imports the shared helpers for internal use.

Attributes:
    find_route_pso: TSP solver using Velocity Tentative PSO.
    find_route_aco: TSP solver using Ant Colony Optimization.
    find_route_ga: TSP solver using a Genetic Algorithm.

Example:
    >>> import numpy as np
    >>> from logic.src.policies.routing.matheuristics.cluster_first_route_second.tsp_metaheuristics import find_route_pso
    >>> dm = np.zeros((3, 3))
    >>> tour = find_route_pso(dm, [0, 1, 2])
"""

import random
import time
from typing import List, Optional

import numpy as np

from logic.src.policies.routing.matheuristics.cluster_first_route_second.tsp_genetic import (
    find_route_ga as find_route_ga,
)
from logic.src.policies.routing.matheuristics.cluster_first_route_second.tsp_local_search import (
    _calculate_tour_distance,
    _get_swap_operators,
)


def find_route_pso(
    distance_matrix: np.ndarray,
    cluster: List[int],
    time_limit: float = 60.0,
    seed: int = 42,
    n_particles: int = 50,
    c1: float = 1.5,
    c2: float = 1.5,
) -> List[int]:
    """
    Solve TSP using Velocity Tentative PSO (VTPSO).

    Ref: Akhand et al. (2015); Sultana & Akhand (2017). VTPSO computes a
    tentative velocity and uses partial search to explore intermediate
    tours, selecting the best one. This methodology enables the mapping
    of continuous velocity vectors to discrete TSP topologies.

    Args:
        distance_matrix: Pre-computed all-pairs distance matrix.
        cluster: List of node indices to visit (including depot at index 0).
        time_limit: Maximum time in seconds.
        seed: Random seed.
        n_particles: Number of particles.
        c1: Cognitive coefficient.
        c2: Social coefficient.

    Returns:
        TSP tour as list of node indices [0, i1, i2, ..., in, 0].
    """
    np.random.seed(seed)
    random.seed(seed)
    start_time = time.time()

    if not cluster:
        return [0, 0]
    if len(cluster) <= 3:
        return list(cluster) + [0] if cluster[-1] != 0 else cluster

    customers = [node for node in cluster if node != 0]

    # Initialize particles
    particles = []
    for _ in range(n_particles):
        perm = np.random.permutation(customers).tolist()
        tour = [0] + perm + [0]
        particles.append(
            {"pos": tour, "pbest": tour.copy(), "pbest_dist": _calculate_tour_distance(tour, distance_matrix)}
        )

    gbest_tour = min(particles, key=lambda p: p["pbest_dist"])["pbest"].copy()
    gbest_dist = _calculate_tour_distance(gbest_tour, distance_matrix)

    while time.time() - start_time < time_limit:
        for p in particles:
            if time.time() - start_time >= time_limit:
                break

            # 1. Calculate Tentative Velocity
            v_pbest = _get_swap_operators(p["pos"], p["pbest"])
            v_gbest = _get_swap_operators(p["pos"], gbest_tour)

            # Stochastic selection of swaps
            v_pbest = [s for s in v_pbest if random.random() < 0.5]  # Simplified for VTPSO
            v_gbest = [s for s in v_gbest if random.random() < 0.5]

            tentative_velocity = v_pbest + v_gbest

            # 2. VTPSO Partial Search
            current_tour = list(p["pos"])
            best_inter_tour = list(current_tour)
            best_inter_dist = _calculate_tour_distance(best_inter_tour, distance_matrix)

            for idx1, idx2 in tentative_velocity:
                current_tour[idx1], current_tour[idx2] = current_tour[idx2], current_tour[idx1]
                d = _calculate_tour_distance(current_tour, distance_matrix)
                if d < best_inter_dist:
                    best_inter_dist = d
                    best_inter_tour = list(current_tour)

            p["pos"] = best_inter_tour

            # Update personal best
            if best_inter_dist < p["pbest_dist"]:
                p["pbest"] = list(p["pos"])
                p["pbest_dist"] = best_inter_dist

                # Update global best
                if best_inter_dist < gbest_dist:
                    gbest_tour = list(p["pbest"])
                    gbest_dist = best_inter_dist

    return gbest_tour


def find_route_aco(
    distance_matrix: np.ndarray,
    cluster: List[int],
    time_limit: float = 60.0,
    seed: int = 42,
    n_ants: Optional[int] = None,
    alpha: float = 1.0,
    beta: float = 3.0,
    rho: float = 0.1,
    q: float = 100.0,
) -> List[int]:
    """
    Solve TSP using Ant Colony Optimization (ACO).

    Args:
        distance_matrix: Pre-computed all-pairs distance matrix.
        cluster: List of node indices visit.
        time_limit: Maximum time.
        seed: Random seed.
        n_ants: Number of ants (defaults to len(cluster) per paper).
        alpha: Pheromone importance (Paper default: 1.0).
        beta: Heuristic importance (Paper default: 3.0).
        rho: Evaporation rate (Standard ACO: 0.1).
        q: Pheromone deposit amount (Standard ACO: 100.0).

    Note:
        The paper Sultana & Akhand (2017) specifies alpha=1 and beta=3 but is
        vague on evaporation and deposit scale. We adopt rho=0.1 and Q=100
        as robust defaults for the pheromone update Delta_tau = Q / L, where
        L is the tour distance.

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

    n_nodes = len(cluster)
    if n_ants is None:
        n_ants = n_nodes  # Paper: "number of ants in ACO was equal to the number of cities"

    # Methodological Note on Pheromone Initialization:
    # Standard OR practice often initializes tau_0 = 1 / (N * L_nn), where L_nn is
    # the cost of a nearest-neighbor tour. We use a uniform initialization of 0.1
    # here as a robust baseline for simplicity and comparability.
    pheromones = np.ones((n_nodes, n_nodes)) * 0.1
    # Heuristic: 1/distance. Small epsilon to avoid div by zero.
    heuristics = 1.0 / (distance_matrix[cluster][:, cluster] + 1e-10)

    best_tour = list(cluster)
    if best_tour[-1] != 0:
        best_tour.append(0)
    best_dist = _calculate_tour_distance(best_tour, distance_matrix)

    while time.time() - start_time < time_limit:
        all_tours = []
        all_dists = []

        for _ in range(n_ants):
            if time.time() - start_time >= time_limit:
                break

            curr_idx = random.randint(0, n_nodes - 1)
            tour_indices = [curr_idx]
            unvisited = set(range(n_nodes))
            unvisited.remove(curr_idx)

            while unvisited:
                unvisited_arr = np.array(list(unvisited))

                # Vectorized probability calculation
                # (pheromones ** alpha) * (heuristics ** beta)
                p_values = (pheromones[curr_idx, unvisited_arr] ** alpha) * (
                    heuristics[curr_idx, unvisited_arr] ** beta
                )

                total = np.sum(p_values)
                if total == 0:
                    next_idx = random.choice(list(unvisited))
                else:
                    probs = p_values / total
                    next_idx = np.random.choice(unvisited_arr, p=probs)

                tour_indices.append(next_idx)
                unvisited.remove(next_idx)
                curr_idx = next_idx

            tour = [cluster[i] for i in tour_indices]
            # Ensure it ends at depot if it starts at depot, or just make it circular
            if 0 in tour:
                z_idx = tour.index(0)
                tour = tour[z_idx:] + tour[:z_idx]
            tour.append(tour[0])

            d = _calculate_tour_distance(tour, distance_matrix)
            all_tours.append(tour_indices)
            all_dists.append(d)

            if d < best_dist:
                best_dist = d
                best_tour = list(tour)

        pheromones *= 1.0 - rho
        for tour_indices, d in zip(all_tours, all_dists, strict=False):
            deposit = q / (d + 1e-10)
            for i in range(len(tour_indices)):
                n1, n2 = tour_indices[i], tour_indices[(i + 1) % len(tour_indices)]
                pheromones[n1, n2] += deposit
                pheromones[n2, n1] += deposit

    return best_tour

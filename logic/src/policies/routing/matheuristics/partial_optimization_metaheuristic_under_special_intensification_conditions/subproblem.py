"""Sub-problem optimisers of POPMUSIC (fast-TSP, HGS, ALNS).

Attributes:
    _optimize_subproblem: Re-optimise a group of neighbouring routes with the configured solver.

Example:
    >>> new_routes = _optimize_subproblem(routes, dist, wastes, capacity, R, C, cfg)
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.configs.policies.routing.meta_heuristics.hgs import HGSConfig
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.alns import ALNSSolver
from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.hgs import HGSSolver
from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.split import LinearSplit
from logic.src.policies.routing.other_algorithms.travelling_salesman_problem.tsp import find_route
from logic.src.utils.routing.tours import get_route_cost

from .state import POPMUSICState


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
                profit_aware_operators=profit_aware_operators,
                seed=seed,
            )
    else:
        params = ALNSConfig(max_iterations=1000, time_limit=time_limit)

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

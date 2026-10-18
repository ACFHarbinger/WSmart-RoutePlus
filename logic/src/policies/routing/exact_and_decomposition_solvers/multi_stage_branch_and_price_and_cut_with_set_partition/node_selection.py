"""Phase 1 of MS-BPC-SP: knapsack selection of the bins the exact phase routes.

Attributes:
    _select_nodes_knapsack: Pick the mandatory bins plus the best-value optional bins within the fleet capacity.

Example:
    >>> selected = _select_nodes_knapsack(dist, wastes, capacity, R, C, mandatory, n_vehicles)
"""

import logging
from typing import Any, Dict, Optional, Set

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from logic.src.policies.helpers.solvers_and_matheuristics.branching.pruning import (
    reset_master_constraints as _reset_master_constraints,  # noqa: F401  (re-exported for tests)
)
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.smoothing import (
    detect_cycles as _detect_cycles,  # noqa: F401  (re-exported for tests)
)
from logic.src.policies.helpers.solvers_and_matheuristics.search.column_generation import (
    column_generation_loop as _column_generation_loop,  # noqa: F401  (re-exported for P-33 tests)
)

logger = logging.getLogger(__name__)


def _select_nodes_knapsack(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory: Set[int],
    n_nodes: int,
    vehicle_limit: Optional[int] = None,
    target_reduction: float = 0.60,  # aim to keep ~40% of optional nodes
    time_limit: float = 10.0,
    env: Optional[Any] = None,
) -> Set[int]:
    """Select profitable optional nodes via MIP knapsack before BPC routing.

    Solves a relaxed node-selection problem: ignore routing structure,
    approximate the routing cost of each node as its average distance to
    its nearest neighbors × C, then solve a knapsack to pick the subset
    of optional nodes where marginal revenue exceeds marginal routing cost.

    This reduces the effective instance size from n to |selected| before
    the RCSPP pricing subproblem is ever called.

    Args:
        dist_matrix: Full (n+1)×(n+1) distance matrix.
        wastes: Node waste amounts.
        capacity: Vehicle capacity.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        mandatory: Nodes that must be included regardless.
        n_nodes: Number of customer nodes.
        vehicle_limit: Optional vehicle limit.
        target_reduction: Target reduction in the number of nodes.
        time_limit: MIP time limit in seconds.
        env: Optional Gurobi environment.

    Returns:
        Set of selected node indices (mandatory ∪ profitable optional).
    """
    optional = [i for i in range(1, n_nodes + 1) if i not in mandatory]
    if not optional:
        return set(mandatory)

    # --- Routing cost estimate ---
    # Use Clarke-Wright savings-based approximation:
    # Cost of inserting node i into a route ≈ detour cost from its nearest
    # already-mandatory neighbor. This is tighter than average-distance.
    # For isolated optional nodes, use distance to depot × 2 as upper bound.
    depot = 0
    node_insertion_cost: Dict[int, float] = {}
    for i in optional:
        # Best insertion cost: min over all mandatory nodes j of
        #   dist(depot, i) + dist(i, j) - dist(depot, j)   (if inserting after depot)
        # Simplified: use half the round-trip cost to nearest mandatory node
        if mandatory:
            nearest_mand_dist = min(dist_matrix[i, j] for j in mandatory)
            insertion = C * (dist_matrix[depot, i] + nearest_mand_dist) / 2.0
        else:
            insertion = C * dist_matrix[depot, i]
        node_insertion_cost[i] = insertion

    # Net value: revenue minus realistic insertion cost
    node_net_value: Dict[int, float] = {}
    for i in optional:
        revenue = R * wastes.get(i, 0.0)
        node_net_value[i] = revenue - node_insertion_cost[i]

    # Pairwise synergy check: a node i might have individual insertion cost > revenue,
    # but when paired with neighbor j, the shared routing cost makes the pair profitable.
    synergistic_nodes: Set[int] = set()
    for i in optional:
        rev_i = R * wastes.get(i, 0.0)
        w_i = wastes.get(i, 0.0)
        for j in range(1, n_nodes + 1):
            if i == j:
                continue
            w_j = wastes.get(j, 0.0)
            if w_i + w_j > capacity:
                continue
            rev_pair = rev_i + (R * w_j if j in optional else 0.0)
            cost_pair = C * (dist_matrix[depot, i] + dist_matrix[i, j] + dist_matrix[j, depot])
            if rev_pair - cost_pair > 0:
                synergistic_nodes.add(i)
                break

    # Only consider nodes with positive net value or pairwise synergy
    candidates = [i for i in optional if node_net_value[i] > 0 or i in synergistic_nodes]
    if not candidates:
        return set(mandatory)

    # --- Capacity budget ---
    # Total load that can be served = vehicle_limit × capacity.
    # Subtract mandatory load to get budget for optional nodes.
    n_vehicles = (
        vehicle_limit
        if vehicle_limit is not None
        else max(2, int(np.ceil(sum(wastes.get(i, 0.0) for i in range(1, n_nodes + 1)) / capacity)))
    )
    total_capacity = n_vehicles * capacity
    mandatory_load = sum(wastes.get(i, 0.0) for i in mandatory)
    optional_budget = max(0.0, total_capacity - mandatory_load)

    # --- Hard size limit ---
    # Target at most (1 - target_reduction) × n_nodes selected nodes total,
    # floored at 20 so small-to-moderate instances are not over-reduced.
    max_optional = max(20, int(n_nodes * (1 - target_reduction)) - len(mandatory))

    try:
        m = gp.Model("node_selection", env=env) if env else gp.Model("node_selection")
        m.Params.OutputFlag = 0
        m.Params.TimeLimit = time_limit
        m.Params.MIPGap = 0.01

        y = m.addVars(candidates, vtype=GRB.BINARY, name="y")

        m.setObjective(
            gp.quicksum(node_net_value[i] * y[i] for i in candidates),
            GRB.MAXIMIZE,
        )

        # Hard capacity constraint
        m.addConstr(gp.quicksum(wastes.get(i, 0.0) * y[i] for i in candidates) <= optional_budget, name="capacity")

        # Hard size constraint — this is what was missing before
        m.addConstr(gp.quicksum(y[i] for i in candidates) <= max_optional, name="max_nodes")

        m.optimize()

        selected = set(mandatory)
        if m.SolCount > 0:
            for i in candidates:
                if y[i].X > 0.5:
                    selected.add(i)
        else:
            # Fallback: take top-k by net value
            top_k = sorted(candidates, key=lambda i: node_net_value[i], reverse=True)[:max_optional]
            selected = set(mandatory) | set(top_k)

        logger.info(
            f"[Node Selection] Knapsack selected {len(selected)} nodes "
            f"({len(selected) - len(mandatory)} optional, max_optional={max_optional}) "
            f"from {n_nodes} total. Reduced by {n_nodes - len(selected)} nodes."
        )
        return selected

    except Exception as e:
        logger.warning(f"[Node Selection] Knapsack failed ({e}), using mandatory + top-k.")
        top_k = sorted(candidates, key=lambda i: node_net_value[i], reverse=True)[:max_optional]
        return set(mandatory) | set(top_k)

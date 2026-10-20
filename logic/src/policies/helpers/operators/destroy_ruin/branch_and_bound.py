"""
Branch-and-Bound Removal (Destroy) Operator Module.

This module implements a destroy operator that selects which customer visits to
remove from the current plan using a **Branch-and-Bound** guided search
enhanced with **Limited Discrepancy Search** (LDS).

Conceptual Foundation
---------------------
Shaw (1998) Section 2 frames the ALNS cycle as a two-phase search:

    1. **Removal** (Figure 1): Select a set *V_r* of visits to remove from the
       current plan *P*.
    2. **Re-insertion** (Figure 2): Use a B&B/LDS search to find the best
       way to restore those visits.

This module implements a principled destroy companion to the
``branch_bound.py`` repair operator.  Rather than removing visits at random or
solely by worst-case detour cost, the operator scores each candidate node by
its **worst-recovery cost** — i.e., how expensive it would be to re-insert
that node into a plan from which it has just been removed.  Nodes with the
highest recovery cost are the most disruptive to remove, and are therefore
preferred (they create the largest perturbation in the plan's cost landscape).

The removal set is built iteratively using an LDS-bounded tree search that
controls how many "sub-optimal" choices (i.e., not the single most-disruptive
node) are allowed per selection step.  A discrepancy budget ``max_discrepancy``
caps the branching width; at ``max_discrepancy = 0`` the operator degenerates
to a deterministic greedy worst-recovery removal.

Algorithm
---------
    Step 1 — Score all routed nodes by their worst-recovery cost:

        recovery_cost(v) = min over all feasible positions (i, j) of
                           dist[prev(v), v] + dist[v, next(v)] - dist[prev(v), next(v)]

        This is the cheapest cost delta needed to re-insert *v* back into
        a plan that already has the other selected nodes removed.  A *high*
        recovery cost means *v* is expensive to put back = most disruptive to
        remove.

    Step 2 — Sort candidates by recovery cost descending (worst first).

    Step 3 — LDS branching: the first candidate costs 0 discrepancy; each
        subsequent candidate costs 1 discrepancy unit.  For each LDS branch,
        tentatively remove the chosen node and recurse to select the next.

    Step 4 — Pruning: if the current partial removal set's estimated cost
        savings (sum of detour costs of removed nodes) already cannot improve
        on the best removal set found so far, prune the branch.

Two public function pairs are provided:

    * ``bb_removal``         — standard cost-minimising variant (CVRP).
    * ``bb_profit_removal``  — profit-maximising VRPP variant that biases
                              selection toward low-profit nodes (located in
                              :mod:`.branch_and_bound_profit`).

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.destroy.branch_bound import (
    ...     bb_removal, bb_profit_removal
    ... )
    >>> routes, removed = bb_removal(routes, n_remove=5, dist_matrix=d, wastes=w, capacity=Q)
    >>> routes, removed = bb_profit_removal(routes, n_remove=5, dist_matrix=d,
    ...                                     wastes=w, capacity=Q, R=1.0, C=1.0)
"""

from random import Random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .branch_and_bound_profit import (
    _marginal_detour as _marginal_detour,
)

__all__ = ["bb_removal"]

# ---------------------------------------------------------------------------
# Internal helpers: per-node cost scores
# ---------------------------------------------------------------------------


def _recovery_insertion_cost(
    node: int,
    routes: List[List[int]],
    loads: List[float],
    wastes: Dict[int, float],
    dist_matrix: np.ndarray,
    capacity: float,
) -> float:
    """Cheapest re-insertion cost for *node* into the current partial plan.

    Used to score how disruptive removing *node* would be.  A high recovery
    cost (expensive to re-insert) means the node is difficult to recover →
    removal is highly disruptive.

    Args:
        node: The customer node to evaluate.
        routes: Partial plan (without *node*, which is assumed already absent).
        loads: Per-route accumulated load (without *node*'s contribution).
        wastes: Node demand lookup.
        dist_matrix: Distance matrix.
        capacity: Maximum vehicle load.

    Returns:
        Cheapest feasible insertion cost delta, or +inf if no feasible route
        exists (node would need its own route — maximally disruptive).
    """
    node_waste = wastes.get(node, 0.0)
    best_cost = float("inf")

    for r_idx, route in enumerate(routes):
        if loads[r_idx] + node_waste > capacity:
            continue
        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0
            delta = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
            if delta < best_cost:
                best_cost = delta

    # New route is always feasible (fallback): treat as round-trip
    new_rt_cost = dist_matrix[0, node] + dist_matrix[node, 0]
    return min(best_cost, new_rt_cost)


# ---------------------------------------------------------------------------
# Internal LDS engine — cost-minimising removal
# ---------------------------------------------------------------------------


def _lds_remove(  # noqa: C901
    routes: List[List[int]],
    loads: List[float],
    node_positions: Dict[int, Tuple[int, int]],  # node -> (r_idx, pos)
    candidates: List[Tuple[float, int]],  # sorted [(score_desc, node), ...]
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    discrep: int,
    current_removal: List[int],
    current_savings: float,
    best_state: List[Optional[Tuple[float, List[int]]]],
) -> None:
    """Recursive LDS tree search to select the best removal set.

    Selects *n_remove* nodes from *candidates* to remove, maximising cumulative
    detour savings subject to the LDS discrepancy budget.

    Args:
        routes: Current (mutable) plan state.
        loads: Per-route loads (mutable, parallel to *routes*).
        node_positions: Mapping from node to ``(route_idx, position)``in the
            *current* (post-removal) plan.  Updated as nodes are removed.
        candidates: Sorted list of ``(score_desc, node)`` — nodes sorted by
            descending disruption score (recovery cost or detour savings).
        n_remove: Number of nodes still to select.
        dist_matrix: Distance matrix.
        wastes: Node demand lookup.
        capacity: Vehicle capacity.
        discrep: Remaining LDS discrepancy budget.
        current_removal: Nodes selected so far in this branch.
        current_savings: Cumulative detour savings of *current_removal*.
        best_state: Single-element list; ``best_state[0]`` is either ``None``
            or ``(best_savings, best_removal_list)``.
    """
    if n_remove == 0:
        # Leaf: record if this removal set gives more savings
        current_best = best_state[0]
        if current_best is None or current_savings > current_best[0]:
            best_state[0] = (current_savings, list(current_removal))
        return

    if not candidates:
        # Ran out of candidates; record partial removal as-is
        current_best = best_state[0]
        if current_best is None or current_savings > current_best[0]:
            best_state[0] = (current_savings, list(current_removal))
        return

    # Pruning upper bound: current savings + sum of the top-n remaining scores
    remaining_scores = [sc for sc, _ in candidates[:n_remove]]
    ub = current_savings + sum(remaining_scores)
    current_best = best_state[0]
    if current_best is not None and ub <= current_best[0]:
        return  # This branch cannot improve the best known

    # LDS branching: try candidates in order, each beyond the first costs 1 discrepancy
    for i, (_score, node) in enumerate(candidates):
        remaining_discrep = discrep - i  # 0-th choice is free; i-th costs i units
        if remaining_discrep < 0:
            break

        # Look up current position of this node (may have shifted after removals)
        loc = node_positions.get(node)
        if loc is None:
            continue  # Already removed in a prior step

        r_idx, pos = loc

        # --- Tentatively remove the node ---
        node_waste = wastes.get(node, 0.0)
        detour = _marginal_detour(node, r_idx, pos, routes, dist_matrix)

        # Update route (in-place) and fix position map for nodes in this route
        route = routes[r_idx]
        route.pop(pos)
        loads[r_idx] -= node_waste

        # Update position map for nodes that shifted right after pos
        for shift_pos in range(pos, len(route)):
            shifted_node = route[shift_pos]
            node_positions[shifted_node] = (r_idx, shift_pos)
        del node_positions[node]

        # Build new candidate list (excluding just-removed node) with updated scores
        # Re-score remaining candidates against the modified route set
        new_candidates: List[Tuple[float, int]] = []
        for _, c_node in candidates:
            if c_node == node:
                continue
            c_loc = node_positions.get(c_node)
            if c_loc is None:
                continue
            c_r_idx, c_pos = c_loc
            # Recovery cost: how expensive to re-insert c_node given current partial plan
            c_score = _recovery_insertion_cost(c_node, routes, loads, wastes, dist_matrix, capacity)
            new_candidates.append((c_score, c_node))
        # Descending by recovery cost (most disruptive first)
        new_candidates.sort(key=lambda x: x[0], reverse=True)

        _lds_remove(
            routes,
            loads,
            node_positions,
            new_candidates,
            n_remove - 1,
            dist_matrix,
            wastes,
            capacity,
            remaining_discrep,
            current_removal + [node],
            current_savings + detour,
            best_state,
        )

        # --- Backtrack: restore node ---
        route.insert(pos, node)
        loads[r_idx] += node_waste
        # Restore node_positions for shifted nodes
        for shift_pos in range(pos + 1, len(route)):
            shifted_node = route[shift_pos]
            node_positions[shifted_node] = (r_idx, shift_pos)
        node_positions[node] = (r_idx, pos)


# ---------------------------------------------------------------------------
# Public API — cost-minimising removal (CVRP)
# ---------------------------------------------------------------------------


def bb_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    max_discrepancy: int = 1,
    rng: Optional[Random] = None,
    noise: float = 0.0,
) -> Tuple[List[List[int]], List[int]]:
    """Select nodes to remove using Branch-and-Bound with Limited Discrepancy Search.

    Implements a principled destroy operator that pairs with
    ``repair.branch_bound.bb_insertion`` (Shaw 1998, Section 2).

    The operator scores each routed node by its **worst-recovery cost** —
    the cheapest cost delta needed to re-insert that node back into the plan
    (computed against the *current* partial plan as removals are applied).
    A node with a high recovery cost is maximally disruptive to remove.

    An LDS tree search enumerates removal sets of size *n_remove*, branching
    over the sorted candidate list.  The first candidate (highest recovery
    cost) is the heuristic choice (0 discrepancy); each alternative costs 1
    discrepancy unit.  The search terminates when the budget is exhausted or
    all removal sets of the required size have been enumerated.

    The removal set with the highest cumulative detour savings (i.e., the most
    disruptive removal) is chosen among all explored branches.

    Args:
        routes: Current plan (list of customer sequences, depot implicit).
        n_remove: Number of nodes to remove.
        dist_matrix: Square distance matrix of shape ``(N+1, N+1)`` where
            index 0 is the depot.
        wastes: Mapping from node index to demand (waste volume).
        capacity: Maximum vehicle load.
        max_discrepancy: LDS discrepancy budget *d*.  At *d = 0* only the
            greedy-best removal sequence is explored (deterministic).  At
            *d = 1* one alternative per step is tried.  Recommended: 0–2.
        rng: Optional random number generator.  If provided, a small random
            perturbation (scaled by *noise*) is added to candidate scores to
            break deterministic ties and introduce stochastic diversity.
        noise: Magnitude of Gaussian noise added to each recovery-cost score
            when *rng* is provided.  Set to 0 for deterministic mode.

    Returns:
        Tuple[List[List[int]], List[int]]: A tuple of:
            - The partial routes with the selected nodes removed.
            - The list of removed node IDs.

    Raises:
        ValueError: If *n_remove* is negative or *max_discrepancy* is negative.
    """
    if n_remove < 0:
        raise ValueError(f"n_remove must be non-negative, got {n_remove}")
    if max_discrepancy < 0:
        raise ValueError(f"max_discrepancy must be non-negative, got {max_discrepancy}")

    if n_remove == 0:
        return routes, []

    # Flatten node positions
    node_positions: Dict[int, Tuple[int, int]] = {}
    all_nodes: List[int] = []
    for r_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            node_positions[node] = (r_idx, pos)
            all_nodes.append(node)

    if not all_nodes:
        return routes, []

    n_remove = min(n_remove, len(all_nodes))

    # Build initial loads
    loads: List[float] = [sum(wastes.get(n, 0.0) for n in r) for r in routes]

    # Score every node by its recovery cost (cheapest re-insertion delta)
    # Copy routes/loads for the temporary removal simulation
    routes_sim = [list(r) for r in routes]
    loads_sim = list(loads)
    np_sim = dict(node_positions)

    initial_scores: List[Tuple[float, int]] = []
    for node in all_nodes:
        score = _recovery_insertion_cost(node, routes_sim, loads_sim, wastes, dist_matrix, capacity)
        if rng is not None and noise > 0.0:
            score = max(0.0, score + rng.gauss(0.0, noise))
        initial_scores.append((score, node))

    # Sort descending: highest recovery cost first (most disruptive)
    initial_scores.sort(key=lambda x: x[0], reverse=True)

    best_state: List[Optional[Tuple[float, List[int]]]] = [None]

    _lds_remove(
        routes_sim,
        loads_sim,
        np_sim,
        initial_scores,
        n_remove,
        dist_matrix,
        wastes,
        capacity,
        max_discrepancy,
        [],
        0.0,
        best_state,
    )

    if best_state[0] is None:
        # Fallback: greedy removal of highest-score nodes
        removed_ids = [node for _, node in initial_scores[:n_remove]]
    else:
        _, removed_ids = best_state[0]

    # Apply removal to the original routes
    removed_set: Set[int] = set(removed_ids)
    final_removed: List[int] = []
    modified_routes: List[List[int]] = []

    for route in routes:
        new_route: List[int] = []
        for node in route:
            if node in removed_set:
                final_removed.append(node)
            else:
                new_route.append(node)
        if new_route:
            modified_routes.append(new_route)

    return modified_routes, final_removed

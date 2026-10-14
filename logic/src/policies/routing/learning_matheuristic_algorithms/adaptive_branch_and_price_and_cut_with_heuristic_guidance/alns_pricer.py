"""
ALNS Metaheuristic Pricing.

Attributes:
    ALNSMultiPeriodPricer: A class that implements the ALNS heuristic pricing for the Benders decomposition.

Example:
    >>> from alns_pricer import ALNSMultiPeriodPricer
    >>> pricer = ALNSMultiPeriodPricer()
    >>> pricer.solve()
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics import RCSPPSolver, Route


class ALNSMultiPeriodPricer:
    """
    ALNS-based column generator that navigates the reduced-cost landscape
    defined by scenario-augmented prizes from ``ScenarioPrizeEngine``.

    Role in the Pipeline
    --------------------
    Acts as the primary heuristic pricing subproblem solver in the BPC
    column-generation loop.  For each call to ``solve()``:

    1. Applies iterative destroy-and-repair cycles to a set of seed routes
       using scenario-aware operators.
    2. Collects routes with positive reduced cost (VRPP maximisation: rc > tol).
    3. Falls back to the exact ESPPRC solver (``RCSPPSolver``) when no
       improving column is found, guaranteeing column-generation optimality.

    Operators
    ---------
    ``scenario_overflow_removal`` (destroy):
        Removes the ``num_remove`` nodes with the lowest urgency score
        π_i^{scenario} − π_i^{dual}.  Low-urgency nodes contribute little
        to the multi-period objective and are strong candidates for ejection.
        Depot (node 0) is always protected.

    ``scenario_aware_insertion`` (repair):
        Inserts unrouted nodes greedily by maximising a combined spatial–prize
        score:  prize(u) − insertion_cost(u, position) at each candidate
        insertion position.  Only profitable insertions (score > 0) are
        accepted, preserving the reduced-cost improvement invariant.

    Reduced Cost Convention
    -----------------------
    This codebase uses the VRPP *maximisation* convention throughout:
    a column with **positive** reduced cost rc > tol is an improving column.
    This is the sign-flip of the standard LP minimisation convention
    (rc < −tol).

    Attributes:
    ----------
    exact_pricer : RCSPPSolver
        Exact ESPPRC solver used as fallback when ALNS fails to find
        improving columns.
    rng : np.random.Generator
        Seeded random number generator for reproducible ALNS behaviour.
    """

    def __init__(self, exact_pricer: RCSPPSolver, rng_seed: int = 42):
        """
        Args:
            exact_pricer: Exact ``RCSPPSolver`` instance used when ALNS cannot
                produce a route with reduced cost > ``rc_tolerance``.
            rng_seed: Seed for the NumPy random number generator.  Fix to
                a constant for reproducible experiments; vary across restarts
                for diversification.
        """
        self.exact_pricer = exact_pricer
        self.rng = np.random.default_rng(rng_seed)

    def scenario_overflow_removal(
        self,
        route_nodes: List[int],
        scenario_prizes: Dict[int, float],
        dual_values: Dict[int, float],
        num_remove: int,
    ) -> List[int]:
        """
        Destroy operator: remove the ``num_remove`` lowest-urgency nodes from
        the route.

        Urgency score for node i: π_i^{scenario} − π_i^{dual}.

        Nodes with low urgency contribute little net benefit to the current
        pricing objective and are the first candidates for ejection.  This
        targets bins that are either near-empty (low scenario prize) or whose
        dual value has already absorbed most of their contribution.

        The depot (node 0) is always assigned score +∞ and is never removed.

        Args:
            route_nodes: Current route as an ordered node list (may include
                depot at position 0 and/or as the last node).
            scenario_prizes: Node prizes {node_id: π_i^{scenario}} from
                ``ScenarioPrizeEngine`` for the current day and scenario.
            dual_values: RMP dual variables {node_id: π_i^{dual}} for the
                current covering constraints.
            num_remove: Number of non-depot nodes to remove.

        Returns:
            Pruned route node list with the ``num_remove`` lowest-urgency
            non-depot nodes removed.
        """
        if not route_nodes:
            return []

        scores = []
        for i in route_nodes:
            if i == 0:
                scores.append(float("inf"))  # Protect depot
            else:
                scores.append(scenario_prizes.get(i, 0.0) - dual_values.get(i, 0.0))

        sorted_indices = np.argsort(scores)

        to_remove = set()
        count = 0
        for idx in sorted_indices:
            if route_nodes[idx] != 0 and count < num_remove:
                to_remove.add(route_nodes[idx])
                count += 1

        return [n for n in route_nodes if n not in to_remove]

    def scenario_aware_insertion(
        self,
        route_nodes: List[int],
        unrouted: List[int],
        scenario_prizes: Dict[int, float],
        dual_values: Dict[int, float],
        dist_matrix: np.ndarray,
        num_insert: int,
    ) -> List[int]:
        """
        Repair operator: greedily insert up to ``num_insert`` unrouted nodes
        into the route by maximising the spatial–prize insertion score.

        For each candidate node u and each feasible insertion position (after
        route_nodes[i]):

            cost(u, i)  = dist[route[i], u] + dist[u, route[i+1]] − dist[route[i], route[i+1]]
            score(u, i) = (π_u^{scenario} − π_u^{dual}) − cost(u, i)

        The (u, i) pair with the highest positive score is inserted first.
        Insertion stops when no profitable candidate remains or ``num_insert``
        nodes have been placed.

        Args:
            route_nodes: Current (possibly partial) route after the destroy step.
            unrouted: List of non-depot nodes not currently in ``route_nodes``.
                Modified in-place: successfully inserted nodes are removed.
            scenario_prizes: Node prizes {node_id: π_i^{scenario}}.
            dual_values: RMP dual variables {node_id: π_i^{dual}}.
            dist_matrix: Square distance / cost matrix.
            num_insert: Maximum number of nodes to insert in this repair step.

        Returns:
            Updated route node list with inserted nodes in their optimal
            spatial positions.
        """
        current_route = list(route_nodes)

        for _ in range(num_insert):
            if not unrouted:
                break

            best_node = -1
            best_pos = -1
            best_score = -float("inf")

            for u in unrouted:
                prize = scenario_prizes.get(u, 0.0) - dual_values.get(u, 0.0)

                for i in range(len(current_route) - 1):
                    cost = (
                        dist_matrix[current_route[i], u]
                        + dist_matrix[u, current_route[i + 1]]
                        - dist_matrix[current_route[i], current_route[i + 1]]
                    )
                    score = prize - cost
                    if score > best_score:
                        best_score = score
                        best_node = u
                        best_pos = i + 1

            if best_node != -1 and best_score > 0:
                current_route.insert(best_pos, best_node)
                unrouted.remove(best_node)
            else:
                break  # No profitable insertion remains

        return current_route

    def _calculate_reduced_cost(
        self,
        route_nodes: List[int],
        scenario_prizes: Dict[int, float],
        dual_values: Dict[int, float],
        dist_matrix: np.ndarray,
    ) -> float:
        """
        Evaluate the reduced cost of a route under the VRPP maximisation convention.

            rc_k = −dist(k) + Σ_{i ∈ k, i≠0} (π_i^{scenario} − π_i^{dual})

        A route with rc_k > 0 improves the current master LP objective and is
        a valid new column.

        Args:
            route_nodes: Ordered node list including depot (0) at both ends.
            scenario_prizes: Node prizes {node_id: π_i^{scenario}}.
            dual_values: RMP dual variables {node_id: π_i^{dual}}.
            dist_matrix: Square distance matrix.

        Returns:
            Reduced cost rc_k (VRPP maximisation: positive = improving).
        """
        dist_cost = 0.0
        prize_sum = 0.0

        for i in range(len(route_nodes) - 1):
            dist_cost += dist_matrix[route_nodes[i], route_nodes[i + 1]]

        for n in route_nodes:
            if n != 0:
                prize_sum += scenario_prizes.get(n, 0.0) - dual_values.get(n, 0.0)

        return prize_sum - dist_cost

    def solve(
        self,
        dual_values: Dict[int, float],
        scenario_prizes: Dict[int, float],
        dist_matrix: np.ndarray,
        initial_routes: List[List[int]],
        all_nodes: List[int],
        max_routes: int = 5,
        rc_tolerance: float = 1e-4,
        alns_iterations: int = 50,
        **kwargs: Any,
    ) -> Tuple[List[Route], bool]:
        """
        Run ALNS pricing and return improving columns, falling back to exact
        ESPPRC if none are found.

        For each route in ``initial_routes``, the ALNS loop applies
        ``alns_iterations`` rounds of (destroy → repair), collecting routes
        with rc > ``rc_tolerance`` into a candidate pool.  The top
        ``max_routes`` columns by reduced cost are returned.

        If no improving columns are produced across all seeds, the exact
        ESPPRC solver (``self.exact_pricer``) is invoked as a fallback to
        guarantee completeness of the pricing step.

        Args:
            dual_values: RMP dual variables {node_id: π_i^{dual}} for the
                current covering constraints.
            scenario_prizes: Node prizes {node_id: π_i^{scenario}} from
                ``ScenarioPrizeEngine``.
            dist_matrix: Square distance matrix over all nodes (0-indexed,
                depot at 0).
            initial_routes: Seed routes for the ALNS loop.  Typically drawn
                from the current RMP column pool or a construction heuristic.
            all_nodes: Complete list of non-depot customer nodes.
            max_routes: Maximum number of improving columns to return per
                pricing call.  Capped at this value to bound the number of
                master LP pivots per CG iteration.
            rc_tolerance: Minimum reduced cost threshold (VRPP maximisation:
                rc > tol is improving).  Corresponds to −1e-4 in the standard
                LP minimisation convention.
            alns_iterations: Number of destroy-and-repair cycles per seed
                route.
            kwargs: Forwarded verbatim to ``self.exact_pricer.solve()`` when
                the fallback is triggered.

        Returns:
            routes : List[Route]
                Improving columns sorted descending by reduced cost.  If ALNS
                succeeds, these are ALNS-generated routes; if the fallback is
                used, these are exact ESPPRC routes.
            used_exact : bool
                ``True`` if the exact ESPPRC fallback was invoked,
                ``False`` if ALNS produced at least one improving column.
        """
        best_routes: List[Route] = []

        for init_route in initial_routes:
            current_route = list(init_route)

            for _ in range(alns_iterations):
                unrouted = [n for n in all_nodes if n not in current_route and n != 0]

                # Destroy step
                num_remove = max(1, len(current_route) // 4)
                current_route = self.scenario_overflow_removal(current_route, scenario_prizes, dual_values, num_remove)

                # Repair step
                current_route = self.scenario_aware_insertion(
                    current_route,
                    unrouted,
                    scenario_prizes,
                    dual_values,
                    dist_matrix,
                    num_remove,
                )

                rc = self._calculate_reduced_cost(current_route, scenario_prizes, dual_values, dist_matrix)

                if rc > rc_tolerance:
                    r = Route(
                        nodes=[n for n in current_route if n != 0],
                        cost=0.0,
                        revenue=0.0,
                        load=0.0,
                        node_coverage=set(current_route),
                    )
                    r.reduced_cost = rc
                    best_routes.append(r)

        best_routes = sorted(
            best_routes,
            key=lambda x: x.reduced_cost if x.reduced_cost else 0.0,
            reverse=True,
        )[:max_routes]

        if best_routes:
            return best_routes, False

        # Fallback: exact ESPPRC guarantees an optimal pricing solution
        return self.exact_pricer.solve(dual_values=dual_values, max_routes=max_routes, **kwargs), True

"""bp routes methods relocated from bp."""

from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

from logic.src.policies.helpers.solvers_and_matheuristics import (
    AnyBranchingConstraint,
    RCSPPSolver,
    Route,
    VRPPMasterProblem,
)

from .pricing_subproblem import PricingSubproblem


class BranchAndPriceRoutesMixin:
    """Relocated bp routes methods."""

    def _call_pricing(
        self,
        pricing: Union[PricingSubproblem, RCSPPSolver],
        dual_values: Dict[Union[int, frozenset[int], str, Tuple[int, int]], float],
        constraints: Optional[List[AnyBranchingConstraint]],
    ) -> List[Tuple[List[int], float]]:
        """Dispatch a pricing call to the correct solver with the right kwargs.

        RCSPPSolver uses ``branching_constraints``; PricingSubproblem uses
        ``active_constraints``.

        Args:
            pricing: The pricing solver instance.
            dual_values: Map from constraints to their dual values.
            constraints: Branching constraints to enforce.

        Returns:
            List of generated routes as (path, reduced_cost) pairs.
        """
        from logic.src.policies.helpers.solvers_and_matheuristics import RCSPPSolver

        if isinstance(pricing, RCSPPSolver):
            routes = pricing.solve(
                dual_values=cast(Any, dual_values),
                max_routes=self.params.max_routes_per_iteration,
                branching_constraints=constraints,
            )
            # RCSPPSolver now returns Route objects; convert to (path, reduced_cost) for BP backward compatibility
            return [(r.nodes, r.reduced_cost if r.reduced_cost is not None else 0.0) for r in routes]
        else:
            return pricing.solve(
                dual_values=cast(Any, dual_values),
                max_routes=self.params.max_routes_per_iteration,
                active_constraints=constraints,
            )

    def _generate_initial_routes(self, pricing: Any) -> List[Route]:
        """Generate high-quality initial feasible columns for the master problem.

        Uses a Greedy Nearest Neighbor heuristic to cover all mandatory nodes
        with feasible routes, obeying capacity constraints. This provides a
        stronger initial LP bound and more stable early dual variables than
        simple artificial out-and-back routes.

        Args:
            pricing: The pricing subproblem (used to compute cost/revenue).

        Returns:
            List of feasible Route objects covering all mandatory nodes.
        """
        routes: List[Route] = []
        uncovered = sorted(list(self.mandatory_nodes))

        while uncovered:
            current_route: List[int] = []
            current_load = 0.0
            current_node = self.depot

            while True:
                # Find nearest uncovered mandatory node that fits.
                best_next = None
                best_dist = float("inf")

                for candidate in uncovered:
                    demand = self.wastes.get(candidate, 0.0)
                    if current_load + demand <= self.capacity:
                        dist = self.cost_matrix[current_node, candidate]
                        if dist < best_dist:
                            best_dist = dist
                            best_next = candidate

                if best_next is None:
                    break  # Capacity reached or no closer nodes.

                current_route.append(best_next)
                current_load += self.wastes.get(best_next, 0.0)
                uncovered.remove(best_next)
                current_node = best_next

            # Finalize route return to depot.
            if current_route:
                cost, revenue, load, coverage = pricing.compute_route_details(current_route)
                routes.append(
                    Route(
                        nodes=current_route,
                        cost=cost,
                        revenue=revenue,
                        load=load,
                        node_coverage=coverage,
                    )
                )

        return routes

    def _add_routes_to_master(
        self,
        master: VRPPMasterProblem,
        pricing: Any,
        new_routes: List[Tuple[List[int], float]],
    ) -> None:
        """Add a batch of new route candidates to the master problem.

        Args:
            master: The Restricted Master Problem instance.
            pricing: The pricing solver used to compute route metadata.
            new_routes: List of (path, reduced_cost) pairs to add.
        """
        for route_nodes, _ in new_routes:
            cost, revenue, load, cov = pricing.compute_route_details(route_nodes)
            route = Route(nodes=route_nodes, cost=cost, revenue=revenue, load=load, node_coverage=cov)
            master.add_route(route)
            self.num_columns_generated += 1

    def _routes_to_tour(self, routes: List[Route]) -> List[int]:
        """Convert selected routes to a flat tour, deduplicating customer visits.

        Because the master problem uses Set Covering (≥ 1), the IP may select
        overlapping routes. Each customer node is included exactly once in the
        output; subsequent routes that would revisit a node simply skip it.

        Args:
            routes: List of Route objects selected in the integer solution.

        Returns:
            Flat tour: [depot, n1, n2, ..., depot, n3, ..., depot].
        """
        tour: List[int] = [self.depot]
        visited_nodes: Set[int] = {self.depot}

        for route in routes:
            unique_nodes = [n for n in route.nodes if n not in visited_nodes]
            if not unique_nodes:
                continue
            tour.extend(unique_nodes)
            tour.append(self.depot)
            visited_nodes.update(unique_nodes)

        return tour

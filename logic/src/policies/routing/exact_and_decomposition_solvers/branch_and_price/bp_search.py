"""bp search methods relocated from bp."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from logic.src.configs.policies.routing.exact_and_decomposition_solvers.bp import BPConfig
from logic.src.policies.helpers.solvers_and_matheuristics import (
    AnyBranchingConstraint,
    BranchAndBoundTree,
    BranchNode,
    RCSPPSolver,
    Route,
    VRPPMasterProblem,
)

from .pricing_subproblem import PricingSubproblem


class BranchAndPriceSearchMixin:
    """Relocated bp search methods."""

    def _bind_bp_master(self, master: VRPPMasterProblem) -> None:
        """Point master cut/column notes at this solve. No-op when untraced."""
        trace = getattr(self, "trace", None)
        if trace is None:
            return
        from logic.src.policies.helpers.solvers_and_matheuristics.exact_trace import attach_master

        stage = getattr(self, "_trace_stage", None)
        attach_master(master, trace, stage)
        parent = getattr(self, "_trace_node", None) or stage
        master._exec_trace_parent = parent
        master._exec_trace_node = parent

    def _note_bp_pricing(self, n_added: int) -> None:
        """Record one ordinary pricing step. No-op when untraced."""
        trace = getattr(self, "trace", None)
        if trace is None:
            return
        from logic.src.policies.helpers.solvers_and_matheuristics.exact_trace import emit_pricing_on

        parent = getattr(self, "_trace_node", None) or getattr(self, "_trace_stage", None)
        emit_pricing_on(trace, parent, "ordinary", 2, n_added, node_id=parent)

    # Declared by BranchAndPriceSolver.__init__; annotation only.
    params: BPConfig
    n_nodes: int
    mandatory_nodes: Set[int]
    num_iterations: int
    num_columns_generated: int
    lp_bound: float
    ip_solution: float
    proven_optimal: bool
    tree: Optional[BranchAndBoundTree]

    if TYPE_CHECKING:

        def _make_master(self) -> VRPPMasterProblem: ...

        def _make_pricing(self) -> Tuple[PricingSubproblem, Optional[RCSPPSolver]]: ...

        def _generate_initial_routes(self, pricing: Any) -> List[Route]: ...

        def _is_integer_solution(self, route_values: Dict[int, float], tol: float = 1e-5) -> bool: ...

        def _routes_to_tour(self, routes: List[Route]) -> List[int]: ...

        def _get_statistics(self) -> Dict[str, Any]: ...

        def _build_master_for_node(self, node: BranchNode, routes: List[Route]) -> VRPPMasterProblem: ...

        def _call_pricing(
            self,
            pricing: Any,
            dual_values: Dict[Any, float],
            constraints: Optional[List[AnyBranchingConstraint]],
        ) -> List[Tuple[List[int], float]]: ...

        def _add_routes_to_master(
            self,
            master: VRPPMasterProblem,
            pricing: Any,
            new_routes: List[Tuple[List[int], float]],
        ) -> None: ...

    def _solve_without_branching(self) -> Tuple[List[int], float, Dict[str, Any]]:
        """Solve with column generation followed by a direct MIP solve.

        Returns:
            A 3-tuple containing (tour, objective, statistics).
        """
        master = self._make_master()
        heuristic_pricing, exact_pricing = self._make_pricing()

        initial_routes = self._generate_initial_routes(heuristic_pricing)
        master.build_model(initial_routes)
        self._bind_bp_master(master)

        lp_obj, route_values = self._column_generation(master, (heuristic_pricing, exact_pricing))
        self.lp_bound = lp_obj

        if self._is_integer_solution(route_values):
            selected = [master.routes[i] for i, v in route_values.items() if v > 0.5]
            return self._routes_to_tour(selected), lp_obj, self._get_statistics()

        ip_obj, selected = master.solve_ip()
        self.ip_solution = ip_obj
        return self._routes_to_tour(selected), ip_obj, self._get_statistics()

    def _solve_with_branching(self) -> Tuple[List[int], Optional[float], Dict[str, Any]]:  # noqa: C901
        """Solve using the configured branching strategy in a B&B tree.

        Returns:
            A 3-tuple containing (tour, objective, statistics).
        """
        self.tree = BranchAndBoundTree(
            v_model=None,
            strategy=self.params.branching_strategy,
            search_strategy=self.params.tree_search_strategy,
        )

        if getattr(self, "trace", None) is not None:
            self._trace_node = self.trace.add(
                name="node",
                kind="stage",
                parent_id=getattr(self, "_trace_stage", None),
                arguments={"depth": 0, "node_id": 0},
            )
        root_lp, root_values, root_routes = self._solve_node(self.tree.root)
        self.tree.root.route_values = root_values
        self.tree.root.routes = root_routes
        self.tree.root.is_integer = self._is_integer_solution(root_values)
        self.lp_bound = self.tree.root.lp_bound  # type: ignore[assignment]

        if self.tree.root.is_integer:
            selected = [root_routes[i] for i, v in root_values.items() if v > 0.5]
            return self._routes_to_tour(selected), root_lp, self._get_statistics()

        self.tree.nodes_explored += 1

        # Remove root from open list — it has already been evaluated above.
        if self.tree.root in self.tree.open_nodes:
            self.tree.open_nodes.remove(self.tree.root)

        routes: List[Route] = root_routes
        while not self.tree.is_empty() and self.tree.nodes_explored < self.params.max_branch_nodes:
            node = self.tree.get_next_node()
            if node is None:
                break
            if getattr(self, "trace", None) is not None:
                self._trace_node = self.trace.add(
                    name="node",
                    kind="stage",
                    parent_id=getattr(self, "_trace_stage", None),
                    arguments={"depth": int(node.depth), "node_id": int(self.tree.nodes_explored)},
                )

            if (
                self.tree.best_integer_solution is not None
                and node.lp_bound is not None
                and node.lp_bound <= self.tree.best_integer_solution
            ):
                self.tree.nodes_pruned += 1
                continue

            lp_obj, route_values, routes = self._solve_node(node, parent_routes=routes)
            node.route_values = route_values
            node.routes = routes

            if lp_obj is None:
                node.is_infeasible = True
                self.tree.nodes_pruned += 1
                continue

            if self._is_integer_solution(route_values):
                node.is_integer = True
                node.ip_solution = lp_obj
                if self.tree.update_incumbent(node, lp_obj):
                    self.ip_solution = lp_obj
                    self.tree.prune_by_bound()
                    if getattr(self, "trace", None) is not None:
                        from logic.src.policies.helpers.solvers_and_matheuristics.exact_trace import emit_incumbent

                        selected_nodes = [
                            list(routes[i].nodes)
                            for i, value in route_values.items()
                            if value > 0.5 and i < len(routes)
                        ]
                        emit_incumbent(self.trace, getattr(self, "_trace_node", None), selected_nodes, lp_obj)
                continue

            result = self.tree.branch(node, routes, route_values, mandatory_nodes=self.mandatory_nodes)
            if result is None:
                logging.warning(
                    "Branching stalled (no fractional candidates found). Falling back to restricted "
                    "MIP solve. This is a Primal Heuristic; global optimality is no longer guaranteed."
                )
                self.proven_optimal = False
                master = self._build_master_for_node(node, routes)
                ip_obj, selected = master.solve_ip()
                if ip_obj is not None:
                    node.is_integer = True
                    node.ip_solution = ip_obj
                    node.routes = routes
                    node.route_values = {i: 1.0 if r in selected else 0.0 for i, r in enumerate(routes)}
                    if self.tree.update_incumbent(node, ip_obj):
                        self.ip_solution = ip_obj
                        self.tree.prune_by_bound()
            else:
                left_child, right_child = result
                if getattr(self, "trace", None) is not None:
                    from logic.src.policies.helpers.solvers_and_matheuristics.exact_trace import emit_branch

                    holder = type("_Holder", (), {})()
                    holder._exec_trace = self.trace
                    holder._exec_trace_node = getattr(self, "_trace_node", None)
                    holder._exec_trace_parent = getattr(self, "_trace_node", None)
                    emit_branch(holder, left_child, right_child)
                self.tree.add_node(left_child)
                self.tree.add_node(right_child)

        if self.tree.best_integer_node is not None:
            rv = self.tree.best_integer_node.route_values
            node_routes = self.tree.best_integer_node.routes
            if rv and node_routes:
                selected = [node_routes[i] for i, v in rv.items() if v > 0.5 and i < len(node_routes)]
                return (
                    self._routes_to_tour(selected),
                    self.tree.best_integer_solution,  # type: ignore[return-value]
                    self._get_statistics(),
                )

        root_master = self._build_master_for_node(self.tree.root, root_routes)
        ip_obj, selected = root_master.solve_ip()
        if ip_obj is not None:
            self.ip_solution = ip_obj
            return self._routes_to_tour(selected), ip_obj, self._get_statistics()

        return [0], 0.0, self._get_statistics()

    def _solve_node(
        self,
        node: BranchNode,
        parent_routes: Optional[List[Route]] = None,
    ) -> Tuple[Optional[float], Dict[int, float], List[Route]]:
        """Solve the LP relaxation at a single B&B node via column generation.

        Args:
            node: The Branch-and-Bound node to evaluate.
            parent_routes: Optional pool of routes from the parent node.

        Returns:
            A 3-tuple containing (lp_objective, route_values, routes).
        """
        master = self._make_master()
        self._bind_bp_master(master)
        constraints = node.get_all_constraints()
        heuristic_pricing, exact_pricing = self._make_pricing()

        # Use parent column pool (filtered to node constraints) when available.
        # Fall back to greedy initial routes only if no feasible parent columns exist.
        feasible = [r for r in parent_routes if node.is_route_feasible(r)] if parent_routes else []

        if not feasible:
            initial_routes = self._generate_initial_routes(heuristic_pricing)
            feasible = [r for r in initial_routes if node.is_route_feasible(r)]

        master.build_model(feasible)

        try:
            lp_obj, route_values, best_bound = self._column_generation_with_constraints(
                master, (heuristic_pricing, exact_pricing), node, constraints
            )
            node.lp_bound = min(lp_obj, best_bound)

            # Part 5 Fix 5: Rounded Capacity Cut (RCC) separation at root node
            # We only separate at depth 0 to avoid cut management complexity in the B&B tree.
            if node.depth == 0:
                logging.info("Starting Root RCC separation...")
                cuts_added = master.find_and_add_violated_rcc(route_values, master.routes)
                if cuts_added > 0:
                    logging.info(f"Added {cuts_added} capacity cuts at root. Re-solving...")
                    lp_obj, route_values, best_bound = self._column_generation_with_constraints(
                        master, (heuristic_pricing, exact_pricing), node, constraints
                    )
                    node.lp_bound = min(lp_obj, best_bound)
            if getattr(self, "trace", None) is not None and node.depth == 0:
                from logic.src.policies.helpers.solvers_and_matheuristics.exact_trace import emit_node_bound

                emit_node_bound(master, 0, float(lp_obj), None, root=True)
        except RuntimeError:
            return None, {}, []

        return lp_obj, route_values, master.routes

    def _column_generation(
        self,
        master: VRPPMasterProblem,
        pricing: Tuple[PricingSubproblem, Optional[RCSPPSolver]],
    ) -> Tuple[float, Dict[int, float]]:
        """Perform unconstrained column generation with heuristic fallback.

        Args:
            master: The Restricted Master Problem (RMP).
            pricing: Pair of (heuristic, exact) pricing solvers.

        Returns:
            A tuple containing (lp_objective, route_values).
        """
        heuristic_pricer, exact_pricer = pricing
        last_basis: Optional[Tuple[List[int], List[int]]] = None
        self._bind_bp_master(master)
        noted_root = False

        for i in range(self.params.max_iterations):
            self.num_iterations += 1

            if last_basis is not None:
                master.restore_basis(*last_basis)

            lp_obj, route_values = master.solve_lp_relaxation()
            last_basis = master.save_basis()
            if not noted_root:
                noted_root = True
                if getattr(self, "trace", None) is not None:
                    from logic.src.policies.helpers.solvers_and_matheuristics.exact_trace import emit_node_bound

                    emit_node_bound(master, 0, float(lp_obj), None, root=True)

            dual_values = master.get_reduced_cost_coefficients()

            # 1. Try heuristic pricer first (Two-Phase Pricing)
            new_routes = self._call_pricing(heuristic_pricer, dual_values, constraints=None)  # type: ignore[arg-type]
            self._note_bp_pricing(0 if not new_routes else len(new_routes))

            # 2. Fallback to exact pricer if heuristic fails
            if (
                not new_routes or max(rc for _, rc in new_routes) < self.params.optimality_gap
            ) and exact_pricer is not None:
                new_routes = self._call_pricing(exact_pricer, dual_values, constraints=None)  # type: ignore[arg-type]
            self._note_bp_pricing(0 if not new_routes else len(new_routes))

            if not new_routes:
                break

            max_rc = max(rc for _, rc in new_routes)
            if max_rc < self.params.optimality_gap:
                break

            # Estimate fleet limit for unconstrained instances to calculate Lagrangian bound
            limit = self.params.vehicle_limit if self.params.vehicle_limit is not None else self.n_nodes
            lagrangian_bound = lp_obj + max_rc * limit
            if abs(lp_obj - lagrangian_bound) < self.params.early_termination_gap:
                break

            self._add_routes_to_master(master, heuristic_pricer, new_routes)
            last_basis = None  # basis is stale after column additions; force fresh solve

            # Task 3: Column Pool Management
            if (i + 1) % self.params.cleanup_frequency == 0:
                master.purge_useless_columns(self.params.cleanup_threshold)

        if last_basis is not None:
            master.restore_basis(*last_basis)
        return master.solve_lp_relaxation()

    def _column_generation_with_constraints(
        self,
        master: VRPPMasterProblem,
        pricing: Tuple[PricingSubproblem, Optional[RCSPPSolver]],
        node: BranchNode,
        constraints: List[AnyBranchingConstraint],
    ) -> Tuple[float, Dict[int, float], float]:
        """Perform constrained column generation with heuristic fallback.

        Args:
            master: The Restricted Master Problem (RMP).
            pricing: Pair of (heuristic, exact) pricing solvers.
            node: The current Branch-and-Price search node.
            constraints: List of branching constraints to respect in pricing.

        Returns:
            A 3-tuple containing (lp_objective, route_values, best_lagrangian_bound).
        """
        heuristic_pricer, exact_pricer = pricing
        best_lagrangian = float("inf")
        last_basis: Optional[Tuple[List[int], List[int]]] = None
        self._bind_bp_master(master)

        for i in range(self.params.max_iterations):
            self.num_iterations += 1
            if last_basis is not None:
                master.restore_basis(*last_basis)

            lp_obj, route_values = master.solve_lp_relaxation()
            last_basis = master.save_basis()
            dual_values = master.get_reduced_cost_coefficients()

            # 1. Try heuristic pricer first (Two-Phase Pricing)
            new_routes = self._call_pricing(heuristic_pricer, dual_values, constraints=constraints)  # type: ignore[arg-type]
            self._note_bp_pricing(0 if not new_routes else len(new_routes))

            # 2. Fallback to exact pricer if heuristic fails
            if (
                not new_routes or max(rc for _, rc in new_routes) < self.params.optimality_gap
            ) and exact_pricer is not None:
                new_routes = self._call_pricing(exact_pricer, dual_values, constraints=constraints)  # type: ignore[arg-type]
            self._note_bp_pricing(0 if not new_routes else len(new_routes))

            if not new_routes:
                break

            feasible: List[Tuple[Route, float]] = []
            for route_nodes, rc in new_routes:
                cost, revenue, load, cov = heuristic_pricer.compute_route_details(route_nodes)
                route = Route(nodes=route_nodes, cost=cost, revenue=revenue, load=load, node_coverage=cov)
                # Exact solver already respects constraints via state enforcement.
                # Heuristic pricer requires post-hoc check for non-edge constraints (like Ryan-Foster).
                if (self.params.use_exact_pricing and exact_pricer is not None) or node.is_route_feasible(route):
                    feasible.append((route, rc))

            if not feasible:
                break

            # Task 4 & Part 5 Task 3: LP Termination via Lagrangian Bound
            max_rc = max(rc for _, rc in feasible)
            if max_rc < self.params.optimality_gap:
                break

            limit = self.params.vehicle_limit if self.params.vehicle_limit is not None else self.n_nodes
            lagrangian_bound = lp_obj + max_rc * limit
            best_lagrangian = min(best_lagrangian, lagrangian_bound)

            # Task 4: Basis Restoration
            # Basis is reset to None if the variable count changes significantly,
            # but Gurobi handles basis extension efficiently after add_route.
            # We explicitly save AFTER solving and Restore BEFORE solving.
            if abs(lp_obj - lagrangian_bound) < self.params.early_termination_gap:
                break

            for route, reduced_cost in feasible:
                if master.add_route(route) and getattr(self, "trace", None) is not None:
                    from logic.src.policies.helpers.solvers_and_matheuristics.exact_trace import emit_column

                    emit_column(master, route.nodes, reduced_cost, "ordinary", 2)
            self.num_columns_generated += len(feasible)

            # Task 3: Column Pool Management
            if (i + 1) % self.params.cleanup_frequency == 0:
                master.purge_useless_columns(self.params.cleanup_threshold)

        # Final solve for clean duals/solution
        if last_basis is not None:
            master.restore_basis(*last_basis)
        lp_obj, route_values = master.solve_lp_relaxation()
        return lp_obj, route_values, best_lagrangian

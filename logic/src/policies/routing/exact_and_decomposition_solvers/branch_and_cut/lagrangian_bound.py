"""Lagrangian k-tree pre-optimisation of the branch-and-cut solver.

Attributes:
    BCLagrangianMixin: k-tree Lagrangian bound and arc pre-filtering of ``BranchAndCutSolver``.

Example:
    >>> class BranchAndCutSolver(BCLagrangianMixin, BCCutSeparationMixin): ...
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.configs.policies.routing.exact_and_decomposition_solvers.bc import BCConfig
from logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model import VRPPModel


class BCLagrangianMixin:
    """Lagrangian k-tree bound computed before the MIP of :class:`BranchAndCutSolver`."""

    # Declared by BranchAndCutSolver.__init__; annotation only.
    params: BCConfig
    model: VRPPModel
    x_vars: Dict[Tuple[int, int], Any]

    def _pre_optimize_lagrangian(self) -> List[Set[int]]:  # noqa: C901
        """Root Node Strengthening via Lagrangian Relaxation.

        Refactored to extract violated Basic GSECs (1-tree cycles) for direct injection.

        Returns:
            List of unique violated node sets (cuts) discovered during relaxation.
        """
        if self.params.verbose:
            print("Strengthening Root Node via Lagrangian Relaxation...")

        n = self.model.n_nodes
        # Initial multipliers
        lambda_mult = np.zeros(n)

        # Held-Karp subgradient parameters (Fischetti et al. 1997, Section 5.1)
        hk_mu = 2.0  # scalar, starts at 2, halved on stagnation
        hk_no_improve = 0  # consecutive iterations without LB improvement
        HK_PATIENCE = 50  # halve mu after this many non-improving iterations
        best_upper_bound = float("inf")  # updated if a primal bound is available
        best_lower_bound = -float("inf")

        # Track unique sets S for injection
        unique_violated_sets: List[Set[int]] = []
        seen_sets = set()

        # Max 1000 iterations as per paper
        for _iter_count in range(1000):
            # 1. Update edge weights with multipliers
            modified_costs = {}
            for i, j in self.model.edges:
                cost = self.model.get_edge_cost(i, j)
                m_i = lambda_mult[i] if not np.isnan(lambda_mult[i]) else 0.0
                m_j = lambda_mult[j] if not np.isnan(lambda_mult[j]) else 0.0
                modified_costs[(i, j)] = cost - m_i - m_j

            # 2. Solve Minimum Cost K-Tree (Generalized 1-tree for fleet K)
            # We extract all intermediate components (Basic PC-SECs) from Kruskal's
            k_tree_edges, lb, components = self._solve_k_tree(modified_costs)

            # Identify cycles in the K-tree
            cycle = self._find_k_tree_cycle(k_tree_edges)
            if cycle and self.model.depot not in cycle:
                unique_violated_sets.append(cycle)

            # Add all discovered components S
            for s_set in components:
                if self.model.depot not in s_set and len(s_set) >= 2:
                    s_tuple = tuple(sorted(list(s_set)))
                    if s_tuple not in seen_sets:
                        unique_violated_sets.append(s_set)
                        seen_sets.add(s_tuple)

            # Add constant part of Lagrangian: 2 * sum(lambda_i)
            current_lb = float(lb) + 2 * np.sum(lambda_mult)

            # Record whether this iteration improved the lower bound BEFORE
            # updating best_lower_bound, so the stagnation check below is correct.
            improved_this_iter = current_lb > best_lower_bound
            if improved_this_iter:
                best_lower_bound = float(current_lb)

            # 3. Subgradient Update
            degrees = np.zeros(n)
            for i, j in k_tree_edges:
                degrees[i] += 1
                degrees[j] += 1

            # Subgradient for relaxed degree constraints: b - Ax
            # For customers: 2 - degree_i. For depot: 2K - degree_0.
            subgradient = 2.0 - degrees
            subgradient[0] = 2.0 * self.model.num_vehicles - degrees[0]

            if np.linalg.norm(subgradient) < 1e-6:
                break

            # Held-Karp Subgradient Update: t_k = μ * (UB - LB) / ||g||²
            grad_norm_sq = float(np.dot(subgradient, subgradient))
            if grad_norm_sq > 1e-10:
                ub = best_upper_bound
                if ub == float("inf"):
                    # Placeholder UB: 1.2x current LB to allow non-trivial steps
                    ub = float(current_lb) + abs(float(current_lb)) * 0.2 + 10.0

                hk_step = hk_mu * (ub - current_lb) / grad_norm_sq

                # Convergence safety: skip if step or grad is infinite
                if not np.isfinite(hk_step) or not np.all(np.isfinite(subgradient)):
                    break

                lambda_mult += hk_step * subgradient
                # Depot multiplier must be >= 0 (inequality constraint, not equality)
                lambda_mult[0] = max(0.0, lambda_mult[0])

            # Halve μ if no improvement for HK_PATIENCE consecutive iterations.
            # Uses improved_this_iter captured before best_lower_bound was updated.
            if improved_this_iter:
                hk_no_improve = 0
            else:
                hk_no_improve += 1
                if hk_no_improve >= HK_PATIENCE:
                    hk_mu /= 2.0
                    hk_no_improve = 0
                    if hk_mu < 1e-6:
                        break  # Multipliers have converged

        # 4. Initialize variable set J using edges with small reduced costs
        for i, j in self.model.edges:
            cost = self.model.get_edge_cost(i, j)
            red_cost = cost - lambda_mult[i] - lambda_mult[j]
            if red_cost < 1e-4:
                self.x_vars[(i, j)].VarHintVal = 1.0

        if self.params.verbose:
            print(f"Lagrangian phase complete. Root LB: {best_lower_bound:.2f}")

        return unique_violated_sets

    def _find_k_tree_cycle(self, edges: List[Tuple[int, int]]) -> Optional[Set[int]]:
        """Find a cycle in a K-tree graph using an iterative stack-based DFS.

        Replaces recursive implementation to prevent RecursionError on large graphs.

        Correctness note: each stack frame carries its own copy of the path from
        the start node to the current node. When we encounter a neighbour v that
        already appears in the current path (a back-edge), we have found a cycle
        and return it immediately. Globally-visited nodes are only skipped when
        they do not appear in the current path, preventing false negatives from
        cross-edges between separate DFS branches.

        Args:
            edges: List of edges (u, v) in the K-tree.

        Returns:
            A set of node indices forming the first cycle found, or None if acyclic.
        """
        adj: Dict[int, List[int]] = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        globally_visited: Set[int] = set()

        for start_node in range(self.model.n_nodes):
            if start_node in globally_visited:
                continue

            # Stack entries: (current_node, parent_node, path_from_start_to_current)
            # Each frame carries its own path copy so back-edge detection is local
            # to the branch, not polluted by other branches' visits.
            stack = [(start_node, -1, [start_node])]

            while stack:
                u, parent, path = stack.pop()

                # If u was already fully explored in a previous branch of this
                # component and is NOT in the current path, it cannot form a new
                # cycle through the current branch — skip it.
                if u in globally_visited and u not in path:
                    continue

                globally_visited.add(u)

                for v in adj.get(u, []):
                    if v == parent:
                        # Tree edge back to parent — not a cycle
                        continue
                    if v in path:
                        # Back-edge: v is an ancestor in the current DFS path.
                        # Extract the cycle as the subpath from v to u.
                        idx = path.index(v)
                        return set(path[idx:])
                    # Forward edge: extend the path and push to stack.
                    stack.append((v, u, path + [v]))

        # No cycle found in any component
        return None

    def _solve_k_tree(self, costs: Dict[Tuple[int, int], float]) -> Tuple[List[Tuple[int, int]], float, List[Set[int]]]:
        """Solve Minimum Cost K-Tree subproblem (Fischetti et al. 1997).

        Generalized from 1-tree to handle fleet size K.

        Args:
            costs: Dictionary of modified edge costs (cost - multipliers).

        Returns:
            A 3-tuple containing:
                - k_tree_edges: List of edges selected for the K-tree.
                - k_tree_cost: Total cost of the selected edges.
                - history_components: List of all intermediate components formed by Kruskal's.
        """
        n = self.model.n_nodes
        # MST on nodes 1...n-1
        customers = self.model.customers
        edges_to_mst = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                u, v = customers[i], customers[j]
                edges_to_mst.append((u, v, costs[(u, v)]))

        edges_to_mst.sort(key=lambda x: x[2])

        # Kruskal's with component tracking
        parent = list(range(n))
        node_to_comp = {i: {i} for i in range(n)}
        history_components = []

        def find(i):
            """Find set representative with path compression."""
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        mst_edges = []
        mst_cost = 0.0
        for u, v, cost in edges_to_mst:
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                mst_edges.append((u, v))
                mst_cost += cost

                # Merge components and store history
                new_comp = node_to_comp[root_u] | node_to_comp[root_v]
                history_components.append(new_comp)

                parent[root_u] = root_v
                node_to_comp[root_v] = new_comp

        # Select depot edges for the K-tree:
        # Always include the 2 cheapest depot edges (minimum required for connectivity),
        # then greedily add edges with strictly negative reduced cost up to 2K total.
        # Edges with non-negative reduced cost are excluded beyond the mandatory 2,
        # as they cannot improve the Lagrangian lower bound.
        k = self.model.num_vehicles
        depot_edges = []
        for j in self.model.customers:
            depot_edges.append((0, j, costs[(0, j)]))
        depot_edges.sort(key=lambda x: x[2])

        # Always select exactly 2*k cheapest depot edges.
        # For k=1 this enforces a valid 1-tree (exactly 2 depot edges).
        selected_depot_edges = depot_edges[: 2 * k]

        k_tree_edges = mst_edges + [(e[0], e[1]) for e in selected_depot_edges]
        k_tree_cost = mst_cost + sum(e[2] for e in selected_depot_edges)

        return k_tree_edges, k_tree_cost, history_components

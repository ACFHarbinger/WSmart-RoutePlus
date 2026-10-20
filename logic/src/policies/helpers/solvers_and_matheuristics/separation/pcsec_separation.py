"""pcsec separation methods relocated from engine."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from logic.src.policies.helpers.solvers_and_matheuristics.separation.inequality import (
    Inequality,
    PCSubtourEliminationCut,
)


class PCSECSeparationMixin:
    """Relocated pcsec separation methods."""

    def _separate_pcsec_exact(self, x_vals: np.ndarray, y_vals: Optional[np.ndarray], root_node: bool = False) -> None:
        """Exact prize-collecting subtour separation using minimum cut (max-flow).

        Fischetti et al. (1997), Section 3.1: GSEC_SEP.

        Finds violated PC-SECs by solving max-flow problems between a reference
        source node s and the depot.

        Args:
            x_vals: Edge values.
            y_vals: Node visit values.
            root_node: Whether to run exhaustive separation.
        """

        n = self.model.n_nodes
        adj = np.zeros((n, n))
        for idx, (i, j) in enumerate(self.model.edges):
            val = x_vals[idx]
            if val > 1e-4:
                adj[i, j] = val
                adj[j, i] = val

        # 1. Source-node selection (s)
        # We select node_i such that y*_i is maximum among nodes i for which y*_i >= eps.
        eps = 0.01
        visited_customers = [c for c in self.model.customers if y_vals is None or y_vals[c - 1] >= eps]
        if not visited_customers:
            return

        source_node = (
            max(visited_customers, key=lambda c: y_vals[c - 1]) if y_vals is not None else visited_customers[0]
        )

        for sink_node in visited_customers:
            if sink_node == source_node or sink_node == self.model.depot:
                continue

            try:
                # Solve max-flow from source_node to sink_node (isolation of S from depot)
                flow_result = maximum_flow(csr_matrix(adj), source_node, sink_node)
                max_flow_value = flow_result.flow_value
                flow_matrix = flow_result.flow.toarray()

                # Extract cut set S reachable from source in residual graph
                s_set = self._extract_min_cut(adj, flow_matrix, source_node, sink_node)
            except Exception:
                # Padberg-Rinaldi Min-Cut Fallback
                try:
                    G = nx.Graph()
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u, v] > 1e-4:
                                G.add_edge(u, v, capacity=adj[u, v])
                    max_flow_value, partition = nx.minimum_cut(G, source_node, sink_node)
                    s_side, t_side = partition
                    s_set = s_side if source_node in s_side else t_side
                except Exception:
                    continue

            if (
                s_set
                and self.model.depot not in s_set
                and len(s_set) >= 2
                and not any(
                    set(s_set) == set(existing.node_set)
                    for existing in self.pool
                    if isinstance(existing, PCSubtourEliminationCut)
                )
            ):
                # Re-select j from N\S to satisfy Form 2.3 requirement j ∉ S.
                not_in_s = set(range(self.model.n_nodes)) - s_set - {self.model.depot}
                if not not_in_s:
                    continue
                if y_vals is not None:
                    actual_j, actual_yj = max(
                        [(v, y_vals[v - 1] if v > 0 else 1.0) for v in not_in_s],
                        key=lambda x: x[1],
                    )
                else:
                    actual_j, actual_yj = next(iter(not_in_s)), 1.0

                yi = y_vals[source_node - 1] if y_vals is not None else 1.0
                actual_rhs = 2.0 * (yi + actual_yj - 1.0)

                # Only add if this recomputed rhs still gives a violated cut.
                if actual_rhs - max_flow_value > 0.01:
                    cut = PCSubtourEliminationCut(
                        set(s_set),
                        actual_rhs - max_flow_value,
                        facet_form="2.3",
                        node_i=source_node,
                        node_j=actual_j,
                    )
                    # RHS = 2*(yi + yj - 1) uses fractional LP values.
                    # This cut is valid for the current LP relaxation but may tighten
                    # below 2*(1+1-1)=2 for integer solutions. Mark as local so it is
                    # removed when backtracking past this B&B node.
                    cut.local_only = True
                    self.pool.append(cut)

    def _extract_min_cut(self, capacity: np.ndarray, flow: np.ndarray, source: int, sink: int) -> Set[int]:
        """Extract source-side of minimum cut using BFS on residual graph.

        Paper Section 3.1: the cut set is the set of nodes reachable from
        the source in the residual graph.

        Args:
            capacity: Capacity matrix (support graph edges).
            flow: Flow matrix from max-flow result.
            source: Source node index (depot or reference node).
            sink: Sink node index (customer).

        Returns:
            Set of nodes reachable from source in the residual graph.
        """

        residual = capacity - flow
        visited: Set[int] = set()
        queue = [source]
        visited.add(source)

        while queue:
            u = queue.pop(0)
            for v in range(len(residual)):
                if residual[u, v] > 1e-6 and v not in visited:
                    visited.add(v)
                    queue.append(v)

        return visited

    def _separate_gsec_h2(self, x_vals: np.ndarray, y_vals: Optional[np.ndarray]) -> None:
        """Kruskal-based Heuristic (GSEC_H2) for GSEC separation.

        Algorithm:
        1. Sort edges e with x_e* > 0 in descending order.
        2. Construct forest using Kruskal's merging logic.
        3. For each merge (Va, Vb), evaluate GSEC violation for S = Va U Vb.
        4. GSEC: sum_{e in delta(S)} x_e >= 2(y_i + y_j - 1)
           where i in S and j not in S have highest y* values.

        Args:
            x_vals: Edge values.
            y_vals: Node visit values.

        """

        # Step 1: Edge Sorting
        support_edges = []
        for idx, (i, j) in enumerate(self.model.edges):
            if x_vals[idx] > 1e-4:
                support_edges.append((i, j, x_vals[idx]))

        support_edges.sort(key=lambda x: x[2], reverse=True)

        # Step 2: Forest Construction (DSU)
        n = self.model.n_nodes
        parent = list(range(n))
        components: Dict[int, Set[int]] = {i: {i} for i in range(n)}

        def find(i):
            """Finds the root of the disjoint set for node i."""
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        y_vals_full = np.ones(n)
        if y_vals is not None:
            y_vals_full[1:] = y_vals

        for u, v, _ in support_edges:
            root_u = find(u)
            root_v = find(v)

            if root_u != root_v:
                # Potential Merge: Check violation for S = Va U Vb
                # Merge
                new_component = components[root_u] | components[root_v]
                parent[root_u] = root_v
                components[root_v] = new_component
                del components[root_u]

                # Step 3: Violation Check for S = new_component
                if self.model.depot in new_component:
                    continue  # S must not contain depot for SEC/GSEC cut

                if len(new_component) < 2:
                    continue

                # Find i in S with max y*
                max_y_in = max(y_vals_full[node] for node in new_component)
                # Find j not in S with max y*
                not_in_s = set(range(n)) - new_component
                max_y_out = max(y_vals_full[node] for node in not_in_s)

                gsec_rhs = 2.0 * (max_y_in + max_y_out - 1.0)
                if gsec_rhs <= 0.0:
                    continue

                cut_val = self._get_cut_value(new_component, x_vals)
                violation = gsec_rhs - cut_val

                if violation > 0.01:
                    self.pool.append(PCSubtourEliminationCut(set(new_component), violation))

    def _strengthen_pool(self, ineq_list: List[Inequality], x_vals: np.ndarray, y_vals: Optional[np.ndarray]) -> None:
        """Apply Procedure PCSEC_BUILD to all subtour sets.

        Refines sets S and selects the strongest facet form (2.1, 2.2, 2.3).

        Args:
            ineq_list: List of inequalities to strengthen.
            x_vals: Edge values.
            y_vals: Node visit values.
        """

        for i in range(len(ineq_list)):
            cut = ineq_list[i]
            if not isinstance(cut, PCSubtourEliminationCut):
                continue

            # 1. Refine set S using PCSEC_BUILD logic
            s_set = self._refine_pcsec_build(cut.node_set, x_vals, y_vals)
            cut.node_set = s_set

            # 2. Select strongest facet form
            # Identify i in S and j not in S with max y*
            max_yi = 1.0
            best_i = next(iter(s_set))
            if y_vals is not None:
                y_in = [(node, y_vals[node - 1]) for node in s_set if node > 0]
                if y_in:
                    best_i, max_yi = max(y_in, key=lambda x: x[1])

            max_yj = 1.0
            best_j = self.model.depot
            if y_vals is not None:
                not_s = set(range(self.model.n_nodes)) - s_set
                y_out = [(node, y_vals[node - 1] if node > 0 else 1.0) for node in not_s]
                if y_out:
                    best_j, max_yj = max(y_out, key=lambda x: x[1])

            # Recalculate violation for each form (2.1, 2.2, 2.3)
            cut_val = self._get_cut_value(s_set, x_vals)

            # Form 2.3: sum x_e >= 2(yi + yj - 1)
            vio_2_3 = (2.0 * (max_yi + max_yj - 1.0) - cut_val) if y_vals is not None else (2.0 - cut_val)

            # Form 2.2: sum x_e >= 2yi
            vio_2_2 = (2.0 * max_yi - cut_val) if y_vals is not None else (2.0 - cut_val)

            # Form 2.1: sum x_e >= 2
            vio_2_1 = 2.0 - cut_val

            # Select strongest (max violation)
            if vio_2_1 > vio_2_2 + 1e-4 and vio_2_1 > vio_2_3 + 1e-4:
                cut.facet_form = "2.1"
                cut.rhs = 2.0
                cut.violation = vio_2_1
                cut.local_only = False  # Form 2.1 is always global
            # Form 2.2: Σ x_e - 2y_i >= 0
            elif vio_2_2 > vio_2_3 + 1e-4:
                cut.facet_form = "2.2"
                cut.node_i = best_i
                cut.rhs = 0.0
                cut.violation = vio_2_2
                cut.local_only = True  # Depends on specific y_i, mark local
            # Form 2.3: Σ x_e - 2y_i - 2y_j >= -2
            else:
                cut.facet_form = "2.3"
                cut.node_i = best_i
                cut.node_j = best_j
                cut.rhs = -2.0
                cut.violation = vio_2_3
                cut.local_only = True  # Depends on y_i, y_j, mark local

    def _refine_pcsec_build(self, s_set: Set[int], x_vals: np.ndarray, y_vals: Optional[np.ndarray]) -> Set[int]:
        """Procedure PCSEC_BUILD (Fischetti et al. 1997, Section 3.1).

        Attempts to refine the set S by moving nodes to minimize cut value sum_{e in delta(S)} x_e.

        Args:
            s_set: Initial set of nodes.
            x_vals: Edge values.
            y_vals: Node visit values.

        Returns:
            Refined set of nodes.
        """

        refined_s = set(s_set)
        improved = True

        while improved:
            improved = False
            for u in list(refined_s):
                # Delta cut value if u is moved from S to V\S
                # delta_val = edges(u, V\S) - edges(u, S)
                in_s_sum = sum(x_vals[self.model.edge_to_idx[(u, v)]] for v in refined_s if u != v)
                all_u_sum = sum(x_vals[self.model.edge_to_idx[(u, v)]] for v in range(self.model.n_nodes) if v != u)
                out_s_sum = all_u_sum - in_s_sum

                # We want to minimize sum_{e in delta(S)} x_e.
                # Moving u out changes it from out_s_sum to in_s_sum inside the cut.
                # Actually, sum_{e in delta(S)} = sum_{i in S, j not in S} x_ij.
                # Moving u from S to V\S removes sum_{j in V\S} x_uj and adds sum_{j in S} x_uj.
                if in_s_sum - out_s_sum < -0.01 and len(refined_s) > 2:
                    refined_s.remove(u)
                    improved = True
                    break

            if improved:
                continue

            for u in range(self.model.n_nodes):
                if u in refined_s or u == self.model.depot:
                    continue

                # Check moving u from V\S into S
                in_s_sum = sum(x_vals[self.model.edge_to_idx[(u, v)]] for v in refined_s)
                all_u_sum = sum(x_vals[self.model.edge_to_idx[(u, v)]] for v in range(self.model.n_nodes) if v != u)
                out_s_sum = all_u_sum - in_s_sum

                if out_s_sum - in_s_sum < -0.01 and u != self.model.depot:
                    # Adding u to S reduces cut capacity
                    refined_s.add(u)
                    improved = True
                    break

        return refined_s

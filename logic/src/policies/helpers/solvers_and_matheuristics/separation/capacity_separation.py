"""capacity separation methods relocated from engine."""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from logic.src.policies.helpers.solvers_and_matheuristics.separation.inequality import (
    CapacityCut,
)


class CapacitySeparationMixin:
    """Relocated capacity separation methods."""

    def _separate_capacity_cuts(self, x_vals: np.ndarray, y_vals: Optional[np.ndarray]) -> None:
        """Separate capacity inequalities using demand-based clustering.

        For sets S with high demand, check if enough vehicles (edges crossing cut) exist.

        Args:
            x_vals: Edge values.
            y_vals: Node visit values.
        """

        # Try different seed nodes and grow sets based on demand
        for seed in self.model.customers:
            # Skip if seed node not visited
            if y_vals is not None and y_vals[seed - 1] < 0.5:
                continue
            node_set = {seed}
            remaining = set(self.model.customers) - {seed}
            current_demand = self.model.get_node_demand(seed)

            # Grow set until capacity is reached
            while current_demand < self.model.capacity * 0.9 and remaining:
                # Find nearest node to current set
                best_node = None
                best_dist = float("inf")

                for candidate in remaining:
                    min_dist = min(self.model.cost_matrix[candidate, node] for node in node_set)
                    if min_dist < best_dist:
                        best_dist = min_dist
                        best_node = candidate

                if best_node is None:
                    break

                candidate_demand = self.model.get_node_demand(best_node)
                if current_demand + candidate_demand > self.model.capacity * 1.5:
                    break  # Don't grow too large

                node_set.add(best_node)
                remaining.remove(best_node)
                current_demand += candidate_demand

            # Check capacity inequality violation
            if len(node_set) >= 2:
                # In VRPP, we only care about demand of nodes we actually visit
                if y_vals is not None:
                    total_demand = sum(self.model.get_node_demand(i) for i in node_set if i > 0 and y_vals[i - 1] > 0.5)
                else:
                    total_demand = self.model.total_demand(node_set)

                if total_demand <= 0.01:
                    continue

                min_vehicles = int(np.ceil(total_demand / self.model.capacity))

                cut_edges = self.model.delta(node_set)
                cut_value = sum(
                    x_vals[self.model.edge_to_idx[(min(e), max(e))]]
                    for e in cut_edges
                    if (min(e), max(e)) in self.model.edge_to_idx
                )

                required_value = 2.0 * min_vehicles
                violation = required_value - cut_value

                if violation > 0.01:
                    self.pool.append(CapacityCut(node_set, total_demand, self.model.capacity, violation))

    def _separate_capacity_cuts_maxflow_heuristic(  # noqa: C901
        self,
        x_vals: np.ndarray,
        y_vals: Optional[np.ndarray],
        root_node: bool = False,
        max_cuts: int = 50,
    ) -> None:
        """Max-Flow-based heuristic separation of Rounded Capacity Cuts (RCCs).

        Args:
            x_vals: Edge values.
            y_vals: Node visit values.
            root_node: Whether to run exhaustive separation.
            max_cuts: Maximum number of cuts to identify.

        Mathematical Formulation:
        -------------------------
        The Rounded Capacity Cut (RCC) is defined as:
            Σ_{i ∈ S, j ∉ S} x_{ij} ≥ 2 * ⌈d(S) / Q⌉
        where S ⊆ Customers, d(S) is total demand of S, and Q is vehicle capacity.

        Separation Strategy:
        --------------------
        We reformulate the problem as finding a set S such that the cut capacity
        is less than the required fleet size. For a fixed set S, we solve
        max-flow problems between the depot (Node 0) and each customer node i
        in the support graph G*=(V, E*) where edge weights correspond to fractional
        x* values. Following Fischetti et al. (1997), the customer-side of each
        computed min-cut is a candidate set S for an RCC.

        This heuristic method performs extensive separation by solving max-flow to ALL
        nodes with non-zero visitation probability y*_i. While highly rigorous, it operates
        on the fractional support graph and does not strictly guarantee finding all
        mathematically violated facets.

        References:
        -----------
        - Fischetti, M., Salazar-González, J. J., & Toth, P. (1997). "A branch-and-cut
          algorithm for the symmetric generalized traveling salesman problem."
          Operations Research, 45(3), 378-394.
        - Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004). "A new branch-and-cut
          algorithm for the capacitated vehicle routing problem."
          Mathematical Programming, 100(2), 423-445.
        """
        n = self.model.n_nodes
        # Build support graph adjacency
        adj = np.zeros((n, n))
        for idx, (i, j) in enumerate(self.model.edges):
            val = x_vals[idx]
            if val > 1e-4:
                adj[i, j] = val
                adj[j, i] = val

        # Exact Separation: isolate most fractional nodes to avoid O(N) max-flows.
        # Sort candidates by fractional visitation value descending.
        if y_vals is not None:
            visited_customers = [c for c in self.model.customers if y_vals[c - 1] > 1e-4]
            visited_customers.sort(key=lambda c: y_vals[c - 1], reverse=True)
        else:
            visited_customers = self.model.customers[:]

        # RCIs separate the depot from customers to ensure enough vehicles leave.
        source_node = self.model.depot

        added = 0
        for sink in visited_customers:
            if sink == source_node:
                continue
            if added >= max_cuts:
                break
            try:
                flow_result = maximum_flow(csr_matrix(adj), source_node, sink)
                source_side = self._extract_min_cut(adj, flow_result.flow.toarray(), source_node, sink)
                cut_set = set(range(self.model.n_nodes)) - source_side
            except Exception:
                # Padberg-Rinaldi Min-Cut Fallback
                try:
                    G = nx.Graph()
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u, v] > 1e-4:
                                G.add_edge(u, v, capacity=adj[u, v])
                    cut_value, partition = nx.minimum_cut(G, source_node, sink)
                    s_side, t_side = partition
                    cut_set = t_side if source_node in s_side else s_side
                except Exception:
                    continue

            # Prune invalid cuts: single-node sets, depot in set, or empty sets
            # Single-node sets are already handled by degree constraints (∑ x_ij = 2y_i)
            if not cut_set or len(cut_set) <= 1 or self.model.depot in cut_set:
                continue

            current_y_vals = y_vals if y_vals is not None else np.ones(len(self.model.customers))
            total_demand = sum(self.model.get_node_demand(i) for i in cut_set if i > 0 and current_y_vals[i - 1] > 0.1)

            if total_demand <= 1e-4:
                continue

            min_vehicles = int(np.ceil(total_demand / self.model.capacity))
            if min_vehicles < 1:
                continue

            cut_val = self._get_cut_value(cut_set, x_vals)
            required_val = 2.0 * min_vehicles

            violation = required_val - cut_val
            if violation > 0.01 and not any(
                set(cut_set) == set(existing.node_set) for existing in self.pool if isinstance(existing, CapacityCut)
            ):
                self.pool.append(CapacityCut(set(cut_set), total_demand, self.model.capacity, violation))
                added += 1

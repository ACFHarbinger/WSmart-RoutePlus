"""Separation Algorithms for VRPP Cutting Planes.

Implements exact and heuristic separation procedures for identifying violated
valid inequalities in the VRPP linear programming relaxation.

References:
    Fischetti, M., Lodi, A., & Toth, P. (1997). "A Branch-and-Cut Algorithm for the Symmetric
    Generalized Traveling Salesman Problem". Operations Research, 45(2), 326-349.

    Lysgaard, J., Letchford, A. N., & Eglese, R. W. (2004). "A new branch-and-cut algorithm
    for the capacitated vehicle routing problem". Mathematical Programming, 100(2), 423-445.

    Padberg, M., & Rinaldi, G. (1991). "A Branch-and-cut Algorithm for the Resolution of
    Large-scale Symmetric Traveling Salesman Problems". SIAM Review, 33(1), 60-100.

    - Prize-Collecting Subtour Elimination Constraints (PC-SEC) follow Fischetti et al. (1997)
    - Capacity Constraints and fractional separation follow Lysgaard et al. (2004)
    - Comb inequalities are disabled in favor of Gurobi's internal polyhedral cuts

Attributes:
    SeparationEngine: Core engine for identifying violated inequalities.

Example:
    >>> engine = SeparationEngine(model, enable_heuristic_rcc_separation=True)
    >>> cuts = engine.separate_fractional(x_star, y_star)

Parts are relocated to pcsec_separation, capacity_separation.
"""

from __future__ import annotations

from typing import List, Optional, Set

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from logic.src.policies.helpers.solvers_and_matheuristics.separation.inequality import (
    Inequality,
    PCSubtourEliminationCut,
)

from .capacity_separation import CapacitySeparationMixin
from .pcsec_separation import PCSECSeparationMixin


class SeparationEngine(CapacitySeparationMixin, PCSECSeparationMixin):
    """Separation algorithms for finding violated inequalities.

    Implements exact and heuristic separation procedures for identifying violated
    valid inequalities in the VRPP linear programming relaxation.

    Attributes:
        model: VRPPModel instance.
        pool (List[Inequality]): Pool of stored inequalities.
        enable_heuristic_rcc_separation (bool): Whether to enable RCC separation.
        enable_comb_cuts (bool): Whether to enable comb inequality separation.

    Example:
        >>> engine = SeparationEngine(model)
        >>> violated_cuts = engine.separate_integer(x_vals)
    """

    # Global toggle for comb inequality separation
    # Set to False to disable custom comb cuts and rely on Gurobi's internal heuristics

    # Task 12c: Period for running expensive exact separation at non-root nodes.
    # Set to 1 for exhaustive separation every iteration.
    _EXACT_SEP_PERIOD = 3

    def __init__(
        self,
        model,
        enable_heuristic_rcc_separation: bool = True,
        enable_comb_cuts: bool = False,
    ):
        """
        Initialize separation engine.

        Args:
            model: VRPPModel instance.
            enable_heuristic_rcc_separation: Whether to enable heuristic connected-component RCC separation.
            enable_comb_cuts: Whether to enable comb inequality separation.

        """
        self.model = model
        self.pool: List[Inequality] = []  # Pool of stored inequalities
        self.enable_heuristic_rcc_separation = enable_heuristic_rcc_separation
        self.enable_comb_cuts = enable_comb_cuts

    def separate_integer(
        self,
        x_vals: np.ndarray,
        y_vals: Optional[np.ndarray] = None,
        max_cuts: int = 100,
        iteration: int = 0,
        sec_only: bool = False,
    ) -> List[Inequality]:
        """
        Fast separation for integer solutions (MIPSOL callback).

        Uses efficient heuristic methods suitable for integer values:
        - Connected components for subtours (O(n²))
        - Demand-based clustering for capacity cuts (O(n²))

        Args:
            x_vals: Integer edge values from MIP solution.
            y_vals: Integer node visit values from MIP solution.
            max_cuts: Maximum number of cuts to return.
            iteration: Current iteration number.
            sec_only: Whether to only separate Subtour Elimination Constraints.


        Returns:
            List of violated inequalities sorted by degree of violation.
        """
        self.pool = []

        # Step 1: Fast subtour separation using connected components
        self._separate_disconnected_components(x_vals, y_vals)

        # Step 2: Heuristic SEC separation (Kruskal-based GSEC_H2)
        if not sec_only:
            self._separate_gsec_h2(x_vals, y_vals)

        # Step 3: Procedure PCSEC_BUILD: Strengthen sets and select strongest facets
        self._strengthen_pool(self.pool, x_vals, y_vals)

        # Step 4: Filter and return most violated cuts
        violated = [ineq for ineq in self.pool if ineq.violation > 0.01]
        violated.sort()  # Sort by violation descending
        return violated[:max_cuts]

    def separate_fractional(
        self,
        x_vals: np.ndarray,
        y_vals: Optional[np.ndarray] = None,
        max_cuts: int = 50,
        iteration: int = 0,
        node_count: int = 0,
    ) -> List[Inequality]:
        """Exact separation for fractional LP nodes (MIPNODE callback).

        WARNING: This is a heuristic separation procedure based on connected
        components of the LP support graph. It finds violated RCCs only for
        subsets S that form a connected component. Violated RCCs for
        non-component subsets are missed. Exact RCC separation requires
        solving a max-flow problem per candidate set (see Padberg & Rinaldi 1991;
        Lysgaard et al. 2004). The enable_heuristic_rcc_separation parameter
        does NOT switch to exact separation — it enables this heuristic.

        Args:
            x_vals: Fractional edge values from LP solution.
            y_vals: Fractional node visit values from LP solution.
            max_cuts: Maximum number of cuts to return.
            iteration: Current iteration number.
            node_count: Number of nodes in the B&B tree explored so far.

        Returns:
            List of violated inequalities sorted by degree of violation.
        """

        self.pool = []

        # Strategy: More aggressive separation at root node
        if node_count == 0:
            # Root node: Full exact separation
            self._separate_subtours_heuristic(x_vals, y_vals)
            self._separate_capacity_cuts(x_vals, y_vals)
            self._separate_gsec_h2(x_vals, y_vals)
            self._separate_pcsec_exact(x_vals, y_vals, root_node=True)

            # Exact fractional capacity separation (toggle-controlled)
            if self.enable_heuristic_rcc_separation:
                self._separate_capacity_cuts_maxflow_heuristic(x_vals, y_vals, root_node=True)
        else:
            # Shallow nodes: Limited exact separation
            self._separate_subtours_heuristic(x_vals, y_vals)
            self._separate_capacity_cuts(x_vals, y_vals)

            # Run exact separation less frequently
            if iteration % self._EXACT_SEP_PERIOD == 0:
                self._separate_gsec_h2(x_vals, y_vals)
                self._separate_pcsec_exact(x_vals, y_vals, root_node=False)

                # Exact fractional capacity separation (toggle-controlled)
                if self.enable_heuristic_rcc_separation:
                    self._separate_capacity_cuts_maxflow_heuristic(x_vals, y_vals, root_node=False)

        # Filter and return most violated cuts
        violated = [ineq for ineq in self.pool if ineq.violation > 0.01]

        # Procedure GSEC_BUILD: Strengthen sets and select strongest facets
        self._strengthen_pool(violated, x_vals, y_vals)

        violated.sort()  # Sort by violation descending
        return violated[:max_cuts]

    def _separate_subtours_heuristic(self, x_vals: np.ndarray, y_vals: Optional[np.ndarray]) -> None:
        """Heuristic subtour elimination using connected components and greedy expansion.

        Args:
            x_vals: Edge values.
            y_vals: Node visit values.
        """

        self._separate_disconnected_components(x_vals, y_vals)
        self._separate_weak_subtours(x_vals, y_vals)

    def _separate_disconnected_components(self, x_vals: np.ndarray, y_vals: Optional[np.ndarray]) -> None:
        """Separate subtours based on disconnected components in the support graph.

        Args:
            x_vals: Edge values.
            y_vals: Node visit values.
        """

        threshold = 0.5
        support_edges = [(i, j) for idx, (i, j) in enumerate(self.model.edges) if x_vals[idx] >= threshold]

        if not support_edges:
            return

        n = self.model.n_nodes
        graph = np.zeros((n, n))
        for i, j in support_edges:
            graph[i, j] = 1
            graph[j, i] = 1

        n_components, labels = connected_components(csgraph=csr_matrix(graph), directed=False)
        if n_components <= 1:
            return

        depot_component = labels[self.model.depot]
        for comp_id in range(n_components):
            if comp_id == depot_component:
                continue

            component = set(np.where(labels == comp_id)[0])
            if len(component) < 2:
                continue

            # In VRPP, SEC only applies if at least one node in component is visited
            is_visited = True
            if y_vals is not None:
                is_visited = any(y_vals[node - 1] > 0.5 for node in component if node != self.model.depot)

            if not is_visited:
                continue

            cut_value = self._get_cut_value(component, x_vals)
            violation = 2.0 - cut_value
            if violation > 0.01:
                self.pool.append(PCSubtourEliminationCut(component, violation))

    def _separate_weak_subtours(self, x_vals: np.ndarray, y_vals: Optional[np.ndarray]) -> None:
        """Separate subtours by greedily expanding sets with high internal connectivity.

        Args:
            x_vals: Edge values.
            y_vals: Node visit values.
        """

        threshold_weak = 0.1
        n = self.model.n_nodes
        graph_weak = np.zeros((n, n))
        for idx, (i, j) in enumerate(self.model.edges):
            if x_vals[idx] >= threshold_weak:
                val = x_vals[idx]
                graph_weak[i, j] = val
                graph_weak[j, i] = val

        for center in self.model.customers:
            # Skip if center node not visited
            if y_vals is not None and y_vals[center - 1] < 0.5:
                continue
            node_set = {center}
            candidates = set(self.model.customers) - {center}

            while len(node_set) < min(5, len(self.model.customers)):
                best_node, best_conn = None, 0.0
                for candidate in candidates:
                    conn = sum(graph_weak[candidate, node] for node in node_set)
                    if conn > best_conn:
                        best_conn = conn
                        best_node = candidate

                if best_node is None or best_conn < 0.1:
                    break

                node_set.add(best_node)
                candidates.remove(best_node)

            if len(node_set) >= 2:
                cut_value = self._get_cut_value(node_set, x_vals)
                violation = 2.0 - cut_value
                if violation > 0.01:
                    self.pool.append(PCSubtourEliminationCut(node_set, violation))

    def _get_cut_value(self, node_set: Set[int], x_vals: np.ndarray) -> float:
        """Compute the sum of x values for edges in the cut delta(S).

        Args:
            node_set: Set of nodes in the component.
            x_vals: Edge values.

        Returns:
            The capacity of the cut delta(S).
        """

        cut_edges = self.model.delta(node_set)
        return sum(
            x_vals[self.model.edge_to_idx[(min(e), max(e))]]
            for e in cut_edges
            if (min(e), max(e)) in self.model.edge_to_idx
        )

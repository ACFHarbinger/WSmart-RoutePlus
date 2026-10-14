"""
Fix-and-Optimize Corridor Method

Attributes:
    FixAndOptimizeRefiner: A class that solves the Benders decomposition sub-problem for a single day.

Example:
    >>> from fix_and_optimize import FixAndOptimizeRefiner
    >>> refiner = FixAndOptimizeRefiner()
    >>> refiner.solve()
"""

from typing import Any, List, Set

import numpy as np


class FixAndOptimizeRefiner:
    """
    Exact decomposition sub-routine that polishes an incumbent solution by
    iteratively unfixing targeted bin clusters and re-optimising them with
    the exact BPC engine.

    Role in the Pipeline
    --------------------
    Called after the Benders loop converges (or reaches its iteration limit)
    to improve the incumbent solution quality via a sequence of restricted
    re-optimisations.  Each iteration:

    1. Selects a cluster of ``max_unfix`` bins to release using one of the
       two cluster-selection strategies.
    2. Checks the cluster against the Jaccard-similarity tabu list to avoid
       revisiting recently explored neighbourhoods.
    3. Passes the corridor (fixed background + unfixed cluster) to the exact
       BPC engine via ``bpc_engine.solve_corridor(cluster, warm_routes)``.

    Cluster Selection Strategies
    ----------------------------
    ``"overflow_urgency"``
        Ranks all bins by ascending days-to-overflow and selects the
        ``max_unfix`` most urgent bins.  Prioritises bins at risk of
        unserved overflow, improving solution feasibility under capacity
        constraints.

    ``"scenario_divergence"``
        Ranks bins by descending cross-scenario fill variance at the current
        day and selects the ``max_unfix`` most uncertain bins.  Focuses
        refinement effort on bins whose optimal service day is most ambiguous
        across the scenario tree.

    Tabu List
    ---------
    To prevent short cycles, clusters with Jaccard similarity > 0.8 to any
    recently used cluster are rejected.  The list stores the last
    ``tabu_length`` clusters in insertion order (FIFO eviction).

    Warm Start
    ----------
    Before invoking the BPC engine, the global column pool is filtered for
    routes that exclusively service bins within the unfixed cluster.  These
    routes are passed as warm-start columns to the BPC RMP, reducing the
    initial column-generation overhead for the corridor sub-problem.

    Attributes:
    ----------
    tabu_list : List[Set[int]]
        Recently used cluster sets, bounded to the last ``tabu_length`` entries.
    tabu_length : int
        Maximum number of clusters retained in the tabu list.
    max_unfix : int
        Number of bins simultaneously unfixed in each corridor.
    """

    def __init__(self, tabu_length: int = 10, max_unfix: int = 5):
        """
        Args:
            tabu_length: Capacity of the tabu list.  Clusters older than
                the last ``tabu_length`` iterations are evicted and may be
                re-selected.
            max_unfix: Maximum corridor width — the number of bins
                simultaneously unfixed per iteration.  Larger values yield
                better improvement per iteration at the cost of sub-MIP
                solve time.
        """
        self.tabu_list: List[Set[int]] = []
        self.tabu_length = tabu_length
        self.max_unfix = max_unfix

    def _is_tabu(self, candidate_set: Set[int]) -> bool:
        """
        Check whether a candidate cluster is tabu.

        A cluster is rejected if it has Jaccard similarity > 0.8 with any
        entry currently in the tabu list:

            J(A, B) = |A ∩ B| / |A ∪ B|  >  0.8

        An empty candidate set is unconditionally tabu to avoid degenerate
        no-op iterations.

        Args:
            candidate_set: Proposed cluster of bin ids to unfix.

        Returns:
            ``True`` if the cluster is tabu (should be skipped),
            ``False`` otherwise.
        """
        if not candidate_set:
            return True

        for tabu_set in self.tabu_list:
            intersection = len(candidate_set.intersection(tabu_set))
            union = len(candidate_set.union(tabu_set))
            if union > 0 and (intersection / union) > 0.8:
                return True
        return False

    def _add_to_tabu(self, cluster: Set[int]) -> None:
        """
        Add a cluster to the tabu list, evicting the oldest entry if needed.

        Args:
            cluster: Cluster of bin ids to record as recently explored.
        """
        self.tabu_list.append(cluster)
        if len(self.tabu_list) > self.tabu_length:
            self.tabu_list.pop(0)

    def select_cluster_overflow_urgency(self, unrouted: List[int], days_to_overflow: np.ndarray) -> Set[int]:
        """
        Select the ``max_unfix`` most overflow-urgent bins.

        Bins are ranked by ascending ``days_to_overflow`` (0-indexed array,
        bin k corresponds to index k-1).  The ``max_unfix`` bins with the
        fewest days until projected overflow are selected for the corridor.

        Args:
            unrouted: List of all candidate bin ids (1-based).
            days_to_overflow: 1-D array of shape (n_bins,) where entry i
                gives the predicted days until bin i+1 overflows.

        Returns:
            Set of at most ``max_unfix`` bin ids selected by overflow urgency.
        """
        if not unrouted:
            return set()

        urn_indices = np.argsort(days_to_overflow)
        cluster: Set[int] = set()
        for idx in urn_indices:
            bin_id = idx + 1
            if len(cluster) >= self.max_unfix:
                break
            cluster.add(bin_id)

        return cluster

    def select_cluster_scenario_divergence(
        self, all_nodes: List[int], scenario_tree: Any, current_day: int
    ) -> Set[int]:
        """
        Select the ``max_unfix`` bins with the highest cross-scenario fill
        variance at ``current_day``.

        Bins with large variance are those whose optimal service decision is
        most sensitive to the scenario realisation, making them the most
        valuable targets for exact re-optimisation.

        Args:
            all_nodes: All candidate bin ids (1-based).
            scenario_tree: ScenarioTree; ``get_scenarios_at_day(day)`` must
                return a list of scenario objects with a ``.wastes`` array of
                shape (n_bins,).
            current_day: Planning day index used to retrieve the scenario
                fill distribution.

        Returns:
            Set of at most ``max_unfix`` bin ids ranked by descending fill
            variance.  Falls back to the first ``max_unfix`` nodes in
            ``all_nodes`` if no scenarios are available for ``current_day``.
        """
        scenarios = scenario_tree.get_scenarios_at_day(current_day)
        if not scenarios:
            return set(all_nodes[: self.max_unfix])

        all_wastes = np.stack([s.wastes for s in scenarios])
        variances = np.var(all_wastes, axis=0)

        urn_indices = np.argsort(variances)[::-1]
        cluster: Set[int] = set()
        for idx in urn_indices:
            bin_id = idx + 1
            if len(cluster) >= self.max_unfix:
                break
            cluster.add(bin_id)

        return cluster

    def refine(
        self,
        current_incumbent: Any,
        bpc_engine: Any,
        scenario_tree: Any,
        current_day: int,
        days_to_overflow: np.ndarray,
        global_column_pool: List[Any],
        strategy: str = "overflow_urgency",
    ) -> Any:
        """
        Execute one Fix-and-Optimize corridor refinement pass.

        Selects a cluster using the specified strategy, validates it against
        the tabu list, warm-starts the BPC engine with filtered corridor
        routes, and returns the refined solution (or the unmodified incumbent
        if the cluster is tabu or the BPC engine interface is unavailable).

        Args:
            current_incumbent: Current best solution object; returned unchanged
                if the corridor is tabu or BPC cannot be invoked.
            bpc_engine: Exact BPC engine object.  Must expose
                ``solve_corridor(cluster: Set[int], warm_routes: List[Route])``
                returning a refined solution object.
            scenario_tree: ScenarioTree for scenario-divergence cluster
                selection.
            current_day: Planning day index for scenario retrieval.
            days_to_overflow: 1-D array of overflow urgency scores; entry i
                corresponds to bin i+1.
            global_column_pool: Full set of currently generated routes.
                Routes whose nodes are a subset of the unfixed cluster are
                filtered out and used to warm-start the corridor BPC solve.
            strategy: Cluster selection strategy — ``"overflow_urgency"`` or
                ``"scenario_divergence"``.

        Returns:
            Refined solution from ``bpc_engine.solve_corridor``, or
            ``current_incumbent`` if the corridor was tabu or the engine
            interface was unavailable.
        """
        all_nodes = [i + 1 for i in range(len(days_to_overflow))]

        if strategy == "overflow_urgency":
            cluster = self.select_cluster_overflow_urgency(all_nodes, days_to_overflow)
        else:
            cluster = self.select_cluster_scenario_divergence(all_nodes, scenario_tree, current_day)

        if self._is_tabu(cluster):
            return current_incumbent

        self._add_to_tabu(cluster)

        # Warm-start: filter routes that exclusively service the unfixed corridor
        filtered_routes = [route for route in global_column_pool if all(n in cluster or n == 0 for n in route.nodes)]

        if hasattr(bpc_engine, "solve_corridor"):
            return bpc_engine.solve_corridor(cluster, filtered_routes)

        return current_incumbent

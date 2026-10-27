"""Destroy operators of the SARSA-driven ALNS.

This module provides the destruction operator mixin for the ALNS-SARSA solver.
"""

from random import Random
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

from logic.src.policies.helpers.operators.destroy_ruin import (
    cluster_removal as cluster_removal_op,
)
from logic.src.policies.helpers.operators.destroy_ruin import (
    random_removal as random_removal_op,
)
from logic.src.policies.helpers.operators.destroy_ruin import (
    shaw_profit_removal as shaw_profit_removal_op,
)
from logic.src.policies.helpers.operators.destroy_ruin import (
    shaw_removal as shaw_removal_op,
)
from logic.src.policies.helpers.operators.destroy_ruin import (
    string_removal as string_removal_op,
)
from logic.src.policies.helpers.operators.destroy_ruin import (
    worst_profit_removal as worst_profit_removal_op,
)
from logic.src.policies.helpers.operators.destroy_ruin import (
    worst_removal as worst_removal_op,
)


class ALNSSARSADestroyMixin:
    """Destroy operators of :class:`ALNSSARSASolver`."""

    # Attribute types from ALNSSARSASolver.__init__. Annotation only.
    # Unstring methods are owned by ALNSSARSARepairMixin (earlier in the MRO).
    dist_matrix: np.ndarray
    wastes: Dict[int, float]
    R: float
    C: float
    nodes: List[int]
    params: Any
    profit_aware_operators: bool
    random: Random

    if TYPE_CHECKING:

        def _unstring_type_i(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]: ...

        def _unstring_type_ii(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]: ...

        def _unstring_type_iii(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]: ...

        def _unstring_type_iv(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]: ...

    def _init_destroy_operators(self):
        """Initialize all destroy operators, including standard and unstringing ones."""

        self.destroy_ops = [
            # Standard operators
            self._destroy_random,
            self._destroy_worst,
            self._destroy_cluster,
            # Advanced operators
            self._destroy_shaw,
            self._destroy_string,
            # Unstringing operators
            self._unstring_type_i,
            self._unstring_type_ii,
            self._unstring_type_iii,
            self._unstring_type_iv,
        ]

        self.destroy_names = [
            "Random",
            "Worst",
            "Cluster",
            "Shaw",
            "String",
            "Unstring-I",
            "Unstring-II",
            "Unstring-III",
            "Unstring-IV",
        ]

    # ===== Destroy Operators =====

    def _destroy_random(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Random removal.

        Args:
            routes: Current set of routes.
            n: Number of nodes to remove.

        Returns:
            Tuple of (new_routes, removed_nodes).
        """

        return random_removal_op(routes, n, rng=self.random)

    def _destroy_worst(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Worst removal (highest cost nodes).

        Args:
            routes: Current set of routes.
            n: Number of nodes to remove.

        Returns:
            Tuple of (new_routes, removed_nodes).
        """

        if self.profit_aware_operators:
            return worst_profit_removal_op(routes, n, self.dist_matrix, self.wastes, self.R, self.C)
        return worst_removal_op(routes, n, self.dist_matrix)

    def _destroy_cluster(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Cluster removal (spatially related nodes).

        Args:
            routes: Current set of routes.
            n: Number of nodes to remove.

        Returns:
            Tuple of (new_routes, removed_nodes).
        """

        return cluster_removal_op(routes, n, self.dist_matrix, self.nodes, rng=self.random)

    def _destroy_shaw(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """Shaw removal (multi-criteria relatedness).

        Args:
            routes: Current set of routes.
            n: Number of nodes to remove.

        Returns:
            Tuple of (new_routes, removed_nodes).
        """

        if self.profit_aware_operators:
            return shaw_profit_removal_op(routes, n, self.dist_matrix, self.wastes, self.R, self.C, rng=self.random)
        return shaw_removal_op(
            routes,
            n,
            self.dist_matrix,
            wastes=self.wastes,
            rng=self.random,
        )

    def _destroy_string(self, routes: List[List[int]], n: int) -> Tuple[List[List[int]], List[int]]:
        """String removal (contiguous segments).

        Args:
            routes: Current set of routes.
            n: Number of nodes to remove.

        Returns:
            Tuple of (new_routes, removed_nodes).
        """

        return string_removal_op(routes, n, self.dist_matrix, rng=self.random)

    def _calc_removal_size(self, routes: List[List[int]]) -> int:
        """Calculate number of nodes to remove based on current solution size.

        Args:
            routes: Current set of routes.

        Returns:
            Random number of nodes to remove.
        """

        current_n_nodes = sum(len(r) for r in routes)

        if current_n_nodes == 0:
            return 0

        # Scale removal percentage with minimum of 2 nodes
        lower_bound = min(current_n_nodes, 2)
        max_pct_remove = int(current_n_nodes * self.params.max_removal_pct)
        upper_bound = max(lower_bound + 1, max_pct_remove)
        upper_bound = min(upper_bound, current_n_nodes)

        return self.random.randint(lower_bound, upper_bound)

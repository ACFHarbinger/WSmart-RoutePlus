"""Destroy/repair operators of the LKH-3 ALNS.

The operator selection, perturbation, history and elite-pool methods moved to
:mod:`alns_selection` and are mixed back in via ``LKH3ALNSSelectionMixin``;
this module keeps the destroy/repair operator wrappers.

Attributes:
    LKH3ALNSOperatorsMixin: Operator selection and wrappers, perturbation, history and elite pool of ``LKH3_ALNS``.

Example:
    >>> class LKH3_ALNS(LKH3ALNSOperatorsMixin): ...
"""

from __future__ import annotations

from typing import List, Tuple

from logic.src.policies.helpers.operators.destroy_ruin.historical import (
    historical_profit_removal,
    historical_removal,
)
from logic.src.policies.helpers.operators.destroy_ruin.neighbor import (
    neighbor_profit_removal,
    neighbor_removal,
)
from logic.src.policies.helpers.operators.destroy_ruin.sector import (
    sector_profit_removal,
    sector_removal,
)
from logic.src.policies.helpers.operators.recreate_repair.deep import (
    deep_insertion,
    deep_profit_insertion,
)
from logic.src.policies.helpers.operators.recreate_repair.nearest import (
    nearest_insertion,
    nearest_profit_insertion,
)
from logic.src.policies.helpers.operators.recreate_repair.savings import (
    savings_insertion,
    savings_profit_insertion,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.alns_selection import (
    LKH3ALNSSelectionMixin,
)


class LKH3ALNSOperatorsMixin(LKH3ALNSSelectionMixin):
    """Operator selection and wrappers, perturbation, history and elite pool of :class:`LKH3_ALNS`."""

    def _wrap_historical_removal(self, routes: List[List[int]], n_remove: int) -> Tuple[List[List[int]], List[int]]:
        """Wrap historical_removal with instance state.

        Args:
            routes: Current solution routes.
            n_remove: Number of nodes to remove.

        Returns:
            Tuple[List[List[int]], List[int]]: (modified_routes, removed_nodes).
        """
        return historical_removal(routes, n_remove, self.history, rng=self.rng, noise=0.1)

    def _wrap_neighbor_removal(self, routes: List[List[int]], n_remove: int) -> Tuple[List[List[int]], List[int]]:
        """Wrap neighbor_removal with instance state.

        Args:
            routes: Current solution routes.
            n_remove: Number of nodes to remove.

        Returns:
            Tuple[List[List[int]], List[int]]: (modified_routes, removed_nodes).
        """
        return neighbor_removal(routes, n_remove, self.distance_matrix, rng=self.rng)

    def _wrap_sector_removal(self, routes: List[List[int]], n_remove: int) -> Tuple[List[List[int]], List[int]]:
        """Wrap sector_removal, falling back to neighbor_removal if no coords.

        Args:
            routes: Current solution routes.
            n_remove: Number of nodes to remove.

        Returns:
            Tuple[List[List[int]], List[int]]: (modified_routes, removed_nodes).
        """
        if self.coords is not None:
            return sector_removal(routes, n_remove, self.coords, depot=(0.0, 0.0), rng=self.rng)
        else:
            return self._wrap_neighbor_removal(routes, n_remove)

    def _wrap_savings_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]:
        """Wrap savings_insertion with instance state.

        Args:
            routes: Current solution routes (after destroy).
            removed_nodes: Nodes removed by the destroy operator.

        Returns:
            List[List[int]]: Repaired routes with removed nodes reinserted.
        """
        m_nodes = list(self.mandatory_nodes) if self.mandatory_nodes else []
        if not self.profit_aware_operators:
            m_nodes = list(set(m_nodes) | set(removed_nodes))

        return savings_insertion(
            routes,
            removed_nodes,
            self.distance_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=m_nodes,
            expand_pool=True,
        )

    def _wrap_deep_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]:
        """Wrap deep_insertion with instance state.

        Args:
            routes: Current solution routes (after destroy).
            removed_nodes: Nodes removed by the destroy operator.

        Returns:
            List[List[int]]: Repaired routes with removed nodes reinserted.
        """
        m_nodes = list(self.mandatory_nodes) if self.mandatory_nodes else []
        if not self.profit_aware_operators:
            m_nodes = list(set(m_nodes) | set(removed_nodes))

        return deep_insertion(
            routes,
            removed_nodes,
            self.distance_matrix,
            self.wastes,
            self.capacity,
            alpha=0.3,
            mandatory_nodes=m_nodes,
            expand_pool=True,
        )

    def _wrap_nearest_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]:
        """Wrap nearest_insertion with instance state.

        Args:
            routes: Current solution routes (after destroy).
            removed_nodes: Nodes removed by the destroy operator.

        Returns:
            List[List[int]]: Repaired routes with removed nodes reinserted.
        """
        m_nodes = list(self.mandatory_nodes) if self.mandatory_nodes else []
        if not self.profit_aware_operators:
            m_nodes = list(set(m_nodes) | set(removed_nodes))

        return nearest_insertion(
            routes,
            removed_nodes,
            self.distance_matrix,
            self.wastes,
            self.capacity,
            mandatory_nodes=m_nodes,
            expand_pool=True,
        )

    def _wrap_historical_profit_removal(
        self, routes: List[List[int]], n_remove: int
    ) -> Tuple[List[List[int]], List[int]]:
        """Wrap historical_profit_removal with instance state.

        Args:
            routes: Current solution routes.
            n_remove: Number of nodes to remove.

        Returns:
            Tuple[List[List[int]], List[int]]: (modified_routes, removed_nodes).
        """
        return historical_profit_removal(
            routes,
            n_remove,
            self.history,
            self.distance_matrix,
            self.wastes,
            self.revenue,
            self.cost_unit,
            alpha=0.5,
            rng=self.rng,
            noise=0.1,
        )

    def _wrap_neighbor_profit_removal(
        self, routes: List[List[int]], n_remove: int
    ) -> Tuple[List[List[int]], List[int]]:
        """Wrap neighbor_profit_removal with instance state.

        Args:
            routes: Current solution routes.
            n_remove: Number of nodes to remove.

        Returns:
            Tuple[List[List[int]], List[int]]: (modified_routes, removed_nodes).
        """
        return neighbor_profit_removal(
            routes,
            n_remove,
            self.distance_matrix,
            self.wastes,
            self.revenue,
            self.cost_unit,
            rng=self.rng,
        )

    def _wrap_sector_profit_removal(self, routes: List[List[int]], n_remove: int) -> Tuple[List[List[int]], List[int]]:
        """Wrap sector_profit_removal, falling back to neighbor_profit_removal if no coords.

        Args:
            routes: Current solution routes.
            n_remove: Number of nodes to remove.

        Returns:
            Tuple[List[List[int]], List[int]]: (modified_routes, removed_nodes).
        """
        if self.coords is not None:
            return sector_profit_removal(
                routes,
                n_remove,
                self.coords,
                self.distance_matrix,
                self.wastes,
                self.revenue,
                self.cost_unit,
                depot=(0.0, 0.0),
                rng=self.rng,
            )
        else:
            return self._wrap_neighbor_profit_removal(routes, n_remove)

    def _wrap_savings_profit_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]:
        """Wrap savings_profit_insertion with expand_pool=True.

        Args:
            routes: Current solution routes (after destroy).
            removed_nodes: Nodes removed by the destroy operator.

        Returns:
            List[List[int]]: Repaired routes with profit-aware node reinsertion.
        """
        return savings_profit_insertion(
            routes,
            removed_nodes,
            self.distance_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=list(self.mandatory_nodes),
            expand_pool=True,  # Consider unvisited nodes for VRPP
        )

    def _wrap_deep_profit_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]:
        """Wrap deep_profit_insertion with expand_pool=True.

        Args:
            routes: Current solution routes (after destroy).
            removed_nodes: Nodes removed by the destroy operator.

        Returns:
            List[List[int]]: Repaired routes with profit-aware node reinsertion.
        """
        return deep_profit_insertion(
            routes,
            removed_nodes,
            self.distance_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=list(self.mandatory_nodes),
            expand_pool=True,
        )

    def _wrap_nearest_profit_insertion(self, routes: List[List[int]], removed_nodes: List[int]) -> List[List[int]]:
        """Wrap nearest_profit_insertion with expand_pool=True.

        Args:
            routes: Current solution routes (after destroy).
            removed_nodes: Nodes removed by the destroy operator.

        Returns:
            List[List[int]]: Repaired routes with profit-aware node reinsertion.
        """
        return nearest_profit_insertion(
            routes,
            removed_nodes,
            self.distance_matrix,
            self.wastes,
            self.capacity,
            R=self.R,
            C=self.C,
            mandatory_nodes=list(self.mandatory_nodes),
            expand_pool=True,  # Consider unvisited nodes for VRPP
        )

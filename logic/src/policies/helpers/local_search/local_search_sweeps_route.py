"""Route-operator sweeps of :class:`SweepLocalSearch`.

:class:`SweepRouteMixin` holds the first-improvement sweeps for 2-opt*, SWAP*, 2-opt,
3-opt, 4-opt, three-permutation, 3-opt* and 4-opt* over the current routes of
:class:`~logic.src.policies.helpers.local_search.local_search_sweeps.SweepLocalSearch`.
Moved unchanged from ``local_search_sweeps.py``.
"""

from logic.src.policies.helpers.operators.inter_route_local_search import (
    move_2opt_star,
    move_swap_star,
)
from logic.src.policies.helpers.operators.intra_route_local_search import (
    move_2opt_intra,
    move_3opt_intra,
    move_kopt_intra,
)


class SweepRouteMixin:
    """Route-operator sweeps of :class:`SweepLocalSearch` (moved unchanged from ``local_search_sweeps.py``)."""

    # ===== Route Operators =====

    def two_opt_star(self) -> bool:
        """2-opt* inter-route operator using imported implementation.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 2:
            return False

        for r_u in range(len(self.routes)):
            for r_v in range(r_u + 1, len(self.routes)):
                route_u = self.routes[r_u]
                route_v = self.routes[r_v]

                if not route_u or not route_v:
                    continue

                for p_u in range(len(route_u)):
                    u = route_u[p_u]
                    for p_v in range(len(route_v)):
                        v = route_v[p_v]
                        if move_2opt_star(self, u, v, r_u, p_u, r_v, p_v):
                            return True

        return False

    def swap_star(self) -> bool:
        """SWAP* inter-route operator using imported implementation.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 2:
            return False

        for r_u in range(len(self.routes)):
            for r_v in range(r_u + 1, len(self.routes)):
                route_u = self.routes[r_u]
                route_v = self.routes[r_v]

                if not route_u or not route_v:
                    continue

                for p_u in range(len(route_u)):
                    u = route_u[p_u]
                    for p_v in range(len(route_v)):
                        v = route_v[p_v]
                        if move_swap_star(self, u, v, r_u, p_u, r_v, p_v):
                            return True

        return False

    def two_opt_intra(self) -> bool:
        """2-opt intra-route: reverse segments within routes.

        Position -1 stands for the depot, so segments that start at the route's
        first bin (including the whole route) can be reversed too; on directed
        distances that changes the cost.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        for r_u in range(len(self.routes)):
            route = self.routes[r_u]
            if len(route) < 2:
                continue

            for p_u in range(-1, len(route) - 1):
                u = route[p_u] if p_u >= 0 else 0
                for p_v in range(p_u + 2, len(route)):
                    v = route[p_v]
                    if move_2opt_intra(self, u, v, r_u, p_u, r_u, p_v):
                        return True
        return False

    def three_opt_intra(self) -> bool:
        """3-opt intra-route: reconnect three segments within routes.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        for r_u in range(len(self.routes)):
            route = self.routes[r_u]
            if len(route) < 4:
                continue

            for p_u in range(len(route) - 2):
                u = route[p_u]
                for p_v in range(p_u + 2, len(route)):
                    v = route[p_v]
                    if move_3opt_intra(self, u, v, r_u, p_u, r_u, p_v, self.random):
                        return True
        return False

    def four_opt_intra(self) -> bool:
        """4-opt intra-route: permute four segments within routes.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        for r_u in range(len(self.routes)):
            route = self.routes[r_u]
            if len(route) < 5:
                continue

            for p_u in range(len(route) - 2):
                u = route[p_u]
                for p_v in range(p_u + 2, len(route)):
                    v = route[p_v]
                    if move_kopt_intra(self, u, v, r_u, p_u, r_u, p_v, k=4, rng=self.random):
                        return True
        return False

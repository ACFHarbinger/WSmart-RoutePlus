"""Exchange-operator sweeps of :class:`SweepLocalSearch`.

:class:`SweepExchangeMixin` holds the first-improvement sweeps for Or-opt, cross
exchange, improved cross exchange, lambda interchange, ejection chains, cyclic transfer
and exchange chains over the current routes of
:class:`~logic.src.policies.helpers.local_search.local_search_sweeps.SweepLocalSearch`.
Moved unchanged from ``local_search_sweeps.py``.
"""

from logic.src.policies.helpers.operators.inter_route_local_search import (
    cross_exchange,
    cyclic_transfer,
    ejection_chain,
    exchange_2_0,
    exchange_2_1,
    improved_cross_exchange,
    lambda_interchange,
)
from logic.src.policies.helpers.operators.intra_route_local_search import (
    move_or_opt,
)


class SweepExchangeMixin:
    """Exchange-operator sweeps of :class:`SweepLocalSearch` (moved unchanged from ``local_search_sweeps.py``)."""

    # ===== Exchange Operators =====

    def or_opt(self, chain_len: int = 2) -> bool:
        """
        OR-opt: relocate chain of nodes using the imported operator.

        Tries all possible OR-opt moves and applies the first improving one.

        Args:
            chain_len: Length of the chain to relocate. Defaults to 2.

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if not self.routes:
            return False

        for r_idx, route in enumerate(self.routes):
            if len(route) < chain_len:
                continue

            for pos in range(len(route)):
                if pos + chain_len > len(route):
                    continue

                if move_or_opt(self, r_idx, pos, chain_len):
                    return True

        return False

    def cross_exchange_op(self, max_seg_len: int = 2) -> bool:
        """
        Cross-exchange: swap segments between routes.

        Systematically tries cross-exchange moves between route pairs.

        Args:
            max_seg_len: Maximum length of segments to swap. Defaults to 2.

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 2:
            return False

        for r_a in range(len(self.routes)):
            for r_b in range(r_a + 1, len(self.routes)):
                for seg_a_len in range(max_seg_len + 1):
                    for seg_b_len in range(max_seg_len + 1):
                        if seg_a_len == 0 and seg_b_len == 0:
                            continue

                        for seg_a_start in range(len(self.routes[r_a]) - seg_a_len + 1):
                            for seg_b_start in range(len(self.routes[r_b]) - seg_b_len + 1):
                                if cross_exchange(self, r_a, seg_a_start, seg_a_len, r_b, seg_b_start, seg_b_len):
                                    return True
        return False

    def improved_cross_exchange_op(self, max_seg_len: int = 2) -> bool:
        """
        Improved Cross-exchange: swap segments between routes with optional internal reversal.

        Args:
            max_seg_len: Maximum length of segments to swap. Defaults to 2.

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 2:
            return False

        for r_a in range(len(self.routes)):
            for r_b in range(r_a + 1, len(self.routes)):
                for seg_a_len in range(max_seg_len + 1):
                    for seg_b_len in range(max_seg_len + 1):
                        if seg_a_len == 0 and seg_b_len == 0:
                            continue

                        for seg_a_start in range(len(self.routes[r_a]) - seg_a_len + 1):
                            for seg_b_start in range(len(self.routes[r_b]) - seg_b_len + 1):
                                if improved_cross_exchange(
                                    self, r_a, seg_a_start, seg_a_len, r_b, seg_b_start, seg_b_len
                                ):
                                    return True
        return False

    def lambda_interchange_op(self, lambda_max: int = 2) -> bool:
        """
        λ-interchange: generalized cross-exchange with segment lengths up to λ.

        Args:
            lambda_max: Maximum length of segments to swap. Defaults to 2.

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        return lambda_interchange(self, lambda_max)

    def ejection_chain_op(self, max_depth: int = 3) -> bool:
        """
        Ejection chain: attempt to empty routes for fleet minimization.

        Args:
            max_depth: Maximum depth of the ejection chain. Defaults to 3.

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 2:
            return False

        # Try to eject from smallest routes first
        route_sizes = [(i, len(r)) for i, r in enumerate(self.routes) if r]
        route_sizes.sort(key=lambda x: x[1])

        # Try ejection on up to 3 smallest routes
        return any(ejection_chain(self, r_idx, max_depth) for r_idx, _ in route_sizes[:3])

    def cyclic_transfer_op(self) -> bool:
        """
        Cyclic transfer: 3-way cyclical node exchange.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 3:
            return False

        # Attempt to find an improving 3-way cycle
        for r_a in range(len(self.routes)):
            if not self.routes[r_a]:
                continue
            for r_b in range(r_a + 1, len(self.routes)):
                if not self.routes[r_b]:
                    continue
                for r_c in range(r_b + 1, len(self.routes)):
                    if not self.routes[r_c]:
                        continue

                    for p_a in range(len(self.routes[r_a])):
                        for p_b in range(len(self.routes[r_b])):
                            for p_c in range(len(self.routes[r_c])):
                                if cyclic_transfer(self, [(r_a, p_a), (r_b, p_b), (r_c, p_c)]):
                                    return True
        return False

    def exchange_chain_op(self) -> bool:
        """
        Exchange chain: evaluate block transfers composed of exchange_2_0 and exchange_2_1.

        Args:
            None

        Returns:
            bool: True if an improving move was found and applied, False otherwise.
        """
        if len(self.routes) < 2:
            return False

        for r_a in range(len(self.routes)):
            for r_b in range(len(self.routes)):
                if r_a == r_b:
                    continue
                # Try exchange (2, 0)
                for p_a in range(len(self.routes[r_a]) - 1):
                    for p_b in range(len(self.routes[r_b]) + 1):
                        if exchange_2_0(self, r_a, p_a, r_b, p_b):
                            return True

                # Try exchange (2, 1) mutually
                for p_a in range(len(self.routes[r_a]) - 1):
                    for p_b in range(len(self.routes[r_b])):
                        if exchange_2_1(self, r_a, p_a, r_b, p_b):
                            return True
        return False

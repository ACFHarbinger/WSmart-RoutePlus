"""Exchange-operator sweeps of :class:`SweepLocalSearch`.

:class:`SweepExchangeMixin` holds the first-improvement sweeps for Or-opt, cross
exchange, improved cross exchange, lambda interchange, ejection chains, cyclic transfer
and exchange chains over the current routes of
:class:`~logic.src.policies.helpers.local_search.local_search_sweeps.SweepLocalSearch`.
Moved unchanged from ``local_search_sweeps.py``.
"""

from logic.src.policies.helpers.operators.inter_route_local_search import (
    cross_exchange,
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

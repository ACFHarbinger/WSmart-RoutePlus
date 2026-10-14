r"""Shared route pool data structures for the TCF → ALNS → BPC → SP pipeline.

Every stage (TCF, ALNS, BPC) produces ``VRPPRoute`` objects that are appended
to a global ``RoutePool``.  The SP-merge stage then reads from this pool.

Attributes:
    VRPPRoute:  Immutable route column with profit/cost metadata.
    RoutePool:  Thread-safe (append-only) container for de-duplicated routes.

Example:
    >>> pool = RoutePool()
    >>> pool.add(VRPPRoute(nodes=[1, 2], profit=42.0, revenue=50.0,
    ...                    cost=8.0, load=70.0, source="tcf"))
    >>> print(len(pool))  # 1
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterator, List, Optional


@dataclass
class VRPPRoute:
    """A single feasible route (column) in the master route pool.

    Attributes:
        nodes:   Ordered customer-only node sequence (local 1-based indices;
                 depot = 0 is *excluded*).
        profit:  Net profit = revenue - travel_cost.
        revenue: Gross revenue from collected waste.
        cost:    Travel cost (distance × C).
        load:    Total waste collected (sum of wastes for visited nodes).
        source:  Stage that generated this route ('tcf', 'alns', 'bpc').
    """

    nodes: List[int]
    profit: float
    revenue: float
    cost: float
    load: float
    source: str = "unknown"

    # ------------------------------------------------------------------
    # Canonical identity — two routes with the same *ordered* sequence
    # are identical.  For deduplication by node-set only (order-agnostic),
    # use RoutePool which indexes by frozenset.
    # ------------------------------------------------------------------

    def canonical_key(self) -> FrozenSet[int]:
        """Return the canonical (order-agnostic) identity key.

        Two routes visiting the same customers in any order share a key.
        Within the pool, the higher-profit route is retained on collision.

        Args:
            None

        Returns:
            Canonical key.
        """
        return frozenset(self.nodes)

    def __repr__(self) -> str:
        """Return a string representation of the route.

        Args:
            None

        Returns:
            String representation of the route.
        """
        return (
            f"VRPPRoute(nodes={self.nodes!r}, profit={self.profit:.4f}, load={self.load:.2f}, source={self.source!r})"
        )


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class RoutePool:
    """Thread-safe append-only container of de-duplicated VRPPRoute objects.

    De-duplication is by ``frozenset(route.nodes)`` — i.e., two routes that
    visit the same customers in different orders are considered equivalent, and
    only the higher-profit representative is kept.  This avoids bloating the
    SP-merge MIP with dominated columns.

    Attributes:
        _lock:   Mutex for thread-safe additions.
        _pool:   Internal dict mapping canonical key → best route.
    """

    def __init__(self) -> None:
        """Initialize the route pool.

        Returns:
            None
        """
        self._lock: threading.Lock = threading.Lock()
        self._pool: Dict[FrozenSet[int], VRPPRoute] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, route: VRPPRoute) -> bool:
        """Add a route to the pool; retain only the higher-profit representative.

        Args:
            route: Route to add.

        Returns:
            True if the pool was modified (new entry or replacement), False
            if an equal-or-better route already exists.
        """
        if not route.nodes:
            return False
        key = route.canonical_key()
        with self._lock:
            existing = self._pool.get(key)
            if existing is None or route.profit > existing.profit + 1e-9:
                self._pool[key] = route
                return True
        return False

    def add_all(self, routes: List[VRPPRoute]) -> int:
        """Bulk-add a list of routes.

        Args:
            routes: Routes to add.

        Returns:
            Number of routes that modified the pool.
        """
        return sum(1 for r in routes if self.add(r))

    def filter_feasible(self, capacity: float) -> None:
        """Remove routes that violate the capacity constraint in-place.

        Args:
            capacity: Vehicle capacity Q.

        Returns:
            None
        """
        with self._lock:
            self._pool = {k: r for k, r in self._pool.items() if r.load <= capacity + 1e-6}

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def routes(self) -> List[VRPPRoute]:
        """Return a snapshot of all routes currently in the pool.

        Returns:
            List of VRPPRoute objects (copy of pool values).
        """
        with self._lock:
            return list(self._pool.values())

    def best(self) -> Optional[VRPPRoute]:
        """Return the single route with the highest profit, or None.

        Returns:
            Highest-profit VRPPRoute, or None if the pool is empty.
        """
        routes = self.routes()
        return max(routes, key=lambda r: r.profit) if routes else None

    def __len__(self) -> int:
        """Return the number of routes in the pool.

        Returns:
            Number of routes in the pool.
        """
        with self._lock:
            return len(self._pool)

    def __iter__(self) -> Iterator[VRPPRoute]:
        """Return an iterator over the routes in the pool.

        Returns:
            Iterator over the routes in the pool.
        """
        return iter(self.routes())

    def __repr__(self) -> str:
        """Return a string representation of the route pool.

        Returns:
            String representation of the route pool.
        """
        return f"RoutePool(n={len(self)})"

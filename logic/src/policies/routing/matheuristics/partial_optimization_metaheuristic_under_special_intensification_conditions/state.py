"""
This file contains the POPMUSICState class, which is
used to store the state of the POPMUSIC algorithm.


Attributes:
    POPMUSICState: Class for storing the state of the POPMUSIC algorithm.

Example:
    >>> from logic.src.policies.routing.matheuristics
        .partial_optimization_metaheuristic_under_special_intensification_conditions.state import POPMUSICState
    >>> state = POPMUSICState(n_nodes=10, coord_array=np.random.rand(11, 2))
    >>> print(state)
    POPMUSICState(n_nodes=10, coord_array=array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0],
       [1.1, 1.2], [1.3, 1.4], [1.5, 1.6], [1.7, 1.8], [1.9, 2.0], [2.1, 2.2]]))  # noqa: E501

    >>> state.insert_route([0, 1, 2], np.array([0.1, 0.2]))
    >>> state.insert_singleton(3)
    >>> state.delete_slot(0)
    >>> state.update_coords(1, np.array([0.3, 0.4]))
    >>> print(state.customer_nodes(1))
    [1, 2]
    >>> print(state.active_route_slots)
    {1}
    >>> print(state.active_singleton_slots)
    {3}
    >>> print(state.free_slots)
    [9, 8, 7, 6, 5, 4, 2, 0]
"""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np

# ---------------------------------------------------------------------------
# POPMUSICState — unified O(1) insert/delete state container (Issue 4)
# ---------------------------------------------------------------------------


class POPMUSICState:
    """
    Stable-slot state container for the POPMUSIC main loop.

    Maintains a pre-allocated centroid buffer of size MAX_PARTS = n so that
    slot indices (= route identity) never shift under insert/delete. Eliminated
    slots are reclaimed via a free-slot stack in O(1), making compaction
    unnecessary and keeping the proximity network G permanently valid.

    Time complexity per operation:
        insert_part  : O(1) amortised
        delete_part  : O(1)
        update_coords: O(1)
        active_count : O(1)

    Space complexity: O(n) for coords buffer + O(p) for G and route dict.

    Attributes:
        coords: Pre-allocated stable centroid buffer — slot identity is permanent.
        route_nodes: Dict mapping slot to list of global node indices (incl. depot sentinels).
        singleton_slots: Slots that represent unvisited singleton nodes (Issue 3).
        active: Currently live slot indices.
        free_slots: Stack of reclaimed positions — O(1) pop/push.
        G: Adjacency list on slot indices, sorted by d_prox ascending (Issue 2).
        _coord_array: Raw coordinate lookup for all nodes (depot + customers).
    """

    def __init__(self, n_nodes: int, coord_array: np.ndarray) -> None:  # noqa: D107
        """
        Initialize the POPMUSICState.

        Args:
            n_nodes: Total number of customer nodes (excluding depot). Used to
                pre-allocate the centroid buffer. MAX_PARTS = n_nodes (worst case:
                every node is a singleton unvisited part).
            coord_array: Float64 array of shape (n_nodes+1, 2) with [Lat, Lng]
                per node index (including depot at 0).
        """
        max_parts = n_nodes + 1  # +1 for safety; depot excluded at runtime
        # Pre-allocated stable centroid buffer — slot identity is permanent.
        self.coords: np.ndarray = np.empty((max_parts, 2), dtype=np.float64)
        # route_nodes[slot] = list of global node indices (incl. depot sentinels).
        # None means the slot is free.
        self.route_nodes: Dict[int, List[int]] = {}
        # singleton_slots: slots that represent unvisited singleton nodes (Issue 3).
        self.singleton_slots: Set[int] = set()
        # active: currently live slot indices.
        self.active: Set[int] = set()
        # free_slots: stack of reclaimed positions — O(1) pop/push.
        self.free_slots: List[int] = list(range(max_parts - 1, -1, -1))
        # G: adjacency list on slot indices, sorted by d_prox ascending (Issue 2).
        self.G: Dict[int, List[int]] = {}
        # raw coordinate lookup for all nodes (depot + customers).
        self._coord_array: np.ndarray = coord_array

    # ------------------------------------------------------------------
    # Slot lifecycle
    # ------------------------------------------------------------------

    def alloc_slot(self) -> int:
        """Pop a free slot from the stack.

        Returns:
            int: Allocated slot index.

        Raises:
            RuntimeError: If no free slots are available.
        """
        if not self.free_slots:
            raise RuntimeError("POPMUSICState: free_slots exhausted. MAX_PARTS underestimated.")
        return self.free_slots.pop()

    def insert_route(self, nodes: List[int], centroid: np.ndarray) -> int:
        """
        Insert a new route part and return its slot index. O(1).

        Args:
            nodes: Global node indices comprising the route (incl. depot sentinels).
            centroid: Geographic centroid of the route, shape (2,).

        Returns:
            Allocated slot index.
        """
        slot = self.alloc_slot()
        self.route_nodes[slot] = nodes
        self.coords[slot] = centroid
        self.active.add(slot)
        return slot

    def insert_singleton(self, node: int) -> int:
        """
        Insert an unvisited node as a singleton part. O(1).

        The centroid is simply the node's own coordinate (Issue 3).

        Args:
            node: Global node index of the unvisited customer.

        Returns:
            Allocated slot index.
        """
        slot = self.alloc_slot()
        self.route_nodes[slot] = [node]
        self.coords[slot] = self._coord_array[node]
        self.active.add(slot)
        self.singleton_slots.add(slot)
        return slot

    def delete_slot(self, slot: int) -> None:
        """
        Reclaim a slot. O(1). G entries referencing this slot become stale
        (tolerated per the static-topology assumption — see Issue 2 rationale).

        Args:
            slot: Slot index to reclaim.

        Returns:
            None
        """
        self.active.discard(slot)
        self.singleton_slots.discard(slot)
        self.route_nodes.pop(slot, None)
        self.G.pop(slot, None)
        self.free_slots.append(slot)

    def update_coords(self, slot: int, centroid: np.ndarray) -> None:
        """Update centroid in-place.

        Args:
            slot: Slot index.
            centroid: New centroid for the given slot.

        Returns:
            None
        """
        self.coords[slot] = centroid

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def customer_nodes(self, slot: int) -> List[int]:
        """Return non-depot nodes for a slot.

        Args:
            slot: Slot index.

        Returns:
            List[int]: List of non-depot nodes for the given slot.
        """
        return [n for n in self.route_nodes.get(slot, []) if n != 0]

    @property
    def active_route_slots(self) -> Set[int]:
        """Live non-singleton slots.

        Returns:
            Set[int]: Set of active non-singleton slots.
        """
        return self.active - self.singleton_slots

    @property
    def active_singleton_slots(self) -> Set[int]:
        """Live singleton slots.

        Returns:
            Set[int]: Set of active singleton slots.
        """
        return self.active & self.singleton_slots

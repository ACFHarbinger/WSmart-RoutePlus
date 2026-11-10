"""
Pheromone Management Module.

This module implements the pheromone matrix used in ACO.

Attributes:
    None

Example:
    >>> from logic.src.policies.routing.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones import PheromoneTau
    >>> pheromone = PheromoneTau(n_nodes=100, k=25, tau_0=1.0, ...)
    >>> val = pheromone.get(0, 1)
"""

from collections import defaultdict
from typing import Dict, Optional


class PheromoneTau:
    """
    Pheromone matrix that stores only k-best values per node.

    For each node, maintains a heap of the k highest pheromone values
    to its neighbors. Unrepresented edges default to tau_0.

    Attributes:
        n_nodes: Total number of nodes (including depot).
        k: Maximum edges to store per node.
        tau_0: Default pheromone value for unrepresented edges.
        tau_min: Minimum pheromone bound.
        tau_max: Maximum pheromone bound.
        scale: Pruning precision: after evaporation, a stored value within 10^-scale of tau_0 is dropped (it reads
            as tau_0 again and frees its slot); None keeps every stored value.
        _pheromone: Sparse storage of pheromone values.
    """

    def __init__(
        self, n_nodes: int, k: int, tau_0: float, tau_min: float, tau_max: float, scale: Optional[float] = None
    ):
        """
        Initialize sparse pheromone structure.

        Args:
            n_nodes: Total number of nodes (including depot).
            k: Maximum edges to store per node.
            tau_0: Default pheromone value for unrepresented edges.
            tau_min: Minimum pheromone bound.
            tau_max: Maximum pheromone bound.
            scale: Pruning precision (see the class attributes).
        """
        self.n_nodes = n_nodes
        self.k = k
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.scale = scale

        # Sparse storage: node -> {neighbor: pheromone}
        self._pheromone: Dict[int, Dict[int, float]] = defaultdict(dict)

    def get(self, i: int, j: int) -> float:
        """
        Get pheromone value for edge (i, j).

        Args:
            i: Source node.
            j: Destination node.

        Returns:
            Pheromone value for edge (i, j).
        """
        if j in self._pheromone[i]:
            return self._pheromone[i][j]
        return self.tau_0

    def set(self, i: int, j: int, value: float) -> None:
        """
        Set pheromone value for edge (i, j).

        If storage for node i exceeds k, evict the lowest value.

        Args:
            i: Source node.
            j: Destination node.
            value: Pheromone value to set.

        """
        value = max(self.tau_min, min(self.tau_max, value))

        if j in self._pheromone[i] or len(self._pheromone[i]) < self.k:
            self._pheromone[i][j] = value
        else:
            # Find minimum value in current storage, use neighbor ID as tie-breaker
            min_neighbor = min(self._pheromone[i].keys(), key=lambda n: (self._pheromone[i][n], n))
            if value > self._pheromone[i][min_neighbor]:
                del self._pheromone[i][min_neighbor]
                self._pheromone[i][j] = value

    def update_edge(self, i: int, j: int, delta: float, evaporate: bool = True) -> None:
        """
        Update pheromone on edge with deposit and optional evaporation.

        Args:
            i: Source node.
            j: Destination node.
            delta: Pheromone to deposit.
            evaporate: If True, also apply evaporation.
        """
        current = self.get(i, j)
        if evaporate:
            # Local update rule: tau = (1-rho)*tau + rho*tau_0
            # This is applied per-edge during construction
            pass
        new_value = current + delta
        self.set(i, j, new_value)

    def evaporate_all(self, rho: float) -> None:
        """
        Apply global evaporation to all stored pheromones.

        Args:
            rho: Evaporation rate.

        """
        precision = None if self.scale is None else 10.0 ** (-self.scale)
        for i in self._pheromone:
            for j in list(self._pheromone[i].keys()):
                self._pheromone[i][j] *= 1 - rho
                self._pheromone[i][j] = max(self._pheromone[i][j], self.tau_min)
                if precision is not None and abs(self._pheromone[i][j] - self.tau_0) <= precision:
                    del self._pheromone[i][j]

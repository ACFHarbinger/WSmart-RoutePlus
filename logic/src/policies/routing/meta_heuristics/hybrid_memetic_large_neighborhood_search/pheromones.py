r"""Sparse Pheromone Management Module.

This module implements the sparse pheromone matrix used in Memetic Ant Colony
Optimization (MACO) following the canonical MAX-MIN Ant System (MMAS) of
Stützle & Hoos (2000) together with the memory-efficient MMAS_exp variant
from Hale (2021).

Key differences from a naïve dense matrix implementation:

* **Dynamic bounds** – ``tau_max`` is recomputed whenever a new best solution
  cost is provided (``tau_max = 1 / (rho * C_bs)``), and ``tau_min`` is
  recomputed from ``tau_max`` and ``p_best`` using the formula from Stützle &
  Hoos (2000, Eq. 14).  Fixed override values can be supplied to disable
  dynamic computation.

* **Sparse storage with precision-based pruning** – Only edges that deviate
  from the current ``default_value`` (the background level shared by all
  non-reinforced edges) by more than ``10^{-scale}`` are stored explicitly.
  This is the MMAS_exp approach from Hale (2021) which avoids O(n²) memory.

* **Global evaporation** – A single call to ``evaporate_all`` simultaneously
  decays the ``default_value`` and all explicitly stored edges, then prunes
  edges that have converged back to the background level.

* **Reinitialization** – The ``reinitialize`` method resets all pheromones to
  a target level (typically ``tau_max``) and clears the sparse store, enabling
  MMAS-style stagnation restarts without constructing a new object.

Attributes:
    SparsePheromoneTau: Sparse pheromone matrix with dynamic MMAS bounds.

Example:
    >>> pheromone = SparsePheromoneTau(
    ...     n_nodes=100, tau_0=1.0, scale=5.0,
    ...     rho=0.1, n_nodes_formula=100,
    ...     tau_min_fixed=None, tau_max_fixed=None, p_best=0.05,
    ... )
    >>> val = pheromone.get(0, 1)

Reference:
    Stützle, T. & Hoos, H. H. (2000). MAX-MIN Ant System.
        Future Generation Computer Systems, 16(8), 889–914.
    Hale, D. (2021). Investigation of Ant Colony Optimization Implementation
        Strategies For Low-Memory Operating Environments.
        Section 4.2.3: MMAS_exp with scale-based precision pruning.
"""

from collections import defaultdict
from typing import Dict, Optional


class SparsePheromoneTau:
    """Sparse pheromone matrix with dynamic MMAS bounds and precision pruning.

    Dynamic bounds (Stützle & Hoos 2000)
    -------------------------------------
    ``tau_max`` is updated whenever ``update_bounds(C_bs)`` is called with a
    new best solution cost C_bs::

        tau_max = 1 / (rho * C_bs)

    ``tau_min`` is then derived from ``tau_max`` via::

        tau_min = tau_max * (1 - p_best^(1/n))^((avg_nn - 1) / 2)

    where ``n`` is the number of customer nodes and ``avg_nn`` is the average
    number of candidate neighbours per node (approximated as k_sparse/2 + 1).
    When ``tau_min_fixed`` or ``tau_max_fixed`` are provided, the corresponding
    dynamic formula is disabled and the fixed value is used instead.

    Sparse storage (Hale 2021)
    --------------------------
    ``default_value`` plays the role of the background pheromone for all edges
    that have never been reinforced (or whose reinforcement has decayed back to
    the background level).  Only edges with
    ``|tau(i,j) - default_value| > 10^{-scale}`` are stored explicitly.

    Attributes:
        n_nodes: Total number of nodes (including depot).
        rho: Evaporation rate – stored so bounds can be recomputed without
            requiring callers to pass it again.
        scale: Precision exponent for pruning threshold.
        p_best: Convergence probability for dynamic tau_min formula.
        avg_candidates: Average number of candidate neighbours (k_sparse/2 + 1).
        tau_min_fixed: Fixed tau_min override (None → dynamic).
        tau_max_fixed: Fixed tau_max override (None → dynamic).
        tau_min: Current lower pheromone bound.
        tau_max: Current upper pheromone bound.
        default_value: Background pheromone level (evaporates globally).
    """

    def __init__(
        self,
        n_nodes: int,
        tau_0: float,
        scale: float,
        rho: float,
        *,
        tau_min_fixed: Optional[float] = None,
        tau_max_fixed: Optional[float] = None,
        p_best: float = 0.05,
        avg_candidates: float = 8.0,
    ) -> None:
        """Initialise the sparse pheromone structure.

        Args:
            n_nodes: Total number of nodes (including depot).
            tau_0: Initial pheromone value; becomes the initial ``default_value``
                and is also used for the very first ``tau_max`` when no fixed
                value is given.
            scale: Precision parameter for pruning (10^{-scale} threshold).
            rho: Evaporation rate; stored for dynamic bound recomputation.
            tau_min_fixed: Optional fixed lower bound.  If supplied, the
                p_best-based formula is never evaluated.
            tau_max_fixed: Optional fixed upper bound.  If supplied,
                ``update_bounds`` becomes a no-op for tau_max.
            p_best: Probability of constructing the best solution at
                convergence (Stützle & Hoos 2000 recommend 0.05).
            avg_candidates: Average number of candidate neighbours per node,
                used in the tau_min formula.  Pass ``k_sparse / 2 + 1`` for a
                reasonable approximation.
        """
        self.n_nodes = n_nodes
        self.rho = rho
        self.scale = scale
        self.p_best = p_best
        self.avg_candidates = avg_candidates
        self.tau_min_fixed = tau_min_fixed
        self.tau_max_fixed = tau_max_fixed

        # Derive initial bounds.
        # If no fixed tau_max, the first real update comes from update_bounds().
        self.tau_max: float = tau_max_fixed if tau_max_fixed is not None else tau_0
        self.tau_min: float = tau_min_fixed if tau_min_fixed is not None else self._compute_tau_min()

        # Background pheromone level that evaporates globally.
        self.default_value: float = tau_0

        # Sparse storage: node → {neighbour: pheromone}.
        # Only stores edges that differ significantly from default_value.
        self._pheromone: Dict[int, Dict[int, float]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Dynamic bound management
    # ------------------------------------------------------------------

    def _compute_tau_min(self) -> float:
        """Compute tau_min from the current tau_max via Stützle & Hoos (2000).

        The formula is::

            tau_min = tau_max * (1 - p_best^(1/n)) / ((avg_candidates - 1) / 2)

        where n is the number of *customer* nodes (n_nodes - 1 for the depot).

        Returns:
            Computed tau_min, guaranteed to be positive.
        """
        n = max(self.n_nodes - 1, 1)
        # (1 − p_best^{1/n})
        decay = 1.0 - (self.p_best ** (1.0 / n))
        # avoid division by zero when avg_candidates == 1
        denom = max((self.avg_candidates - 1.0) / 2.0, 1e-9)
        tau_min = self.tau_max * decay / denom
        # Ensure tau_min < tau_max and > 0
        return max(1e-10, min(tau_min, self.tau_max * 0.5))

    def update_bounds(self, best_cost: float) -> None:
        """Recompute pheromone bounds from the latest best solution cost.

        This should be called once per iteration whenever ``best_cost``
        improves.  The updated bounds take effect on the *next* ``set`` or
        ``evaporate_all`` call.

        tau_max = 1 / (rho * C_bs)  [Stützle & Hoos 2000, Eq. 13]
        tau_min = f(tau_max, p_best)  [Stützle & Hoos 2000, Eq. 14]

        Args:
            best_cost: Cost of the best solution found so far (C_bs > 0).
        """
        if best_cost <= 0:
            return
        if self.tau_max_fixed is None:
            self.tau_max = 1.0 / (self.rho * best_cost)
        if self.tau_min_fixed is None:
            self.tau_min = self._compute_tau_min()

    # ------------------------------------------------------------------
    # Core pheromone access
    # ------------------------------------------------------------------

    def get(self, i: int, j: int) -> float:
        """Return the pheromone value for edge (i, j).

        Args:
            i: Source node index.
            j: Destination node index.

        Returns:
            Explicit stored value if present, otherwise ``default_value``.
        """
        nbrs = self._pheromone.get(i)
        if nbrs is not None and j in nbrs:
            return nbrs[j]
        return self.default_value

    def set(self, i: int, j: int, value: float) -> None:
        """Set the pheromone for edge (i, j), clamped to [tau_min, tau_max].

        Precision pruning does *not* occur here; it is deferred to
        ``evaporate_all`` to keep deposits cheap.

        Args:
            i: Source node index.
            j: Destination node index.
            value: Desired pheromone value (will be clamped).
        """
        value = max(self.tau_min, min(self.tau_max, value))
        self._pheromone[i][j] = value

    def deposit_edge(self, i: int, j: int, delta: float) -> None:
        """Add ``delta`` to the pheromone on edge (i, j).

        Args:
            i: Source node index.
            j: Destination node index.
            delta: Amount of pheromone to deposit (must be ≥ 0).
        """
        self.set(i, j, self.get(i, j) + delta)

    # ------------------------------------------------------------------
    # Global evaporation with precision pruning
    # ------------------------------------------------------------------

    def evaporate_all(self, rho: float) -> None:
        """Apply MMAS_exp global evaporation with precision-based pruning.

        Steps (Hale 2021, §4.2.3):

        1. Decay ``default_value`` by ``(1 − rho)`` and clamp to
           ``[tau_min, tau_max]``.
        2. Decay every explicitly stored edge by ``(1 − rho)`` and clamp.
        3. Delete edges that are within ``10^{-scale}`` of ``default_value``
           (they are effectively indistinguishable from the background).
        4. Delete empty neighbour dicts to reclaim memory.

        Args:
            rho: Evaporation rate (0 < rho < 1).
        """
        # Step 1: Evaporate and clamp the global background level.
        self.default_value = max(self.tau_min, min(self.tau_max, self.default_value * (1.0 - rho)))

        precision = 10.0 ** (-self.scale)

        # Steps 2–4: Evaporate explicit edges with pruning.
        for i in list(self._pheromone.keys()):
            nbrs = self._pheromone[i]
            for j in list(nbrs.keys()):
                val = max(self.tau_min, min(self.tau_max, nbrs[j] * (1.0 - rho)))
                if abs(val - self.default_value) <= precision:
                    del nbrs[j]
                else:
                    nbrs[j] = val
            if not nbrs:
                del self._pheromone[i]

    # ------------------------------------------------------------------
    # Stagnation restart
    # ------------------------------------------------------------------

    def reinitialize(self, tau_reset: Optional[float] = None) -> None:
        """Reset all pheromones to a uniform level and clear sparse storage.

        Called by the solver on stagnation detection.  After this call the
        matrix behaves as if newly constructed with ``tau_0 = tau_reset``.

        Args:
            tau_reset: Level to reset to.  Defaults to current ``tau_max``
                (the canonical MMAS restart level).
        """
        level = tau_reset if tau_reset is not None else self.tau_max
        level = max(self.tau_min, min(self.tau_max, level))
        self.default_value = level
        self._pheromone.clear()

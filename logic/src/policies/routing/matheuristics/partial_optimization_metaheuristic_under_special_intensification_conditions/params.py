"""
Configuration parameters for the POPMUSIC matheuristic.

Attributes:
    POPMUSICParams (dataclass): Configuration parameters for POPMUSIC.

Example:
    >>> params = POPMUSICParams.from_config(config)
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class POPMUSICParams:
    """
    Configuration parameters for POPMUSIC.

    Attributes:
        subproblem_size: Total number of parts per subproblem (the unique
            POPMUSIC parameter r). Controls the trade-off between subproblem
            quality and solve time.
        max_iterations: Maximum number of main-loop iterations
            (None = run until U is empty, i.e. full convergence).
        base_solver: Solver for subproblem optimisation
            ("fast_tsp", "hgs", "alns").
        base_solver_config: Configuration object/dict for base_solver.
        cluster_solver: Solver used for warm-starting initial clusters
            (only active when WARMSTART_INITIAL=True in solver.py).
        cluster_solver_config: Configuration for cluster_solver.
        initial_solver: Initial solution construction method.
            "pmedian"  — canonical O(n^{3/2}) two-metric p-median heuristic
                         (Alvim & Taillard 2013). Recommended for publication.
            "greedy"   — greedy insertion (legacy, O(n^2), no proximity network).
            "nearest_neighbor" — nearest-neighbour (legacy, O(n^2), no G).
        seed: Random seed for reproducibility.
        vrpp: If True, nodes may be left unvisited (VRPP mode). Unvisited
            nodes are represented as singleton parts in the proximity network.
        profit_aware_operators: If True, uses profit-biased assignment metric
            d_assign(i,c) = d(i,c)/d_max / (rev(i)/rev_max + ε) during
            p-median initialisation (Issue 1 refactor).
        k_prox: Retained for API backwards-compatibility. No longer used
            internally — the proximity network degree is derived from
            subproblem_size during initialisation.
        seed_strategy: Seed-part selection strategy for U.
            "lifo"   — stack (recommended by Alvim & Taillard 2013).
            "fifo"   — queue.
            "random" — random shuffle of initial U.
    """

    subproblem_size: int = 3
    max_iterations: Optional[int] = None
    base_solver: str = "fast_tsp"
    base_solver_config: Optional[Any] = None
    cluster_solver: str = "fast_tsp"
    cluster_solver_config: Optional[Any] = None
    initial_solver: str = "pmedian"
    seed: int = 42
    vrpp: bool = True
    profit_aware_operators: bool = False
    k_prox: int = 10
    seed_strategy: str = "lifo"

    @classmethod
    def from_config(cls, config: Any) -> POPMUSICParams:
        """Create POPMUSICParams from a configuration object or dictionary.

        Args:
            config (Any): Configuration object.

        Returns:
            POPMUSICParams: Instance of POPMUSICParams.
        """
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            subproblem_size=getattr(config, "subproblem_size", 3),
            max_iterations=getattr(config, "max_iterations", None),
            base_solver=getattr(config, "base_solver", "fast_tsp"),
            base_solver_config=getattr(config, "base_solver_config", None),
            cluster_solver=getattr(config, "cluster_solver", "fast_tsp"),
            cluster_solver_config=getattr(config, "cluster_solver_config", None),
            initial_solver=getattr(config, "initial_solver", "pmedian"),
            seed=getattr(config, "seed", 42),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            k_prox=getattr(config, "k_prox", 10),
            seed_strategy=getattr(config, "seed_strategy", "lifo"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert POPMUSICParams to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing POPMUSICParams.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

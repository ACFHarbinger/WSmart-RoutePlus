r"""ACO Parameters Module.

This module defines the configuration parameters for the K-Sparse Ant Colony
Optimization algorithm. It uses a dataclass to store and validate hyperparameters.

Attributes:
    KSACOParams: Parameters for K-Sparse Ant Colony Optimization (MMAS_exp variant).

Example:
    >>> from logic.src.policies.routing.meta_heuristics.ant_colony_optimization_k_sparse.params import KSACOParams
    >>> params = KSACOParams(n_ants=20, rho=0.1)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion

if TYPE_CHECKING:
    from logic.src.configs.policies import KSparseACOConfig


@dataclass
class KSACOParams:
    """
    Parameters for K-Sparse Ant Colony Optimization (MMAS_exp variant).

    This parameter set follows the experimental MAX-MIN Ant System (MMAS_exp)
    from Hale (2021), using scale-based precision pruning instead of fixed
    capacity pheromone storage.

    Attributes:
        n_ants: Number of ants per iteration.
        k_sparse: Size of candidate lists (k-nearest neighbors for each node).
        alpha: Pheromone importance exponent.
        beta: Heuristic (distance) importance exponent.
        rho: Pheromone evaporation rate (0 < rho < 1).
        scale: Precision parameter for pheromone pruning. Edges within 10^-scale
               of default_value are removed from explicit storage.
        tau_0: Initial pheromone level (if None, computed from nearest neighbor heuristic).
        tau_min: Minimum pheromone level (MMAS lower bound).
        tau_max: Maximum pheromone level (MMAS upper bound).
        max_iterations: Maximum number of iterations.
        time_limit: Maximum runtime in seconds.
        local_search: Whether to apply local search (2-opt) to solutions.
        local_search_iterations: Number of iterations for local search.
        elitist_weight: Weight for best-so-far solution in pheromone update.
        vrpp: Whether to use VRPP (Vehicle Routing Problem with Profits) mode.
        profit_aware_operators: Whether to use profit-aware feasibility checks.
        seed: Random seed for reproducibility.
        acceptance_criterion: Acceptance criterion for solutions.

    Reference:
        Hale, D. "Investigation of Ant Colony Optimization Implementation
        Strategies For Low-Memory Operating Environments", 2021.
    """

    n_ants: int = 20
    k_sparse: int = 15
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    scale: float = 5.0
    tau_0: Optional[float] = None
    tau_min: float = 0.001
    tau_max: float = 10.0
    max_iterations: int = 100
    time_limit: float = 30.0

    # Disabled by default to isolate the effect of memory limitations
    # purely on the stigmergic operations, per Hale (2021).
    local_search: bool = False
    local_search_iterations: int = 100

    elitist_weight: float = 1.0
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = 42
    acceptance_criterion: Optional[IAcceptanceCriterion] = None

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if not (0 < self.rho < 1):
            raise ValueError(f"Evaporation rate rho must be in (0, 1), got {self.rho}")
        if self.scale < 0:
            raise ValueError(f"Scale parameter must be non-negative, got {self.scale}")

    @classmethod
    def from_config(cls, config: Union[KSparseACOConfig, Dict[str, Any]]) -> KSACOParams:
        """Create KSACOParams from an KSparseACOConfig dataclass or dict.

        Performs explicit type casting for numeric fields to ensure consistency.

        Args:
            config: Configuration object or dictionary.

        Returns:
            KSACOParams instance.
        """
        if config is None:
            return cls()

        raw_data: Dict[str, Any] = {}
        if isinstance(config, dict):
            raw_data = config
        else:
            for f in dataclasses.fields(cls):
                if hasattr(config, f.name):
                    raw_data[f.name] = getattr(config, f.name)

        kwargs: Dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name == "acceptance_criterion":
                continue
            val = raw_data.get(f.name, getattr(cls, f.name, f.default))
            if val is not None:
                if f.type is float or f.type == "float":
                    val = float(val)
                elif f.type is int or f.type == "int":
                    val = int(val)
                elif f.type is bool or f.type == "bool":
                    val = val.lower() in ("true", "1", "yes") if isinstance(val, str) else bool(val)
            kwargs[f.name] = val

        params = cls(**kwargs)

        # Handle Acceptance Criterion Injection
        from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory

        acceptance_cfg = (
            raw_data.get("acceptance_criterion")
            if isinstance(config, dict)
            else getattr(config, "acceptance_criterion", None)
        )
        if acceptance_cfg:
            # acceptance_cfg might be a dict or a Config object
            if hasattr(acceptance_cfg, "method"):
                name = acceptance_cfg.method
                params_cfg = getattr(acceptance_cfg, "params", {})
            else:
                name = acceptance_cfg.get("method", "oi")
                params_cfg = acceptance_cfg.get("params", {})

            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=name,
                config=params_cfg,
            )
        else:
            # Default to oi (only improving) for standard ACO
            params.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

        return params

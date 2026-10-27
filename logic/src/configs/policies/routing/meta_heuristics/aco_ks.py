"""
KS-ACO (K-Sparse Ant Colony Optimization) configuration.

Attributes:
    KSparseACOConfig: Attributes for KS-ACO configuration.

Example:
    >>> from configs.policies.aco_ks import KSparseACOConfig
    >>> config = KSparseACOConfig()
    >>> config.n_ants
    20
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from logic.src.configs.policies.acceptance_criteria import AcceptanceConfig
from logic.src.configs.policies.packing.dispatch import PackingConfig
from logic.src.configs.policies.route_improvement import RouteImprovingConfig
from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion


@dataclass
class KSparseACOConfig:
    """Configuration for K-Sparse Ant Colony Optimization (KS-ACO).

    Attributes:
        n_ants: Number of ants in the colony.
        alpha: Pheromone importance factor.
        beta: Heuristic (distance) importance factor.
        rho: Pheromone evaporation rate.
        tau_0: Initial pheromone level.
        tau_min: Minimum pheromone bound.
        tau_max: Maximum pheromone bound.
        max_iterations: Maximum number of ACO iterations.
        scale: Scaling factor for heuristic information.
        time_limit: Maximum time in seconds for the solver.
        k_sparse: Number of nearest neighbors to consider.
        local_search: Whether to apply local search after construction.
        local_search_iterations: Number of local search iterations.
        elitist_weight: Weight for elitist pheromone update.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
        acceptance_criterion: Acceptance criterion config for local search.
    """

    n_ants: int = 20
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    tau_0: Optional[float] = 1.0
    tau_min: float = 0.01
    tau_max: float = 10.0
    max_iterations: int = 50
    scale: float = 5.0
    time_limit: float = 30.0
    seed: Optional[int] = None
    k_sparse: int = 15
    q0: float = 0.9
    local_search: bool = True
    local_search_iterations: int = 500
    elitist_weight: float = 1.0
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    acceptance_criterion: Optional[Union[AcceptanceConfig, IAcceptanceCriterion]] = field(
        default_factory=lambda: AcceptanceConfig(method="oi")
    )
    profit_aware_operators: bool = False

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if not (0 < self.rho < 1):
            raise ValueError(f"Evaporation rate rho must be in (0, 1), got {self.rho}")
        if self.scale < 0:
            raise ValueError(f"Scale parameter must be non-negative, got {self.scale}")

    @classmethod
    def from_config(cls, config: Union[KSparseACOConfig, Dict[str, Any]]) -> "KSparseACOConfig":
        """Create KSparseACOConfig from an KSparseACOConfig dataclass or dict.

        Performs explicit type casting for numeric fields to ensure consistency.

        Args:
            config: Configuration object or dictionary.

        Returns:
            "KSparseACOConfig" instance.
        """
        import dataclasses

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

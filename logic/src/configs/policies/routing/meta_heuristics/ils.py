"""
ILS (Iterated Local Search) configuration for Hydra.

Attributes:
    ILSConfig: Configuration for the Iterated Local Search policy.

Example:
    >>> from configs.policies.ils import ILSConfig
    >>> config = ILSConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.policies.acceptance_criteria import AcceptanceConfig


@dataclass
class ILSConfig:
    """Configuration for the Iterated Local Search policy.

    Attributes:
        engine (str): Name of the engine.
        n_restarts (int): Number of restarts.
        inner_iterations (int): Number of inner iterations.
        n_removal (int): Number of removals for perturbation.
        n_llh (int): Number of local heuristic calls per restart.
        perturbation_strength (float): Strength of perturbation (0-1).
        time_limit (float): Time limit in seconds.
        seed (Optional[int]): Random seed for reproducibility.
        profit_aware_operators (bool): Use profit-aware operators.
        packing (Optional[List[Any]]): Mandatory customers/requests selection.
        route_improvement (Optional[List[Any]]): Route improvement strategies.
        acceptance_criterion (AcceptanceConfig): Acceptance criteria for new solutions.
    """

    engine: str = "ils"
    n_restarts: int = 30
    inner_iterations: int = 20
    n_removal: int = 2
    n_llh: int = 5
    perturbation_strength: float = 0.15
    time_limit: float = 60.0
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
    acceptance_criterion: AcceptanceConfig = field(default_factory=lambda: AcceptanceConfig(method="oi"))


    def __post_init__(self):
        """Ensure acceptance criterion is initialized.

        Returns:
            None.
        """
        if self.acceptance_criterion is None:
            # Standard ILS uses Improving-Only acceptance
            from logic.src.policies.acceptance_criteria.base.factory import (
                AcceptanceCriterionFactory,
            )

            self.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

    @classmethod
    def from_config(cls, config: Any) -> "ILSConfig":
        """Build parameters from a configuration object.

        Args:
            config: Configuration source.

        Returns:
            Instantiated ILSConfig.
        """
        # Build parameters
        params = cls(
            n_restarts=getattr(config, "n_restarts", 30),
            inner_iterations=getattr(config, "inner_iterations", 20),
            n_removal=getattr(config, "n_removal", 2),
            n_llh=getattr(config, "n_llh", 5),
            perturbation_strength=getattr(config, "perturbation_strength", 0.15),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )

        # Handle Acceptance Criterion Injection
        from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory

        acceptance_cfg = getattr(config, "acceptance_criterion", None)
        if acceptance_cfg:
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acceptance_cfg.method,
                config=acceptance_cfg.params,
            )
        else:
            # Default to only_improving for standard ILS
            params.acceptance_criterion = AcceptanceCriterionFactory.create(name="oi")

        return params

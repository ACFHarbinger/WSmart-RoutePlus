"""
RTS (Reactive Tabu Search) configuration for Hydra.

Attributes:
    RTSConfig: Configuration for the Reactive Tabu Search (RTS) policy.

Example:
    >>> from configs.policies.rts import RTSConfig
    >>> config = RTSConfig()
    >>> config.initial_tenure
    7
    >>> config.max_iterations
    500
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .other.acceptance_criteria import AcceptanceConfig, AspirationConfig


@dataclass
class RTSConfig:
    """Configuration for the Reactive Tabu Search policy.

    Attributes:
        initial_tenure: Initial tenure for the tabu list.
        min_tenure: Minimum tenure for the tabu list.
        max_tenure: Maximum tenure for the tabu list.
        tenure_increase: Rate of increase for the tenure.
        tenure_decrease: Rate of decrease for the tenure.
        max_iterations: Maximum number of iterations.
        n_removal: Number of nodes to remove in each iteration.
        n_llh: Number of local search operators to apply in each iteration.
        time_limit: Time limit for the algorithm in seconds.
        seed: Random seed for the algorithm.
        profit_aware_operators: Whether to use profit-aware operators.
        packing: List of mandatory nodes.
        route_improvement: List of route improvements.
        acceptance_criterion: Acceptance criterion configuration.
    """

    initial_tenure: int = 7
    min_tenure: int = 3
    max_tenure: int = 20
    tenure_increase: float = 1.5
    tenure_decrease: float = 0.9
    max_iterations: int = 500
    n_removal: int = 2
    n_llh: int = 5
    time_limit: float = 60.0
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
    acceptance_criterion: AcceptanceConfig = AcceptanceConfig(method="ac", params=AspirationConfig())


    @classmethod
    def from_config(cls, config: "RTSConfig") -> "RTSConfig":
        """Create RTSConfig from a Hydra configuration object.

        Args:
            config (RTSConfig): The configuration object.

        Returns:
            RTSConfig: An instance of RTSConfig populated from the config.
        """
        params = cls(
            initial_tenure=getattr(config, "initial_tenure", 7),
            min_tenure=getattr(config, "min_tenure", 3),
            max_tenure=getattr(config, "max_tenure", 20),
            tenure_increase=getattr(config, "tenure_increase", 1.5),
            tenure_decrease=getattr(config, "tenure_decrease", 0.9),
            max_iterations=getattr(config, "max_iterations", 500),
            n_removal=getattr(config, "n_removal", 2),
            n_llh=getattr(config, "n_llh", 5),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )

        # Handle Acceptance Criterion Injection (Aspiration folding)
        from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory

        acceptance_cfg = getattr(config, "acceptance_criterion", None)
        if acceptance_cfg:
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name=acceptance_cfg.method,
                config=acceptance_cfg.params,
            )
        else:
            # Default to aspiration for standard RTS
            params.acceptance_criterion = AcceptanceCriterionFactory.create(name="ac")

        return params

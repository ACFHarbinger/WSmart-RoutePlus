"""
SANS (Simulated Annealing Neighborhood Search) configuration.

Attributes:
    SAConfig: Configuration for the Simulated Annealing policy.

Example:
    >>> from configs.policies.sa import SAConfig
    >>> config = SAConfig()
    >>> config.initial_temperature
    100.0
    >>> config.cooling_rate
    0.95
    >>> config.min_temperature
    0.01
    >>> config.nb_granular
    20
    >>> config.time_limit
    60.0
    >>> config.seed
    42
    >>> config.profit_aware_operators
    False
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from .other.acceptance_criteria import AcceptanceConfig, BoltzmannAcceptanceConfig
from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class SAConfig:
    """
    Configuration for the Simulated Annealing policy.

    Attributes:
        initial_temperature (float): Starting temperature T_0. Must be high enough
            to ensure an initial acceptance probability of approx 0.8 for worsening moves.
        cooling_rate (float): Geometric cooling coefficient alpha in (0, 1).
            T_{k+1} = alpha * T_k.
        min_temperature (float): Termination temperature threshold T_{min}.
            Allows the system to reach thermal equilibrium before cooling.
        nb_granular (int): Number of neighborhood search granular moves to try at each
            temperature step.
        time_limit (float): Absolute wall-clock time limit in seconds.
        seed (Optional[int]): Random seed for stochastic reproducibility.
        profit_aware_operators (bool): Flag to bias ruin/recreate toward high-revenue nodes.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
        acceptance_criterion: Acceptance criterion config for local search.
    """

    initial_temperature: float = 100.0
    cooling_rate: float = 0.95
    min_temperature: float = 0.01
    nb_granular: int = 20
    time_limit: float = 60.0
    seed: Optional[int] = 42
    profit_aware_operators: bool = False
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    acceptance_criterion: AcceptanceConfig = AcceptanceConfig(
        method="bmc", params=BoltzmannAcceptanceConfig(initial_temp=100.0, alpha=0.95)
    )
    target_acceptances_per_node: int = 10
    max_attempts_multiplier: int = 100
    frozen_streak_limit: int = 3
    auto_calibrate_temperature: bool = True
    target_initial_acceptance: float = 0.80
    calibration_samples: int = 200
    n_restarts: int = 1


    @classmethod
    def from_config(cls, config: Any) -> "SAConfig":
        """Create SAConfig from a configuration source.

        Args:
            config: Configuration source.

        Returns:
            Instantiated SAConfig.
        """
        from dataclasses import fields
        # Get base params
        if isinstance(config, dict):
            params = cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})
        else:
            params = cls(
                initial_temperature=getattr(config, "initial_temperature", 100.0),
                cooling_rate=getattr(config, "cooling_rate", 0.95),
                min_temperature=getattr(config, "min_temperature", 0.01),
                target_acceptances_per_node=getattr(config, "target_acceptances_per_node", 10),
                max_attempts_multiplier=getattr(config, "max_attempts_multiplier", 100),
                frozen_streak_limit=getattr(config, "frozen_streak_limit", 3),
                auto_calibrate_temperature=getattr(config, "auto_calibrate_temperature", True),
                target_initial_acceptance=getattr(config, "target_initial_acceptance", 0.80),
                calibration_samples=getattr(config, "calibration_samples", 200),
                n_restarts=getattr(config, "n_restarts", 1),
                time_limit=getattr(config, "time_limit", 60.0),
                seed=getattr(config, "seed", 42),
                profit_aware_operators=getattr(config, "profit_aware_operators", False),
                nb_granular=getattr(config, "nb_granular", 20),
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
            # Legacy mapping
            params.acceptance_criterion = AcceptanceCriterionFactory.create(
                name="bmc",
                initial_temp=params.initial_temperature,
                alpha=params.cooling_rate,
                seed=params.seed,
            )

        return params

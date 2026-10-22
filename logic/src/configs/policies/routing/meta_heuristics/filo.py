"""
FILO (Fast Iterative Localized Optimization) configuration dataclasses.

Attributes:
    FILOConfig: Configuration for the FILO policy.

Example:
    >>> from configs.policies.filo import FILOConfig
    >>> config = FILOConfig()
    >>> config.time_limit
    60.0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from logic.src.configs.policies.routing.meta_heuristics.abc import ABCConfig


@dataclass
class FILOConfig(ABCConfig):
    """
    Configuration parameters for the Fast Iterative Localized Optimization (FILO) solver.

    Attributes:
        time_limit: Maximum allowed runtime in seconds.
        max_iterations: Maximum number of core optimization iterations.
        initial_temperature_factor: Divisor for initial SA temperature.
        final_temperature_factor: Divisor for final SA temperature.
        shaking_lb_factor: Lower bound intensification factor for shaking.
        shaking_ub_factor: Upper bound intensification factor for shaking.
        shaking_lb_intensity: Lower intensity bound for adaptive omega.
        shaking_ub_intensity: Upper intensity bound for adaptive omega.
        omega_alpha: Probability of jumping to a neighbor route during ruin.
        omega_base_multiplier: Base shaking intensity multiplier.
        delta_gamma: Coefficient for dynamic gamma threshold calculation.
        gamma_base: Base probability for neighborhood evaluation.
        gamma_lambda: Expansion factor for stagnant nodes.
        svc_size: Maximum size of the Selective Vertex Cache.
        n_cw: Number of nearest neighbors for Clarke-Wright savings.
        local_search_iterations: Maximum local search iterations per call.
        seed: Random seed for reproducibility.
        profit_aware_operators: Whether to use profit-aware destroy/repair operators.
    """

    time_limit: float = 60.0
    max_iterations: int = 50000
    initial_temperature_factor: float = 10.0
    final_temperature_factor: float = 100.0
    shaking_lb_factor: float = 0.5
    shaking_ub_factor: float = 2.0
    shaking_lb_intensity: float = 0.5
    shaking_ub_intensity: float = 1.5
    omega_alpha: float = 0.5
    omega_base_multiplier: float = 1.0
    delta_gamma: float = 0.25
    gamma_base: float = 1.0
    gamma_lambda: float = 2.0
    svc_size: int = 50
    n_cw: int = 100
    local_search_iterations: int = 500
    seed: int = 42
    profit_aware_operators: bool = False
    route_min_sa_temp: float = 1.0


    @classmethod
    def from_config(cls, config: Any) -> "FILOConfig":
        """Create FILOConfig from a configuration object.

        Args:
            config: Configuration object or dictionary.

        Returns:
            "FILOConfig" instance.
        """
        import dataclasses
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})

        return cls(
            time_limit=getattr(config, "time_limit", 60.0),
            max_iterations=getattr(config, "max_iterations", 50000),
            initial_temperature_factor=getattr(config, "initial_temperature_factor", 10.0),
            final_temperature_factor=getattr(config, "final_temperature_factor", 100.0),
            omega_alpha=getattr(config, "omega_alpha", 0.5),
            shaking_lb_intensity=getattr(config, "shaking_lb_intensity", 0.5),
            shaking_ub_intensity=getattr(config, "shaking_ub_intensity", 1.5),
            omega_base_multiplier=getattr(config, "omega_base_multiplier", 1.0),
            delta_gamma=getattr(config, "delta_gamma", 0.25),
            gamma_base=getattr(config, "gamma_base", 1.0),
            gamma_lambda=getattr(config, "gamma_lambda", 2.0),
            local_search_iterations=getattr(config, "local_search_iterations", 500),
            n_cw=getattr(config, "n_cw", 100),
            svc_size=getattr(config, "svc_size", 50),
            seed=getattr(config, "seed", 42),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )

"""
PSOMA (Particle Swarm Optimization Memetic Algorithm) configuration for Hydra.

Attributes:
    PSOMAConfig: Configuration for the PSOMA policy.

Example:
    >>> from configs.policies.psoma import PSOMAConfig
    >>> config = PSOMAConfig()
    >>> config.pop_size
    20
    >>> config.max_iterations
    200
    >>> config.seed
    42
    >>> config.vrpp
    True
    >>> config.profit_aware_operators
    False
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .other.acceptance_criteria import AcceptanceConfig, BoltzmannAcceptanceConfig
from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class PSOMAConfig:
    """
    Configuration for the PSOMA policy.

    Attributes:
        pop_size: Swarm size.
        omega: Inertia weight.
        c1: Cognitive acceleration coefficient.
        c2: Social acceleration coefficient.
        max_iterations: Maximum PSO iterations.
        x_min: Minimum value for continuous position vector.
        x_max: Maximum value for continuous position vector.
        v_min: Minimum value for continuous velocity vector.
        v_max: Maximum value for continuous velocity vector.
        L: Number of iterations for simulated annealing cooling.
        T0: Initial temperature for simulated annealing.
        lambda_cooling: Cooling rate for simulated annealing.
        time_limit: Wall-clock time limit in seconds.
        vrpp: If True, solver operates in full VRPP mode.
        profit_aware_operators: Whether to use profit-aware operators.
        seed: Random seed for reproducibility.
        packing: Packing algorithm config list.
        route_improvement: Route improvement operation config list.
        acceptance_criterion: Acceptance criterion config for local search.
    """

    pop_size: int = 20
    omega: float = 1.0
    c1: float = 2.0
    c2: float = 2.0
    max_iterations: int = 200
    x_min: float = 0.0
    x_max: float = 4.0
    v_min: float = -4.0
    v_max: float = 4.0
    L: int = 30
    T0: float = 3.0
    lambda_cooling: float = 0.9
    time_limit: float = 60.0
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = 42
    packing: Optional[List[PackingConfig]] = field(default_factory=list)
    route_improvement: Optional[List[RouteImprovingConfig]] = field(default_factory=list)
    acceptance_criterion: AcceptanceConfig = field(
        default_factory=lambda: AcceptanceConfig(
            method="bmc",
            params=BoltzmannAcceptanceConfig(
                initial_temp=3.0,
                alpha=0.9,
                seed=42,
            ),
        )
    )

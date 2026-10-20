"""
Particle Swarm Optimization with Velocity Momentum configuration.

**TRUE PSO** implementation with inertia-weighted velocity updates.
Replaces the deceptive "Firefly Algorithm" which lacked velocity momentum.

Attributes:
    DistancePSOConfig: Configuration for the Particle Swarm Optimization (PSO) policy.

Example:
    >>> from configs.policies.psoda import DistancePSOConfig
    >>> config = DistancePSOConfig()
    >>> config.population_size
    20
    >>> config.max_iterations
    500
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from logic.src.configs.policies.other.packing_dispatch import PackingConfig

from .other.route_improvement import RouteImprovingConfig


@dataclass
class DistancePSOConfig:
    """Configuration for PSO with velocity momentum policy.

    **TRUE PARTICLE SWARM OPTIMIZATION** (Kennedy & Eberhart 1995).

    Core PSO Components:
    - Velocity vectors with momentum (inertia term)
    - Personal best (pbest) tracking per particle (cognitive term)
    - Global best (gbest) for the swarm (social term)
    - Linearly decreasing inertia weight

    References:
        Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization."
        Ai, T. J., & Kachitvichyanukul, V. (2009). "A particle swarm optimization
        for the vehicle routing problem with simultaneous pickup and delivery."
    Attributes:
        None
    """

    # Swarm Configuration
    population_size: int = 20

    # PSO Velocity Momentum Parameters
    inertia_weight_start: float = 0.9  # w(0) - high for exploration
    inertia_weight_end: float = 0.4  # w(T) - low for exploitation
    cognitive_coef: float = 2.0  # c₁ - personal best attraction
    social_coef: float = 2.0  # c₂ - global best attraction

    # Discrete Adaptation Parameters
    n_removal: int = 3  # Nodes in velocity perturbation
    velocity_to_mutation_rate: float = 0.1  # Velocity → mutation scaling

    # Optimization Parameters
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0

    # Guided Insertion Scoring Weights
    alpha_profit: float = 1.0
    beta_will: float = 0.5
    gamma_cost: float = 0.3

    # Infrastructure
    profit_aware_operators: bool = False
    seed: Optional[int] = None
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None

    def get_inertia_weight(self, iteration: int) -> float:
        """
        Compute linearly decreasing inertia weight.

        High inertia weight early → exploration (particles move freely).
        Low inertia weight late → exploitation (particles converge to best).

        Args:
            iteration: Current iteration number (0-indexed).

        Returns:
            Inertia weight w(t) ∈ [w_end, w_start].

        Formula (Shi & Eberhart 1998):
            w(t) = w_max - (w_max - w_min) × (t / T_max)
        """
        if self.max_iterations <= 1:
            return self.inertia_weight_end

        decay_factor = iteration / (self.max_iterations - 1)
        return self.inertia_weight_start - (self.inertia_weight_start - self.inertia_weight_end) * decay_factor

    @property
    def c1(self) -> float:
        """Alias for cognitive_coef (standard PSO notation).

        Returns:
            float: Cognitive acceleration coefficient.
        """
        return self.cognitive_coef

    @property
    def c2(self) -> float:
        """Alias for social_coef (standard PSO notation).

        Returns:
            float: Social acceleration coefficient.
        """
        return self.social_coef

    @property
    def w_start(self) -> float:
        """Alias for inertia_weight_start.

        Returns:
            float: Initial inertia weight.
        """
        return self.inertia_weight_start

    @property
    def w_end(self) -> float:
        """Alias for inertia_weight_end.

        Returns:
            float: Final inertia weight.
        """
        return self.inertia_weight_end

    @property
    def pop_size(self) -> int:
        """Alias for population_size.

        Returns:
            int: Number of particles in the swarm.
        """
        return self.population_size

    @classmethod
    def from_config(cls, config: Any) -> "DistancePSOConfig":
        """Create parameters from a configuration object.

        Args:
            config: Configuration object with PSO parameter attributes.

        Returns:
            DistancePSOConfig: Populated parameter dataclass.
        """
        return cls(
            population_size=getattr(config, "population_size", 20),
            max_iterations=getattr(config, "max_iterations", 500),
            inertia_weight_start=getattr(config, "inertia_weight_start", 0.9),
            inertia_weight_end=getattr(config, "inertia_weight_end", 0.4),
            cognitive_coef=getattr(config, "cognitive_coef", 2.0),
            social_coef=getattr(config, "social_coef", 2.0),
            n_removal=getattr(config, "n_removal", 3),
            velocity_to_mutation_rate=getattr(config, "velocity_to_mutation_rate", 0.1),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            alpha_profit=getattr(config, "alpha_profit", 1.0),
            beta_will=getattr(config, "beta_will", 0.5),
            gamma_cost=getattr(config, "gamma_cost", 0.3),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            seed=getattr(config, "seed", None),
        )

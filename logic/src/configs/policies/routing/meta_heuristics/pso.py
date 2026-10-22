"""
Particle Swarm Optimization (PSO) with velocity momentum configuration.

**TRUE PSO** replacing the Sine Cosine Algorithm (SCA).

Attributes:
    PSOConfig: Configuration for the PSO policy.

Example:
    >>> from configs.policies.pso import PSOConfig
    >>> config = PSOConfig()
    >>> config.pop_size
    30
    >>> config.max_iterations
    500
    >>> config.seed
    None
    >>> config.packing
    None
    >>> config.route_improvement
    None
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from logic.src.configs.policies.packing.dispatch import PackingConfig

from logic.src.configs.policies.route_improvement import RouteImprovingConfig


@dataclass
class PSOConfig:
    """Configuration for PSO with velocity momentum policy.

    **Replaces SCA** - Proper PSO with all components intact.

    Core PSO Components (Kennedy & Eberhart 1995):
    - Velocity vectors with momentum (inertia term)
    - Personal best (pbest) tracking per particle (cognitive term)
    - Global best (gbest) for the swarm (social term)
    - Linearly decreasing inertia weight

    SCA is mathematically equivalent to PSO without velocity momentum,
    but with expensive trigonometric operations providing no benefit.

    References:
        Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization."
        Proceedings of ICNN'95 - International Conference on Neural Networks.

        Shi, Y., & Eberhart, R. (1998). "A modified particle swarm optimizer."
        IEEE International Conference on Evolutionary Computation.
    Attributes:
        None
    """

    # Swarm Configuration
    pop_size: int = 30

    # PSO Velocity Momentum Parameters
    inertia_weight_start: float = 0.9  # w(0) - high for exploration
    inertia_weight_end: float = 0.4  # w(T) - low for exploitation
    cognitive_coef: float = 2.0  # c₁ - personal best attraction
    social_coef: float = 2.0  # c₂ - global best attraction

    # Continuous Space Bounds
    position_min: float = -1.0  # Lower bound for positions
    position_max: float = 1.0  # Upper bound for positions
    velocity_max: float = 0.5  # Maximum velocity (clamping)

    # Runtime Control
    max_iterations: int = 500
    time_limit: float = 60.0

    # Infrastructure
    profit_aware_operators: bool = False
    seed: Optional[int] = None
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None

    def get_inertia_weight(self, iteration: int) -> float:
        """Compute linearly decreasing inertia weight.

        Args:
            iteration: Current iteration number (0-indexed).

        Returns:
            Inertia weight w(t).
        """
        if self.max_iterations <= 1:
            return self.inertia_weight_end

        decay_factor = iteration / (self.max_iterations - 1)
        return self.inertia_weight_start - (self.inertia_weight_start - self.inertia_weight_end) * decay_factor

    @property
    def c1(self) -> float:
        """Standard PSO notation for cognitive_coef.

        Returns:
            Cognitive coefficient.
        """
        return self.cognitive_coef

    @property
    def c2(self) -> float:
        """Standard PSO notation for social_coef.

        Returns:
            Social coefficient.
        """
        return self.social_coef

    @property
    def w_start(self) -> float:
        """Alias for inertia_weight_start.

        Returns:
            Initial inertia weight.
        """
        return self.inertia_weight_start

    @property
    def w_end(self) -> float:
        """Alias for inertia_weight_end.

        Returns:
            Final inertia weight.
        """
        return self.inertia_weight_end

    @classmethod
    def from_config(cls, config: Any) -> "PSOConfig":
        """Create parameters from a configuration object.

        Args:
            config: Configuration source.

        Returns:
            Instantiated PSOConfig.
        """
        return cls(
            pop_size=getattr(config, "pop_size", 30),
            inertia_weight_start=getattr(config, "inertia_weight_start", 0.9),
            inertia_weight_end=getattr(config, "inertia_weight_end", 0.4),
            cognitive_coef=getattr(config, "cognitive_coef", 2.0),
            social_coef=getattr(config, "social_coef", 2.0),
            position_min=getattr(config, "position_min", -1.0),
            position_max=getattr(config, "position_max", 1.0),
            velocity_max=getattr(config, "velocity_max", 0.5),
            max_iterations=getattr(config, "max_iterations", 500),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )

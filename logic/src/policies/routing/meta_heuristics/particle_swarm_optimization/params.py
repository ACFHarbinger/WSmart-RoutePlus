"""Configuration parameters for Particle Swarm Optimization (PSO).

Attributes:
    PSOParams: Parameter dataclass for Particle Swarm Optimization.

Example:
    >>> params = PSOParams(pop_size=50)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PSOParams:
    """Parameters for standard Particle Swarm Optimization.

    Attributes:
        pop_size: Number of particles in swarm.
        inertia_weight_start: Initial inertia weight w(0).
        inertia_weight_end: Final inertia weight w(T).
        cognitive_coef: c1 - personal best acceleration constant.
        social_coef: c2 - global best acceleration constant.
        position_min: Minimum value for position vector components.
        position_max: Maximum value for position vector components.
        velocity_max: Maximum velocity magnitude.
        max_iterations: Maximum number of PSO iterations.
        time_limit: Wall-clock time limit in seconds.
        seed: Random seed.
        vrpp: Whether solving VRP with Profits.
        profit_aware_operators: Whether to use profit-aware operators.
    """

    # Swarm Configuration
    pop_size: int = 30

    # PSO Velocity Momentum Parameters
    inertia_weight_start: float = 0.9  # w(0) - high for initial exploration
    inertia_weight_end: float = 0.4  # w(T) - low for final exploitation
    cognitive_coef: float = 2.0  # c₁ - personal best attraction
    social_coef: float = 2.0  # c₂ - global best attraction

    # Continuous Space Bounds
    position_min: float = -1.0  # Lower bound for position vector
    position_max: float = 1.0  # Upper bound for position vector
    velocity_max: float = 0.5  # Maximum velocity (prevents divergence)

    # Runtime Control
    max_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

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
    def from_config(cls, config: Any) -> "PSOParams":
        """Create parameters from a configuration object.

        Args:
            config: Configuration source.

        Returns:
            Instantiated PSOParams.
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
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )

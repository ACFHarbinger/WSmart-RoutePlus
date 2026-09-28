"""Configuration for various acceptance criteria used in local search and metaheuristics.

Attributes:
    OnlyImprovingConfig: Configuration for Only Improving acceptance.
    ImprovingAndEqualConfig: Configuration for Improving and Equal acceptance.
    AllMovesAcceptedConfig: Configuration for All Moves Accepted (random walk).
    BoltzmannAcceptanceConfig: Configuration for Boltzmann-Metropolis (Simulated Annealing).
    DemonAlgorithmConfig: Configuration for Demon Algorithm.
    GeneralizedTsallisSAConfig: Configuration for Generalized Tsallis SA.
    NonLinearGreatDelugeConfig: Configuration for Non-Linear Great Deluge.
    EMCQConfig: Configuration for Exponential Monte Carlo with Counter.
    AdaptiveBoltzmannConfig: Configuration for Adaptive Boltzmann Metropolis.
    LateAcceptanceConfig: Configuration for Late Acceptance Hill Climbing.
    StepCountingConfig: Configuration for Step Counting Hill Climbing.
    GreatDelugeConfig: Configuration for Great Deluge.
    ThresholdAcceptingConfig: Configuration for Threshold Accepting.
    MonteCarloConfig: Configuration for Fixed Monte Carlo.
    AspirationConfig: Configuration for Aspiration Criterion.
    AcceptanceConfig: Unified configuration wrapper for selecting an acceptance criterion.

Example:
    >>> from configs.policies.other.acceptance_criteria import AcceptanceConfig
    >>> config = AcceptanceConfig(method='boltzmann', params= BoltzmannAcceptanceConfig())
    >>> config.method
    'boltzmann'
    >>> config.params
    BoltzmannAcceptanceConfig(initial_temp=100.0, alpha=0.95, seed=42)
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OnlyImprovingConfig:
    """Configuration for Only Improving acceptance.

    Attributes:
        None
    """

    pass


@dataclass
class ImprovingAndEqualConfig:
    """Configuration for Improving and Equal acceptance.

    Attributes:
        None
    """

    pass


@dataclass
class AllMovesAcceptedConfig:
    """Configuration for All Moves Accepted (random walk).

    Attributes:
        None
    """

    pass


@dataclass
class BoltzmannAcceptanceConfig:
    """Configuration for Boltzmann-Metropolis (Simulated Annealing).

    Attributes:
        initial_temp: Initial temperature for the Boltzmann distribution.
        alpha: Cooling rate for the temperature.
        seed: Random seed for reproducibility.
    """

    initial_temp: float = 100.0
    alpha: float = 0.95
    seed: int = 42


@dataclass
class DemonAlgorithmConfig:
    """Configuration for Demon Algorithm.

    Attributes:
        initial_credit: Initial credit for the demon.
        is_stochastic: Whether the demon algorithm is stochastic.
        maximization: Whether the problem is a maximization problem.
        seed: Random seed for reproducibility.
    """

    initial_credit: float = 0.0
    is_stochastic: bool = False
    maximization: bool = True
    seed: int = 42


@dataclass
class GeneralizedTsallisSAConfig:
    """Configuration for Generalized Tsallis SA.

    Attributes:
        q: Temperature parameter for the Tsallis distribution.
        initial_temp: Initial temperature for the Boltzmann distribution.
        alpha: Cooling rate for the temperature.
        seed: Random seed for reproducibility.
        maximization: Whether the problem is a maximization problem.
    """

    q: float = 1.0
    initial_temp: float = 100.0
    alpha: float = 0.95
    seed: int = 42
    maximization: bool = True


@dataclass
class NonLinearGreatDelugeConfig:
    """Configuration for Non-Linear Great Deluge.

    Attributes:
        initial_level: Initial level for the great deluge.
        beta: Decay rate for the level.
        t_max: Maximum number of iterations.
        maximization: Whether the problem is a maximization problem.
    """

    initial_level: float = 100.0
    beta: float = 0.01
    t_max: int = 1000
    maximization: bool = True


@dataclass
class EMCQConfig:
    """Configuration for Exponential Monte Carlo with Counter.

    Attributes:
        p: Probability of accepting a solution.
        p_boost: Probability boost for accepting better solutions.
        q_threshold: Threshold for the counter.
        seed: Random seed for reproducibility.
    """

    p: float = 0.1
    p_boost: float = 0.5
    q_threshold: int = 50
    seed: int = 42


@dataclass
class AdaptiveBoltzmannConfig:
    """Configuration for Adaptive Boltzmann Metropolis.

    Attributes:
        p0: Initial probability of accepting a solution.
        window_size: Size of the acceptance window.
        alpha: Cooling rate for the temperature.
        min_temp: Minimum temperature.
        seed: Random seed for reproducibility.
        maximization: Whether the problem is a maximization problem.
    """

    p0: float = 0.5
    window_size: int = 100
    alpha: float = 0.95
    min_temp: float = 1e-6
    seed: int = 42
    maximization: bool = True


@dataclass
class LateAcceptanceConfig:
    """Configuration for Late Acceptance Hill Climbing.

    Attributes:
        history_length: Length of the acceptance history.
        maximization: Whether the problem is a maximization problem.
    """

    history_length: int = 50
    maximization: bool = True


@dataclass
class StepCountingConfig:
    """Configuration for Step Counting Hill Climbing.

    Attributes:
        maximization: Whether the problem is a maximization problem.
    """

    maximization: bool = True


@dataclass
class GreatDelugeConfig:
    """Configuration for Great Deluge.

    Attributes:
        initial_level: Initial level for the great deluge.
        decay_rate: Decay rate for the level.
        maximization: Whether the problem is a maximization problem.
    """

    initial_level: float = 100.0
    decay_rate: float = 0.01
    maximization: bool = True


@dataclass
class ThresholdAcceptingConfig:
    """Configuration for Threshold Accepting.

    Attributes:
        initial_threshold: Initial threshold for accepting solutions.
        decay_rate: Decay rate for the threshold.
        maximization: Whether the problem is a maximization problem.
    """

    initial_threshold: float = 100.0
    decay_rate: float = 0.95
    maximization: bool = True


@dataclass
class MonteCarloConfig:
    """Configuration for Fixed Monte Carlo.

    Attributes:
        p: Probability of accepting a solution.
        seed: Random seed for reproducibility.
    """

    p: float = 0.1
    seed: int = 42


@dataclass
class AspirationConfig:
    """Configuration for Aspiration Criterion.

    Attributes:
        None
    """

    pass


@dataclass
class AcceptanceConfig:
    """
    Unified configuration wrapper for selecting an acceptance criterion.

    Attributes:
        method: The identifier for the criterion (e.g., 'boltzmann', 'demon').
        params: The specific configuration object for the chosen method.
    """

    method: str = "oi"
    params: Any = field(default_factory=OnlyImprovingConfig)

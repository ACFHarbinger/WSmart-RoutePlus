"""
QDE (Quantum-Inspired Differential Evolution) configuration for Hydra.

Attributes:
    QDEConfig: Configuration for the Quantum-Inspired Differential Evolution policy.

Example:
    >>> from configs.policies.qde import QDEConfig
    >>> config = QDEConfig()
    >>> config.pop_size
    20
    >>> config.max_iterations
    200
    >>> config.vrpp
    True
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class QDEConfig:
    """
    Configuration for the Quantum-Inspired Differential Evolution policy.

    Attributes:
        pop_size: Population size.
        F: Mutation scaling factor ∈ (0, 2].
        CR: Crossover probability ∈ [0, 1].
        max_iterations: Maximum DE generations.
        time_limit: Wall-clock time limit in seconds.
        n_removal: Nodes removed per collapse-repair step.
        vrpp: If True, solver operates in full VRPP mode (consider all bins).
        packing: Packing algorithm config list.
        route_improvement: Route improvement operation config list.
    """

    pop_size: int = 20
    F: float = 0.5
    CR: float = 0.7
    max_iterations: int = 200
    time_limit: float = 60.0
    seed: Optional[int] = None
    n_removal: int = 2
    vrpp: bool = True
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)

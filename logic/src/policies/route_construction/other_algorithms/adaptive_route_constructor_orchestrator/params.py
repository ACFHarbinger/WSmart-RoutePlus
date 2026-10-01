"""
Configuration parameters for the Adaptive Route Constructor Orchestrator (ARCO).

Attributes:
    ARCOParams: Parameters for the ARCO algorithm.

Example:
    >>> from logic.src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator import ARCOParams
    >>> params = ARCOParams()
    >>> params
    ARCOParams(constructors=['tsp', 'alns'], time_limit=120.0, selection_strategy='epsilon_greedy', epsilon=0.15, temperature=1.0, alpha_ema=0.15, weight_init=1.0, weight_floor=0.01, decay=1.0, seed=42)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class ARCOParams:
    """
    Runtime parameters for the Adaptive Route Constructor Orchestrator.

    Attributes:
        constructors: Candidate constructor names; ARCO selects an ordered
            permutation of these at each call using adaptive weights.
        time_limit: Maximum wall-clock seconds for the full execution chain.
        selection_strategy: ``"epsilon_greedy"``, ``"greedy"``, or
            ``"softmax"``.
        epsilon: Exploration probability for ε-greedy ∈ [0, 1].
        temperature: Boltzmann temperature for softmax selection > 0.
        alpha_ema: EMA factor for online weight updates ∈ (0, 1].
        weight_init: Starting value for all pair-wise weights.
        weight_floor: Lower bound on any weight (prevents total collapse).
        decay: Per-call multiplicative decay on all weights ∈ (0, 1].
        seed: Random seed for reproducible exploration.
    """

    constructors: List[str] = field(default_factory=lambda: ["tsp", "alns"])
    time_limit: float = 120.0
    selection_strategy: str = "epsilon_greedy"
    epsilon: float = 0.15
    temperature: float = 1.0
    alpha_ema: float = 0.15
    weight_init: float = 1.0
    weight_floor: float = 0.01
    decay: float = 1.0
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> ARCOParams:
        """Build parameters from a config object or dict.

        Args:
            config: Configuration object.

        Returns:
            ARCOParams: The parameters for the ARCO algorithm.
        """
        return cls(
            constructors=getattr(config, "constructors", ["tsp", "alns"]),
            time_limit=getattr(config, "time_limit", 120.0),
            selection_strategy=getattr(config, "selection_strategy", "epsilon_greedy"),
            epsilon=getattr(config, "epsilon", 0.15),
            temperature=getattr(config, "temperature", 1.0),
            alpha_ema=getattr(config, "alpha_ema", 0.15),
            weight_init=getattr(config, "weight_init", 1.0),
            weight_floor=getattr(config, "weight_floor", 0.01),
            decay=getattr(config, "decay", 1.0),
            seed=getattr(config, "seed", 42),
        )

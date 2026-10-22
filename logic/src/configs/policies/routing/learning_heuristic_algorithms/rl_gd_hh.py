"""
RL-GD-HH (Reinforcement Learning Great Deluge Hyper-Heuristic) configuration for Hydra.

This module defines the schema and default values for the RL-GD-HH policy,
aligned directly with the Ozcan et al. (2010) paper specification.

Paper mappings (Section 3, Table 1):
    - maxUtilityValue = utility_upper_bound = 40
    - Initial utility  = 0.75 × 40 = 30
    - qualityLB        = quality_lb = 0
    - Reward           = +1 (additive, RL1)
    - Penalty          = −1 (RL1), ÷2 (RL2), √ (RL3)

Attributes:
    RLGDHHConfig: Configuration for the RL-GD-HH policy.

Example:
    >>> from configs.policies.rl_gd_hh import RLGDHHConfig
    >>> config = RLGDHHConfig()
    >>> config.max_iterations
    5000
    >>> config.time_limit
    60.0
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, List, Optional

from logic.src.configs.policies.acceptance_criteria import AcceptanceConfig


@dataclass
class RLGDHHConfig:
    """
    Configuration for the RL-GD-HH policy.

    The values defined here reproduce the best-performing configuration
    from Ozcan et al. (2010) unless otherwise noted.

    Attributes:
        max_iterations: Maximum number of iterations.
        time_limit: Time limit for the algorithm in seconds.
        seed: Random seed.
        reward_improvement: Reward for improving the solution.
        penalty_worsening: Penalty for worsening the solution.
        punishment_type: Type of punishment.
        utility_upper_bound: Upper bound for utility.
        min_utility: Minimum utility.
        initial_utility: Initial utility.
        quality_lb: Lower bound for quality.
        packing: List of mandatory nodes.
        route_improvement: List of route improvements.
        acceptance_criterion: Acceptance criterion configuration.
    """

    # --- Search Control ---
    max_iterations: int = 5000
    time_limit: float = 60.0
    seed: Optional[int] = None

    # --- RL Adaptation Settings (Section 3.2) ---
    # Additive reward when a LLH strictly improves the solution (Step 14).
    reward_improvement: float = 1.0
    # Penalty magnitude applied for worsening AND neutral moves (Step 17).
    # The paper treats neutral moves identically to worsening ones.
    penalty_worsening: float = 1.0
    # Punishment variant: "RL1" (subtract), "RL2" (halve), "RL3" (root).
    punishment_type: str = "RL1"

    # --- Utility Constraints (Table 1: best maxUtilityValue = 40) ---
    utility_upper_bound: float = 40.0
    min_utility: float = 0.0
    # Initial utility = 0.75 × maxUtilityValue = 30 (per paper p. 16).
    initial_utility: float = 30.0

    # --- Great Deluge: Declining Boundary (Fig 2, Step 18) ---
    # level(t) = quality_lb + (f0 - quality_lb) × (1 − t / T)
    # qualityLB = 0 for all benchmark instances in the paper.
    quality_lb: float = 0.0

    # --- Problem Flags ---

    # --- Infrastructure Hooks ---
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
    acceptance_criterion: Optional[AcceptanceConfig] = None
    profit_aware_operators: bool = False

    def __post_init__(self):
        # Keep this declarative: the solver creates its own criterion each run.
        if self.acceptance_criterion is None:
            self.acceptance_criterion = AcceptanceConfig(
                method="gd",
                params={"target_objective": self.quality_lb, "max_iterations": self.max_iterations},
            )
        else:
            spec = self.acceptance_criterion
            method = spec.get("method") if isinstance(spec, Mapping) else getattr(spec, "method", None)
            if method == "great_deluge":
                params = dict(spec.get("params", {}) if isinstance(spec, Mapping) else spec.params)
                params["target_objective"] = params.pop("quality_lb", self.quality_lb)
                params.setdefault("max_iterations", self.max_iterations)
                self.acceptance_criterion = AcceptanceConfig(method="gd", params=params)

    @classmethod
    def from_config(cls, config: Any) -> "RLGDHHConfig":
        """Load a mapping or attribute config without constructing search state."""
        if isinstance(config, Mapping):
            values = {item.name: config[item.name] for item in fields(cls) if item.name in config}
        else:
            values = {item.name: getattr(config, item.name) for item in fields(cls) if hasattr(config, item.name)}
        return cls(**values)

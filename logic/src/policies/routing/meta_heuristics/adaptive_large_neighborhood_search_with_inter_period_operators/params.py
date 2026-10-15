r"""Configuration parameters for the ALNS-IPO solver.

Attributes:
    ALNSIPOParams: Configuration parameters for the ALNS-IPO solver.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.params import ALNSIPOParams
    >>> params = ALNSIPOParams(horizon=7)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from logic.src.configs.policies.alns import ALNSConfig


@dataclass
class ALNSIPOParams(ALNSConfig):
    """
    Configuration parameters for the ALNS-IPO solver.

    Extends ALNSConfig with multi-period stochastic routing parameters.

    Attributes:
        horizon: Planning horizon T (number of days).
        stockout_penalty: Penalty per unit of bin overflow.
        forward_looking_depth: Lookahead depth H for ForwardLookingInsertion.
        inter_period_operators: If False, disable inter-period operators.
        shift_direction: Direction for ShiftVisitRemoval ("both", "forward", "backward").
        inventory_lambda: Weight on inventory term in ForwardLookingInsertion.
        inter_period_weight: Initial roulette weight for inter-period operators.
        stochastic_repair: If True, multi-period repair ops use ScenarioTree evaluation.
    """

    horizon: int = 7
    stockout_penalty: float = 500.0
    forward_looking_depth: int = 3
    inter_period_operators: bool = True
    shift_direction: str = "both"
    inventory_lambda: float = 1.0
    inter_period_weight: float = 1.0
    stochastic_repair: bool = False

    @classmethod
    def from_config(cls, config: Any) -> ALNSIPOParams:
        """Create ALNSIPOParams from an ALNSIPOConfig dataclass or dict.

        Args:
            config: ALNSIPOConfig dataclass or dict with solver parameters.

        Returns:
            ALNSIPOParams instance with values from config.
        """
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in valid_keys})

        kwargs = {}
        for k in valid_keys:
            if hasattr(config, k):
                kwargs[k] = getattr(config, k)
        return cls(**kwargs)

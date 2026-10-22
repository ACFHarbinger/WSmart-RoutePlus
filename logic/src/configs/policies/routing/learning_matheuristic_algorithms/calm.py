"""
Configuration for the Concurrent Adaptive Lagrangian Matheuristic (CALM).

Attributes:
    CALMConfig: Attributes for CALM configuration.

Example:
    >>> from configs.policies.calm import CALMConfig
    >>> config = CALMConfig()
    >>> config.time_limit
    600.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class CALMConfig:
    """
    Configuration for the Concurrent Adaptive Lagrangian Matheuristic (CALM) policy.

    Attributes:
        lookahead (Dict[str, Any]): Lookahead evaluation and scenario generation parameters.
        lagrangian (Dict[str, Any]): Lagrangian multiplier coordination subsystem configuration.
        dual_bound (Dict[str, Any]): Dual-bound tracking subsystem configuration.
        bandit (Dict[str, Any]): LinUCB contextual bandit algorithm choice configuration.
        regret (Dict[str, Any]): Regret-based preprocessing module configuration.
        tpks (Dict[str, Any]): Two-phase kernel search parameters (used by solver engines).
        time_limit (float): Global time budget across outer iterations.
        seed (int): Random seed for reproducibility.
        verbose (bool): Whether to enable verbose execution logs.
        stockout_penalty (float): Overflow (stockout) penalty metric.
    """

    lookahead: Dict[str, Any] = field(default_factory=dict)
    lagrangian: Dict[str, Any] = field(default_factory=dict)
    dual_bound: Dict[str, Any] = field(default_factory=dict)
    bandit: Dict[str, Any] = field(default_factory=dict)
    regret: Dict[str, Any] = field(default_factory=dict)
    tpks: Dict[str, Any] = field(default_factory=dict)

    time_limit: float = 600.0
    seed: int = 42
    verbose: bool = False
    stockout_penalty: float = 500.0

"""Configuration parameters for the Harmony Search (HS) solver.

Attributes:
    HSParams: Parameter dataclass for the Harmony Search.

Example:
    >>> params = HSParams(hm_size=20)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class HSParams:
    """Configuration parameters for the Harmony Search solver.

    Attributes:
        hm_size: Size of the Harmony Memory (archive capacity).
        HMCR: Harmony Memory Considering Rate.
        PAR: Pitch Adjusting Rate.
        BW: Bandwidth for adjustment.
        max_iterations: Total number of improvisation cycles.
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
        seed: Random seed for reproducibility.
        vrpp: Whether solving VRP with Profits.
        profit_aware_operators: Whether to use profit-aware operators.
    """

    hm_size: int = 10
    HMCR: float = 0.95
    PAR: float = 0.3
    BW: float = 0.05  # Bandwidth for continuous adjustment (or discrete neighbor range)
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> HSParams:
        """Build parameters from a configuration object.

        Args:
            config: Configuration source.

        Returns:
            Instantiated HSParams.
        """
        return cls(
            hm_size=getattr(config, "hm_size", 10),
            HMCR=getattr(config, "HMCR", 0.95),
            PAR=getattr(config, "PAR", 0.3),
            BW=getattr(config, "BW", 0.05),
            max_iterations=getattr(config, "max_iterations", 500),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )

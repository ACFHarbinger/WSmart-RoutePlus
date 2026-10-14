"""Configuration parameters for the Exact Stochastic Dynamic Programming (ESDP) solver.

Attributes:
    SDPParams (class): Standardized parameters for ESDP.

Example:
    >>> params = SDPParams(num_days=3, max_nodes=5)
    >>> config_dict = params.to_dict()
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SDPParams:
    """Standardized parameters for Exact Stochastic Dynamic Programming.

    Attributes:
        time_limit (float): Maximum time in seconds for the solver.
        num_days (int): Planning horizon in days.
        discrete_levels (int): Number of discretization levels for waste.
        max_fill_rate (float): Maximum assumed daily fill rate.
        max_nodes (int): Maximum nodes to consider in the state space.
        discount_factor (float): Discount factor for future rewards.
        overflow_penalty (float): Penalty for bin overflow.
        cost_weight (float): Multiplier for travel cost.
        waste_weight (float): Multiplier for waste revenue.
    """

    time_limit: float = 3600.0
    num_days: int = 5
    discrete_levels: int = 4
    max_fill_rate: float = 0.3
    max_nodes: int = 8
    discount_factor: float = 0.99
    overflow_penalty: float = 100.0
    cost_weight: float = 1.0
    waste_weight: float = 10.0

    @classmethod
    def from_config(cls, config: dict) -> "SDPParams":
        """Create SDPParams from configuration dictionary.

        Args:
            config (dict): Configuration dictionary with parameter overrides.

        Returns:
            SDPParams: An instance of the parameter class.
        """
        return cls(
            time_limit=config.get("time_limit", 3600.0),
            num_days=config.get("num_days", 5),
            discrete_levels=config.get("discrete_levels", 4),
            max_fill_rate=config.get("max_fill_rate", 0.3),
            max_nodes=config.get("max_nodes", 8),
            discount_factor=config.get("discount_factor", 0.99),
            overflow_penalty=config.get("overflow_penalty", 100.0),
            cost_weight=config.get("cost_weight", 1.0),
            waste_weight=config.get("waste_weight", 10.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of parameters.
        """
        return {
            "time_limit": self.time_limit,
            "num_days": self.num_days,
            "discrete_levels": self.discrete_levels,
            "max_fill_rate": self.max_fill_rate,
            "max_nodes": self.max_nodes,
            "discount_factor": self.discount_factor,
            "overflow_penalty": self.overflow_penalty,
            "cost_weight": self.cost_weight,
            "waste_weight": self.waste_weight,
        }

"""Learned and economic packing selection configs.

Moved from ``logic.src.configs.policies.other.packing``.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MandatoryManagerSelectionConfig:
    """Configuration for neural network-based selection (MandatoryManager).

    Attributes:
        hidden_dim: Dimension of the hidden layers.
        lstm_hidden: Dimension of the LSTM hidden layers.
        history_length: Length of the history.
        manager_weights: Path to the manager weights.
        device: Device to use for training.
    """

    hidden_dim: int = 128
    lstm_hidden: int = 64
    history_length: int = 10
    manager_weights: Optional[str] = None
    device: str = "cuda"


@dataclass
class KnapsackSelectionConfig:
    """Base configuration for knapsack-based economic coupling.

    Attributes:
        n_vehicles: Number of vehicles.
        cost_per_km: Cost per kilometer.
        use_eoq_threshold: Whether to use EOQ threshold.
    """

    n_vehicles: Optional[int] = None
    cost_per_km: Optional[float] = None
    use_eoq_threshold: Optional[bool] = None
    holding_cost_per_kg_day: Optional[float] = None
    ordering_cost_per_visit: Optional[float] = None


@dataclass
class MIPKnapsackSelectionConfig(KnapsackSelectionConfig):
    """Configuration for exact 0/1 multiple-knapsack MIP selection.

    Attributes:
        overflow_penalty_frac: Additional penalty for overflow event.
    """

    overflow_penalty_frac: Optional[float] = None


@dataclass
class FractionalKnapsackSelectionConfig(KnapsackSelectionConfig):
    """Configuration for greedy net-profit density selection.

    Attributes:
        revenue_kg: Revenue per kilogram.
        bin_capacity: Capacity of the bin.
        holding_cost_per_kg_day: Holding cost per kilogram per day.
        ordering_cost_per_visit: Ordering cost per visit.
    """

    revenue_kg: Optional[float] = None
    bin_capacity: Optional[float] = None
    holding_cost_per_kg_day: Optional[float] = None
    ordering_cost_per_visit: Optional[float] = None


@dataclass
class RolloutSelectionConfig:
    """Configuration for rollout-based predictive selection.

    Attributes:
        rollout_horizon: Horizon for rollout.
        rollout_base_policy: Base policy for rollout.
        rollout_n_scenarios: Number of scenarios for rollout.
        rollout_discount: Discount factor for rollout.
    """

    rollout_horizon: int = 5
    rollout_base_policy: str = "last_minute"
    rollout_n_scenarios: int = 1
    rollout_discount: float = 0.95


@dataclass
class WhittleSelectionConfig:
    """Configuration for Whittle index-based allocation.

    Attributes:
        whittle_discount: Discount factor for Whittle index.
        whittle_grid_size: Grid size for Whittle index.
        n_vehicles: Number of vehicles.
    """

    whittle_discount: float = 0.95
    whittle_grid_size: int = 21
    n_vehicles: int = 1


@dataclass
class CVaRSelectionConfig:
    """Configuration for Conditional Value at Risk selection.

    Attributes:
        cvar_alpha: Confidence level for CVaR.
        threshold: Threshold for CVaR selection.
    """

    cvar_alpha: float = 0.95
    threshold: float = 0.0


@dataclass
class SavingsSelectionConfig:
    """Configuration for savings-based (Clarke-Wright) pre-selection.

    Attributes:
        savings_min_fill_ratio: Minimum fill ratio for savings selection.
    """

    savings_min_fill_ratio: float = 0.5


@dataclass
class SetCoverSelectionConfig:
    """Configuration for set-covering based selection.

    Attributes:
        service_radius: Radius for set cover.
        critical_threshold: Threshold for critical set cover.
    """

    service_radius: float = 5.0
    critical_threshold: float = 0.90


@dataclass
class ModularGreedySelectionConfig:
    """Configuration for (Super/Sub-)Modular greedy selection.

    Attributes:
        modular_alpha: Alpha parameter for modularity.
        modular_budget: Budget for modular selection.
    """

    modular_alpha: float = 1.0
    modular_budget: int = 0


@dataclass
class LearnedSelectionConfig:
    """Configuration for imitation/learned selection models.

    Attributes:
        learned_model_path: Path to the learned model.
        threshold: Threshold for learned selection.
    """

    learned_model_path: Optional[str] = None
    threshold: float = 0.5

"""Threshold and heuristic packing selection configs.

Moved from ``logic.src.configs.policies.other.packing``.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LastMinuteSelectionConfig:
    """Configuration for threshold-based last-minute selection.

    1.0.0 last-minute is fill-threshold only (RQ-cursor-3). EOQ last-minute
    is the unreleased ``eoq_last_minute`` key.

    Attributes:
        threshold: Fill level in percent (0-100) at or above which a bin is mandatory
            (one unit for the simulator and the vectorized selector).
    """

    threshold: float = 70.0


@dataclass
class EOQLastMinuteSelectionConfig:
    """Configuration for unreleased EOQ last-minute (excluded from 1.0.0).

    Attributes:
        threshold: Fallback fill trigger when EOQ parameters are missing.
        holding_cost_per_kg_day: Holding cost ``h`` in the Harris EOQ formula.
        ordering_cost_per_visit: Ordering cost ``S`` per dispatch.
        bin_volume: Bin volume used to recover mass capacity.
        bin_density: Waste density used to recover mass capacity.
    """

    threshold: float = 90.0
    holding_cost_per_kg_day: float = 0.02
    ordering_cost_per_visit: float = 50.0
    bin_volume: float = 2.5
    bin_density: float = 400.0


@dataclass
class RegularSelectionConfig:
    """Configuration for periodic collection on scheduled days.

    Attributes:
        threshold: Collection period in days (every N days).
    """

    threshold: int = 3


@dataclass
class ServiceLevelSelectionConfig:
    """Configuration for statistical overflow prediction.

    Attributes:
        threshold: Confidence factor for statistical overflow prediction.
        horizon_days: Days projected ahead (1 = SL1, 2 = SL2 in ``pk_service_level.yaml``).
        estimates: ``full`` estimates the increments from the full deposits (the reported rule); ``sensor`` from
            the capped level changes a sensor reveals, which miss the waste lost to overflow.
    """

    threshold: float = 1.0
    horizon_days: int = 1
    estimates: str = "full"


@dataclass
class RevenueSelectionConfig:
    """Configuration for revenue-based selection.

    Attributes:
        revenue_kg: Revenue per kilogram.
        bin_capacity: Capacity of the bin.
        revenue_threshold: Threshold for revenue-based selection.
    """

    revenue_kg: Optional[float] = None
    bin_capacity: float = 1.0
    revenue_threshold: Optional[float] = None


@dataclass
class LookaheadSelectionConfig:
    """Configuration for predictive collection within a horizon.

    Attributes:
        current_collection_day: Current day in the collection cycle.
        estimates: ``full`` (the reported rule) or ``sensor``, as for the service-level rule.
    """

    current_collection_day: int = 0
    estimates: str = "full"


@dataclass
class DeadlineSelectionConfig:
    """Configuration for deterministic days-to-overflow selection.

    Attributes:
        horizon_days: Number of days to look ahead.
        threshold: Threshold for deadline selection.
    """

    horizon_days: int = 3
    threshold: float = 0.9


@dataclass
class MultiDayProbSelectionConfig:
    """Configuration for stochastic overflow risk over K days.

    Attributes:
        horizon_days: Number of days to look ahead.
        threshold: Threshold for multi-day probability selection.
    """

    horizon_days: int = 3
    threshold: float = 0.5


@dataclass
class ParetoFrontSelectionConfig:
    """Configuration for multi-objective (urgency x distance) optimization.

    Attributes:
        threshold: Threshold for Pareto front selection.
    """

    threshold: float = 0.5


@dataclass
class ProfitPerKmSelectionConfig:
    """Configuration for spatial ROI (Expected revenue / distance).

    Attributes:
        threshold: Threshold for profit per km selection.
        revenue_kg: Revenue per kilogram.
    """

    threshold: float = 0.0
    revenue_kg: float = 1.0


@dataclass
class SpatialSynergySelectionConfig:
    """Configuration for critical bins + opportunistic neighbors.

    Attributes:
        critical_threshold: Threshold for critical bins.
        synergy_threshold: Threshold for synergistic bins.
        radius: Radius for opportunistic neighbors.
    """

    critical_threshold: float = 0.90
    synergy_threshold: float = 0.60
    radius: float = 10.0


@dataclass
class StochasticRegretSelectionConfig:
    """Configuration for expected overflow volume minimization.

    Attributes:
        threshold: Threshold for stochastic regret selection.
    """

    threshold: float = 0.1


@dataclass
class CombinedSelectionConfig:
    """Configuration for combining multiple strategies with OR/AND logic.

    Attributes:
        combined_strategies: List of strategies to combine.
        logic: Logic to combine the strategies (or/and).
    """

    combined_strategies: Optional[List[Dict[str, Any]]] = None
    logic: str = "or"

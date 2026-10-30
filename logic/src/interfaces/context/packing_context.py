"""
Selection Context Module.

This module defines the `PackingContext` data class, which serves as a
container for all relevant data needed by selection strategies to make decisions.

Attributes:
    PackingContext: Context container for selection strategy inputs.

Example:
    >>> from logic.src.policies.mandatory.base.packing_context import PackingContext
    >>> ctx = PackingContext(bin_ids=np.array([1, 2]), current_fill=np.array([0.8, 0.5]))
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class PackingContext:
    """Context container for all potential inputs required by selection strategies.

    Attributes:
        bin_ids (NDArray[np.int32]): Array of unique bin identifiers.
        current_fill (NDArray[np.float64]): Current waste fill levels of the bins.
        accumulation_rates (Optional[NDArray[np.float64]]): Daily waste accumulation rates.
        std_deviations (Optional[NDArray[np.float64]]): Standard deviations of accumulation.
        current_day (int): The current simulation day.
        threshold (float): Selection threshold (e.g., fill level percentage).
        next_collection_day (Optional[int]): Predicted next collection day.
        distance_matrix (Optional[NDArray[Any]]): Matrix of distances between nodes.
        paths_between_states (Optional[List[List[List[int]]]]): Precomputed paths.
        vehicle_capacity (float): Total mass capacity of the collection vehicle.
        revenue_kg (float): Revenue generated per kg of waste collected.
        bin_density (float): Average density of waste in the bins.
        bin_volume (float): Total volume of each bin.
        max_fill (float): Maximum possible fill level (typically 100).
        overflow_penalty_frac (float): Fraction of revenue lost on overflow.
        coordinates (Optional[NDArray[np.float64]]): GPS coordinates of bins.
        current_collection_day (int): Today's day in the collection cycle.
        seed (Optional[int]): Random seed for stochastic strategies.

    Example:
        >>> ctx = PackingContext(bin_ids=np.array([0, 1]), current_fill=np.array([80, 20]))
    """

    bin_ids: NDArray[np.int32]
    current_fill: NDArray[np.float64]
    accumulation_rates: Optional[NDArray[np.float64]] = None
    std_deviations: Optional[NDArray[np.float64]] = None
    current_day: int = 0
    threshold: float = 0.0
    next_collection_day: Optional[int] = None
    distance_matrix: Optional[NDArray[Any]] = None
    paths_between_states: Optional[List[List[List[int]]]] = None
    vehicle_capacity: float = 0.0
    revenue_kg: float = 0.0
    bin_density: float = 0.0
    bin_volume: float = 0.0
    max_fill: float = 100.0
    overflow_penalty_frac: float = 1.0  # default to 100% of bin waste capacity
    coordinates: Optional[NDArray[np.float64]] = None
    current_collection_day: int = 0
    seed: Optional[int] = None

    # --- New fields for advanced strategies ---
    # DeadlineDrivenSelection / MultiDayOverflowSelection
    horizon_days: int = 3
    # SpatialSynergySelection
    critical_threshold: float = 0.90
    synergy_threshold: float = 0.60
    radius: float = 10.0

    # --- Knapsack / economic coupling ---
    n_vehicles: int = 1
    cost_per_km: float = 0.0
    use_eoq_threshold: bool = False
    holding_cost_per_kg_day: float = 0.0
    ordering_cost_per_visit: float = 0.0

    # --- Rollout strategy ---
    rollout_horizon: int = 5
    rollout_base_policy: str = "last_minute"
    rollout_n_scenarios: int = 1  # 1 = deterministic rollout; >1 = Monte Carlo
    rollout_discount: float = 0.95

    # --- Whittle index ---
    whittle_discount: float = 0.95
    whittle_grid_size: int = 21  # number of subsidy values probed for index computation

    # --- CVaR ---
    cvar_alpha: float = 0.95  # tail level; strategy compares CVaR to context.threshold

    # --- Savings / Clarke-Wright ---
    savings_min_fill_ratio: float = 0.5

    # --- Regular with a fill filter (threshold Regular) ---
    min_fill: float = 0.0  # percent of capacity; scheduled days select only bins at or above it

    # --- Set-cover ---
    service_radius: float = 5.0

    # --- (Super/Sub-)Modular greedy ---
    modular_alpha: float = 1.0  # trade-off between revenue and route length
    modular_budget: int = 0  # 0 = unbounded; otherwise cardinality cap

    # --- Learned / imitation selection ---
    learned_model_path: Optional[str] = None
    learned_threshold: float = 0.5  # classification cutoff

    # --- Contextual Thompson sampling dispatcher ---
    dispatcher_state_path: Optional[str] = None  # pickle of posterior params
    dispatcher_candidate_strategies: Optional[List[str]] = None
    dispatcher_exploration: float = 1.0  # posterior-sample temperature
    dispatcher_mode: str = "union"

    # --- Distributionally robust ---
    wasserstein_radius: float = 0.1
    wasserstein_p: int = 1  # only p=1 is implemented

    # --- Stochastic scenario tree ---
    # Optional ScenarioTree (pipeline.simulations.bins.prediction.ScenarioTree) injected
    # by the simulation engine before calling pack().  Strategies that do not
    # consume it simply ignore the field.
    scenario_tree: Optional[Any] = None

    # --- Filter-and-Fan ---
    ff_filter_width: int = 0  # 0 = auto (max(5, n_bins // 3))
    ff_fan_depth: int = 3  # number of add/remove sweep passes

    # --- Execution Trace ---
    trace: Optional[Any] = None
    trace_parent_id: Optional[str] = None

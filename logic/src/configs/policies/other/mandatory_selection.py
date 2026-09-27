"""
Mandatory Selection Config module.

Configures the strategies used to determine which bins must be collected
during simulation or training. This module follows the modular pattern of
nested dataclasses for each algorithm.

Attributes:
    last_minute (LastMinuteSelectionConfig): Configuration for last-minute selection.
    regular (RegularSelectionConfig): Configuration for regular selection.
    service_level (ServiceLevelSelectionConfig): Configuration for service-level selection.
    revenue (RevenueSelectionConfig): Configuration for revenue-based selection.
    lookahead (LookaheadSelectionConfig): Configuration for lookahead selection.
    deadline (DeadlineSelectionConfig): Configuration for deadline-based selection.
    multi_day_prob (MultiDayProbSelectionConfig): Configuration for multi-day probability selection.
    pareto_front (ParetoFrontSelectionConfig): Configuration for Pareto front selection.
    profit_per_km (ProfitPerKmSelectionConfig): Configuration for profit per km selection.
    spatial_synergy (SpatialSynergySelectionConfig): Configuration for spatial synergy selection.
    stochastic_regret (StochasticRegretSelectionConfig): Configuration for stochastic regret selection.
    combined (CombinedSelectionConfig): Configuration for combined selection.
    mandatory_manager (MandatoryManagerSelectionConfig): Configuration for neural network-based selection.
    knapsack (KnapsackSelectionConfig): Base configuration for knapsack-based economic coupling.
    mip_knapsack (MIPKnapsackSelectionConfig): Configuration for exact 0/1 multiple-knapsack MIP selection.
    fractional_knapsack (FractionalKnapsackSelectionConfig): Configuration for greedy net-profit density selection.
    rollout (RolloutSelectionConfig): Configuration for rollout-based predictive selection.
    whittle (WhittleSelectionConfig): Configuration for Whittle index-based allocation.
    cvar (CVaRSelectionConfig): Configuration for Conditional Value at Risk selection.
    savings (SavingsSelectionConfig): Configuration for savings-based pre-selection.
    set_cover (SetCoverSelectionConfig): Configuration for set-covering based selection.
    modular_greedy (ModularGreedySelectionConfig): Configuration for modular greedy selection.
    learned (LearnedSelectionConfig): Configuration for learned selection models.
    thompson_dispatcher (ThompsonDispatcherSelectionConfig): Configuration for contextual Thompson sampling dispatcher.
    wasserstein (WassersteinSelectionConfig): Configuration for distributionally robust Wasserstein selection.
    lagrangian (LagrangianSelectionConfig): Configuration for Lagrangian relaxation based selection.
    bernoulli (BernoulliSelectionConfig): Configuration for independent Bernoulli trial selection.
    kmeans_sector (KMeansSectorSelectionConfig): Configuration for geographic sector-based selection.
    staggered_regular (StaggeredRegularSelectionConfig): Configuration for staggered regular selection.
    method: The method to be used for selection.

Example:
    >>> from configs.policies.other.mandatory_selection import MandatorySelectionConfig
    >>> config = MandatorySelectionConfig(method='last_minute', last_minute=LastMinuteSelectionConfig(threshold=80))
    >>> config.method
    'last_minute'
    >>> config.last_minute.threshold
    80
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LastMinuteSelectionConfig:
    """Configuration for threshold-based last-minute selection.

    Attributes:
        threshold: Fill level in percent (0-100) at or above which a bin is mandatory
            (one unit for the simulator and the vectorized selector).
    """

    threshold: float = 70.0


@dataclass
class RegularSelectionConfig:
    """Configuration for periodic collection on scheduled days.

    Attributes:
        frequency: Frequency of regular selection.
    """

    frequency: int = 3


@dataclass
class ServiceLevelSelectionConfig:
    """Configuration for statistical overflow prediction.

    Attributes:
        confidence_factor: Confidence factor for statistical overflow prediction.
        horizon_days: Days projected ahead (1 = SL1, 2 = SL2 in ``ms_service_level.yaml``).
    """

    confidence_factor: float = 1.0
    horizon_days: int = 1


@dataclass
class RevenueSelectionConfig:
    """Configuration for revenue-based selection.

    Attributes:
        revenue_kg: Revenue per kilogram.
        bin_capacity: Capacity of the bin.
        revenue_threshold: Threshold for revenue-based selection.
    """

    revenue_kg: float = 1.0
    bin_capacity: float = 1.0
    revenue_threshold: float = 0.0


@dataclass
class LookaheadSelectionConfig:
    """Configuration for predictive collection within a horizon.

    Attributes:
        current_collection_day: Current day in the collection cycle.
    """

    current_collection_day: int = 0


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
        strategies: List of strategies to combine.
        logic: Logic to combine the strategies (or/and).
    """

    strategies: Optional[List[Dict[str, Any]]] = None
    logic: str = "or"


@dataclass
class MandatoryManagerSelectionConfig:
    """Configuration for neural network-based selection (MandatoryManager).

    Attributes:
        hidden_dim: Dimension of the hidden layers.
        lstm_hidden: Dimension of the LSTM hidden layers.
        history_length: Length of the history.
        manager_critical_threshold: Threshold for manager selection.
        manager_weights: Path to the manager weights.
        device: Device to use for training.
    """

    hidden_dim: int = 128
    lstm_hidden: int = 64
    history_length: int = 10
    manager_critical_threshold: float = 0.9
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

    n_vehicles: int = 1
    cost_per_km: float = 0.1
    use_eoq_threshold: bool = False
    holding_cost_per_kg_day: float = 0.0
    ordering_cost_per_visit: float = 0.0


@dataclass
class MIPKnapsackSelectionConfig(KnapsackSelectionConfig):
    """Configuration for exact 0/1 multiple-knapsack MIP selection.

    Attributes:
        overflow_penalty_frac: Additional penalty for overflow event.
    """

    overflow_penalty_frac: float = 1.0


@dataclass
class FractionalKnapsackSelectionConfig(KnapsackSelectionConfig):
    """Configuration for greedy net-profit density selection.

    Attributes:
        revenue_kg: Revenue per kilogram.
        bin_capacity: Capacity of the bin.
        holding_cost_per_kg_day: Holding cost per kilogram per day.
        ordering_cost_per_visit: Ordering cost per visit.
    """

    revenue_kg: float = 1.0
    bin_capacity: float = 1.0
    holding_cost_per_kg_day: float = 0.01
    ordering_cost_per_visit: float = 10.0


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
        learned_threshold: Threshold for learned selection.
    """

    learned_model_path: Optional[str] = None
    learned_threshold: float = 0.5


@dataclass
class ThompsonDispatcherSelectionConfig:
    """Configuration for contextual Thompson sampling dispatcher.

    Attributes:
        dispatcher_state_path: Path to the dispatcher state.
        dispatcher_candidate_strategies: List of candidate strategies.
        dispatcher_exploration: Exploration parameter for Thompson sampling.
        dispatcher_mode: Mode for combining strategies (union/intersection).
    """

    dispatcher_state_path: Optional[str] = None
    dispatcher_candidate_strategies: Optional[List[str]] = None
    dispatcher_exploration: float = 1.0
    dispatcher_mode: str = "union"


@dataclass
class WassersteinSelectionConfig:
    """Configuration for distributionally robust Wasserstein selection.

    Attributes:
        wasserstein_radius: Radius for Wasserstein distance.
        wasserstein_p: Power parameter for Wasserstein distance.
    """

    wasserstein_radius: float = 0.1
    wasserstein_p: int = 1


@dataclass
class LagrangianSelectionConfig:
    """Configuration for Lagrangian relaxation based selection.

    Attributes:
        n_vehicles: Number of vehicles.
        cost_per_km: Cost per kilometer.
    """

    n_vehicles: int = 1
    cost_per_km: float = 0.1


@dataclass
class BernoulliSelectionConfig:
    """Configuration for independent Bernoulli trial selection.

    Attributes:
        p: Probability of selecting a bin.
    """

    p: float = 0.5


@dataclass
class KMeansSectorSelectionConfig:
    """Configuration for geographic sector-based selection.

    Attributes:
        n_sectors: Number of sectors for clustering.
    """

    n_sectors: int = 5


@dataclass
class StaggeredRegularSelectionConfig:
    """Configuration for phased-staggered periodic selection.

    Attributes:
        period: Period for staggered selection.
    """

    period: int = 1


@dataclass
class MandatorySelectionConfig:
    """Main configuration for mandatory bin selection.

    Composes algorithm-specific parameters and execution settings.

    Attributes:
        strategy: Primary selection strategy name.
        last_minute: Params for LastMinuteSelection.
        regular: Params for RegularSelection.
        service_level: Params for ServiceLevelSelection.
        revenue: Params for RevenueSelection.
        lookahead: Params for LookaheadSelection.
        deadline: Params for DeadlineDrivenSelection.
        multi_day_prob: Params for MultiDayOverflowSelection.
        pareto_front: Params for ParetoFrontSelection.
        profit_per_km: Params for ProfitPerKmSelection.
        spatial_synergy: Params for SpatialSynergySelection.
        stochastic_regret: Params for StochasticRegretSelection.
        combined: Params for CombinedSelection.
        manager: Params for neural MandatoryManager.
        mip_knapsack: Params for MIPKnapsackSelection.
        fractional_knapsack: Params for FractionalKnapsackSelection.
        rollout: Params for RolloutSelection.
        whittle: Params for WhittleIndexSelection.
        cvar: Params for CVaRSelection.
        savings: Params for SavingsSelection.
        set_cover: Params for SetCoverSelection.
        modular_greedy: Params for ModularGreedySelection.
        learned: Params for LearnedSelection.
        thompson_dispatcher: Params for ThompsonDispatcher.
        wasserstein: Params for WassersteinSelection.
        lagrangian: Params for LagrangianSelection.
        params: Additional strategy-specific parameters as a dictionary.
    """

    strategy: Optional[str] = None

    # Strategy-specific sub-configs
    last_minute: LastMinuteSelectionConfig = field(default_factory=LastMinuteSelectionConfig)
    regular: RegularSelectionConfig = field(default_factory=RegularSelectionConfig)
    service_level: ServiceLevelSelectionConfig = field(default_factory=ServiceLevelSelectionConfig)
    revenue: RevenueSelectionConfig = field(default_factory=RevenueSelectionConfig)
    lookahead: LookaheadSelectionConfig = field(default_factory=LookaheadSelectionConfig)
    deadline: DeadlineSelectionConfig = field(default_factory=DeadlineSelectionConfig)
    multi_day_prob: MultiDayProbSelectionConfig = field(default_factory=MultiDayProbSelectionConfig)
    pareto_front: ParetoFrontSelectionConfig = field(default_factory=ParetoFrontSelectionConfig)
    profit_per_km: ProfitPerKmSelectionConfig = field(default_factory=ProfitPerKmSelectionConfig)
    spatial_synergy: SpatialSynergySelectionConfig = field(default_factory=SpatialSynergySelectionConfig)
    stochastic_regret: StochasticRegretSelectionConfig = field(default_factory=StochasticRegretSelectionConfig)
    combined: CombinedSelectionConfig = field(default_factory=CombinedSelectionConfig)
    manager: MandatoryManagerSelectionConfig = field(default_factory=MandatoryManagerSelectionConfig)
    mip_knapsack: MIPKnapsackSelectionConfig = field(default_factory=MIPKnapsackSelectionConfig)
    fractional_knapsack: FractionalKnapsackSelectionConfig = field(default_factory=FractionalKnapsackSelectionConfig)
    rollout: RolloutSelectionConfig = field(default_factory=RolloutSelectionConfig)
    whittle: WhittleSelectionConfig = field(default_factory=WhittleSelectionConfig)
    cvar: CVaRSelectionConfig = field(default_factory=CVaRSelectionConfig)
    savings: SavingsSelectionConfig = field(default_factory=SavingsSelectionConfig)
    set_cover: SetCoverSelectionConfig = field(default_factory=SetCoverSelectionConfig)
    modular_greedy: ModularGreedySelectionConfig = field(default_factory=ModularGreedySelectionConfig)
    learned: LearnedSelectionConfig = field(default_factory=LearnedSelectionConfig)
    thompson_dispatcher: ThompsonDispatcherSelectionConfig = field(default_factory=ThompsonDispatcherSelectionConfig)
    wasserstein: WassersteinSelectionConfig = field(default_factory=WassersteinSelectionConfig)
    lagrangian: LagrangianSelectionConfig = field(default_factory=LagrangianSelectionConfig)
    bernoulli: BernoulliSelectionConfig = field(default_factory=BernoulliSelectionConfig)
    kmeans_sector: KMeansSectorSelectionConfig = field(default_factory=KMeansSectorSelectionConfig)
    staggered_regular: StaggeredRegularSelectionConfig = field(default_factory=StaggeredRegularSelectionConfig)

    # Additional parameters
    params: Dict[str, Any] = field(default_factory=dict)

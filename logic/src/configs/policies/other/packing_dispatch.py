"""Dispatcher and special packing selection configs.

Moved from ``logic.src.configs.policies.other.packing``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .packing_learned import (
    CVaRSelectionConfig,
    FractionalKnapsackSelectionConfig,
    LearnedSelectionConfig,
    MandatoryManagerSelectionConfig,
    MIPKnapsackSelectionConfig,
    ModularGreedySelectionConfig,
    RolloutSelectionConfig,
    SavingsSelectionConfig,
    SetCoverSelectionConfig,
    WhittleSelectionConfig,
)
from .packing_threshold import (
    CombinedSelectionConfig,
    DeadlineSelectionConfig,
    EOQLastMinuteSelectionConfig,
    LastMinuteSelectionConfig,
    LookaheadSelectionConfig,
    MultiDayProbSelectionConfig,
    ParetoFrontSelectionConfig,
    ProfitPerKmSelectionConfig,
    RegularSelectionConfig,
    RevenueSelectionConfig,
    ServiceLevelSelectionConfig,
    SpatialSynergySelectionConfig,
    StochasticRegretSelectionConfig,
)


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
class KMeansSectorSelectionConfig:
    """Configuration for geographic sector-based selection.

    Attributes:
        threshold: Number of sectors for clustering.
    """

    threshold: int = 5


@dataclass
class StaggeredRegularSelectionConfig:
    """Configuration for phased-staggered periodic selection.

    Attributes:
        threshold: Period for staggered selection.
    """

    threshold: int = 1


@dataclass
class BernoulliRandomSelectionConfig:
    """Configuration for independent Bernoulli trial selection.

    Attributes:
        p: Selection probability in [0, 1]; None uses the context threshold.
        min_fill: Minimum fill ratio for a bin to enter the trial.
    """

    p: Optional[float] = None
    min_fill: float = 0.0


@dataclass
class FilterAndFanSelectionConfig:
    """Configuration for the filter-and-fan selection.

    Attributes:
        filter_width: Seed size; None uses the context value.
        fan_depth: Add/remove passes; None uses the context value.
    """

    filter_width: Optional[int] = None
    fan_depth: Optional[int] = None


@dataclass
class FPTASKnapsackSelectionConfig:
    """Configuration for the FPTAS knapsack selection.

    Attributes:
        epsilon: Precision parameter.
        alpha: Weight between overflow risk and net profit in the scalarisation.
        overflow_penalty_frac: Overflow penalty fraction; None uses the context value.
    """

    epsilon: float = 0.1
    alpha: float = 0.5
    overflow_penalty_frac: Optional[float] = None


@dataclass
class PortfolioDispatcherSelectionConfig:
    """Configuration for the portfolio dispatcher.

    Attributes:
        dispatcher_mode: 'union' or 'intersection'; None uses the context value.
        dispatcher_candidate_strategies: Candidate strategy names; None uses the context value.
    """

    dispatcher_mode: Optional[str] = None
    dispatcher_candidate_strategies: Optional[List[str]] = None


@dataclass
class PackingConfig:
    """Main configuration for mandatory bin selection.

    Composes algorithm-specific parameters and execution settings.

    Attributes:
        strategy: Primary packing algorithm name.
        last_minute: Params for LastMinuteSelection.
        eoq_last_minute: Params for unreleased LastMinuteEOQSelection.
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
    eoq_last_minute: EOQLastMinuteSelectionConfig = field(default_factory=EOQLastMinuteSelectionConfig)
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
    kmeans_sector: KMeansSectorSelectionConfig = field(default_factory=KMeansSectorSelectionConfig)
    staggered_regular: StaggeredRegularSelectionConfig = field(default_factory=StaggeredRegularSelectionConfig)
    bernoulli_random: BernoulliRandomSelectionConfig = field(default_factory=BernoulliRandomSelectionConfig)
    filter_and_fan: FilterAndFanSelectionConfig = field(default_factory=FilterAndFanSelectionConfig)
    fptas_knapsack: FPTASKnapsackSelectionConfig = field(default_factory=FPTASKnapsackSelectionConfig)
    portfolio_dispatcher: PortfolioDispatcherSelectionConfig = field(default_factory=PortfolioDispatcherSelectionConfig)

    # Additional parameters
    params: Dict[str, Any] = field(default_factory=dict)

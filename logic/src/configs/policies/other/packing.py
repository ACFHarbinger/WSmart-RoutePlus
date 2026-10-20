"""

Per-strategy packing dataclasses live in ``packing_threshold``, ``packing_learned`` and ``packing_dispatch``; this module keeps ``PackingConfig`` and re-exports those names.
Packing Config module.

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
    >>> from configs.policies.other.packing import PackingConfig
    >>> config = PackingConfig(method='last_minute', last_minute=LastMinuteSelectionConfig(threshold=80))
    >>> config.method
    'last_minute'
    >>> config.last_minute.threshold
    80
"""


from .packing_dispatch import (
    BernoulliRandomSelectionConfig as BernoulliRandomSelectionConfig,
)
from .packing_dispatch import (
    FilterAndFanSelectionConfig as FilterAndFanSelectionConfig,
)
from .packing_dispatch import (
    FPTASKnapsackSelectionConfig as FPTASKnapsackSelectionConfig,
)
from .packing_dispatch import (
    KMeansSectorSelectionConfig as KMeansSectorSelectionConfig,
)
from .packing_dispatch import (
    LagrangianSelectionConfig as LagrangianSelectionConfig,
)
from .packing_dispatch import (
    PackingConfig as PackingConfig,
)
from .packing_dispatch import (
    PortfolioDispatcherSelectionConfig as PortfolioDispatcherSelectionConfig,
)
from .packing_dispatch import (
    StaggeredRegularSelectionConfig as StaggeredRegularSelectionConfig,
)
from .packing_dispatch import (
    ThompsonDispatcherSelectionConfig as ThompsonDispatcherSelectionConfig,
)
from .packing_dispatch import (
    WassersteinSelectionConfig as WassersteinSelectionConfig,
)
from .packing_learned import (
    CVaRSelectionConfig as CVaRSelectionConfig,
)
from .packing_learned import (
    FractionalKnapsackSelectionConfig as FractionalKnapsackSelectionConfig,
)
from .packing_learned import (
    KnapsackSelectionConfig as KnapsackSelectionConfig,
)
from .packing_learned import (
    LearnedSelectionConfig as LearnedSelectionConfig,
)
from .packing_learned import (
    MandatoryManagerSelectionConfig as MandatoryManagerSelectionConfig,
)
from .packing_learned import (
    MIPKnapsackSelectionConfig as MIPKnapsackSelectionConfig,
)
from .packing_learned import (
    ModularGreedySelectionConfig as ModularGreedySelectionConfig,
)
from .packing_learned import (
    RolloutSelectionConfig as RolloutSelectionConfig,
)
from .packing_learned import (
    SavingsSelectionConfig as SavingsSelectionConfig,
)
from .packing_learned import (
    SetCoverSelectionConfig as SetCoverSelectionConfig,
)
from .packing_learned import (
    WhittleSelectionConfig as WhittleSelectionConfig,
)
from .packing_threshold import (
    CombinedSelectionConfig as CombinedSelectionConfig,
)
from .packing_threshold import (
    DeadlineSelectionConfig as DeadlineSelectionConfig,
)
from .packing_threshold import (
    EOQLastMinuteSelectionConfig as EOQLastMinuteSelectionConfig,
)
from .packing_threshold import (
    LastMinuteSelectionConfig as LastMinuteSelectionConfig,
)
from .packing_threshold import (
    LookaheadSelectionConfig as LookaheadSelectionConfig,
)
from .packing_threshold import (
    MultiDayProbSelectionConfig as MultiDayProbSelectionConfig,
)
from .packing_threshold import (
    ParetoFrontSelectionConfig as ParetoFrontSelectionConfig,
)
from .packing_threshold import (
    ProfitPerKmSelectionConfig as ProfitPerKmSelectionConfig,
)
from .packing_threshold import (
    RegularSelectionConfig as RegularSelectionConfig,
)
from .packing_threshold import (
    RevenueSelectionConfig as RevenueSelectionConfig,
)
from .packing_threshold import (
    ServiceLevelSelectionConfig as ServiceLevelSelectionConfig,
)
from .packing_threshold import (
    SpatialSynergySelectionConfig as SpatialSynergySelectionConfig,
)
from .packing_threshold import (
    StochasticRegretSelectionConfig as StochasticRegretSelectionConfig,
)



































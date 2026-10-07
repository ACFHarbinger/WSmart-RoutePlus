"""Selection Strategy Implementations for WSmart-Route.

This package contains concrete implementations of the `MandatorySelectionStrategy`
interface. Strategies determine which bins must be collected on a given day
based on different criteria (fill levels, revenue, schedule, etc.).

Attributes:
    MandatorySelectionFactory: Factory for creating strategy instances.
    MandatorySelectionRegistry: Registry for mapping names to strategy classes.
    SelectionContext: Context object containing state for selection decisions.
    CombinedSelection: Strategy that combines multiple selection criteria.
    LastMinuteSelection: Simple threshold-based reactive strategy.
    RevenueThresholdSelection: Strategy based on estimated revenue.
    LookaheadSelection: Predictive strategy using future fill simulations.
    WhittleIndexSelection: RMAB-based priority ranking strategy.
    MIPKnapsackSelection: Exact multiple-knapsack optimization strategy.
    FPTASKnapsackSelection: FPTAS-based multiple-knapsack strategy.
    BernoulliRandomSelection: Stochastic selection with eligibility thresholds.
    CVaRSelection: Tail-risk based selection.
    DeadlineDrivenSelection: Temporal deadline-based selection.
    PortfolioDispatcher: Orchestrator for multiple concurrent strategies.
    ThompsonDispatcher: MAB-based adaptive strategy selection.
    FilterAndFanSelection: Local search based selection refinement.
    FractionalKnapsackSelection: Greedy net-profit density selection.
    KMeansGeographicSectorSelection: Cyclic zone-day selection.
    LagrangianSelection: Reduced-cost based selection.
    LearnedSelection: ML-imitation selection strategy.
    MultiDayOverflowSelection: Stochastic multi-period overflow probability.
    ParetoFrontSelection: Bi-objective urgency/cost selection.
    ProfitPerKmSelection: ROI-based selection proxy.
    RegularSelection: Fixed-frequency periodic selection.
    RolloutSelection: One-step rollout simulation strategy.
    SavingsSelection: Spatial-savings based selection.
    ServiceLevelSelection: Statistical confidence-bound selection.
    SetCoverSelection: Hub-based spatial coverage selection.
    SpatialSynergySelection: Neighbourhood-based opportunistic selection.
    StaggeredRegularSelection: Phase-staggered periodic selection.
    StochasticRegretSelection: Expected overflow regret selection.
    SubmodularGreedySelection: Facility-location coverage selection.
    SupermodularGreedySelection: Synergetic cluster selection.
    WassersteinRobustSelection: Distributionally robust selection.

Example:
    >>> from logic.src.policies.mandatory_selection import MandatorySelectionFactory
    >>> strategy = MandatorySelectionFactory.create_strategy("last_minute")
    >>> bins, ctx = strategy.select_bins(selection_context)
"""

from logic.src.interfaces import IMandatorySelectionStrategy
from logic.src.interfaces.context import SelectionContext
from logic.src.policies.mandatory_selection.vectorized import (
    CombinedSelector,
    LastMinuteSelector,
    LookaheadSelector,
    ManagerSelector,
    RegularSelector,
    RevenueSelector,
    ServiceLevelSelector,
    VectorizedSelector,
    create_selector_from_config,
    get_vectorized_selector,
)

from .base import MandatorySelectionFactory, MandatorySelectionRegistry
from .selection_bernoulli_random import BernoulliRandomSelection
from .selection_combined import CombinedSelection
from .selection_cvar import CVaRSelection
from .selection_deadline import DeadlineDrivenSelection
from .selection_dispatcher_portfolio import PortfolioDispatcher
from .selection_dispatcher_thompson import ThompsonDispatcher
from .selection_filter_and_fan import FilterAndFanSelection
from .selection_fptas_knapsack import FPTASKnapsackSelection
from .selection_fractional_knapsack import FractionalKnapsackSelection
from .selection_kmeans_sector import KMeansGeographicSectorSelection
from .selection_lagrangian import LagrangianSelection
from .selection_last_minute import LastMinuteSelection
from .selection_learned import LearnedSelection
from .selection_lookahead import LookaheadSelection
from .selection_mip_knapsack import MIPKnapsackSelection
from .selection_multi_day_prob import MultiDayOverflowSelection
from .selection_pareto import ParetoFrontSelection
from .selection_profit_per_km import ProfitPerKmSelection
from .selection_regular import RegularSelection
from .selection_revenue import RevenueThresholdSelection
from .selection_rollout import RolloutSelection
from .selection_savings import SavingsSelection
from .selection_service_level import ServiceLevelSelection
from .selection_set_cover import SetCoverSelection
from .selection_spatial_synergy import SpatialSynergySelection
from .selection_staggered_regular import StaggeredRegularSelection
from .selection_stochastic_regret import StochasticRegretSelection
from .selection_submodular_greedy import SubmodularGreedySelection
from .selection_supermodular_greedy import SupermodularGreedySelection
from .selection_wasserstein import WassersteinRobustSelection
from .selection_whittle import WhittleIndexSelection

__all__ = [
    # Vectorized selectors for training
    "VectorizedSelector",
    "LastMinuteSelector",
    "RegularSelector",
    "LookaheadSelector",
    "RevenueSelector",
    "ServiceLevelSelector",
    "CombinedSelector",
    "ManagerSelector",
    "get_vectorized_selector",
    "create_selector_from_config",
    "IMandatorySelectionStrategy",
    "MandatorySelectionFactory",
    "MandatorySelectionRegistry",
    "SelectionContext",
    # New simulation strategies
    "CombinedSelection",
    "LastMinuteSelection",
    "RevenueThresholdSelection",
    "DeadlineDrivenSelection",
    "MultiDayOverflowSelection",
    "ParetoFrontSelection",
    "ProfitPerKmSelection",
    "SpatialSynergySelection",
    "StochasticRegretSelection",
    "LagrangianSelection",
    "RolloutSelection",
    "WhittleIndexSelection",
    "CVaRSelection",
    "SavingsSelection",
    "SetCoverSelection",
    "SubmodularGreedySelection",
    "SupermodularGreedySelection",
    "LearnedSelection",
    "WassersteinRobustSelection",
    "ThompsonDispatcher",
    "PortfolioDispatcher",
    "RegularSelection",
    "LookaheadSelection",
    "ServiceLevelSelection",
    "FractionalKnapsackSelection",
    "MIPKnapsackSelection",
    "FilterAndFanSelection",
    "BernoulliRandomSelection",
    "KMeansGeographicSectorSelection",
    "StaggeredRegularSelection",
    "FPTASKnapsackSelection",
]

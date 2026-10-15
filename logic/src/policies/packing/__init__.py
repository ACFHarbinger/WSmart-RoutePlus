"""Packing algorithms. Public names load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

from .base import PackingFactory, PackingRegistry

_LAZY = {
    "IPackingAlgorithm": "logic.src.interfaces.packing:IPackingAlgorithm",
    "SelectionContext": "logic.src.interfaces.context:SelectionContext",
    "VectorizedSelector": "logic.src.policies.packing.vectorized:VectorizedSelector",
    "LastMinuteSelector": "logic.src.policies.packing.vectorized:LastMinuteSelector",
    "RegularSelector": "logic.src.policies.packing.vectorized:RegularSelector",
    "LookaheadSelector": "logic.src.policies.packing.vectorized:LookaheadSelector",
    "RevenueSelector": "logic.src.policies.packing.vectorized:RevenueSelector",
    "ServiceLevelSelector": "logic.src.policies.packing.vectorized:ServiceLevelSelector",
    "CombinedSelector": "logic.src.policies.packing.vectorized:CombinedSelector",
    "ManagerSelector": "logic.src.policies.packing.vectorized:ManagerSelector",
    "get_vectorized_selector": "logic.src.policies.packing.vectorized:get_vectorized_selector",
    "create_selector_from_config": "logic.src.policies.packing.vectorized:create_selector_from_config",
    "CombinedSelection": "logic.src.policies.packing.selection_combined:CombinedSelection",
    "LastMinuteSelection": "logic.src.policies.packing.selection_last_minute:LastMinuteSelection",
    "RevenueThresholdSelection": "logic.src.policies.packing.selection_revenue:RevenueThresholdSelection",
    "DeadlineDrivenSelection": "logic.src.policies.packing.selection_deadline:DeadlineDrivenSelection",
    "MultiDayOverflowSelection": "logic.src.policies.packing.selection_multi_day_prob:MultiDayOverflowSelection",
    "ParetoFrontSelection": "logic.src.policies.packing.selection_pareto:ParetoFrontSelection",
    "ProfitPerKmSelection": "logic.src.policies.packing.selection_profit_per_km:ProfitPerKmSelection",
    "SpatialSynergySelection": "logic.src.policies.packing.selection_spatial_synergy:SpatialSynergySelection",
    "StochasticRegretSelection": "logic.src.policies.packing.selection_stochastic_regret:StochasticRegretSelection",
    "LagrangianSelection": "logic.src.policies.packing.selection_lagrangian:LagrangianSelection",
    "RolloutSelection": "logic.src.policies.packing.selection_rollout:RolloutSelection",
    "WhittleIndexSelection": "logic.src.policies.packing.selection_whittle:WhittleIndexSelection",
    "CVaRSelection": "logic.src.policies.packing.selection_cvar:CVaRSelection",
    "SavingsSelection": "logic.src.policies.packing.selection_savings:SavingsSelection",
    "SetCoverSelection": "logic.src.policies.packing.selection_set_cover:SetCoverSelection",
    "SubmodularGreedySelection": "logic.src.policies.packing.selection_submodular_greedy:SubmodularGreedySelection",
    "SupermodularGreedySelection": "logic.src.policies.packing.selection_supermodular_greedy:SupermodularGreedySelection",
    "LearnedSelection": "logic.src.policies.packing.selection_learned:LearnedSelection",
    "WassersteinRobustSelection": "logic.src.policies.packing.selection_wasserstein:WassersteinRobustSelection",
    "ThompsonDispatcher": "logic.src.policies.packing.selection_dispatcher_thompson:ThompsonDispatcher",
    "PortfolioDispatcher": "logic.src.policies.packing.selection_dispatcher_portfolio:PortfolioDispatcher",
    "RegularSelection": "logic.src.policies.packing.selection_regular:RegularSelection",
    "LookaheadSelection": "logic.src.policies.packing.selection_lookahead:LookaheadSelection",
    "ServiceLevelSelection": "logic.src.policies.packing.selection_service_level:ServiceLevelSelection",
    "FractionalKnapsackSelection": "logic.src.policies.packing.selection_fractional_knapsack:FractionalKnapsackSelection",
    "MIPKnapsackSelection": "logic.src.policies.packing.selection_mip_knapsack:MIPKnapsackSelection",
    "FilterAndFanSelection": "logic.src.policies.packing.selection_filter_and_fan:FilterAndFanSelection",
    "BernoulliRandomSelection": "logic.src.policies.packing.selection_bernoulli_random:BernoulliRandomSelection",
    "KMeansGeographicSectorSelection": "logic.src.policies.packing.selection_kmeans_sector:KMeansGeographicSectorSelection",
    "StaggeredRegularSelection": "logic.src.policies.packing.selection_staggered_regular:StaggeredRegularSelection",
    "FPTASKnapsackSelection": "logic.src.policies.packing.selection_fptas_knapsack:FPTASKnapsackSelection",
}

__all__ = ["PackingFactory", "PackingRegistry", *list(_LAZY)]


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)

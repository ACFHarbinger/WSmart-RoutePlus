"""Packing algorithms. Public names load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

from .base import PackingFactory, PackingRegistry

_LAZY = {
    "IPackingAlgorithm": "logic.src.interfaces.packing:IPackingAlgorithm",
    "PackingContext": "logic.src.interfaces.context:PackingContext",
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
    "CombinedSelection": "logic.src.policies.packing.combined:CombinedSelection",
    "LastMinuteSelection": "logic.src.policies.packing.last_minute:LastMinuteSelection",
    "RevenueThresholdSelection": "logic.src.policies.packing.revenue:RevenueThresholdSelection",
    "DeadlineDrivenSelection": "logic.src.policies.packing.deadline:DeadlineDrivenSelection",
    "MultiDayOverflowSelection": "logic.src.policies.packing.multi_day_prob:MultiDayOverflowSelection",
    "ParetoFrontSelection": "logic.src.policies.packing.pareto:ParetoFrontSelection",
    "ProfitPerKmSelection": "logic.src.policies.packing.profit_per_km:ProfitPerKmSelection",
    "SpatialSynergySelection": "logic.src.policies.packing.spatial_synergy:SpatialSynergySelection",
    "StochasticRegretSelection": "logic.src.policies.packing.stochastic_regret:StochasticRegretSelection",
    "LagrangianSelection": "logic.src.policies.packing.lagrangian:LagrangianSelection",
    "RolloutSelection": "logic.src.policies.packing.rollout:RolloutSelection",
    "WhittleIndexSelection": "logic.src.policies.packing.whittle:WhittleIndexSelection",
    "CVaRSelection": "logic.src.policies.packing.cvar:CVaRSelection",
    "SavingsSelection": "logic.src.policies.packing.savings:SavingsSelection",
    "SetCoverSelection": "logic.src.policies.packing.set_cover:SetCoverSelection",
    "SubmodularGreedySelection": "logic.src.policies.packing.submodular_greedy:SubmodularGreedySelection",
    "SupermodularGreedySelection": "logic.src.policies.packing.supermodular_greedy:SupermodularGreedySelection",
    "LearnedSelection": "logic.src.policies.packing.learned:LearnedSelection",
    "WassersteinRobustSelection": "logic.src.policies.packing.wasserstein:WassersteinRobustSelection",
    "ThompsonDispatcher": "logic.src.policies.packing.dispatcher_thompson:ThompsonDispatcher",
    "PortfolioDispatcher": "logic.src.policies.packing.dispatcher_portfolio:PortfolioDispatcher",
    "RegularSelection": "logic.src.policies.packing.regular:RegularSelection",
    "LookaheadSelection": "logic.src.policies.packing.lookahead:LookaheadSelection",
    "ServiceLevelSelection": "logic.src.policies.packing.service_level:ServiceLevelSelection",
    "FractionalKnapsackSelection": "logic.src.policies.packing.fractional_knapsack:FractionalKnapsackSelection",
    "MIPKnapsackSelection": "logic.src.policies.packing.mip_knapsack:MIPKnapsackSelection",
    "FilterAndFanSelection": "logic.src.policies.packing.filter_and_fan:FilterAndFanSelection",
    "BernoulliRandomSelection": "logic.src.policies.packing.bernoulli_random:BernoulliRandomSelection",
    "KMeansGeographicSectorSelection": "logic.src.policies.packing.kmeans_sector:KMeansGeographicSectorSelection",
    "StaggeredRegularSelection": "logic.src.policies.packing.staggered_regular:StaggeredRegularSelection",
    "FPTASKnapsackSelection": "logic.src.policies.packing.fptas_knapsack:FPTASKnapsackSelection",
}

__all__ = ["PackingFactory", "PackingRegistry", *list(_LAZY)]


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)

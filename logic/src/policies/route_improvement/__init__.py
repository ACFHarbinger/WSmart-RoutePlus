"""Route improvement. Public names load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

from .base.factory import RouteImproverFactory
from .base.registry import RouteImproverRegistry

_LAZY = {
    "IRouteImprovement": "logic.src.interfaces.route_improvement:IRouteImprovement",
    "AdaptiveEnsembleRouteImprover": "logic.src.policies.route_improvement.adaptive_ensemble:AdaptiveEnsembleRouteImprover",
    "AdaptiveLargeNeighborhoodSearchRouteImprover": "logic.src.policies.route_improvement.adaptive_large_neighborhood_search:AdaptiveLargeNeighborhoodSearchRouteImprover",
    "BranchAndPriceRouteImprover": "logic.src.policies.route_improvement.branch_and_price:BranchAndPriceRouteImprover",
    "CheapestInsertionRouteImprover": "logic.src.policies.route_improvement.cheapest_insertion:CheapestInsertionRouteImprover",
    "CrossExchangeRouteImprover": "logic.src.policies.route_improvement.cross_exchange:CrossExchangeRouteImprover",
    "DPRouteReoptRouteImprover": "logic.src.policies.route_improvement.dp_route_reopt:DPRouteReoptRouteImprover",
    "FastTSPRouteImprover": "logic.src.policies.route_improvement.fast_tsp:FastTSPRouteImprover",
    "FixAndOptimizeRouteImprover": "logic.src.policies.route_improvement.fix_and_optimize:FixAndOptimizeRouteImprover",
    "GuidedLocalSearchRouteImprover": "logic.src.policies.route_improvement.guided_local_search:GuidedLocalSearchRouteImprover",
    "LearnedRouteImprover": "logic.src.policies.route_improvement.learned:LearnedRouteImprover",
    "LinKernighanRouteImprover": "logic.src.policies.route_improvement.lk:LinKernighanRouteImprover",
    "LinKernighanHelsgaunRouteImprover": "logic.src.policies.route_improvement.lkh:LinKernighanHelsgaunRouteImprover",
    "LinKernighanHelsgaunTwoRouteImprover": "logic.src.policies.route_improvement.lkh2:LinKernighanHelsgaunTwoRouteImprover",
    "ClassicalLocalSearchRouteImprover": "logic.src.policies.route_improvement.local_search:ClassicalLocalSearchRouteImprover",
    "MIPLNSRouteImprover": "logic.src.policies.route_improvement.mip_lns:MIPLNSRouteImprover",
    "MultiPhaseRouteImprover": "logic.src.policies.route_improvement.multi_phase:MultiPhaseRouteImprover",
    "NeuralSelectorRouteImprover": "logic.src.policies.route_improvement.neural_selector:NeuralSelectorRouteImprover",
    "NodeExchangeSteepestRouteImprover": "logic.src.policies.route_improvement.node_exchange_steepest:NodeExchangeSteepestRouteImprover",
    "OrOptRouteImprover": "logic.src.policies.route_improvement.or_opt:OrOptRouteImprover",
    "OrOptSteepestRouteImprover": "logic.src.policies.route_improvement.or_opt_steepest:OrOptSteepestRouteImprover",
    "PathRouteImprover": "logic.src.policies.route_improvement.path:PathRouteImprover",
    "ProfitableDetourRouteImprover": "logic.src.policies.route_improvement.profitable_detour:ProfitableDetourRouteImprover",
    "RandomLocalSearchRouteImprover": "logic.src.policies.route_improvement.random_local_search:RandomLocalSearchRouteImprover",
    "RegretKInsertionRouteImprover": "logic.src.policies.route_improvement.regret_k_insertion:RegretKInsertionRouteImprover",
    "RuinRecreateRouteImprover": "logic.src.policies.route_improvement.ruin_recreate:RuinRecreateRouteImprover",
    "SimulatedAnnealingRouteImprover": "logic.src.policies.route_improvement.simulated_annealing:SimulatedAnnealingRouteImprover",
    "SteepestTwoOptRouteImprover": "logic.src.policies.route_improvement.steepest_two_opt:SteepestTwoOptRouteImprover",
    "SetCoverRouteImprover": "logic.src.policies.route_improvement.set_covering:SetCoverRouteImprover",
    "SetPartitioningRouteImprover": "logic.src.policies.route_improvement.set_partitioning:SetPartitioningRouteImprover",
    "SetPartitioningPolishRouteImprover": "logic.src.policies.route_improvement.set_partitioning_polish:SetPartitioningPolishRouteImprover",
}

__all__ = ["RouteImproverRegistry", "RouteImproverFactory", *list(_LAZY)]  # pyrefly: ignore[unresolvable-dunder-all]  # splat of the lazy export map


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)

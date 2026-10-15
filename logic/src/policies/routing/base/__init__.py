"""Routing algorithm base package. Heavy bases load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

from .factory import RoutingAlgorithmFactory
from .registry import RoutingAlgorithmRegistry

_LAZY = {
    "BaseRoutingPolicy": "logic.src.policies.routing.base.base_routing_policy:BaseRoutingPolicy",
    "BaseMultiPeriodRoutingPolicy": "logic.src.policies.routing.base.base_multi_period_policy:BaseMultiPeriodRoutingPolicy",
    "GenericRoutingPolicy": "logic.src.policies.routing.base.generic_policy:GenericRoutingPolicy",
    "IRoutingAlgorithm": "logic.src.interfaces.routing_algorithm:IRoutingAlgorithm",
    "SolveResult": "logic.src.policies.routing.base.sub_problem:SolveResult",
    "Solver": "logic.src.policies.routing.base.sub_problem:Solver",
    "SubProblem": "logic.src.policies.routing.base.sub_problem:SubProblem",
    "build_subproblem": "logic.src.policies.routing.base.sub_problem:build_subproblem",
}

__all__ = [
    "BaseRoutingPolicy",
    "BaseMultiPeriodRoutingPolicy",
    "GenericRoutingPolicy",
    "IRoutingAlgorithm",
    "RoutingAlgorithmFactory",
    "RoutingAlgorithmRegistry",
    "SolveResult",
    "Solver",
    "SubProblem",
    "build_subproblem",
]


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)

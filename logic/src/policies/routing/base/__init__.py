"""Routing algorithm base package. Heavy bases load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

from .factory import RoutingAlgorithmFactory
from .registry import RoutingAlgorithmRegistry

_LAZY = {
    "BaseRoutingPolicy": "logic.src.policies.routing.base.base_routing_policy:BaseRoutingPolicy",
    "BaseMultiPeriodRoutingPolicy": "logic.src.policies.routing.base.base_multi_period_policy:BaseMultiPeriodRoutingPolicy",
    "IRoutingAlgorithm": "logic.src.interfaces.routing_algorithm:IRoutingAlgorithm",
}

__all__ = [
    "BaseRoutingPolicy",
    "BaseMultiPeriodRoutingPolicy",
    "IRoutingAlgorithm",
    "RoutingAlgorithmFactory",
    "RoutingAlgorithmRegistry",
]


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)

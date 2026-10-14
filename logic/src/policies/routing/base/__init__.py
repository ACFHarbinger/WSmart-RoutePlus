"""
Routing Algorithm Base Package.

This package provides adapter implementations for various routing
optimization policies (HGS, ALNS, VRPP, etc.) and a factory pattern
for dynamic routing algorithm instantiation.

Attributes:
    BaseRoutingPolicy: Template base class for all routing policies.
    BaseMultiPeriodRoutingPolicy: Base class for multi-period (T-day) routing.
    IRoutingAlgorithm: Interface defining the routing contract.
    RoutingAlgorithmFactory: Factory for instantiating registered routing algorithms.
    RoutingAlgorithmRegistry: Registry for dynamic routing algorithm lookup.

Example:
    >>> from logic.src.policies.routing.base import RoutingAlgorithmFactory
    >>> adapter = RoutingAlgorithmFactory.get_adapter("hgs")
    >>> tour, cost, profit, _, _ = adapter.execute(mandatory=[1, 2, 3], ...)

"""

from .base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from .base_routing_policy import BaseRoutingPolicy
from .factory import IRoutingAlgorithm, RoutingAlgorithmFactory, RoutingAlgorithmRegistry

__all__ = [
    "BaseRoutingPolicy",
    "BaseMultiPeriodRoutingPolicy",
    "IRoutingAlgorithm",
    "RoutingAlgorithmFactory",
    "RoutingAlgorithmRegistry",
]

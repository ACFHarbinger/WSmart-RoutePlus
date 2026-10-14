"""Adaptive Routing Algorithm Orchestrator (ARCO) package.

This package implements an adaptive meta-constructor that learns optimal
ordering of routing heuristics through online experience.

Attributes:
    ARCOParams: Parameters for the ARCO algorithm.
    AdaptiveRoutingAlgorithmOrchestrator: The ARCO algorithm itself.

Example:
    >>> from logic.src.policies.routing.other_algorithms.adaptive_routing_algorithm_orchestrator import ARCOParams, AdaptiveRoutingAlgorithmOrchestrator
    >>> params = ARCOParams()
    >>> arco = AdaptiveRoutingAlgorithmOrchestrator(params)
    >>> routes = arco.build_routes(context)
"""

from .params import ARCOParams as ARCOParams
from .policy_arco import AdaptiveRoutingAlgorithmOrchestrator as AdaptiveRoutingAlgorithmOrchestrator

__all__ = ["AdaptiveRoutingAlgorithmOrchestrator", "ARCOParams"]

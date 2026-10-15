"""Adaptive Routing Algorithm Orchestrator (ARCO) package.

This package implements an adaptive meta-constructor that learns optimal
ordering of routing heuristics through online experience.

Attributes:
    ARCOConfig: Parameters for the ARCO algorithm.
    AdaptiveRoutingAlgorithmOrchestrator: The ARCO algorithm itself.

Example:
    >>> from logic.src.policies.routing.other_algorithms.adaptive_routing_algorithm_orchestrator import ARCOConfig, AdaptiveRoutingAlgorithmOrchestrator
    >>> params = ARCOConfig()
    >>> arco = AdaptiveRoutingAlgorithmOrchestrator(params)
    >>> routes = arco.build_routes(context)
"""

from logic.src.configs.policies.arco import ARCOConfig as ARCOConfig
from .policy_arco import AdaptiveRoutingAlgorithmOrchestrator as AdaptiveRoutingAlgorithmOrchestrator

__all__ = ["AdaptiveRoutingAlgorithmOrchestrator", "ARCOConfig"]

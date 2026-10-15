"""Miscellaneous routing algorithms. Implementations load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "AdaptiveRoutingAlgorithmOrchestrator": "logic.src.policies.routing.other_algorithms.adaptive_routing_algorithm_orchestrator.policy_arco:AdaptiveRoutingAlgorithmOrchestrator",
    "SequentialRoutingAlgorithm": "logic.src.policies.routing.other_algorithms.sequential_routing_algorithm.policy_src:SequentialRoutingAlgorithm",
    "policy_cvrp": "logic.src.policies.routing.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp",
    "policy_tsp": "logic.src.policies.routing.other_algorithms.travelling_salesman_problem.policy_tsp",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)

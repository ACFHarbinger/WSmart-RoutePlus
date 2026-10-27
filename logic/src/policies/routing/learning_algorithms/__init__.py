"""Learning-based routing policies. Implementations load on first access."""

from typing import Any

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "ADPRolloutPolicy": "logic.src.policies.routing.learning_algorithms.approximate_dynamic_programming_with_rollout.policy_adp:ADPRolloutPolicy",
    "HierarchicalNeuralAgentPolicy": "logic.src.policies.routing.learning_algorithms.hierarchical_neural_agent.policy_hna:HierarchicalNeuralAgentPolicy",
    "NeuralAgentPolicy": "logic.src.policies.routing.learning_algorithms.neural_agent.policy_na:NeuralAgentPolicy",
}

__all__ = [
    "NeuralAgentPolicy",
]


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _LAZY, name)

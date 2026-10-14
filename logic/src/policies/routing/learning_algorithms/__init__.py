"""
Module documentation.

Attributes:
    ADPRolloutPolicy: Approximate Dynamic Programming with Rollout Policy.
    HierarchicalNeuralAgentPolicy: Hierarchical Neural Agent Policy.
    NeuralAgentPolicy: Neural Agent Policy.

Example:
    >>> from logic.src.policies.routing.learning_algorithms import ADPRolloutPolicy
    >>> policy = ADPRolloutPolicy()
    >>> policy.construct_routes(env)
"""

from .approximate_dynamic_programming_with_rollout import ADPRolloutPolicy as ADPRolloutPolicy
from .hierarchical_neural_agent import HierarchicalNeuralAgentPolicy as HierarchicalNeuralAgentPolicy
from .neural_agent import NeuralAgentPolicy as NeuralAgentPolicy

__all__ = ["ADPRolloutPolicy", "HierarchicalNeuralAgentPolicy", "NeuralAgentPolicy"]

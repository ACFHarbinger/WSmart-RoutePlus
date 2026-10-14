"""
Module documentation.

Attributes:
    HierarchicalNeuralAgentPolicy: Hierarchical Neural Agent Policy.

Example:
    >>> from logic.src.policies.routing.learning_algorithms import HierarchicalNeuralAgentPolicy
    >>> policy = HierarchicalNeuralAgentPolicy()
    >>> routes, metrics = policy.run_day(env)
    >>> print(f"Best routes: {routes}")
    >>> print(f"Metrics: {metrics}")
"""

from .policy_hna import HierarchicalNeuralAgentPolicy

__all__ = ["HierarchicalNeuralAgentPolicy"]

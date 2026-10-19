"""Agents Module.

This module contains various RL agent implementations, such as bandit agents,
which can be used within routing policies to adaptively select operators or
parameters.

Attributes:
    bandits: Sub-package containing Multi-Armed Bandit (MAB) implementations.

Example:
    >>> from logic.src.policies.helpers.learning.agents.bandits import EpsilonGreedyAgent
    >>> agent = EpsilonGreedyAgent(...)
"""

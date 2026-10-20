"""Critic Network components for VRP.

This package provides estimators for state-value functions, and includes
both canonical (torch-based) and legacy (problem-based) implementations.

Attributes:
    CriticNetwork: Modern value function network.
    create_critic_from_actor: Factory to initialize shared-backbone critics.

"""

from .policy import CriticNetwork as CriticNetwork
from .policy import create_critic_from_actor as create_critic_from_actor

__all__ = [
    "CriticNetwork",
    "create_critic_from_actor",
]

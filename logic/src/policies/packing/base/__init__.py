"""Mandatory Base Package.

This package defines the core infrastructure for the "Mandatory" selection
policy, including the factory pattern for strategy creation,
and the registry for available strategies.

Attributes:
    PackingFactory: Factory for creating strategy instances.
    PackingRegistry: Registry for mapping names to strategy classes.

Example:
    >>> from logic.src.policies.packing.base import PackingFactory
    >>> factory = PackingFactory()
    >>> strategy = factory.get_strategy("some_strategy")
"""

from .factory import PackingFactory
from .registry import PackingRegistry

__all__ = ["PackingFactory", "PackingRegistry"]

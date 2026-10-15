"""Acceptance Criterion Base Package.

This package provides the foundational interfaces and infrastructure for
move acceptance criteria, including the abstract registry and factory.

Attributes:
    AcceptanceCriterionFactory: Factory for instantiating criteria.
    AcceptanceCriterionRegistry: Registry for tracking criteria implementations.

"""

from .factory import AcceptanceCriterionFactory
from .registry import AcceptanceCriterionRegistry

__all__ = [
    "AcceptanceCriterionFactory",
    "AcceptanceCriterionRegistry",
]

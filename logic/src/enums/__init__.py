"""
Enums and Registry Module.

This package centralizes the tags used for algorithm
classification, discovery, and orchestration across the WSmart-Route framework.

Attributes:
    PolicyTag: Policy tags
    OperatorTag: Operator tags
    EnvironmentTag: Environment tags
    ModelTag: Model tags
    TrainerTag: Trainer tags

Example:
    >>> from logic.src.enums import (
        PolicyTag,
        OperatorTag,
        EnvironmentTag,
        ModelTag,
        TrainerTag,
    )
"""

from .environment_tags import EnvironmentTag
from .model_tags import ModelTag
from .operator_tags import OperatorTag
from .policy_tags import PolicyTag
from .trainer_tags import TrainerTag

__all__ = [
    "PolicyTag",
    "OperatorTag",
    "EnvironmentTag",
    "ModelTag",
    "TrainerTag",
]
